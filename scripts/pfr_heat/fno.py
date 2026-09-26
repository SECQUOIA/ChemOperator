"""Model definition, scientific losses, and canonical experiment entry point."""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections.abc import Sequence
import time
from physicsnemo.models.fno import FNO
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from ray import tune
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from chem_operator.reactors.pfr_heat.dataset_generator import CylindricalWall, ModelConstants, PlugFlowReactor
from chem_operator.experiments import CompositeLossTrainer, LossTerm, EvaluationOutcome, StreamingRegressionMetrics
from chem_operator.experiments import run_operator
import numpy as np
from scripts.pfr_heat.common import (
    PATHS,
    FILE_STEM,
    CHECKPOINT,
    SEED,
    METRIC,
    PFR_INPUTS,
    PFR_OUTPUTS,
    WALL_CONDITIONS,
    WALL_GAS,
    WALL_OUTPUTS,
    PHYSICS_CONSTANTS,
    EXPECTED_PDES,
    TUNE_SAMPLES,
    TUNE_EPOCHS,
    FINAL_EPOCHS,
    EVALUATION_BATCH_SIZE,
    CPUS_PER_TRIAL,
    GPUS_PER_TRIAL,
    MAX_CONCURRENT_TRIALS,
    HISTORY_KEYS,
    raw_dataset,
    complete_field,
    coordinate,
    uniform_spacing,
    validate_sample,
    Moments,
    make_normalizer,
    fit_normalizers,
    CoupledPFRHeatDataset,
    make_adapter,
    PROBLEM_ID,
    experiment_spec,
    parse_args,
)

class CoupledFNO(nn.Module):
    """Acyclic PFR-to-wall pair of PhysicsNeMo FNOs."""

    def __init__(self, config, pfr_normalizer, wall_normalizer):
        super().__init__()
        self.pfr_normalizer = pfr_normalizer
        self.wall_normalizer = wall_normalizer
        common = {
            "latent_channels": int(config["latent_channels"]),
            "num_fno_layers": int(config["n_layers"]),
            "padding": int(config.get("padding", 4)),
            "decoder_layers": int(config.get("decoder_layers", 1)),
            "decoder_layer_size": int(config.get("decoder_layer_size", 32)),
            "coord_features": True,
        }
        self.pfr = FNO(
            in_channels=len(PFR_INPUTS), out_channels=4, dimension=1,
            num_fno_modes=[int(config["pfr_modes"])], **common,
        )
        self.wall = FNO(
            in_channels=1 + len(WALL_CONDITIONS), out_channels=1, dimension=2,
            num_fno_modes=[int(config["wall_modes_z"]), int(config["wall_modes_r"])],
            **common,
        )

    def forward(self, pfr_x, wall_conditions) -> dict[str, torch.Tensor]:
        pfr = self.pfr(pfr_x)
        gas = self.pfr_normalizer.denormalize(pfr[:, 3], "T_gas")
        gas = self.wall_normalizer.normalize(gas, WALL_GAS)
        gas = gas[..., None].expand(-1, -1, wall_conditions.shape[-1])
        wall = self.wall(torch.cat((gas[:, None], wall_conditions), dim=1))
        return {"pfr": pfr, "wall": wall}


def model_from_config(config, pfr_normalizer, wall_normalizer, device):
    return CoupledFNO(config, pfr_normalizer, wall_normalizer).to(device)


class InformerCache:
    """Case-coefficient cache for finite-difference PhysicsInformers."""

    def __init__(self, device: torch.device):
        self.device = device
        self.reactors = {}
        self.walls = {}

    def reactor(self, c: Sequence[float], dz: float) -> PhysicsInformer:
        key = (c[1], c[2], c[4], dz)
        if key not in self.reactors:
            self.reactors[key] = PhysicsInformer(
                ["flow_a", "flow_b", "flow_c", "gas_energy"],
                PlugFlowReactor(ModelConstants(), c[1], c[2], c[4]),
                "finite_difference", fd_dx=dz, device=str(self.device),
            )
        return self.reactors[key]

    def wall(self, c: Sequence[float], dz: float, dr: float) -> PhysicsInformer:
        key = (c[5], c[6], dz, dr)
        if key not in self.walls:
            self.walls[key] = PhysicsInformer(
                ["solid_heat"], CylindricalWall(c[5], c[6]),
                "finite_difference", fd_dx=[dz, dr], device=str(self.device),
            )
        return self.walls[key]


def physical_predictions(prediction, pfr_normalizer, wall_normalizer):
    flows = torch.stack([
        pfr_normalizer.denormalize(prediction["pfr"][:, i], name)
        for i, name in enumerate(PFR_OUTPUTS[:3])
    ], dim=1)
    gas = pfr_normalizer.denormalize(prediction["pfr"][:, 3], "T_gas")
    solid = wall_normalizer.denormalize(prediction["wall"][:, 0], "T_solid")
    return flows, gas, solid


def physics_losses(prediction, batch, pfr_normalizer, wall_normalizer, cache):
    """Return species, gas-energy, wall-conduction, and BC losses."""
    flows, gas, solid = physical_predictions(
        prediction, pfr_normalizer, wall_normalizer
    )
    constants = batch["physics_constants"].to(flows.device, flows.dtype)
    z, r = batch["z"].to(flows.device), batch["r"].to(flows.device)
    totals = {name: flows.new_zeros(()) for name in ("species", "gas", "solid", "bc")}
    for i in range(flows.shape[0]):
        c = [float(v) for v in constants[i].detach().cpu()]
        dz, dr = float(z[i, 1] - z[i, 0]), float(r[i, 1] - r[i, 0])
        flow_scale, temperature_scale = c[-2:]
        fi = flows[i:i + 1] / flow_scale
        tg = gas[i:i + 1, None] / temperature_scale
        ts = solid[i:i + 1, None] / temperature_scale
        residuals = cache.reactor(c, dz).forward({
            "f_a": fi[:, 0:1], "f_b": fi[:, 1:2], "f_c": fi[:, 2:3],
            "t_gas": tg, "t_wall": ts[..., 0],
        })
        totals["species"] += torch.stack([
            residuals[name][..., 2:-2].square().mean()
            for name in ("flow_a", "flow_b", "flow_c")
        ]).mean()
        totals["gas"] += residuals["gas_energy"][..., 2:-2].square().mean()
        radial = r[i][None, None, None, :].expand_as(ts)
        wall_residual = cache.wall(c, dz, dr).forward(
            {"t_solid": ts, "y": radial}
        )["solid_heat"]
        totals["solid"] += wall_residual[..., 2:-2, 2:-2].square().mean()

        inlet = fi.new_tensor([c[0] / flow_scale, 0.0, 0.0])
        inlet_flow = (fi[0, :, 0] - inlet).square().mean()
        inlet_gas = (tg[0, 0, 0] - c[2] / temperature_scale).square()
        outer = (ts[0, 0, :, -1] - c[3] / temperature_scale).square().mean()
        radial_gradient = (-3 * ts[..., 0] + 4 * ts[..., 1] - ts[..., 2]) / (2 * dr)
        interface = (radial_gradient - c[6] * (ts[..., 0] - tg)).square().mean()
        flux_0 = (-3 * ts[..., 0, :] + 4 * ts[..., 1, :] - ts[..., 2, :]) / (2 * dz)
        flux_1 = (3 * ts[..., -1, :] - 4 * ts[..., -2, :] + ts[..., -3, :]) / (2 * dz)
        axial = 0.5 * (flux_0.square().mean() + flux_1.square().mean())
        totals["bc"] += torch.stack((inlet_flow, inlet_gas, outer, interface, axial)).mean()
    return {name: value / flows.shape[0] for name, value in totals.items()}


def supervised_loss(prediction, batch):
    return 0.5 * (
        F.mse_loss(prediction["pfr"], batch["pfr_y"])
        + F.mse_loss(prediction["wall"], batch["wall_y"])
    )


def relative_channels(prediction: torch.Tensor, target: torch.Tensor):
    error = torch.linalg.vector_norm((prediction - target).flatten(2), dim=2)
    scale = torch.linalg.vector_norm(target.flatten(2), dim=2).clamp_min(1e-12)
    return error / scale


PHYSICS_WEIGHT_KEYS = ("lambda_f", "lambda_g", "lambda_s", "lambda_bc")


def search_space(variant="physics"):
    space = {'pfr_modes': tune.choice([8, 12, 16]), 'wall_modes_z': tune.choice([8, 12, 16]), 'wall_modes_r': tune.choice([4, 6, 8]), 'latent_channels': tune.choice([8, 16, 24]), 'n_layers': tune.choice([3, 4]), 'padding': tune.choice([0, 4]), 'decoder_layers': tune.choice([1, 2]), 'decoder_layer_size': tune.choice([16, 32]), 'learning_rate': tune.loguniform(0.0001, 0.003), 'weight_decay': tune.loguniform(1e-08, 0.0001), 'batch_size': tune.choice([1, 2, 4])}
    if variant == "data":
        space.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant == "physics":
        space.update({
            key: tune.loguniform(0.0001, 0.1)
            for key in PHYSICS_WEIGHT_KEYS
        })
    else:
        raise ValueError(f"Unknown variant {variant!r}.")
    return space

MODEL_IDS = {"data": "fno_data", "physics": "fno_physics"}
MODEL_ID = MODEL_IDS["physics"]

def make_trainer(config, context, pfr, wall, shape, *, variant="physics"):
    config = dict(config)
    if variant == "data":
        config.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant != "physics":
        raise ValueError(f"Unknown variant {variant!r}.")
    physics_active = any(
        float(config.get(key, 0.0)) > 0.0 for key in PHYSICS_WEIGHT_KEYS
    )
    cache = InformerCache(context.device)
    def adapt(batch, device, dtype):
        batch = {key: value.to(device=device, dtype=dtype) for key, value in batch.items()}
        return (batch["pfr_x"], batch["wall_conditions"]), {"pfr": batch["pfr_y"], "wall": batch["wall_y"]}, batch
    def forward(model, inputs, batch):
        prediction = model(*inputs)
        if physics_active or not model.training:
            batch["physics_losses"] = physics_losses(
                prediction, batch, pfr, wall, cache
            )
        else:
            zero = prediction["pfr"].new_zeros(())
            batch["physics_losses"] = {
                name: zero for name in ("species", "gas", "solid", "bc")
            }
        return prediction
    def metric(p, t, batch):
        pp, tt = physical_predictions(p, pfr, wall), physical_predictions(t, pfr, wall)
        return tuple(torch.cat([v.flatten(1) for v in values], dim=1) for values in (pp,tt))
    terms = [LossTerm("data_loss", lambda p,t,b: supervised_loss(p,b))]
    for name, key in (("species", "lambda_f"), ("gas", "lambda_g"), ("solid", "lambda_s"), ("bc", "lambda_bc")):
        terms.append(LossTerm(name + "_loss", lambda p,t,b,n=name: b["physics_losses"][n], float(config[key])))
    return CompositeLossTrainer(lambda cfg: model_from_config(cfg,pfr,wall,context.device), config,
        batch_adapter=adapt, forward_adapter=forward, loss_terms=terms, metric_adapter=metric,
        checkpoint_metadata={"pfr_normalizer": pfr.state_dict(), "wall_normalizer": wall.state_dict(),
                             "shape": list(shape), "pdes": EXPECTED_PDES})

def evaluate_run(trainer, data, context, pfr, wall, cases=2):
    scores = StreamingRegressionMetrics()
    fields = {name: StreamingRegressionMetrics() for name in PFR_OUTPUTS + WALL_OUTPUTS}
    stored = {name: [] for name in ("pfr_reference", "pfr_prediction", "wall_reference", "wall_prediction", "z", "r")}
    started = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=int(trainer.config["batch_size"])):
            prediction = trainer.model(batch["pfr_x"].to(context.device), batch["wall_conditions"].to(context.device))
            pp = tuple(v.cpu() for v in physical_predictions(prediction, pfr, wall))
            tt = physical_predictions({"pfr": batch["pfr_y"], "wall": batch["wall_y"]}, pfr, wall)
            pred = torch.cat((pp[0], pp[1][:,None]), dim=1)
            ref = torch.cat((tt[0], tt[1][:,None]), dim=1)
            scores.update(torch.cat([v.flatten(1) for v in pp],dim=1), torch.cat([v.flatten(1) for v in tt],dim=1))
            for i,name in enumerate(PFR_OUTPUTS): fields[name].update(pred[:,i], ref[:,i])
            fields["T_solid"].update(pp[2],tt[2])
            count = max(0,min(cases-len(stored["z"]), len(ref)))
            for i in range(count):
                for key,value in (("pfr_reference",ref),("pfr_prediction",pred),("wall_reference",tt[2][:,None]),
                                  ("wall_prediction",pp[2][:,None]),("z",batch["z"]),("r",batch["r"])):
                    stored[key].append(value[i].numpy())
    elapsed = time.perf_counter()-started
    metrics = scores.compute()
    for name,value in fields.items(): metrics.update({f"{name}/{k}": v for k,v in value.compute().items()})
    diagnostics = trainer.evaluate_losses(data, context)
    metrics.update({k: v for k,v in diagnostics.items() if k.endswith("_loss")})
    return EvaluationOutcome(metrics, {**{k: np.asarray(v) for k,v in stored.items()},
        "case_ids": np.arange(len(stored["z"])), "pfr_labels": np.asarray(PFR_OUTPUTS), "wall_labels": np.asarray(WALL_OUTPUTS)}, elapsed)


def parse_cli_args():
    """Parse the shared workflow arguments plus the loss variant."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant", choices=("both", "data", "physics"), default="physics"
    )
    return parse_args(parser=parser)


def main():
    args = parse_cli_args()
    raw = raw_dataset(args.data_dir,"train")
    try: pfr,wall,shape = fit_normalizers(raw)
    finally: raw.close()
    variants = ("data", "physics") if args.variant == "both" else (args.variant,)
    for variant in variants:
        model_id = MODEL_IDS[variant]
        run_operator(
            args, PATHS, experiment_spec(args, model_id),
            lambda split: make_adapter(args.data_dir,split,pfr,wall,shape),
            lambda config,context,selected=variant: make_trainer(
                config,context,pfr,wall,shape,variant=selected
            ),
            search_space(variant),
            lambda trainer,data,context: evaluate_run(
                trainer,data,context,pfr,wall,args.plot_cases
            ),
        )

if __name__ == "__main__":
    main()
