"""Model definition, scientific losses, and canonical experiment entry point."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import time
from typing import Any, Mapping
from omegaconf import OmegaConf
from ray import tune
import torch
from torch import nn
from torch.utils.data import DataLoader
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from chem_operator.reactors.pipe_flow.dataset_generator import HagenPoiseuille, hagen_poiseuille_velocity
from chem_operator.experiments import CompositeLossTrainer, LossTerm, EvaluationOutcome, StreamingRegressionMetrics
from chem_operator.experiments import parse_operator_args, run_operator
import argparse
import numpy as np
from scripts.pipe_flow.common import PATHS, BRANCH_NAMES, Normalization, load_physics_split, fit_physics_normalization, PhysicsPipeDataset, physics_experiment_spec
class DeepONet(nn.Module):
    """PhysicsNeMo branch/trunk MLPs with a scalar DeepONet product."""

    def __init__(
        self,
        *,
        width: int,
        depth: int,
        latent_width: int,
        activation: str,
    ) -> None:
        super().__init__()
        options = {
            "layer_size": width,
            "out_features": latent_width,
            "num_layers": depth,
            "activation_fn": activation,
        }
        self.branch = FullyConnected(in_features=len(BRANCH_NAMES), **options)
        self.trunk = FullyConnected(in_features=1, **options)
        self.bias = nn.Parameter(torch.zeros(1))

    def points(
        self,
        branch: torch.Tensor,
        coordinates: torch.Tensor,
        radius: torch.Tensor,
        case_index: torch.Tensor,
    ) -> torch.Tensor:
        branch_latent = self.branch(branch)[case_index]
        case_radius = radius.reshape(-1, 1)[case_index]
        relative_coordinate = 2.0 * coordinates / case_radius - 1.0
        trunk_latent = self.trunk(relative_coordinate)
        return (branch_latent * trunk_latent).sum(dim=-1, keepdim=True) + self.bias

    def forward(
        self,
        branch: torch.Tensor,
        coordinates: torch.Tensor,
        radius: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, n_points = coordinates.shape[:2]
        case_index = torch.arange(batch_size, device=branch.device).repeat_interleave(
            n_points
        )
        values = self.points(
            branch,
            coordinates.reshape(-1, 1),
            radius,
            case_index,
        )
        return values.reshape(batch_size, n_points, 1)


def model_from_config(config: Mapping[str, Any], device: torch.device) -> DeepONet:
    return DeepONet(
        width=int(config["width"]),
        depth=int(config["depth"]),
        latent_width=int(config["latent_width"]),
        activation=str(config["activation"]),
    ).to(device)


def physics_mse(
    model: DeepONet,
    physics: PhysicsInformer,
    branch: torch.Tensor,
    coordinates: torch.Tensor,
    radius: torch.Tensor,
    viscosity: torch.Tensor,
    pressure_gradient: torch.Tensor,
    statistics: Normalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized predictions and a dimensionless residual MSE."""

    batch_size, n_points = coordinates.shape[:2]
    point_coordinates = coordinates.reshape(-1, 1).detach().clone()
    point_coordinates.requires_grad_(True)
    case_index = torch.arange(batch_size, device=branch.device).repeat_interleave(
        n_points
    )
    prediction_normalized = model.points(
        branch, point_coordinates, radius, case_index
    )
    velocity = (
        prediction_normalized * statistics.velocity_std.to(branch.device)
        + statistics.velocity_mean.to(branch.device)
    )
    point_radius = radius[case_index]
    point_viscosity = viscosity[case_index]
    point_gradient = pressure_gradient[case_index]
    residual = physics.forward(
        {
            "coordinates": point_coordinates,
            "x": point_coordinates,
            "velocity": velocity,
            "dynamic_viscosity": point_viscosity,
            "pressure_gradient": point_gradient,
        }
    )["momentum"]
    residual_scale = (point_gradient.abs() * point_radius).clamp_min(1.0e-12)
    return (
        prediction_normalized.reshape(batch_size, n_points, 1),
        (residual / residual_scale).square().mean(),
    )


def verify_exact_residual() -> None:
    coordinate = torch.linspace(0.0, 1.0e-3, 32, dtype=torch.float64).reshape(-1, 1)
    coordinate.requires_grad_(True)
    viscosity = torch.full_like(coordinate, 1.0e-3)
    gradient = torch.full_like(coordinate, -80.0)
    velocity = hagen_poiseuille_velocity(
        coordinate,
        radius=1.0e-3,
        dynamic_viscosity=viscosity,
        pressure_gradient=gradient,
    )
    physics = PhysicsInformer(["momentum"], HagenPoiseuille(), "autodiff")
    residual = physics.forward(
        {
            "coordinates": coordinate,
            "x": coordinate,
            "velocity": velocity,
            "dynamic_viscosity": viscosity,
            "pressure_gradient": gradient,
        }
    )["momentum"]
    if float(residual.detach().abs().max()) > 1.0e-10:
        raise RuntimeError("PhysicsInformer failed the exact-profile residual check.")

def make_trainer(config, context, statistics):
    physics = PhysicsInformer(required_outputs=["momentum"], equations=HagenPoiseuille(),
                              grad_method="autodiff",device=str(context.device))
    def adapt(batch, device, dtype):
        batch = {k: v.to(device=device,dtype=dtype) for k,v in batch.items()}
        return (batch["branch"],batch["coordinates"],batch["radius"]),batch["y"],batch
    def forward(model, inputs, batch):
        if float(config.get("physics_weight",0)) > 0 or not model.training:
            prediction,residual = physics_mse(model,physics,*inputs,batch["viscosity"],batch["pressure_gradient"],statistics)
        else:
            prediction = model(*inputs)
            residual = prediction.new_zeros(())
        batch["physics_loss"] = residual
        return prediction
    def physical(value):
        return value * statistics.velocity_std.to(value.device) + statistics.velocity_mean.to(value.device)
    return CompositeLossTrainer(lambda cfg: model_from_config(cfg,context.device),config,
        batch_adapter=adapt,forward_adapter=forward,validation_requires_grad=True,
        loss_terms=[LossTerm("data_loss",lambda p,t,b: torch.nn.functional.mse_loss(p,t)),
                    LossTerm("physics_loss",lambda p,t,b: b["physics_loss"],float(config.get("physics_weight",0)))],
        metric_adapter=lambda p,t,b: (physical(p),physical(t)),
        checkpoint_metadata={"normalizer": {name: getattr(statistics,name).cpu() for name in
                            ("branch_mean","branch_std","velocity_mean","velocity_std")},
                             "pde_name": "HagenPoiseuille", "branch_names": list(BRANCH_NAMES)})


def evaluate_run(trainer,data,context,statistics,cases=3):
    scores = StreamingRegressionMetrics()
    stored = {k: [] for k in ("reference","prediction","coordinates")}
    started = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(data,batch_size=int(trainer.config["batch_size"])):
            prediction = trainer.model(batch["branch"].to(context.device),batch["coordinates"].to(context.device),batch["radius"].to(context.device)).cpu()
            prediction = prediction * statistics.velocity_std + statistics.velocity_mean
            scores.update(prediction,batch["reference"])
            take = max(0,min(cases-len(stored["reference"]),len(prediction)))
            for i in range(take):
                stored["prediction"].append(prediction[i].numpy())
                stored["reference"].append(batch["reference"][i].numpy())
                stored["coordinates"].append(batch["coordinates"][i].numpy())
    elapsed = time.perf_counter()-started
    diagnostics = trainer.evaluate_losses(data, context)
    return EvaluationOutcome({**scores.compute(), "physics_loss": diagnostics["physics_loss"]},
        {**{k: np.asarray(v) for k,v in stored.items()},"case_ids": np.arange(len(stored["reference"])),"labels": np.asarray(["velocity"])},elapsed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=Path(__file__).with_suffix(".yaml"))
    parser.add_argument("--variant",choices=("both","data","physics"),default="both")
    parser.add_argument("--physics-samples", type=int, help="Physics-weight trials; defaults to the YAML budget for --variant both.")
    args = parse_operator_args(PATHS,epochs=40,tune_epochs=20,samples=8,parser=parser)
    if args.physics_samples is not None and args.physics_samples < 1:
        parser.error("--physics-samples must be positive.")
    cfg = OmegaConf.load(args.config)
    verify_exact_residual()
    statistics = fit_physics_normalization(load_physics_split(args.data_dir,"train",args.max_cases))
    def dataset_factory(split):
        data = PhysicsPipeDataset(load_physics_split(args.data_dir,split,args.max_cases).normalized(statistics))
        return data,data
    base_space = {"width": tune.choice(list(cfg.search.width)), "depth": tune.choice(list(cfg.search.depth)),
        "latent_width": tune.choice(list(cfg.search.latent_width)), "activation": tune.choice(list(cfg.search.activation)),
        "learning_rate": tune.loguniform(*map(float,cfg.search.learning_rate)),
        "weight_decay": tune.loguniform(*map(float,cfg.search.weight_decay)), "batch_size": tune.choice(list(cfg.search.batch_size))}
    from chem_operator.experiments import ArtifactStore
    base_context = None
    for variant in (("data","physics") if args.variant == "both" else (args.variant,)):
        model_id = "physics_deeponet_data" if variant == "data" else "physics_deeponet"
        space = dict(base_space, physics_weight=0.0)
        if variant == "physics":
            if base_context is not None:
                space = ArtifactStore(base_context).read_best_config()
            space["physics_weight"] = tune.loguniform(float(cfg.physics.min_weight),float(cfg.physics.max_weight))
        variant_args = argparse.Namespace(**vars(args))
        if variant == "physics":
            variant_args.samples = args.physics_samples or (int(cfg.physics.num_samples) if args.variant == "both" else args.samples)
        context = run_operator(variant_args,PATHS,physics_experiment_spec(variant_args,model_id),dataset_factory,
            lambda config,context: make_trainer(dict(config, physics_weight=0.0) if variant == "data" else config,context,statistics), space,
            lambda trainer,data,context: evaluate_run(trainer,data,context,statistics,args.plot_cases),seed=int(cfg.seed))
        if variant == "data": base_context = context

if __name__ == "__main__":
    main()
