"""PhysicsNeMo DeepONet for one-step transient pipe-flow rollouts."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from ray import tune
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.experiments import (
    CompositeLossTrainer,
    EvaluationOutcome,
    LossTerm,
    StreamingRegressionMetrics,
    run_operator,
)
from chem_operator.models import fit_zscore_normalizer
from scripts.pipe_flow_transient.common import (
    FIELD_NAMES,
    FILE_STEM,
    PATHS,
    PDE_REGISTRY,
    PHYSICS_CONSTANT_NAMES,
    dataset_pde_name,
    experiment_spec,
    parse_args,
)

CONSTANT_NAMES = PHYSICS_CONSTANT_NAMES
PHYSICS_WEIGHT_KEYS = ("physics_weight", "constraint_weight")
MODEL_IDS = {"data": "nemo_deeponet_data", "physics": "nemo_deeponet_physics"}


def raw_dataset(data_dir: Path, split: str, *, full_trajectory: bool = False) -> ChemOperatorDataset:
    """Read adjacent pairs, or one whole trajectory per case for evaluation."""
    return ChemOperatorDataset(
        data_dir / f"{FILE_STEM}_{split}.h5",
        task="operator_cartesian" if full_trajectory else "rollout",
        coordinate_name="t",
        input_fields=FIELD_NAMES,
        output_fields=FIELD_NAMES,
        constant_inputs=CONSTANT_NAMES,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )


def _physics_constants(sample: Mapping[str, Any]) -> torch.Tensor:
    return torch.stack([sample["constant_inputs"][name].reshape(()) for name in CONSTANT_NAMES])


def _fourier_times(t_now: torch.Tensor, t_next: torch.Tensor, constants: torch.Tensor):
    radius, _, viscosity, _, density = constants.unbind(dim=-1)
    scale = (viscosity / density) / radius.square()
    return t_now * scale, (t_next - t_now) * scale


class RolloutAdapter(Dataset):
    """Normalize one radial profile and retain physical coordinates for physics."""

    def __init__(self, dataset: ChemOperatorDataset, normalizer, n_radial: int):
        self.dataset = dataset
        self.normalizer = normalizer
        self.n_radial = n_radial

    def __len__(self) -> int:
        return len(self.dataset)

    def branch(self, velocity: torch.Tensor, constants: torch.Tensor,
               t_now: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        profile = self.normalizer.normalize(velocity, "velocity")
        parameters = torch.stack([
            self.normalizer.normalize(constants[..., index], name)
            for index, name in enumerate(CONSTANT_NAMES)
        ], dim=-1)
        tau, dtau = _fourier_times(t_now, t_next, constants)
        return torch.cat((profile, parameters, tau.unsqueeze(-1), dtau.unsqueeze(-1)), dim=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        current = sample["input_fields"]["velocity"][0]
        following = sample["output_fields"]["velocity"][0]
        radius_grid = sample["output_coordinates"]["r"]
        if current.numel() != self.n_radial or following.numel() != self.n_radial or radius_grid.numel() != self.n_radial:
            raise ValueError("Rollout sample radial resolution differs from training resolution.")
        t_now = sample["input_coordinates"]["t"][0]
        t_next = sample["output_coordinates"]["t"][0]
        if t_next <= t_now:
            raise ValueError("Rollout times must increase.")
        constants = _physics_constants(sample)
        radius_fraction = radius_grid / constants[0]
        if not torch.all(radius_fraction[1:] > radius_fraction[:-1]):
            raise ValueError("Radial coordinates must increase.")
        return {
            "branch": self.branch(current, constants, t_now, t_next),
            "x": radius_fraction,
            "y": self.normalizer.normalize(following, "velocity"),
            "current": current,
            "t_now": t_now,
            "t_next": t_next,
            "r": radius_grid,
            "constants": constants,
            "physicsnemo_pde": str(sample["metadata"]["physicsnemo_pde"]),
        }


def fit_normalizer(data_dir: Path):
    # Read each case once while retaining statistics for every radial node.
    dataset = raw_dataset(data_dir, "train", full_trajectory=True)
    try:
        dataset_pde_name(dataset)
        return fit_zscore_normalizer(dataset, FIELD_NAMES, CONSTANT_NAMES)
    finally:
        dataset.close()


def make_adapter(data_dir: Path, split: str, normalizer, n_radial: int):
    dataset = raw_dataset(data_dir, split)
    return dataset, RolloutAdapter(dataset, normalizer, n_radial)


class DeepONet(nn.Module):
    """PhysicsNeMo MLP branch and trunk with a scalar product output."""

    def __init__(self, config: Mapping[str, Any], n_radial: int):
        super().__init__()
        latent = int(config["latent_width"])
        options = {"num_layers": int(config["depth"]), "activation_fn": str(config["activation"])}
        self.branch = FullyConnected(
            in_features=n_radial + len(CONSTANT_NAMES) + 2,
            out_features=latent,
            layer_size=int(config["branch_width"]),
            **options,
        )
        self.trunk = FullyConnected(
            in_features=1,
            out_features=latent,
            layer_size=int(config["trunk_width"]),
            **options,
        )
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, branch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if branch.ndim != 2 or x.ndim != 2 or branch.shape[0] != x.shape[0]:
            raise ValueError("Expected branch [batch, features] and x [batch, radius].")
        branch_latent = self.branch(branch)
        trunk_latent = self.trunk((2.0 * x - 1.0).unsqueeze(-1))
        return torch.einsum("bl,brl->br", branch_latent, trunk_latent) + self.bias


def model_from_config(config: Mapping[str, Any], n_radial: int, device: torch.device) -> DeepONet:
    return DeepONet(config, n_radial).to(device)


def make_physics_informer(pde_name: str, radial_spacing: float, device: torch.device):
    try:
        equation = PDE_REGISTRY[pde_name]()
    except KeyError as exc:
        raise ValueError(f"Unsupported PhysicsNeMo PDE {pde_name!r}.") from exc
    return PhysicsInformer(
        required_outputs=["momentum"], equations=equation,
        grad_method="meshless_finite_difference", fd_dx=radial_spacing,
        device=str(device),
    )


def physics_losses(prediction: torch.Tensor, batch: Mapping[str, Any],
                   normalizer, informer: PhysicsInformer, *, pde_name: str,
                   radial_spacing: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Backward-time momentum residual and next-profile boundary conditions."""
    names = batch["physicsnemo_pde"]
    if isinstance(names, str):
        names = [names]
    if set(names) != {pde_name}:
        raise ValueError("Batch PDE metadata does not match the configured equation.")
    if prediction.ndim != 2 or prediction.shape[-1] < 3:
        raise ValueError("Physics loss requires [batch, radius] with at least three nodes.")
    constants = batch["constants"].to(device=prediction.device, dtype=prediction.dtype)
    radius, length, viscosity, pressure_drop, _density = constants.unbind(dim=-1)
    velocity_scale = pressure_drop * radius.square() / (4.0 * viscosity * length)
    current = batch["current"].to(device=prediction.device, dtype=prediction.dtype) / velocity_scale[:, None]
    following = normalizer.denormalize(prediction, "velocity") / velocity_scale[:, None]
    _, dtau = _fourier_times(batch["t_now"], batch["t_next"], constants)
    if torch.any(dtau <= 0):
        raise ValueError("Dimensionless time step must be positive.")
    x = batch["x"].to(device=prediction.device, dtype=prediction.dtype)
    dx = x[:, 1:] - x[:, :-1]
    if not torch.allclose(dx, torch.full_like(dx, radial_spacing), rtol=1e-4, atol=1e-6):
        raise ValueError("Batch radial spacing differs from the PhysicsInformer spacing.")
    center = following[:, 1:-1]
    residual = informer.forward({
        "velocity": center,
        "velocity>>x::1": following[:, 2:],
        "velocity>>x::-1": following[:, :-2],
        "velocity__t": (following[:, 1:-1] - current[:, 1:-1]) / dtau[:, None],
        "x": x[:, 1:-1],
    })["momentum"]
    wall_loss = following[:, -1].square().mean()
    center_gradient = (-3.0 * following[:, 0] + 4.0 * following[:, 1] - following[:, 2]) / (2.0 * radial_spacing)
    return residual.square().mean(), (wall_loss + center_gradient.square().mean()) / 2.0


def search_space(variant: str = "physics") -> dict[str, Any]:
    space = {
        "branch_width": tune.choice([64, 128]),
        "trunk_width": tune.choice([16, 32]),
        "latent_width": tune.choice([16, 32]),
        "depth": tune.choice([2, 3, 4]),
        "activation": tune.choice(["gelu", "tanh"]),
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-4),
        "batch_size": tune.choice([4, 8, 16]),
    }
    if variant == "data":
        space.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant == "physics":
        space.update({key: tune.loguniform(1e-4, 1e-1) for key in PHYSICS_WEIGHT_KEYS})
    else:
        raise ValueError(f"Unknown variant {variant!r}.")
    return space


def make_trainer(config, context, normalizer, n_radial: int, *, pde_name: str,
                 radial_spacing: float, variant: str = "physics"):
    config = dict(config)
    if variant == "data":
        config.update({key: 0.0 for key in PHYSICS_WEIGHT_KEYS})
    elif variant != "physics":
        raise ValueError(f"Unknown variant {variant!r}.")
    active = any(float(config[key]) > 0.0 for key in PHYSICS_WEIGHT_KEYS)
    informer = make_physics_informer(pde_name, radial_spacing, context.device)

    def adapt(batch, device, dtype):
        moved = {key: value.to(device=device, dtype=dtype) if isinstance(value, torch.Tensor) else value
                 for key, value in batch.items()}
        return (moved["branch"], moved["x"]), moved["y"], moved

    def forward(model, inputs, batch):
        prediction = model(*inputs)
        if active or not model.training:
            batch["physics_loss"], batch["constraint_loss"] = physics_losses(
                prediction, batch, normalizer, informer,
                pde_name=pde_name, radial_spacing=radial_spacing,
            )
        else:
            zero = prediction.new_zeros(())
            batch["physics_loss"], batch["constraint_loss"] = zero, zero
        return prediction

    return CompositeLossTrainer(
        lambda cfg: model_from_config(cfg, n_radial, context.device), config,
        batch_adapter=adapt, forward_adapter=forward,
        loss_terms=[
            LossTerm("data_loss", lambda p, t, b: F.mse_loss(p, t)),
            LossTerm("physics_loss", lambda p, t, b: b["physics_loss"], float(config["physics_weight"])),
            LossTerm("constraint_loss", lambda p, t, b: b["constraint_loss"], float(config["constraint_weight"])),
        ],
        metric_adapter=lambda p, t, b: (normalizer.denormalize(p, "velocity"), normalizer.denormalize(t, "velocity")),
        checkpoint_metadata={
            "normalizer": normalizer.state_dict(), "n_radial": n_radial,
            "pde_name": pde_name, "radial_spacing": radial_spacing,
            "input_fields": list(FIELD_NAMES), "constant_names": list(CONSTANT_NAMES),
            "branch_time_features": ["fourier_time", "fourier_step"],
            "variant": variant,
        },
    )


def evaluate_run(trainer, data, context, normalizer, data_dir: Path,
                 n_radial: int, *, cases: int = 2, max_cases: int | None = None):
    """Score teacher-forced windows and recursive trajectories in physical units."""
    model = trainer.model.to(context.device).eval()
    one_step = StreamingRegressionMetrics()
    rollout = StreamingRegressionMetrics()
    arrays = {key: [] for key in ("case_ids", "reference", "prediction", "t", "r")}
    started = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=int(trainer.config["batch_size"])):
            predicted = model(batch["branch"].to(context.device), batch["x"].to(context.device))
            predicted = normalizer.denormalize(predicted.cpu(), "velocity")
            reference = normalizer.denormalize(batch["y"], "velocity")
            one_step.update(predicted, reference)

        raw = raw_dataset(data_dir, "test", full_trajectory=True)
        adapter = RolloutAdapter(raw, normalizer, n_radial)
        try:
            count = len(raw) if max_cases is None else min(len(raw), max_cases)
            for case_index in range(count):
                sample = raw[case_index]
                constants = _physics_constants(sample)
                r = sample["output_coordinates"]["r"]
                if r.numel() != n_radial:
                    raise ValueError("Test trajectory radial resolution differs from training resolution.")
                x = (r / constants[0]).to(context.device).unsqueeze(0)
                times = torch.cat((sample["input_coordinates"]["t"], sample["output_coordinates"]["t"]))
                reference = torch.cat((sample["input_fields"]["velocity"], sample["output_fields"]["velocity"]))
                current = reference[0]
                profiles = [current]
                for step in range(1, len(times)):
                    branch = adapter.branch(current, constants, times[step - 1], times[step]).to(context.device).unsqueeze(0)
                    predicted = model(branch, x)
                    current = normalizer.denormalize(predicted.squeeze(0).cpu(), "velocity")
                    profiles.append(current)
                prediction = torch.stack(profiles)
                rollout.update(prediction[1:], reference[1:])
                if len(arrays["case_ids"]) < cases:
                    arrays["case_ids"].append(sample["metadata"]["case_index"])
                    arrays["reference"].append(reference.unsqueeze(0).numpy())
                    arrays["prediction"].append(prediction.unsqueeze(0).numpy())
                    arrays["t"].append(times.numpy())
                    arrays["r"].append(r.numpy())
        finally:
            raw.close()
    elapsed = time.perf_counter() - started
    scores = {f"one_step_{key}": value for key, value in one_step.compute().items()}
    scores.update(rollout.compute())
    diagnostics = trainer.evaluate_losses(data, context)
    scores.update({key: diagnostics[key] for key in ("data_loss", "physics_loss", "constraint_loss")})
    return EvaluationOutcome(
        scores,
        {**{key: np.asarray(value) for key, value in arrays.items()},
         "labels": np.asarray(FIELD_NAMES), "coordinate_names": np.asarray(("t", "r"))},
        elapsed,
    )


def parse_cli_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("both", "data", "physics"), default="physics")
    return parse_args(epochs=20, tune_epochs=10, samples=8, parser=parser)


def main():
    args = parse_cli_args()
    normalizer = fit_normalizer(args.data_dir)
    raw = raw_dataset(args.data_dir, "train")
    try:
        pde_name = dataset_pde_name(raw)
        first = raw[0]
        n_radial = first["input_fields"]["velocity"].shape[-1]
        x = first["input_coordinates"]["r"] / first["constant_inputs"]["radius"]
        if n_radial < 3 or not torch.allclose(x[1:] - x[:-1], (x[1] - x[0]).expand_as(x[1:]), rtol=1e-5, atol=1e-7):
            raise ValueError("PhysicsNeMo rollout requires at least three uniformly spaced radial nodes.")
        radial_spacing = float(x[1] - x[0])
    finally:
        raw.close()
    variants = ("data", "physics") if args.variant == "both" else (args.variant,)
    for variant in variants:
        spec = replace(experiment_spec(args, MODEL_IDS[variant]), benchmark_protocol_id="rollout-one-step-v1")
        run_operator(
            args, PATHS, spec,
            lambda split: make_adapter(args.data_dir, split, normalizer, n_radial),
            lambda config, context, selected=variant: make_trainer(
                config, context, normalizer, n_radial, pde_name=pde_name,
                radial_spacing=radial_spacing, variant=selected,
            ),
            search_space(variant),
            lambda trainer, data, context: evaluate_run(
                trainer, data, context, normalizer, args.data_dir, n_radial,
                cases=args.plot_cases, max_cases=args.max_cases,
            ),
        )


if __name__ == "__main__":
    main()
