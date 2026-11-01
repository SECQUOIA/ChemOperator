"""PhysicsNeMo DeepONet for one-step transient pipe-flow rollouts."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from ray import tune
from torch import nn
from torch.nn import functional as F

from chem_operator.experiments import (
    CompositeLossTrainer,
    LossTerm,
    run_operator,
)
from scripts.pipe_flow_transient.common import (
    FIELD_NAMES,
    PATHS,
    PDE_REGISTRY,
    PHYSICS_CONSTANT_NAMES,
    RolloutAdapter,
    dataset_pde_name,
    evaluate_rollout_run as evaluate_run,
    experiment_spec,
    fit_rollout_normalizer as fit_normalizer,
    fourier_times as _fourier_times,
    make_rollout_adapter as make_adapter,
    parse_args,
    rollout_constants as _physics_constants,
    rollout_raw_dataset as raw_dataset,
)

CONSTANT_NAMES = PHYSICS_CONSTANT_NAMES
PHYSICS_WEIGHT_KEYS = ("physics_weight", "constraint_weight")
MODEL_IDS = {"data": "nemo_deeponet_data", "physics": "nemo_deeponet_physics"}


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
