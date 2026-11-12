"""Tune and train data-only or physics-informed PhysicsNeMo pipe-flow DeepONets."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from physicsnemo.models.mlp.fully_connected import FullyConnected
from physicsnemo.sym.eq.phy_informer import PhysicsInformer
from ray import tune
from torch import nn
from torch.utils.data import DataLoader

from chem_operator.experiments import (
    ArtifactStore,
    EvaluationOutcome,
    LossTerm,
    OperatorTrainer,
    StreamingRegressionMetrics,
    parse_operator_args,
    run_operator,
)
from chem_operator.reactors.pipe_flow.dataset_generator import HagenPoiseuille, hagen_poiseuille_velocity
from scripts.pipe_flow.common import (
    BRANCH_NAMES,
    PATHS,
    Normalization,
    PhysicsPipeDataset,
    fit_physics_normalization,
    load_physics_split,
    physics_experiment_spec,
)


MODEL_IDS = {"data": "physics_deeponet_data", "physics": "physics_deeponet_physics"}
SEED = 42
PHYSICS_WEIGHT_RANGE = (1.0e-4, 1.0)


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


def search_space() -> dict[str, Any]:
    return {
        "width": tune.choice([64, 128]),
        "depth": tune.choice([2, 3]),
        "latent_width": tune.choice([16, 32]),
        "activation": tune.choice(["silu", "gelu", "tanh"]),
        "learning_rate": tune.loguniform(1.0e-4, 3.0e-3),
        "weight_decay": tune.loguniform(1.0e-8, 1.0e-4),
        "batch_size": tune.choice([64, 128]),
        "seed": SEED,
    }


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
    return OperatorTrainer(lambda cfg: model_from_config(cfg,context.device),config,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("data", "physics", "both"), default="both")
    parser.add_argument("--physics-samples", type=int, default=5)
    args = parse_operator_args(PATHS, epochs=40, tune_epochs=20, samples=8, parser=parser)
    if args.physics_samples < 1:
        parser.error("--physics-samples must be positive.")

    verify_exact_residual()
    statistics = fit_physics_normalization(load_physics_split(args.data_dir, "train", args.max_cases))

    def dataset_factory(split: str):
        data = PhysicsPipeDataset(
            load_physics_split(args.data_dir, split, args.max_cases).normalized(statistics)
        )
        return data, data

    data_context = None
    variants = ("data", "physics") if args.variant == "both" else (args.variant,)
    for variant in variants:
        model_id = MODEL_IDS[variant]
        space = dict(search_space(), physics_weight=0.0)
        if variant == "physics":
            if data_context is not None:
                space = ArtifactStore(data_context).read_best_config()
            space["physics_weight"] = tune.loguniform(*PHYSICS_WEIGHT_RANGE)

        variant_args = argparse.Namespace(**vars(args))
        if variant == "physics" and args.variant == "both":
            variant_args.samples = args.physics_samples
        context = run_operator(
            variant_args, PATHS, physics_experiment_spec(variant_args, model_id), dataset_factory,
            lambda config, run_context, selected=variant: make_trainer(
                dict(config, physics_weight=0.0) if selected == "data" else config, run_context, statistics
            ),
            space,
            lambda trainer, data, run_context: evaluate_run(trainer, data, run_context, statistics, args.plot_cases),
            seed=SEED,
        )
        if variant == "data":
            data_context = context


if __name__ == "__main__":
    main()
