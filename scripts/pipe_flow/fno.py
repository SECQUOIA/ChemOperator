"""Tune and train data-only or physics-informed NeuralOperator pipe-flow FNOs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from neuralop.models import FNO
from ray import tune
from torch import nn
from torch.utils.data import Dataset

from chem_operator.experiments import (
    ArtifactStore,
    LossTerm,
    OperatorTrainer,
    evaluate_fields,
    parse_operator_args,
    run_operator,
)
from scripts.pipe_flow.common import (
    BRANCH_NAMES,
    FIELDS,
    FILE_STEM,
    PATHS,
    PROBLEM_ID,
    UNITS,
    Normalization,
    PipeData,
    fit_physics_normalization,
    load_physics_split,
)
from chem_operator.experiments import operator_spec


MODEL_IDS = {"data": "fno_data", "physics": "fno_physics"}
SEED = 42


class PipeFNODataset(Dataset):
    """Broadcast normalized controls over each case's radial profile."""

    def __init__(self, data: PipeData, statistics: Normalization) -> None:
        self.data = data.normalized(statistics)
        self.statistics = statistics
        if not len(data.branch) or data.target.ndim != 3 or data.target.shape[-1] != 1:
            raise ValueError("Pipe FNO data must contain velocity profiles.")
        points = data.target.shape[1]
        if points < 3 or data.coordinates.shape != (len(data.branch), points, 1):
            raise ValueError("Pipe FNO requires aligned radial grids of at least 3 points.")
        if data.branch.shape != (len(data.branch), len(BRANCH_NAMES)):
            raise ValueError("Pipe controls do not match the FNO input channels.")

    def __len__(self) -> int:
        return len(self.data.branch)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        data = self.data
        radius = data.radius[index].reshape(())
        viscosity = data.viscosity[index].reshape(())
        pressure_gradient = data.pressure_gradient[index].reshape(())
        radial = data.coordinates[index, :, 0]
        spacing = radial[1:] - radial[:-1]
        if (
            not torch.isfinite(radial).all()
            or not torch.isfinite(data.target[index]).all()
            or not torch.isfinite(data.branch[index]).all()
            or not torch.isfinite(pressure_gradient)
            or radius <= 0
            or viscosity <= 0
            or not torch.all(spacing > 0)
            or not torch.allclose(spacing, spacing[:1].expand_as(spacing), rtol=1e-4, atol=0)
            or not torch.isclose(radial[0], radial.new_zeros(()), atol=1e-8)
            or not torch.isclose(radial[-1], radius, rtol=1e-5, atol=1e-8)
        ):
            raise ValueError("Pipe FNO requires a finite, uniform centerline-to-wall grid.")
        points = radial.numel()
        return {
            "x": data.branch[index, :, None].expand(-1, points),
            "y": data.target_normalized[index].transpose(0, 1).contiguous(),
            "r": radial,
            "radius": radius,
            "viscosity": viscosity,
            "pressure_gradient": pressure_gradient,
        }

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        if output.shape[-2] != 1:
            raise ValueError("Pipe FNO output must have one velocity channel.")
        return (
            output * self.statistics.velocity_std.to(output.device)
            + self.statistics.velocity_mean.to(output.device)
        )

    def close(self) -> None:
        """The HDF5 reader is closed by load_physics_split."""


def model_from_config(config: Mapping[str, Any], device: torch.device) -> FNO:
    return FNO(
        n_modes=(int(config["modes"]),),
        in_channels=len(BRANCH_NAMES),
        out_channels=1,
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
    ).to(device)


def search_space() -> dict[str, Any]:
    return {
        "modes": tune.choice([8, 16, 24]),
        "hidden_channels": tune.choice([16, 32]),
        "n_layers": tune.choice([3, 4]),
        "learning_rate": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-4),
        "batch_size": tune.choice([8, 16]),
        "seed": SEED,
    }


def physics_losses(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    statistics: Normalization,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dimensionless Hagen–Poiseuille momentum and boundary MSEs."""
    if prediction.ndim != 3 or prediction.shape[1] != 1 or prediction.shape[-1] < 3:
        raise ValueError("Physics loss requires a full radial velocity profile.")
    velocity = (
        prediction[:, 0] * statistics.velocity_std.to(prediction.device)
        + statistics.velocity_mean.to(prediction.device)
    )
    radial = batch["r"]
    radius = batch["radius"]
    viscosity = batch["viscosity"]
    gradient = batch["pressure_gradient"]
    spacing = (radial[:, 1] - radial[:, 0])[:, None]
    first = (velocity[:, 2:] - velocity[:, :-2]) / (2 * spacing)
    second = (
        velocity[:, 2:] - 2 * velocity[:, 1:-1] + velocity[:, :-2]
    ) / spacing.square()
    residual = (
        viscosity[:, None] * (radial[:, 1:-1] * second + first)
        - gradient[:, None] * radial[:, 1:-1]
    )
    force_scale = (gradient.abs() * radius).clamp_min(1e-8)
    pde_loss = (residual / force_scale[:, None]).square().mean()

    speed_scale = (
        gradient.abs() * radius.square() / (4 * viscosity)
    ).clamp_min(1e-8)
    center_gradient = (
        -3 * velocity[:, 0] + 4 * velocity[:, 1] - velocity[:, 2]
    ) / (2 * spacing[:, 0])
    center_loss = (center_gradient * radius / speed_scale).square().mean()
    wall_loss = (velocity[:, -1] / speed_scale).square().mean()
    return pde_loss, 0.5 * (center_loss + wall_loss)


def make_trainer(
    config: Mapping[str, Any], context: Any, statistics: Normalization
) -> OperatorTrainer:
    weight = float(config.get("physics_weight", 0.0))

    def pde(prediction: torch.Tensor, _target: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return physics_losses(prediction, batch, statistics)[0]

    def boundary(prediction: torch.Tensor, _target: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return physics_losses(prediction, batch, statistics)[1]

    def physical_metrics(prediction: torch.Tensor, target: torch.Tensor, _batch: Any):
        scale = statistics.velocity_std.to(prediction.device)
        offset = statistics.velocity_mean.to(prediction.device)
        return prediction * scale + offset, target * scale + offset

    return OperatorTrainer(
        lambda values: model_from_config(values, context.device),
        config,
        loss_terms=(
            LossTerm("data_loss", lambda prediction, target, _: nn.functional.mse_loss(prediction, target)),
            LossTerm("pde_loss", pde, weight),
            LossTerm("boundary_loss", boundary, weight),
        ),
        metric_adapter=physical_metrics,
        selection_metric="data_loss",
        checkpoint_metadata={
            "normalizer": {
                name: getattr(statistics, name).cpu()
                for name in ("branch_mean", "branch_std", "velocity_mean", "velocity_std")
            },
            "input_channels": list(BRANCH_NAMES),
            "output_channels": list(FIELDS),
            "pde_name": "HagenPoiseuille",
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("data", "physics", "both"), default="both")
    parser.add_argument("--physics-samples", type=int, default=5)
    args = parse_operator_args(PATHS, epochs=40, tune_epochs=20, samples=8, parser=parser)
    if args.physics_samples < 1:
        parser.error("--physics-samples must be positive.")

    training = load_physics_split(args.data_dir, "train", args.max_cases)
    statistics = fit_physics_normalization(training)

    def dataset_factory(split: str):
        data = PipeFNODataset(
            load_physics_split(args.data_dir, split, args.max_cases), statistics
        )
        return data, data

    data_context = None
    variants = ("data", "physics") if args.variant == "both" else (args.variant,)
    for variant in variants:
        model_id = MODEL_IDS[variant]
        space = search_space()
        if variant == "data":
            space["physics_weight"] = 0.0
        elif data_context is not None:
            space = ArtifactStore(data_context).read_best_config()
        if variant == "physics":
            space["physics_weight"] = tune.loguniform(1e-4, 1.0)

        variant_args = argparse.Namespace(**vars(args))
        if variant == "physics" and args.variant == "both":
            variant_args.samples = args.physics_samples
        spec = operator_spec(
            variant_args, PATHS, PROBLEM_ID, model_id, FILE_STEM,
            FIELDS, UNITS, ("r",),
        )
        context = run_operator(
            variant_args, PATHS, spec, dataset_factory,
            lambda config, run_context: make_trainer(config, run_context, statistics),
            space,
            lambda trainer, data, run_context: evaluate_fields(
                trainer, data, run_context,
                labels=FIELDS, coordinate_names=("r",), cases=args.plot_cases,
                extra_metrics=lambda trained, held_out, ctx: {
                    key: value for key, value in trained.evaluate_losses(held_out, ctx).items()
                    if key in ("pde_loss", "boundary_loss")
                },
            ),
            seed=SEED,
            selection_metric="data_loss",
        )
        if variant == "data":
            data_context = context


if __name__ == "__main__":
    main()
