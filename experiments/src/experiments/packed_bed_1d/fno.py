"""Tune and train a NeuralOperator FNO for packed-bed axial profiles."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path
import time
from typing import Any, Iterator, Mapping

import numpy as np
from neuralop.models import FNO
from ray import tune
import torch
from torch.utils.data import DataLoader, Dataset

from chem_operator.datasets import TrajectoryResamplingDataset
from chem_operator.experiments import (
    EvaluationOutcome,
    ExperimentRunner,
    LossTerm,
    OperatorTrainer,
    StreamingRegressionMetrics,
    Tuner,
    WorkflowStages,
    final_epochs,
    resolve_train_config,
    run_context_from_namespace,
    train_final,
)
from chem_operator.normalization import ZScoreNormalizer

from experiments.packed_bed_1d.common import (
    CONSTANTS,
    FIELDS,
    PATHS,
    PIN_MEMORY,
    PROBLEM_ID,
    RESAMPLE_POINTS,
    SEED,
    experiment_spec,
    limited,
    open_raw,
    parse_args,
    prepare_normalizer,
    tuning_settings,
)


MODEL_ID = "fno"
GRID_POINTS = RESAMPLE_POINTS
BED_LENGTH = 0.05


class PackedBedFNOAdapter(Dataset):
    """Broadcast inlet state and constants over a resampled axial grid."""

    def __init__(self, dataset: Dataset, normalizer: ZScoreNormalizer) -> None:
        if not len(dataset):
            raise ValueError("Packed-bed FNO dataset must contain a trajectory.")
        self.dataset = dataset
        self.normalizer = normalizer
        self.output_channels = sum(normalizer.means[name].numel() for name in FIELDS)
        self.input_channels = self.output_channels + len(CONSTANTS)
        self.labels = tuple(
            name if normalizer.means[name].numel() == 1 else f"{name}[{index}]"
            for name in FIELDS
            for index in range(normalizer.means[name].numel())
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        coordinate = torch.cat(
            (
                sample["input_coordinates"]["z"].reshape(-1),
                sample["output_coordinates"]["z"].reshape(-1),
            )
        )
        expected_grid = torch.linspace(
            0.0, BED_LENGTH, GRID_POINTS, dtype=coordinate.dtype
        )
        if coordinate.shape != (GRID_POINTS,) or not torch.allclose(
            coordinate, expected_grid, rtol=1e-6, atol=1e-7
        ):
            raise ValueError("Packed-bed trajectory must use the shared axial grid.")

        def fields(window: str, points: int) -> torch.Tensor:
            values = sample[f"{window}_fields"]
            if any(values[name].shape[0] != points for name in FIELDS):
                raise ValueError(f"Packed-bed {window} fields do not match the grid.")
            packed = torch.cat(
                tuple(values[name].reshape(points, -1) for name in FIELDS), dim=-1
            )
            if packed.shape != (points, self.output_channels):
                raise ValueError(f"Packed-bed {window} fields have wrong channels.")
            return packed

        initial_state = fields("input", 1).squeeze(0)
        target = torch.cat(
            (initial_state.unsqueeze(0), fields("output", GRID_POINTS - 1))
        )
        constants = torch.stack(
            tuple(sample["constant_inputs"][name].reshape(()) for name in CONSTANTS)
        )
        if not torch.isfinite(initial_state).all() or not torch.isfinite(target).all():
            raise ValueError("Packed-bed state contains NaN or infinity.")
        if not torch.isfinite(constants).all():
            raise ValueError("Packed-bed constants contain NaN or infinity.")

        normalized_input = torch.cat(
            (
                self.normalizer.normalize_flattened(initial_state, "variable"),
                self.normalizer.normalize_flattened(constants, "constant"),
            )
        )
        return {
            "x": normalized_input[:, None].expand(-1, GRID_POINTS),
            "y": self.normalizer.normalize_flattened(
                target, "variable"
            ).transpose(0, 1).contiguous(),
            "z": coordinate,
        }

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Return physical values with axial position before channel."""
        if (
            output.ndim < 2
            or output.shape[-2:] != (self.output_channels, GRID_POINTS)
        ):
            raise ValueError("FNO output shape differs from packed-bed data.")
        return self.normalizer.denormalize_flattened(
            output.transpose(-2, -1), "variable"
        )


def resampled(
    raw: Dataset, maximum: int | None, *, cache: bool = False
) -> TrajectoryResamplingDataset:
    """Resample the first *maximum* cases of one raw split onto the shared axial grid.

    Raw arrays are never preloaded: the resampling cache serves every sample
    after the first epoch, so a raw copy would only use memory.
    """
    return TrajectoryResamplingDataset(
        limited(raw, maximum),
        coordinate_name="z",
        target_coordinate=torch.linspace(0.0, BED_LENGTH, GRID_POINTS),
        cache=cache,
    )


def resampled_split(data_dir: Path, split: str) -> TrajectoryResamplingDataset:
    """Return an uncached, complete resampled split for ``--data-cache`` estimates."""
    return resampled(open_raw(data_dir, split), None)


@contextmanager
def opened_data(
    splits: tuple[str, ...], normalizer: ZScoreNormalizer, args: argparse.Namespace
) -> Iterator[tuple[PackedBedFNOAdapter, ...]]:
    cache_data = getattr(args, "cache_data", False)
    raw = []
    try:
        for split in splits:
            raw.append(open_raw(args.data_dir, split))
        yield tuple(
            PackedBedFNOAdapter(
                resampled(item, args.max_cases, cache=cache_data), normalizer
            )
            for item in raw
        )
    finally:
        for item in raw:
            item.close()


def model_from_config(
    config: Mapping[str, Any], device: torch.device, output_channels: int
) -> FNO:
    return FNO(
        n_modes=(int(config["modes"]),),
        in_channels=output_channels + len(CONSTANTS),
        out_channels=output_channels,
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
    ).to(device)


def search_space() -> dict[str, Any]:
    return {
        "modes": tune.choice([8, 16, 32]),
        "hidden_channels": tune.choice([32, 64]),
        "n_layers": tune.choice([3, 4, 5]),
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([16, 32, 64]),
    }


def relative_l2(
    prediction: torch.Tensor, target: torch.Tensor, _: Mapping[str, Any]
) -> torch.Tensor:
    error = torch.linalg.vector_norm((prediction - target).flatten(1), dim=1)
    reference = torch.linalg.vector_norm(target.flatten(1), dim=1)
    return (error / reference.clamp_min(1e-8)).mean()


def make_trainer(
    config: Mapping[str, Any], context: Any, normalizer: ZScoreNormalizer
) -> OperatorTrainer:
    output_channels = sum(normalizer.means[name].numel() for name in FIELDS)
    return OperatorTrainer(
        lambda values: model_from_config(values, context.device, output_channels),
        config,
        loss_terms=(LossTerm("data_loss", relative_l2),),
        selection_metric="data_loss",
        pin_memory=PIN_MEMORY,
        checkpoint_metadata={
            "normalizer": normalizer.state_dict(),
            "input_channels": output_channels + len(CONSTANTS),
            "output_channels": output_channels,
            "grid_points": GRID_POINTS,
            "grid_end_m": BED_LENGTH,
        },
    )


def evaluate(
    trainer: OperatorTrainer,
    data: PackedBedFNOAdapter,
    context: Any,
    *,
    reconstruction_cases: int = 2,
) -> EvaluationOutcome:
    """Score the full test set and save bounded plot arrays."""
    if reconstruction_cases < 1:
        raise ValueError("reconstruction_cases must be positive.")
    if trainer.model is None:
        raise RuntimeError("Train or load an FNO before evaluation.")
    metrics = StreamingRegressionMetrics()
    arrays: dict[str, list[Any]] = {
        "case_ids": [],
        "coordinates": [],
        "reference": [],
        "prediction": [],
    }
    started = time.perf_counter()
    predicted = trainer.predict(data, context)
    inference_seconds = time.perf_counter() - started
    loader = DataLoader(data, batch_size=int(trainer.config["batch_size"]))
    offset = 0
    for batch in loader:
        count = len(batch["y"])
        prediction = data.denormalize_output(predicted[offset : offset + count])
        reference = data.denormalize_output(batch["y"])
        metrics.update(prediction, reference)
        take = min(reconstruction_cases - len(arrays["case_ids"]), count)
        for index in range(max(0, take)):
            arrays["case_ids"].append(offset + index)
            arrays["coordinates"].append(batch["z"][index].numpy())
            arrays["reference"].append(reference[index].numpy())
            arrays["prediction"].append(prediction[index].numpy())
        offset += count
    return EvaluationOutcome(
        metrics=metrics.compute(),
        reconstructions={
            **{name: np.asarray(values) for name, values in arrays.items()},
            "labels": np.asarray(data.labels),
        },
        inference_seconds=inference_seconds,
    )


def main() -> None:
    args = parse_args(cache_factory=resampled_split)
    stages = WorkflowStages.from_namespace(args)
    normalizer = prepare_normalizer(args)
    context = run_context_from_namespace(
        args, problem=PROBLEM_ID, model=MODEL_ID, seed=SEED,
        provenance={"max_cases": args.max_cases},
    )
    runner = ExperimentRunner(context, experiment_spec(MODEL_ID, args))
    tuning = None

    if stages.tune:
        PATHS.ray.mkdir(parents=True, exist_ok=True)

        def trial_data(_context: Any):
            return opened_data(("train", "valid"), normalizer, args)

        def trial_trainer(config: Mapping[str, Any], *, context: Any):
            return make_trainer(config, context, normalizer)

        tuning = runner.tune(
            Tuner(
                trial_trainer,
                search_space(),
                config=tuning_settings(args, context, metric="best_valid_data_loss"),
                dataset_factory=trial_data,
            ),
            None,
            None,
            experiment_name=f"{PROBLEM_ID}_{MODEL_ID}_{context.paths.run_dir.name}",
        )

    if stages.train:
        config, source = resolve_train_config(stages.train_config, runner, tuning=tuning)
        config["epochs"] = final_epochs(args)

        def train_one(run_runner, run_context, run_config):
            trainer = make_trainer(run_config, run_context, normalizer)
            with opened_data(("train", "valid", "test"), normalizer, args) as datasets:
                return run_runner.run(
                    trainer,
                    *datasets,
                    config=run_config,
                    tuning=tuning if run_runner is runner else None,
                    evaluator=partial(evaluate, reconstruction_cases=args.plot_cases),
                )

        train_final(
            runner, stages, config=config, tuning=tuning,
            train_one=train_one, train_config_source=source,
        )
    print(f"Run artifacts: {context.paths.run_dir}")


if __name__ == "__main__":
    main()
