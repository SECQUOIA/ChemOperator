"""Tune and train a NeuralOperator FNO for PFR-chain axial profiles."""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial
import os
from pathlib import Path
import sys
import time
from typing import Any, Iterator, Mapping

os.environ.setdefault("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from neuralop.models import FNO
from ray import tune
import torch
from torch.utils.data import DataLoader, Dataset

from chem_operator.experiments import (
    EvaluationOutcome,
    ExperimentRunner,
    LossTerm,
    OperatorTrainer,
    RayRuntimeConfig,
    StreamingRegressionMetrics,
    Tuner,
    TuningConfig,
    WorkflowStages,
    run_context_from_namespace,
)
from chem_operator.normalization import ZScoreNormalizer

from scripts.pfr.common import (
    CONSTANTS,
    CPUS_PER_TRIAL,
    DATALOADER_WORKERS,
    GPUS_PER_TRIAL,
    MAX_CONCURRENT_TRIALS,
    MAX_EPOCHS,
    NUM_SAMPLES,
    PATHS,
    PIN_MEMORY,
    PROBLEM_ID,
    RAY_OBJECT_STORE_BYTES,
    RECONSTRUCTION_CASES,
    SEED,
    TIME_BUDGET_S,
    experiment_spec,
    limited,
    parse_args,
    prepare_normalizer,
    raw_dataset,
)


MODEL_ID = "fno"
GRID_POINTS = 50
FINAL_EPOCHS = 2 * MAX_EPOCHS


class PFRFNOAdapter(Dataset):
    """Broadcast inlet state and case constants over the shared axial grid."""

    def __init__(self, dataset: Dataset, normalizer: ZScoreNormalizer) -> None:
        if not len(dataset):
            raise ValueError("PFR FNO dataset must contain at least one trajectory.")
        self.dataset = dataset
        self.normalizer = normalizer
        self.n_species = int(normalizer.means["X"].numel())
        self.output_channels = self.n_species + 3
        self.input_channels = self.output_channels + len(CONSTANTS)
        self.labels = (
            ("T", "P")
            + tuple(f"X[{index}]" for index in range(self.n_species))
            + ("velocity",)
        )
        self.grid = self._coordinate(dataset[0])

    @staticmethod
    def _coordinate(sample: Mapping[str, Any]) -> torch.Tensor:
        inlet = sample["input_coordinates"]["z"].reshape(-1)
        future = sample["output_coordinates"]["z"].reshape(-1)
        coordinate = torch.cat((inlet, future))
        if (
            inlet.numel() != 1
            or coordinate.numel() != GRID_POINTS
            or not torch.isfinite(coordinate).all()
            or not torch.all(coordinate[1:] > coordinate[:-1])
        ):
            raise ValueError("PFR trajectory does not have a valid 50-point axial grid.")
        return coordinate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        coordinate = self._coordinate(sample)
        if not torch.allclose(coordinate, self.grid, rtol=1e-6, atol=1e-12):
            raise ValueError("PFR FNO trajectories must use the same axial grid.")

        inlet = sample["input_fields"]
        future = sample["output_fields"]
        if any(tuple(inlet[name].shape) != (1,) for name in ("T", "P", "velocity")):
            raise ValueError("PFR inlet scalar fields must each contain one value.")
        if tuple(inlet["X"].shape) != (1, self.n_species):
            raise ValueError("PFR inlet species count differs from the normalizer.")
        if any(
            tuple(future[name].shape) != (GRID_POINTS - 1,)
            for name in ("T", "P", "velocity")
        ) or tuple(future["X"].shape) != (GRID_POINTS - 1, self.n_species):
            raise ValueError("PFR future fields do not match the FNO grid or channels.")

        initial_state = torch.cat(
            (inlet["T"], inlet["P"], inlet["X"].reshape(-1), inlet["velocity"])
        )
        future_state = torch.cat(
            (
                future["T"].unsqueeze(-1),
                future["P"].unsqueeze(-1),
                future["X"],
                future["velocity"].unsqueeze(-1),
            ),
            dim=-1,
        )
        target = torch.cat((initial_state.unsqueeze(0), future_state), dim=0)
        constants = torch.stack(
            tuple(sample["constant_inputs"][name].reshape(()) for name in CONSTANTS)
        )
        if not torch.isfinite(initial_state).all() or not torch.isfinite(target).all():
            raise ValueError("PFR state contains NaN or infinity.")
        if not torch.isfinite(constants).all():
            raise ValueError("PFR constants contain NaN or infinity.")

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
        if output.ndim < 2 or output.shape[-2:] != (self.output_channels, GRID_POINTS):
            raise ValueError("FNO output shape differs from PFR data.")
        return self.normalizer.denormalize_flattened(
            output.transpose(-2, -1), "variable"
        )


@contextmanager
def opened_data(
    splits: tuple[str, ...], normalizer: ZScoreNormalizer
) -> Iterator[tuple[PFRFNOAdapter, ...]]:
    raw = []
    try:
        for split in splits:
            raw.append(raw_dataset(split))
        yield tuple(PFRFNOAdapter(limited(item), normalizer) for item in raw)
    finally:
        for item in raw:
            item.close()


def model_from_config(
    config: Mapping[str, Any], device: torch.device, n_species: int
) -> FNO:
    return FNO(
        n_modes=(int(config["modes"]),),
        in_channels=n_species + 3 + len(CONSTANTS),
        out_channels=n_species + 3,
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
    ).to(device)


def search_space() -> dict[str, Any]:
    return {
        "modes": tune.choice([8, 12, 20]),
        "hidden_channels": tune.choice([32, 64]),
        "n_layers": tune.choice([3, 4]),
        "learning_rate": tune.loguniform(5e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([2, 4]),
        "seed": SEED,
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
    n_species = int(normalizer.means["X"].numel())
    return OperatorTrainer(
        lambda values: model_from_config(values, context.device, n_species),
        config,
        loss_terms=(LossTerm("data_loss", relative_l2),),
        selection_metric="data_loss",
        num_workers=DATALOADER_WORKERS,
        pin_memory=PIN_MEMORY,
        checkpoint_metadata={
            "normalizer": normalizer.state_dict(),
            "input_channels": n_species + 3 + len(CONSTANTS),
            "output_channels": n_species + 3,
            "grid_points": GRID_POINTS,
        },
    )


def evaluate(
    trainer: OperatorTrainer,
    data: PFRFNOAdapter,
    context: Any,
    *,
    reconstruction_cases: int = RECONSTRUCTION_CASES,
) -> EvaluationOutcome:
    """Score the full test set and save bounded DeepONet-layout plot arrays."""
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


def tuning_settings() -> TuningConfig:
    return TuningConfig(
        metric="best_valid_data_loss",
        mode="min",
        num_samples=NUM_SAMPLES,
        max_epochs=MAX_EPOCHS,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        time_budget_s=TIME_BUDGET_S,
        resources_per_trial={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
        optuna_seed=SEED,
        ray_runtime=RayRuntimeConfig(
            num_cpus=CPUS_PER_TRIAL,
            num_gpus=GPUS_PER_TRIAL,
            object_store_memory=RAY_OBJECT_STORE_BYTES,
            temp_dir=PATHS.ray.resolve(),
            runtime_env={
                "env_vars": {
                    "PYTHONPATH": os.pathsep.join(
                        filter(
                            None,
                            (
                                str(PATHS.root),
                                str(PATHS.root / "src"),
                                str(PATHS.example),
                                os.environ.get("PYTHONPATH"),
                            ),
                        )
                    )
                }
            },
        ),
    )


def main() -> None:
    args = parse_args(description=__doc__)
    stages = WorkflowStages.from_namespace(args)
    if not (stages.tune or stages.train):
        raise ValueError("Select --tune or --train.")
    if stages.train_config != "best":
        raise ValueError("PFR FNO training requires --train-config best.")
    normalizer = prepare_normalizer()
    context = run_context_from_namespace(
        args, problem=PROBLEM_ID, model=MODEL_ID, seed=SEED
    )
    runner = ExperimentRunner(context, experiment_spec(MODEL_ID))
    tuning = None

    if stages.tune:
        PATHS.ray.mkdir(parents=True, exist_ok=True)

        def trial_data(_context: Any):
            return opened_data(("train", "valid"), normalizer)

        def trial_trainer(config: Mapping[str, Any], *, context: Any):
            return make_trainer(config, context, normalizer)

        tuning = runner.tune(
            Tuner(
                trial_trainer,
                {**search_space(), "epochs": MAX_EPOCHS},
                config=tuning_settings(),
                dataset_factory=trial_data,
            ),
            None,
            None,
            experiment_name=f"{PROBLEM_ID}_{MODEL_ID}_{context.paths.run_dir.name}",
        )

    if stages.train:
        config = (
            dict(tuning.best_config)
            if tuning is not None
            else runner.artifacts.read_best_config()
        )
        config["epochs"] = FINAL_EPOCHS
        trainer = make_trainer(config, context, normalizer)
        with opened_data(("train", "valid", "test"), normalizer) as datasets:
            runner.run(
                trainer,
                *datasets,
                config=config,
                tuning=tuning,
                evaluator=partial(
                    evaluate, reconstruction_cases=RECONSTRUCTION_CASES
                ),
            )
    print(f"Run artifacts: {context.paths.run_dir}")


if __name__ == "__main__":
    main()
