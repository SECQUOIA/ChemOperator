"""Tune and train a NeuralOperator FNO for full CSTR trajectories."""

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
    OperatorTrainer,
    LossTerm,
    RayRuntimeConfig,
    StreamingRegressionMetrics,
    Tuner,
    TuningConfig,
    WorkflowStages,
    run_context_from_namespace,
)
from chem_operator.models import relative_l2_loss
from chem_operator.normalization import ZScoreNormalizer

from scripts.cstr.common import (
    CPUS_PER_TRIAL,
    DATALOADER_WORKERS,
    GPUS_PER_TRIAL,
    LOG_TIME_POINTS,
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
FINAL_EPOCHS = 2 * MAX_EPOCHS


class CSTRFNOAdapter(Dataset):
    """Broadcast the normalized initial state over the shared log-time grid."""

    def __init__(self, dataset: Dataset, normalizer: ZScoreNormalizer) -> None:
        self.dataset = dataset
        self.normalizer = normalizer
        self.n_species = int(normalizer.means["X"].numel())
        self.labels = ("T", "P") + tuple(
            f"X[{index}]" for index in range(self.n_species)
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.dataset[index]
        initial = torch.cat(
            (
                sample["input_fields"]["T"][-1:].reshape(-1),
                sample["input_fields"]["P"][-1:].reshape(-1),
                sample["input_fields"]["X"][-1].reshape(-1),
            )
        )
        future = torch.cat(
            (
                sample["output_fields"]["T"].reshape(-1, 1),
                sample["output_fields"]["P"].reshape(-1, 1),
                sample["output_fields"]["X"],
            ),
            dim=-1,
        )
        target = torch.cat((initial.unsqueeze(0), future), dim=0)
        coordinate = torch.cat(
            (
                sample["input_coordinates"]["t"].reshape(-1)[-1:],
                sample["output_coordinates"]["t"].reshape(-1),
            )
        )
        if (
            initial.numel() != self.n_species + 2
            or target.shape != (LOG_TIME_POINTS, self.n_species + 2)
            or coordinate.numel() != LOG_TIME_POINTS
        ):
            raise ValueError("CSTR trajectory does not match the shared FNO grid.")
        state = self.normalizer.normalize_flattened(initial, "variable")
        coefficient = self.normalizer.normalize(
            sample["constant_inputs"]["heat_transfer_coefficient"],
            "heat_transfer_coefficient",
        ).reshape(1)
        inputs = torch.cat((state, coefficient)).unsqueeze(-1)
        return {
            "x": inputs.expand(-1, LOG_TIME_POINTS),
            "y": self.normalizer.normalize_flattened(
                target, "variable"
            ).transpose(0, 1).contiguous(),
            "t": coordinate,
        }

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Return physical values with time before channel."""
        if output.shape[-2] != self.n_species + 2:
            raise ValueError("FNO output channel count differs from CSTR data.")
        return self.normalizer.denormalize_flattened(
            output.transpose(-2, -1), "variable"
        )


@contextmanager
def opened_data(
    splits: tuple[str, ...], normalizer: ZScoreNormalizer
) -> Iterator[tuple[CSTRFNOAdapter, ...]]:
    raw = []
    try:
        for split in splits:
            raw.append(raw_dataset(split))
        yield tuple(CSTRFNOAdapter(limited(item), normalizer) for item in raw)
    finally:
        for item in raw:
            item.close()


def model_from_config(
    config: Mapping[str, Any], device: torch.device, n_species: int
) -> FNO:
    return FNO(
        n_modes=(int(config["modes"]),),
        in_channels=n_species + 3,
        out_channels=n_species + 2,
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
    ).to(device)


def search_space() -> dict[str, Any]:
    return {
        "modes": tune.choice([12, 24, 48]),
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
    return relative_l2_loss(prediction, target, eps=1e-8)


def make_trainer(
    config: Mapping[str, Any],
    context: Any,
    normalizer: ZScoreNormalizer,
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
            "input_channels": n_species + 3,
            "output_channels": n_species + 2,
            "time_points": LOG_TIME_POINTS,
        },
    )


def evaluate(
    trainer: OperatorTrainer,
    data: CSTRFNOAdapter,
    context: Any,
    *,
    reconstruction_cases: int = RECONSTRUCTION_CASES,
) -> EvaluationOutcome:
    """Score the full test set and retain bounded DeepONet-layout plot arrays."""
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
            arrays["coordinates"].append(batch["t"][index].numpy())
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
        raise ValueError("CSTR FNO training requires --train-config best.")
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
