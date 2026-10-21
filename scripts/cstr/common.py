"""Shared CSTR dataset and DeepONet experiment configuration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from typing import Iterator

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from chem_operator.datasets import (
    ChemOperatorDataset,
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    TargetTransformConfig,
    TrajectoryResamplingDataset,
)
from chem_operator.example_paths import ExamplePaths
from chem_operator.experiments import (
    DeepONetTuningSettings,
    ExperimentSpec,
    RayRuntimeConfig,
    RunContext,
    WorkflowStages,
    add_run_arguments,
    add_workflow_arguments,
    fingerprint_path,
)
from chem_operator.models import (
    DeepONetAdapter,
    fit_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = ExamplePaths.from_script(__file__, dataset="cstr")
DATASET_NAME = "cstr_non_isothermal"
PROBLEM_ID = "cstr_non_isothermal"
PROTOCOL_ID = "operator-cartesian-logtime-v1"
FIELDS = ("T", "P", "X")
CONSTANTS = ("heat_transfer_coefficient",)
UNITS = {"T": "K", "P": "Pa", "X": "-"}

COORDINATE_STRIDE = 1
LOG_TIME_POINTS = 501
LOG_TIME_FINAL = 100.0
LOG_TIME_SCALE = 1.0e-2
CACHE_RESAMPLED_TRAJECTORIES = True
SEED = 42
RECONSTRUCTION_CASES = 2
MAX_TRAJECTORIES: int | None = None

MAX_EPOCHS = 40
NUM_SAMPLES = 10
MAX_CONCURRENT_TRIALS = 1
TIME_BUDGET_S = 30 * 60
CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0
PIN_MEMORY = bool(GPUS_PER_TRIAL)
RAY_OBJECT_STORE_BYTES = 100 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    add_run_arguments(
        parser,
        default_runs_root=PATHS.root / "artifacts" / "runs",
    )
    return parser.parse_args()


def log_time_grid() -> np.ndarray:
    normalized = np.linspace(0.0, 1.0, LOG_TIME_POINTS)
    grid = LOG_TIME_SCALE * np.expm1(
        normalized * np.log1p(LOG_TIME_FINAL / LOG_TIME_SCALE)
    )
    grid[0] = 0.0
    grid[-1] = LOG_TIME_FINAL
    return grid


def raw_dataset(split: str) -> TrajectoryResamplingDataset:
    raw = ChemOperatorDataset(
        PATHS.data / f"{DATASET_NAME}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="t",
        input_fields=FIELDS,
        output_fields=FIELDS,
        constant_inputs=CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        index_stride=COORDINATE_STRIDE,
        dtype=torch.float32,
    )
    return TrajectoryResamplingDataset(
        raw,
        coordinate_name="t",
        target_coordinate=log_time_grid(),
        cache=CACHE_RESAMPLED_TRAJECTORIES,
    )


def limited(dataset: Dataset) -> Dataset:
    if MAX_TRAJECTORIES is None:
        return dataset
    return Subset(dataset, range(min(MAX_TRAJECTORIES, len(dataset))))


def adapter(dataset: Dataset, normalizer: ZScoreNormalizer) -> DeepONetAdapter:
    processor = DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=FIELDS,
            constant_field_order=CONSTANTS,
        ),
        normalizer=normalizer,
        normalization_config=NormalizationConfig(enabled=True),
        target_transform=TargetTransformConfig(mode="state"),
    )
    return DeepONetAdapter(
        dataset,
        processor,
        format="cartesian_product",
        coordinate_name="t",
        include_constants=True,
        coordinate_mode="log1p",
        coordinate_scale=LOG_TIME_SCALE,
    )


def resampled_dataset_fingerprint(split: str) -> str:
    source = fingerprint_path(PATHS.data / f"{DATASET_NAME}_{split}.h5")
    payload = {
        "source": source,
        "method": "linear",
        "coordinate": "t",
        "points": LOG_TIME_POINTS,
        "final": LOG_TIME_FINAL,
        "scale": LOG_TIME_SCALE,
        "trunk_transform": "log1p",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_normalizer() -> ZScoreNormalizer:
    """Fit training-only normalization once on the driver."""
    raw = raw_dataset("train")
    try:
        normalizer = fit_zscore_normalizer(
            DataLoader(limited(raw), batch_size=None, shuffle=False),
            FIELDS,
            CONSTANTS,
        )
    finally:
        raw.close()
    return normalizer


@contextmanager
def tuning_data(
    context: RunContext,
    *,
    normalizer: ZScoreNormalizer,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter]]:
    del context
    train_raw, valid_raw = raw_dataset("train"), raw_dataset("valid")
    try:
        yield (
            adapter(limited(train_raw), normalizer),
            adapter(limited(valid_raw), normalizer),
        )
    finally:
        train_raw.close()
        valid_raw.close()


@contextmanager
def final_data(
    *,
    normalizer: ZScoreNormalizer,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter, DeepONetAdapter]]:
    raw = tuple(raw_dataset(split) for split in ("train", "valid", "test"))
    try:
        yield tuple(adapter(limited(item), normalizer) for item in raw)  # type: ignore[misc]
    finally:
        for item in raw:
            item.close()


def experiment_spec(model_id: str) -> ExperimentSpec:
    return ExperimentSpec(
        problem_id=PROBLEM_ID,
        model_id=model_id,
        benchmark_protocol_id=PROTOCOL_ID,
        dataset_fingerprints={
            name: resampled_dataset_fingerprint(split)
            for name, split in (
                ("train", "train"),
                ("validation", "valid"),
                ("test", "test"),
            )
        },
        fields=FIELDS,
        channels=FIELDS,
        units=UNITS,
        coordinates=("t",),
        selected_test_case_ids=range(RECONSTRUCTION_CASES),
        tuning_budget={"samples": NUM_SAMPLES, "epochs": MAX_EPOCHS},
        project_root=PATHS.root,
    )


def validate_stages(stages: WorkflowStages) -> None:
    if stages.train_config != "best":
        raise ValueError("DeepONet training currently requires --train-config best.")


def tuning_settings() -> DeepONetTuningSettings:
    return DeepONetTuningSettings(
        max_epochs=MAX_EPOCHS,
        num_samples=NUM_SAMPLES,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        time_budget_s=TIME_BUDGET_S,
        cpus_per_trial=CPUS_PER_TRIAL,
        gpus_per_trial=GPUS_PER_TRIAL,
        ray_runtime=RayRuntimeConfig(
            num_cpus=CPUS_PER_TRIAL,
            num_gpus=GPUS_PER_TRIAL,
            object_store_memory=RAY_OBJECT_STORE_BYTES,
            temp_dir=PATHS.ray.resolve(),
        ),
    )
