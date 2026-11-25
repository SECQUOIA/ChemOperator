"""Shared pipe-flow dataset and DeepONet experiment configuration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from typing import Iterator

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from chem_operator.datasets import (
    ChemOperatorDataset,
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    TargetTransformConfig,
)
from chem_operator.example_paths import ExamplePaths
from chem_operator.experiments import (
    DeepONetTuningSettings,
    ExperimentSpec,
    RayRuntimeConfig,
    RunContext,
    add_run_arguments,
    add_workflow_arguments,
    fingerprint_path,
    parse_workflow_args,
    resolve_data_cache,
)
from chem_operator.models import (
    DeepONetAdapter,
    fit_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = ExamplePaths.from_script(__file__, dataset="pipe_flow")
PROBLEM_ID = "pipe_flow"
PROTOCOL_ID = "operator-cartesian-v1"
FIELDS = ("velocity",)
CONSTANTS = ("radius", "length", "dynamic_viscosity", "pressure_drop")
UNITS = {"velocity": "m/s"}

SEED = 42
RECONSTRUCTION_CASES = 3
MAX_TRAJECTORIES: int | None = None

MAX_EPOCHS = 20
NUM_SAMPLES = 15
MAX_CONCURRENT_TRIALS = 1
CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
PIN_MEMORY = bool(GPUS_PER_TRIAL)
RAY_OBJECT_STORE_BYTES = 100 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    add_run_arguments(
        parser,
        default_runs_root=PATHS.root / "artifacts" / "runs",
    )
    args = parse_workflow_args(parser)
    resolve_data_cache(args, raw_dataset)
    return args


def raw_dataset(split: str, *, preload: bool = False) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        PATHS.data / f"{PROBLEM_ID}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="r",
        input_fields=FIELDS,
        output_fields=FIELDS,
        constant_inputs=CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
        preload=preload,
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
        coordinate_name="r",
        include_constants=True,
        include_input_state=False,
        coordinate_mode="relative",
    )


def prepare_normalizer() -> ZScoreNormalizer:
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
    cache_data: bool = False,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter]]:
    del context
    train_raw = raw_dataset("train", preload=cache_data)
    valid_raw = raw_dataset("valid", preload=cache_data)
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
    cache_data: bool = False,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter, DeepONetAdapter]]:
    raw = tuple(
        raw_dataset(split, preload=cache_data) for split in ("train", "valid", "test")
    )
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
            name: fingerprint_path(PATHS.data / f"{PROBLEM_ID}_{split}.h5")
            for name, split in (
                ("train", "train"),
                ("validation", "valid"),
                ("test", "test"),
            )
        },
        fields=FIELDS,
        channels=FIELDS,
        units=UNITS,
        coordinates=("r",),
        selected_test_case_ids=range(RECONSTRUCTION_CASES),
        tuning_budget={"samples": NUM_SAMPLES, "epochs": MAX_EPOCHS},
        project_root=PATHS.root,
    )


def tuning_settings() -> DeepONetTuningSettings:
    return DeepONetTuningSettings(
        max_epochs=MAX_EPOCHS,
        num_samples=NUM_SAMPLES,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        cpus_per_trial=CPUS_PER_TRIAL,
        gpus_per_trial=GPUS_PER_TRIAL,
        ray_runtime=RayRuntimeConfig(
            num_cpus=CPUS_PER_TRIAL,
            num_gpus=GPUS_PER_TRIAL,
            object_store_memory=RAY_OBJECT_STORE_BYTES,
            temp_dir=PATHS.ray.resolve(),
        ),
    )
