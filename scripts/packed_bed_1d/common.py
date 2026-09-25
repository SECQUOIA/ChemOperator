"""Shared packed-bed dataset and DeepONet experiment configuration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from pathlib import Path
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


PATHS = ExamplePaths.from_script(__file__, dataset="packed_bed_1D")
FILE_STEM = "packed_bed_1d"
PROBLEM_ID = "packed_bed_1d"
PROTOCOL_ID = "operator-cartesian-v1"
FIELDS = ("T", "P", "X", "Z", "rhou", "velocity")
CONSTANTS = (
    "length",
    "radius",
    "porosity",
    "tortuosity",
    "particle_diameter",
    "specific_surface_area",
    "inlet_velocity",
    "wall_temperature",
    "heat_transfer_coefficient",
    "solve_energy",
    "membrane_present",
    "membrane_permeability",
    "membrane_thickness",
    "sweep_pressure",
    "inlet_nh3_mole_fraction",
)
UNITS = {
    "T": "K",
    "P": "Pa",
    "X": "-",
    "Z": "-",
    "rhou": "kg/(m2 s)",
    "velocity": "m/s",
}

COORDINATE_STRIDE = 1
RESAMPLE_POINTS = 128
SEED = 7
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


def raw_dataset(data_dir: Path, split: str, stride: int) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        data_dir / f"{FILE_STEM}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="z",
        input_fields=FIELDS,
        output_fields=FIELDS,
        constant_inputs=CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        index_stride=stride,
        dtype=torch.float32,
    )


def limited(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None:
        return dataset
    return Subset(dataset, range(min(maximum, len(dataset))))


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
        coordinate_name="z",
        include_constants=True,
        resample_points=RESAMPLE_POINTS,
        coordinate_mode="relative",
    )


def open_raw(split: str) -> ChemOperatorDataset:
    return raw_dataset(PATHS.data, split, COORDINATE_STRIDE)


def prepare_normalizer() -> ZScoreNormalizer:
    raw = open_raw("train")
    try:
        normalizer = fit_zscore_normalizer(
            DataLoader(
                limited(raw, MAX_TRAJECTORIES),
                batch_size=None,
                shuffle=False,
            ),
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
    train_raw, valid_raw = open_raw("train"), open_raw("valid")
    try:
        yield (
            adapter(limited(train_raw, MAX_TRAJECTORIES), normalizer),
            adapter(limited(valid_raw, MAX_TRAJECTORIES), normalizer),
        )
    finally:
        train_raw.close()
        valid_raw.close()


@contextmanager
def final_data(
    *,
    normalizer: ZScoreNormalizer,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter, DeepONetAdapter]]:
    raw = tuple(open_raw(split) for split in ("train", "valid", "test"))
    try:
        yield tuple(
            adapter(limited(item, MAX_TRAJECTORIES), normalizer) for item in raw
        )  # type: ignore[misc]
    finally:
        for item in raw:
            item.close()


def experiment_spec(model_id: str) -> ExperimentSpec:
    return ExperimentSpec(
        problem_id=PROBLEM_ID,
        model_id=model_id,
        benchmark_protocol_id=PROTOCOL_ID,
        dataset_fingerprints={
            name: fingerprint_path(PATHS.data / f"{FILE_STEM}_{split}.h5")
            for name, split in (
                ("train", "train"),
                ("validation", "valid"),
                ("test", "test"),
            )
        },
        fields=FIELDS,
        channels=FIELDS,
        units=UNITS,
        coordinates=("z",),
        selected_test_case_ids=range(RECONSTRUCTION_CASES),
        tuning_budget={"samples": NUM_SAMPLES, "epochs": MAX_EPOCHS},
        project_root=PATHS.root,
    )


def validate_stages(stages: WorkflowStages) -> None:
    if stages.generate:
        raise ValueError("Use the packed-bed dataset-generation script first.")
    if stages.plot:
        raise ValueError("Use scripts/packed_bed_1d/plot.py for saved runs.")
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
