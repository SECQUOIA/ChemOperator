"""Shared PFR-chain dataset and operator experiment configuration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from typing import Iterator

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from chem_operator.datasets import (
    ChemOperatorDataset,
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    TargetTransformConfig,
)
from experiments.paths import example_paths
from chem_operator.experiments import (
    DeepONetTuningSettings,
    ExperimentSpec,
    RayRuntimeConfig,
    RunContext,
    fingerprint_path,
    parse_operator_args,
    resolve_data_cache,
)
from chem_operator.models import (
    DeepONetAdapter,
    fit_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = example_paths(__file__, dataset="pfr")
PROBLEM_ID = "pfr_chain"
PROTOCOL_ID = "operator-cartesian-v1"
FIELDS = ("T", "P", "X", "velocity")
CONSTANTS = (
    "length",
    "area",
    "inlet_velocity",
    "pressure_controller_K",
    "phi",
    "ar_o2_ratio",
)
UNITS = {"T": "K", "P": "Pa", "X": "-", "velocity": "m/s"}

COORDINATE_STRIDE = 4
SEED = 42

# Defaults for --tune-epochs and --samples. Final training runs 2 x --tune-epochs
# unless --epochs is given (see chem_operator.experiments.final_epochs).
MAX_EPOCHS = 40
NUM_SAMPLES = 15
MAX_CONCURRENT_TRIALS = 1
TIME_BUDGET_S = 30 * 60
CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
PIN_MEMORY = bool(GPUS_PER_TRIAL)
RAY_OBJECT_STORE_BYTES = 100 * 1024**2


def parse_args(*, description: str | None = None) -> argparse.Namespace:
    args = parse_operator_args(
        PATHS,
        epochs=None,
        tune_epochs=MAX_EPOCHS,
        samples=NUM_SAMPLES,
        description=description or __doc__,
    )
    resolve_data_cache(args, partial(raw_dataset, args.data_dir))
    return args


def raw_dataset(
    data_dir: Path, split: str, *, preload: bool = False
) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        data_dir / f"{PROBLEM_ID}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="z",
        input_fields=FIELDS,
        output_fields=FIELDS,
        constant_inputs=CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        index_stride=COORDINATE_STRIDE,
        dtype=torch.float32,
        preload=preload,
    )


def limited(dataset: Dataset, maximum: int | None) -> Dataset:
    """Return the first *maximum* cases (``--max-cases``), or all of them."""
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
    )


def prepare_normalizer(args: argparse.Namespace) -> ZScoreNormalizer:
    """Fit normalization on the selected training cases."""
    raw = raw_dataset(args.data_dir, "train")
    try:
        normalizer = fit_zscore_normalizer(
            DataLoader(limited(raw, args.max_cases), batch_size=None, shuffle=False),
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
    args: argparse.Namespace,
    normalizer: ZScoreNormalizer,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter]]:
    del context
    cache_data = getattr(args, "cache_data", False)
    train_raw = raw_dataset(args.data_dir, "train", preload=cache_data)
    valid_raw = raw_dataset(args.data_dir, "valid", preload=cache_data)
    try:
        yield (
            adapter(limited(train_raw, args.max_cases), normalizer),
            adapter(limited(valid_raw, args.max_cases), normalizer),
        )
    finally:
        train_raw.close()
        valid_raw.close()


@contextmanager
def final_data(
    args: argparse.Namespace,
    *,
    normalizer: ZScoreNormalizer,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter, DeepONetAdapter]]:
    cache_data = getattr(args, "cache_data", False)
    raw = tuple(
        raw_dataset(args.data_dir, split, preload=cache_data)
        for split in ("train", "valid", "test")
    )
    try:
        yield tuple(  # type: ignore[misc]
            adapter(limited(item, args.max_cases), normalizer) for item in raw
        )
    finally:
        for item in raw:
            item.close()


def experiment_spec(model_id: str, args: argparse.Namespace) -> ExperimentSpec:
    return ExperimentSpec(
        problem_id=PROBLEM_ID,
        model_id=model_id,
        benchmark_protocol_id=PROTOCOL_ID,
        dataset_fingerprints={
            name: fingerprint_path(args.data_dir / f"{PROBLEM_ID}_{split}.h5")
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
        selected_test_case_ids=range(
            min(args.plot_cases, args.max_cases or args.plot_cases)
        ),
        tuning_budget={"samples": args.samples, "epochs": args.tune_epochs},
        project_root=PATHS.root,
    )


def tuning_settings(args: argparse.Namespace) -> DeepONetTuningSettings:
    return DeepONetTuningSettings(
        max_epochs=args.tune_epochs,
        num_samples=args.samples,
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
