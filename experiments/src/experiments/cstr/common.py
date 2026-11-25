"""Shared CSTR dataset and DeepONet experiment configuration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Iterator

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
from experiments.paths import example_paths
from chem_operator.experiments import (
    ExperimentSpec,
    RunContext,
    TuningConfig,
    build_tuning_config,
    fingerprint_path,
    parse_operator_args,
    resolve_data_cache,
)
from chem_operator.models import (
    DeepONetAdapter,
    fit_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = example_paths(__file__, dataset="cstr")

PROBLEM_ID = "cstr_non_isothermal"
PROTOCOL_ID = "operator-cartesian-logtime-v1"
FIELDS = ("T", "P", "X")
CONSTANTS = ("heat_transfer_coefficient",)
UNITS = {"T": "K", "P": "Pa", "X": "-"}

COORDINATE_STRIDE = 1
LOG_TIME_POINTS = 501
LOG_TIME_FINAL = 100.0
LOG_TIME_SCALE = 1.0e-2
SEED = 42

# Defaults for --tune-epochs and --samples. Final training runs 2 x --tune-epochs
# unless --epochs is given (see chem_operator.experiments.final_epochs).
MAX_EPOCHS = 40
NUM_SAMPLES = 15
MAX_CONCURRENT_TRIALS = 1
TIME_BUDGET_S = 30 * 60
CPUS_PER_TRIAL = 2
PIN_MEMORY = torch.cuda.is_available()


def parse_args(*, description: str | None = None) -> argparse.Namespace:
    args = parse_operator_args(
        PATHS,
        tune_epochs=MAX_EPOCHS,
        samples=NUM_SAMPLES,
        description=description or __doc__,
    )
    resolve_data_cache(args, partial(raw_dataset, args.data_dir))
    return args


def log_time_grid() -> np.ndarray:
    normalized = np.linspace(0.0, 1.0, LOG_TIME_POINTS)
    grid = LOG_TIME_SCALE * np.expm1(
        normalized * np.log1p(LOG_TIME_FINAL / LOG_TIME_SCALE)
    )
    grid[0] = 0.0
    grid[-1] = LOG_TIME_FINAL
    return grid


def raw_dataset(
    data_dir: Path, split: str, *, cache: bool = False
) -> TrajectoryResamplingDataset:
    """Resample one split; ``cache`` keeps resampled trajectories in memory.

    The raw arrays are never preloaded: after the first epoch the resampling
    cache serves every sample, so a raw copy would only use memory.
    """
    raw = ChemOperatorDataset(
        data_dir / f"{PROBLEM_ID}_{split}.h5",
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
        cache=cache,
    )


def limited(dataset: Dataset, maximum: int | None) -> Dataset:
    """Return the first *maximum* trajectories (``--max-cases``), or all of them."""
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
        coordinate_name="t",
        include_constants=True,
        coordinate_mode="log1p",
        coordinate_scale=LOG_TIME_SCALE,
    )


def resampled_dataset_fingerprint(data_dir: Path, split: str) -> str:
    source = fingerprint_path(data_dir / f"{PROBLEM_ID}_{split}.h5")
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


def prepare_normalizer(args: argparse.Namespace) -> ZScoreNormalizer:
    """Fit normalization once on the driver, on the selected training cases."""
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
    train_raw = raw_dataset(args.data_dir, "train", cache=cache_data)
    valid_raw = raw_dataset(args.data_dir, "valid", cache=cache_data)
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
        raw_dataset(args.data_dir, split, cache=cache_data)
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
            name: resampled_dataset_fingerprint(args.data_dir, split)
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
        selected_test_case_ids=range(
            min(args.plot_cases, args.max_cases or args.plot_cases)
        ),
        tuning_budget={"samples": args.samples, "epochs": args.tune_epochs},
        project_root=PATHS.root,
    )


def tuning_settings(
    args: argparse.Namespace, context: RunContext, *, metric: str = "best_valid_loss"
) -> TuningConfig:
    """Tuning policy of the hand-built scripts here; GPUs per trial follow ``--device``."""
    return build_tuning_config(
        metric=metric,
        samples=args.samples,
        epochs=args.tune_epochs,
        seed=context.seed,
        device=context.device,
        cpus_per_trial=CPUS_PER_TRIAL,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        time_budget_s=TIME_BUDGET_S,
        temp_dir=PATHS.ray,
    )
