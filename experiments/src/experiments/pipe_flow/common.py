"""Shared pipe-flow dataset and DeepONet experiment configuration."""

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
    ExperimentSpec,
    RunContext,
    TuningConfig,
    build_tuning_config,
    fingerprint_path,
    parse_operator_args,
    resolve_data_cache,
    tuning_resources,
)
from chem_operator.models import (
    DeepONetAdapter,
    fit_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = example_paths(__file__, dataset="pipe_flow")
PROBLEM_ID = "pipe_flow"
PROTOCOL_ID = "operator-cartesian-v1"
FIELDS = ("velocity",)
CONSTANTS = ("radius", "length", "dynamic_viscosity", "pressure_drop")
UNITS = {"velocity": "m/s"}

SEED = 42

# Defaults for --tune-epochs and --samples. Final training runs 2 x --tune-epochs
# unless --epochs is given (see chem_operator.experiments.final_epochs).
MAX_EPOCHS = 20
NUM_SAMPLES = 15
PIN_MEMORY = torch.cuda.is_available()


def parse_args() -> argparse.Namespace:
    args = parse_operator_args(
        PATHS,
        tune_epochs=MAX_EPOCHS,
        samples=NUM_SAMPLES,
        description=__doc__,
    )
    resolve_data_cache(args, partial(raw_dataset, args.data_dir))
    return args


def raw_dataset(
    data_dir: Path, split: str, *, preload: bool = False
) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        data_dir / f"{PROBLEM_ID}_{split}.h5",
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
        coordinate_name="r",
        include_constants=True,
        include_input_state=False,
        coordinate_mode="relative",
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
        coordinates=("r",),
        selected_test_case_ids=range(
            min(args.plot_cases, args.max_cases or args.plot_cases)
        ),
        tuning_budget={"samples": args.samples, "epochs": args.tune_epochs},
        project_root=PATHS.root,
    )


def tuning_settings(
    args: argparse.Namespace, context: RunContext, *, metric: str = "best_valid_loss"
) -> TuningConfig:
    """Tuning policy of the hand-built scripts here; resources come from the CLI flags."""
    return build_tuning_config(
        metric=metric,
        samples=args.samples,
        epochs=args.tune_epochs,
        seed=context.seed,
        device=context.device,
        **tuning_resources(args),
        temp_dir=PATHS.ray,
    )
