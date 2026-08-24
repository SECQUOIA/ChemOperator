"""Train direct and IPCA-POD DeepONets for complete CSTR trajectories."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from torch.utils.data import Dataset, Subset

from chem_operator.datasets import (
    ChemOperatorDataset,
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    TargetTransformConfig,
)
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import (
    DeepONetBenchmarkConfig,
    DeepXDEAdapter,
    fit_zscore_normalizer,
    run_deepxde_benchmark,
)


PATHS = ExamplePaths.from_script(__file__, dataset="cstr")
FIELDS = ("T", "P", "X")
CONSTANTS: tuple[str, ...] = ()
EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
WIDTH = 512
LATENT_WIDTH = 64
DISPLAY_EVERY = 10
COORDINATE_STRIDE = 5
SEED = 7
PLOT_CASES = 2
MAX_TRAJECTORIES: int | None = None


def raw_dataset(data_dir: Path, split: str, stride: int) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        data_dir / f"cstr_non_isothermal_{split}.h5",
        task="operator_cartesian",
        coordinate_name="t",
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


def main() -> None:
    train_raw = raw_dataset(PATHS.data, "train", COORDINATE_STRIDE)
    valid_raw = raw_dataset(PATHS.data, "valid", COORDINATE_STRIDE)
    test_raw = raw_dataset(PATHS.data, "test", COORDINATE_STRIDE)
    try:
        train_data = limited(train_raw, MAX_TRAJECTORIES)
        valid_data = limited(valid_raw, MAX_TRAJECTORIES)
        test_data = limited(test_raw, MAX_TRAJECTORIES)
        print("Fitting CSTR Z-score statistics from training trajectories ...")
        normalizer = fit_zscore_normalizer(train_data, FIELDS, CONSTANTS)

        def processor() -> DataProcessor:
            return DataProcessor(
                field_packer=FieldPacker(
                    channel_axis="last",
                    variable_field_order=FIELDS,
                    constant_field_order=CONSTANTS,
                ),
                normalizer=normalizer,
                normalization_config=NormalizationConfig(enabled=True),
                target_transform=TargetTransformConfig(mode="state"),
            )

        train = DeepXDEAdapter(
            train_data,
            processor(),
            format="cartesian_product",
            coordinate_name="t",
            include_constants=False,
        )
        validation = DeepXDEAdapter(
            valid_data,
            processor(),
            format="cartesian_product",
            coordinate_name="t",
            include_constants=False,
        )
        test = DeepXDEAdapter(
            test_data,
            processor(),
            format="cartesian_product",
            coordinate_name="t",
            include_constants=False,
        )
        run_deepxde_benchmark(
            train,
            validation,
            test,
            normalizer,
            output_dir=PATHS.output,
            plot_labels=("T", "P", "X[0]"),
            coordinate_label="Time [s]",
            config=DeepONetBenchmarkConfig(
                epochs=EPOCHS,
                learning_rate=LEARNING_RATE,
                batch_size=BATCH_SIZE,
                width=WIDTH,
                latent_width=LATENT_WIDTH,
                display_every=DISPLAY_EVERY,
                variance_threshold=0.9995,
                seed=SEED,
                plot_cases=PLOT_CASES,
            ),
        )
        print(f"Results written to {PATHS.output}")
    finally:
        train_raw.close()
        valid_raw.close()
        test_raw.close()


if __name__ == "__main__":
    main()
