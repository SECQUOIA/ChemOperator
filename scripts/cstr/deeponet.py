"""Train direct and IPCA-POD DeepONets for non-isothermal CSTR trajectories."""

from __future__ import annotations

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")
# DeepXDE imports Matplotlib internally even though this runner does not plot.
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
    DeepONetTrainingConfig,
    DeepXDEAdapter,
    fit_zscore_normalizer,
    run_deeponet_comparison,
    save_deeponet_comparison,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = ExamplePaths.from_script(__file__, dataset="cstr")
DATASET_NAME = "cstr_non_isothermal"
FIELDS = ("T", "P", "X")
CONSTANTS = ("heat_transfer_coefficient",)

EPOCHS = 30
LEARNING_RATE = 1e-3
BATCH_SIZE = 16
WIDTH = 512
LATENT_WIDTH = 64
DISPLAY_EVERY = 10
COORDINATE_STRIDE = 5
POD_VARIANCE_THRESHOLD = 0.9995
SEED = 42
RECONSTRUCTION_CASES = 2
MAX_TRAJECTORIES: int | None = None
DATALOADER_WORKERS = 0
PIN_MEMORY = torch.cuda.is_available()


def raw_dataset(split: str) -> ChemOperatorDataset:
    return ChemOperatorDataset(
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


def limited(dataset: Dataset) -> Dataset:
    if MAX_TRAJECTORIES is None:
        return dataset
    return Subset(dataset, range(min(MAX_TRAJECTORIES, len(dataset))))


def processor(normalizer: ZScoreNormalizer) -> DataProcessor:
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


def adapter(dataset: Dataset, normalizer: ZScoreNormalizer) -> DeepXDEAdapter:
    return DeepXDEAdapter(
        dataset,
        processor(normalizer),
        format="cartesian_product",
        coordinate_name="t",
        include_constants=True,
    )


def training_config() -> DeepONetTrainingConfig:
    return DeepONetTrainingConfig(
        loss="relative_l2",
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        width=WIDTH,
        latent_width=LATENT_WIDTH,
        display_every=DISPLAY_EVERY,
        seed=SEED,
    )


def main() -> None:
    train_raw = raw_dataset("train")
    valid_raw = raw_dataset("valid")
    test_raw = raw_dataset("test")
    try:
        train_data = limited(train_raw)
        valid_data = limited(valid_raw)
        test_data = limited(test_raw)
        print("Fitting CSTR Z-score statistics from training trajectories ...")
        normalizer = fit_zscore_normalizer(train_data, FIELDS, CONSTANTS)
        result = run_deeponet_comparison(
            adapter(train_data, normalizer),
            adapter(valid_data, normalizer),
            adapter(test_data, normalizer),
            normalizer,
            direct_config=training_config(),
            pod_variance_threshold=POD_VARIANCE_THRESHOLD,
            reconstruction_cases=RECONSTRUCTION_CASES,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        save_deeponet_comparison(
            result,
            PATHS.output,
            problem=DATASET_NAME,
        )
        print(f"Artifacts written to {PATHS.output}")
    finally:
        train_raw.close()
        valid_raw.close()
        test_raw.close()


if __name__ == "__main__":
    main()
