"""Train direct and IPCA-POD DeepONets for 1D packed-bed profiles."""

from __future__ import annotations

import os
from pathlib import Path

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


PATHS = ExamplePaths.from_script(__file__, dataset="packed_bed_1D")
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
EPOCHS = 200
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
WIDTH = 128
LATENT_WIDTH = 8
DISPLAY_EVERY = 100
COORDINATE_STRIDE = 1
RESAMPLE_POINTS = 128
SEED = 7
POD_VARIANCE_THRESHOLD = 0.999
RECONSTRUCTION_CASES = 2
MAX_TRAJECTORIES: int | None = None


def raw_dataset(data_dir: Path, split: str, stride: int) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        data_dir / f"packed_bed_1d_{split}.h5",
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


def main() -> None:
    train_raw = raw_dataset(PATHS.data, "train", COORDINATE_STRIDE)
    valid_raw = raw_dataset(PATHS.data, "valid", COORDINATE_STRIDE)
    test_raw = raw_dataset(PATHS.data, "test", COORDINATE_STRIDE)
    try:
        train_data = limited(train_raw, MAX_TRAJECTORIES)
        valid_data = limited(valid_raw, MAX_TRAJECTORIES)
        test_data = limited(test_raw, MAX_TRAJECTORIES)
        print("Fitting packed-bed Z-score statistics from training trajectories ...")
        normalizer = fit_zscore_normalizer(train_data, FIELDS, CONSTANTS)

        def adapter(dataset: Dataset) -> DeepXDEAdapter:
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
            return DeepXDEAdapter(
                dataset,
                processor,
                format="cartesian_product",
                coordinate_name="z",
                include_constants=True,
                resample_points=RESAMPLE_POINTS,
                coordinate_mode="relative",
            )

        config = DeepONetTrainingConfig(
            epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            width=WIDTH,
            latent_width=LATENT_WIDTH,
            display_every=DISPLAY_EVERY,
            seed=SEED,
        )
        result = run_deeponet_comparison(
            adapter(train_data),
            adapter(valid_data),
            adapter(test_data),
            normalizer,
            direct_config=config,
            pod_variance_threshold=POD_VARIANCE_THRESHOLD,
            reconstruction_cases=RECONSTRUCTION_CASES,
        )
        save_deeponet_comparison(
            result,
            PATHS.output,
            problem="packed_bed_1d",
        )
        print(f"Artifacts written to {PATHS.output}")
    finally:
        train_raw.close()
        valid_raw.close()
        test_raw.close()


if __name__ == "__main__":
    main()
