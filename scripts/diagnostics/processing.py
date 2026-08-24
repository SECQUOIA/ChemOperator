"""Smoke-test and plot the processed tensors for four reactor datasets.

Run from any directory with::

    uv run python scripts/diagnostics/processing.py

The script checks one DataLoader batch and delta round-trip reconstruction for
each dataset, then writes ``scripts/diagnostics/processing.png``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from chem_operator.datasets import (
    ChemOperatorDataset,
    DataProcessor,
    FieldPacker,
    IdentityNormalizer,
    NormalizationConfig,
    ProcessedDataset,
    TargetTransformConfig,
)
from chem_operator.example_paths import ExamplePaths


PATHS = ExamplePaths.from_script(__file__)
OUTPUT_PATH = PATHS.example / "processing.png"
MAX_POINTS = 250


@dataclass(frozen=True)
class DatasetSpec:
    title: str
    path: Path
    coordinate: str
    fields: tuple[str, ...]
    constants: tuple[str, ...]


DATASETS = (
    DatasetSpec(
        title="CSTR",
        path=PATHS.datasets / "cstr/cstr_test.h5",
        coordinate="t",
        fields=("T", "P", "X"),
        constants=(
            "reactor_volume",
            "residence_time",
            "pressure_controller_K",
            "phi",
            "reactive_fraction",
        ),
    ),
    DatasetSpec(
        title="PFR: chain of reactors",
        path=PATHS.datasets / "pfr/pfr_chain_of_reactors_test.h5",
        coordinate="z",
        fields=("T", "P", "X", "velocity"),
        constants=(
            "length",
            "area",
            "inlet_velocity",
            "pressure_controller_K",
            "phi",
            "ar_o2_ratio",
        ),
    ),
    DatasetSpec(
        title="PFR: Lagrangian particle",
        path=PATHS.datasets / "pfr/pfr_lagrangian_particle_test.h5",
        coordinate="t",
        fields=("T", "P", "X", "velocity", "z"),
        constants=(
            "length",
            "area",
            "inlet_velocity",
            "phi",
            "ar_o2_ratio",
        ),
    ),
    DatasetSpec(
        title="1D packed bed",
        path=PATHS.datasets / "packed_bed_1D/packed_bed_1d_test.h5",
        coordinate="z",
        fields=("T", "P", "X", "Z", "rhou", "velocity"),
        constants=(
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
        ),
    ),
)


def make_dataset(
    spec: DatasetSpec,
) -> tuple[ChemOperatorDataset, DataProcessor, ProcessedDataset]:
    raw_dataset = ChemOperatorDataset(
        spec.path,
        task="next_step",
        coordinate_name=spec.coordinate,
        input_fields=spec.fields,
        output_fields=spec.fields,
        constant_inputs=spec.constants,
        n_steps_input=4,
        n_steps_output=1,
        index_stride=1,
        dtype=torch.float32,
    )
    processor = DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=spec.fields,
            constant_field_order=spec.constants,
        ),
        normalizer=IdentityNormalizer(),
        normalization_config=NormalizationConfig(enabled=False),
        target_transform=TargetTransformConfig(
            mode="delta",
            multi_step_delta="direct",
        ),
    )
    return raw_dataset, processor, ProcessedDataset(raw_dataset, processor)


def evenly_spaced_case_indices(
    dataset: ChemOperatorDataset, max_points: int
) -> list[int]:
    case_name = dataset.sample_index[0][0]
    case_indices = [
        index
        for index, (sample_case, _, _) in enumerate(dataset.sample_index)
        if sample_case == case_name
    ]
    positions = np.linspace(
        0,
        len(case_indices) - 1,
        min(max_points, len(case_indices)),
        dtype=int,
    )
    return [case_indices[position] for position in np.unique(positions)]


def validate_and_collect(
    spec: DatasetSpec,
    max_points: int,
) -> tuple[np.ndarray, torch.Tensor, tuple[str, ...], tuple[int, ...]]:
    raw_dataset, processor, dataset = make_dataset(spec)
    try:
        batch = next(iter(DataLoader(dataset, batch_size=4, shuffle=False)))
        batch_shapes = (
            tuple(batch["x"].shape),
            tuple(batch["y"].shape),
            tuple(batch["constants"].shape),
        )
        print(
            f"{spec.title}: x={batch_shapes[0]}, y={batch_shapes[1]}, "
            f"constants={batch_shapes[2]}"
        )

        coordinates: list[float] = []
        states: list[torch.Tensor] = []
        labels: tuple[str, ...] | None = None
        for index in evenly_spaced_case_indices(raw_dataset, max_points):
            raw = raw_dataset[index]
            sample = processor(raw)
            reconstructed = processor.inverse_reconstruct(sample["y"], sample["x"])
            expected = processor.field_packer.pack_variable(raw["output_fields"])
            torch.testing.assert_close(reconstructed, expected)

            coordinate = raw["output_coordinates"][spec.coordinate].reshape(-1)[0]
            coordinates.append(float(coordinate))
            states.append(reconstructed[0].detach().cpu())
            labels = tuple(sample["labels"]["y"])

        if labels is None:
            raise RuntimeError(f"No samples were collected from {spec.path}.")
        return (
            np.asarray(coordinates),
            torch.stack(states),
            labels,
            batch_shapes[0],
        )
    finally:
        dataset.close()


def plot_dataset(
    axis: plt.Axes,
    spec: DatasetSpec,
    coordinates: np.ndarray,
    states: torch.Tensor,
    labels: tuple[str, ...],
) -> None:
    temperature_index = labels.index("T")
    axis.plot(
        coordinates,
        states[:, temperature_index].numpy(),
        color="tab:red",
        linewidth=2,
        label="T",
    )
    axis.set_title(spec.title)
    axis.set_xlabel(spec.coordinate)
    axis.set_ylabel("Temperature [K]", color="tab:red")
    axis.tick_params(axis="y", labelcolor="tab:red")
    axis.grid(alpha=0.25)

    composition_indices = [
        index for index, label in enumerate(labels) if label.startswith("X[")
    ]
    if composition_indices:
        composition = states[:, composition_indices]
        variation = torch.max(composition, dim=0).values - torch.min(
            composition, dim=0
        ).values
        local_index = int(torch.argmax(variation))
        channel_index = composition_indices[local_index]
        second_axis = axis.twinx()
        second_axis.plot(
            coordinates,
            states[:, channel_index].numpy(),
            color="tab:blue",
            linewidth=1.5,
            label=labels[channel_index],
        )
        second_axis.set_ylabel(labels[channel_index], color="tab:blue")
        second_axis.tick_params(axis="y", labelcolor="tab:blue")


def main() -> None:
    if MAX_POINTS < 2:
        raise ValueError("MAX_POINTS must be at least 2.")

    figure, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    for axis, spec in zip(axes.flat, DATASETS):
        coordinates, states, labels, _ = validate_and_collect(
            spec, MAX_POINTS
        )
        plot_dataset(axis, spec, coordinates, states, labels)

    figure.suptitle("Processed reactor datasets (delta targets reconstructed)")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, dpi=180)
    plt.close(figure)
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
