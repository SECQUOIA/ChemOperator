
import math

import matplotlib.pyplot as plt
import numpy as np
import torch

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths

PATHS = ExamplePaths.from_script(__file__, dataset="q2d_cmr")

# Match q2d_fno.py: field_map returns one complete steady Q2D field per case.
DATASET_PATH = PATHS.data / "q2d_cmr_test.h5"
dataset = ChemOperatorDataset(
    DATASET_PATH,
    task="field_map",
    coordinate_name="z",
    n_steps_input=1,
    dtype=torch.float32,
    constant_inputs=(),
)

try:
    sample = dataset[0]
finally:
    dataset.close()

# The dataset separates the inlet plane from the rest of the axial solution;
# concatenate them to recover each full (z, r) steady field.
fields = {
    name: torch.cat(
        (sample["input_fields"][name], sample["output_fields"][name]), dim=0
    ).cpu().numpy()
    for name in sample["output_fields"]
}
coordinates = {
    "z": torch.cat(
        (sample["input_coordinates"]["z"], sample["output_coordinates"]["z"])
    ).cpu().numpy(),
    "r": sample["output_coordinates"]["r"].cpu().numpy(),
}

print(f"Dataset: {DATASET_PATH.name}")
print(f"Case: {sample['metadata']['case_name']}")
print(f"Grid: {len(coordinates['z'])} axial x {len(coordinates['r'])} radial points")
print(f"Stored field groups: {', '.join(fields)}")
print(f"{len(fields) = }")
print(f"{fields["theta"] = }")
print(f"{sample["metadata"]["field_species"]["sdot"] = }")

def expanded_fields(field_groups, metadata):
    """Yield every scalar map, expanding grouped species dimensions."""
    species_by_group = metadata.get("field_species", {})
    for field_name, values in field_groups.items():
        if values.ndim == 2:
            yield field_name, values
            continue

        species_names = species_by_group.get(field_name, [])
        if values.ndim != 3 or values.shape[-1] != len(species_names):
            raise ValueError(
                f"Cannot label field {field_name!r} with shape {values.shape}."
            )
        for species_index, species_name in enumerate(species_names):
            yield f"{field_name}_{species_name}", values[..., species_index]


scalar_fields = list(expanded_fields(fields, sample["metadata"]))
z_mm = 1e3 * np.asarray(coordinates["z"], dtype=float)
r_mm = 1e3 * np.asarray(coordinates["r"], dtype=float)

z_pad = 0.5 * np.ptp(z_mm) / max(len(z_mm) - 1, 1) or 0.5
r_pad = 0.5 * np.ptp(r_mm) / max(len(r_mm) - 1, 1) or 0.5
extent = [z_mm.min() - z_pad, z_mm.max() + z_pad,
          r_mm.min() - r_pad, r_mm.max() + r_pad]

ncols = 3
nrows = math.ceil(len(scalar_fields) / ncols)
figure, axes = plt.subplots(
    nrows, ncols, figsize=(5 * ncols, 3.5 * nrows),
    squeeze=False, constrained_layout=True,
)

for axis, (field_name, values) in zip(axes.flat, scalar_fields):
    image = axis.imshow(
        values.T,
        origin="lower",
        aspect="auto",
        extent=extent,
        interpolation="nearest",
    )
    print(field_name)
    axis.set_title(field_name)
    axis.set_xlabel("Axial position z [mm]")
    axis.set_ylabel("Radius r [mm]")
    figure.colorbar(image, ax=axis)

for axis in axes.flat[len(scalar_fields):]:
    axis.set_visible(False)

figure.suptitle(
    f"All Q2D fields: {sample['metadata']['case_name']} "
    f"({len(scalar_fields)} scalar maps)",
    fontsize=15,
)
figure.savefig(PATHS.example / "all_fields.png")

N_SAMPLES = 5
FIELD_SPECS = (
    ("X", "CH4", r"$X_{CH_4}$"),
    ("X", "H2", r"$X_{H_2}$"),
    ("theta", "C(s)", r"$\theta_{C(s)}$"),
    ("sdot", "C(s)", r"$\dot{s}_{C(s)}$ [kmol m$^{-2}$ s$^{-1}$]"),
)

comparison_dataset = ChemOperatorDataset(
    DATASET_PATH,
    task="field_map",
    coordinate_name="z",
    input_fields=("X", "theta", "sdot"),
    output_fields=("X", "theta", "sdot"),
    constant_inputs=(),
    n_steps_input=1,
    dtype=torch.float32,
)

sample_indices = np.linspace(
    0, len(comparison_dataset) - 1,
    num=min(N_SAMPLES, len(comparison_dataset)),
    dtype=int,
)
comparison_samples = []
try:
    for sample_index in sample_indices:
        current = comparison_dataset[int(sample_index)]
        full_groups = {
            name: torch.cat(
                (current["input_fields"][name], current["output_fields"][name]),
                dim=0,
            ).cpu().numpy()
            for name in ("X", "theta", "sdot")
        }
        selected = []
        for group, species, _ in FIELD_SPECS:
            species_names = current["metadata"]["field_species"][group]
            selected.append(full_groups[group][..., species_names.index(species)])
        comparison_samples.append((current, selected))
finally:
    comparison_dataset.close()

# Fix each column's scale across all samples for direct visual comparison.
column_limits = []
for column in range(len(FIELD_SPECS)):
    values = np.concatenate(
        [selected[column].ravel() for _, selected in comparison_samples]
    )
    column_limits.append((float(np.nanmin(values)), float(np.nanmax(values))))

figure, axes = plt.subplots(
    len(comparison_samples), len(FIELD_SPECS),
    figsize=(17, 3.2 * len(comparison_samples)),
    squeeze=False,
    constrained_layout=True,
)
column_images = []
for row, (current, selected) in enumerate(comparison_samples):
    z = 1e3 * torch.cat(
        (current["input_coordinates"]["z"], current["output_coordinates"]["z"])
    ).cpu().numpy()
    r = 1e3 * current["output_coordinates"]["r"].cpu().numpy()
    z_pad = 0.5 * np.ptp(z) / max(len(z) - 1, 1) or 0.5
    r_pad = 0.5 * np.ptp(r) / max(len(r) - 1, 1) or 0.5
    current_extent = [
        z.min() - z_pad, z.max() + z_pad,
        r.min() - r_pad, r.max() + r_pad,
    ]
    params = current["metadata"]["params"]

    for column, (values, (_, _, title)) in enumerate(zip(selected, FIELD_SPECS)):
        low, high = column_limits[column]
        image = axes[row, column].imshow(
            values.T,
            origin="lower",
            aspect="auto",
            extent=current_extent,
            interpolation="nearest",
            vmin=low,
            vmax=high,
        )
        if row == 0:
            axes[row, column].set_title(title)
        if row == len(comparison_samples) - 1:
            axes[row, column].set_xlabel("Axial position z [mm]")
        if column == 0:
            axes[row, column].set_ylabel(
                f"T0={params['T0']:.1f} K\n"
                f"sccm={params['sccm']:.1f}\nRadius r [mm]"
            )
        if row == 0:
            column_images.append(image)

for column, image in enumerate(column_images):
    figure.colorbar(image, ax=axes[:, column], shrink=0.9)

figure.suptitle("Selected Q2D chemistry fields across test samples", fontsize=15)
figure.savefig(PATHS.example / "chem.png")
