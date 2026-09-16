"""Tests for the configurable, resolution-independent FNO adapter."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from chem_operator.models import (
    FNOAdapter,
    FNOChannel,
    ModelDataAdapter,
    fit_fno_zscore_normalizer,
)


class _SampleDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def _sample(n_z: int, n_r: int, offset: float) -> dict:
    z = torch.linspace(0.0, 1.0, n_z)
    r = torch.linspace(0.0, 0.5, n_r)
    velocity = (
        torch.arange(n_z * n_r, dtype=torch.float32).reshape(n_z, n_r)
        + offset
    )
    fractions = torch.stack(
        (
            0.2 + 0.01 * velocity,
            0.3 + 0.02 * velocity,
            0.5 - 0.03 * velocity,
        ),
        dim=-1,
    )
    return {
        "input_fields": {
            "velocity": velocity[:1],
            "X": fractions[:1],
        },
        "output_fields": {
            "velocity": velocity[1:],
            "X": fractions[1:],
        },
        "constant_inputs": {"SCCM": torch.tensor(500.0 + offset)},
        "input_coordinates": {"z": z[:1], "r": r},
        "output_coordinates": {"z": z[1:], "r": r},
        "metadata": {
            "params": {"T0": 1170.0 + offset},
            "field_species": {"X": ["CH4", "H2", "AR"]},
        },
    }


INPUT_CHANNELS = (
    FNOChannel("T0", "parameter", "T0"),
    FNOChannel("SCCM", "constant", "SCCM"),
)
OUTPUT_CHANNELS = (
    FNOChannel("velocity", "field", "velocity"),
    FNOChannel("X_CH4", "species", "X", species="CH4"),
    FNOChannel("X_H2", "species", "X", species="H2"),
)


def test_configurable_channels_and_global_normalization() -> None:
    """Parameters, constants, fields, and species share one FNO interface."""
    raw = _SampleDataset([_sample(3, 2, 0.0), _sample(3, 2, 2.0)])
    normalizer = fit_fno_zscore_normalizer(
        raw,
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
    )
    adapter = FNOAdapter(
        raw,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("z", "r"),
    )
    assert isinstance(adapter, ModelDataAdapter)

    item = adapter[0]
    assert item["x"].shape == (2, 3, 2)
    assert item["y"].shape == (3, 3, 2)
    assert item["z"].shape == (3,)
    assert item["r"].shape == (2,)
    assert all(stat.ndim == 0 for stat in normalizer.means.values())
    for channel in item["x"]:
        torch.testing.assert_close(channel, channel[0, 0].expand_as(channel))

    physical = adapter.physical_item(0)
    reconstructed = adapter.denormalize_output(item["y"])
    torch.testing.assert_close(reconstructed, physical["y"])
    torch.testing.assert_close(
        physical["y"][1],
        torch.cat(
            (
                raw[0]["input_fields"]["X"],
                raw[0]["output_fields"]["X"],
            )
        )[..., 0],
    )

    reference = adapter.reference_item(0)
    assert reference.labels == ("velocity", "X_CH4", "X_H2")
    assert reference.coordinates.shape == (6, 2)
    assert reference.values.shape == (6, 3)
    torch.testing.assert_close(
        reference.values,
        physical["y"].movedim(0, -1).reshape(6, 3),
    )
    assert adapter.checkpoint_config()["coordinate_names"] == ["z", "r"]


def test_normalizer_broadcasts_on_a_different_resolution() -> None:
    """Training-grid scalar statistics work on a finer axial-radial mesh."""
    training = _SampleDataset([_sample(3, 2, 0.0), _sample(3, 2, 2.0)])
    normalizer = fit_fno_zscore_normalizer(
        training,
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
    )
    fine = _SampleDataset([_sample(5, 4, 1.0)])
    adapter = FNOAdapter(
        fine,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("z", "r"),
    )
    item = adapter[0]
    assert item["x"].shape == (2, 5, 4)
    assert item["y"].shape == (3, 5, 4)
    assert torch.isfinite(item["x"]).all()
    assert torch.isfinite(item["y"]).all()


def test_physical_field_inputs_are_configuration_only() -> None:
    """A known physical field can be added without changing adapter logic."""
    inputs = INPUT_CHANNELS + (
        FNOChannel("known_velocity", "field", "velocity"),
    )
    raw = _SampleDataset([_sample(3, 2, 0.0), _sample(3, 2, 2.0)])
    normalizer = fit_fno_zscore_normalizer(raw, inputs, OUTPUT_CHANNELS)
    adapter = FNOAdapter(
        raw,
        normalizer,
        input_channels=inputs,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("z", "r"),
    )
    assert adapter[0]["x"].shape == (3, 3, 2)
    assert FNOAdapter.required_field_names(inputs, OUTPUT_CHANNELS) == (
        "velocity",
        "X",
    )
    assert FNOAdapter.required_constant_names(inputs, OUTPUT_CHANNELS) == (
        "SCCM",
    )
