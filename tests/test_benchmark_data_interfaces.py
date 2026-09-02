"""Focused tests for benchmark serialization and dataset discovery interfaces."""

from __future__ import annotations

import json
import pickle
from io import BytesIO

import h5py
import numpy as np
import pytest
import torch

from chem_operator.datasets import ChemOperatorDataset
from chem_operator.normalization import (IdentityNormalizer, MinMaxNormalizer,
                                         RMSNormalizer, ZScoreNormalizer,
                                         normalizer_from_state_dict)


def _normalizers():
    variable_fields = ("T", "X")
    constant_fields = ("radius",)
    zscore_stats = {
        "mean": {
            "T": torch.tensor(900.0, dtype=torch.float64),
            "X": torch.tensor([0.25, 0.75], dtype=torch.float64),
            "radius": torch.tensor(0.001, dtype=torch.float64),
        },
        "std": {
            "T": torch.tensor(100.0, dtype=torch.float64),
            "X": torch.tensor([0.1, 0.2], dtype=torch.float64),
            "radius": torch.tensor(0.0002, dtype=torch.float64),
        },
        "mean_delta": {
            "T": torch.tensor(2.0, dtype=torch.float64),
            "X": torch.tensor([0.01, -0.01], dtype=torch.float64),
        },
        "std_delta": {
            "T": torch.tensor(5.0, dtype=torch.float64),
            "X": torch.tensor([0.02, 0.03], dtype=torch.float64),
        },
    }
    rms_stats = {
        "rms": zscore_stats["std"],
        "rms_delta": zscore_stats["std_delta"],
    }
    minmax_stats = {
        "min": {
            "T": torch.tensor(700.0, dtype=torch.float64),
            "X": torch.tensor([0.0, 0.0], dtype=torch.float64),
            "radius": torch.tensor(0.0005, dtype=torch.float64),
        },
        "max": {
            "T": torch.tensor(1100.0, dtype=torch.float64),
            "X": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "radius": torch.tensor(0.0015, dtype=torch.float64),
        },
        "min_delta": {
            "T": torch.tensor(-10.0, dtype=torch.float64),
            "X": torch.tensor([-0.1, -0.1], dtype=torch.float64),
        },
        "max_delta": {
            "T": torch.tensor(10.0, dtype=torch.float64),
            "X": torch.tensor([0.1, 0.1], dtype=torch.float64),
        },
    }
    return (
        IdentityNormalizer(),
        ZScoreNormalizer(
            zscore_stats,
            variable_fields,
            constant_fields,
            min_denom=1.0e-12,
        ),
        RMSNormalizer(
            rms_stats,
            variable_fields,
            constant_fields,
            min_denom=1.0e-12,
        ),
        MinMaxNormalizer(
            minmax_stats,
            variable_fields,
            constant_fields,
            feature_range=(-1.0, 1.0),
            min_denom=1.0e-12,
        ),
    )


@pytest.mark.parametrize("normalizer", _normalizers())
def test_normalizer_state_is_json_and_torch_safe(normalizer) -> None:
    state = normalizer.state_dict()
    json_state = json.loads(json.dumps(state))
    json_restored = normalizer_from_state_dict(json_state)

    buffer = BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    torch_state = torch.load(buffer, weights_only=True)
    torch_restored = normalizer_from_state_dict(torch_state)

    assert json_restored.state_dict() == state
    assert torch_restored.state_dict() == state

    variable = torch.tensor(
        [[800.0, 0.2, 0.8], [1000.0, 0.3, 0.7]],
        dtype=torch.float64,
    )
    delta = torch.tensor(
        [[1.0, 0.01, -0.01], [3.0, -0.02, 0.02]],
        dtype=torch.float64,
    )
    for restored in (json_restored, torch_restored):
        normalized = restored.normalize_flattened(variable, "variable")
        torch.testing.assert_close(
            restored.denormalize_flattened(normalized, "variable"),
            variable,
        )
        normalized_delta = restored.delta_normalize_flattened(
            delta,
            "variable",
        )
        torch.testing.assert_close(
            restored.delta_denormalize_flattened(
                normalized_delta,
                "variable",
            ),
            delta,
        )


def test_normalizer_loader_rejects_unknown_schema() -> None:
    with pytest.raises(ValueError, match="schema"):
        normalizer_from_state_dict(
            {"schema": "other", "version": 1, "type": "identity"}
        )


def test_normalizer_loader_rejects_unknown_version_and_type() -> None:
    with pytest.raises(ValueError, match="version"):
        normalizer_from_state_dict(
            {
                "schema": "chem-operator-normalizer",
                "version": 2,
                "type": "identity",
            }
        )
    with pytest.raises(ValueError, match="type"):
        normalizer_from_state_dict(
            {
                "schema": "chem-operator-normalizer",
                "version": 1,
                "type": "unknown",
            }
        )


def _write_discovery_dataset(path) -> None:
    with h5py.File(path, "w") as h5:
        h5.attrs["schema"] = "simulation-record-split-v0"
        h5.attrs["simulator"] = "discovery_fixture"
        h5.attrs["split"] = "train"
        h5.attrs["n_cases"] = 2
        h5.create_dataset("field_names", data=np.asarray(["velocity", "X"], dtype="S"))
        cases = h5.create_group("cases")
        for case_index in range(2):
            case = cases.create_group(f"{case_index:06d}")
            coordinates = case.create_group("coordinates")
            coordinates.create_dataset("z", data=np.linspace(0.0, 1.0, 3))
            coordinates.create_dataset("r", data=np.linspace(0.0, 0.5, 2))

            fields = case.create_group("fields")
            fields.create_dataset(
                "velocity",
                data=np.arange(6, dtype=np.float64).reshape(3, 2),
            )
            fractions = np.zeros((3, 2, 3), dtype=np.float64)
            fractions[..., 0] = 0.1
            fractions[..., 1] = 0.2
            fractions[..., 2] = 0.7
            fields.create_dataset("X", data=fractions)

            constants = case.create_group("constants")
            constants.attrs["T0"] = 900.0 + case_index
            inlet = constants.create_group("inlet")
            inlet.attrs["pressure"] = 101325.0
            case.attrs["metadata"] = json.dumps(
                {
                    "record_idx": case_index,
                    "field_species": {"X": ["CH4", "O2", "N2"]},
                    "units": {"velocity": "m/s"},
                }
            )


def test_dataset_manifest_and_channel_resolution_are_worker_safe(tmp_path) -> None:
    path = tmp_path / "discovery.h5"
    _write_discovery_dataset(path)
    dataset = ChemOperatorDataset(
        path,
        task="field_map",
        coordinate_name="z",
        input_fields=("velocity", "X"),
        output_fields=("velocity", "X"),
        constant_inputs=("T0", "inlet/pressure"),
        dtype=torch.float64,
    )
    try:
        fields = dataset.inspect_fields()
        assert fields["velocity"]["channel_count"] == 1
        assert fields["velocity"]["channel_shape"] == []
        assert fields["X"]["shape"] == [3, 2, 3]
        assert fields["X"]["channel_shape"] == [3]
        assert fields["X"]["species"] == ["CH4", "O2", "N2"]
        assert dataset.resolve_field("fields/x") == "X"
        assert dataset.species_names("x") == ("CH4", "O2", "N2")
        assert dataset.resolve_species("X", "ch4") == 0
        assert dataset.resolve_channel("X", "X_O2") == 1
        assert dataset.resolve_channel("velocity") == 0
        assert dataset.resolve_channel_reference("X:N2") == ("X", 2)

        manifest = dataset.manifest()
        assert manifest["schema"] == "chem-operator-dataset-manifest-v1"
        assert manifest["case_count"] == 2
        assert manifest["hdf5_attributes"]["simulator"] == "discovery_fixture"
        assert len(manifest["fingerprint"]) == 64
        json.dumps(manifest)

        manifest["fields"]["X"]["species"].append("mutated")
        assert dataset.species_names("X") == ("CH4", "O2", "N2")

        # Opening a sample creates the lazy handle; pickling must omit it.
        assert dataset[0]["output_fields"]["X"].shape == (2, 2, 3)
        worker_dataset = pickle.loads(pickle.dumps(dataset))
        try:
            assert worker_dataset._file_handle is None
            assert worker_dataset.fingerprint() == dataset.fingerprint()
            assert worker_dataset[0]["constant_inputs"]["T0"].ndim == 0
        finally:
            worker_dataset.close()

        fingerprint = dataset.fingerprint()
        dataset.close()
        assert dataset.fingerprint() == fingerprint
        assert dataset.resolve_species("X", "O2") == 1
    finally:
        dataset.close()


def test_dataset_channel_resolution_errors_are_explicit(tmp_path) -> None:
    path = tmp_path / "discovery.h5"
    _write_discovery_dataset(path)
    dataset = ChemOperatorDataset(
        path,
        task="field_map",
        coordinate_name="z",
    )
    try:
        with pytest.raises(KeyError, match="Available fields"):
            dataset.resolve_field("missing")
        with pytest.raises(ValueError, match="channel name or index"):
            dataset.resolve_channel("X")
        with pytest.raises(IndexError, match="outside"):
            dataset.resolve_channel("X", 10)
        with pytest.raises(KeyError, match="species metadata"):
            dataset.resolve_species("velocity", "CH4")
    finally:
        dataset.close()
