"""Shared contract tests for model-family data adapters."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from chem_operator.datasets import DataProcessor, FieldPacker
from chem_operator.models import (
    DeepONetAdapter,
    DeepXDEAdapter,
    ModelDataAdapter,
)


class _TrajectoryDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, _index: int):
        return {
            "input_fields": {"u": torch.tensor([1.0])},
            "output_fields": {"u": torch.tensor([2.0, 3.0])},
            "constant_inputs": {"forcing": torch.tensor(4.0)},
            "input_coordinates": {"t": torch.tensor([0.0])},
            "output_coordinates": {"t": torch.tensor([1.0, 2.0])},
            "metadata": {"record_idx": 17},
        }


def test_deeponet_adapter_exposes_model_independent_reference() -> None:
    processor = DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=("u",),
            constant_field_order=("forcing",),
        )
    )
    adapter = DeepONetAdapter(
        _TrajectoryDataset(),
        processor,
        coordinate_name="t",
        include_constants=True,
        include_initial=True,
    )

    assert DeepXDEAdapter is DeepONetAdapter
    assert isinstance(adapter, ModelDataAdapter)
    assert set(adapter[0]) >= {"branch", "trunk", "target"}

    reference = adapter.reference_item(0)
    assert reference.case_id == 17
    assert reference.labels == ("u",)
    torch.testing.assert_close(
        reference.coordinates, torch.tensor([[0.0], [1.0], [2.0]])
    )
    torch.testing.assert_close(
        reference.values, torch.tensor([[1.0], [2.0], [3.0]])
    )
    assert adapter.checkpoint_config() == {
        "format": "cartesian_product",
        "coordinate_names": ["t"],
        "include_constants": True,
        "include_input_state": True,
        "include_initial": True,
        "resample_points": None,
        "coordinate_mode": "physical",
        "max_trajectories": 1,
        "dtype": "float32",
    }


def test_deeponet_adapter_can_use_constants_only_as_branch_input() -> None:
    processor = DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=("u",),
            constant_field_order=("forcing",),
        )
    )
    adapter = DeepONetAdapter(
        _TrajectoryDataset(),
        processor,
        coordinate_name="t",
        include_constants=True,
        include_input_state=False,
    )

    torch.testing.assert_close(adapter[0]["branch"], torch.tensor([4.0]))
