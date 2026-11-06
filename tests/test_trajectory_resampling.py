from __future__ import annotations

import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from chem_operator.datasets import (
    DataProcessor,
    FieldPacker,
    TrajectoryResamplingDataset,
)
from chem_operator.models import CoordinateScaler, DeepONetAdapter
from chem_operator._models.deeponet import make_deeponet_dataloader


def _sample(times: list[float], *, scale: float = 1.0) -> dict:
    coordinate = torch.tensor(times, dtype=torch.float32)
    values = scale * coordinate
    return {
        "input_fields": {"u": values[:1, None]},
        "output_fields": {"u": values[1:, None]},
        "constant_inputs": {"forcing": torch.tensor(scale)},
        "input_coordinates": {"t": coordinate[:1]},
        "output_coordinates": {"t": coordinate[1:]},
        "metadata": {"record_idx": int(scale)},
    }


class _RaggedDataset(Dataset):
    def __init__(self, samples: list[dict]):
        self.samples = samples
        self.calls = 0
        self.closed = False

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        self.calls += 1
        return self.samples[index]

    def close(self) -> None:
        self.closed = True


def test_resampling_aligns_ragged_overshooting_trajectories() -> None:
    source = _RaggedDataset(
        [
            _sample([0.0, 0.5, 2.0, 120.0]),
            _sample([0.0, 0.25, 5.0, 150.0], scale=2.0),
        ]
    )
    target = torch.tensor([0.0, 0.25, 1.0, 10.0, 100.0])
    dataset = TrajectoryResamplingDataset(
        source,
        coordinate_name="t",
        target_coordinate=target,
    )

    first, second = dataset[0], dataset[1]
    for sample in (first, second):
        actual = torch.cat(
            (sample["input_coordinates"]["t"], sample["output_coordinates"]["t"])
        )
        torch.testing.assert_close(actual, target)
        assert sample["metadata"]["resampling"]["target_points"] == len(target)
    torch.testing.assert_close(
        torch.cat((first["input_fields"]["u"], first["output_fields"]["u"]))[:, 0],
        target,
    )
    torch.testing.assert_close(
        torch.cat((second["input_fields"]["u"], second["output_fields"]["u"]))[:, 0],
        2.0 * target,
    )


def test_resampling_keeps_last_state_at_repeated_float_coordinate() -> None:
    sample = _sample([0.0, 1.0, 1.0, 2.0])
    sample["output_fields"]["u"][1] = 3.0
    dataset = TrajectoryResamplingDataset(
        _RaggedDataset([sample]),
        coordinate_name="t",
        target_coordinate=[0.0, 1.0, 2.0],
    )

    resampled = dataset[0]
    values = torch.cat(
        (resampled["input_fields"]["u"], resampled["output_fields"]["u"])
    )[:, 0]
    torch.testing.assert_close(values, torch.tensor([0.0, 3.0, 2.0]))
    assert resampled["metadata"]["resampling"]["source_points"] == 4
    assert resampled["metadata"]["resampling"]["distinct_source_points"] == 3


def test_resampling_cache_is_reused_not_pickled_and_closed() -> None:
    source = _RaggedDataset([_sample([0.0, 1.0, 2.0])])
    dataset = TrajectoryResamplingDataset(
        source,
        coordinate_name="t",
        target_coordinate=[0.0, 0.5, 2.0],
        cache=True,
    )

    assert dataset[0] is dataset[0]
    assert source.calls == 1
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored._cache == {}
    dataset.close()
    assert dataset._cache == {}
    assert source.closed


@pytest.mark.parametrize(
    ("sample", "match"),
    [
        (_sample([0.0, 2.0, 1.0]), "strictly increasing"),
        (_sample([0.0, 1.0, 2.0]), "do not cover"),
    ],
)
def test_resampling_rejects_invalid_source_coordinates(sample, match) -> None:
    target = [0.0, 1.0, 3.0] if match == "do not cover" else [0.0, 1.0]
    dataset = TrajectoryResamplingDataset(
        _RaggedDataset([sample]),
        coordinate_name="t",
        target_coordinate=target,
    )

    with pytest.raises(ValueError, match=match):
        dataset[0]


def test_resampling_rejects_nonfinite_and_mismatched_fields() -> None:
    nonfinite = _sample([0.0, 1.0, 2.0])
    nonfinite["output_fields"]["u"][0] = torch.nan
    mismatched = _sample([0.0, 1.0, 2.0])
    mismatched["output_fields"]["u"] = torch.tensor([[1.0]])

    with pytest.raises(ValueError, match="NaN or infinity"):
        TrajectoryResamplingDataset(
            _RaggedDataset([nonfinite]),
            coordinate_name="t",
            target_coordinate=[0.0, 1.0, 2.0],
        )[0]
    with pytest.raises(ValueError, match="Output field"):
        TrajectoryResamplingDataset(
            _RaggedDataset([mismatched]),
            coordinate_name="t",
            target_coordinate=[0.0, 1.0, 2.0],
        )[0]


def test_ragged_trajectories_form_one_cartesian_deeponet_batch() -> None:
    target = np.array([0.0, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    resampled = TrajectoryResamplingDataset(
        _RaggedDataset(
            [
                _sample([0.0, 0.3, 3.0, 110.0]),
                _sample([0.0, 0.2, 8.0, 180.0], scale=2.0),
            ]
        ),
        coordinate_name="t",
        target_coordinate=target,
    )
    processor = DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=("u",),
            constant_field_order=("forcing",),
        )
    )
    adapter = DeepONetAdapter(
        resampled,
        processor,
        coordinate_name="t",
        include_constants=True,
        coordinate_mode="log1p",
        coordinate_scale=0.01,
    )
    scaler = CoordinateScaler.fit(adapter[0]["trunk"].numpy())
    batch = next(
        iter(
            make_deeponet_dataloader(
                adapter,
                batch_size=2,
                shuffle=False,
                coordinate_scaler=scaler,
            )
        )
    )

    assert batch["branch"].shape[0] == 2
    assert batch["target"].shape[:2] == (2, len(target))
    torch.testing.assert_close(batch["coordinates"][0], torch.from_numpy(target))
    torch.testing.assert_close(batch["coordinates"][1], torch.from_numpy(target))
