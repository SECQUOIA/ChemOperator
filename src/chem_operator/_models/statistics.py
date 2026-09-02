"""Streaming normalizer fitting for model adapters."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import Dataset

from chem_operator.normalization import ZScoreNormalizer

from .arrays import FNOChannel
from .fno import FNOAdapter


@dataclass
class _RunningMoments:
    count: int = 0
    mean: torch.Tensor | None = None
    m2: torch.Tensor | None = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64, device="cpu")
        if values.ndim == 0:
            values = values.reshape(1)
        batch_count = int(values.shape[0])
        if batch_count == 0:
            return
        batch_mean = values.mean(dim=0)
        batch_m2 = torch.sum((values - batch_mean) ** 2, dim=0)
        if self.mean is None:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        assert self.m2 is not None
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = (
            self.m2
            + batch_m2
            + delta**2 * (self.count * batch_count / total)
        )
        self.count = total

    def result(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.count == 0 or self.mean is None or self.m2 is None:
            raise RuntimeError("No values were accumulated.")
        variance = self.m2 / self.count
        return self.mean.float(), torch.sqrt(variance).float()


def fit_fno_zscore_normalizer(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    input_channels: Sequence[FNOChannel],
    output_channels: Sequence[FNOChannel],
    *,
    min_denom: float = 1e-8,
) -> ZScoreNormalizer:
    """Fit scalar, resolution-independent statistics for an FNO adapter.

    Every value of a spatial channel contributes to one global channel mean
    and standard deviation. Parameter and constant channels contribute once
    per sample. Consequently, fitted statistics broadcast on grids whose
    resolution differs from the training grid.

    Delta statistics are set to identity values because ``FNOAdapter`` models
    complete fields rather than one-step deltas.
    """

    inputs = tuple(input_channels)
    outputs = tuple(output_channels)
    if not inputs:
        raise ValueError("input_channels must contain at least one channel.")
    if not outputs:
        raise ValueError("output_channels must contain at least one channel.")
    labels = [channel.label for channel in inputs + outputs]
    if len(labels) != len(set(labels)):
        raise ValueError("FNO channel labels must be globally unique.")

    moments = {label: _RunningMoments() for label in labels}
    if isinstance(dataset, Dataset):
        samples: Iterable[Mapping[str, Any]] = (
            dataset[index] for index in range(len(dataset))
        )
    else:
        samples = dataset

    for sample in samples:
        for channel in inputs + outputs:
            value = FNOAdapter.resolve_channel(sample, channel)
            moments[channel.label].update(value.reshape(-1))

    means: dict[str, torch.Tensor] = {}
    stds: dict[str, torch.Tensor] = {}
    for label in labels:
        means[label], stds[label] = moments[label].result()
    output_labels = tuple(channel.label for channel in outputs)
    return ZScoreNormalizer(
        {
            "mean": means,
            "std": stds,
            "mean_delta": {
                label: torch.tensor(0.0, dtype=torch.float32)
                for label in output_labels
            },
            "std_delta": {
                label: torch.tensor(1.0, dtype=torch.float32)
                for label in output_labels
            },
        },
        variable_field_order=output_labels,
        constant_field_order=tuple(channel.label for channel in inputs),
        min_denom=min_denom,
    )


def fit_zscore_normalizer(
    dataset: Dataset | Iterable[Mapping[str, Any]],
    variable_field_order: Sequence[str],
    constant_field_order: Sequence[str] = (),
    *,
    min_denom: float = 1e-8,
) -> ZScoreNormalizer:
    """Fit field and one-step-delta statistics from raw trajectories.

    The function streams one trajectory at a time and never concatenates a
    complete HDF5 split in memory.
    """

    variables = tuple(variable_field_order)
    constants = tuple(constant_field_order)
    moments = {name: _RunningMoments() for name in variables}
    delta_moments = {name: _RunningMoments() for name in variables}
    constant_moments = {name: _RunningMoments() for name in constants}

    if isinstance(dataset, Dataset):
        samples: Iterable[Mapping[str, Any]] = (
            dataset[index] for index in range(len(dataset))
        )
    else:
        samples = dataset

    for sample in samples:
        for name in variables:
            trajectory = torch.cat(
                (sample["input_fields"][name], sample["output_fields"][name]),
                dim=0,
            )
            moments[name].update(trajectory)
            delta_moments[name].update(trajectory[1:] - trajectory[:-1])
        for name in constants:
            value = sample["constant_inputs"][name]
            constant_moments[name].update(value.unsqueeze(0))

    means: dict[str, torch.Tensor] = {}
    stds: dict[str, torch.Tensor] = {}
    delta_means: dict[str, torch.Tensor] = {}
    delta_stds: dict[str, torch.Tensor] = {}
    for name in variables:
        means[name], stds[name] = moments[name].result()
        delta_means[name], delta_stds[name] = delta_moments[name].result()
    for name in constants:
        means[name], stds[name] = constant_moments[name].result()

    return ZScoreNormalizer(
        {
            "mean": means,
            "std": stds,
            "mean_delta": delta_means,
            "std_delta": delta_stds,
        },
        variable_field_order=variables,
        constant_field_order=constants,
        min_denom=min_denom,
    )
