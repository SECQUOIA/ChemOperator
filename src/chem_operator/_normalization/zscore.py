"""Z-score normalization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from .base import (FieldMode, NormalizerState, as_stat, for_input,
                   normalizer_state, ordered_flattened, state_common)


class ZScoreNormalizer:
    """Normalize fields with means and standard deviations."""

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]],
        variable_field_order: Sequence[str],
        constant_field_order: Sequence[str] = (),
        min_denom: float = 1e-8,
    ):
        self.variable_field_order = tuple(variable_field_order)
        self.constant_field_order = tuple(constant_field_order)
        self.min_denom = float(min_denom)
        all_fields = self.variable_field_order + self.constant_field_order
        self.means = self._read(stats, "mean", all_fields)
        self.stds = {
            name: value.abs().clamp_min(self.min_denom)
            for name, value in self._read(stats, "std", all_fields).items()
        }
        self.delta_means = self._read(
            stats, "mean_delta", self.variable_field_order
        )
        self.delta_stds = {
            name: value.abs().clamp_min(self.min_denom)
            for name, value in self._read(
                stats, "std_delta", self.variable_field_order
            ).items()
        }
        self._make_flattened_statistics()

    def state_dict(self) -> NormalizerState:
        return normalizer_state(
            "zscore",
            **state_common(
                self,
                {
                    "mean": self.means,
                    "std": self.stds,
                    "mean_delta": self.delta_means,
                    "std_delta": self.delta_stds,
                },
            ),
        )

    @staticmethod
    def _read(
        stats: Mapping[str, Mapping[str, Any]],
        statistic: str,
        fields: Sequence[str],
    ) -> dict[str, torch.Tensor]:
        if statistic not in stats:
            raise KeyError(f"Normalization statistics do not contain {statistic!r}.")
        values = stats[statistic]
        missing = [field for field in fields if field not in values]
        if missing:
            raise KeyError(f"{statistic!r} is missing fields: {', '.join(missing)}")
        return {field: as_stat(values[field]) for field in fields}

    def _make_flattened_statistics(self) -> None:
        self.flattened_means = {
            "variable": ordered_flattened(self.means, self.variable_field_order),
            "constant": ordered_flattened(self.means, self.constant_field_order),
        }
        self.flattened_stds = {
            "variable": ordered_flattened(self.stds, self.variable_field_order),
            "constant": ordered_flattened(self.stds, self.constant_field_order),
        }
        self.flattened_delta_means = {
            "variable": ordered_flattened(
                self.delta_means, self.variable_field_order
            )
        }
        self.flattened_delta_stds = {
            "variable": ordered_flattened(
                self.delta_stds, self.variable_field_order
            )
        }

    @staticmethod
    def _check_field(field: str, statistics: Mapping[str, torch.Tensor]) -> None:
        if field not in statistics:
            raise KeyError(f"No normalization statistics exist for field {field!r}.")

    @staticmethod
    def _flat_stats(
        x: torch.Tensor,
        mode: FieldMode,
        offsets: Mapping[str, torch.Tensor],
        scales: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if mode not in offsets:
            raise ValueError(f"Unsupported normalization mode {mode!r}.")
        offset = for_input(offsets[mode], x)
        scale = for_input(scales[mode], x)
        if x.shape[-1] != offset.numel():
            raise ValueError(
                "Packed channel count does not match normalization statistics: "
                f"got {x.shape[-1]}, expected {offset.numel()}."
            )
        return offset, scale

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.means)
        return (x - for_input(self.means[field], x)) / for_input(
            self.stds[field], x
        )

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.means)
        return x * for_input(self.stds[field], x) + for_input(
            self.means[field], x
        )

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.delta_means)
        return (x - for_input(self.delta_means[field], x)) / for_input(
            self.delta_stds[field], x
        )

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        self._check_field(field, self.delta_means)
        return x * for_input(self.delta_stds[field], x) + for_input(
            self.delta_means[field], x
        )

    def normalize_flattened(self, x: torch.Tensor, mode: FieldMode) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_means, self.flattened_stds
        )
        return (x - mean) / std

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_means, self.flattened_stds
        )
        return x * std + mean

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_delta_means, self.flattened_delta_stds
        )
        return (x - mean) / std

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        mean, std = self._flat_stats(
            x, mode, self.flattened_delta_means, self.flattened_delta_stds
        )
        return x * std + mean
