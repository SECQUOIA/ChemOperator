"""Minimum-maximum normalization."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from .base import (FieldMode, NormalizerState, for_input, normalizer_state,
                   ordered_flattened, state_common)
from .zscore import ZScoreNormalizer


class MinMaxNormalizer:
    """Linearly map field ranges to a configurable output range."""

    def __init__(
        self,
        stats: Mapping[str, Mapping[str, Any]],
        variable_field_order: Sequence[str],
        constant_field_order: Sequence[str] = (),
        feature_range: tuple[float, float] = (0.0, 1.0),
        min_denom: float = 1e-8,
    ):
        low, high = feature_range
        if high <= low:
            raise ValueError("feature_range must have an increasing (low, high).")
        self.variable_field_order = tuple(variable_field_order)
        self.constant_field_order = tuple(constant_field_order)
        self.feature_range = (float(low), float(high))
        self.min_denom = float(min_denom)
        all_fields = self.variable_field_order + self.constant_field_order
        self.minimums = ZScoreNormalizer._read(stats, "min", all_fields)
        self.maximums = ZScoreNormalizer._read(stats, "max", all_fields)
        self.delta_minimums = ZScoreNormalizer._read(stats, "min_delta", self.variable_field_order)
        self.delta_maximums = ZScoreNormalizer._read(stats, "max_delta", self.variable_field_order)
        self.flattened_minimums = {
            "variable": ordered_flattened(self.minimums, self.variable_field_order),
            "constant": ordered_flattened(self.minimums, self.constant_field_order),
        }
        self.flattened_maximums = {
            "variable": ordered_flattened(self.maximums, self.variable_field_order),
            "constant": ordered_flattened(self.maximums, self.constant_field_order),
        }
        self.flattened_delta_minimums = {
            "variable": ordered_flattened(
                self.delta_minimums, self.variable_field_order
            )
        }
        self.flattened_delta_maximums = {
            "variable": ordered_flattened(
                self.delta_maximums, self.variable_field_order
            )
        }

    def state_dict(self) -> NormalizerState:
        state = state_common(
            self,
            {
                "min": self.minimums,
                "max": self.maximums,
                "min_delta": self.delta_minimums,
                "max_delta": self.delta_maximums,
            },
        )
        state["feature_range"] = list(self.feature_range)
        return normalizer_state("minmax", **state)

    def _transform(
        self,
        x: torch.Tensor,
        minimum: torch.Tensor,
        maximum: torch.Tensor,
        *,
        inverse: bool,
    ) -> torch.Tensor:
        minimum = for_input(minimum, x)
        maximum = for_input(maximum, x)
        scale = (maximum - minimum).abs().clamp_min(self.min_denom)
        low, high = self.feature_range
        if inverse:
            return (x - low) * scale / (high - low) + minimum
        return (x - minimum) * (high - low) / scale + low

    def _field_transform(
        self,
        x: torch.Tensor,
        field: str,
        minimums: Mapping[str, torch.Tensor],
        maximums: Mapping[str, torch.Tensor],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        ZScoreNormalizer._check_field(field, minimums)
        return self._transform(x, minimums[field], maximums[field], inverse=inverse)

    def _flat_transform(
        self,
        x: torch.Tensor,
        mode: FieldMode,
        minimums: Mapping[str, torch.Tensor],
        maximums: Mapping[str, torch.Tensor],
        *,
        inverse: bool,
    ) -> torch.Tensor:
        if mode not in minimums:
            raise ValueError(f"Unsupported normalization mode {mode!r}.")
        if x.shape[-1] != minimums[mode].numel():
            raise ValueError(
                "Packed channel count does not match normalization statistics: "
                f"got {x.shape[-1]}, expected {minimums[mode].numel()}."
            )
        return self._transform(x, minimums[mode], maximums[mode], inverse=inverse)

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(x, field, self.minimums, self.maximums, inverse=False)

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(x, field, self.minimums, self.maximums, inverse=True)

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(
            x,
            field,
            self.delta_minimums,
            self.delta_maximums,
            inverse=False,
        )

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._field_transform(
            x,
            field,
            self.delta_minimums,
            self.delta_maximums,
            inverse=True,
        )

    def normalize_flattened(self, x: torch.Tensor, mode: FieldMode) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_minimums,
            self.flattened_maximums,
            inverse=False,
        )

    def denormalize_flattened(self, x: torch.Tensor, mode: FieldMode) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_minimums,
            self.flattened_maximums,
            inverse=True,
        )

    def delta_normalize_flattened(self, x: torch.Tensor, mode: Literal["variable"]) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_delta_minimums,
            self.flattened_delta_maximums,
            inverse=False,
        )

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return self._flat_transform(
            x,
            mode,
            self.flattened_delta_minimums,
            self.flattened_delta_maximums,
            inverse=True,
        )
