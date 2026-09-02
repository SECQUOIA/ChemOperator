"""Root-mean-square normalization."""

from collections.abc import Mapping, Sequence
from typing import Any, Literal

import torch

from .base import (FieldMode, NormalizerState, for_input, normalizer_state,
                   ordered_flattened, state_common)
from .zscore import ZScoreNormalizer


class RMSNormalizer:
    """Normalize fields by root-mean-square statistics."""

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
        self.rmss = {
            name: value.abs().clamp_min(self.min_denom)
            for name, value in ZScoreNormalizer._read(stats, "rms", all_fields).items()
        }
        self.delta_rmss = {
            name: value.abs().clamp_min(self.min_denom)
            for name, value in ZScoreNormalizer._read(
                stats, "rms_delta", self.variable_field_order
            ).items()
        }
        self.flattened_rmss = {
            "variable": ordered_flattened(self.rmss, self.variable_field_order),
            "constant": ordered_flattened(self.rmss, self.constant_field_order),
        }
        self.flattened_delta_rmss = {
            "variable": ordered_flattened(
                self.delta_rmss, self.variable_field_order
            )
        }

    def state_dict(self) -> NormalizerState:
        return normalizer_state(
            "rms",
            **state_common(
                self, {"rms": self.rmss, "rms_delta": self.delta_rmss}
            ),
        )

    @staticmethod
    def _scale_field(
        x: torch.Tensor,
        field: str,
        statistics: Mapping[str, torch.Tensor],
        inverse: bool,
    ) -> torch.Tensor:
        ZScoreNormalizer._check_field(field, statistics)
        scale = for_input(statistics[field], x)
        return x * scale if inverse else x / scale

    @staticmethod
    def _scale_flattened(
        x: torch.Tensor,
        mode: FieldMode,
        statistics: Mapping[str, torch.Tensor],
        inverse: bool,
    ) -> torch.Tensor:
        if mode not in statistics:
            raise ValueError(f"Unsupported normalization mode {mode!r}.")
        scale = for_input(statistics[mode], x)
        if x.shape[-1] != scale.numel():
            raise ValueError(
                "Packed channel count does not match normalization statistics: "
                f"got {x.shape[-1]}, expected {scale.numel()}."
            )
        return x * scale if inverse else x / scale

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.rmss, inverse=False)

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.rmss, inverse=True)

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.delta_rmss, inverse=False)

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return self._scale_field(x, field, self.delta_rmss, inverse=True)

    def normalize_flattened(self, x: torch.Tensor, mode: FieldMode) -> torch.Tensor:
        return self._scale_flattened(x, mode, self.flattened_rmss, inverse=False)

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return self._scale_flattened(x, mode, self.flattened_rmss, inverse=True)

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return self._scale_flattened(
            x, mode, self.flattened_delta_rmss, inverse=False
        )

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return self._scale_flattened(
            x, mode, self.flattened_delta_rmss, inverse=True
        )
