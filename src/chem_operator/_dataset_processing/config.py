"""Configuration and layout records for dataset processing."""

from dataclasses import dataclass
from typing import Literal

ChannelAxis = Literal["last", "first"]
TargetMode = Literal["state", "absolute", "identity", "delta"]
MultiStepDelta = Literal["direct", "incremental", "sequential"]


@dataclass(frozen=True)
class TargetTransformConfig:
    """Configure the representation learned by a model."""

    mode: TargetMode = "state"
    multi_step_delta: MultiStepDelta = "direct"

    def __post_init__(self) -> None:
        if self.mode not in {"state", "absolute", "identity", "delta"}:
            raise ValueError(f"Unsupported target transform mode {self.mode!r}.")
        if self.multi_step_delta not in {"direct", "incremental", "sequential"}:
            raise ValueError("multi_step_delta must be 'direct' or 'incremental'.")

    @property
    def is_delta(self) -> bool:
        return self.mode == "delta"

    @property
    def is_direct_delta(self) -> bool:
        return self.multi_step_delta == "direct"


@dataclass(frozen=True)
class NormalizationConfig:
    """Choose which parts of a processed sample are normalized."""

    enabled: bool = True
    normalize_inputs: bool = True
    normalize_targets: bool = True
    normalize_constants: bool = True


@dataclass(frozen=True)
class PackedFieldLayout:
    """Description of fields concatenated into a packed channel dimension."""

    names: tuple[str, ...]
    feature_shapes: tuple[tuple[int, ...], ...]
    widths: tuple[int, ...]
    labels: tuple[str, ...]

    @property
    def n_channels(self) -> int:
        return sum(self.widths)

    @property
    def slices(self) -> dict[str, slice]:
        result: dict[str, slice] = {}
        start = 0
        for name, width in zip(self.names, self.widths):
            result[name] = slice(start, start + width)
            start += width
        return result
