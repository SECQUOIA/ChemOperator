"""Shared model adapter records and channel descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
import torch

ChannelAxis = Literal["first", "last"]
FNOChannelSource = Literal["parameter", "constant", "field", "species"]


@dataclass(frozen=True)
class ReferenceSample:
    """One model-independent physical reference case.

    Coordinates are stored as ``(points, dimensions)`` and values as
    ``(points, channels)``. Model adapters may expose any layout from
    ``__getitem__``; this record is solely for common evaluation and
    comparison code.
    """

    case_id: str | int
    coordinates: torch.Tensor
    values: torch.Tensor
    labels: tuple[str, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.coordinates.ndim != 2:
            raise ValueError("Reference coordinates must have shape (points, dimensions).")
        if self.values.ndim != 2:
            raise ValueError("Reference values must have shape (points, channels).")
        if self.coordinates.shape[0] != self.values.shape[0]:
            raise ValueError("Reference coordinate and value point counts differ.")
        if self.values.shape[1] != len(self.labels):
            raise ValueError("Reference labels do not match the value channels.")
        if not torch.isfinite(self.coordinates).all() or not torch.isfinite(
            self.values
        ).all():
            raise ValueError("Reference samples cannot contain NaN or infinity.")


@runtime_checkable
class ModelDataAdapter(Protocol):
    """Common boundary implemented by model-family dataset adapters."""

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Mapping[str, Any]: ...

    def reference_item(self, index: int) -> ReferenceSample: ...

    def checkpoint_config(self) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class FNOChannel:
    """Describe one input or output channel for :class:`FNOAdapter`.

    Parameters
    ----------
    label:
        Name used by the normalizer and model interface.
    source:
        ``"parameter"`` reads ``sample["metadata"]["params"][key]``;
        ``"constant"`` reads ``sample["constant_inputs"][key]``;
        ``"field"`` reconstructs a complete scalar field from the input and
        output windows; and ``"species"`` selects one species from a grouped
        field using ``sample["metadata"]["field_species"]``.
    key:
        Parameter, constant, or field name in the source sample.
    species:
        Species name required when ``source="species"``.
    display_name, unit:
        Optional presentation metadata for downstream plots and reports.

    Notes
    -----
    A converged solution field should only be configured as an input when it
    is genuinely available at inference time. Otherwise it leaks target data
    into the model input.
    """

    label: str
    source: FNOChannelSource
    key: str
    species: str | None = None
    display_name: str | None = None
    unit: str = "-"

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("FNO channel labels must be non-empty.")
        if not self.key:
            raise ValueError(f"FNO channel {self.label!r} needs a source key.")
        if self.source == "species" and not self.species:
            raise ValueError(f"Species channel {self.label!r} needs a species.")
        if self.source != "species" and self.species is not None:
            raise ValueError(
                f"Only species channels may set species; got {self.label!r}."
            )

    @property
    def title(self) -> str:
        """Return the preferred human-readable channel name."""
        return self.display_name or self.label


@dataclass(frozen=True)
class OperatorArrays:
    """Materialized operator-learning arrays and reconstruction context."""

    branch: np.ndarray
    trunk: np.ndarray
    targets: np.ndarray
    coordinates: tuple[np.ndarray, ...]
    model_inputs: np.ndarray
    labels: tuple[str, ...]
    constant_labels: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]
    trajectory_slices: tuple[slice, ...]

    @property
    def n_trajectories(self) -> int:
        return len(self.coordinates)


@dataclass(frozen=True)
class _Trajectory:
    branch: np.ndarray
    target: np.ndarray
    coordinate: np.ndarray
    model_input: np.ndarray
    labels: tuple[str, ...]
    constant_labels: tuple[str, ...]
    metadata: Mapping[str, Any]
