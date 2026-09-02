"""Shared model adapter records and channel descriptions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

ChannelAxis = Literal["first", "last"]
FNOChannelSource = Literal["parameter", "constant", "field", "species"]

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

