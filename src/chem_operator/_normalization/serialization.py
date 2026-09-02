"""Restore normalizers from their versioned, JSON-safe state."""

from collections.abc import Mapping, Sequence
from typing import Any, cast

from .base import (NORMALIZER_STATE_SCHEMA, NORMALIZER_STATE_VERSION,
                   IdentityNormalizer, Normalizer, field_order,
                   statistics_from_state)
from .minmax import MinMaxNormalizer
from .rms import RMSNormalizer
from .zscore import ZScoreNormalizer


def normalizer_from_state_dict(state: Mapping[str, Any]) -> Normalizer:
    """Restore a normalizer from ``Normalizer.state_dict()`` output."""
    if not isinstance(state, Mapping):
        raise TypeError("Normalizer state must be a mapping.")
    if state.get("schema") != NORMALIZER_STATE_SCHEMA:
        raise ValueError(
            "Unsupported normalizer state schema "
            f"{state.get('schema')!r}; expected {NORMALIZER_STATE_SCHEMA!r}."
        )
    if state.get("version") != NORMALIZER_STATE_VERSION:
        raise ValueError(
            "Unsupported normalizer state version "
            f"{state.get('version')!r}; expected {NORMALIZER_STATE_VERSION}."
        )
    normalizer_type = state.get("type")
    if normalizer_type == "identity":
        return cast(Normalizer, IdentityNormalizer())
    if normalizer_type not in {"zscore", "rms", "minmax"}:
        raise ValueError(f"Unsupported normalizer type {normalizer_type!r}.")
    variable_fields = field_order(state.get("variable_field_order"), name="variable_field_order")
    constant_fields = field_order(state.get("constant_field_order"), name="constant_field_order")
    try:
        min_denom = float(state["min_denom"])
    except (KeyError, TypeError, ValueError) as exc:
        raise TypeError("Serialized normalizer min_denom must be numeric.") from exc
    statistics = statistics_from_state(state.get("statistics"))
    if normalizer_type == "zscore":
        return ZScoreNormalizer(statistics, variable_fields, constant_fields, min_denom=min_denom)
    if normalizer_type == "rms":
        return RMSNormalizer(statistics, variable_fields, constant_fields, min_denom=min_denom)
    raw_range = state.get("feature_range")
    if (
        not isinstance(raw_range, Sequence)
        or isinstance(raw_range, (str, bytes))
        or len(raw_range) != 2
    ):
        raise TypeError("Serialized MinMax feature_range must contain two values.")
    try:
        feature_range = (float(raw_range[0]), float(raw_range[1]))
    except (TypeError, ValueError) as exc:
        raise TypeError("Serialized MinMax feature_range values must be numeric.") from exc
    return MinMaxNormalizer(
        statistics,
        variable_fields,
        constant_fields,
        feature_range=feature_range,
        min_denom=min_denom,
    )
