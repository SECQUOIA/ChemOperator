"""Shared normalizer protocols, state codecs, and tensor helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol

import torch

FieldMode = Literal["variable", "constant"]
NormalizerState = dict[str, Any]

NORMALIZER_STATE_SCHEMA = "chem-operator-normalizer"
NORMALIZER_STATE_VERSION = 1


class Normalizer(Protocol):
    """Interface required by :class:`DataProcessor`."""

    def state_dict(self) -> NormalizerState: ...

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor: ...

    def normalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor: ...

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor: ...

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor: ...

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor: ...


def normalizer_state(normalizer_type: str, **values: Any) -> NormalizerState:
    return {
        "schema": NORMALIZER_STATE_SCHEMA,
        "version": NORMALIZER_STATE_VERSION,
        "type": normalizer_type,
        **values,
    }


class IdentityNormalizer:
    """A no-op normalizer with the complete normalizer interface."""

    def state_dict(self) -> NormalizerState:
        return normalizer_state("identity")

    def normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def delta_normalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def delta_denormalize(self, x: torch.Tensor, field: str) -> torch.Tensor:
        return x

    def normalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return x

    def denormalize_flattened(
        self, x: torch.Tensor, mode: FieldMode
    ) -> torch.Tensor:
        return x

    def delta_normalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return x

    def delta_denormalize_flattened(
        self, x: torch.Tensor, mode: Literal["variable"]
    ) -> torch.Tensor:
        return x


def as_stat(value: Any) -> torch.Tensor:
    tensor = torch.as_tensor(value)
    if not tensor.is_floating_point():
        tensor = tensor.to(torch.get_default_dtype())
    return tensor


def for_input(stat: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype if x.is_floating_point() else stat.dtype
    return stat.to(device=x.device, dtype=dtype)


def ordered_flattened(
    statistics: Mapping[str, torch.Tensor],
    fields: Sequence[str],
) -> torch.Tensor:
    if not fields:
        return torch.empty(0)
    return torch.cat([statistics[field].reshape(-1) for field in fields])


def tensor_state(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu().contiguous()
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "data": tensor.tolist(),
    }


def tensor_from_state(value: Any, *, name: str) -> torch.Tensor:
    if not isinstance(value, Mapping):
        raise TypeError(f"Serialized statistic {name!r} must be a mapping.")
    dtype_name = value.get("dtype")
    shape = value.get("shape")
    if not isinstance(dtype_name, str):
        raise TypeError(f"Serialized statistic {name!r} has no string dtype.")
    if (
        not isinstance(shape, Sequence)
        or isinstance(shape, (str, bytes))
        or not all(isinstance(size, int) and size >= 0 for size in shape)
    ):
        raise TypeError(f"Serialized statistic {name!r} has an invalid shape.")
    dtype = getattr(torch, dtype_name, None)
    if not isinstance(dtype, torch.dtype):
        raise ValueError(
            f"Serialized statistic {name!r} uses unsupported dtype {dtype_name!r}."
        )
    if "data" not in value:
        raise KeyError(f"Serialized statistic {name!r} has no data.")
    try:
        return torch.as_tensor(value["data"], dtype=dtype).reshape(tuple(shape))
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Serialized statistic {name!r} does not match shape {list(shape)!r}."
        ) from exc


def statistics_state(
    statistics: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        statistic: {
            field: tensor_state(value)
            for field, value in sorted(fields.items())
        }
        for statistic, fields in sorted(statistics.items())
    }


def statistics_from_state(
    value: Any,
) -> dict[str, dict[str, torch.Tensor]]:
    if not isinstance(value, Mapping):
        raise TypeError("Serialized normalizer statistics must be a mapping.")
    decoded: dict[str, dict[str, torch.Tensor]] = {}
    for statistic, raw_fields in value.items():
        if not isinstance(statistic, str) or not isinstance(raw_fields, Mapping):
            raise TypeError(
                "Serialized normalizer statistics must map names to field mappings."
            )
        decoded[statistic] = {}
        for field, raw_value in raw_fields.items():
            if not isinstance(field, str):
                raise TypeError("Serialized statistic field names must be strings.")
            decoded[statistic][field] = tensor_from_state(
                raw_value, name=f"{statistic}/{field}"
            )
    return decoded


def field_order(value: Any, *, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not all(isinstance(field, str) for field in value)
    ):
        raise TypeError(f"{name} must be a sequence of strings.")
    return tuple(value)


def state_common(
    normalizer: Any,
    statistics: Mapping[str, Mapping[str, torch.Tensor]],
) -> dict[str, Any]:
    return {
        "variable_field_order": list(normalizer.variable_field_order),
        "constant_field_order": list(normalizer.constant_field_order),
        "min_denom": float(normalizer.min_denom),
        "statistics": statistics_state(statistics),
    }
