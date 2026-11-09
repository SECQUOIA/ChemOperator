"""Lazy resampling views for trajectory datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class TrajectoryResamplingDataset(Dataset):
    """Resample complete one-dimensional trajectories onto a shared grid.

    The wrapped dataset must return the structured samples produced by
    :class:`ChemOperatorDataset`. The selected input and output field windows
    are joined, interpolated in physical-coordinate space, and split back into
    one initial input point followed by the remaining output points.
    """

    def __init__(
        self,
        dataset: Dataset,
        *,
        coordinate_name: str,
        target_coordinate: Sequence[float] | np.ndarray | torch.Tensor,
        cache: bool = True,
    ) -> None:
        if not coordinate_name:
            raise ValueError("coordinate_name must be non-empty.")
        target = torch.as_tensor(target_coordinate).detach().cpu().reshape(-1)
        if target.numel() < 2:
            raise ValueError("target_coordinate must contain at least two points.")
        if not target.is_floating_point():
            target = target.to(torch.float64)
        if not torch.isfinite(target).all():
            raise ValueError("target_coordinate must contain only finite values.")
        if not torch.all(target[1:] > target[:-1]):
            raise ValueError("target_coordinate must be strictly increasing.")

        self.dataset = dataset
        self.coordinate_name = coordinate_name
        self.target_coordinate = target
        self.cache = bool(cache)
        self._cache: dict[int, dict[str, Any]] = {}

    def __len__(self) -> int:
        return len(self.dataset)

    @staticmethod
    def _coordinate(
        sample: Mapping[str, Any],
        window: str,
        name: str,
    ) -> torch.Tensor:
        coordinates = sample.get(f"{window}_coordinates")
        if not isinstance(coordinates, Mapping) or name not in coordinates:
            raise KeyError(f"Coordinate {name!r} is absent from {window}_coordinates.")
        value = torch.as_tensor(coordinates[name]).detach().cpu().reshape(-1)
        if not value.is_floating_point():
            value = value.to(torch.float64)
        return value

    @staticmethod
    def _interpolate(
        source: torch.Tensor,
        values: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        original_dtype = values.dtype
        work_dtype = torch.promote_types(source.dtype, target.dtype)
        if values.is_floating_point():
            work_dtype = torch.promote_types(work_dtype, values.dtype)
        else:
            work_dtype = torch.promote_types(work_dtype, torch.float32)

        source_work = source.to(work_dtype)
        target_work = target.to(work_dtype)
        values_work = values.to(work_dtype)
        right = torch.searchsorted(source_work, target_work, right=False)
        right = right.clamp(1, source_work.numel() - 1)
        left = right - 1
        span = source_work[right] - source_work[left]
        weight = (target_work - source_work[left]) / span
        weight_shape = (target_work.numel(),) + (1,) * (values_work.ndim - 1)
        result = (
            values_work[left]
            + weight.reshape(weight_shape)
            * (values_work[right] - values_work[left])
        )
        if values.is_floating_point():
            return result.to(original_dtype)
        return result

    def _resample(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        input_coordinate = self._coordinate(
            sample, "input", self.coordinate_name
        )
        output_coordinate = self._coordinate(
            sample, "output", self.coordinate_name
        )
        source = torch.cat((input_coordinate, output_coordinate))
        if not torch.isfinite(source).all():
            raise ValueError("Source coordinates must contain only finite values.")
        differences = source[1:] - source[:-1]
        if torch.any(differences < 0):
            raise ValueError("Source coordinates must be strictly increasing.")
        # Float32 dataset views can collapse distinct, very close adaptive
        # solver times. Keep the final state at each repeated coordinate.
        keep = torch.cat(
            (differences > 0, torch.ones(1, dtype=torch.bool))
        )
        source_points = int(source.numel())
        source = source[keep]
        if source.numel() < 2:
            raise ValueError("Source coordinates need at least two distinct points.")

        target = self.target_coordinate.to(dtype=source.dtype)
        if target[0] < source[0] or target[-1] > source[-1]:
            raise ValueError(
                "Source coordinates do not cover the requested target interval "
                f"[{float(target[0])}, {float(target[-1])}]."
            )

        input_fields = sample.get("input_fields")
        output_fields = sample.get("output_fields")
        if not isinstance(input_fields, Mapping) or not isinstance(
            output_fields, Mapping
        ):
            raise TypeError("Samples must contain input_fields and output_fields mappings.")
        if tuple(input_fields) != tuple(output_fields):
            raise ValueError("Input and output field names must match for resampling.")

        resampled_input: dict[str, torch.Tensor] = {}
        resampled_output: dict[str, torch.Tensor] = {}
        for name in input_fields:
            input_value = torch.as_tensor(input_fields[name]).detach().cpu()
            output_value = torch.as_tensor(output_fields[name]).detach().cpu()
            if input_value.ndim == 0 or output_value.ndim == 0:
                raise ValueError(f"Field {name!r} must have a leading time dimension.")
            if input_value.shape[0] != input_coordinate.numel():
                raise ValueError(
                    f"Input field {name!r} has {input_value.shape[0]} steps; "
                    f"expected {input_coordinate.numel()}."
                )
            if output_value.shape[0] != output_coordinate.numel():
                raise ValueError(
                    f"Output field {name!r} has {output_value.shape[0]} steps; "
                    f"expected {output_coordinate.numel()}."
                )
            if input_value.shape[1:] != output_value.shape[1:]:
                raise ValueError(
                    f"Input and output shapes differ for field {name!r}."
                )
            values = torch.cat((input_value, output_value), dim=0)
            if not torch.isfinite(values).all():
                raise ValueError(f"Field {name!r} contains NaN or infinity.")
            values = values[keep]
            interpolated = self._interpolate(source, values, target)
            resampled_input[name] = interpolated[:1]
            resampled_output[name] = interpolated[1:]

        input_coordinates = dict(sample["input_coordinates"])
        output_coordinates = dict(sample["output_coordinates"])
        input_coordinates[self.coordinate_name] = target[:1]
        output_coordinates[self.coordinate_name] = target[1:]
        metadata = deepcopy(dict(sample.get("metadata", {})))
        metadata["resampling"] = {
            "method": "linear",
            "coordinate_name": self.coordinate_name,
            "source_points": source_points,
            "distinct_source_points": int(source.numel()),
            "target_points": int(target.numel()),
            "target_min": float(target[0]),
            "target_max": float(target[-1]),
        }
        return {
            **dict(sample),
            "input_fields": resampled_input,
            "output_fields": resampled_output,
            "input_coordinates": input_coordinates,
            "output_coordinates": output_coordinates,
            "metadata": metadata,
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self.cache and index in self._cache:
            return self._cache[index]
        sample = self._resample(index)
        if self.cache:
            self._cache[index] = sample
        return sample

    def close(self) -> None:
        """Clear cached trajectories and close the underlying dataset."""

        self._cache.clear()
        close = getattr(self.dataset, "close", None)
        if callable(close):
            close()

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state
