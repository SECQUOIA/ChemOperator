"""Low-level HDF5 name, path, and constant helpers."""

from __future__ import annotations

from typing import Any

import h5py
import numpy as np


class _HDF5Mixin:
    @staticmethod
    def _decode(value: Any) -> Any:
        if isinstance(value, bytes):
            return value.decode()
        if isinstance(value, np.generic):
            return value.item()
        return value

    @classmethod
    def _decode_attr_dict(cls, attrs: h5py.AttributeManager) -> dict[str, Any]:
        return {key: cls._decode(value) for key, value in attrs.items()}

    @classmethod
    def _decode_names(cls, values: Any) -> tuple[str, ...]:
        return tuple(str(cls._decode(value)) for value in np.asarray(values))
    @staticmethod
    def _dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
        paths: list[str] = []
        for name, value in group.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(value, h5py.Dataset):
                paths.append(path)
            elif isinstance(value, h5py.Group):
                paths.extend(_HDF5Mixin._dataset_paths(value, path))
        return paths

    @staticmethod
    def _constant_paths(group: h5py.Group, prefix: str = "") -> list[str]:
        paths = [f"{prefix}/{name}" if prefix else name for name in group.attrs]
        for name, value in group.items():
            path = f"{prefix}/{name}" if prefix else name
            if isinstance(value, h5py.Dataset):
                paths.append(path)
            elif isinstance(value, h5py.Group):
                paths.extend(_HDF5Mixin._constant_paths(value, path))
        return paths

    @staticmethod
    def _get_dataset(group: h5py.Group, name: str) -> h5py.Dataset:
        value: h5py.Group | h5py.Dataset = group
        for part in name.split("/"):
            value = value[part]
        if not isinstance(value, h5py.Dataset):
            raise KeyError(f"{name!r} is not a dataset.")
        return value

    @staticmethod
    def _read_constant(group: h5py.Group, name: str) -> Any:
        parts = name.split("/")
        current: h5py.Group | h5py.Dataset = group

        for part in parts[:-1]:
            current = current[part]
            if not isinstance(current, h5py.Group):
                raise KeyError(f"{name!r} does not name a constant.")

        leaf = parts[-1]
        if isinstance(current, h5py.Group) and leaf in current.attrs:
            return current.attrs[leaf]
        value = current[leaf]
        if isinstance(value, h5py.Dataset):
            return value[()]
        raise KeyError(f"{name!r} does not name a tensor-valued constant.")
