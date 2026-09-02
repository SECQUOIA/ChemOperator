"""Dataset discovery, channel resolution, manifests, and fingerprints."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import cantera as ct
import h5py
import numpy as np


class _InspectionMixin:
    @classmethod
    def _json_safe(cls, value: Any) -> Any:
        """Convert HDF5/numpy metadata to a deterministic JSON-safe value."""
        value = cls._decode(value)
        if value is None or isinstance(value, (str, bool, int)):
            return value
        if isinstance(value, float):
            if np.isfinite(value):
                return value
            return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return cls._json_safe(value.tolist())
        if isinstance(value, Mapping):
            return {
                str(key): cls._json_safe(subvalue)
                for key, subvalue in sorted(
                    value.items(),
                    key=lambda item: str(item[0]),
                )
            }
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [cls._json_safe(subvalue) for subvalue in value]
        return str(value)

    @staticmethod
    def _dataset_descriptor(dataset: h5py.Dataset) -> dict[str, Any]:
        return {
            "shape": [int(size) for size in dataset.shape],
            "dtype": str(dataset.dtype),
            "chunks": (
                [int(size) for size in dataset.chunks]
                if dataset.chunks is not None
                else None
            ),
            "compression": dataset.compression,
        }

    @classmethod
    def _value_descriptor(cls, value: Any) -> dict[str, Any]:
        array = np.asarray(cls._decode(value))
        return {
            "shape": [int(size) for size in array.shape],
            "dtype": str(array.dtype),
            "storage": "attribute",
        }

    @staticmethod
    def _canonical_digest(value: Any) -> str:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
    @classmethod
    def _constant_descriptor(
        cls,
        group: h5py.Group,
        name: str,
    ) -> dict[str, Any]:
        parts = name.split("/")
        current: h5py.Group | h5py.Dataset = group
        for part in parts[:-1]:
            current = current[part]
            if not isinstance(current, h5py.Group):
                raise KeyError(f"{name!r} does not name a constant.")

        leaf = parts[-1]
        if isinstance(current, h5py.Group) and leaf in current.attrs:
            return cls._value_descriptor(current.attrs[leaf])
        value = current[leaf]
        if isinstance(value, h5py.Dataset):
            return {
                **cls._dataset_descriptor(value),
                "storage": "dataset",
            }
        raise KeyError(f"{name!r} does not name a tensor-valued constant.")
    def _field_channel_shape(self, field: str) -> tuple[int, ...]:
        shape = list(self.field_descriptors[field]["shape"])
        if not shape:
            return ()

        case_name = self.case_names[0]
        n_steps = self.case_n_steps[case_name]
        if shape and shape[0] == n_steps:
            shape.pop(0)

        for name, descriptor in self.coordinate_descriptors.items():
            if name == self._representative_source_coordinate_name:
                continue
            coordinate_shape = descriptor["shape"]
            if len(coordinate_shape) != 1:
                continue
            coordinate_size = int(coordinate_shape[0])
            try:
                matching_axis = shape.index(coordinate_size)
            except ValueError:
                continue
            shape.pop(matching_axis)
        return tuple(int(size) for size in shape)

    @staticmethod
    def _metadata_names(metadata: Mapping[str, Any], name: str) -> tuple[str, ...]:
        raw_names = metadata.get(name)
        if (
            not isinstance(raw_names, Sequence)
            or isinstance(raw_names, (str, bytes))
        ):
            return ()
        return tuple(str(value) for value in raw_names)

    def _discover_species_by_field(self) -> dict[str, tuple[str, ...]]:
        metadata = self._representative_metadata
        discovered: dict[str, tuple[str, ...]] = {}

        field_species = metadata.get("field_species", {})
        if isinstance(field_species, Mapping):
            for raw_field, raw_names in field_species.items():
                if (
                    not isinstance(raw_names, Sequence)
                    or isinstance(raw_names, (str, bytes))
                ):
                    continue
                try:
                    field = self.resolve_field(str(raw_field))
                except KeyError:
                    continue
                discovered[field] = tuple(str(name) for name in raw_names)

        gas_species = (
            self._metadata_names(metadata, "gas_species")
            or self._metadata_names(metadata, "species_names")
        )
        if not gas_species and isinstance(metadata.get("mechanism"), str):
            try:
                gas_species = tuple(
                    ct.Solution(str(metadata["mechanism"])).species_names
                )
            except (ct.CanteraError, OSError, TypeError, ValueError):
                gas_species = ()
        surface_species = self._metadata_names(metadata, "surface_species")

        for field in self.field_names:
            if field in discovered:
                continue
            leaf = field.rsplit("/", maxsplit=1)[-1].casefold()
            if leaf in {"x", "y"} and gas_species:
                discovered[field] = gas_species
            elif leaf in {"z", "theta", "coverage", "coverages"} and surface_species:
                discovered[field] = surface_species

        validated: dict[str, tuple[str, ...]] = {}
        for field, names in discovered.items():
            channel_shape = self._field_channel_shape(field)
            channel_count = int(np.prod(channel_shape)) if channel_shape else 1
            if len(names) == channel_count:
                validated[field] = names
        return validated

    def _annotate_field_descriptors(self) -> None:
        for field, descriptor in self.field_descriptors.items():
            channel_shape = self._field_channel_shape(field)
            channel_count = int(np.prod(channel_shape)) if channel_shape else 1
            species = self.species_by_field.get(field, ())
            if species:
                channel_names = species
            elif channel_count == 1:
                channel_names = (field,)
            else:
                channel_names = tuple(
                    f"{field}[{index}]" for index in range(channel_count)
                )
            descriptor.update(
                {
                    "channel_shape": list(channel_shape),
                    "channel_count": channel_count,
                    "channel_names": list(channel_names),
                    "species": list(species),
                    "follows_primary_coordinate": bool(
                        descriptor["shape"]
                        and descriptor["shape"][0]
                        == self.case_n_steps[self.case_names[0]]
                    ),
                }
            )

    def resolve_field(self, field: str) -> str:
        """Resolve a field name exactly or by an unambiguous case-insensitive name."""
        if not isinstance(field, str) or not field.strip():
            raise TypeError("Field names must be non-empty strings.")
        candidate = field.strip().removeprefix("fields/")
        if candidate in self.field_names:
            return candidate

        matches = [
            name for name in self.field_names
            if name.casefold() == candidate.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        available = ", ".join(self.field_names)
        raise KeyError(
            f"Field {field!r} is unavailable. Available fields: {available}."
        )

    def inspect_fields(self) -> dict[str, dict[str, Any]]:
        """Return JSON-safe field shapes, dtypes, channels, and species."""
        return deepcopy(self.field_descriptors)

    def inspect_constants(self) -> dict[str, dict[str, Any]]:
        """Return JSON-safe constant names and storage descriptors."""
        return deepcopy(self.constant_descriptors)

    def inspect_coordinates(self) -> dict[str, dict[str, Any]]:
        """Return JSON-safe coordinate shapes and dtypes."""
        return deepcopy(self.coordinate_descriptors)

    def species_names(self, field: str) -> tuple[str, ...]:
        """Return ordered species names for a composition field, if known."""
        return self.species_by_field.get(self.resolve_field(field), ())

    def resolve_species(self, field: str, species: str) -> int:
        """Resolve a species name to its channel index."""
        resolved_field = self.resolve_field(field)
        if not isinstance(species, str) or not species.strip():
            raise TypeError("Species names must be non-empty strings.")
        names = self.species_by_field.get(resolved_field, ())
        if not names:
            raise KeyError(
                f"Field {resolved_field!r} has no species metadata."
            )
        candidate = species.strip()
        if candidate in names:
            return names.index(candidate)
        matches = [
            index for index, name in enumerate(names)
            if name.casefold() == candidate.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(
            f"Species {species!r} is unavailable in field {resolved_field!r}."
        )

    def channel_names(self, field: str) -> tuple[str, ...]:
        """Return ordered physical or synthetic channel names for a field."""
        resolved_field = self.resolve_field(field)
        return tuple(self.field_descriptors[resolved_field]["channel_names"])

    def resolve_channel(
        self,
        field: str,
        channel: str | int | None = None,
    ) -> int:
        """Resolve an integer, species, or channel label within one field."""
        resolved_field = self.resolve_field(field)
        names = self.channel_names(resolved_field)
        if channel is None:
            if len(names) == 1:
                return 0
            raise ValueError(
                f"Field {resolved_field!r} has {len(names)} channels; "
                "a channel name or index is required."
            )
        if isinstance(channel, int) and not isinstance(channel, bool):
            if 0 <= channel < len(names):
                return channel
            raise IndexError(
                f"Channel index {channel} is outside [0, {len(names)})."
            )
        if not isinstance(channel, str) or not channel.strip():
            raise TypeError("Channel must be an integer or non-empty string.")

        candidate = channel.strip()
        for separator in (":", "/", "_"):
            prefix = f"{resolved_field}{separator}"
            if candidate.casefold().startswith(prefix.casefold()):
                candidate = candidate[len(prefix):]
                break
        if candidate in names:
            return names.index(candidate)
        matches = [
            index for index, name in enumerate(names)
            if name.casefold() == candidate.casefold()
        ]
        if len(matches) == 1:
            return matches[0]
        raise KeyError(
            f"Channel {channel!r} is unavailable in field {resolved_field!r}. "
            f"Available channels: {', '.join(names)}."
        )

    def resolve_channel_reference(self, reference: str) -> tuple[str, int]:
        """Resolve ``field``, ``field:channel``, or ``field_channel`` syntax."""
        if not isinstance(reference, str) or not reference.strip():
            raise TypeError("Channel references must be non-empty strings.")
        candidate = reference.strip()
        try:
            field = self.resolve_field(candidate)
        except KeyError:
            field = ""
        else:
            return field, self.resolve_channel(field)

        for possible_field in sorted(self.field_names, key=len, reverse=True):
            for separator in (":", "/", "_"):
                prefix = f"{possible_field}{separator}"
                if candidate.casefold().startswith(prefix.casefold()):
                    channel = candidate[len(prefix):]
                    return (
                        possible_field,
                        self.resolve_channel(possible_field, channel),
                    )
        raise KeyError(f"Cannot resolve channel reference {reference!r}.")

    def manifest(self) -> dict[str, Any]:
        """Return a lightweight, JSON-safe manifest for this dataset view."""
        stat = self.path.stat()
        reader = {
            "task": self.task,
            "coordinate_name": self.coordinate_name,
            "input_fields": list(self.input_fields or ()),
            "output_fields": list(self.output_fields or ()),
            "constant_inputs": list(self.constant_inputs or ()),
            "n_steps_input": self.n_steps_input,
            "n_steps_output": self.n_steps_output,
            "index_stride": self.index_stride,
            "prediction_horizon": self.prediction_horizon,
            "full_trajectory_mode": self.full_trajectory_mode,
            "dtype": str(self.dtype) if self.dtype is not None else None,
        }
        identity = {
            "schema": "chem-operator-dataset-manifest-v1",
            "file": {
                "size_bytes": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            },
            "hdf5_attributes": self.file_attributes,
            "case_count": len(self.case_names),
            "case_names_digest": self._canonical_digest(self.case_names),
            "case_metadata_digest": self._case_metadata_digest,
            "fields": self.field_descriptors,
            "constants": self.constant_descriptors,
            "coordinates": self.coordinate_descriptors,
            "reader": reader,
        }
        return {
            **deepcopy(identity),
            "path": str(self.path.resolve()),
            "fingerprint": self._canonical_digest(identity),
        }

    def fingerprint(self) -> str:
        """Return the manifest fingerprint without reading field array contents."""
        return str(self.manifest()["fingerprint"])

    @property
    def dataset_fingerprint(self) -> str:
        """Property alias useful in artifact manifests."""
        return self.fingerprint()
