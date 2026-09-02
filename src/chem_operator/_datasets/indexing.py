"""Sample-window indexing for the lazy operator dataset."""

from __future__ import annotations

import hashlib
import json

import h5py
import numpy as np


class _IndexingMixin:
    def _resolve_coordinate_name(self, coordinates: h5py.Group) -> str:
        if self.coordinate_name in coordinates:
            return self.coordinate_name
        if self.coordinate_name == "time" and "t" in coordinates:
            return "t"
        if len(coordinates) == 1:
            return next(iter(coordinates.keys()))

        available = ", ".join(sorted(coordinates.keys()))
        raise KeyError(
            f"Coordinate {self.coordinate_name!r} was not found. "
            f"Available coordinates: {available}."
        )

    def _case_steps(self, case: h5py.Group, source_coordinate_name: str) -> int:
        coordinate = np.asarray(case["coordinates"][source_coordinate_name])
        if coordinate.ndim == 0:
            raise ValueError(
                f"Coordinate {source_coordinate_name!r} must have at least one step."
            )
        return int(coordinate.shape[0])

    def _field_shape_matches_steps(
        self,
        dataset: h5py.Dataset,
        n_steps: int,
    ) -> bool:
        return dataset.ndim > 0 and dataset.shape[0] == n_steps

    def _read_coordinate_values(
        self,
        case: h5py.Group,
        source_coordinate_name: str,
    ) -> np.ndarray:
        values = np.asarray(case["coordinates"][source_coordinate_name])
        if values.ndim != 1:
            values = values.reshape(values.shape[0], -1)[:, 0]
        return values

    def _index_at_horizon(
        self,
        coordinates: np.ndarray,
        input_last_index: int,
    ) -> int:
        if self.prediction_horizon is None:
            return input_last_index + self.index_stride

        target = coordinates[input_last_index] + self.prediction_horizon
        after_input = np.arange(input_last_index + 1, coordinates.shape[0])
        candidates = after_input[coordinates[after_input] >= target]
        if candidates.size == 0:
            return coordinates.shape[0]
        return int(candidates[0])

    def _first_output_index(
        self,
        case: h5py.Group,
        input_start: int,
        source_coordinate_name: str,
    ) -> int:
        input_last = input_start + (self.n_steps_input - 1) * self.index_stride
        if self.prediction_horizon is None:
            return input_last + self.index_stride
        coordinates = self._read_coordinate_values(case, source_coordinate_name)
        return self._index_at_horizon(coordinates, input_last)

    def _output_count(self, n_steps: int, output_start: int) -> int:
        if self.full_trajectory_mode or self.task in {
            "operator_cartesian",
            "field_map",
        }:
            return max(0, 1 + (n_steps - 1 - output_start) // self.index_stride)
        return self.n_steps_output

    def _output_indices(self, n_steps: int, output_start: int) -> np.ndarray:
        count = self._output_count(n_steps, output_start)
        return output_start + np.arange(count, dtype=np.int64) * self.index_stride

    def _sample_is_valid(
        self,
        n_steps: int,
        input_start: int,
        output_start: int,
    ) -> bool:
        input_last = input_start + (self.n_steps_input - 1) * self.index_stride
        output_indices = self._output_indices(n_steps, output_start)
        return (
            input_start >= 0
            and input_last < n_steps
            and output_indices.size > 0
            and int(output_indices[-1]) < n_steps
        )
    def _build_index(self) -> None:
        field_names: tuple[str, ...] | None = None
        constant_names: tuple[str, ...] | None = None
        coordinate_names: tuple[str, ...] | None = None
        metadata_hasher = hashlib.sha256()

        with h5py.File(self.path, "r") as file:
            if "cases" not in file:
                raise KeyError(f"{self.path} does not contain a 'cases' group.")

            self.file_attributes = self._json_safe(
                self._decode_attr_dict(file.attrs)
            )
            cases = file["cases"]
            case_names = sorted(cases.keys())
            if not case_names:
                raise ValueError(f"{self.path} contains no cases.")
            self.case_names = tuple(case_names)

            for case_idx, case_name in enumerate(case_names):
                case = cases[case_name]
                case_attrs = self._json_safe(self._decode_attr_dict(case.attrs))
                metadata_hasher.update(case_name.encode("utf-8"))
                metadata_hasher.update(
                    json.dumps(
                        case_attrs,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=True,
                    ).encode("utf-8")
                )
                source_coordinate_name = self._resolve_coordinate_name(
                    case["coordinates"]
                )
                n_steps = self._case_steps(case, source_coordinate_name)

                self.case_n_steps[case_name] = n_steps
                self.source_coordinate_names[case_name] = source_coordinate_name
                self.case_indices[case_name] = case_idx

                if field_names is None:
                    if "field_names" in file:
                        field_names = self._decode_names(file["field_names"][:])
                    else:
                        field_names = tuple(sorted(self._dataset_paths(case["fields"])))
                    constant_names = tuple(
                        sorted(self._constant_paths(case["constants"]))
                    )
                    coordinate_names = tuple(sorted(case["coordinates"].keys()))
                    self._representative_metadata = self._json_safe(
                        self._case_metadata(case)
                    )
                    self._representative_source_coordinate_name = (
                        source_coordinate_name
                    )
                    self.field_descriptors = {
                        name: self._dataset_descriptor(
                            self._get_dataset(case["fields"], name)
                        )
                        for name in field_names
                    }
                    self.constant_descriptors = {
                        name: self._constant_descriptor(
                            case["constants"],
                            name,
                        )
                        for name in constant_names
                    }
                    self.coordinate_descriptors = {
                        name: self._dataset_descriptor(dataset)
                        for name, dataset in case["coordinates"].items()
                    }

                self._add_case_samples(
                    case_name,
                    case,
                    n_steps,
                    source_coordinate_name,
                )

        self.field_names = field_names or ()
        self.constant_names = constant_names or ()
        self.coordinate_names = coordinate_names or ()
        self._case_metadata_digest = metadata_hasher.hexdigest()
        self.species_by_field = self._discover_species_by_field()
        self._annotate_field_descriptors()
    def _add_case_samples(
        self,
        case_name: str,
        case: h5py.Group,
        n_steps: int,
        source_coordinate_name: str,
    ) -> None:
        if self.task == "steady_map":
            output_start = n_steps - 1 - (self.n_steps_output - 1) * self.index_stride
            if self._sample_is_valid(n_steps, 0, output_start):
                self.sample_index.append((case_name, 0, output_start))
            return

        if self.full_trajectory_mode or self.task in {
            "operator_cartesian",
            "field_map",
        }:
            output_start = self._first_output_index(case, 0, source_coordinate_name)
            if self._sample_is_valid(n_steps, 0, output_start):
                self.sample_index.append((case_name, 0, output_start))
            return

        if self.task == "operator_pointwise":
            output_start = self._first_output_index(case, 0, source_coordinate_name)
            while self._sample_is_valid(n_steps, 0, output_start):
                self.sample_index.append((case_name, 0, output_start))
                output_start += self.index_stride
            return

        for input_start in range(n_steps):
            output_start = self._first_output_index(
                case,
                input_start,
                source_coordinate_name,
            )
            if not self._sample_is_valid(n_steps, input_start, output_start):
                break
            self.sample_index.append((case_name, input_start, output_start))

    def _validate_requested_names(self) -> None:
        available_fields = set(self.field_names)
        requested_fields = set(self.input_fields or ()) | set(self.output_fields or ())
        missing_fields = sorted(requested_fields - available_fields)
        if missing_fields:
            raise KeyError(
                "Requested fields are not present in the dataset: "
                + ", ".join(missing_fields)
            )

        available_constants = set(self.constant_names)
        missing_constants = sorted(set(self.constant_inputs or ()) - available_constants)
        if missing_constants:
            raise KeyError(
                "Requested constants are not present in the dataset: "
                + ", ".join(missing_constants)
            )

