"""Lazy HDF5-backed operator dataset."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .hdf5 import _HDF5Mixin
from .indexing import _IndexingMixin
from .inspection import _InspectionMixin


def raw_steps_to_possible_sample_t0s(
    total_steps: int,
    n_steps_input: int,
    n_steps_output: int,
    dt_stride: int,
) -> int:
    required_steps = 1 + dt_stride * (
        n_steps_input + n_steps_output - 1
    )
    return max(0, total_steps - required_steps + 1)

class ChemOperatorDataset(
    _HDF5Mixin,
    _IndexingMixin,
    _InspectionMixin,
    Dataset,
):
    """Lazy reader for one SimulationDatasetGenerator HDF5 file.

    Samples are returned as structured dictionaries of raw, unnormalized
    tensors. The file is scanned once for names and valid windows, then each
    worker opens its own HDF5 handle on demand.

    Parameters
    ----------
    path:
        Path to the HDF5 file produced by ``SimulationDatasetGenerator``.
    task:
        Sampling layout used to construct input/output pairs. Supported values
        are ``"next_step"``, ``"rollout"``, ``"operator_pointwise"``,
        ``"operator_cartesian"``, ``"steady_map"``, and ``"field_map"``.
    input_fields:
        Field names to include in each sample's ``input_fields`` mapping. By
        default, all fields in the file are included.
    output_fields:
        Field names to include in each sample's ``output_fields`` mapping. By
        default, the selected input fields are used.
    constant_inputs:
        Constant names to include in each sample's ``constant_inputs``
        mapping. By default, all constants in the file are included.
    coordinate_name:
        Coordinate along which samples are windowed. ``"time"`` also resolves
        to a stored ``"t"`` coordinate; a sole available coordinate is used as
        a fallback.
    n_steps_input:
        Number of coordinate steps in each input window.
    n_steps_output:
        Number of coordinate steps in each fixed-length output window. This is
        ignored when the selected task or ``full_trajectory_mode`` requests
        all remaining output steps.
    index_stride:
        Step interval used to subsample input and output windows.
    prediction_horizon:
        Minimum coordinate distance from the last input step to the first
        output step. If omitted, the first output is one ``index_stride``
        beyond the final input index.
    full_trajectory_mode:
        If true, return every remaining strided output step rather than a
        fixed-length output window.
    dtype:
        Optional PyTorch or NumPy dtype, or dtype name, to which every numeric
        tensor is cast. If omitted, stored dtypes are preserved.
    """

    VALID_TASKS = {
        "next_step",
        "rollout",
        "operator_pointwise",
        "operator_cartesian",
        "steady_map",
        "field_map",
    }
    def __init__(
        self,
        path: str | Path,
        *,
        task: Literal[
            "next_step",
            "rollout",
            "operator_pointwise",
            "operator_cartesian",
            "steady_map",
            "field_map",
        ] = "next_step",
        input_fields: Sequence[str] | None = None,
        output_fields: Sequence[str] | None = None,
        constant_inputs: Sequence[str] | None = None,
        coordinate_name: str = "time",
        n_steps_input: int = 1,
        n_steps_output: int = 1,
        index_stride: int = 1,
        prediction_horizon: float | None = None,
        full_trajectory_mode: bool = False,
        dtype: torch.dtype | str | np.dtype | type | None = None,
    ):
        super().__init__()

        if task not in self.VALID_TASKS:
            raise ValueError(f"Unsupported task {task!r}.")
        if n_steps_input < 1 or n_steps_output < 1 or index_stride < 1:
            raise ValueError(
                "n_steps_input, n_steps_output, and index_stride must be positive."
            )
        if prediction_horizon is not None and prediction_horizon < 0:
            raise ValueError("prediction_horizon must be non-negative.")

        self.path = Path(path)
        self.task = task
        self.coordinate_name = coordinate_name
        self.n_steps_input = n_steps_input
        self.n_steps_output = n_steps_output
        self.index_stride = index_stride
        self.prediction_horizon = prediction_horizon
        self.full_trajectory_mode = full_trajectory_mode
        self.dtype = self._resolve_dtype(dtype)

        if not self.path.is_file():
            raise FileNotFoundError(
                f"ChemOperatorDataset expects one HDF5 file: {self.path}"
            )

        self._file_handle: h5py.File | None = None

        self.input_fields = tuple(input_fields) if input_fields is not None else None
        self.output_fields = tuple(output_fields) if output_fields is not None else None
        self.constant_inputs = (
            tuple(constant_inputs) if constant_inputs is not None else None
        )

        self.sample_index: list[tuple[str, int, int]] = []
        self.case_n_steps: dict[str, int] = {}
        self.source_coordinate_names: dict[str, str] = {}
        self.case_indices: dict[str, int] = {}

        self.field_names: tuple[str, ...] = ()
        self.constant_names: tuple[str, ...] = ()
        self.coordinate_names: tuple[str, ...] = ()
        self.case_names: tuple[str, ...] = ()
        self.file_attributes: dict[str, Any] = {}
        self.field_descriptors: dict[str, dict[str, Any]] = {}
        self.constant_descriptors: dict[str, dict[str, Any]] = {}
        self.coordinate_descriptors: dict[str, dict[str, Any]] = {}
        self.species_by_field: dict[str, tuple[str, ...]] = {}
        self._representative_metadata: dict[str, Any] = {}
        self._representative_source_coordinate_name = ""
        self._case_metadata_digest = ""

        self._build_index()

        if self.input_fields is None:
            self.input_fields = self.field_names
        if self.output_fields is None:
            self.output_fields = self.input_fields
        if self.constant_inputs is None:
            self.constant_inputs = self.constant_names

        self._validate_requested_names()

        if not self.sample_index:
            raise ValueError("No valid samples were found for the requested windowing.")
    @staticmethod
    def _resolve_dtype(
        dtype: torch.dtype | str | np.dtype | type | None,
    ) -> torch.dtype | None:
        if dtype is None:
            return None
        if isinstance(dtype, torch.dtype):
            return dtype
        if isinstance(dtype, str):
            dtype_name = dtype.removeprefix("torch.")
            torch_dtype = getattr(torch, dtype_name, None)
            if isinstance(torch_dtype, torch.dtype):
                return torch_dtype

        try:
            dtype_name = np.dtype(dtype).name
        except TypeError as exc:
            raise TypeError(f"Unsupported dtype {dtype!r}.") from exc

        torch_dtype = getattr(torch, dtype_name, None)
        if not isinstance(torch_dtype, torch.dtype):
            raise TypeError(f"Unsupported dtype {dtype!r}.")
        return torch_dtype

    def _to_tensor(self, value: Any, *, name: str) -> torch.Tensor:
        value = ChemOperatorDataset._decode(value)
        array = np.asarray(value)
        if array.dtype.kind in {"O", "S", "U"}:
            raise TypeError(f"{name!r} is non-numeric and cannot be a tensor.")
        tensor = torch.as_tensor(array)
        if self.dtype is not None:
            tensor = tensor.to(dtype=self.dtype)
        return tensor
    def _file(self) -> h5py.File:
        if self._file_handle is None:
            self._file_handle = h5py.File(self.path, "r")
        return self._file_handle

    def _slice_field(
        self,
        dataset: h5py.Dataset,
        indices: np.ndarray,
        n_steps: int,
        *,
        name: str,
    ) -> torch.Tensor:
        if self._field_shape_matches_steps(dataset, n_steps):
            return self._to_tensor(np.asarray(dataset[indices]), name=name)
        return self._to_tensor(np.asarray(dataset), name=name)

    def _load_fields(
        self,
        case: h5py.Group,
        names: Sequence[str],
        indices: np.ndarray,
        n_steps: int,
    ) -> dict[str, torch.Tensor]:
        fields = {}
        for name in names:
            dataset = self._get_dataset(case["fields"], name)
            fields[name] = self._slice_field(
                dataset,
                indices,
                n_steps,
                name=f"fields/{name}",
            )
        return fields

    def _load_coordinates(
        self,
        case: h5py.Group,
        indices: np.ndarray,
        n_steps: int,
        source_coordinate_name: str,
    ) -> dict[str, torch.Tensor]:
        coordinates: dict[str, torch.Tensor] = {}
        for name, dataset in case["coordinates"].items():
            output_name = (
                self.coordinate_name
                if name == source_coordinate_name
                else name
            )
            if name == source_coordinate_name:
                coordinates[output_name] = self._to_tensor(
                    np.asarray(dataset[indices]),
                    name=f"coordinates/{name}",
                )
            else:
                coordinates[output_name] = self._to_tensor(
                    np.asarray(dataset),
                    name=f"coordinates/{name}",
                )
        return coordinates

    def _load_constants(self, case: h5py.Group) -> dict[str, torch.Tensor]:
        constants = {}
        for name in self.constant_inputs or ():
            constants[name] = self._to_tensor(
                self._read_constant(case["constants"], name),
                name=f"constants/{name}",
            )
        return constants

    def _case_metadata(self, case: h5py.Group) -> dict[str, Any]:
        metadata = self._decode_attr_dict(case.attrs)
        raw_metadata = metadata.pop("metadata", None)
        if raw_metadata is not None:
            try:
                decoded = json.loads(raw_metadata)
            except TypeError:
                decoded = json.loads(str(raw_metadata))
            metadata.update(decoded)
        return metadata

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        case_name, input_start, output_start = self.sample_index[index]
        file = self._file()
        case = file["cases"][case_name]
        n_steps = self.case_n_steps[case_name]

        input_indices = (
            input_start
            + np.arange(self.n_steps_input, dtype=np.int64) * self.index_stride
        )
        output_indices = self._output_indices(n_steps, output_start)

        metadata = self._case_metadata(case)
        source_coordinate_name = self.source_coordinate_names[case_name]
        metadata.update(
            {
                "task": self.task,
                "file_path": str(self.path),
                "case_name": case_name,
                "case_index": self.case_indices[case_name],
                "input_start_index": input_start,
                "output_start_index": output_start,
                "input_indices": input_indices.tolist(),
                "output_indices": output_indices.tolist(),
                "index_stride": self.index_stride,
                "coordinate_name": self.coordinate_name,
                "source_coordinate_name": source_coordinate_name,
            }
        )

        return {
            "input_fields": self._load_fields(
                case,
                self.input_fields or (),
                input_indices,
                n_steps,
            ),
            "output_fields": self._load_fields(
                case,
                self.output_fields or (),
                output_indices,
                n_steps,
            ),
            "constant_inputs": self._load_constants(case),
            "input_coordinates": self._load_coordinates(
                case,
                input_indices,
                n_steps,
                source_coordinate_name,
            ),
            "output_coordinates": self._load_coordinates(
                case,
                output_indices,
                n_steps,
                source_coordinate_name,
            ),
            "metadata": metadata,
        }

    def close(self) -> None:
        if getattr(self, "_file_handle", None) is not None:
            self._file_handle.close()
            self._file_handle = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file_handle"] = None
        return state

    def __del__(self):
        self.close()
