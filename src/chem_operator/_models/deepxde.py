"""DeepXDE dataset adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from chem_operator.dataset_processing import DataProcessor
from chem_operator.utils import to_numpy

from .arrays import OperatorArrays, _Trajectory
from .deepxde_geometry import ArrayTransform, DeepXDEFormat


class DeepXDEAdapter(Dataset):
    """Adapt processed trajectories to DeepXDE operator-data layouts.

    Parameters
    ----------
    dataset:
        A map-style dataset, normally a ``ChemOperatorDataset`` configured with
        ``task="operator_cartesian"`` and one input step.
    processor:
        A state-target ``DataProcessor``. Its packed final input state becomes
        the branch input. Selected constants may be appended to that branch.
    format:
        ``"cartesian_product"`` retains one branch row per trajectory and a
        shared trunk grid. ``"pointwise"`` repeats each branch row at every
        coordinate and permits trajectories with different grids.
    resample_points:
        If set, interpolate every trajectory onto a shared grid. Relative
        coordinates are useful for adaptively stepped trajectories. Resampling
        is only defined for one-dimensional trajectories.
    coordinate_names:
        Ordered coordinate names for a tensor-product spatial grid. The first
        coordinate must be the leading step dimension used by
        ``ChemOperatorDataset``; subsequent coordinates must be leading feature
        dimensions of every packed output field. ``coordinate_name`` remains
        the one-dimensional shorthand.
    """

    def __init__(
        self,
        dataset: Dataset,
        processor: DataProcessor,
        *,
        format: DeepXDEFormat = "cartesian_product",
        coordinate_name: str | None = None,
        coordinate_names: Sequence[str] | None = None,
        include_constants: bool = False,
        include_initial: bool = True,
        resample_points: int | None = None,
        coordinate_mode: Literal["physical", "relative"] = "physical",
        indices: Sequence[int] | None = None,
        max_trajectories: int | None = None,
        dtype: np.dtype | type = np.float32,
    ):
        if format not in {"cartesian_product", "pointwise"}:
            raise ValueError(
                "format must be 'cartesian_product' or 'pointwise'."
            )
        if coordinate_mode not in {"physical", "relative"}:
            raise ValueError("coordinate_mode must be 'physical' or 'relative'.")
        if resample_points is not None and resample_points < 2:
            raise ValueError("resample_points must be at least two.")
        if coordinate_name is not None and coordinate_names is not None:
            raise ValueError(
                "Use coordinate_name or coordinate_names, not both."
            )
        if isinstance(coordinate_names, str):
            raise TypeError("coordinate_names must be a sequence of names.")
        resolved_coordinate_names = (
            tuple(coordinate_names)
            if coordinate_names is not None
            else ((coordinate_name,) if coordinate_name is not None else ())
        )
        if resolved_coordinate_names and (
            any(not name for name in resolved_coordinate_names)
            or len(set(resolved_coordinate_names))
            != len(resolved_coordinate_names)
        ):
            raise ValueError(
                "coordinate_names must contain distinct, non-empty names."
            )
        if resample_points is not None and len(resolved_coordinate_names) > 1:
            raise ValueError(
                "resample_points is only supported for one-dimensional grids."
            )
        if max_trajectories is not None and max_trajectories < 1:
            raise ValueError("max_trajectories must be positive.")
        if processor.target_transform.is_delta:
            raise ValueError(
                "DeepXDEAdapter requires state targets; configure "
                "TargetTransformConfig(mode='state')."
            )

        self.dataset = dataset
        self.processor = processor
        self.format = format
        self.coordinate_names = resolved_coordinate_names
        # Keep the original attribute useful for one-dimensional callers.
        self.coordinate_name = (
            resolved_coordinate_names[0]
            if len(resolved_coordinate_names) == 1
            else None
        )
        self.include_constants = include_constants
        self.include_initial = include_initial
        self.resample_points = resample_points
        self.coordinate_mode = coordinate_mode
        if indices is None:
            selected = tuple(range(len(dataset)))
        else:
            selected = tuple(indices)
        if max_trajectories is not None:
            selected = selected[:max_trajectories]
        if not selected:
            raise ValueError("The adapter received no trajectory indices.")
        self.indices = selected
        self.dtype = np.dtype(dtype)
        self._arrays: OperatorArrays | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        """Load and process exactly one trajectory for a ``DataLoader``."""

        item = self._load_trajectory(self.indices[position])
        if self.resample_points is not None:
            resampled, trunk = self._resample([item])
            item = resampled[0]
        else:
            trunk = self._trunk(item.coordinate)
        return {
            "branch": torch.from_numpy(item.branch),
            "trunk": torch.from_numpy(
                np.asarray(trunk, dtype=self.dtype)
            ),
            "target": torch.from_numpy(item.target),
            "coordinate": torch.from_numpy(item.coordinate),
            "labels": item.labels,
        }

    @staticmethod
    def _numpy(tensor: torch.Tensor) -> np.ndarray:
        return to_numpy(tensor)

    def _coordinate_keys(self, sample: Mapping[str, Any]) -> tuple[str, ...]:
        if self.coordinate_names:
            return self.coordinate_names
        metadata = sample.get("metadata", {})
        name = metadata.get("coordinate_name")
        if name is None:
            coordinates = sample.get("output_coordinates", {})
            if len(coordinates) != 1:
                raise KeyError(
                    "coordinate_name is required when a sample has multiple "
                    "output coordinates."
                )
            return (next(iter(coordinates)),)
        return (str(name),)

    @staticmethod
    def _coordinate_vector(
        sample: Mapping[str, Any],
        window: Literal["input", "output"],
        name: str,
    ) -> np.ndarray:
        coordinates = sample[f"{window}_coordinates"]
        if name not in coordinates:
            raise KeyError(
                f"Coordinate {name!r} is absent from {window}_coordinates."
            )
        value = DeepXDEAdapter._numpy(coordinates[name]).reshape(-1)
        if value.size == 0:
            raise ValueError(f"Coordinate {name!r} has no points.")
        return value

    @staticmethod
    def _mesh_coordinate(vectors: Sequence[np.ndarray]) -> np.ndarray:
        if len(vectors) == 1:
            return vectors[0]
        mesh = np.meshgrid(*vectors, indexing="ij")
        return np.stack(mesh, axis=-1).reshape(-1, len(vectors))

    def _coordinate_vectors(
        self,
        processed: Mapping[str, Any],
        output_steps: int,
    ) -> tuple[np.ndarray, ...]:
        names = self._coordinate_keys(processed)
        if self.resample_points is not None and len(names) > 1:
            raise ValueError(
                "resample_points is only supported for one-dimensional grids."
            )
        output_first = self._coordinate_vector(processed, "output", names[0])
        if output_first.size != output_steps:
            raise ValueError(
                f"Leading coordinate {names[0]!r} has {output_first.size} "
                f"output points but the target has {output_steps} steps."
            )
        if self.include_initial:
            input_first = self._coordinate_vector(
                processed, "input", names[0]
            )
            first = np.concatenate((input_first[-1:], output_first))
        else:
            first = output_first

        vectors = [first]
        for name in names[1:]:
            input_values = self._coordinate_vector(processed, "input", name)
            output_values = self._coordinate_vector(processed, "output", name)
            if (
                input_values.shape != output_values.shape
                or not np.allclose(
                    input_values,
                    output_values,
                    rtol=1e-6,
                    atol=1e-8,
                )
            ):
                raise ValueError(
                    f"Coordinate {name!r} must be unchanged between input "
                    "and output windows."
                )
            vectors.append(output_values)

        for name, values in zip(names, vectors):
            if values.size < 2 or np.any(np.diff(values) <= 0):
                raise ValueError(
                    f"Coordinate {name!r} must contain at least two strictly "
                    "increasing points."
                )
        return tuple(
            np.asarray(values, dtype=self.dtype) for values in vectors
        )

    def _spatial_target(
        self,
        tensor: torch.Tensor,
        secondary_shape: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        """Restore spatial feature axes hidden by ``FieldPacker``."""

        if not secondary_shape:
            values = self._numpy(
                self.processor.field_packer.to_channel_last(tensor)
            )
            return values, ()

        layout = getattr(self.processor, "_output_layout", None)
        if layout is None:
            raise RuntimeError(
                "DataProcessor did not expose an output field layout."
            )
        fields = self.processor.field_packer.unpack_variable(
            tensor,
            layout=layout,
        )
        spatial_fields: list[np.ndarray] = []
        labels: list[str] = []
        for name in layout.names:
            field = self._numpy(fields[name])
            actual_secondary = tuple(
                field.shape[1 : 1 + len(secondary_shape)]
            )
            if actual_secondary != secondary_shape:
                raise ValueError(
                    f"Field {name!r} has secondary spatial shape "
                    f"{actual_secondary}; expected {secondary_shape} from "
                    "coordinate_names."
                )
            channel_shape = field.shape[1 + len(secondary_shape) :]
            channel_width = int(np.prod(channel_shape)) if channel_shape else 1
            spatial_fields.append(
                field.reshape(
                    (field.shape[0], *secondary_shape, channel_width)
                )
            )
            if channel_width == 1:
                labels.append(name)
            else:
                labels.extend(
                    f"{name}[{index}]" for index in range(channel_width)
                )
        return np.concatenate(spatial_fields, axis=-1), tuple(labels)

    def _trunk(self, coordinate: np.ndarray) -> np.ndarray:
        values = (
            coordinate.reshape(-1, 1)
            if coordinate.ndim == 1
            else coordinate
        )
        if self.coordinate_mode == "relative":
            minimum = np.min(values, axis=0)
            span = np.max(values, axis=0) - minimum
            if np.any(span <= 0):
                raise ValueError("Cannot normalize a zero-width coordinate.")
            values = (values - minimum) / span
        return np.asarray(values, dtype=self.dtype)

    def _load_trajectory(self, index: int) -> _Trajectory:
        processed = self.processor(self.dataset[index])
        packer = self.processor.field_packer
        x = self._numpy(packer.to_channel_last(processed["x"]))
        y = self._numpy(packer.to_channel_last(processed["y"]))
        constants = self._numpy(processed["constants"]).reshape(-1)
        branch = x[-1].reshape(-1)
        if self.include_constants and constants.size:
            branch = np.concatenate((branch, constants))

        vectors = self._coordinate_vectors(processed, y.shape[0])
        secondary_shape = tuple(vector.size for vector in vectors[1:])
        if secondary_shape:
            spatial_y, labels = self._spatial_target(
                processed["y"], secondary_shape
            )
            spatial_x, input_labels = self._spatial_target(
                processed["x"], secondary_shape
            )
            if input_labels != labels:
                raise ValueError(
                    "Input and output field labels differ on the spatial grid."
                )
            target_grid = spatial_y
            if self.include_initial:
                target_grid = np.concatenate(
                    (spatial_x[-1:], target_grid),
                    axis=0,
                )
            target = target_grid.reshape(-1, target_grid.shape[-1])
        else:
            labels = tuple(processed["labels"]["y"])
            target = y
            if self.include_initial:
                target = np.concatenate((x[-1:], target), axis=0)

        coordinate = self._mesh_coordinate(vectors)
        if coordinate.shape[0] != target.shape[0]:
            raise ValueError(
                f"Coordinate mesh has {coordinate.shape[0]} points but the "
                f"target has {target.shape[0]}."
            )

        return _Trajectory(
            branch=branch.astype(self.dtype, copy=False),
            target=target.astype(self.dtype, copy=False),
            coordinate=coordinate.astype(self.dtype, copy=False),
            model_input=x.astype(self.dtype, copy=False),
            labels=labels,
            constant_labels=tuple(processed["labels"]["constants"]),
            metadata=processed.get("metadata", {}),
        )

    @staticmethod
    def _interpolate(
        source_coordinate: np.ndarray,
        values: np.ndarray,
        target_coordinate: np.ndarray,
    ) -> np.ndarray:
        columns = [
            np.interp(target_coordinate, source_coordinate, values[:, channel])
            for channel in range(values.shape[-1])
        ]
        return np.stack(columns, axis=-1).astype(values.dtype, copy=False)

    def _resample(
        self, trajectories: Sequence[_Trajectory]
    ) -> tuple[list[_Trajectory], np.ndarray]:
        assert self.resample_points is not None
        if any(item.coordinate.ndim != 1 for item in trajectories):
            raise ValueError(
                "resample_points is only supported for one-dimensional grids."
            )
        if self.coordinate_mode == "relative":
            common = np.linspace(0.0, 1.0, self.resample_points, dtype=self.dtype)
        else:
            low = max(float(item.coordinate[0]) for item in trajectories)
            high = min(float(item.coordinate[-1]) for item in trajectories)
            if high <= low:
                raise ValueError("Trajectory coordinate ranges do not overlap.")
            common = np.linspace(low, high, self.resample_points, dtype=self.dtype)

        result: list[_Trajectory] = []
        for item in trajectories:
            if self.coordinate_mode == "relative":
                start = float(item.coordinate[0])
                span = float(item.coordinate[-1] - item.coordinate[0])
                if span <= 0:
                    raise ValueError("Cannot normalize a zero-width coordinate.")
                source = (item.coordinate - start) / span
                plot_coordinate = start + common * span
            else:
                source = item.coordinate
                plot_coordinate = common
            target = self._interpolate(source, item.target, common)
            result.append(
                _Trajectory(
                    branch=item.branch,
                    target=target,
                    coordinate=plot_coordinate.astype(self.dtype, copy=False),
                    model_input=item.model_input,
                    labels=item.labels,
                    constant_labels=item.constant_labels,
                    metadata=item.metadata,
                )
            )
        trunk = common.reshape(-1, 1)
        return result, trunk

    def _materialize(self) -> OperatorArrays:
        trajectories = [self._load_trajectory(index) for index in self.indices]
        labels = trajectories[0].labels
        constant_labels = trajectories[0].constant_labels
        branch_width = trajectories[0].branch.size
        target_width = trajectories[0].target.shape[-1]
        for item in trajectories[1:]:
            if item.labels != labels or item.constant_labels != constant_labels:
                raise ValueError("Field labels vary between trajectories.")
            if (
                item.branch.size != branch_width
                or item.target.shape[-1] != target_width
            ):
                raise ValueError("Packed channel widths vary between trajectories.")

        if self.resample_points is not None:
            trajectories, common_trunk = self._resample(trajectories)
        else:
            common_trunk = self._trunk(trajectories[0].coordinate)

        branches = np.stack([item.branch for item in trajectories])
        model_inputs = np.stack([item.model_input for item in trajectories])
        coordinates = tuple(item.coordinate for item in trajectories)
        metadata = tuple(item.metadata for item in trajectories)

        if self.format == "cartesian_product":
            for item in trajectories[1:]:
                candidate = self._trunk(item.coordinate)
                if self.resample_points is None and (
                    candidate.shape != common_trunk.shape
                    or not np.allclose(candidate, common_trunk, rtol=1e-5, atol=1e-8)
                ):
                    raise ValueError(
                        "Cartesian-product data requires a shared coordinate "
                        "grid. Set resample_points or use format='pointwise'."
                    )
            targets = np.stack([item.target for item in trajectories])
            slices = tuple(
                slice(index * targets.shape[1], (index + 1) * targets.shape[1])
                for index in range(len(trajectories))
            )
            return OperatorArrays(
                branch=branches,
                trunk=common_trunk.astype(self.dtype, copy=False),
                targets=targets,
                coordinates=coordinates,
                model_inputs=model_inputs,
                labels=labels,
                constant_labels=constant_labels,
                metadata=metadata,
                trajectory_slices=slices,
            )

        repeated_branch: list[np.ndarray] = []
        trunks: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        slices_list: list[slice] = []
        start = 0
        for item in trajectories:
            count = item.target.shape[0]
            repeated_branch.append(np.repeat(item.branch[None, :], count, axis=0))
            trunks.append(self._trunk(item.coordinate))
            targets_list.append(item.target)
            slices_list.append(slice(start, start + count))
            start += count
        return OperatorArrays(
            branch=np.concatenate(repeated_branch).astype(self.dtype, copy=False),
            trunk=np.concatenate(trunks).astype(self.dtype, copy=False),
            targets=np.concatenate(targets_list).astype(self.dtype, copy=False),
            coordinates=coordinates,
            model_inputs=model_inputs,
            labels=labels,
            constant_labels=constant_labels,
            metadata=metadata,
            trajectory_slices=tuple(slices_list),
        )

    @property
    def arrays(self) -> OperatorArrays:
        if self._arrays is None:
            self._arrays = self._materialize()
        return self._arrays

    def to_deepxde_data(
        self,
        validation: "DeepXDEAdapter",
        *,
        train_targets: np.ndarray | None = None,
        validation_targets: np.ndarray | None = None,
        trunk_transform: ArrayTransform | None = None,
    ):
        """Create a DeepXDE ``Triple`` or ``TripleCartesianProd`` dataset."""

        if validation.format != self.format:
            raise ValueError("Training and validation adapters must share a format.")
        import deepxde as dde

        train = self.arrays
        valid = validation.arrays
        train_y = train.targets if train_targets is None else train_targets
        valid_y = valid.targets if validation_targets is None else validation_targets
        train_trunk = train.trunk
        valid_trunk = valid.trunk
        if trunk_transform is not None:
            train_trunk = trunk_transform(train_trunk)
            valid_trunk = trunk_transform(valid_trunk)
        train_x = (train.branch, train_trunk.astype(self.dtype, copy=False))
        valid_x = (valid.branch, valid_trunk.astype(self.dtype, copy=False))
        if self.format == "cartesian_product":
            return dde.data.TripleCartesianProd(train_x, train_y, valid_x, valid_y)
        return dde.data.Triple(train_x, train_y, valid_x, valid_y)
