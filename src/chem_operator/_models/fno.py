"""Fourier neural operator dataset adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

import torch
from torch.utils.data import Dataset

from chem_operator.normalization import ZScoreNormalizer

from .arrays import FNOChannel, ReferenceSample


class FNOAdapter(Dataset):
    """Adapt complete tensor-product fields for NeuralOperator FNOs.

    The adapter supports one-dimensional profiles, transient ``(time, space)``
    trajectories, and steady ``(axial, radial)`` fields through configurable
    :class:`FNOChannel` definitions. Scalar parameters and constants are
    spatially broadcast; scalar fields and selected species retain their
    complete tensor-product grids.

    The legacy ``field_names`` / ``constant_names`` interface remains a
    shorthand for constant inputs, scalar-field outputs, and coordinates
    ``(time_coordinate, space_coordinate)``.
    """

    def __init__(  # pylint: disable=too-many-arguments
        self,
        dataset: Dataset,
        normalizer: ZScoreNormalizer,
        *,
        input_channels: Sequence[FNOChannel] | None = None,
        output_channels: Sequence[FNOChannel] | None = None,
        coordinate_names: Sequence[str] | None = None,
        field_names: Sequence[str] | None = None,
        constant_names: Sequence[str] | None = None,
        time_coordinate: str = "t",
        space_coordinate: str = "r",
        max_trajectories: int | None = None,
    ) -> None:
        configured_interface = (
            input_channels is not None
            or output_channels is not None
            or coordinate_names is not None
        )
        legacy_interface = field_names is not None or constant_names is not None
        if configured_interface and legacy_interface:
            raise ValueError(
                "Use either input_channels/output_channels/coordinate_names or "
                "the legacy field_names/constant_names interface, not both."
            )
        if configured_interface:
            if input_channels is None or output_channels is None:
                raise ValueError(
                    "input_channels and output_channels must be supplied together."
                )
            resolved_inputs = tuple(input_channels)
            resolved_outputs = tuple(output_channels)
            resolved_coordinates = coordinate_names or (
                time_coordinate,
                space_coordinate,
            )
        else:
            if not field_names:
                raise ValueError("field_names must contain at least one field.")
            if not constant_names:
                raise ValueError("constant_names must contain at least one constant.")
            resolved_inputs = tuple(
                FNOChannel(name, "constant", name) for name in constant_names
            )
            resolved_outputs = tuple(
                FNOChannel(name, "field", name) for name in field_names
            )
            resolved_coordinates = (time_coordinate, space_coordinate)

        if not resolved_inputs:
            raise ValueError("input_channels must contain at least one channel.")
        if not resolved_outputs:
            raise ValueError("output_channels must contain at least one channel.")
        if isinstance(resolved_coordinates, str):
            raise TypeError("coordinate_names must be a sequence of names.")
        resolved_coordinates = tuple(resolved_coordinates)
        if not resolved_coordinates:
            raise ValueError("coordinate_names must contain at least one name.")
        if (
            any(not name for name in resolved_coordinates)
            or len(set(resolved_coordinates)) != len(resolved_coordinates)
        ):
            raise ValueError(
                "FNO coordinate names must be distinct and non-empty."
            )
        labels = [
            channel.label for channel in resolved_inputs + resolved_outputs
        ]
        if len(labels) != len(set(labels)):
            raise ValueError("FNO channel labels must be globally unique.")
        if max_trajectories is not None and max_trajectories < 1:
            raise ValueError("max_trajectories must be positive.")

        self.dataset = dataset
        self.normalizer = normalizer
        self.input_channels = resolved_inputs
        self.output_channels = resolved_outputs
        self.coordinate_names = resolved_coordinates
        # Compatibility attributes used by existing scripts and checkpoints.
        self.field_names = tuple(
            channel.label for channel in self.output_channels
        )
        self.constant_names = tuple(
            channel.label for channel in self.input_channels
        )
        self.time_coordinate = self.coordinate_names[0]
        self.space_coordinate = (
            self.coordinate_names[1]
            if len(self.coordinate_names) > 1
            else None
        )
        self.max_trajectories = max_trajectories

    def checkpoint_config(self) -> dict[str, Any]:
        """Return the JSON-compatible adapter definition for a checkpoint."""

        return {
            "input_channels": [
                asdict(channel) for channel in self.input_channels
            ],
            "output_channels": [
                asdict(channel) for channel in self.output_channels
            ],
            "coordinate_names": list(self.coordinate_names),
            "max_trajectories": self.max_trajectories,
        }

    @staticmethod
    def required_field_names(
        *channel_groups: Sequence[FNOChannel],
    ) -> tuple[str, ...]:
        """Return unique raw field names required by channel definitions."""
        return tuple(
            dict.fromkeys(
                channel.key
                for channels in channel_groups
                for channel in channels
                if channel.source in {"field", "species"}
            )
        )

    @staticmethod
    def required_constant_names(
        *channel_groups: Sequence[FNOChannel],
    ) -> tuple[str, ...]:
        """Return unique raw constants required by channel definitions."""
        return tuple(
            dict.fromkeys(
                channel.key
                for channels in channel_groups
                for channel in channels
                if channel.source == "constant"
            )
        )

    def __len__(self) -> int:
        if self.max_trajectories is None:
            return len(self.dataset)
        return min(self.max_trajectories, len(self.dataset))

    @staticmethod
    def _trajectory(
        sample: Mapping[str, Any],
        field_name: str,
    ) -> torch.Tensor:
        try:
            initial = sample["input_fields"][field_name]
            future = sample["output_fields"][field_name]
        except KeyError as exc:
            raise KeyError(f"FNO field {field_name!r} is unavailable.") from exc
        return torch.cat((initial, future), dim=0)

    @classmethod
    def resolve_channel(
        cls,
        sample: Mapping[str, Any],
        channel: FNOChannel,
    ) -> torch.Tensor:
        """Resolve one physical channel before normalization or broadcasting."""
        if channel.source == "parameter":
            params = sample.get("metadata", {}).get("params", {})
            if channel.key not in params:
                raise KeyError(
                    f"FNO parameter {channel.key!r} is unavailable in "
                    "sample metadata."
                )
            value = torch.as_tensor(params[channel.key])
        elif channel.source == "constant":
            try:
                value = sample["constant_inputs"][channel.key]
            except KeyError as exc:
                raise KeyError(
                    f"FNO constant {channel.key!r} is unavailable."
                ) from exc
        elif channel.source == "field":
            value = cls._trajectory(sample, channel.key)
        elif channel.source == "species":
            grouped = cls._trajectory(sample, channel.key)
            species_by_field = sample.get("metadata", {}).get(
                "field_species",
                {},
            )
            species_names = species_by_field.get(channel.key)
            if species_names is None:
                raise KeyError(
                    f"FNO species metadata for field {channel.key!r} "
                    "is unavailable."
                )
            try:
                species_index = list(species_names).index(channel.species)
            except ValueError as exc:
                raise KeyError(
                    f"Species {channel.species!r} is absent from "
                    f"field {channel.key!r}."
                ) from exc
            if grouped.ndim < 2 or grouped.shape[-1] != len(species_names):
                raise ValueError(
                    f"FNO species field {channel.key!r} must have "
                    "(*spatial coordinates, species) shape; "
                    f"received {tuple(grouped.shape)}."
                )
            value = grouped[..., species_index]
        else:
            raise ValueError(f"Unsupported FNO channel source {channel.source!r}.")

        value = torch.as_tensor(value)
        if not value.is_floating_point():
            value = value.to(torch.get_default_dtype())
        if not torch.isfinite(value).all():
            raise ValueError(
                f"FNO channel {channel.label!r} contains NaN or infinity."
            )
        return value

    @staticmethod
    def _as_grid(
        value: torch.Tensor,
        shape: tuple[int, ...],
        label: str,
    ) -> torch.Tensor:
        if value.numel() == 1:
            return value.reshape(()).expand(shape)
        if tuple(value.shape) != shape:
            raise ValueError(
                f"FNO channel {label!r} must be scalar or have shape {shape}; "
                f"received {tuple(value.shape)}."
            )
        return value

    def _coordinates(
        self,
        sample: Mapping[str, Any],
        shape: tuple[int, ...],
    ) -> tuple[torch.Tensor, ...]:
        first_name = self.coordinate_names[0]
        first_values = torch.cat(
            (
                sample["input_coordinates"][first_name],
                sample["output_coordinates"][first_name],
            )
        ).reshape(-1)
        coordinates = [first_values]
        for name in self.coordinate_names[1:]:
            input_values = sample["input_coordinates"][name].reshape(-1)
            output_values = sample["output_coordinates"][name].reshape(-1)
            if input_values.shape != output_values.shape or not torch.equal(
                input_values,
                output_values,
            ):
                raise ValueError(
                    f"Coordinate {name!r} must be unchanged "
                    "across the complete trajectory."
                )
            coordinates.append(output_values)

        coordinate_shape = tuple(value.numel() for value in coordinates)
        if coordinate_shape != shape:
            raise ValueError(
                "Coordinate sizes do not match the FNO field grid: "
                f"coordinates {coordinate_shape}, "
                f"field {shape}."
            )
        for name, values in zip(self.coordinate_names, coordinates):
            if values.numel() < 2 or not torch.all(
                values[1:] > values[:-1]
            ):
                raise ValueError(
                    f"FNO coordinate {name!r} must contain at least two "
                    "strictly increasing points."
                )
        return tuple(coordinates)

    def physical_item(self, index: int) -> dict[str, Any]:
        """Return one unnormalized model input, target, and grid."""
        sample = self.dataset[index]
        resolved_outputs = [
            self.resolve_channel(sample, channel)
            for channel in self.output_channels
        ]
        field_outputs = [
            value for value in resolved_outputs if value.numel() != 1
        ]
        if not field_outputs:
            raise ValueError(
                "At least one FNO output channel must define the spatial grid."
            )
        if field_outputs[0].ndim != len(self.coordinate_names):
            raise ValueError(
                "FNO spatial channels must have one dimension per configured "
                f"coordinate ({len(self.coordinate_names)}); received "
                f"{tuple(field_outputs[0].shape)}."
            )
        shape = tuple(field_outputs[0].shape)
        output = torch.stack(
            [
                self._as_grid(value, shape, channel.label)
                for value, channel in zip(
                    resolved_outputs,
                    self.output_channels,
                )
            ]
        )
        model_input = torch.stack(
            [
                self._as_grid(
                    self.resolve_channel(sample, channel),
                    shape,
                    channel.label,
                )
                for channel in self.input_channels
            ]
        )
        coordinates = self._coordinates(sample, shape)
        return {
            "x": model_input,
            "y": output,
            **dict(zip(self.coordinate_names, coordinates)),
            "metadata": sample.get("metadata", {}),
        }

    def reference_item(self, index: int) -> ReferenceSample:
        """Return the physical target in the shared point/channel layout."""
        physical = self.physical_item(index)
        coordinate_mesh = torch.meshgrid(
            *(physical[name] for name in self.coordinate_names),
            indexing="ij",
        )
        coordinates = torch.stack(coordinate_mesh, dim=-1).reshape(
            -1,
            len(self.coordinate_names),
        )
        values = physical["y"].movedim(0, -1).reshape(
            -1,
            len(self.output_channels),
        )
        metadata = physical["metadata"]
        case_id = metadata.get("record_idx", index)
        return ReferenceSample(
            case_id=case_id,
            coordinates=coordinates,
            values=values,
            labels=tuple(channel.label for channel in self.output_channels),
            metadata=metadata,
        )

    def __getitem__(self, index: int) -> dict[str, Any]:
        physical = self.physical_item(index)
        target = torch.stack(
            [
                self.normalizer.normalize(
                    physical["y"][index],
                    channel.label,
                )
                for index, channel in enumerate(self.output_channels)
            ],
            dim=0,
        )
        model_input = torch.stack(
            [
                self.normalizer.normalize(
                    physical["x"][index],
                    channel.label,
                )
                for index, channel in enumerate(self.input_channels)
            ],
            dim=0,
        )
        return {
            "x": model_input,
            "y": target,
            **{
                name: physical[name]
                for name in self.coordinate_names
            },
        }

    def denormalize_output(self, output: torch.Tensor) -> torch.Tensor:
        """Convert channel-first FNO output back to physical field values."""
        channel_axis = -(len(self.coordinate_names) + 1)
        if (
            output.ndim < len(self.coordinate_names) + 1
            or output.shape[channel_axis] != len(self.output_channels)
        ):
            raise ValueError(
                "FNO output must end in (channel, *spatial) with "
                f"{len(self.output_channels)} channels; received "
                f"{tuple(output.shape)}."
            )
        fields = [
            self.normalizer.denormalize(
                output.select(channel_axis, index),
                channel.label,
            )
            for index, channel in enumerate(self.output_channels)
        ]
        return torch.stack(fields, dim=channel_axis)
