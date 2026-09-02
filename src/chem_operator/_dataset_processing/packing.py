"""Field packing and unpacking utilities."""

from collections.abc import Mapping, Sequence

import torch

from .config import ChannelAxis, PackedFieldLayout


class FieldPacker:
    """Pack ordered field dictionaries into channel-based model tensors."""

    def __init__(
        self,
        *,
        channel_axis: ChannelAxis = "last",
        variable_field_order: Sequence[str] = (),
        constant_field_order: Sequence[str] = (),
    ):
        if channel_axis not in {"last", "first"}:
            raise ValueError("channel_axis must be 'last' or 'first'.")
        self.channel_axis = channel_axis
        self.variable_field_order = tuple(variable_field_order)
        self.constant_field_order = tuple(constant_field_order)
        self._variable_layout: PackedFieldLayout | None = None
        self._constant_layout: PackedFieldLayout | None = None

    @staticmethod
    def _labels(name: str, width: int) -> tuple[str, ...]:
        if width == 1:
            return (name,)
        return tuple(f"{name}[{index}]" for index in range(width))

    @staticmethod
    def _ordered_names(
        fields: Mapping[str, torch.Tensor],
        configured_order: tuple[str, ...],
        *,
        kind: str,
    ) -> tuple[str, ...]:
        names = configured_order or tuple(fields.keys())
        missing = [name for name in names if name not in fields]
        extras = [name for name in fields if name not in names]
        if missing or extras:
            details = []
            if missing:
                details.append(f"missing {kind} fields: {', '.join(missing)}")
            if extras:
                details.append(f"unordered {kind} fields: {', '.join(extras)}")
            raise KeyError("; ".join(details))
        return names

    def variable_layout(self, fields: Mapping[str, torch.Tensor]) -> PackedFieldLayout:
        names = self._ordered_names(fields, self.variable_field_order, kind="variable")
        shapes: list[tuple[int, ...]] = []
        widths: list[int] = []
        labels: list[str] = []
        n_steps: int | None = None
        for name in names:
            tensor = fields[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Variable field {name!r} is not a tensor.")
            if tensor.ndim == 0:
                raise ValueError(f"Variable field {name!r} has no leading step dimension.")
            if n_steps is None:
                n_steps = tensor.shape[0]
            elif tensor.shape[0] != n_steps:
                raise ValueError(
                    "Variable fields do not share a step dimension: "
                    f"{name!r} has {tensor.shape[0]}, expected {n_steps}."
                )
            feature_shape = tuple(tensor.shape[1:])
            width = tensor[0].numel() if tensor.shape[0] else 0
            if width == 0:
                raise ValueError(f"Variable field {name!r} has no channels.")
            shapes.append(feature_shape)
            widths.append(width)
            labels.extend(self._labels(name, width))
        return PackedFieldLayout(
            names=names,
            feature_shapes=tuple(shapes),
            widths=tuple(widths),
            labels=tuple(labels),
        )

    def constant_layout(self, fields: Mapping[str, torch.Tensor]) -> PackedFieldLayout:
        names = self._ordered_names(fields, self.constant_field_order, kind="constant")
        shapes: list[tuple[int, ...]] = []
        widths: list[int] = []
        labels: list[str] = []
        for name in names:
            tensor = fields[name]
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"Constant field {name!r} is not a tensor.")
            shape = tuple(tensor.shape)
            width = tensor.numel()
            if width == 0:
                raise ValueError(f"Constant field {name!r} has no values.")
            shapes.append(shape)
            widths.append(width)
            labels.extend(self._labels(name, width))
        return PackedFieldLayout(
            names=names,
            feature_shapes=tuple(shapes),
            widths=tuple(widths),
            labels=tuple(labels),
        )

    @staticmethod
    def _validate_cat_tensors(tensors: Sequence[torch.Tensor], *, kind: str) -> None:
        if not tensors:
            return
        first = tensors[0]
        for tensor in tensors[1:]:
            if tensor.dtype != first.dtype or tensor.device != first.device:
                raise ValueError(f"All {kind} fields must have the same dtype and device.")

    def pack_variable(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        layout = self.variable_layout(fields)
        tensors = [
            fields[name].reshape(fields[name].shape[0], width)
            for name, width in zip(layout.names, layout.widths)
        ]
        self._validate_cat_tensors(tensors, kind="variable")
        packed = torch.cat(tensors, dim=-1) if tensors else torch.empty((0, 0))
        self._variable_layout = layout
        if self.channel_axis == "first":
            packed = packed.movedim(-1, 0)
        return packed

    def pack_constants(self, fields: Mapping[str, torch.Tensor]) -> torch.Tensor:
        layout = self.constant_layout(fields)
        tensors = [fields[name].reshape(-1) for name in layout.names]
        self._validate_cat_tensors(tensors, kind="constant")
        packed = torch.cat(tensors) if tensors else torch.empty(0)
        self._constant_layout = layout
        return packed

    pack_fields = pack_variable
    pack_constant_fields = pack_constants

    def to_channel_last(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.channel_axis == "last":
            return tensor
        if tensor.ndim < 2:
            raise ValueError("A packed variable tensor must have at least two axes.")
        return tensor.movedim(-2, -1)

    def from_channel_last(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.channel_axis == "last":
            return tensor
        if tensor.ndim < 2:
            raise ValueError("A packed variable tensor must have at least two axes.")
        return tensor.movedim(-1, -2)

    def unpack_variable(
        self,
        tensor: torch.Tensor,
        *,
        layout: PackedFieldLayout | None = None,
    ) -> dict[str, torch.Tensor]:
        layout = layout or self._variable_layout
        if layout is None:
            raise RuntimeError("Pack variable fields before unpacking a tensor.")
        channel_last = self.to_channel_last(tensor)
        if channel_last.shape[-1] != layout.n_channels:
            raise ValueError(
                f"Packed tensor has {channel_last.shape[-1]} channels; "
                f"the field layout expects {layout.n_channels}."
            )
        result: dict[str, torch.Tensor] = {}
        for name, shape in zip(layout.names, layout.feature_shapes):
            value = channel_last[..., layout.slices[name]]
            result[name] = value.reshape(value.shape[:-1] + shape)
        return result

    def unpack_constants(
        self,
        tensor: torch.Tensor,
        *,
        layout: PackedFieldLayout | None = None,
    ) -> dict[str, torch.Tensor]:
        layout = layout or self._constant_layout
        if layout is None:
            raise RuntimeError("Pack constant fields before unpacking a tensor.")
        if tensor.shape[-1] != layout.n_channels:
            raise ValueError(
                f"Packed tensor has {tensor.shape[-1]} channels; "
                f"the constant layout expects {layout.n_channels}."
            )
        result: dict[str, torch.Tensor] = {}
        for name, shape in zip(layout.names, layout.feature_shapes):
            value = tensor[..., layout.slices[name]]
            result[name] = value.reshape(value.shape[:-1] + shape)
        return result
