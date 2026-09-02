"""Generic neural-operator and autoencoder adapters."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import Dataset

from chem_operator.dataset_processing import DataProcessor

from .arrays import ChannelAxis


class NeuralOperatorAdapter(Dataset):
    """Return processed tensors with a neural-operator channel convention."""

    def __init__(
        self,
        dataset: Dataset,
        processor: DataProcessor,
        *,
        channel_axis: ChannelAxis = "first",
        append_constants: bool = False,
        spatial_ndim: int | None = None,
    ):
        if channel_axis not in {"first", "last"}:
            raise ValueError("channel_axis must be 'first' or 'last'.")
        if spatial_ndim is not None and spatial_ndim < 1:
            raise ValueError("spatial_ndim must be positive when supplied.")
        self.dataset = dataset
        self.processor = processor
        self.channel_axis = channel_axis
        self.append_constants = append_constants
        self.spatial_ndim = spatial_ndim

    def __len__(self) -> int:
        return len(self.dataset)

    def _axis(self, tensor: torch.Tensor) -> torch.Tensor:
        channel_last = self.processor.field_packer.to_channel_last(tensor)
        if self.channel_axis == "first":
            result = channel_last.movedim(-1, 0)
        else:
            result = channel_last
        if self.spatial_ndim is not None:
            current_spatial_ndim = result.ndim - 1
            while current_spatial_ndim < self.spatial_ndim:
                axis = -1 if self.channel_axis == "first" else -2
                result = result.unsqueeze(axis)
                current_spatial_ndim += 1
        return result

    def _with_constants(
        self, tensor: torch.Tensor, constants: torch.Tensor
    ) -> torch.Tensor:
        if not self.append_constants or constants.numel() == 0:
            return tensor
        if self.channel_axis == "first":
            shape = (constants.numel(),) + (1,) * (tensor.ndim - 1)
            expanded = constants.reshape(shape).expand(
                (constants.numel(),) + tuple(tensor.shape[1:])
            )
            return torch.cat((tensor, expanded), dim=0)
        shape = (1,) * (tensor.ndim - 1) + (constants.numel(),)
        expanded = constants.reshape(shape).expand(
            tuple(tensor.shape[:-1]) + (constants.numel(),)
        )
        return torch.cat((tensor, expanded), dim=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        processed = self.processor(self.dataset[index])
        x = self._with_constants(
            self._axis(processed["x"]), processed["constants"]
        )
        y = self._axis(processed["y"])
        return {**processed, "x": x, "y": y}



class AutoencoderAdapter(Dataset):
    """Expose complete processed trajectories as reconstruction pairs."""

    def __init__(
        self,
        dataset: Dataset,
        processor: DataProcessor,
        *,
        flatten: bool = False,
    ):
        if processor.target_transform.is_delta:
            raise ValueError(
                "AutoencoderAdapter requires state targets so inputs and "
                "reconstruction targets use one representation."
            )
        self.dataset = dataset
        self.processor = processor
        self.flatten = flatten

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        processed = self.processor(self.dataset[index])
        packer = self.processor.field_packer
        x = packer.to_channel_last(processed["x"])
        y = packer.to_channel_last(processed["y"])
        states = torch.cat((x, y), dim=-2)
        if self.flatten:
            states = states.reshape(-1)
        return {
            **processed,
            "x": states,
            "y": states.clone(),
            "target": states.clone(),
        }

