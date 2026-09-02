"""Structured sample to model-tensor processing."""

from collections.abc import Mapping
from typing import Any

import torch

from chem_operator.normalization import IdentityNormalizer, Normalizer

from .config import (NormalizationConfig, PackedFieldLayout,
                     TargetTransformConfig)
from .packing import FieldPacker


class DataProcessor:
    """Transform one raw ``ChemOperatorDataset`` sample into model tensors."""

    def __init__(
        self,
        *,
        field_packer: FieldPacker,
        normalizer: Normalizer | None = None,
        normalization_config: NormalizationConfig | None = None,
        target_transform: TargetTransformConfig | None = None,
        preserve_auxiliary: bool = True,
    ):
        self.field_packer = field_packer
        self.normalizer = normalizer or IdentityNormalizer()
        self.normalization_config = normalization_config or NormalizationConfig()
        self.target_transform = target_transform or TargetTransformConfig()
        self.preserve_auxiliary = preserve_auxiliary
        self._output_layout: PackedFieldLayout | None = None

    def _target_fields(
        self,
        input_fields: Mapping[str, torch.Tensor],
        output_fields: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not self.target_transform.is_delta:
            return dict(output_fields)
        targets: dict[str, torch.Tensor] = {}
        for name, output in output_fields.items():
            if name not in input_fields:
                raise KeyError(f"Delta target {name!r} is absent from the input fields.")
            input_field = input_fields[name]
            if input_field.shape[1:] != output.shape[1:]:
                raise ValueError(
                    f"Input and output shapes for {name!r} are incompatible: "
                    f"{tuple(input_field.shape)} and {tuple(output.shape)}."
                )
            reference = input_field[-1]
            if self.target_transform.is_direct_delta:
                targets[name] = output - reference
            else:
                first = output[:1] - reference
                later = output[1:] - output[:-1]
                targets[name] = torch.cat((first, later), dim=0)
        return targets

    def _normalize_fields(
        self, fields: Mapping[str, torch.Tensor], *, delta: bool
    ) -> dict[str, torch.Tensor]:
        operation = (
            self.normalizer.delta_normalize
            if delta
            else self.normalizer.normalize
        )
        return {name: operation(value, name) for name, value in fields.items()}

    def __call__(self, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
        required = {"input_fields", "output_fields", "constant_inputs"}
        missing = required - raw_sample.keys()
        if missing:
            raise KeyError("Raw sample is missing keys: " + ", ".join(sorted(missing)))
        input_fields = dict(raw_sample["input_fields"])
        output_fields = dict(raw_sample["output_fields"])
        constant_fields = dict(raw_sample["constant_inputs"])
        target_fields = self._target_fields(input_fields, output_fields)
        config = self.normalization_config
        if config.enabled:
            if config.normalize_inputs:
                input_fields = self._normalize_fields(input_fields, delta=False)
            if config.normalize_targets:
                target_fields = self._normalize_fields(
                    target_fields, delta=self.target_transform.is_delta
                )
            if config.normalize_constants:
                constant_fields = self._normalize_fields(constant_fields, delta=False)
        input_layout = self.field_packer.variable_layout(input_fields)
        output_layout = self.field_packer.variable_layout(target_fields)
        constant_layout = self.field_packer.constant_layout(constant_fields)
        x = self.field_packer.pack_variable(input_fields)
        y = self.field_packer.pack_variable(target_fields)
        constants = self.field_packer.pack_constants(constant_fields)
        self._output_layout = output_layout
        processed: dict[str, Any] = {
            "x": x,
            "y": y,
            "constants": constants,
            "labels": {
                "x": input_layout.labels,
                "y": output_layout.labels,
                "constants": constant_layout.labels,
            },
        }
        if self.preserve_auxiliary:
            for key, value in raw_sample.items():
                if key not in required:
                    processed[key] = value
        return processed

    process = __call__

    def inverse_reconstruct(
        self,
        prediction: torch.Tensor,
        model_input: torch.Tensor,
        *,
        as_fields: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Convert model targets back to physical future states."""
        target = self.field_packer.to_channel_last(prediction)
        inputs = self.field_packer.to_channel_last(model_input)
        config = self.normalization_config
        if config.enabled and config.normalize_inputs:
            inputs = self.normalizer.denormalize_flattened(inputs, "variable")
        if config.enabled and config.normalize_targets:
            if self.target_transform.is_delta:
                target = self.normalizer.delta_denormalize_flattened(target, "variable")
            else:
                target = self.normalizer.denormalize_flattened(target, "variable")
        if self.target_transform.is_delta:
            if inputs.shape[-1] != target.shape[-1]:
                raise ValueError(
                    "Delta reconstruction requires matching input and output "
                    f"channels, got {inputs.shape[-1]} and {target.shape[-1]}."
                )
            reference = inputs[..., -1, :].unsqueeze(-2)
            if self.target_transform.is_direct_delta:
                target = reference + target
            else:
                target = reference + torch.cumsum(target, dim=-2)
        reconstructed = self.field_packer.from_channel_last(target)
        if not as_fields:
            return reconstructed
        if self._output_layout is None:
            raise RuntimeError("Process at least one sample before requesting fields.")
        return self.field_packer.unpack_variable(reconstructed, layout=self._output_layout)

    inverse_transform = inverse_reconstruct
    reconstruct = inverse_reconstruct
