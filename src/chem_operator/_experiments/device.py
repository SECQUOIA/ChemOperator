"""Worker-local device, precision, seeding, and hardware helpers."""

from __future__ import annotations

import os
import platform
import random
from typing import Any

import numpy as np
import torch


def resolve_device(requested: str | torch.device) -> torch.device:
    """Resolve a device in the calling worker without changing global state."""
    value = str(requested).lower()
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable in this worker.")
        index = 0 if device.index is None else device.index
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} was requested, but this worker exposes "
                f"{torch.cuda.device_count()} device(s)."
            )
        return torch.device("cuda", index)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable in this worker.")
    return device


def resolve_dtype(value: str | torch.dtype) -> torch.dtype:
    """Resolve a supported floating-point precision without setting defaults."""
    if isinstance(value, torch.dtype):
        dtype = value
    else:
        name = value.lower().removeprefix("torch.")
        aliases = {
            "float16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
            "float": torch.float32,
            "float64": torch.float64,
            "double": torch.float64,
        }
        try:
            dtype = aliases[name]
        except KeyError as exc:
            raise ValueError(f"Unsupported floating-point dtype {value!r}.") from exc
    if not dtype.is_floating_point:
        raise ValueError("Experiment dtype must be floating point.")
    return dtype


def seed_worker(seed: int) -> None:
    """Seed libraries in the current worker without selecting a default device."""
    if seed < 0:
        raise ValueError("seed cannot be negative.")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def hardware_metadata(device: str | torch.device) -> dict[str, Any]:
    """Describe the hardware visible to the current process."""
    resolved = resolve_device(device)
    metadata: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "device_type": resolved.type,
    }
    if resolved.type == "cuda":
        index = 0 if resolved.index is None else resolved.index
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "device_index": index,
                "device_name": properties.name,
                "device_memory_bytes": properties.total_memory,
                "cuda_version": torch.version.cuda,
            }
        )
    elif resolved.type == "mps":
        metadata["device_name"] = "Apple Metal Performance Shaders"
    else:
        metadata["device_name"] = platform.processor() or "CPU"
    return metadata
