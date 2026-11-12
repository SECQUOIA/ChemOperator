"""Import DeepXDE without changing PyTorch's process-wide default device."""

from __future__ import annotations

import os
from types import ModuleType

import torch


def import_deepxde() -> ModuleType:
    """Import DeepXDE's PyTorch backend and keep the caller's default device.

    On import, DeepXDE calls ``torch.set_default_device("cuda")`` whenever a GPU
    is visible. Every later allocation without an explicit device then lands
    on CUDA, which breaks CPU-resident datasets, samplers, and normalizers.
    ChemOperator places networks and batches on explicit devices, so the
    previous default is restored after the import.
    """
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    device = torch.get_default_device()
    try:
        import deepxde
    finally:
        torch.set_default_device(None if device.type == "cpu" else device)
    return deepxde
