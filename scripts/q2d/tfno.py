"""Tune, train, and evaluate a tensorized NeuralOperator FNO on Q2D data."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuralop.models import TFNO
from ray import tune
import torch

from scripts.q2d.common import INPUT_CHANNELS, OUTPUT_CHANNELS, parse_args, run_model

MODEL_ID = "tfno"


def model_from_config(config: Mapping[str, Any], device: torch.device) -> TFNO:
    """Construct a Tucker-factorized Q2D Fourier neural operator."""
    return TFNO(
        n_modes=(int(config["modes_z"]), int(config["modes_r"])),
        in_channels=len(INPUT_CHANNELS),
        out_channels=len(OUTPUT_CHANNELS),
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
        domain_padding=float(config.get("domain_padding", 0.0)),
        rank=float(config["rank"]),
    ).to(device)


def search_space() -> dict[str, Any]:
    """Match the FNO search budget while tuning tensor rank."""
    return {
        "modes_z": tune.choice([4, 5, 6]),
        "modes_r": tune.choice([2, 3, 4]),
        "hidden_channels": tune.choice([16, 20, 24, 28, 32]),
        "n_layers": tune.choice([6, 7, 8, 9]),
        "rank": tune.choice([0.05, 0.1, 0.2]),
        "learning_rate": tune.loguniform(0.0001, 0.004),
        "weight_decay": tune.loguniform(1e-08, 0.0001),
        "batch_size": 2,
        "domain_padding": tune.choice([0.0, 0.05, 0.1, 0.15]),
    }


def main() -> None:
    args = parse_args(model_name="TFNO", benchmark=False)
    run_model(args, MODEL_ID, model_from_config, search_space())


if __name__ == "__main__":
    main()
