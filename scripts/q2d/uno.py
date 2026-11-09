"""Tune, train, and evaluate a NeuralOperator UNO on Q2D data."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from neuralop.models import UNO
from ray import tune
import torch

from scripts.q2d.common import INPUT_CHANNELS, OUTPUT_CHANNELS, parse_args, run_model

MODEL_ID = "uno"


def model_from_config(config: Mapping[str, Any], device: torch.device) -> UNO:
    """Construct a four-layer U-shaped operator that preserves the Q2D mesh."""
    width = int(config["hidden_channels"])
    modes_z = int(config["modes_z"])
    return UNO(
        in_channels=len(INPUT_CHANNELS),
        out_channels=len(OUTPUT_CHANNELS),
        hidden_channels=width,
        lifting_channels=2 * width,
        projection_channels=2 * width,
        n_layers=4,
        uno_out_channels=[width, 2 * width, 2 * width, width],
        uno_n_modes=[[modes_z, 2] for _ in range(4)],
        uno_scalings=[[1.0, 1.0], [0.5, 0.5], [1.0, 1.0], [2.0, 2.0]],
        positional_embedding="grid",
        channel_mlp_skip="linear",
    ).to(device)


def search_space() -> dict[str, Any]:
    return {
        "hidden_channels": tune.choice([16, 24, 32]),
        "modes_z": tune.choice([3, 4]),
        "learning_rate": tune.loguniform(0.0001, 0.004),
        "weight_decay": tune.loguniform(1e-08, 0.0001),
        "batch_size": 2,
    }


def main() -> None:
    args = parse_args(model_name="UNO", benchmark=False)
    run_model(args, MODEL_ID, model_from_config, search_space())


if __name__ == "__main__":
    main()
