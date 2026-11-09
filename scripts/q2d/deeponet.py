"""Tune, train, and evaluate a direct DeepONet on fixed-mesh Q2D fields."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ray import tune

from scripts.q2d.common import (
    deeponet_search_space,
    parse_args,
    prepare_deeponet_data,
    run_deeponet,
)

MODEL_ID = "deeponet"


def search_space() -> dict[str, Any]:
    space = deeponet_search_space()
    space.update({
        "latent_width": tune.choice([16, 32, 64]),
        "trunk_hidden_layers": tune.choice([2, 3, 4]),
    })
    return space


def main() -> None:
    args = parse_args(model_name="DeepONet", benchmark=False)
    normalizer, geometry, coordinates = prepare_deeponet_data(args.data_dir)
    run_deeponet(args, MODEL_ID, search_space(), normalizer, geometry, coordinates)


if __name__ == "__main__":
    main()
