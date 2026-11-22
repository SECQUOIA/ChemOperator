"""Tune, train, and evaluate a POD-DeepONet on fixed-mesh Q2D fields."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from chem_operator.models import PODTransform, fit_incremental_pod_dataset
from chem_operator.normalization import ZScoreNormalizer
from scripts.q2d.common import (
    deeponet_data,
    deeponet_search_space,
    parse_args,
    prepare_deeponet_data,
    run_deeponet,
)

MODEL_ID = "pod_deeponet"
POD_VARIANCE_THRESHOLD = 0.999


def prepare_pod(
    data_dir: Path,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    coordinates: tuple[torch.Tensor, torch.Tensor],
    max_cases: int | None = None,
) -> PODTransform:
    """Fit a joint spatial/channel basis using only selected training cases."""
    with deeponet_data(
        data_dir, ("train",), normalizer, geometry, coordinates, max_cases,
    ) as (training,):
        if len(training) < 2:
            raise ValueError("Q2D POD fitting requires at least two training cases.")
        return fit_incremental_pod_dataset(
            training, variance_threshold=POD_VARIANCE_THRESHOLD,
        )


def search_space(pod_components: int) -> dict[str, Any]:
    space = deeponet_search_space()
    space.update({"latent_width": pod_components, "trunk_hidden_layers": 0})
    return space


def main() -> None:
    args = parse_args(model_name="POD-DeepONet", benchmark=False)
    normalizer, geometry, coordinates = prepare_deeponet_data(args.data_dir)
    pod = prepare_pod(args.data_dir, normalizer, geometry, coordinates, args.max_cases)
    run_deeponet(
        args, MODEL_ID, search_space(pod.n_components), normalizer, geometry,
        coordinates, pod=pod,
    )


if __name__ == "__main__":
    main()
