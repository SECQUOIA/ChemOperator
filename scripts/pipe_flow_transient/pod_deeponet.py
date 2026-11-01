"""Train a data-only DeepXDE POD-DeepONet for transient pipe-flow rollouts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from torch.utils.data import Subset

from chem_operator._models.deeponet import _PODOnlyDeepONet
from chem_operator.models import PODTransform, fit_incremental_pod_dataset
from chem_operator.normalization import normalizer_from_state_dict
from scripts.pipe_flow_transient.common import (
    ROLLOUT_CONSTANT_NAMES,
    RolloutDeepXDEGrid,
    RolloutDeepXDETrainerBase,
    make_rollout_adapter,
    prepare_rollout_run,
    rollout_architecture,
    rollout_search_space,
    run_rollout_experiment,
)

MODEL_ID = "pod_deeponet"
POD_VARIANCE_THRESHOLD = 0.999


def pod_state(pod: PODTransform) -> dict[str, Any]:
    """Keep the fitted basis and mean in a weights-only checkpoint."""
    return {
        "mean": torch.as_tensor(np.array(pod.mean, copy=True)),
        "basis": torch.as_tensor(np.array(pod.basis, copy=True)),
        "explained_variance_ratio": torch.as_tensor(
            np.array(pod.explained_variance_ratio, copy=True)
        ),
        "cumulative_explained_variance": pod.cumulative_explained_variance,
        "output_shape": list(pod.output_shape),
    }


def pod_from_state(state: Mapping[str, Any]) -> PODTransform:
    return PODTransform(
        mean=state["mean"].cpu().numpy(),
        basis=state["basis"].cpu().numpy(),
        explained_variance_ratio=state["explained_variance_ratio"].cpu().numpy(),
        cumulative_explained_variance=float(state["cumulative_explained_variance"]),
        output_shape=tuple(int(value) for value in state["output_shape"]),
    )


class PODRolloutDeepXDE(RolloutDeepXDEGrid):
    """Cartesian POD-DeepONet on normalized radial profiles."""

    def __init__(self, config: Mapping[str, Any], n_radial: int, pod: PODTransform):
        if pod.output_shape != (n_radial,):
            raise ValueError("POD basis must represent one radial profile.")
        branch_width = n_radial + len(ROLLOUT_CONSTANT_NAMES) + 2
        super().__init__(
            _PODOnlyDeepONet(branch_width, pod, rollout_architecture(config)),
            n_radial,
        )


class PODRolloutDeepXDETrainer(RolloutDeepXDETrainerBase):
    def __init__(self, config, normalizer, n_radial: int, pod: PODTransform):
        self.pod = pod
        super().__init__(
            config, normalizer, n_radial, model_id=MODEL_ID,
            model_builder=lambda cfg, radial: PODRolloutDeepXDE(cfg, radial, self.pod),
            pod_metadata=pod_state(pod),
        )

    def _restore_model_metadata(self, metadata: Mapping[str, Any]) -> None:
        self.pod = pod_from_state(metadata["pod"])

    @classmethod
    def from_checkpoint(cls, path: str | Path, context):
        """Restore a POD rollout from its checkpoint and run context."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        trainer = cls(
            payload["config"], normalizer_from_state_dict(metadata["normalizer"]),
            int(metadata["n_radial"]), pod_from_state(metadata["pod"]),
        )
        trainer.load_checkpoint(path, context)
        return trainer


def fit_rollout_pod(data_dir: Path, normalizer, n_radial: int,
                    max_cases: int | None = None) -> PODTransform:
    raw, adapter = make_rollout_adapter(data_dir, "train", normalizer, n_radial)
    try:
        selected = (
            adapter if max_cases is None
            else Subset(adapter, range(min(len(adapter), max_cases)))
        )
        return fit_incremental_pod_dataset(
            selected, variance_threshold=POD_VARIANCE_THRESHOLD
        )
    finally:
        raw.close()


def search_space(pod: PODTransform) -> dict[str, Any]:
    space = rollout_search_space()
    space.update({"latent_width": pod.n_components, "trunk_hidden_layers": 0})
    return space


def main() -> None:
    args, normalizer, n_radial = prepare_rollout_run()
    pod = fit_rollout_pod(args.data_dir, normalizer, n_radial, args.max_cases)
    run_rollout_experiment(
        args, normalizer, n_radial, MODEL_ID,
        lambda config, context: PODRolloutDeepXDETrainer(config, normalizer, n_radial, pod),
        search_space(pod),
    )


if __name__ == "__main__":
    main()
