"""Train a data-only DeepXDE DeepONet for transient pipe-flow rollouts."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from ray import tune

from chem_operator._models.deeponet import _network
from chem_operator.normalization import normalizer_from_state_dict
from scripts.pipe_flow_transient.common import (
    ROLLOUT_CONSTANT_NAMES,
    RolloutDeepXDEGrid,
    RolloutDeepXDETrainerBase,
    prepare_rollout_run,
    rollout_architecture,
    rollout_search_space,
    run_rollout_experiment,
)

MODEL_ID = "deeponet"


class RolloutDeepXDE(RolloutDeepXDEGrid):
    """Direct Cartesian DeepONet on normalized radial profiles."""

    def __init__(self, config: Mapping[str, Any], n_radial: int):
        branch_width = n_radial + len(ROLLOUT_CONSTANT_NAMES) + 2
        super().__init__(_network(branch_width, 1, 1, rollout_architecture(config)), n_radial)


class RolloutDeepXDETrainer(RolloutDeepXDETrainerBase):
    def __init__(self, config, normalizer, n_radial: int):
        super().__init__(
            config, normalizer, n_radial, model_id=MODEL_ID,
            model_builder=RolloutDeepXDE,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, context):
        """Restore a direct rollout from its checkpoint and run context."""
        payload = torch.load(path, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        trainer = cls(
            payload["config"], normalizer_from_state_dict(metadata["normalizer"]),
            int(metadata["n_radial"]),
        )
        trainer.load_checkpoint(path, context)
        return trainer


def search_space() -> dict[str, Any]:
    space = rollout_search_space()
    space.update({
        "latent_width": tune.choice([16, 32]),
        "trunk_hidden_layers": tune.choice([2, 3]),
    })
    return space


def main() -> None:
    args, normalizer, n_radial = prepare_rollout_run()
    run_rollout_experiment(
        args, normalizer, n_radial, MODEL_ID,
        lambda config, context: RolloutDeepXDETrainer(config, normalizer, n_radial),
        search_space(),
    )


if __name__ == "__main__":
    main()
