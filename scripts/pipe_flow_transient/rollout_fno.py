"""Train a data-only neuraloperator FNO for transient pipe-flow rollouts.

The FNO maps one normalized radial velocity profile to the next, using the
same branch features as the rollout DeepONets: the current profile, the
normalized case constants, and the Fourier time and step. Scalars are
broadcast across the radial grid as extra channels, with ``r/R`` supplied as a
coordinate channel.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from neuralop.models import FNO
from ray import tune
from torch import nn

from chem_operator.normalization import normalizer_from_state_dict
from scripts.pipe_flow_transient.common import (
    BRANCH_CONSTANT_NAMES,
    RolloutTrainerBase,
    prepare_rollout_run,
    rollout_search_space,
    run_rollout_experiment,
)

MODEL_ID = "rollout_fno"
# Current profile, branch constants, Fourier time and step, and r/R.
IN_CHANNELS = 1 + len(BRANCH_CONSTANT_NAMES) + 2 + 1


class RolloutFNO(nn.Module):
    """One-dimensional FNO over the radial grid for one rollout step."""

    def __init__(self, config: Mapping[str, Any], n_radial: int):
        super().__init__()
        self.n_radial = n_radial
        self.fno = FNO(
            n_modes=(int(config["modes"]),),
            in_channels=IN_CHANNELS,
            out_channels=1,
            hidden_channels=int(config["hidden_channels"]),
            n_layers=int(config["n_layers"]),
            positional_embedding=None,
        )

    def forward(self, branch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if branch.ndim != 2 or x.ndim != 2 or branch.shape[0] != x.shape[0]:
            raise ValueError("Expected branch [batch, features] and x [batch, radius].")
        if x.shape[1] != self.n_radial or branch.shape[1] != self.n_radial + IN_CHANNELS - 2:
            raise ValueError("Rollout inputs differ from the model resolution.")
        profile = branch[:, : self.n_radial].unsqueeze(1)
        scalars = branch[:, self.n_radial:].unsqueeze(-1).expand(-1, -1, self.n_radial)
        coordinate = (2.0 * x - 1.0).unsqueeze(1)
        channels = torch.cat((profile, scalars, coordinate), dim=1)
        return self.fno(channels).squeeze(1)


class RolloutFNOTrainer(RolloutTrainerBase):
    def __init__(self, config, normalizer, n_radial: int):
        super().__init__(
            config, normalizer, n_radial, model_id=MODEL_ID,
            model_builder=RolloutFNO,
        )

    @classmethod
    def from_checkpoint(cls, path: str | Path, context):
        """Restore a rollout FNO from its checkpoint and run context."""
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
    for key in ("width", "branch_hidden_layers", "activation"):
        space.pop(key)
    space.update({
        "modes": tune.choice([8, 16, 32]),
        "hidden_channels": tune.choice([16, 32]),
        "n_layers": tune.choice([3, 4, 5]),
    })
    return space


def main() -> None:
    args, normalizer, n_radial = prepare_rollout_run()
    run_rollout_experiment(
        args, normalizer, n_radial, MODEL_ID,
        lambda config, context: RolloutFNOTrainer(config, normalizer, n_radial),
        search_space(),
    )


if __name__ == "__main__":
    main()
