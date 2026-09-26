"""Model definition, scientific losses, and canonical experiment entry point."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from typing import Any, Mapping
from neuralop.losses import LpLoss
from neuralop.models import FNO
from ray import tune
import torch
from chem_operator.experiments import FNOTrainer, LossTerm
from chem_operator.experiments import run_operator, evaluate_fields
from scripts.pipe_flow_transient.common import (
    PATHS,
    FIELD_NAMES,
    MODEL_CONSTANT_NAMES,
    PHYSICS_CONSTANT_NAMES,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    FILE_STEM,
    METRIC,
    SEED,
    PDE_REGISTRY,
    MAX_TRAIN_TRAJECTORIES,
    CONSTANT_NAMES,
    PhysicsFNOAdapter,
    raw_dataset,
    dataset_pde_name,
    make_adapter,
    fit_normalizer,
    normalizer_state,
    normalizer_from_state,
    adapter_radial_spacing,
    training_pde_name,
    PROBLEM_ID,
    experiment_spec,
    parse_args,
)

def model_from_config(
    config: Mapping[str, Any],
    device: torch.device,
) -> FNO:
    """Construct the configured NeuralOperator FNO."""
    modes = int(config["modes"])
    return FNO(
        n_modes=(modes, modes),
        in_channels=len(CONSTANT_NAMES),
        out_channels=len(FIELD_NAMES),
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
    ).to(device)
def search_space():
    return {'modes': tune.choice([8, 12]), 'hidden_channels': tune.choice([8, 16]), 'n_layers': 3, 'learning_rate': tune.loguniform(0.0001, 0.003), 'weight_decay': tune.loguniform(1e-08, 0.0001), 'batch_size': tune.choice([2, 4, 8, 16, 32])}

MODEL_ID = "transient_fno"

def make_trainer(config, context, normalizer):
    loss = LpLoss(d=2, p=2, reduction="mean")
    return FNOTrainer(lambda cfg: model_from_config(cfg, context.device), config,
        loss_terms=[LossTerm("data_loss", lambda p,t,b: loss(p,t))], selection_metric="data_loss",
        metric_adapter=lambda p,t,b: (normalizer.denormalize(p, "velocity"), normalizer.denormalize(t, "velocity")),
        checkpoint_metadata={"normalizer": normalizer.state_dict(), "input_channels": list(CONSTANT_NAMES),
                             "output_channels": list(FIELD_NAMES)})

def main():
    args = parse_args(epochs=10, tune_epochs=6, samples=12)
    normalizer = fit_normalizer(args.data_dir)
    run_operator(args, PATHS, experiment_spec(args, MODEL_ID),
        lambda split: make_adapter(args.data_dir, split, normalizer, args.max_cases),
        lambda config, context: make_trainer(config, context, normalizer), search_space(),
        lambda trainer, data, context: evaluate_fields(trainer, data, context,
            labels=FIELD_NAMES, coordinate_names=("t", "r"), cases=args.plot_cases), selection_metric="data_loss")

if __name__ == "__main__":
    main()
