"""Neuraloperator FNO for one-step transient pipe-flow rollouts."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.experiments import RunContext
from chem_operator.reactors.pipe_flow_transient.dataset_generator import (
    TransientHagenPoiseuillePipeFlowSim,
)
from chem_operator.sampling import Constant, Uniform
from scripts.pipe_flow_transient.common import (
    evaluate_rollout_run,
    fit_rollout_normalizer,
    make_rollout_adapter,
)
from scripts.pipe_flow_transient.rollout_fno import (
    RolloutFNO,
    RolloutFNOTrainer,
    search_space,
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("transient_rollout_fno")
    simulator = TransientHagenPoiseuillePipeFlowSim(parameter_space={
        "radius": Uniform(0.8e-3, 1.2e-3),
        "length": Constant(1.0),
        "dynamic_viscosity": Uniform(0.9e-3, 1.1e-3),
        "pressure_drop": Uniform(40.0, 80.0),
        "density": Constant(1000.0),
        "n_time_points": Constant(8),
        "n_radial_points": Constant(8),
        "max_fourier_number": Constant(2.0),
    })
    generator = SimulationDatasetGenerator(simulator, directory, seed=31)
    generator.save_splits(generator.generate_splits(n_cases=10))
    return directory


def config():
    return {
        "modes": 4,
        "hidden_channels": 8,
        "n_layers": 2,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
        "epochs": 1,
    }


def test_search_space_replaces_deeponet_architecture_keys():
    space = search_space()
    assert {"modes", "hidden_channels", "n_layers"} <= set(space)
    assert not {"width", "branch_hidden_layers", "activation"} & set(space)


def test_forward_maps_branch_features_to_next_profile(data_dir):
    normalizer = fit_rollout_normalizer(data_dir)
    raw, data = make_rollout_adapter(data_dir, "train", normalizer, 8)
    try:
        sample = data[0]
        assert sample["branch"].shape == (14,)
        assert sample["x"].shape == sample["y"].shape == (8,)
        batch = next(iter(DataLoader(data, batch_size=2)))
        model = RolloutFNO(config(), 8)
        prediction = model(batch["branch"], batch["x"])
        assert prediction.shape == (2, 8)
        prediction.square().mean().backward()
        with pytest.raises(ValueError, match="resolution"):
            model(batch["branch"][:, :-1], batch["x"])
    finally:
        raw.close()


def test_training_checkpoint_and_recursive_evaluation(data_dir, tmp_path):
    normalizer = fit_rollout_normalizer(data_dir)
    train_raw, train = make_rollout_adapter(data_dir, "train", normalizer, 8)
    valid_raw, valid = make_rollout_adapter(data_dir, "valid", normalizer, 8)
    test_raw, test = make_rollout_adapter(data_dir, "test", normalizer, 8)
    context = RunContext.create(
        tmp_path, problem="pipe_flow_transient", model="rollout_fno",
        run_id="smoke", seed=42,
    )
    try:
        trainer = RolloutFNOTrainer(config(), normalizer, 8)
        outcome = trainer.fit(train, valid, context)
        assert outcome.metadata["checkpoint_reload_verified"]
        payload = torch.load(outcome.checkpoint, weights_only=True)
        assert payload["metadata"]["model_id"] == "rollout_fno"

        restored = RolloutFNOTrainer.from_checkpoint(outcome.checkpoint, context)
        torch.testing.assert_close(restored.predict(valid, context), trainer.predict(valid, context))

        evaluation = evaluate_rollout_run(
            trainer, test, context, normalizer, data_dir, 8, cases=1, max_cases=1
        )
        assert all(math.isfinite(value) for value in evaluation.metrics.values())
        assert "one_step_relative_l2" in evaluation.metrics
        assert "relative_l2" in evaluation.metrics
        assert evaluation.reconstructions["prediction"].shape == (1, 1, 8, 8)
    finally:
        train_raw.close()
        valid_raw.close()
        test_raw.close()
