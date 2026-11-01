"""Data-only DeepXDE rollout models and shared rollout preparation."""

from __future__ import annotations

import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.experiments import ExperimentRunner, RunContext
from chem_operator.plotting import plot_operator_fields, plot_operator_runs
from chem_operator.reactors.pipe_flow_transient.dataset_generator import (
    TransientHagenPoiseuillePipeFlowSim,
)
from chem_operator.sampling import Constant, Uniform
from scripts.pipe_flow_transient.common import (
    experiment_spec,
    evaluate_rollout_run,
    fit_rollout_normalizer,
    make_rollout_adapter,
)
from scripts.pipe_flow_transient.deeponet import (
    RolloutDeepXDE,
    RolloutDeepXDETrainer,
)
from scripts.pipe_flow_transient.pod_deeponet import (
    PODRolloutDeepXDETrainer,
    fit_rollout_pod,
)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("transient_deepxde")
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
    generator = SimulationDatasetGenerator(simulator, directory, seed=21)
    generator.save_splits(generator.generate_splits(n_cases=10))
    return directory


def config(pod=False):
    return {
        "width": 16,
        "latent_width": 8,
        "branch_hidden_layers": 2,
        "trunk_hidden_layers": 0 if pod else 2,
        "activation": "tanh",
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "batch_size": 2,
        "epochs": 1,
    }


def test_shared_rollout_inputs_and_cartesian_grid(data_dir):
    normalizer = fit_rollout_normalizer(data_dir)
    raw, data = make_rollout_adapter(data_dir, "train", normalizer, 8)
    try:
        sample = data[0]
        assert raw.task == "rollout"
        assert sample["branch"].shape == (15,)
        assert sample["x"].shape == sample["y"].shape == (8,)
        torch.testing.assert_close(sample["target"], sample["y"])
        batch = next(iter(DataLoader(data, batch_size=2)))
        model = RolloutDeepXDE(config(), 8)
        assert model(batch["branch"], batch["x"]).shape == (2, 8)
        shifted = batch["x"].clone()
        shifted[1, 1] += 0.01
        with pytest.raises(ValueError, match="shared radial grid"):
            model(batch["branch"], shifted)
    finally:
        raw.close()


@pytest.mark.parametrize("kind", ("direct", "pod"))
def test_training_checkpoint_and_recursive_evaluation(data_dir, tmp_path, kind):
    normalizer = fit_rollout_normalizer(data_dir)
    pod = fit_rollout_pod(data_dir, normalizer, 8) if kind == "pod" else None
    if pod is not None:
        assert pod.output_shape == (8,)
        assert pod.basis.shape[0] == 8
    train_raw, train = make_rollout_adapter(data_dir, "train", normalizer, 8)
    valid_raw, valid = make_rollout_adapter(data_dir, "valid", normalizer, 8)
    test_raw, test = make_rollout_adapter(data_dir, "test", normalizer, 8)
    model_id = "pod_deeponet" if kind == "pod" else "deeponet"
    context = RunContext.create(
        tmp_path, problem="pipe_flow_transient", model=model_id,
        run_id="smoke", seed=42,
    )
    try:
        if pod is not None:
            training_mean = torch.stack([train[index]["y"] for index in range(len(train))]).mean(0)
            np.testing.assert_allclose(pod.mean, training_mean.numpy(), rtol=1e-5, atol=1e-6)
        trainer = (
            PODRolloutDeepXDETrainer(config(True), normalizer, 8, pod)
            if pod is not None else RolloutDeepXDETrainer(config(False), normalizer, 8)
        )
        outcome = trainer.fit(train, valid, context)
        assert outcome.metadata["checkpoint_reload_verified"]
        payload = torch.load(outcome.checkpoint, weights_only=True)
        assert payload["metadata"]["normalizer"]
        assert (payload["metadata"]["pod"] is not None) == (pod is not None)

        restored = type(trainer).from_checkpoint(outcome.checkpoint, context)
        torch.testing.assert_close(restored.predict(valid, context), trainer.predict(valid, context))
        assert (getattr(restored, "pod", None) is not None) == (pod is not None)

        evaluation = evaluate_rollout_run(
            trainer, test, context, normalizer, data_dir, 8, cases=1, max_cases=1
        )
        assert all(math.isfinite(value) for value in evaluation.metrics.values())
        assert "one_step_relative_l2" in evaluation.metrics
        assert "relative_l2" in evaluation.metrics
        assert evaluation.reconstructions["reference"].shape == (1, 1, 8, 8)
        assert evaluation.reconstructions["prediction"].shape == (1, 1, 8, 8)
        arrays = evaluation.reconstructions
        plotted = plot_operator_fields(
            tmp_path / f"{kind}.png", arrays["reference"][0],
            arrays["prediction"][0], arrays["labels"],
            [arrays["t"][0], arrays["r"][0]], arrays["coordinate_names"],
        )
        assert plotted.is_file()
    finally:
        train_raw.close()
        valid_raw.close()
        test_raw.close()


def test_canonical_runs_compare_in_operator_plot(data_dir, tmp_path):
    normalizer = fit_rollout_normalizer(data_dir)
    pod = fit_rollout_pod(data_dir, normalizer, 8)
    run_paths = []
    args = SimpleNamespace(
        data_dir=data_dir, plot_cases=1, max_cases=1, samples=1, tune_epochs=1
    )
    for model_id, basis in (("deeponet", None), ("pod_deeponet", pod)):
        train_raw, train = make_rollout_adapter(data_dir, "train", normalizer, 8)
        valid_raw, valid = make_rollout_adapter(data_dir, "valid", normalizer, 8)
        test_raw, test = make_rollout_adapter(data_dir, "test", normalizer, 8)
        try:
            context = RunContext.create(
                tmp_path, problem="pipe_flow_transient", model=model_id,
                run_id="comparison", seed=42,
            )
            spec = replace(
                experiment_spec(args, model_id),
                benchmark_protocol_id="rollout-one-step-v1",
            )
            trainer = (
                PODRolloutDeepXDETrainer(config(True), normalizer, 8, basis)
                if basis is not None else RolloutDeepXDETrainer(config(False), normalizer, 8)
            )
            ExperimentRunner(context, spec).run(
                trainer, train, valid, test, config=trainer.config,
                evaluator=lambda fitted, data, run_context: evaluate_rollout_run(
                    fitted, data, run_context, normalizer, data_dir, 8,
                    cases=1, max_cases=1,
                ),
            )
            run_paths.append(context.paths.run_dir)
        finally:
            train_raw.close()
            valid_raw.close()
            test_raw.close()
    saved = plot_operator_runs(run_paths, tmp_path / "comparison_plots", cases=1)
    assert any(path.name == "comparison.png" for path in saved)
    assert all(path.is_file() for path in saved)
