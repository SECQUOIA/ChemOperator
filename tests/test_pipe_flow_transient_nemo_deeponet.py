"""Focused rollout and PhysicsNeMo DeepONet checks."""

from __future__ import annotations

import math
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.experiments import RunContext
from chem_operator.normalization import normalizer_from_state_dict
from chem_operator.plotting import plot_operator_fields
from chem_operator.reactors.pipe_flow_transient.dataset_generator import TransientHagenPoiseuillePipeFlowSim
from chem_operator.sampling import Constant, Uniform

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "pipe_flow_transient" / "nemo_deeponet.py"
SCRIPT_SPEC = importlib.util.spec_from_file_location("pipe_flow_transient_nemo_deeponet", SCRIPT_PATH)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
script = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(script)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("pipe_flow_deeponet")
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


def config():
    return {
        "branch_width": 16, "trunk_width": 16, "latent_width": 8,
        "depth": 2, "activation": "tanh", "learning_rate": 1e-3,
        "weight_decay": 0.0, "batch_size": 2, "epochs": 1,
        "physics_weight": 1e-4, "constraint_weight": 1e-4,
    }


def test_rollout_windows_and_physics(data_dir):
    normalizer = script.fit_normalizer(data_dir)
    raw, data = script.make_adapter(data_dir, "train", normalizer, 8)
    try:
        assert raw.task == "rollout"
        sample = data[0]
        assert sample["branch"].shape == (15,)
        assert sample["x"].shape == (8,)
        assert sample["y"].shape == (8,)
        assert sample["physicsnemo_pde"] == "TransientHagenPoiseuille"
        torch.testing.assert_close(normalizer.denormalize(sample["y"], "velocity"), raw[0]["output_fields"]["velocity"][0])
        assert sample["t_next"] > sample["t_now"]

        batch = next(iter(DataLoader(data, batch_size=2)))
        prediction = batch["y"].clone().requires_grad_(True)
        informer = script.make_physics_informer("TransientHagenPoiseuille", 1 / 7, torch.device("cpu"))
        physics, constraint = script.physics_losses(
            prediction, batch, normalizer, informer,
            pde_name="TransientHagenPoiseuille", radial_spacing=1 / 7,
        )
        assert math.isfinite(float(physics.detach()))
        assert math.isfinite(float(constraint.detach()))
        assert float(constraint.detach()) < 1e-5
        (physics + constraint).backward()
        assert torch.isfinite(prediction.grad).all()
    finally:
        raw.close()


def test_data_mode_skips_training_physics(data_dir, monkeypatch):
    normalizer = script.fit_normalizer(data_dir)
    context = SimpleNamespace(device=torch.device("cpu"))
    trainer = script.make_trainer(config(), context, normalizer, 8,
                                  pde_name="TransientHagenPoiseuille",
                                  radial_spacing=1 / 7, variant="data")
    assert all(trainer.config[key] == 0.0 for key in script.PHYSICS_WEIGHT_KEYS)
    monkeypatch.setattr(script, "physics_losses", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("physics called")))
    model = script.model_from_config(config(), 8, torch.device("cpu"))
    model.train()
    prediction = trainer.forward_adapter(model, (torch.zeros(1, 15), torch.linspace(0, 1, 8).unsqueeze(0)), {})
    assert prediction.shape == (1, 8)


def test_training_checkpoint_and_recursive_evaluation(data_dir, tmp_path):
    normalizer = script.fit_normalizer(data_dir)
    train_raw, train = script.make_adapter(data_dir, "train", normalizer, 8)
    valid_raw, valid = script.make_adapter(data_dir, "valid", normalizer, 8)
    context = RunContext.create(tmp_path, problem="pipe_flow_transient", model="nemo_deeponet_physics", run_id="smoke", seed=42)
    try:
        trainer = script.make_trainer(config(), context, normalizer, 8,
                                      pde_name="TransientHagenPoiseuille", radial_spacing=1 / 7)
        outcome = trainer.fit(train, valid, context)
        assert outcome.metadata["checkpoint_reload_verified"]
        evaluation = script.evaluate_run(trainer, valid, context, normalizer, data_dir, 8, cases=1, max_cases=1)
        assert all(math.isfinite(value) for value in evaluation.metrics.values())
        assert evaluation.reconstructions["reference"].shape == (1, 1, 8, 8)
        assert evaluation.reconstructions["prediction"].shape == (1, 1, 8, 8)
        arrays = evaluation.reconstructions
        plot_path = plot_operator_fields(
            tmp_path / "rollout.png", arrays["reference"][0], arrays["prediction"][0],
            arrays["labels"], [arrays["t"][0], arrays["r"][0]], arrays["coordinate_names"],
        )
        assert plot_path.is_file()
        restored_normalizer = normalizer_from_state_dict(torch.load(outcome.checkpoint, weights_only=True)["metadata"]["normalizer"])
        restored = script.make_trainer(config(), context, restored_normalizer, 8,
                                       pde_name="TransientHagenPoiseuille", radial_spacing=1 / 7)
        restored.load_checkpoint(outcome.checkpoint, context)
        torch.testing.assert_close(restored.predict(valid, context), trainer.predict(valid, context))
    finally:
        train_raw.close()
        valid_raw.close()


def test_both_variants_have_independent_model_ids(monkeypatch, tmp_path):
    args = SimpleNamespace(variant="both", data_dir=tmp_path, plot_cases=1, max_cases=None)
    monkeypatch.setattr(script, "parse_cli_args", lambda: args)
    monkeypatch.setattr(script, "fit_normalizer", lambda _: object())

    class Raw:
        def __getitem__(self, index):
            return {"input_fields": {"velocity": torch.zeros(1, 8)},
                    "input_coordinates": {"r": torch.linspace(0, 1, 8)},
                    "constant_inputs": {"radius": torch.tensor(1.0)}}

        def close(self):
            pass

    monkeypatch.setattr(script, "raw_dataset", lambda *args, **kwargs: Raw())
    monkeypatch.setattr(script, "dataset_pde_name", lambda _: "TransientHagenPoiseuille")
    monkeypatch.setattr(script, "experiment_spec", lambda _args, model_id: SimpleNamespace(model_id=model_id, benchmark_protocol_id="other"))
    monkeypatch.setattr(script, "replace", lambda spec, **kwargs: spec)
    calls = []
    monkeypatch.setattr(script, "run_operator", lambda _args, _paths, spec, _data, _trainer, space, _eval: calls.append((spec.model_id, space)))
    script.main()
    assert [item[0] for item in calls] == ["nemo_deeponet_data", "nemo_deeponet_physics"]
    assert calls[0][1]["physics_weight"] == 0.0
    assert calls[1][1]["physics_weight"] != 0.0
