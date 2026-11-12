"""Fixed-grid Q2D DeepONet data, training, and artifact contracts."""

from __future__ import annotations

from argparse import Namespace
from dataclasses import replace
import json
import subprocess
import sys

import numpy as np
import pytest
import torch

from chem_operator.datasets import SimulationDatasetGenerator, SimulationRecord
from chem_operator.experiments import (
    DeepONetTrainer, ExperimentRunner, RunContext, deeponet_training_config,
    evaluate_fields, load_run, validate_model_comparison,
)
from chem_operator.reactors.q2d.dataset_generator import CMRSim
from chem_operator.normalization import normalizer_from_state_dict
from scripts.q2d import common, deeponet, fno, plot, pod_deeponet


def record(offset=0.0):
    z = np.linspace(0.0, 1.0, 6)
    r = np.linspace(0.0, 0.5, 4)
    base = z[:, None] + r[None, :]
    return SimulationRecord(
        coordinates={"z": z, "r": r},
        fields={
            # Deliberately store species in a different order from the outputs.
            "X": np.stack((0.4 - 0.03 * base + offset, 0.2 + 0.02 * base + offset), axis=-1),
            "theta": (0.1 + 0.01 * base + offset**2)[..., None],
        },
        constants={"SCCM": 500.0 + 100 * offset},
        metadata={
            "params": {"T0": 1100.0 + 100 * offset, "sccm": 500.0 + 100 * offset},
            "field_species": {"X": ["H2", "CH4"], "theta": ["C(s)"]},
            "wall_time": 0.01,
            "input_values": {"LENGTH_LUMEN": 1.0, "CHANNELRAD_LUMEN": 0.5},
        },
    )


@pytest.fixture
def data_dir(tmp_path):
    directory = tmp_path / "data"
    generator = SimulationDatasetGenerator(CMRSim(), directory)
    for split, offsets in (("train", (0.0, 0.1, 0.2)),
                           ("valid", (0.3, 0.4)), ("test", (0.5, 0.6))):
        generator.save_split(split, [record(value) for value in offsets])
    return directory


def config():
    return {
        "loss": "relative_l2", "width": 8, "latent_width": 4,
        "branch_hidden_layers": 1, "trunk_hidden_layers": 1,
        "activation": "tanh", "learning_rate": 1e-3, "weight_decay": 0.0,
        "batch_size": 2, "epochs": 1, "seed": common.SEED,
    }


def arguments(data_dir, tmp_path, **overrides):
    values = dict(
        data_dir=data_dir, runs_root=tmp_path / "runs", run_id="smoke", device="cpu",
        tune=False, train=True, train_config="best", plot_cases=1, epochs=1,
        tune_epochs=1, samples=1, max_cases=None,
    )
    values.update(overrides)
    return Namespace(**values)


def test_adapter_controls_species_grid_and_round_trip(data_dir):
    normalizer, geometry, coordinates = common.prepare_deeponet_data(data_dir)
    with common.deeponet_data(
        data_dir, ("train", "test"), normalizer, geometry, coordinates,
    ) as (train, test):
        sample = train[0]
        assert sample["branch"].shape == (2,)
        assert sample["trunk"].shape == (24, 2)
        assert sample["target"].shape == (24, 3)
        assert sample["labels"] == ("X_CH4", "X_H2", "theta_C(s)")
        torch.testing.assert_close(sample["branch"], torch.stack((
            normalizer.normalize(torch.tensor(1100.0), "T0"),
            normalizer.normalize(torch.tensor(500.0), "SCCM"),
        )))
        expected_grid = torch.stack(torch.meshgrid(*coordinates, indexing="ij"), -1)
        torch.testing.assert_close(sample["trunk"], expected_grid.reshape(-1, 2))
        physical = train.dataset.denormalize_output(train.field_output(sample["target"]))
        base = coordinates[0][:, None] + coordinates[1][None, :]
        torch.testing.assert_close(physical[0], 0.2 + 0.02 * base)
        torch.testing.assert_close(physical[1], 0.4 - 0.03 * base)
        torch.testing.assert_close(physical[2], 0.1 + 0.01 * base)
        assert physical.shape == (3, 6, 4)
        assert sample["trunk"][0, 0] == coordinates[0][0]
        # Validation/test distributions must not enter training statistics.
        torch.testing.assert_close(normalizer.means["T0"], torch.tensor(1110.0))
        test_physical = test.dataset.denormalize_output(test.field_output(test[0]["target"]))
        torch.testing.assert_close(test_physical[0], 0.7 + 0.02 * base)
    assert train.dataset.dataset._file_handle is None
    assert test.dataset.dataset._file_handle is None


def test_pod_uses_only_selected_training_targets(data_dir):
    normalizer, geometry, coordinates = common.prepare_deeponet_data(data_dir)
    pod = pod_deeponet.prepare_pod(data_dir, normalizer, geometry, coordinates, max_cases=2)
    with common.deeponet_data(
        data_dir, ("train",), normalizer, geometry, coordinates, max_cases=2,
    ) as (train,):
        targets = torch.stack([train[i]["target"] for i in range(len(train))])
    assert pod.output_shape == (24, 3)
    assert pod.basis.shape[0] == 72
    assert pod.cumulative_explained_variance >= pod_deeponet.POD_VARIANCE_THRESHOLD
    np.testing.assert_allclose(pod.mean, targets.mean(0).numpy().reshape(-1), atol=1e-6)
    torch.testing.assert_close(pod.decode_tensor(pod.encode_tensor(targets)), targets,
                               atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match="at least two training cases"):
        pod_deeponet.prepare_pod(data_dir, normalizer, geometry, coordinates, max_cases=1)


def test_evaluation_keeps_reference_on_cpu_with_non_cpu_default(data_dir, tmp_path):
    normalizer, geometry, coordinates = common.prepare_deeponet_data(data_dir)
    context = RunContext.create(
        tmp_path, problem=common.PROBLEM_ID, model="deeponet", run_id="evaluation",
        seed=common.SEED,
    )
    with common.deeponet_data(
        data_dir, ("test",), normalizer, geometry, coordinates,
    ) as (test,):
        predictions = torch.stack([test[i]["target"] for i in range(len(test))])

        class SavedPredictions:
            def predict(self, data, context):
                return predictions

        # Simulate DeepXDE changing the default device without requiring CUDA.
        # The trainer returns CPU predictions; freshly read targets must match.
        with torch.device("meta"):
            outcome = common.evaluate_q2d_deeponet(SavedPredictions(), test, context, cases=1)
            assert torch.empty(0).device.type == "meta"
    assert outcome.metrics["relative_l2"] == 0.0
    assert outcome.metrics["normalized_relative_l2"] == 0.0
    np.testing.assert_array_equal(
        outcome.reconstructions["prediction"], outcome.reconstructions["reference"],
    )


@pytest.mark.parametrize("mismatch", ("grid", "shape", "geometry"))
def test_rejects_incompatible_selected_cases_and_closes_files(data_dir, monkeypatch, mismatch):
    normalizer, geometry, coordinates = common.prepare_deeponet_data(data_dir)
    invalid = record(0.6)
    if mismatch == "grid":
        invalid = replace(invalid, coordinates={**invalid.coordinates, "z": np.linspace(0.0, 0.9, 6)})
    elif mismatch == "shape":
        invalid = replace(
            invalid,
            coordinates={**invalid.coordinates, "z": np.linspace(0.0, 1.0, 5)},
            fields={name: value[:5] for name, value in invalid.fields.items()},
        )
    else:
        invalid = replace(invalid, metadata={**invalid.metadata,
                          "input_values": {"LENGTH_LUMEN": 2.0, "CHANNELRAD_LUMEN": 0.5}})
    SimulationDatasetGenerator(CMRSim(), data_dir).save_split(
        "valid", [record(0.3), invalid], overwrite=True,
    )
    opened = []
    original = common.raw_dataset

    def tracked(path):
        raw = original(path)
        opened.append(raw)
        return raw

    monkeypatch.setattr(common, "raw_dataset", tracked)
    with pytest.raises(ValueError, match="training (coordinate grid|geometry)"):
        with common.deeponet_data(data_dir, ("train", "valid"), normalizer, geometry, coordinates):
            pytest.fail("Incompatible data must fail before entering the context")
    assert all(raw._file_handle is None for raw in opened)
    # A deliberately unselected invalid case is allowed.
    with common.deeponet_data(data_dir, ("valid",), normalizer, geometry, coordinates, max_cases=1):
        pass


@pytest.mark.parametrize("module", (deeponet, pod_deeponet), ids=("direct", "pod"))
def test_cli_training_checkpoint_reload_and_artifacts(data_dir, tmp_path, monkeypatch, module):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config()))
    monkeypatch.setattr(sys, "argv", [
        module.__file__, "--no-tune", "--train-config", str(config_path),
        "--epochs", "1", "--max-cases", "2", "--plot-cases", "1", "--device", "cpu",
        "--data-dir", str(data_dir), "--runs-root", str(tmp_path / "runs"), "--run-id", "smoke",
    ])
    module.main()
    run_dir = tmp_path / "runs" / common.PROBLEM_ID / module.MODEL_ID / "smoke"
    run = load_run(run_dir)
    assert run.manifest["status"] == "completed"
    assert run.manifest["training_metadata"]["checkpoint_reload_verified"]
    assert any(row["metric"] == "normalized_relative_l2" for row in run.metrics)
    assert any(row["metric"] == "X_CH4/relative_l2" for row in run.metrics)
    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["training_shape"] == [6, 4]
    assert checkpoint["metadata"]["input_channels"] == ["T0", "SCCM"]
    assert (checkpoint["pod"] is not None) == (module is pod_deeponet)
    if module is pod_deeponet:
        assert checkpoint["pod"]["output_shape"] == [24, 3]
        assert checkpoint["config"]["latent_width"] == checkpoint["pod"]["basis"].shape[1]
        assert checkpoint["config"]["trunk_hidden_layers"] == 0
    context = RunContext.create(tmp_path / "runs", problem=common.PROBLEM_ID,
                                model=module.MODEL_ID, run_id="smoke", seed=common.SEED)
    restored = DeepONetTrainer(deeponet_training_config(config()))
    restored.load_checkpoint(run_dir / "checkpoint.pt", context)
    metadata = restored.checkpoint_metadata
    normalizer = normalizer_from_state_dict(metadata["normalizer"])
    coordinates = tuple(metadata["training_coordinates"][axis] for axis in ("z", "r"))
    with common.deeponet_data(
        data_dir, ("test",), normalizer, metadata["geometry"], coordinates, max_cases=2,
    ) as (test,):
        prediction = test.dataset.denormalize_output(test.field_output(restored.predict(test, context)))
    with np.load(run_dir / "reconstructions.npz", allow_pickle=False) as arrays:
        assert arrays["reference"].shape == arrays["prediction"].shape == (1, 3, 6, 4)
        np.testing.assert_allclose(arrays["prediction"], prediction[:1].numpy(), rtol=1e-6, atol=1e-7)
    # Train from a saved best configuration, as a tuning-only run would leave it.
    args = arguments(data_dir, tmp_path, max_cases=2, run_id="best-config")
    best_run = run_dir.parent / "best-config"
    best_run.mkdir()
    (best_run / "best_config.json").write_text(json.dumps(checkpoint["config"]))
    basis = restored.pod
    common.run_deeponet(args, module.MODEL_ID, {}, normalizer, metadata["geometry"], coordinates, pod=basis)
    assert load_run(best_run).manifest["status"] == "completed"


def test_deeponet_runs_plot_and_compare_with_fno(data_dir, tmp_path, monkeypatch):
    normalizer, geometry, coordinates = common.prepare_deeponet_data(data_dir)
    pod = pod_deeponet.prepare_pod(data_dir, normalizer, geometry, coordinates)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config()))
    args = arguments(data_dir, tmp_path, train_config=str(config_path))
    paths = []
    for model_id, basis in (("deeponet", None), ("pod_deeponet", pod)):
        context = common.run_deeponet(args, model_id, {}, normalizer, geometry, coordinates, pod=basis)
        paths.append(context.paths.run_dir)
    context = RunContext.create(args.runs_root, problem=common.PROBLEM_ID, model="fno",
                                run_id="smoke", seed=common.SEED)
    fno_config = dict(modes_z=2, modes_r=2, hidden_channels=4, n_layers=1,
                      learning_rate=1e-3, weight_decay=0.0, batch_size=2, epochs=1)
    trainer = common.make_trainer(fno.model_from_config, fno_config, context,
                                  normalizer, geometry, (6, 4))
    raw, data = common.make_adapter(data_dir / "q2d_cmr_test.h5", normalizer, geometry)
    try:
        ExperimentRunner(context, common.experiment_spec(args, "fno")).run(
            trainer, data, data, data, config=fno_config,
            evaluator=lambda trained, dataset, ctx: evaluate_fields(
                trained, dataset, ctx, labels=tuple(c.label for c in common.OUTPUT_CHANNELS),
                coordinate_names=("z", "r"), cases=1,
            ),
        )
    finally:
        raw.close()
    paths.append(context.paths.run_dir)
    validate_model_comparison(tuple(load_run(path) for path in paths))
    monkeypatch.setattr(sys, "argv", [
        "plot.py", *[argument for path in paths for argument in ("--run", str(path))],
        "--cases", "1", "--output-dir", str(tmp_path / "plots"),
    ])
    plot.main()
    for model_id in ("deeponet", "pod_deeponet", "fno"):
        assert (tmp_path / "plots" / model_id / "smoke" / "case_00.png").is_file()
    assert (tmp_path / "plots" / "comparisons" / "group_1" / "validation_history.png").is_file()


@pytest.mark.parametrize("module", (deeponet, pod_deeponet))
def test_tuning_setup_and_train_validation_factory(data_dir, tmp_path, monkeypatch, module):
    normalizer, geometry, coordinates = common.prepare_deeponet_data(data_dir)
    pod = pod_deeponet.prepare_pod(data_dir, normalizer, geometry, coordinates) if module is pod_deeponet else None
    args = arguments(data_dir, tmp_path, tune=True, train=False, tune_epochs=3, samples=2, max_cases=2)
    space = module.search_space(pod.n_components) if pod else module.search_space()
    captured = []

    def tune_without_ray(runner, tuner, train, validation, **kwargs):
        assert train is validation is None
        assert tuner.config.metric == "best_valid_loss"
        assert tuner.config.max_epochs == tuner.search_space["epochs"] == 3
        assert tuner.config.num_samples == 2
        assert tuner.config.resources_per_trial == {"cpu": 2, "gpu": 0}
        with tuner.dataset_factory(runner.context) as (training, valid):
            assert len(training) == len(valid) == 2
            assert training.dataset.dataset.path.name == "q2d_cmr_train.h5"
            assert valid.dataset.dataset.path.name == "q2d_cmr_valid.h5"
            trial = tuner.trainer_factory(config())
            assert trial.pod is pod
        assert training.dataset.dataset._file_handle is None
        captured.append(tuner)

    monkeypatch.setattr(ExperimentRunner, "tune", tune_without_ray)
    common.run_deeponet(args, module.MODEL_ID, space, normalizer, geometry, coordinates, pod=pod)
    assert len(captured) == 1


@pytest.mark.parametrize("module", (deeponet, pod_deeponet))
def test_entrypoint_help(module):
    result = subprocess.run([sys.executable, module.__file__, "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    for option in ("--data-dir", "--train-config", "--tune", "--max-cases", "--device"):
        assert option in result.stdout
    assert "--benchmark" not in result.stdout
