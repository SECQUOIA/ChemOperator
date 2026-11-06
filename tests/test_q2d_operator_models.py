"""Smoke tests for comparable Q2D FNO, TFNO, and UNO runs."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from chem_operator.datasets import SimulationDatasetGenerator, SimulationRecord
from chem_operator.experiments import load_run
from chem_operator.reactors.q2d.dataset_generator import CMRSim
from scripts.q2d import common, fno, plot, tfno, uno


CONFIGS = (
    (fno, {"modes_z": 3, "modes_r": 2, "hidden_channels": 4, "n_layers": 1}),
    (tfno, {"modes_z": 3, "modes_r": 2, "hidden_channels": 8, "n_layers": 1, "rank": 0.1}),
    (uno, {"modes_z": 3, "hidden_channels": 16}),
)


@pytest.mark.parametrize("module, architecture", CONFIGS, ids=["fno", "tfno", "uno"])
def test_q2d_operator_forward_and_backward(module, architecture):
    model = module.model_from_config(architecture, torch.device("cpu"))
    inputs = torch.randn(2, len(common.INPUT_CHANNELS), 14, 6)
    outputs = model(inputs)
    assert outputs.shape == (2, len(common.OUTPUT_CHANNELS), 14, 6)
    outputs.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def _record(offset: float, n_z: int = 14, n_r: int = 6) -> SimulationRecord:
    z = np.linspace(0.0, 1.0, n_z)
    r = np.linspace(0.0, 0.5, n_r)
    base = z[:, None] + r[None, :] + offset
    return SimulationRecord(
        coordinates={"z": z, "r": r},
        fields={
            "X": np.stack((0.2 + 0.1 * base, 0.4 - 0.1 * base), axis=-1),
            "theta": (0.1 + 0.01 * base)[..., None],
        },
        constants={"SCCM": 500.0 + offset},
        metadata={
            "params": {"T0": 1173.0 + offset, "sccm": 500.0 + offset},
            "field_species": {"X": ["CH4", "H2"], "theta": ["C(s)"]},
            "wall_time": 0.01,
            "input_values": {"LENGTH_LUMEN": 1.0, "CHANNELRAD_LUMEN": 0.5},
        },
    )


@pytest.fixture
def data_dir(tmp_path):
    data_dir = tmp_path / "data"
    generator = SimulationDatasetGenerator(CMRSim(), data_dir)
    for split in ("train", "valid", "test"):
        generator.save_split(split, [_record(0.0), _record(0.1)])
    for n_z, n_r in ((14, 6), (18, 8)):
        generator.save_split(f"{n_z}_{n_r}_test", [_record(0.0, n_z, n_r)])
    return data_dir


@pytest.fixture
def train_cli(data_dir, tmp_path, monkeypatch):
    def train(module, architecture, *extra_args):
        config = {
            **architecture,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 1,
            "epochs": 1,
        }
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(config))
        monkeypatch.setattr(
            "sys.argv",
            [module.__file__, "--no-tune", "--train-config", str(config_path),
             "--data-dir", str(data_dir), "--runs-root", str(tmp_path / "runs"),
             "--run-id", "smoke", "--epochs", "1", "--plot-cases", "1", "--device", "cpu",
             *extra_args],
        )
        module.main()
        return tmp_path / "runs" / "q2d_cmr" / module.MODEL_ID / "smoke"
    return train


def test_q2d_operator_runs_compare_and_reload(train_cli, tmp_path, monkeypatch):
    run_paths = []
    for module, architecture in CONFIGS:
        model_id = module.MODEL_ID
        run_path = train_cli(module, architecture, *(["--no-benchmark"] if module is fno else []))
        run = load_run(run_path)
        assert run.manifest["status"] == "completed"
        assert run.manifest["model_id"] == model_id
        assert any(row["metric"] == "normalized_relative_l2" for row in run.metrics)
        assert not (run_path / "mesh_benchmark.csv").exists()
        checkpoint = torch.load(run_path / "checkpoint.pt", map_location="cpu", weights_only=True)
        shape = tuple(checkpoint["metadata"]["training_shape"])
        restored = module.model_from_config(checkpoint["config"], torch.device("cpu"))
        restored.load_state_dict(checkpoint["state_dict"])
        assert restored(torch.zeros(1, len(common.INPUT_CHANNELS), *shape)).shape == (
            1,
            len(common.OUTPUT_CHANNELS),
            *shape,
        )
        assert (run_path / "reconstructions.npz").is_file()
        run_paths.append(run_path)

    monkeypatch.setattr(
        "sys.argv",
        ["plot.py", *[argument for path in run_paths for argument in ("--run", str(path))],
         "--output-dir", str(tmp_path / "plots")],
    )
    plot.main()
    assert (tmp_path / "plots" / "comparison.png").is_file()
    for module, _ in CONFIGS:
        assert (tmp_path / "plots" / module.MODEL_ID / "smoke" / "case_00.png").is_file()


def test_q2d_fno_benchmarks_and_portable_plotting(train_cli, data_dir, tmp_path, monkeypatch):
    monkeypatch.setattr(fno, "LATENCY_WARMUPS", 0)
    monkeypatch.setattr(fno, "LATENCY_REPEATS", 1)
    run_path = train_cli(*CONFIGS[0])
    manifest = load_run(run_path).manifest
    assert len(manifest["benchmark_metadata"]["dataset_fingerprints"]) == 2
    assert (run_path / "mesh_benchmark.csv").is_file()
    assert (run_path / "cost_model.json").is_file()
    checkpoint = torch.load(run_path / "checkpoint.pt", weights_only=True)
    assert any(value.is_complex() for value in checkpoint["state_dict"].values())
    # Plotting must continue working when datasets are inaccessible.
    data_dir.rename(tmp_path / "hidden-data")
    monkeypatch.setattr("sys.argv", ["plot.py", "--run", str(run_path), "--output-dir", str(tmp_path / "plots")])
    plot.main()
    assert (tmp_path / "plots" / "fine_superresolution.png").is_file()
    assert (tmp_path / "plots" / "break_even_speedup_map.png").is_file()


def test_q2d_benchmark_failure_is_recorded(train_cli, data_dir, tmp_path):
    for path in data_dir.glob("q2d_cmr_*_*_test.h5"):
        path.unlink()
    with pytest.raises(FileNotFoundError, match="resolution-sweep"):
        train_cli(*CONFIGS[0])
    run_path = tmp_path / "runs" / "q2d_cmr" / "fno" / "smoke"
    manifest = json.loads((run_path / "manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["failure"]["type"] == "FileNotFoundError"
    assert (run_path / "checkpoint.pt").is_file()
