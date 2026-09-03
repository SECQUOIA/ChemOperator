"""Tests for the transient pipe-flow FNO data and training path."""

from __future__ import annotations

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from matplotlib.axes import Axes
from neuralop.models import FNO

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.models import FNOAdapter, fit_zscore_normalizer
from chem_operator.reactors.pipe_flow_transient.dataset_generator import (
    TransientHagenPoiseuillePipeFlowSim,
)
from chem_operator.sampling import Constant, Uniform


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "pipe_flow_transient"
    / "transient_fno.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "pipe_flow_transient_fno",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
FNO_SCRIPT = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(FNO_SCRIPT)
CONSTANT_NAMES = FNO_SCRIPT.CONSTANT_NAMES
FIELD_NAMES = FNO_SCRIPT.FIELD_NAMES
evaluate = FNO_SCRIPT.evaluate
make_adapter = FNO_SCRIPT.make_adapter
plot_history = FNO_SCRIPT.plot_history
plot_reconstructions = FNO_SCRIPT.plot_reconstructions
raw_dataset = FNO_SCRIPT.raw_dataset
load_checkpoint = FNO_SCRIPT.load_checkpoint
save_checkpoint = FNO_SCRIPT.save_checkpoint
train_model = FNO_SCRIPT.train_model
use_saved_model = FNO_SCRIPT.use_saved_model
write_history = FNO_SCRIPT.write_history


@pytest.fixture(scope="module", name="fno_data_dir")
def fixture_fno_data_dir(tmp_path_factory) -> object:
    """Generate small, varied train/validation/test trajectory splits."""
    output = tmp_path_factory.mktemp("pipe_flow_fno")
    simulator = TransientHagenPoiseuillePipeFlowSim(
        parameter_space={
            "radius": Uniform(0.8e-3, 1.2e-3),
            "length": Uniform(0.8, 1.2),
            "dynamic_viscosity": Uniform(0.9e-3, 1.1e-3),
            "pressure_drop": Uniform(40.0, 80.0),
            "density": Constant(1000.0),
            "n_time_points": Constant(8),
            "n_radial_points": Constant(8),
            "max_fourier_number": Constant(2.0),
        }
    )
    generator = SimulationDatasetGenerator(simulator, output, seed=11)
    generator.save_splits(generator.generate_splits(n_cases=10))
    return output


@pytest.fixture(name="fno_normalizer")
def fixture_fno_normalizer(fno_data_dir):
    """Fit normalization using the small training split only."""
    dataset = raw_dataset(fno_data_dir, "train")
    try:
        return fit_zscore_normalizer(dataset, FIELD_NAMES, CONSTANT_NAMES)
    finally:
        dataset.close()


def test_fno_adapter_preserves_grid_and_round_trips(
    fno_data_dir,
    fno_normalizer,
) -> None:
    """The adapter returns complete channel-first fields and physical grids."""
    dataset = raw_dataset(fno_data_dir, "train")
    adapter = FNOAdapter(
        dataset,
        fno_normalizer,
        field_names=FIELD_NAMES,
        constant_names=CONSTANT_NAMES,
    )
    try:
        sample = adapter[0]
        assert sample["x"].shape == (4, 8, 8)
        assert sample["y"].shape == (1, 8, 8)
        assert sample["t"].shape == (8,)
        assert sample["r"].shape == (8,)
        for channel in sample["x"]:
            torch.testing.assert_close(
                channel,
                channel[0, 0].expand_as(channel),
            )

        raw = dataset[0]
        expected = torch.cat(
            (
                raw["input_fields"]["velocity"],
                raw["output_fields"]["velocity"],
            )
        )
        reconstructed = adapter.denormalize_output(sample["y"])[0]
        torch.testing.assert_close(reconstructed, expected)

        model = FNO(
            n_modes=(2, 2),
            in_channels=4,
            out_channels=1,
            hidden_channels=4,
            n_layers=2,
        )
        prediction = model(sample["x"].unsqueeze(0))
        assert prediction.shape == (1, 1, 8, 8)
        prediction.square().mean().backward()
    finally:
        dataset.close()


def test_fno_training_and_plots_smoke(  # pylint: disable=too-many-locals
    fno_data_dir,
    fno_normalizer,
    tmp_path,
    monkeypatch,
) -> None:
    """One CPU epoch produces finite losses, metrics, and both plots."""
    train_raw, train_data = make_adapter(
        fno_data_dir, "train", fno_normalizer, maximum=4
    )
    valid_raw, valid_data = make_adapter(
        fno_data_dir, "valid", fno_normalizer, maximum=1
    )
    config = {
        "modes": 2,
        "hidden_channels": 4,
        "n_layers": 2,
        "learning_rate": 1.0e-3,
        "weight_decay": 1.0e-8,
        "batch_size": 2,
    }
    try:
        model, history, _ = train_model(
            config,
            train_data,
            valid_data,
            epochs=1,
            device=torch.device("cpu"),
        )
        assert len(history["train_loss"]) == 1
        assert len(history["valid_loss"]) == 1
        assert math.isfinite(history["train_loss"][0])
        assert math.isfinite(history["valid_loss"][0])

        metrics = evaluate(
            model,
            valid_data,
            batch_size=1,
            device=torch.device("cpu"),
        )
        assert np.isfinite(metrics["relative_l2"])
        assert np.isfinite(metrics["rmse"])

        checkpoint_path = tmp_path / "fno.pt"
        save_checkpoint(checkpoint_path, model, config, fno_normalizer)
        loaded_model, loaded_normalizer = load_checkpoint(
            checkpoint_path,
            torch.device("cpu"),
        )
        loaded_data = FNOAdapter(
            valid_raw,
            loaded_normalizer,
            field_names=FIELD_NAMES,
            constant_names=CONSTANT_NAMES,
        )
        sample = loaded_data[0]
        with torch.no_grad():
            expected = model(sample["x"].unsqueeze(0))
            actual = loaded_model(sample["x"].unsqueeze(0))
        torch.testing.assert_close(actual, expected)

        history_path = tmp_path / "history.csv"
        loss_path = tmp_path / "loss.png"
        reconstruction_path = tmp_path / "reconstruction.png"
        write_history(history_path, history)
        plot_history(loss_path, history)
        extents = []
        original_imshow = Axes.imshow

        def capture_imshow(axis, values, *args, **kwargs):
            extents.append(kwargs["extent"])
            return original_imshow(axis, values, *args, **kwargs)

        monkeypatch.setattr(Axes, "imshow", capture_imshow)
        plot_reconstructions(
            reconstruction_path,
            model,
            valid_data,
            cases=1,
            device=torch.device("cpu"),
        )
        plotted_sample = valid_data[0]
        assert extents[0] == (
            float(plotted_sample["t"][0]),
            float(plotted_sample["t"][-1]),
            float(1.0e3 * plotted_sample["r"][0]),
            float(1.0e3 * plotted_sample["r"][-1]),
        )

        monkeypatch.setattr(
            FNO_SCRIPT,
            "PATHS",
            replace(FNO_SCRIPT.PATHS, output=tmp_path, data=fno_data_dir),
        )
        monkeypatch.setattr(FNO_SCRIPT, "MAX_TEST_TRAJECTORIES", 1)
        use_saved_model(torch.device("cpu"), calculate_metrics=False)
        for path in (
            checkpoint_path,
            history_path,
            loss_path,
            reconstruction_path,
        ):
            assert path.is_file()
            assert path.stat().st_size > 0
    finally:
        train_raw.close()
        valid_raw.close()


def test_conflicting_python_run_flags_are_rejected(monkeypatch) -> None:
    """Only one abbreviated run mode may be selected."""
    monkeypatch.setattr(FNO_SCRIPT, "TRAIN_BEST_CONFIG_ONLY", True)
    monkeypatch.setattr(FNO_SCRIPT, "PLOT_SAVED_MODEL_ONLY", True)
    with pytest.raises(ValueError, match="cannot both be true"):
        FNO_SCRIPT.main()
