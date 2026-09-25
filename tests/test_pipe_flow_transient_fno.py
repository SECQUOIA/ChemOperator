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
make_adapter = FNO_SCRIPT.make_adapter
raw_dataset = FNO_SCRIPT.raw_dataset


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


def test_fno_training_and_plots_smoke(fno_data_dir, fno_normalizer, tmp_path):
    """A canonical CPU run reloads its checkpoint and plots without its data."""
    from argparse import Namespace
    from chem_operator.experiments import ExperimentRunner, RunContext, load_run
    from chem_operator.plotting import plot_operator_runs
    from chem_operator.normalization import normalizer_from_state_dict
    raw, train = make_adapter(fno_data_dir, "train", fno_normalizer, 4)
    valid_raw, valid = make_adapter(fno_data_dir, "valid", fno_normalizer, 1)
    config = dict(modes=2, hidden_channels=4, n_layers=2, learning_rate=1e-3,
                  weight_decay=1e-8, batch_size=2, epochs=1)
    context = RunContext.create(tmp_path / "runs", problem="pipe_flow_transient",
                               model="transient_fno", run_id="smoke", seed=42)
    args = Namespace(data_dir=fno_data_dir, samples=1, tune_epochs=1, plot_cases=1, max_cases=1)
    try:
        trainer = FNO_SCRIPT.make_trainer(config, context, fno_normalizer)
        result = ExperimentRunner(context, FNO_SCRIPT.experiment_spec(args, "transient_fno")).run(
            trainer, train, valid, valid, config=config,
            evaluator=lambda t,d,c: FNO_SCRIPT.evaluate_fields(t,d,c,labels=("velocity",),coordinate_names=("t","r"),cases=1))
        assert all(math.isfinite(v) for v in result.evaluation.metrics.values())
        assert result.training.metadata["checkpoint_reload_verified"]
        saved = torch.load(context.paths.checkpoint(), weights_only=True)
        normalizer = normalizer_from_state_dict(saved["metadata"]["normalizer"])
        restored = FNO_SCRIPT.make_trainer(config, context, normalizer)
        restored.load_checkpoint(context.paths.checkpoint(), context)
        torch.testing.assert_close(restored.predict(valid,context), trainer.predict(valid,context))
        assert load_run(context.paths.run_dir).manifest["status"] == "completed"
    finally:
        raw.close()
        valid_raw.close()
    for path in plot_operator_runs([context.paths.run_dir], tmp_path / "plots", cases=1):
        assert path.stat().st_size > 0
