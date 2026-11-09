"""Smoke tests for comparable Q2D FNO, TFNO, and UNO runs."""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
import torch

from chem_operator.datasets import SimulationDatasetGenerator, SimulationRecord
from chem_operator.experiments import ExperimentRunner, RunContext, evaluate_fields, load_run
from chem_operator.reactors.q2d.dataset_generator import CMRSim
from scripts.q2d import common, fno, plot, tfno, uno
from scripts.q2d.common import evaluate, make_trainer


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


def _record(offset: float) -> SimulationRecord:
    z = np.linspace(0.0, 1.0, 14)
    r = np.linspace(0.0, 0.5, 6)
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


def test_q2d_operator_runs_compare_and_reload(tmp_path, monkeypatch):
    data_dir = tmp_path / "data"
    generator = SimulationDatasetGenerator(CMRSim(), data_dir)
    for split in ("train", "valid", "test"):
        generator.save_split(split, [_record(0.0), _record(0.1)])

    normalizer, geometry, shape = common.fit_normalizer(data_dir / "q2d_cmr_train.h5")
    args = Namespace(data_dir=data_dir, samples=1, tune_epochs=1, plot_cases=1, max_cases=None)
    run_paths = []

    for module, architecture in CONFIGS:
        model_id = module.MODEL_ID
        config = {
            **architecture,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 1,
            "epochs": 1,
        }
        context = RunContext.create(
            tmp_path / "runs", problem="q2d_cmr", model=model_id, run_id="smoke", seed=42
        )
        raw, data = common.make_adapter(data_dir / "q2d_cmr_test.h5", normalizer, geometry)
        trainer = make_trainer(module.model_from_config, config, context, normalizer, geometry, shape)
        try:
            ExperimentRunner(context, common.experiment_spec(args, model_id)).run(
                trainer,
                data,
                data,
                data,
                config=config,
                evaluator=lambda trained, dataset, run_context: evaluate_fields(
                    trained,
                    dataset,
                    run_context,
                    labels=tuple(channel.label for channel in common.OUTPUT_CHANNELS),
                    coordinate_names=("z", "r"),
                    cases=1,
                    extra_metrics=lambda current, current_data, current_context: evaluate(
                        current.model,
                        current_data,
                        batch_size=1,
                        device=current_context.device,
                    ),
                ),
            )
        finally:
            raw.close()

        run = load_run(context.paths.run_dir)
        assert run.manifest["model_id"] == model_id
        assert any(row["metric"] == "normalized_relative_l2" for row in run.metrics)
        checkpoint = torch.load(context.paths.checkpoint(), map_location="cpu", weights_only=True)
        restored = module.model_from_config(checkpoint["config"], torch.device("cpu"))
        restored.load_state_dict(checkpoint["state_dict"])
        assert restored(torch.zeros(1, len(common.INPUT_CHANNELS), *shape)).shape == (
            1,
            len(common.OUTPUT_CHANNELS),
            *shape,
        )
        assert (context.paths.run_dir / "reconstructions.npz").is_file()
        run_paths.append(context.paths.run_dir)

    monkeypatch.setattr(
        "sys.argv",
        ["plot.py", *[argument for path in run_paths for argument in ("--run", str(path))],
         "--output-dir", str(tmp_path / "plots")],
    )
    plot.main()
    assert (tmp_path / "plots" / "comparisons" / "group_1" / "validation_history.png").is_file()
    for module, _ in CONFIGS:
        assert (tmp_path / "plots" / module.MODEL_ID / "smoke" / "case_00.png").is_file()
