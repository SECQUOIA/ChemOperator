"""PFR FNO data contract and canonical training smoke test."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from chem_operator.experiments import ExperimentRunner, ExperimentSpec, RunContext, load_run
from chem_operator.models import fit_zscore_normalizer
from chem_operator.normalization import normalizer_from_state_dict
from chem_operator.plotting import plot_deeponet_run_set
from scripts.pfr import fno
from scripts.pfr.common import CONSTANTS, FIELDS, UNITS, adapter as deeponet_adapter


class Cases(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def sample(case: int, *, future_shift: float = 0.0) -> dict:
    z = torch.linspace(0.0, 1.5e-7, fno.GRID_POINTS)
    ramp = torch.linspace(0.0, 1.0, fno.GRID_POINTS - 1)
    inlet_temperature = torch.tensor([1200.0 + 20.0 * case])
    inlet_pressure = torch.tensor([101325.0 + 100.0 * case])
    inlet_species = torch.linspace(0.01, 0.1, 10).unsqueeze(0) + 0.001 * case
    inlet_velocity = torch.tensor([0.005 + 0.0001 * case])
    constants = {
        "length": torch.tensor(1.5e-7),
        "area": torch.tensor(1.0e-4),
        "inlet_velocity": inlet_velocity[0],
        "pressure_controller_K": torch.tensor(1.0e-12),
        "phi": torch.tensor(0.5 + 0.1 * case),
        "ar_o2_ratio": torch.tensor(0.02 + 0.001 * case),
    }
    return {
        "input_fields": {
            "T": inlet_temperature,
            "P": inlet_pressure,
            "X": inlet_species,
            "velocity": inlet_velocity,
        },
        "output_fields": {
            "T": inlet_temperature + 10.0 * ramp + future_shift,
            "P": inlet_pressure + 20.0 * ramp + future_shift,
            "X": inlet_species + 0.01 * ramp[:, None] + future_shift,
            "velocity": inlet_velocity + 0.001 * ramp + future_shift,
        },
        "constant_inputs": constants,
        "input_coordinates": {"z": z[:1]},
        "output_coordinates": {"z": z[1:]},
    }


def test_adapter_full_grid_normalization_and_no_future_leakage() -> None:
    training = Cases([sample(0), sample(1), sample(2)])
    normalizer = fit_zscore_normalizer(training, FIELDS, CONSTANTS)
    adapter = fno.PFRFNOAdapter(training, normalizer)
    item = adapter[0]
    assert item["x"].shape == (19, fno.GRID_POINTS)
    assert item["y"].shape == (13, fno.GRID_POINTS)
    assert item["z"].shape == (fno.GRID_POINTS,)
    torch.testing.assert_close(item["x"], item["x"][:, :1].expand_as(item["x"]))

    raw = training[0]
    expected = torch.cat(
        (
            torch.cat(
                (
                    raw["input_fields"]["T"][:, None],
                    raw["input_fields"]["P"][:, None],
                    raw["input_fields"]["X"],
                    raw["input_fields"]["velocity"][:, None],
                ),
                dim=-1,
            ),
            torch.cat(
                (
                    raw["output_fields"]["T"][:, None],
                    raw["output_fields"]["P"][:, None],
                    raw["output_fields"]["X"],
                    raw["output_fields"]["velocity"][:, None],
                ),
                dim=-1,
            ),
        )
    )
    torch.testing.assert_close(adapter.denormalize_output(item["y"]), expected)

    direct = deeponet_adapter(training, normalizer)[0]
    torch.testing.assert_close(item["x"][:, 0], direct["branch"])
    torch.testing.assert_close(item["y"].transpose(0, 1), direct["target"])

    altered = fno.PFRFNOAdapter(
        Cases([sample(0, future_shift=1000.0)]), normalizer
    )[0]
    torch.testing.assert_close(item["x"], altered["x"])
    assert not torch.equal(item["y"], altered["y"])

    shifted = sample(0)
    shifted["output_coordinates"]["z"] += 1e-9
    with pytest.raises(ValueError, match="same axial grid"):
        fno.PFRFNOAdapter(Cases([sample(0), shifted]), normalizer)[1]

    malformed = sample(0)
    malformed["output_fields"]["X"] = malformed["output_fields"]["X"][:, :-1]
    with pytest.raises(ValueError, match="future fields"):
        fno.PFRFNOAdapter(Cases([malformed]), normalizer)[0]


def test_fno_canonical_cpu_run_checkpoint_and_plot(tmp_path) -> None:
    training = Cases([sample(0), sample(1), sample(2)])
    validation = Cases([sample(3)])
    test = Cases([sample(4), sample(5), sample(6)])
    normalizer = fit_zscore_normalizer(training, FIELDS, CONSTANTS)
    train = fno.PFRFNOAdapter(training, normalizer)
    valid = fno.PFRFNOAdapter(validation, normalizer)
    held_out = fno.PFRFNOAdapter(test, normalizer)
    context = RunContext.create(
        tmp_path / "runs", problem="pfr_chain", model="fno", run_id="smoke", seed=42
    )
    spec = ExperimentSpec(
        problem_id="pfr_chain",
        model_id="fno",
        benchmark_protocol_id="operator-cartesian-v1",
        dataset_fingerprints={"train": "test", "validation": "test", "test": "test"},
        fields=FIELDS,
        channels=FIELDS,
        units=UNITS,
        coordinates=("z",),
        selected_test_case_ids=(0, 1),
        project_root=tmp_path,
    )
    config = {
        "modes": 8,
        "hidden_channels": 4,
        "n_layers": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "batch_size": 2,
        "epochs": 1,
        "seed": 42,
    }
    trainer = fno.make_trainer(config, context, normalizer)
    result = ExperimentRunner(context, spec).run(
        trainer,
        train,
        valid,
        held_out,
        config=config,
        evaluator=lambda trained, data, run: fno.evaluate(
            trained, data, run, reconstruction_cases=2
        ),
    )
    assert all(np.isfinite(value) for value in result.evaluation.metrics.values())
    reconstructions = result.evaluation.reconstructions
    assert reconstructions["reference"].shape == (2, fno.GRID_POINTS, 13)
    assert reconstructions["prediction"].shape == (2, fno.GRID_POINTS, 13)
    assert reconstructions["coordinates"].shape == (2, fno.GRID_POINTS)
    assert reconstructions["labels"].tolist() == [
        "T", "P", *(f"X[{index}]" for index in range(10)), "velocity"
    ]
    assert load_run(context.paths.run_dir).manifest["status"] == "completed"
    checkpoint = torch.load(
        context.paths.checkpoint(), map_location="cpu", weights_only=True
    )
    restored_normalizer = normalizer_from_state_dict(
        checkpoint["metadata"]["normalizer"]
    )
    torch.testing.assert_close(
        restored_normalizer.flattened_means["variable"],
        normalizer.flattened_means["variable"],
    )
    assert checkpoint["metadata"]["input_channels"] == 19
    assert checkpoint["metadata"]["output_channels"] == 13
    assert checkpoint["metadata"]["grid_points"] == fno.GRID_POINTS

    restored = fno.make_trainer(config, context, normalizer)
    restored.load_checkpoint(context.paths.checkpoint(), context)
    torch.testing.assert_close(
        restored.predict(held_out, context), trainer.predict(held_out, context)
    )
    plots = plot_deeponet_run_set(
        [context.paths.run_dir],
        selected_labels=("T", "velocity", "X[0]"),
        coordinate_label="Axial position z [m]",
        output_dir=tmp_path / "plots",
        cases=2,
    )
    assert all(path.stat().st_size > 0 for path in plots)
