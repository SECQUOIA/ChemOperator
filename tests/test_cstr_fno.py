"""CSTR FNO data contract and canonical training smoke test."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from chem_operator.experiments import ExperimentRunner, ExperimentSpec, RunContext, load_run
from chem_operator.models import fit_zscore_normalizer
from chem_operator.plotting import plot_deeponet_run_set
from scripts.cstr import fno
from scripts.cstr.common import (
    CONSTANTS,
    FIELDS,
    LOG_TIME_POINTS,
    adapter as deeponet_adapter,
    log_time_grid,
)


class Cases(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def sample(case: int, *, future_shift: float = 0.0) -> dict:
    time = torch.from_numpy(log_time_grid()).float()
    initial_temperature = torch.tensor([900.0 + 10.0 * case])
    initial_pressure = torch.tensor([100000.0 + 100.0 * case])
    initial_species = torch.tensor([[0.01 + 0.001 * case, 0.02, 0.97]])
    ramp = torch.linspace(0.0, 1.0, LOG_TIME_POINTS - 1)
    return {
        "input_fields": {
            "T": initial_temperature,
            "P": initial_pressure,
            "X": initial_species,
        },
        "output_fields": {
            "T": initial_temperature + 5.0 * ramp + future_shift,
            "P": initial_pressure + 20.0 * ramp + future_shift,
            "X": initial_species + torch.stack(
                (0.001 * ramp, 0.002 * ramp, -0.003 * ramp), dim=-1
            ) + future_shift,
        },
        "constant_inputs": {"heat_transfer_coefficient": torch.tensor(float(case + 1))},
        "input_coordinates": {"t": time[:1]},
        "output_coordinates": {"t": time[1:]},
    }


def test_adapter_full_grid_normalization_and_no_future_leakage() -> None:
    training = Cases([sample(0), sample(1), sample(2)])
    normalizer = fit_zscore_normalizer(training, FIELDS, CONSTANTS)
    adapter = fno.CSTRFNOAdapter(training, normalizer)
    item = adapter[0]
    assert item["x"].shape == (6, LOG_TIME_POINTS)
    assert item["y"].shape == (5, LOG_TIME_POINTS)
    assert item["t"].shape == (LOG_TIME_POINTS,)
    torch.testing.assert_close(item["x"], item["x"][:, :1].expand_as(item["x"]))

    raw = training[0]
    expected = torch.cat(
        (
            torch.cat((raw["input_fields"]["T"], raw["output_fields"]["T"]))[:, None],
            torch.cat((raw["input_fields"]["P"], raw["output_fields"]["P"]))[:, None],
            torch.cat((raw["input_fields"]["X"], raw["output_fields"]["X"])),
        ),
        dim=-1,
    )
    torch.testing.assert_close(adapter.denormalize_output(item["y"]), expected)

    direct = deeponet_adapter(training, normalizer)[0]
    torch.testing.assert_close(item["x"][:, 0], direct["branch"])
    torch.testing.assert_close(item["y"].transpose(0, 1), direct["target"])

    altered = fno.CSTRFNOAdapter(Cases([sample(0, future_shift=1000.0)]), normalizer)[0]
    torch.testing.assert_close(item["x"], altered["x"])
    assert not torch.equal(item["y"], altered["y"])

    validation = Cases([sample(100, future_shift=10000.0)])
    assert normalizer.means["T"] < validation[0]["input_fields"]["T"][0]


def test_fno_canonical_cpu_run_and_plot(tmp_path) -> None:
    training = Cases([sample(0), sample(1), sample(2)])
    validation = Cases([sample(3)])
    test = Cases([sample(4)])
    normalizer = fit_zscore_normalizer(training, FIELDS, CONSTANTS)
    train = fno.CSTRFNOAdapter(training, normalizer)
    valid = fno.CSTRFNOAdapter(validation, normalizer)
    held_out = fno.CSTRFNOAdapter(test, normalizer)
    context = RunContext.create(
        tmp_path / "runs", problem="cstr_non_isothermal", model="fno", run_id="smoke", seed=42
    )
    spec = ExperimentSpec(
        problem_id="cstr_non_isothermal",
        model_id="fno",
        benchmark_protocol_id="operator-cartesian-logtime-v1",
        dataset_fingerprints={"train": "test", "validation": "test", "test": "test"},
        fields=FIELDS,
        channels=FIELDS,
        units={"T": "K", "P": "Pa", "X": "-"},
        coordinates=("t",),
        selected_test_case_ids=(0,),
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
    outcome = ExperimentRunner(context, spec).run(
        trainer, train, valid, held_out, config=config,
        evaluator=lambda trained, data, run: fno.evaluate(
            trained, data, run, reconstruction_cases=1
        ),
    )
    assert all(np.isfinite(value) for value in outcome.evaluation.metrics.values())
    assert outcome.evaluation.reconstructions["reference"].shape == (1, LOG_TIME_POINTS, 5)
    assert outcome.evaluation.reconstructions["labels"].tolist() == [
        "T", "P", "X[0]", "X[1]", "X[2]"
    ]
    assert load_run(context.paths.run_dir).manifest["status"] == "completed"

    restored = fno.make_trainer(config, context, normalizer)
    restored.load_checkpoint(context.paths.checkpoint(), context)
    torch.testing.assert_close(
        restored.predict(held_out, context), trainer.predict(held_out, context)
    )
    plots = plot_deeponet_run_set(
        [context.paths.run_dir], selected_labels=("T", "P", "X[0]"),
        coordinate_label="Time [s]", output_dir=tmp_path / "plots", cases=1,
    )
    assert all(path.stat().st_size > 0 for path in plots)
