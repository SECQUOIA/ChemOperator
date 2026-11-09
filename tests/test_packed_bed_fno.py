"""Packed-bed FNO resampling, data contract, and canonical run smoke test."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import Dataset

from chem_operator.datasets import TrajectoryResamplingDataset
from chem_operator.experiments import ExperimentRunner, ExperimentSpec, RunContext, load_run
from chem_operator.models import fit_zscore_normalizer
from chem_operator.normalization import normalizer_from_state_dict
from chem_operator.plotting import plot_deeponet_run_set
from scripts.packed_bed_1d import fno
from scripts.packed_bed_1d.common import CONSTANTS, FIELDS, UNITS


class Cases(Dataset):
    def __init__(self, samples: list[dict]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        return self.samples[index]


def source_cases() -> Cases:
    samples = []
    for case in range(7):
        z = torch.linspace(0.0, 0.052 + 0.001 * case, 5 + case % 3)
        fraction = z / 0.05
        profiles = {
            "T": 650 + case * 3 + 2 * fraction,
            "P": 5e5 + case * 100 + 10 * fraction,
            "X": torch.stack(
                tuple(0.1 * (i + 1) + 0.001 * case + 0.01 * fraction for i in range(4)),
                dim=-1,
            ),
            "Z": torch.stack(
                tuple(0.05 * (i + 1) + 0.001 * case + 0.005 * fraction for i in range(6)),
                dim=-1,
            ),
            "rhou": 0.2 + 0.001 * case + 0.001 * fraction,
            "velocity": 0.002 + 0.0001 * case + 0.0001 * fraction,
        }
        samples.append({
            "input_fields": {name: value[:1] for name, value in profiles.items()},
            "output_fields": {name: value[1:] for name, value in profiles.items()},
            "constant_inputs": {
                name: torch.tensor(0.05 if name == "length" else float(i + 1) + case / 10)
                for i, name in enumerate(CONSTANTS)
            },
            "input_coordinates": {"z": z[:1]},
            "output_coordinates": {"z": z[1:]},
        })
    return Cases(samples)


def adapted(data: Dataset, normalizer) -> fno.PackedBedFNOAdapter:
    resampled = TrajectoryResamplingDataset(
        data,
        coordinate_name="z",
        target_coordinate=torch.linspace(0.0, fno.BED_LENGTH, fno.GRID_POINTS),
        cache=False,
    )
    return fno.PackedBedFNOAdapter(resampled, normalizer)


def test_adapter_resamples_channels_and_excludes_future_from_input() -> None:
    source = source_cases()
    normalizer = fit_zscore_normalizer(source, FIELDS, CONSTANTS)
    data = adapted(source, normalizer)
    first, second = data[0], data[1]
    assert first["x"].shape == (29, fno.GRID_POINTS)
    assert first["y"].shape == (14, fno.GRID_POINTS)
    assert first["z"].shape == (fno.GRID_POINTS,)
    torch.testing.assert_close(first["z"], second["z"])
    torch.testing.assert_close(first["z"], torch.linspace(0, 0.05, 128))
    torch.testing.assert_close(first["x"], first["x"][:, :1].expand_as(first["x"]))
    assert data.labels == (
        "T", "P", *(f"X[{i}]" for i in range(4)),
        *(f"Z[{i}]" for i in range(6)), "rhou", "velocity",
    )
    inlet = torch.cat(tuple(source[0]["input_fields"][name].reshape(-1) for name in FIELDS))
    constants = torch.stack(tuple(source[0]["constant_inputs"][name].reshape(()) for name in CONSTANTS))
    expected = torch.cat((
        normalizer.normalize_flattened(inlet, "variable"),
        normalizer.normalize_flattened(constants, "constant"),
    ))
    torch.testing.assert_close(first["x"][:, 0], expected)
    torch.testing.assert_close(data.denormalize_output(first["y"])[0], inlet)
    torch.testing.assert_close(
        data.denormalize_output(first["y"])[:, 0],
        650 + 2 * first["z"] / 0.05,
        rtol=1e-5,
        atol=1e-4,
    )

    changed = deepcopy(source[0])
    changed["output_fields"]["T"] += 10
    shifted = adapted(Cases([changed]), normalizer)[0]
    torch.testing.assert_close(first["x"], shifted["x"])
    assert not torch.equal(first["y"], shifted["y"])


def test_canonical_cpu_run_checkpoint_and_plot(tmp_path) -> None:
    source = source_cases()
    normalizer = fit_zscore_normalizer(Cases(source.samples[:3]), FIELDS, CONSTANTS)
    train = adapted(Cases(source.samples[:3]), normalizer)
    valid = adapted(Cases(source.samples[3:4]), normalizer)
    test = adapted(Cases(source.samples[4:7]), normalizer)
    context = RunContext.create(
        tmp_path / "runs", problem="packed_bed_1d", model="fno", run_id="smoke", seed=7
    )
    spec = ExperimentSpec(
        problem_id="packed_bed_1d",
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
        "seed": 7,
    }
    trainer = fno.make_trainer(config, context, normalizer)
    result = ExperimentRunner(context, spec).run(
        trainer, train, valid, test, config=config,
        evaluator=lambda trained, data, run: fno.evaluate(
            trained, data, run, reconstruction_cases=2
        ),
    )
    assert all(np.isfinite(value) for value in result.evaluation.metrics.values())
    recon = result.evaluation.reconstructions
    assert recon["reference"].shape == (2, 128, 14)
    assert recon["prediction"].shape == (2, 128, 14)
    assert recon["coordinates"].shape == (2, 128)
    assert recon["labels"].tolist() == list(test.labels)
    assert load_run(context.paths.run_dir).manifest["status"] == "completed"
    checkpoint = torch.load(context.paths.checkpoint(), map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    assert (metadata["input_channels"], metadata["output_channels"], metadata["grid_points"]) == (29, 14, 128)
    restored_normalizer = normalizer_from_state_dict(metadata["normalizer"])
    torch.testing.assert_close(
        restored_normalizer.flattened_means["variable"],
        normalizer.flattened_means["variable"],
    )
    restored = fno.make_trainer(config, context, normalizer)
    restored.load_checkpoint(context.paths.checkpoint(), context)
    torch.testing.assert_close(restored.predict(test, context), trainer.predict(test, context))
    plots = plot_deeponet_run_set(
        [context.paths.run_dir],
        selected_labels=("T", "velocity", "X[0]"),
        coordinate_label="Axial position z [m]",
        output_dir=tmp_path / "plots",
        cases=2,
    )
    assert all(path.stat().st_size > 0 for path in plots)
