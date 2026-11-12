"""Pipe-flow FNO data contract, physical residuals, and run smoke test."""

from __future__ import annotations

import numpy as np
import torch

from chem_operator.experiments import (
    ExperimentRunner,
    ExperimentSpec,
    RunContext,
    evaluate_fields,
    load_run,
)
from scripts.pipe_flow import fno
from scripts.pipe_flow.common import FIELDS, UNITS, PipeData, fit_physics_normalization


def cases(count: int, *, start: int = 0, points: int = 16) -> PipeData:
    branch = []
    coordinates = []
    radius = []
    viscosity = []
    gradient = []
    target = []
    for index in range(start, start + count):
        pipe_radius = 0.001 + index * 0.0001
        length = 1.0 + 0.1 * index
        dynamic_viscosity = 0.001 + 0.00005 * index
        pressure_drop = 60.0 + 2.0 * index
        pressure_gradient = -pressure_drop / length
        radial = torch.linspace(0, pipe_radius, points)
        velocity = -pressure_gradient * (pipe_radius**2 - radial.square()) / (4 * dynamic_viscosity)
        branch.append(torch.tensor([pipe_radius, length, dynamic_viscosity, pressure_drop]))
        coordinates.append(radial[:, None])
        radius.append(torch.tensor([pipe_radius]))
        viscosity.append(torch.tensor([dynamic_viscosity]))
        gradient.append(torch.tensor([pressure_gradient]))
        target.append(velocity[:, None])
    return PipeData(
        branch=torch.stack(branch),
        coordinates=torch.stack(coordinates),
        radius=torch.stack(radius),
        viscosity=torch.stack(viscosity),
        pressure_gradient=torch.stack(gradient),
        target=torch.stack(target),
    )


def test_adapter_uses_controls_only_and_physical_grid() -> None:
    raw = cases(3)
    statistics = fit_physics_normalization(raw)
    dataset = fno.PipeFNODataset(raw, statistics)
    item = dataset[0]
    assert item["x"].shape == (4, 16)
    assert item["y"].shape == (1, 16)
    assert item["r"].shape == (16,)
    torch.testing.assert_close(item["x"], item["x"][:, :1].expand_as(item["x"]))
    torch.testing.assert_close(dataset.denormalize_output(item["y"])[0], raw.target[0, :, 0])

    altered = PipeData(
        branch=raw.branch,
        coordinates=raw.coordinates,
        radius=raw.radius,
        viscosity=raw.viscosity,
        pressure_gradient=raw.pressure_gradient,
        target=raw.target + 1,
    )
    changed = fno.PipeFNODataset(altered, statistics)[0]
    torch.testing.assert_close(item["x"], changed["x"])
    assert not torch.equal(item["y"], changed["y"])


def test_exact_profile_has_small_physics_loss_and_perturbation_does_not() -> None:
    raw = cases(2, points=32)
    statistics = fit_physics_normalization(raw)
    dataset = fno.PipeFNODataset(raw, statistics)
    batch = {key: torch.stack([dataset[i][key] for i in range(2)]) for key in dataset[0]}
    pde, boundary = fno.physics_losses(batch["y"], batch, statistics)
    assert pde.item() < 1e-5
    assert boundary.item() < 1e-5

    changed = batch["y"].clone()
    changed[:, :, -1] += 0.5
    changed_pde, changed_boundary = fno.physics_losses(changed, batch, statistics)
    assert changed_pde > pde
    assert changed_boundary > boundary


def test_canonical_run_and_plot_arrays(tmp_path) -> None:
    training = cases(4)
    statistics = fit_physics_normalization(training)
    train = fno.PipeFNODataset(training, statistics)
    valid = fno.PipeFNODataset(cases(1, start=4), statistics)
    test = fno.PipeFNODataset(cases(2, start=5), statistics)
    context = RunContext.create(
        tmp_path / "runs", problem="pipe_flow", model="fno_physics", run_id="smoke", seed=42
    )
    spec = ExperimentSpec(
        problem_id="pipe_flow",
        model_id="fno_physics",
        benchmark_protocol_id="operator-cartesian-v1",
        dataset_fingerprints={"train": "test", "validation": "test", "test": "test"},
        fields=FIELDS,
        channels=FIELDS,
        units=UNITS,
        coordinates=("r",),
        selected_test_case_ids=(0, 1),
        project_root=tmp_path,
    )
    config = {
        "modes": 4,
        "hidden_channels": 4,
        "n_layers": 2,
        "learning_rate": 1e-3,
        "weight_decay": 1e-6,
        "batch_size": 2,
        "epochs": 1,
        "physics_weight": 0.01,
    }
    trainer = fno.make_trainer(config, context, statistics)
    result = ExperimentRunner(context, spec).run(
        trainer, train, valid, test, config=config,
        evaluator=lambda fitted, held_out, run: evaluate_fields(
            fitted, held_out, run, labels=FIELDS, coordinate_names=("r",), cases=2,
        ),
    )
    assert all(np.isfinite(value) for value in result.evaluation.metrics.values())
    arrays = result.evaluation.reconstructions
    assert arrays["reference"].shape == (2, 1, 16)
    assert arrays["prediction"].shape == (2, 1, 16)
    assert arrays["r"].shape == (2, 16)
    assert load_run(context.paths.run_dir).manifest["status"] == "completed"
    checkpoint = torch.load(context.paths.checkpoint(), map_location="cpu", weights_only=True)
    assert checkpoint["metadata"]["pde_name"] == "HagenPoiseuille"
