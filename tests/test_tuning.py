"""Integration checks for standardized tuning orchestration."""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from chem_operator._experiments.artifacts import ArtifactStore
from chem_operator._experiments.tuning import _configure_validation_physics
from chem_operator.experiments import (
    RayRuntimeConfig,
    RunContext,
    RunPaths,
    Tuner,
    TuningConfig,
)
from ray import tune


def test_tuner_owns_ray_and_returns_portable_best_trial(tmp_path: Path) -> None:
    driver_python = sys.executable

    # Serialize the objective by value: pytest's test module is not an
    # installed module on Ray workers.
    def objective(config, _train, _validation, _context, report):
        import ray

        assert sys.executable == driver_python
        assert "working_dir" not in ray.get_runtime_context().runtime_env
        score = abs(float(config["value"]) - 2.0)
        for epoch in (1, 2):
            report({"epoch": epoch, "valid_relative_l2": 0.1 / epoch, "score": score})

    context = RunContext(
        RunPaths(tmp_path / "run"),
        seed=7,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    orchestrator = Tuner.from_objective(
        objective,
        {"value": tune.choice([2.0])},
        config=TuningConfig(
            metric="score",
            mode="min",
            num_samples=1,
            max_epochs=2,
            resources_per_trial={"cpu": 1},
            max_concurrent_trials=1,
            optuna_seed=7,
            optuna_startup_trials=0,
            resume=False,
            verbose=0,
            ray_runtime=RayRuntimeConfig(
                num_cpus=1,
                # Ray appends a long session/socket suffix; keep this root short.
                temp_dir=Path("/tmp") / f"co-ray-{os.getpid()}",
            ),
        ),
    )

    outcome = orchestrator.fit(
        None,
        None,
        context,
        storage_path=tmp_path / "results",
        experiment_name="single_trial",
        artifact_store=ArtifactStore(context),
    )

    assert outcome.best_config == {"value": 2.0}
    assert outcome.best_metrics["score"] == 0.0
    assert outcome.trials[0]["status"] == "completed"
    assert outcome.tuning_seconds >= 0.0

    with context.paths.tuning_progress.open(encoding="utf-8", newline="") as stream:
        progress = list(csv.DictReader(stream))
    assert [int(row["training_iteration"]) for row in progress] == [1, 2]
    assert [float(row["valid_relative_l2"]) for row in progress] == [0.1, 0.05]
    assert {row["trial_id"] for row in progress} == {outcome.trials[0]["trial_id"]}
    assert all(float(row["timestamp"]) > 0 for row in progress)
    columns = set(progress[0])
    assert not any(name.startswith("config/") for name in columns)
    assert not columns & {"hostname", "node_ip", "pid", "date", "done"}


class _RecordingTrainer:
    """Stub trainer exposing the validation-physics hooks the Tuner sets."""

    def __init__(self, needs_physics: bool = False):
        self.reporter = None
        self.validation_physics = True
        self.needs_physics = needs_physics

    def metric_needs_validation_physics(self, metric):
        del metric
        return self.needs_physics


def test_tuner_disables_validation_physics_by_default() -> None:
    trainer = _RecordingTrainer()
    _configure_validation_physics(trainer, TuningConfig(metric="best_valid_relative_l2"))
    assert trainer.validation_physics is False


def test_tuner_keeps_validation_physics_when_metric_needs_it() -> None:
    trainer = _RecordingTrainer(needs_physics=True)
    with pytest.warns(UserWarning, match="depends on physics terms"):
        _configure_validation_physics(trainer, TuningConfig(metric="best_valid_objective"))
    assert trainer.validation_physics is True


def test_tuner_respects_explicit_validation_physics() -> None:
    trainer = _RecordingTrainer()
    _configure_validation_physics(
        trainer, TuningConfig(metric="best_valid_relative_l2", validation_physics=True)
    )
    assert trainer.validation_physics is True


def test_tuner_ignores_trainers_without_validation_physics() -> None:
    trainer = SimpleNamespace(reporter=None)
    _configure_validation_physics(trainer, TuningConfig())
    assert not hasattr(trainer, "validation_physics")


def test_tuner_trial_trainers_skip_validation_physics(tmp_path: Path) -> None:
    class Trainer:
        def __init__(self):
            self.reporter = None
            self.validation_physics = True

        def metric_needs_validation_physics(self, metric):
            del metric
            return False

        def fit(self, _train, _validation, _context):
            self.reporter({
                "epoch": 1,
                "best_valid_relative_l2": 0.1,
                "validation_physics": float(self.validation_physics),
            })

    context = RunContext(
        RunPaths(tmp_path / "run"), seed=7, dtype=torch.float32, device=torch.device("cpu")
    )
    orchestrator = Tuner(
        lambda config: Trainer(),
        {"value": tune.choice([1.0])},
        dataset_factory=lambda _context: (None, None),
        config=TuningConfig(
            metric="best_valid_relative_l2",
            mode="min",
            num_samples=1,
            max_epochs=1,
            resources_per_trial={"cpu": 1},
            max_concurrent_trials=1,
            optuna_seed=7,
            optuna_startup_trials=0,
            resume=False,
            verbose=0,
            ray_runtime=RayRuntimeConfig(
                num_cpus=1, temp_dir=Path("/tmp") / f"co-ray-{os.getpid()}"
            ),
        ),
    )
    outcome = orchestrator.fit(
        None, None, context, storage_path=tmp_path / "results", experiment_name="physics"
    )
    assert outcome.best_metrics["validation_physics"] == 0.0
