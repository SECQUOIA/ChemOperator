"""Integration checks for standardized tuning orchestration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

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
        report({"score": abs(float(config["value"]) - 2.0)})

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
            max_epochs=1,
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
    )

    assert outcome.best_config == {"value": 2.0}
    assert outcome.best_metrics["score"] == 0.0
    assert outcome.trials[0]["status"] == "completed"
    assert outcome.tuning_seconds >= 0.0
