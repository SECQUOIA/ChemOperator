"""Final-training orchestration shared by every model script.

A script supplies ``train_one(runner, context, config)``, which builds its
trainer from ``config`` (whose ``seed`` equals ``context.seed``), opens its
datasets, and calls ``runner.run(...)``. This module decides whether that
callback runs once in the flat layout or once per seed under
``<run>/seeds/<seed>/`` and where the hyperparameters come from.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .comparison import read_best_config
from .runner import ExperimentResult, ExperimentRunner, SeedTrainer
from .tuning import TuningOutcome
from .workflow import WorkflowStages


def resolve_train_config(
    train_config: str,
    runner: ExperimentRunner,
    *,
    tuning: TuningOutcome | None = None,
) -> tuple[dict[str, Any], str]:
    """Return the final-training hyperparameters and a provenance label.

    ``tuning`` wins when present. Otherwise ``train_config`` is ``best``
    (this run's ``best_config.json``), ``recent`` (the newest tuned run of
    this problem and model, see :func:`recent_tuned_run`), a completed run
    directory, or a JSON file.
    """
    if tuning is not None:
        return dict(tuning.best_config), "tuning"
    if train_config == "best":
        return runner.artifacts.read_best_config(), f"best:{runner.context.paths.run_dir}"
    if train_config == "recent":
        model_dir = runner.context.paths.root_run_dir.parent
        source = recent_tuned_run(model_dir)
        if source is None:
            raise FileNotFoundError(
                f"--train-config recent found no completed tuned run in {model_dir}. "
                "Tune this model first."
            )
        print(f"--train-config recent: using hyperparameters from {source}")
        return _completed_run_config(source, runner), str(source)
    source = Path(train_config)
    if source.is_dir():
        return _completed_run_config(source, runner), str(source)
    if source.is_file():
        config = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping) or not config:
            raise ValueError(f"{source} must contain a non-empty JSON object.")
        return dict(config), str(source)
    raise FileNotFoundError(
        f"--train-config {train_config!r} is neither 'best', a run directory, nor a JSON file."
    )


def _completed_run_config(source: Path, runner: ExperimentRunner) -> dict[str, Any]:
    """Read a completed run's best configuration after checking its problem."""
    store = ArtifactStore(source)
    if not store.paths.manifest.is_file():
        raise FileNotFoundError(f"{source} is not a run directory (no manifest.json).")
    manifest = store.read_manifest()
    if manifest["status"] != "completed":
        raise ValueError(
            f"{source} is not a completed run; cannot read its best configuration."
        )
    if manifest["problem_id"] != runner.spec.problem_id:
        raise ValueError(
            f"{source} belongs to problem {manifest['problem_id']!r}, "
            f"not {runner.spec.problem_id!r}."
        )
    return read_best_config(source)


def recent_tuned_run(model_dir: str | Path) -> Path | None:
    """Return the newest completed run in ``model_dir`` that ran a tuning study.

    A candidate must be completed, hold ``best_config.json`` and a non-empty
    ``tuning_trials.csv``, and not be a ``--max-cases`` smoke run. Runs trained
    from a saved configuration have no trials, so they are never chosen and
    configurations do not chain from one seed run to the next. Runs are
    ordered by ``completed_at`` (falling back to ``created_at``), then name.
    """
    best: tuple[tuple[str, str], Path] | None = None
    directory = Path(model_dir)
    if not directory.is_dir():
        return None
    for run_dir in directory.iterdir():
        store = ArtifactStore(run_dir)
        if not run_dir.is_dir() or not store.paths.manifest.is_file():
            continue
        try:
            manifest = store.read_manifest()
        except (OSError, ValueError):
            continue
        if manifest["status"] != "completed":
            continue
        if (manifest.get("provenance") or {}).get("max_cases") is not None:
            continue
        if not store.paths.best_config.is_file() or not store.paths.tuning_trials.is_file():
            continue
        with store.paths.tuning_trials.open(encoding="utf-8", newline="") as stream:
            if next(csv.DictReader(stream), None) is None:
                continue
        key = (str(manifest.get("completed_at") or manifest.get("created_at") or ""), run_dir.name)
        if best is None or key > best[0]:
            best = (key, run_dir)
    return None if best is None else best[1]


def train_final(
    runner: ExperimentRunner,
    stages: WorkflowStages,
    *,
    config: Mapping[str, Any],
    tuning: TuningOutcome | None,
    train_one: SeedTrainer,
    train_config_source: str = "best",
) -> tuple[ExperimentResult, ...]:
    """Run final training once (flat) or once per ``stages.seeds`` (nested)."""
    if stages.seeds is None:
        # Flat layout: the root context seeds training, so a config loaded
        # from elsewhere (or tuned under --tune-seed) must follow it.
        if tuning is None:
            runner.record_train_config_source(train_config_source)
        return (train_one(runner, runner.context, dict(config, seed=runner.context.seed)),)
    return runner.run_seeds(
        stages.seeds,
        train_one,
        config=config,
        tuning=tuning,
        train_config_source=train_config_source,
    )


__all__ = ["recent_tuned_run", "resolve_train_config", "train_final"]
