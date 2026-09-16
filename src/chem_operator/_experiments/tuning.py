"""Ray/Optuna mechanics for script-defined trainer factories and spaces."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore
from .trainers import accepts_context
from .types import RunContext, Trainer

TrainerFactory = Callable[..., Trainer]


@dataclass(frozen=True, slots=True)
class TuningOutcome:
    """Best configuration and portable, flattened trial records."""

    best_config: Mapping[str, Any]
    trials: Sequence[Mapping[str, Any]]
    tuning_seconds: float


class Tuner:
    """Thin Ray Tune wrapper with Optuna search and resumable storage."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        trainer_factory: TrainerFactory,
        search_space: Mapping[str, Any],
        *,
        metric: str = "valid_relative_l2",
        mode: str = "min",
        num_samples: int = 1,
        resources_per_trial: Mapping[str, float] | None = None,
        max_concurrent_trials: int | None = None,
        scheduler: Any = None,
        search_algorithm: Any = None,
    ) -> None:
        if not search_space:
            raise ValueError("search_space cannot be empty.")
        if not metric:
            raise ValueError("metric must be non-empty.")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'.")
        if num_samples < 1:
            raise ValueError("num_samples must be positive.")
        self.trainer_factory = trainer_factory
        self.search_space = dict(search_space)
        self.metric = metric
        self.mode = mode
        self.num_samples = num_samples
        self.resources_per_trial = dict(resources_per_trial or {"cpu": 1})
        self.max_concurrent_trials = max_concurrent_trials
        self.scheduler = scheduler
        self.search_algorithm = search_algorithm

    def fit(  # pylint: disable=too-many-locals
        self,
        train: Any,
        validation: Any,
        context: RunContext,
        *,
        storage_path: str | Path | None = None,
        experiment_name: str = "tuning",
        resume: bool = True,
        artifact_store: ArtifactStore | None = None,
    ) -> TuningOutcome:
        """Run or resume tuning and optionally persist portable artifacts."""
        from ray import train as ray_train
        from ray import tune

        if not experiment_name or Path(experiment_name).name != experiment_name:
            raise ValueError("experiment_name must be one path component.")

        factory_takes_context = accepts_context(self.trainer_factory)

        def trainable(config: Mapping[str, Any]) -> None:
            trial_id = ray_train.get_context().get_trial_id() or "trial"
            trial_context = context.trial(trial_id)
            trainer = (
                self.trainer_factory(config, context=trial_context)
                if factory_takes_context
                else self.trainer_factory(config)
            )
            reported = False

            def report(values: Mapping[str, float | int]) -> None:
                nonlocal reported
                reported = True
                payload = dict(values)
                if self.metric not in payload:
                    if self.metric == "valid_relative_l2" and "validation_relative_l2" in payload:
                        payload[self.metric] = payload["validation_relative_l2"]
                ray_train.report(payload)

            if hasattr(trainer, "reporter"):
                trainer.reporter = report
            outcome = trainer.fit(train, validation, trial_context)
            if not reported:
                payload = dict(outcome.metrics)
                if self.metric not in payload:
                    if self.metric == "valid_relative_l2" and "validation_relative_l2" in payload:
                        payload[self.metric] = payload["validation_relative_l2"]
                    else:
                        raise KeyError(
                            f"Trainer outcome did not contain tuning metric {self.metric!r}."
                        )
                payload.update(
                    {
                        "best_epoch": outcome.best_epoch,
                        **{
                            key: value
                            for key, value in outcome.metadata.items()
                            if isinstance(value, (int, float))
                        },
                    }
                )
                ray_train.report(payload)

        configured_trainable = tune.with_resources(
            trainable,
            resources=self.resources_per_trial,
        )
        root = (
            context.paths.run_dir / "ray_results"
            if storage_path is None
            else Path(storage_path)
        )
        experiment_path = root / experiment_name
        started = time.perf_counter()
        if resume and tune.Tuner.can_restore(str(experiment_path)):
            tuner = tune.Tuner.restore(
                str(experiment_path),
                trainable=configured_trainable,
                resume_unfinished=True,
                resume_errored=True,
            )
        else:
            search_algorithm = self.search_algorithm
            if search_algorithm is None:
                from ray.tune.search.optuna import OptunaSearch

                search_algorithm = OptunaSearch(metric=self.metric, mode=self.mode)
            tuner = tune.Tuner(
                configured_trainable,
                param_space=self.search_space,
                tune_config=tune.TuneConfig(
                    metric=self.metric,
                    mode=self.mode,
                    num_samples=self.num_samples,
                    scheduler=self.scheduler,
                    search_alg=search_algorithm,
                    max_concurrent_trials=self.max_concurrent_trials,
                ),
                run_config=ray_train.RunConfig(
                    name=experiment_name,
                    storage_path=str(root.resolve()),
                ),
            )
        results = tuner.fit()
        tuning_seconds = time.perf_counter() - started
        best = results.get_best_result(
            metric=self.metric,
            mode=self.mode,
            scope="all",
        )
        best_config = dict(best.config)
        trials = tuple(
            _trial_record(result, metric=self.metric)
            for result in results
        )
        outcome = TuningOutcome(best_config, trials, tuning_seconds)
        if artifact_store is not None:
            artifact_store.write_best_config(best_config)
            artifact_store.write_tuning_trials(trials)
            if artifact_store.paths.manifest.is_file():
                artifact_store.update_manifest(
                    timings={"tuning_seconds": tuning_seconds}
                )
        return outcome


def _trial_record(result: Any, *, metric: str) -> dict[str, Any]:
    metrics = dict(result.metrics or {})
    record: dict[str, Any] = {
        "trial_id": metrics.get("trial_id", Path(str(result.path)).name),
        "status": "error" if result.error is not None else "completed",
        "metric": metric,
        "value": metrics.get(metric),
        "training_iteration": metrics.get("training_iteration"),
    }
    record.update(
        {
            f"config.{name}": value
            for name, value in sorted(dict(result.config).items())
        }
    )
    if result.error is not None:
        record["error"] = str(result.error)
    return record
