"""One lifecycle for tuning, training, evaluation, and run artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, build_manifest
from .tuning import Tuner, TuningOutcome
from .types import EvaluationOutcome, RunContext, Trainer, TrainingOutcome

Evaluator = Callable[[Trainer, Any, RunContext], EvaluationOutcome]

_OWN_RUN_SOURCES = frozenset({"tuning", "best"})


def source_tuning_seconds(train_config_source: str) -> float:
    """Return the tuning time behind hyperparameters loaded from another run.

    ``train_config_source`` is the label from ``resolve_train_config``. Only a
    run directory carries a tuning time; this run's own tuning (``tuning``,
    ``best``, ``best:<run>``) and JSON files give 0. A source that was itself
    trained from a saved configuration is followed back to the run that tuned.
    """
    seen: set[Path] = set()
    source = train_config_source
    while source and source not in _OWN_RUN_SOURCES and not source.startswith("best:"):
        manifest_path = Path(source) / "manifest.json"
        if not manifest_path.is_file():
            break
        resolved = manifest_path.resolve()
        if resolved in seen:
            raise ValueError(f"train_config_source cycle through {source}")
        seen.add(resolved)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        seconds = float((manifest.get("timings") or {}).get("tuning_seconds") or 0.0)
        if seconds:
            return seconds
        source = str((manifest.get("provenance") or {}).get("train_config_source") or "")
    return 0.0


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """Scientific metadata shared by every model trained for one problem."""

    problem_id: str
    model_id: str
    benchmark_protocol_id: str
    dataset_fingerprints: Mapping[str, str | None]
    fields: Sequence[str]
    channels: Sequence[str] | Mapping[str, Sequence[str]]
    units: Mapping[str, str]
    coordinates: Sequence[str] | Mapping[str, Any]
    selected_test_case_ids: Sequence[str | int]
    tuning_budget: Mapping[str, Any] = field(default_factory=dict)
    project_root: str | Path | None = None

    def __post_init__(self) -> None:
        for name in ("problem_id", "model_id", "benchmark_protocol_id"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty.")
        if not {"train", "validation", "test"} <= set(
            self.dataset_fingerprints
        ):
            raise ValueError(
                "dataset_fingerprints must contain train, validation, and test."
            )
        if not self.fields:
            raise ValueError("fields must be non-empty.")
        object.__setattr__(self, "dataset_fingerprints", dict(self.dataset_fingerprints))
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "units", dict(self.units))
        object.__setattr__(self, "selected_test_case_ids", tuple(self.selected_test_case_ids))
        object.__setattr__(self, "tuning_budget", dict(self.tuning_budget))


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    context: RunContext
    training: TrainingOutcome
    evaluation: EvaluationOutcome


class ExperimentRunner:
    """Own model-independent experiment lifecycle and artifact mechanics."""

    def __init__(self, context: RunContext, spec: ExperimentSpec) -> None:
        root = context.paths.root_run_dir
        if root.parent.name != spec.model_id:
            raise ValueError("RunContext model path and ExperimentSpec.model_id differ.")
        if root.parent.parent.name != spec.problem_id:
            raise ValueError("RunContext problem path and ExperimentSpec.problem_id differ.")
        self.context = context
        self.spec = spec
        self.artifacts = ArtifactStore(context)

    def start(
        self,
        *,
        parameter_count: int = 0,
        timings: Mapping[str, float | None] | None = None,
        allow_completed: bool = False,
    ) -> None:
        """Create the manifest or reopen an unfinished (or, if allowed, completed) run."""
        if self.artifacts.paths.manifest.is_file():
            manifest = self.artifacts.read_manifest()
            for name in ("problem_id", "model_id", "benchmark_protocol_id"):
                if manifest[name] != getattr(self.spec, name):
                    raise ValueError(f"Existing manifest has a different {name}.")
            for name in ("dataset_fingerprints", "fields", "channels", "units", "coordinates", "selected_test_case_ids"):
                expected = json.loads(json.dumps(getattr(self.spec, name)))
                if manifest[name] != expected:
                    raise ValueError(f"Existing manifest has a different {name}.")
            if manifest["status"] == "completed" and not allow_completed:
                raise FileExistsError(
                    f"Completed run already exists at {self.context.paths.run_dir}."
                )
            self.artifacts.update_manifest(status="running", failure=None)
            return
        self.artifacts.write_manifest(
            build_manifest(
                self.context,
                problem_id=self.spec.problem_id,
                model_id=self.spec.model_id,
                benchmark_protocol_id=self.spec.benchmark_protocol_id,
                dataset_fingerprints=self.spec.dataset_fingerprints,
                fields=self.spec.fields,
                channels=self.spec.channels,
                units=self.spec.units,
                coordinates=self.spec.coordinates,
                selected_test_case_ids=self.spec.selected_test_case_ids,
                parameter_count=parameter_count,
                tuning_budget=self.spec.tuning_budget,
                project_root=self.spec.project_root,
                timings=timings,
            )
        )

    def tune(
        self,
        tuner: Tuner,
        train: Any,
        validation: Any,
        **kwargs: Any,
    ) -> TuningOutcome:
        self.start()
        try:
            return tuner.fit(
                train,
                validation,
                self.context,
                artifact_store=self.artifacts,
                **kwargs,
            )
        except BaseException as error:
            self.artifacts.mark_failed(error)
            raise

    def record_train_config_source(self, train_config_source: str) -> None:
        """Record where a tuning-free run's hyperparameters came from.

        The run inherits the source run's tuning time unless it already has
        one, so its manifest reports the full cost of its hyperparameters.
        """
        self.start()
        try:
            manifest = self.artifacts.read_manifest()
            self.artifacts.update_manifest(
                provenance={
                    **manifest.get("provenance", {}),
                    "train_config_source": train_config_source,
                },
            )
            if not manifest["timings"].get("tuning_seconds"):
                self.artifacts.update_manifest(
                    timings={"tuning_seconds": source_tuning_seconds(train_config_source)}
                )
        except BaseException as error:
            self.artifacts.mark_failed(error)
            raise

    def run(
        self,
        trainer: Trainer,
        train: Any,
        validation: Any,
        test: Any,
        *,
        config: Mapping[str, Any],
        evaluator: Evaluator,
        tuning: TuningOutcome | None = None,
        data_generation_seconds: float = 0.0,
    ) -> ExperimentResult:
        self.start(timings={"data_generation_seconds": data_generation_seconds})
        try:
            self.artifacts.write_best_config(config)
            if tuning is None:
                if not self.artifacts.paths.tuning_trials.is_file():
                    self.artifacts.write_tuning_trials(())
                tuning_seconds = float(self.artifacts.read_manifest()["timings"].get("tuning_seconds") or 0.0)
            else:
                self.artifacts.write_tuning_trials(tuning.trials)
                tuning_seconds = tuning.tuning_seconds
            training = trainer.fit(train, validation, self.context)
            self.artifacts.write_history(training.history)
            canonical = self.context.paths.checkpoint(training.checkpoint.suffix or ".pt")
            if training.checkpoint.resolve() != canonical.resolve():
                self.artifacts.copy_checkpoint(training.checkpoint)
            parameter_count = int(
                training.metadata.get("n_params", training.metadata.get("parameter_count", 0))
            )
            self.artifacts.update_manifest(
                parameter_count=parameter_count,
                timings={"tuning_seconds": tuning_seconds, **training.timings},
                training_metadata=dict(training.metadata),
            )
            evaluation = evaluator(trainer, test, self.context)
            self.artifacts.write_metrics(evaluation.metrics)
            self.artifacts.write_reconstructions(**evaluation.reconstructions)
            self.artifacts.update_manifest(evaluation_metadata=dict(evaluation.metadata))
            self.artifacts.mark_completed(
                timings={
                    "data_generation_seconds": data_generation_seconds,
                    "tuning_seconds": tuning_seconds,
                    "final_training_seconds": float(training.timings.get("final_training_seconds", 0.0)),
                    "inference_seconds": evaluation.inference_seconds,
                }
            )
            return ExperimentResult(self.context, training, evaluation)
        except BaseException as error:
            self.artifacts.mark_failed(error)
            raise

    def run_seeds(
        self,
        seeds: Sequence[int],
        train_one: SeedTrainer,
        *,
        config: Mapping[str, Any],
        tuning: TuningOutcome | None = None,
        train_config_source: str = "best",
    ) -> tuple[ExperimentResult, ...]:
        """Train ``config`` once per seed into ``<run>/seeds/<seed>/``.

        The parent run keeps tuning artifacts and records the child seeds in
        its manifest. Completed children are skipped so an interrupted or
        extended run can be resumed with the same ``--run-id``.
        """
        if self.context.paths.is_seed_child:
            raise ValueError("run_seeds must be called on the parent run.")
        requested = tuple(int(seed) for seed in seeds)
        if not requested or len(set(requested)) != len(requested):
            raise ValueError("seeds must be non-empty and unique.")
        self.start(allow_completed=True)
        try:
            manifest = self.artifacts.read_manifest()
            self.artifacts.update_manifest(
                tuning_seed=self.context.seed,
                provenance={
                    **manifest.get("provenance", {}),
                    "train_config_source": train_config_source,
                },
            )
            self.artifacts.write_best_config(config)
            if tuning is None:
                if not self.artifacts.paths.tuning_trials.is_file():
                    self.artifacts.write_tuning_trials(())
                tuning_seconds = float(
                    manifest["timings"].get("tuning_seconds") or 0.0
                ) or source_tuning_seconds(train_config_source)
            else:
                self.artifacts.write_tuning_trials(tuning.trials)
                tuning_seconds = tuning.tuning_seconds
            results: list[ExperimentResult] = []
            for seed in requested:
                child = self.context.seed_child(seed)
                child_store = ArtifactStore(child)
                if child_store.paths.manifest.is_file() and child_store.read_manifest()["status"] == "completed":
                    print(f"Seed {seed}: completed run exists at {child.paths.run_dir}; skipping.")
                    continue
                child_runner = ExperimentRunner(child, self.spec)
                results.append(train_one(child_runner, child, dict(config, seed=seed)))
            completed = self.completed_seeds()
            missing = [seed for seed in requested if seed not in completed]
            if missing:
                raise RuntimeError(f"Seeds {missing} did not produce completed runs.")
            training_seconds = 0.0
            inference_seconds = 0.0
            for seed in completed:
                timings = ArtifactStore(self.context.seed_child(seed)).read_manifest()["timings"]
                training_seconds += float(timings.get("final_training_seconds") or 0.0)
                inference_seconds += float(timings.get("inference_seconds") or 0.0)
            self.artifacts.update_manifest(seeds=list(completed))
            self.artifacts.mark_completed(
                timings={
                    "data_generation_seconds": float(manifest["timings"].get("data_generation_seconds") or 0.0),
                    "tuning_seconds": tuning_seconds,
                    "final_training_seconds": training_seconds,
                    "inference_seconds": inference_seconds,
                }
            )
            return tuple(results)
        except BaseException as error:
            self.artifacts.mark_failed(error)
            raise

    def completed_seeds(self) -> tuple[int, ...]:
        """Return the sorted seeds with a completed child run on disk."""
        seeds_dir = self.context.paths.seeds_dir
        if not seeds_dir.is_dir():
            return ()
        completed = []
        for child in seeds_dir.iterdir():
            if not child.is_dir() or not child.name.isdigit():
                continue
            store = ArtifactStore(child)
            if store.paths.manifest.is_file() and store.read_manifest()["status"] == "completed":
                completed.append(int(child.name))
        return tuple(sorted(completed))


SeedTrainer = Callable[[ExperimentRunner, RunContext, dict[str, Any]], ExperimentResult]

__all__ = ["Evaluator", "ExperimentResult", "ExperimentRunner", "ExperimentSpec", "SeedTrainer"]
