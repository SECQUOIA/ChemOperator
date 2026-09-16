"""One lifecycle for tuning, training, evaluation, and run artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import ArtifactStore, build_manifest
from .tuning import Tuner, TuningOutcome
from .types import EvaluationOutcome, RunContext, Trainer, TrainingOutcome

Evaluator = Callable[[Trainer, Any, RunContext], EvaluationOutcome]


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
        if context.paths.run_dir.parent.name != spec.model_id:
            raise ValueError("RunContext model path and ExperimentSpec.model_id differ.")
        if context.paths.run_dir.parent.parent.name != spec.problem_id:
            raise ValueError("RunContext problem path and ExperimentSpec.problem_id differ.")
        self.context = context
        self.spec = spec
        self.artifacts = ArtifactStore(context)

    def start(
        self,
        *,
        parameter_count: int = 0,
        timings: Mapping[str, float | None] | None = None,
    ) -> None:
        if self.artifacts.paths.manifest.is_file():
            manifest = self.artifacts.read_manifest()
            for name in ("problem_id", "model_id", "benchmark_protocol_id"):
                if manifest[name] != getattr(self.spec, name):
                    raise ValueError(f"Existing manifest has a different {name}.")
            if manifest["status"] == "completed":
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
                tuning_seconds = 0.0
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


__all__ = ["Evaluator", "ExperimentResult", "ExperimentRunner", "ExperimentSpec"]
