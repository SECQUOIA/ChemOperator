"""Contract tests for reusable, model-agnostic experiment infrastructure."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from chem_operator.experiments import (
    ArtifactStore,
    ArtifactValidationError,
    EvaluationOutcome,
    ExperimentRunner,
    ExperimentSpec,
    FNOTrainer,
    LossTerm,
    MetricEvent,
    RunContext,
    StreamingRegressionMetrics,
    Trainer,
    WorkflowStages,
    build_manifest,
    comparison_records,
    load_run,
    metric_matrix,
    shared_history,
    validate_model_comparison,
)


def context(tmp_path: Path, model: str = "tiny") -> RunContext:
    return RunContext.create(
        tmp_path,
        problem="synthetic",
        model=model,
        run_id="run-1",
        seed=11,
        device="cpu",
    )


def manifest(run_context: RunContext, tmp_path: Path) -> dict[str, object]:
    return build_manifest(
        run_context,
        problem_id="synthetic",
        model_id=run_context.paths.run_dir.parent.name,
        benchmark_protocol_id="test-v1",
        dataset_fingerprints={
            "train": "sha256:train",
            "validation": "sha256:validation",
            "test": "sha256:test",
        },
        fields=("temperature",),
        channels=("temperature",),
        units={"temperature": "K"},
        coordinates=("time",),
        selected_test_case_ids=(0,),
        parameter_count=2,
        tuning_budget={"samples": 1, "epochs": 1},
        project_root=tmp_path,
    )


def complete_run(tmp_path: Path, model: str) -> Path:
    run_context = context(tmp_path, model)
    store = ArtifactStore(run_context)
    store.write_manifest(manifest(run_context, tmp_path))
    store.write_best_config({"width": 4})
    store.write_history(
        (
            MetricEvent(1, "train", "objective", 2.0),
            MetricEvent(1, "val", "relative_l2", 0.5),
        )
    )
    store.write_metrics({"relative_l2": 0.4})
    store.write_tuning_trials(())
    store.write_reconstructions(
        case_ids=np.asarray((0,)),
        reference=np.ones((1, 2)),
        prediction=np.ones((1, 2)),
    )
    store.write_torch_checkpoint({"state_dict": {"weight": torch.ones(1)}})
    store.mark_completed(
        timings={
            "data_generation_seconds": 0.0,
            "tuning_seconds": 0.0,
            "final_training_seconds": 0.1,
            "inference_seconds": 0.01,
        }
    )
    return run_context.paths.run_dir


def test_artifact_store_writes_complete_atomic_contract(tmp_path: Path) -> None:
    run = complete_run(tmp_path, "fno")
    assert {path.name for path in run.iterdir()} == {
        "manifest.json",
        "best_config.json",
        "history.csv",
        "metrics.csv",
        "checkpoint.pt",
        "reconstructions.npz",
        "tuning_trials.csv",
    }
    with (run / "history.csv").open(encoding="utf-8", newline="") as stream:
        assert tuple(csv.DictReader(stream).fieldnames or ()) == (
            "epoch",
            "split",
            "metric",
            "value",
        )
    with (run / "manifest.json").open(encoding="utf-8") as stream:
        assert json.load(stream)["status"] == "completed"


def test_artifact_store_rejects_incomplete_completion(tmp_path: Path) -> None:
    run_context = context(tmp_path)
    store = ArtifactStore(run_context)
    store.write_manifest(manifest(run_context, tmp_path))
    with pytest.raises(ArtifactValidationError, match="missing artifacts"):
        store.mark_completed()


def test_best_config_fails_clearly_when_tuning_has_not_written_it(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Run or resume tuning"):
        ArtifactStore(tmp_path / "missing").read_best_config()


def test_comparison_selects_shared_metric_and_rejects_objective(
    tmp_path: Path,
) -> None:
    first = complete_run(tmp_path / "a", "deeponet")
    second = complete_run(tmp_path / "b", "fno")
    series = shared_history((load_run(first), load_run(second)))
    assert [values.events[0].value for values in series] == [0.5, 0.5]
    with pytest.raises(ValueError, match="model-specific"):
        shared_history((first, second), metric="objective", split="train")


def test_streaming_metrics_match_materialized_values() -> None:
    state = StreamingRegressionMetrics()
    state.update(torch.tensor((1.0, 3.0)), torch.tensor((1.0, 1.0)))
    state.update(np.asarray((2.0,)), np.asarray((1.0,)))
    assert state.compute() == pytest.approx(
        {
            "rmse": np.sqrt(5.0 / 3.0),
            "relative_l2": np.sqrt(5.0 / 3.0),
            "max_absolute_error": 2.0,
        }
    )


class TinyFields(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = torch.full((1, 2, 2), float(index + 1))
        return {"x": value, "y": 2.0 * value}


def test_fno_trainer_implements_contract_and_verifies_checkpoint(
    tmp_path: Path,
) -> None:
    trainer = FNOTrainer(
        lambda config: torch.nn.Conv2d(1, 1, kernel_size=1),
        {
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1.0e-2,
        },
    )
    assert isinstance(trainer, Trainer)
    run_context = context(tmp_path, "fno")
    outcome = trainer.fit(TinyFields(), TinyFields(), run_context)
    assert outcome.metadata["checkpoint_reload_verified"] is True
    assert outcome.checkpoint.name == "checkpoint.pt"
    assert {event.metric for event in outcome.history} >= {
        "objective",
        "relative_l2",
    }
    assert trainer.predict(TinyFields(), run_context).shape == (4, 1, 2, 2)


def test_workflow_stages_reject_invalid_plot_count() -> None:
    with pytest.raises(ValueError, match="plot_cases"):
        WorkflowStages(plot_cases=0)


def test_experiment_runner_publishes_complete_run(tmp_path: Path) -> None:
    run_context = context(tmp_path, "fno")
    spec = ExperimentSpec(
        problem_id="synthetic",
        model_id="fno",
        benchmark_protocol_id="test-v1",
        dataset_fingerprints={
            "train": "sha256:train",
            "validation": "sha256:validation",
            "test": "sha256:test",
        },
        fields=("temperature",),
        channels=("temperature",),
        units={"temperature": "K"},
        coordinates=("x", "y"),
        selected_test_case_ids=(0,),
        tuning_budget={"samples": 0, "epochs": 1},
        project_root=tmp_path,
    )
    trainer = FNOTrainer(
        lambda config: torch.nn.Conv2d(1, 1, kernel_size=1),
        {"epochs": 1, "batch_size": 2, "learning_rate": 1.0e-2},
    )

    def evaluate(trained, data, evaluation_context):
        started = 0.0
        prediction = trained.predict(data, evaluation_context).numpy()
        reference = np.stack([data[index]["y"].numpy() for index in range(len(data))])
        error = prediction - reference
        return EvaluationOutcome(
            metrics={
                "relative_l2": float(
                    np.linalg.norm(error) / np.linalg.norm(reference)
                )
            },
            reconstructions={
                "case_ids": np.asarray((0,)),
                "reference": reference[:1],
                "prediction": prediction[:1],
            },
            inference_seconds=started,
        )

    runner = ExperimentRunner(run_context, spec)
    runner.start(timings={"tuning_seconds": 2.5})
    from dataclasses import replace
    changed = replace(spec, dataset_fingerprints={**spec.dataset_fingerprints, "train": "sha256:changed"})
    with pytest.raises(ValueError, match="dataset_fingerprints"):
        ExperimentRunner(run_context, changed).start()
    result = runner.run(
        trainer,
        TinyFields(),
        TinyFields(),
        TinyFields(),
        config=trainer.config,
        evaluator=evaluate,
    )
    ArtifactStore(result.context).validate_complete()
    loaded = load_run(result.context.paths.run_dir)
    assert loaded.manifest["timings"]["tuning_seconds"] == 2.5
    assert loaded.manifest["parameter_count"] == 2
    assert loaded.metrics[0]["metric"] == "relative_l2"


def test_fno_trainer_supports_multi_input_mapping_outputs(tmp_path: Path) -> None:
    class PairFields(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int):
            first = torch.full((1, 2), float(index + 1))
            second = torch.full((1, 2), 2.0)
            return {"first": first, "second": second, "target": first + second}

    class PairModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.ones(()))

        def forward(self, first, second):
            value = self.scale * (first + second)
            return {"sum": value}

    def adapt(batch, device, dtype):
        moved = {
            name: value.to(device=device, dtype=dtype)
            for name, value in batch.items()
        }
        return (
            (moved["first"], moved["second"]),
            {"sum": moved["target"]},
            moved,
        )

    def loss(prediction, target, batch):
        del batch
        return torch.nn.functional.mse_loss(prediction["sum"], target["sum"])

    def metrics(prediction, target, batch):
        del batch
        return prediction["sum"], target["sum"]

    trainer = FNOTrainer(
        lambda config: PairModel(),
        {"epochs": 1, "batch_size": 2, "learning_rate": 1.0e-2},
        loss_terms=(LossTerm("data", loss),),
        batch_adapter=adapt,
        metric_adapter=metrics,
    )
    run_context = context(tmp_path, "pair_fno")
    outcome = trainer.fit(PairFields(), PairFields(), run_context)
    assert outcome.metrics["validation_relative_l2"] >= 0
    prediction = trainer.predict(PairFields(), run_context)
    assert prediction["sum"].shape == (4, 1, 2)


def test_comparison_records_form_problem_by_model_matrix(tmp_path: Path) -> None:
    direct = complete_run(tmp_path / "direct", "deeponet")
    fno = complete_run(tmp_path / "fno", "fno")
    records = comparison_records((direct, fno), metric="relative_l2")
    assert {(record.problem, record.model) for record in records} == {
        ("synthetic", "deeponet"),
        ("synthetic", "fno"),
    }
    assert metric_matrix((direct, fno)) == {
        "synthetic": {"deeponet": 0.4, "fno": 0.4}
    }
    assert len(validate_model_comparison((direct, fno))) == 2
