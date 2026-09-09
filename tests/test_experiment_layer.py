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
    FNOTrainer,
    MetricEvent,
    RunContext,
    StreamingRegressionMetrics,
    Trainer,
    WorkflowStages,
    build_manifest,
    load_run,
    shared_history,
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
