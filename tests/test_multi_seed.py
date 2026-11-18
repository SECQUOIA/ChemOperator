"""Multi-seed training runs: CLI stages, nested layout, resume, and config sources."""

from __future__ import annotations

import argparse
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
    OperatorTrainer,
    RunContext,
    WorkflowStages,
    add_run_arguments,
    add_workflow_arguments,
    load_run,
    resolve_train_config,
    run_context_from_namespace,
    train_final,
)


class TinyFields(Dataset):
    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        value = torch.full((1, 2, 2), float(index + 1))
        return {"x": value, "y": 2.0 * value}


def spec(tmp_path: Path, model: str = "fno") -> ExperimentSpec:
    return ExperimentSpec(
        problem_id="synthetic",
        model_id=model,
        benchmark_protocol_id="test-v1",
        dataset_fingerprints={"train": "a", "validation": "b", "test": "c"},
        fields=("temperature",),
        channels=("temperature",),
        units={"temperature": "K"},
        coordinates=("x", "y"),
        selected_test_case_ids=(0,),
        tuning_budget={"samples": 0, "epochs": 1},
        project_root=tmp_path,
    )


def evaluate(trained, data, evaluation_context):
    prediction = trained.predict(data, evaluation_context).numpy()
    reference = np.stack([data[index]["y"].numpy() for index in range(len(data))])
    error = float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))
    return EvaluationOutcome(
        metrics={"relative_l2": error},
        reconstructions={
            "case_ids": np.asarray((0,)),
            "reference": reference[:1],
            "prediction": prediction[:1],
        },
        inference_seconds=0.5,
    )


def train_one(run_runner, run_context, config):
    assert config["seed"] == run_context.seed
    trainer = OperatorTrainer(
        lambda _config: torch.nn.Conv2d(1, 1, kernel_size=1),
        {"epochs": 1, "batch_size": 2, "learning_rate": 1.0e-2, **config},
    )
    return run_runner.run(
        trainer, TinyFields(), TinyFields(), TinyFields(),
        config=trainer.config, evaluator=evaluate,
    )


def parser() -> argparse.ArgumentParser:
    built = argparse.ArgumentParser()
    add_workflow_arguments(built)
    add_run_arguments(built)
    return built


def test_workflow_stages_validate_seed_options() -> None:
    args = parser().parse_args(["--seeds", "3", "1", "2", "--tune-seed", "9"])
    stages = WorkflowStages.from_namespace(args)
    assert stages.seeds == (3, 1, 2)
    assert stages.tune_seed == 9
    assert WorkflowStages.from_namespace(parser().parse_args([])).seeds is None
    with pytest.raises(ValueError, match="requires --train"):
        WorkflowStages.from_namespace(parser().parse_args(["--no-train", "--seeds", "1"]))
    with pytest.raises(ValueError, match="unique"):
        WorkflowStages(seeds=(1, 1))
    with pytest.raises(ValueError, match="negative"):
        WorkflowStages(seeds=(-1,))
    with pytest.raises(ValueError, match="negative"):
        WorkflowStages(tune_seed=-2)


def test_tune_seed_overrides_root_context_seed(tmp_path: Path) -> None:
    args = parser().parse_args(["--runs-root", str(tmp_path), "--run-id", "r", "--device", "cpu"])
    assert run_context_from_namespace(args, problem="p", model="m", seed=7).seed == 7
    args = parser().parse_args(
        ["--runs-root", str(tmp_path), "--run-id", "r", "--device", "cpu", "--tune-seed", "5"]
    )
    assert run_context_from_namespace(args, problem="p", model="m", seed=7).seed == 5


def test_seed_child_context_paths_and_provenance(tmp_path: Path) -> None:
    root = RunContext.create(tmp_path, problem="p", model="m", run_id="r", seed=1, device="cpu")
    child = root.seed_child(42)
    assert child.paths.run_dir == tmp_path / "p" / "m" / "r" / "seeds" / "42"
    assert child.seed == 42
    assert child.paths.is_seed_child and not root.paths.is_seed_child
    assert child.paths.root_run_dir == root.paths.run_dir
    assert child.provenance == {"parent_run_id": "r", "seed": 42}
    with pytest.raises(ValueError, match="nested"):
        child.seed_child(1)


def test_run_seeds_writes_nested_children_and_parent_summary(tmp_path: Path) -> None:
    root = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="multi", seed=1, device="cpu")
    runner = ExperimentRunner(root, spec(tmp_path))
    stages = WorkflowStages(tune=False, seeds=(3, 1))
    results = train_final(
        runner, stages, config={"width": 4, "seed": 1}, tuning=None,
        train_one=train_one, train_config_source="unit-test",
    )
    assert [result.context.seed for result in results] == [3, 1]
    parent = runner.artifacts.read_manifest()
    assert parent["status"] == "completed"
    assert parent["seeds"] == [1, 3]
    assert parent["tuning_seed"] == 1
    assert parent["random_seed"] == 1
    assert parent["provenance"]["train_config_source"] == "unit-test"
    assert parent["timings"]["inference_seconds"] == pytest.approx(1.0)
    assert not (root.paths.run_dir / "metrics.csv").exists()
    runner.artifacts.validate_complete()
    for seed in (1, 3):
        child = ArtifactStore(root.seed_child(seed))
        manifest = child.read_manifest()
        assert manifest["random_seed"] == seed
        assert manifest["provenance"]["parent_run_id"] == "multi"
        assert child.read_best_config()["seed"] == seed
        child.validate_complete()
    assert runner.artifacts.read_best_config() == {"width": 4, "seed": 1}
    loaded = load_run(root.paths.run_dir)
    assert loaded.seeds == (1, 3)
    assert loaded.representative.path == root.seed_child(1).paths.run_dir


def test_run_seeds_resumes_by_skipping_completed_children(tmp_path: Path) -> None:
    root = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="multi", seed=1, device="cpu")
    runner = ExperimentRunner(root, spec(tmp_path))
    runner.run_seeds((1,), train_one, config={"seed": 1})
    first_stamp = ArtifactStore(root.seed_child(1)).read_manifest()["completed_at"]
    results = runner.run_seeds((1, 2), train_one, config={"seed": 1})
    assert [result.context.seed for result in results] == [2]
    assert ArtifactStore(root.seed_child(1)).read_manifest()["completed_at"] == first_stamp
    assert runner.artifacts.read_manifest()["seeds"] == [1, 2]
    with pytest.raises(FileExistsError):
        runner.start()


@pytest.mark.parametrize("parent_failed", [False, True], ids=["completed", "interrupted"])
@pytest.mark.parametrize(
    "original, changed, differing",
    [
        ({"width": 4}, {"width": 8}, "width"),
        ({"weight_decay": 0.2}, {}, "weight_decay"),
        ({}, {"weight_decay": 0.2}, "weight_decay"),
    ],
    ids=["changed-value", "removed-key", "added-key"],
)
def test_run_seeds_rejects_resume_with_different_config(
    tmp_path: Path, original: dict, changed: dict, differing: str, parent_failed: bool,
) -> None:
    root = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="multi", seed=1, device="cpu")
    runner = ExperimentRunner(root, spec(tmp_path))
    runner.run_seeds((1,), train_one, config={"seed": 1, **original})
    if parent_failed:
        runner.artifacts.mark_failed(RuntimeError("interrupted after seed 1"))
    before = {path: path.read_bytes() for path in root.paths.run_dir.rglob("*") if path.is_file()}
    with pytest.raises(ValueError, match=f"different.*{differing}"):
        runner.run_seeds((1, 2), train_one, config={"seed": 1, **changed})
    assert {path: path.read_bytes() for path in root.paths.run_dir.rglob("*") if path.is_file()} == before
    assert not root.seed_child(2).paths.run_dir.exists()
    assert load_run(root.paths.run_dir, require_complete=not parent_failed).seeds == (1,)
    # Per-seed keys may change on resume.
    runner.run_seeds((1, 2), train_one, config={**original, "seed": 9, "epochs": 2})
    assert load_run(root.paths.run_dir).seeds == (1, 2)


def test_run_seeds_rejects_config_unlike_completed_flat_run(tmp_path: Path) -> None:
    root = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="flat", seed=42, device="cpu")
    runner = ExperimentRunner(root, spec(tmp_path))
    train_final(runner, WorkflowStages(tune=False), config={"seed": 42, "width": 4},
                tuning=None, train_one=train_one)
    with pytest.raises(ValueError, match="existing run was trained with different width"):
        train_final(runner, WorkflowStages(tune=False, seeds=(43,)), config={"seed": 42, "width": 8},
                    tuning=None, train_one=train_one)
    assert runner.artifacts.read_manifest()["status"] == "completed"
    assert runner.artifacts.read_best_config()["width"] == 4


def test_run_operator_passes_tune_seed_to_optuna(tmp_path: Path, monkeypatch) -> None:
    from types import SimpleNamespace

    from chem_operator._experiments import operator_workflow

    class Captured(Exception):
        pass

    def fake_tuner(*args, config, **kwargs):
        raise Captured(config)

    class Raw:
        def close(self) -> None:
            pass

    paths = SimpleNamespace(root=tmp_path, data=tmp_path, ray=tmp_path / "ray")
    monkeypatch.setattr(operator_workflow, "Tuner", fake_tuner)
    monkeypatch.setattr("sys.argv", ["fno.py", "--no-train", "--tune-seed", "19", "--device", "cpu",
                                     "--runs-root", str(tmp_path / "runs"), "--run-id", "r"])
    args = operator_workflow.parse_operator_args(paths, epochs=1, tune_epochs=1, samples=1)
    with pytest.raises(Captured) as captured:
        operator_workflow.run_operator(
            args, paths, spec(tmp_path), lambda split: (Raw(), TinyFields()),
            trainer_factory=None, search_space={}, evaluator=evaluate, seed=42,
        )
    assert captured.value.args[0].optuna_seed == 19


def test_run_seeds_marks_child_and_parent_failed(tmp_path: Path) -> None:
    root = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="multi", seed=1, device="cpu")
    runner = ExperimentRunner(root, spec(tmp_path))

    def explode(run_runner, run_context, config):
        run_runner.start()
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        runner.run_seeds((5,), explode, config={"seed": 1})
    assert runner.artifacts.read_manifest()["status"] == "failed"
    child = ArtifactStore(root.seed_child(5)).read_manifest()
    assert child["status"] == "running"  # only the runner's own run() marks children
    with pytest.raises(ArtifactValidationError):
        runner.artifacts.validate_complete()


def test_flat_run_adds_seeds_to_existing_completed_root(tmp_path: Path) -> None:
    root = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="flat", seed=42, device="cpu")
    runner = ExperimentRunner(root, spec(tmp_path))
    train_final(runner, WorkflowStages(tune=False), config={"seed": 42}, tuning=None, train_one=train_one)
    assert (root.paths.run_dir / "metrics.csv").exists()
    with pytest.raises(FileExistsError):
        train_final(runner, WorkflowStages(tune=False), config={"seed": 42}, tuning=None, train_one=train_one)
    train_final(runner, WorkflowStages(tune=False, seeds=(43,)), config={"seed": 42}, tuning=None, train_one=train_one)
    manifest = runner.artifacts.read_manifest()
    assert manifest["seeds"] == [43]
    assert manifest["random_seed"] == 42
    runner.artifacts.validate_complete()
    loaded = load_run(root.paths.run_dir)
    assert loaded.seeds == (42, 43)
    assert loaded.representative.path == root.paths.run_dir


def test_resolve_train_config_sources(tmp_path: Path) -> None:
    root = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="src", seed=1, device="cpu")
    runner = ExperimentRunner(root, spec(tmp_path))
    with pytest.raises(FileNotFoundError, match="Run or resume tuning"):
        resolve_train_config("best", runner)
    train_final(runner, WorkflowStages(tune=False), config={"width": 8, "seed": 1}, tuning=None, train_one=train_one)
    assert resolve_train_config("best", runner)[0]["width"] == 8

    other = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="new", seed=1, device="cpu")
    other_runner = ExperimentRunner(other, spec(tmp_path))
    config, source = resolve_train_config(str(root.paths.run_dir), other_runner)
    assert config["width"] == 8 and source == str(root.paths.run_dir)

    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"width": 16, "seed": 0}), encoding="utf-8")
    assert resolve_train_config(str(json_path), other_runner)[0] == {"width": 16, "seed": 0}

    incomplete = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="half", seed=1, device="cpu")
    ExperimentRunner(incomplete, spec(tmp_path)).start()
    with pytest.raises(ValueError, match="not a completed run"):
        resolve_train_config(str(incomplete.paths.run_dir), other_runner)
    with pytest.raises(FileNotFoundError, match="neither"):
        resolve_train_config(str(tmp_path / "missing.json"), other_runner)


# --- Read side: member discovery, aggregation, plotting, and selection ----------

from chem_operator.experiments import (  # noqa: E402
    MetricEvent,
    RunPaths,
    aggregate_history,
    aggregate_metrics,
    build_manifest,
    comparison_records,
    final_metric_summaries,
    final_metrics,
    metric_matrix,
    validate_complete_run,
)
from chem_operator.plotting import (  # noqa: E402
    add_plot_run_arguments,
    plot_experiment_histories,
    plot_selected_runs,
    select_plot_runs,
)


def _write_member(run_dir: Path, model: str, *, seed: int, stamp: str, epochs: int = 3,
                  error: float = 0.4, status: str = "completed", protocol: str = "shared",
                  tuning_files: bool = True, root: Path | None = None) -> Path:
    context = RunContext(paths=RunPaths(run_dir), seed=seed, dtype=torch.float32, device="cpu")
    store = ArtifactStore(context)
    manifest = build_manifest(
        context, problem_id="synthetic", model_id=model, benchmark_protocol_id=protocol,
        dataset_fingerprints={"train": "a", "validation": "b", "test": "c"},
        fields=("T",), channels=("T",), units={"T": "K"}, coordinates=("t",),
        selected_test_case_ids=(1,), parameter_count=3, tuning_budget={},
        project_root=root or run_dir,
        timings={"tuning_seconds": 0.0, "final_training_seconds": 2.0 * seed,
                 "inference_seconds": 0.1, "data_generation_seconds": 0.0},
    )
    manifest.update(status=status, created_at=stamp, completed_at=stamp,
                    training_metadata={"config": {"epochs": epochs}})
    store.write_manifest(manifest)
    if tuning_files:
        store.write_best_config({"width": 1, "seed": seed})
        store.write_tuning_trials(())
    store.write_history(
        [MetricEvent(epoch, "val", "relative_l2", error * (1.0 + 0.1 * epoch)) for epoch in range(1, epochs + 1)]
        + [MetricEvent(epoch, "train", "objective", 1.0 / epoch) for epoch in range(1, epochs + 1)]
    )
    store.write_metrics({"relative_l2": error, "rmse": 2 * error})
    store.write_reconstructions(
        case_ids=np.array([1]), labels=np.array(["T"]),
        coordinates=np.array([[[0.0], [1.0]]]),
        reference=np.array([[[1.0], [2.0]]]),
        prediction=np.array([[[1.0 + error], [2.0 + error]]]),
    )
    store.write_torch_checkpoint({"state_dict": {}})
    return run_dir


def _seed_group(root: Path, model: str, run_id: str, *, stamp: str,
                seeds=(1, 2, 3), epochs=(3, 2, 3), errors=(0.4, 0.5, 0.6),
                root_seed: int | None = None, listed=None) -> Path:
    run_dir = root / "synthetic" / model / run_id
    if root_seed is not None:
        _write_member(run_dir, model, seed=root_seed, stamp=stamp, root=root)
    else:
        context = RunContext(paths=RunPaths(run_dir), seed=0, dtype=torch.float32, device="cpu")
        store = ArtifactStore(context)
        manifest = build_manifest(
            context, problem_id="synthetic", model_id=model, benchmark_protocol_id="shared",
            dataset_fingerprints={"train": "a", "validation": "b", "test": "c"},
            fields=("T",), channels=("T",), units={"T": "K"}, coordinates=("t",),
            selected_test_case_ids=(1,), parameter_count=0, tuning_budget={}, project_root=root,
            timings={"tuning_seconds": 5.0, "final_training_seconds": 0.0,
                     "inference_seconds": 0.0, "data_generation_seconds": 0.0},
        )
        manifest.update(status="completed", created_at=stamp, completed_at=stamp)
        store.write_manifest(manifest)
        store.write_best_config({"width": 1, "seed": 0})
        store.write_tuning_trials(())
    for seed, count, error in zip(seeds, epochs, errors):
        _write_member(run_dir / "seeds" / str(seed), model, seed=seed, stamp=stamp,
                      epochs=count, error=error, root=root)
    ArtifactStore(run_dir).update_manifest(
        seeds=list(seeds) if listed is None else list(listed), tuning_seed=0,
    )
    return run_dir


def test_load_run_discovers_members_and_aggregates(tmp_path: Path) -> None:
    parent = _seed_group(tmp_path, "model_a", "group", stamp="2025-02-01T00:00:00+00:00")
    run = load_run(parent)
    assert run.path == parent
    assert run.seeds == (1, 2, 3)
    assert run.representative.path == parent / "seeds" / "1"
    assert run.history is run.representative.history
    assert load_run(parent / "seeds" / "2").members == ()
    validate_complete_run(run)

    series = aggregate_history(run.seed_members, metric="relative_l2", split="validation")
    assert series.epochs == (1, 2)  # member 2 stopped after two epochs
    assert series.count == 3
    assert series.mean[0] == pytest.approx(0.5 * 1.1)
    assert series.std[0] == pytest.approx(0.1 * 1.1)
    assert series.minimum[0] == pytest.approx(0.4 * 1.1)
    with pytest.raises(ValueError, match="no val rmse history"):
        aggregate_history(run.seed_members, metric="rmse", split="val")

    summary = aggregate_metrics(run.seed_members)["relative_l2"]
    assert (summary.mean, summary.std, summary.count) == (pytest.approx(0.5), pytest.approx(0.1), 3)
    assert final_metrics((run,)) == {run.label: pytest.approx(0.5)}
    assert final_metric_summaries((run,))[run.label].count == 3
    records = comparison_records((run,), metric="relative_l2")
    assert [(record.run_id, record.seed, record.value) for record in records] == [
        ("group", 1, 0.4), ("group", 2, 0.5), ("group", 3, 0.6),
    ]
    assert metric_matrix((run,)) == {"synthetic": {"model_a": pytest.approx(0.5)}}
    other = _seed_group(tmp_path, "model_a", "group2", stamp="2025-02-02T00:00:00+00:00")
    with pytest.raises(ValueError, match="select one run"):
        metric_matrix((run, load_run(other)))


def test_load_run_includes_root_member_and_validates_children(tmp_path: Path) -> None:
    parent = _seed_group(tmp_path, "model_a", "resumed", stamp="2025-02-01T00:00:00+00:00",
                         root_seed=42, seeds=(43,), epochs=(3,), errors=(0.2,))
    run = load_run(parent)
    assert run.seeds == (42, 43)
    assert run.representative.path == parent
    assert aggregate_metrics(run.seed_members)["relative_l2"].count == 2

    listed = _seed_group(tmp_path, "model_a", "missing", stamp="2025-02-01T00:00:00+00:00",
                         seeds=(1, 2), epochs=(2, 2), errors=(0.1, 0.2), listed=(1, 2, 3))
    with pytest.raises(ValueError, match=r"lists seeds \[3\]"):
        load_run(listed)
    assert load_run(listed, require_complete=False).seeds == (1, 2)

    mismatched = _seed_group(tmp_path, "model_a", "bad", stamp="2025-02-01T00:00:00+00:00",
                             seeds=(1,), epochs=(2,), errors=(0.1,))
    ArtifactStore(mismatched / "seeds" / "1").update_manifest(benchmark_protocol_id="other")
    with pytest.raises(ValueError, match="mismatched benchmark_protocol_id"):
        load_run(mismatched)

    broken = _seed_group(tmp_path, "model_a", "broken", stamp="2025-02-01T00:00:00+00:00",
                         seeds=(1, 2), epochs=(2, 2), errors=(0.1, 0.2))
    (broken / "seeds" / "2" / "reconstructions.npz").unlink()
    with pytest.raises(ArtifactValidationError, match="seeds/2/reconstructions.npz"):
        validate_complete_run(broken)


def _plot_parser(root: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_plot_run_arguments(parser, root=root, problem_id="synthetic", cases=1)
    return parser


def test_recent_treats_seed_group_as_one_run(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    _write_member(root / "synthetic" / "model_a" / "flat", "model_a", seed=1,
                  stamp="2025-01-01T00:00:00+00:00", root=root)
    group = _seed_group(root, "model_a", "group", stamp="2025-02-01T00:00:00+00:00")
    incomplete = _seed_group(root, "model_a", "incomplete", stamp="2025-03-01T00:00:00+00:00",
                             seeds=(1, 2), epochs=(2, 2), errors=(0.1, 0.2), listed=(1, 2, 3))
    parser = _plot_parser(root)
    selected = select_plot_runs(parser.parse_args(["--recent", "--runs-root", str(root)]),
                                parser, problem_id="synthetic")
    assert [run.path for run in selected] == [group]
    assert selected[0].seeds == (1, 2, 3)
    assert incomplete not in {run.path for run in selected}


def test_plot_selected_runs_writes_per_seed_reconstructions_and_bands(tmp_path, monkeypatch) -> None:
    import matplotlib.figure

    root = tmp_path / "runs"
    group = _seed_group(root, "model_a", "group", stamp="2025-02-01T00:00:00+00:00")
    flat = _write_member(root / "synthetic" / "model_b" / "flat", "model_b", seed=7,
                         stamp="2025-02-01T00:00:00+00:00", root=root)
    captured: dict[str, list] = {}
    original = matplotlib.figure.Figure.savefig

    def record(self, path, *args, **kwargs):
        captured[Path(path).name] = [
            (line.get_label(), list(line.get_ydata()), len(axis.collections))
            for axis in self.axes for line in axis.get_lines()
        ]
        return original(self, path, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", record)
    parser = _plot_parser(root)
    runs = select_plot_runs(parser.parse_args(["--run", str(group), "--run", str(flat)]),
                            parser, problem_id="synthetic")
    output = tmp_path / "plots"
    saved = plot_selected_runs(runs, output, cases=1)
    assert all(path.is_file() for path in saved)
    assert (output / "model_a" / "group" / "training_validation_loss.png").is_file()
    assert (output / "model_a" / "group" / "case_00.png").is_file()
    for seed in (1, 2, 3):
        assert (output / "model_a" / "group" / "seeds" / str(seed) / "case_00.png").is_file()
    assert not (output / "model_b" / "flat" / "seeds").exists()
    assert (output / "comparisons" / "group_1" / "validation_history.png").is_file()
    assert (output / "comparisons" / "group_1" / "reconstruction_overlay.png").is_file()

    history = captured["validation_history.png"]
    group_lines = [item for item in history if "model_a" in item[0]]
    flat_lines = [item for item in history if "model_b" in item[0]]
    assert len(group_lines) == 1 and "n=3" in group_lines[0][0]
    assert group_lines[0][1] == pytest.approx([0.5 * 1.1, 0.5 * 1.2])
    assert group_lines[0][2] == 1  # one shaded band on the shared axis
    assert "n=" not in flat_lines[0][0]


def test_plot_experiment_histories_single_member_has_no_band(tmp_path: Path) -> None:
    import matplotlib.pyplot as plt

    flat = _write_member(tmp_path / "synthetic" / "m" / "flat", "m", seed=1,
                         stamp="2025-02-01T00:00:00+00:00", root=tmp_path)
    figures: list = []
    original = plt.close

    def spy(figure=None):
        figures.append(figure)
        return original(figure)

    plt.close = spy
    try:
        plot_experiment_histories((flat,), output=tmp_path / "h.png")
    finally:
        plt.close = original
    assert figures and all(not axis.collections for axis in figures[-1].axes)


def test_train_config_recent_picks_newest_tuned_run(tmp_path: Path) -> None:
    from chem_operator.experiments import recent_tuned_run

    def finished(run_id: str, *, width: int, stamp: str, trials: bool = True,
                 max_cases: int | None = None) -> Path:
        run_context = RunContext.create(
            tmp_path, problem="synthetic", model="fno", run_id=run_id, seed=1, device="cpu",
            provenance={"max_cases": max_cases},
        )
        run_runner = ExperimentRunner(run_context, spec(tmp_path))
        train_final(run_runner, WorkflowStages(tune=False), config={"width": width, "seed": 1},
                    tuning=None, train_one=train_one)
        if trials:
            run_runner.artifacts.write_tuning_trials(
                [{"trial_id": "t0", "status": "TERMINATED", "metric": "loss", "value": 1.0}]
            )
        run_runner.artifacts.update_manifest(completed_at=stamp)
        return run_context.paths.run_dir

    target = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="target", seed=1, device="cpu")
    target_runner = ExperimentRunner(target, spec(tmp_path))
    with pytest.raises(FileNotFoundError, match="no completed tuned run"):
        resolve_train_config("recent", target_runner)

    older = finished("older", width=4, stamp="2025-01-01T00:00:00+00:00")
    newest = finished("newest", width=8, stamp="2025-02-01T00:00:00+00:00")
    finished("untuned", width=16, stamp="2025-03-01T00:00:00+00:00", trials=False)
    finished("smoke", width=32, stamp="2025-04-01T00:00:00+00:00", max_cases=4)
    failed = RunContext.create(tmp_path, problem="synthetic", model="fno", run_id="zz-failed", seed=1, device="cpu")
    ExperimentRunner(failed, spec(tmp_path)).start()
    ArtifactStore(failed).mark_failed(RuntimeError("x"))

    model_dir = tmp_path / "synthetic" / "fno"
    assert recent_tuned_run(model_dir) == newest
    config, source = resolve_train_config("recent", target_runner)
    assert config["width"] == 8 and source == str(newest)
    assert older != newest

    train_final(target_runner, WorkflowStages(tune=False, train_config="recent", seeds=(1, 2)),
                config=config, tuning=None, train_one=train_one, train_config_source=source)
    assert recent_tuned_run(model_dir) == newest  # the seed run has no trials


def test_runs_trained_from_another_run_inherit_its_tuning_time(tmp_path: Path) -> None:
    def context(run_id: str) -> RunContext:
        return RunContext.create(tmp_path, problem="synthetic", model="fno", run_id=run_id, seed=1, device="cpu")

    tuned = context("tuned")
    tuned_runner = ExperimentRunner(tuned, spec(tmp_path))
    train_final(tuned_runner, WorkflowStages(tune=False), config={"seed": 1}, tuning=None, train_one=train_one)
    tuned_runner.artifacts.update_manifest(timings={"tuning_seconds": 12.5})
    source = str(tuned.paths.run_dir)

    flat = ExperimentRunner(context("flat"), spec(tmp_path))
    train_final(flat, WorkflowStages(tune=False), config={"seed": 1}, tuning=None,
                train_one=train_one, train_config_source=source)
    manifest = flat.artifacts.read_manifest()
    assert manifest["timings"]["tuning_seconds"] == pytest.approx(12.5)
    assert manifest["provenance"]["train_config_source"] == source

    nested = ExperimentRunner(context("nested"), spec(tmp_path))
    train_final(nested, WorkflowStages(tune=False, seeds=(1, 2)), config={"seed": 1}, tuning=None,
                train_one=train_one, train_config_source=source)
    assert nested.artifacts.read_manifest()["timings"]["tuning_seconds"] == pytest.approx(12.5)

    # A chain through a run that predates inheritance reaches the tuned run.
    flat.artifacts.update_manifest(timings={"tuning_seconds": 0.0})
    chained = ExperimentRunner(context("chained"), spec(tmp_path))
    train_final(chained, WorkflowStages(tune=False), config={"seed": 1}, tuning=None,
                train_one=train_one, train_config_source=str(flat.context.paths.run_dir))
    assert chained.artifacts.read_manifest()["timings"]["tuning_seconds"] == pytest.approx(12.5)

    own = ExperimentRunner(context("own"), spec(tmp_path))
    train_final(own, WorkflowStages(tune=False), config={"seed": 1}, tuning=None, train_one=train_one)
    assert own.artifacts.read_manifest()["timings"]["tuning_seconds"] == 0.0
