"""Selection and comparison behavior shared by the seven plot CLIs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from chem_operator._experiments.artifacts import ArtifactStore, build_manifest
from chem_operator._experiments.metrics import MetricEvent
from chem_operator._experiments.types import RunContext
from chem_operator.plotting import (
    add_plot_run_arguments, comparison_groups, plot_selected_runs, print_plot_paths,
    select_plot_runs, set_log_xscale,
)


def test_print_plot_paths_uses_one_line_per_path(capsys: pytest.CaptureFixture[str]) -> None:
    paths = (Path("plots/first.png"), Path("plots/second.png"))

    print_plot_paths(iter(paths))

    assert capsys.readouterr().out == (
        "Plots written to:\nplots/first.png\nplots/second.png\n"
    )


def _run(root: Path, model: str, run_id: str, *, stamp: str,
         status: str = "completed", protocol: str = "shared") -> Path:
    context = RunContext.create(root, problem="synthetic", model=model,
                                run_id=run_id, seed=1)
    store = ArtifactStore(context)
    manifest = build_manifest(
        context, problem_id="synthetic", model_id=model,
        benchmark_protocol_id=protocol,
        dataset_fingerprints={"train": "a", "validation": "b", "test": "c"},
        fields=("T",), channels=("T",), units={"T": "K"},
        coordinates=("t",), selected_test_case_ids=(1,), parameter_count=1,
        tuning_budget={}, project_root=root,
    )
    manifest.update(status=status, created_at=stamp)
    if status == "completed":
        manifest["completed_at"] = stamp
    store.write_manifest(manifest)
    store.write_best_config({"width": 1})
    store.write_history((MetricEvent(1, "val", "relative_l2", 0.5),))
    store.write_metrics({"relative_l2": 0.4})
    store.write_tuning_trials(())
    store.write_reconstructions(
        case_ids=np.array([1]), labels=np.array(["T"]),
        coordinates=np.array([[[0.0], [1.0]]]),
        reference=np.array([[[1.0], [2.0]]]),
        prediction=np.array([[[1.1], [2.1]]]),
    )
    store.write_torch_checkpoint({"state_dict": {}})
    return context.paths.run_dir


def _parser(root: Path, *, cases: int = 2) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_plot_run_arguments(parser, root=root, problem_id="synthetic", cases=cases)
    return parser


def test_shared_arguments_default_to_mirrored_plot_tree(tmp_path: Path) -> None:
    args = _parser(tmp_path, cases=3).parse_args(["--recent"])
    assert args.runs_root == tmp_path / "artifacts" / "runs"
    assert args.output_dir == tmp_path / "artifacts" / "plots" / "synthetic"
    assert args.cases == 3
    override = _parser(tmp_path).parse_args(
        ["--recent", "--output-dir", str(tmp_path / "custom"), "--cases", "1"]
    )
    assert override.output_dir == tmp_path / "custom"
    assert override.cases == 1


@pytest.mark.parametrize("argv", [
    ["--recent", "--cases", "0"],
    ["--recent", "--cases", "-2"],
    ["--recent", "--cases", "two"],
    ["--recent", "--run", "some/run"],
    [],
])
def test_shared_arguments_reject_invalid_combinations(
    tmp_path: Path, argv: list[str], capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        _parser(tmp_path).parse_args(argv)
    assert excinfo.value.code == 2
    stderr = capsys.readouterr().err
    if "--cases" in argv:
        assert "--cases" in stderr
    else:
        assert "--run" in stderr and "--recent" in stderr


def test_recent_selects_newest_completed_per_discovered_model(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    older = _run(root, "model_a", "older", stamp="2025-01-01T00:00:00+00:00")
    newer = _run(root, "model_a", "newer", stamp="2025-02-01T00:00:00+00:00")
    _run(root, "model_a", "running", stamp="2025-03-01T00:00:00+00:00", status="running")
    other = _run(root, "future_model", "first", stamp="2025-01-02T00:00:00+00:00")
    ArtifactStore(other).update_manifest(completed_at=None)
    malformed = _run(root, "model_a", "malformed", stamp="2025-04-01T00:00:00+00:00")
    (malformed / "reconstructions.npz").unlink()
    (root / "synthetic" / "model_a" / "broken").mkdir()
    parser = _parser(root)
    args = parser.parse_args(["--recent", "--runs-root", str(root)])
    selected = select_plot_runs(args, parser, problem_id="synthetic")
    assert {run.path for run in selected} == {newer, other}
    assert older not in {run.path for run in selected}


def test_recent_empty_errors_and_explicit_invalid_errors(tmp_path: Path) -> None:
    parser = _parser(tmp_path)
    with pytest.raises(SystemExit, match="2"):
        select_plot_runs(parser.parse_args(["--recent"]), parser, problem_id="synthetic")
    with pytest.raises(FileNotFoundError):
        select_plot_runs(parser.parse_args(["--run", str(tmp_path / "missing")]),
                         parser, problem_id="synthetic")


def test_plot_all_runs_and_only_compatible_groups(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    a = _run(root, "model_a", "a", stamp="2025-01-01T00:00:00+00:00")
    b = _run(root, "model_b", "b", stamp="2025-01-01T00:00:00+00:00")
    c = _run(root, "model_c", "c", stamp="2025-01-01T00:00:00+00:00",
             protocol="different")
    d = _run(root, "model_d", "d", stamp="2025-01-01T00:00:00+00:00")
    ArtifactStore(d).write_reconstructions(
        case_ids=np.array([1]), labels=np.array(["T"]),
        coordinates=np.array([[[0.0], [1.0]]]),
        reference=np.array([[[5.0], [6.0]]]),
        prediction=np.array([[[5.1], [6.1]]]),
    )
    parser = _parser(root)
    runs = select_plot_runs(parser.parse_args(["--run", str(a), "--run", str(b),
                                                "--run", str(c), "--run", str(d)]), parser,
                            problem_id="synthetic")
    assert [[run.path for run in group] for group in comparison_groups(runs)] == [[a, b]]
    output = tmp_path / "plots"
    saved = plot_selected_runs(runs, output, cases=1)
    for model, run_id in (("model_a", "a"), ("model_b", "b"),
                          ("model_c", "c"), ("model_d", "d")):
        assert (output / model / run_id / "case_00.png").is_file()
    assert (output / "comparisons" / "group_1" / "validation_history.png").is_file()
    assert (output / "comparisons" / "group_1" / "reconstruction_overlay.png").is_file()
    assert all(path.is_file() for path in saved)


def test_plot_selected_runs_adds_tuning_progress_when_available(tmp_path: Path) -> None:
    root = tmp_path / "project"
    tuned = _run(root, "model_a", "tuned", stamp="2026-01-01T00:00:00+00:00")
    untuned = _run(root, "model_b", "untuned", stamp="2026-01-01T00:00:00+00:00")
    with (tuned / "tuning_progress.csv").open("w", encoding="utf-8") as stream:
        stream.write("trial_id,training_iteration,timestamp,valid_relative_l2\n"
                     "t1,1,1795000000,0.2\nt1,2,1795000060,0.1\n")
    parser = _parser(root)
    runs = select_plot_runs(parser.parse_args(["--run", str(tuned), "--run", str(untuned)]),
                            parser, problem_id="synthetic")

    saved = plot_selected_runs(runs, tmp_path / "plots", cases=1)

    assert tmp_path / "plots" / "model_a" / "tuned" / "tuning_progress.png" in saved
    assert not (tmp_path / "plots" / "model_b" / "untuned" / "tuning_progress.png").exists()


def test_set_log_xscale_uses_symlog_when_values_include_zero() -> None:
    import matplotlib.pyplot as plt

    figure, (positive, with_zero) = plt.subplots(1, 2)
    set_log_xscale(positive, np.array([1e-4, 1e-2, 1.0]))
    set_log_xscale(with_zero, [np.array([0.0, 2e-4, 1.0]), np.array([0.0, 4e-4])])
    assert positive.get_xscale() == "log"
    assert with_zero.get_xscale() == "symlog"
    assert with_zero.xaxis.get_transform().linthresh == pytest.approx(1e-4)
    plt.close(figure)


def test_set_log_xscale_float32_grid_keeps_first_decade_tick_off_zero() -> None:
    import matplotlib.pyplot as plt

    time = np.r_[0.0, np.geomspace(1.86e-4, 100.0, 20)].astype(np.float32)
    figure, axis = plt.subplots()
    axis.plot(time, time)
    set_log_xscale(axis, time)
    figure.canvas.draw()
    ticks = [tick for tick in axis.get_xticks() if axis.get_xlim()[0] <= tick <= axis.get_xlim()[1]]
    assert ticks[:2] == pytest.approx([0.0, 1e-4])
    plt.close(figure)


def test_plot_selected_runs_supports_log_time_axis(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    # Fixture coordinates start at t = 0, exercising the symlog fallback.
    a = _run(root, "model_a", "a", stamp="2025-01-01T00:00:00+00:00")
    b = _run(root, "model_b", "b", stamp="2025-01-01T00:00:00+00:00")
    parser = _parser(root)
    runs = select_plot_runs(parser.parse_args(["--run", str(a), "--run", str(b)]),
                            parser, problem_id="synthetic")
    output = tmp_path / "plots"

    saved = plot_selected_runs(runs, output, cases=1, labels=("T",), xlog=True)

    assert (output / "model_a" / "a" / "case_00.png").is_file()
    assert (output / "comparisons" / "group_1" / "reconstruction_overlay.png").is_file()
    assert all(path.is_file() for path in saved)
