"""Reading and plotting per-trial tuning progress without running Ray."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from chem_operator.experiments import read_tuning_progress, tuning_source_dir
from chem_operator.plotting import plot_tuning_comparison, plot_tuning_progress


def _write_csv(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def _manifest(run_dir: Path, source: str | None = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"provenance": {} if source is None else {"train_config_source": source}}
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return run_dir


def _ray_progress(run_dir: Path, trial_id: str, values: list[float], *,
                  with_id_column: bool = True) -> None:
    rows = []
    for iteration, value in enumerate(values, start=1):
        row: dict[str, object] = {
            "valid_relative_l2": value,
            "timestamp": 1_795_000_000 + 60 * iteration,
            "training_iteration": iteration,
            "hostname": "host",
        }
        if with_id_column:
            row["trial_id"] = trial_id
        rows.append(row)
    folder = run_dir / "ray_results" / "tuning" / f"trainable_{trial_id}_1_lr=0.001_2026"
    _write_csv(folder / "progress.csv", rows)


def _portable_progress(run_dir: Path) -> None:
    _write_csv(run_dir / "tuning_progress.csv", [
        {"trial_id": "aa", "training_iteration": 1, "timestamp": 1_795_000_000,
         "valid_relative_l2": 0.2},
        {"trial_id": "aa", "training_iteration": 2, "timestamp": 1_795_000_060,
         "valid_relative_l2": 0.1},
    ])


def test_read_prefers_run_folder_progress_over_ray_results(tmp_path: Path) -> None:
    run = _manifest(tmp_path / "run")
    _portable_progress(run)
    _ray_progress(run, "zz", [0.5])

    rows = read_tuning_progress(run)

    assert rows is not None
    assert {row["trial_id"] for row in rows} == {"aa"}
    assert [float(row["valid_relative_l2"]) for row in rows] == [0.2, 0.1]


def test_read_falls_back_to_ray_progress_files(tmp_path: Path) -> None:
    run = _manifest(tmp_path / "run")
    _ray_progress(run, "ab12", [0.3, 0.2])
    _ray_progress(run, "cd34", [0.4], with_id_column=False)

    rows = read_tuning_progress(run)

    assert rows is not None
    assert sorted({row["trial_id"] for row in rows}) == ["ab12", "cd34"]
    assert len(rows) == 3


def test_read_returns_none_without_progress(tmp_path: Path) -> None:
    assert read_tuning_progress(_manifest(tmp_path / "run")) is None


def test_source_dir_follows_train_config_source(tmp_path: Path) -> None:
    tuned = _manifest(tmp_path / "tuned")
    _ray_progress(tuned, "ab12", [0.3])
    middle = _manifest(tmp_path / "middle", source=str(tuned))
    final = _manifest(tmp_path / "final", source=str(middle))

    assert tuning_source_dir(tuned) == tuned
    assert tuning_source_dir(final) == tuned
    assert tuning_source_dir(_manifest(tmp_path / "json", source="config.json")) is None
    assert tuning_source_dir(_manifest(tmp_path / "own", source="tuning")) is None


def test_source_dir_rejects_cycles(tmp_path: Path) -> None:
    first, second = tmp_path / "first", tmp_path / "second"
    _manifest(first, source=str(second))
    _manifest(second, source=str(first))

    with pytest.raises(ValueError, match="cycle"):
        tuning_source_dir(first)


def test_plot_writes_figure_from_source_run(tmp_path: Path) -> None:
    tuned = _manifest(tmp_path / "tuned")
    _portable_progress(tuned)
    trained = _manifest(tmp_path / "trained", source=str(tuned))

    path = plot_tuning_progress(trained, tmp_path / "plots" / "tuning_progress.png")

    assert path == tmp_path / "plots" / "tuning_progress.png"
    assert path.stat().st_size > 0


def test_plot_uses_tuning_metric_when_relative_l2_is_missing(tmp_path: Path) -> None:
    run = _manifest(tmp_path / "run")
    _write_csv(run / "tuning_progress.csv", [
        {"trial_id": "aa", "training_iteration": 1, "timestamp": 1_795_000_000,
         "best_valid_data_loss": 0.2},
    ])
    _write_csv(run / "tuning_trials.csv", [
        {"trial_id": "aa", "status": "completed", "metric": "best_valid_data_loss",
         "value": 0.2},
    ])

    assert plot_tuning_progress(run, tmp_path / "plot.png") is not None


def test_plot_returns_none_without_progress(tmp_path: Path) -> None:
    run = _manifest(tmp_path / "run")

    assert plot_tuning_progress(run, tmp_path / "plot.png") is None
    assert not (tmp_path / "plot.png").exists()


def _run(path: Path, model_id: str) -> SimpleNamespace:
    return SimpleNamespace(path=path, manifest={"model_id": model_id})


def test_comparison_overlays_models(tmp_path: Path) -> None:
    deeponet = _manifest(tmp_path / "deeponet")
    _portable_progress(deeponet)
    fno = _manifest(tmp_path / "fno")
    _ray_progress(fno, "ab12", [0.3, 0.2])

    path = plot_tuning_comparison(
        (_run(deeponet, "deeponet"), _run(fno, "fno_physics")), tmp_path / "cmp.png")

    assert path == tmp_path / "cmp.png"
    assert path.stat().st_size > 0


def test_comparison_needs_two_studies(tmp_path: Path) -> None:
    tuned = _manifest(tmp_path / "tuned")
    _portable_progress(tuned)
    untuned = _manifest(tmp_path / "untuned")
    retrained = _manifest(tmp_path / "retrained", source=str(tuned))
    output = tmp_path / "cmp.png"

    assert plot_tuning_comparison(
        (_run(tuned, "deeponet"), _run(untuned, "fno")), output) is None
    # A run trained from another's study shares that study, so it adds nothing.
    assert plot_tuning_comparison(
        (_run(tuned, "deeponet"), _run(retrained, "fno")), output) is None
    assert not output.exists()


def test_comparison_skips_studies_without_metric(tmp_path: Path) -> None:
    runs = []
    for name in ("deeponet", "pod_deeponet"):
        run = _manifest(tmp_path / name)
        _portable_progress(run)
        runs.append(_run(run, name))
    other = _manifest(tmp_path / "fno")
    _write_csv(other / "tuning_progress.csv", [
        {"trial_id": "aa", "training_iteration": 1, "timestamp": 1_795_000_000,
         "best_valid_data_loss": 0.2},
    ])

    assert plot_tuning_comparison(
        (*runs, _run(other, "fno")), tmp_path / "cmp.png") is not None
    assert plot_tuning_comparison(
        (runs[0], _run(other, "fno")), tmp_path / "none.png") is None
