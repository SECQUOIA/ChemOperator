"""Tests for plotting saved DeepONet artifacts."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from chem_operator.experiments import (
    ArtifactStore,
    MetricEvent,
    RunContext,
    build_manifest,
)
from chem_operator.plotting import (
    HistoryPanel,
    plot_deeponet_artifacts,
    plot_deeponet_run_set,
    plot_deeponet_runs,
    plot_experiment_histories,
    use_publication_style,
)


def write_artifacts(directory: Path) -> None:
    with (directory / "history.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("model", "epoch", "split", "metric", "value"),
        )
        writer.writeheader()
        for model in ("deeponet", "pod_deeponet"):
            for split, value in (("train", 1.0), ("validation", 1.5)):
                writer.writerow(
                    {
                        "model": model,
                        "epoch": 1,
                        "split": split,
                        "metric": "relative_l2",
                        "value": value,
                    }
                )
    coordinates = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
    reference = np.array([[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]])
    predictions = np.stack((reference * 0.9, reference * 1.1))
    np.savez(
        directory / "reconstructions.npz",
        coordinates=coordinates,
        reference=reference,
        predictions=predictions,
        model_ids=np.asarray(("deeponet", "pod_deeponet")),
        labels=np.asarray(("T", "X[0]")),
    )


def test_plot_deeponet_artifacts_round_trip(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    loss, reconstruction = plot_deeponet_artifacts(
        tmp_path,
        selected_labels=("T", "X[0]"),
        coordinate_label="Time [s]",
        cases=5,
    )

    assert loss == tmp_path / "training_validation_loss.png"
    assert reconstruction == tmp_path / "test_reconstructions.png"
    assert loss.stat().st_size > 0
    assert reconstruction.stat().st_size > 0


def test_plot_deeponet_artifacts_rejects_unknown_label(tmp_path: Path) -> None:
    write_artifacts(tmp_path)
    with pytest.raises(KeyError, match="pressure"):
        plot_deeponet_artifacts(
            tmp_path,
            selected_labels=("pressure",),
            coordinate_label="Time [s]",
        )


def write_run(directory: Path, model: str, *, reference_scale: float = 1.0) -> Path:
    context = RunContext.create(
        directory,
        problem="synthetic",
        model=model,
        run_id="run-1",
        seed=7,
        device="cpu",
    )
    store = ArtifactStore(context)
    store.write_manifest(
        build_manifest(
            context,
            problem_id="synthetic",
            model_id=model,
            benchmark_protocol_id="operator-cartesian-v1",
            dataset_fingerprints={
                "train": "sha256:train",
                "validation": "sha256:validation",
                "test": "sha256:test",
            },
            fields=("T", "X"),
            channels=("T", "X"),
            units={"T": "K", "X": "-"},
            coordinates=("t",),
            selected_test_case_ids=(0,),
            parameter_count=4,
            tuning_budget={"samples": 1, "epochs": 1},
            project_root=directory,
        )
    )
    store.write_best_config({"width": 4})
    store.write_history(
        (
            MetricEvent(1, "train", "objective", 1.0),
            MetricEvent(1, "val", "relative_l2", 0.5),
        )
    )
    store.write_metrics({"relative_l2": 0.4})
    store.write_tuning_trials(())
    reference = reference_scale * np.array(
        [[[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]]], dtype=np.float32
    )
    store.write_reconstructions(
        case_ids=np.asarray((0,)),
        coordinates=np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
        reference=reference,
        prediction=reference * (0.9 if model == "deeponet" else 1.1),
        labels=np.asarray(("T", "X[0]")),
    )
    store.write_torch_checkpoint({"state_dict": {"weight": torch.ones(1)}})
    store.mark_completed(
        timings={
            "data_generation_seconds": 0.0,
            "tuning_seconds": 0.1,
            "final_training_seconds": 0.1,
            "inference_seconds": 0.01,
        }
    )
    return context.paths.run_dir


def test_plot_deeponet_runs_round_trip(tmp_path: Path) -> None:
    direct = write_run(tmp_path / "runs", "deeponet")
    pod = write_run(tmp_path / "runs", "pod_deeponet")

    history, reconstruction = plot_deeponet_runs(
        direct,
        pod,
        selected_labels=("T", "X[0]"),
        coordinate_label="Time [s]",
        output_dir=tmp_path / "plots",
        cases=5,
    )

    assert history == tmp_path / "plots" / "training_validation_loss.png"
    assert reconstruction == tmp_path / "plots" / "test_reconstructions.png"
    assert history.stat().st_size > 0
    assert reconstruction.stat().st_size > 0


def test_plot_deeponet_run_set_accepts_one_run_and_reversed_pair(tmp_path: Path) -> None:
    direct = write_run(tmp_path / "runs", "deeponet")
    pod = write_run(tmp_path / "runs", "pod_deeponet")

    for name, runs in (("single", (direct,)), ("pair", (pod, direct))):
        history, reconstruction = plot_deeponet_run_set(
            runs,
            selected_labels=("T", "X[0]"),
            coordinate_label="Time [s]",
            output_dir=tmp_path / name,
        )
        assert history.is_file()
        assert reconstruction.is_file()


def test_plot_deeponet_run_set_requires_a_run(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one run"):
        plot_deeponet_run_set(
            (),
            selected_labels=("T",),
            coordinate_label="Time [s]",
            output_dir=tmp_path / "plots",
        )


def test_plot_deeponet_runs_rejects_mismatched_references(tmp_path: Path) -> None:
    direct = write_run(tmp_path / "runs", "deeponet")
    pod = write_run(
        tmp_path / "runs",
        "pod_deeponet",
        reference_scale=2.0,
    )

    with pytest.raises(ValueError, match="reference"):
        plot_deeponet_runs(
            direct,
            pod,
            selected_labels=("T",),
            coordinate_label="Time [s]",
            output_dir=tmp_path / "plots",
        )


def test_publication_style_saves_without_latex(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setenv("PATH", str(tmp_path))
    with plt.rc_context():
        use_publication_style()
        figure, axis = plt.subplots()
        axis.plot([0., 1.], [0., 1.])
        axis.set_xlabel(r"Time $t$ [s]")
        try:
            figure.savefig(tmp_path / "publication.png")
        finally:
            plt.close(figure)
    assert (tmp_path / "publication.png").stat().st_size > 0


def test_cstr_comparison_uses_mean_per_case_errors(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    runs = [write_run(tmp_path, model) for model in ("deeponet", "pod_deeponet", "fno")]
    for run in runs:
        history = [MetricEvent(1, "val", "relative_l2", 0.55)]
        if run.parent.name == "fno":
            history = [MetricEvent(1, "val", "relative_l2", math.sqrt(2 / 101)),
                       MetricEvent(1, "val", "data_loss", 0.55)]
        ArtifactStore(run).write_history(history)
    curves = []
    original_save = Figure.savefig

    def capture(figure, path, **kwargs):
        if Path(path).name == "training_validation_loss.png":
            curves.extend(line.get_ydata() for line in figure.axes[0].lines)
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    plot_deeponet_run_set(runs, selected_labels=("T",), coordinate_label="Time [s]", output_dir=tmp_path / "plots")
    assert len(curves) == 3
    np.testing.assert_allclose(curves, [[0.55], [0.55], [0.55]])


def test_cstr_cli_plots_saved_runs_without_raw_datasets(tmp_path, monkeypatch):
    from scripts.cstr import plot
    import matplotlib.pyplot as plt

    run = write_run(tmp_path, "deeponet")
    fuel = plot.ct.Solution(plot.MECHANISM).species_index("NC7H16")
    reference = np.ones((1, 3, 3))
    ArtifactStore(run).write_reconstructions(
        case_ids=np.array([0]), coordinates=np.array([[0., 0.5, 1.]]),
        reference=reference, prediction=reference * 0.9,
        labels=np.array(["T", "P", f"X[{fuel}]"]),
    )
    monkeypatch.setattr(plot, "plot_co_histories", lambda *args, **kwargs: pytest.fail("Raw training data must be optional"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["plot.py", "--run", str(run), "--output-dir", str(tmp_path / "plots")])
    with plt.rc_context():
        plot.main()
    assert (tmp_path / "plots" / "training_validation_loss.png").is_file()
    assert (tmp_path / "plots" / "test_reconstructions.png").is_file()


def test_cstr_co_diagnostic_can_run_explicitly_without_run(tmp_path, monkeypatch):
    import h5py
    import matplotlib.pyplot as plt
    from scripts.cstr import plot

    mechanism = plot.ct.Solution(plot.MECHANISM)
    values = np.zeros((3, mechanism.n_species))
    values[:, mechanism.species_index("CO")] = [0., 0.1, 0.2]
    dataset = tmp_path / "training.h5"
    with h5py.File(dataset, "w") as archive:
        case = archive.create_group("cases/case_0")
        case.create_dataset("coordinates/t", data=[0., 0.1, 1.])
        case.create_dataset("fields/X", data=values)
    output = tmp_path / "co.png"
    monkeypatch.setattr("sys.argv", ["plot.py", "--co-histories", "--dataset", str(dataset), "--co-output", str(output)])
    with plt.rc_context():
        plot.main()
    assert output.stat().st_size > 0


def test_publication_tex_requires_explicit_opt_in():
    import matplotlib.pyplot as plt

    with plt.rc_context():
        use_publication_style(use_tex=True)
        assert plt.rcParams["text.usetex"]
        use_publication_style()
        assert not plt.rcParams["text.usetex"]


def test_model_metric_selection_cannot_overlay_raw_objectives(tmp_path):
    runs = [write_run(tmp_path, model) for model in ("deeponet", "fno")]
    with pytest.raises(ValueError, match="Raw objectives"):
        plot_experiment_histories(
            runs, output=tmp_path / "invalid.png",
            panels=(HistoryPanel(("relative_l2",), "Error", metrics_by_model={"fno": ("objective",)}),),
        )
