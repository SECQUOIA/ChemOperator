"""The CSTR plot script draws saved runs without their raw training data."""

from __future__ import annotations

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


def test_cstr_cli_plots_saved_runs_without_raw_datasets(tmp_path, monkeypatch):
    from experiments.cstr import plot
    import matplotlib.pyplot as plt

    run = write_run(tmp_path, "deeponet")
    gas = plot.ct.Solution(plot.MECHANISM)
    labels = ["T", "P", *(f"X[{gas.species_index(name)}]" for name in plot.PLOT_SPECIES)]
    reference = np.ones((1, 3, len(labels)))
    ArtifactStore(run).write_reconstructions(
        case_ids=np.array([0]), coordinates=np.array([[0., 0.5, 1.]]),
        reference=reference, prediction=reference * 0.9,
        labels=np.array(labels),
    )
    monkeypatch.setattr(plot, "plot_co_histories", lambda *args, **kwargs: pytest.fail("Raw training data must be optional"))
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setattr("sys.argv", ["plot.py", "--run", str(run), "--output-dir", str(tmp_path / "plots")])
    with plt.rc_context():
        plot.main()
    run_plots = tmp_path / "plots" / "deeponet" / run.name
    assert (run_plots / "training_validation_loss.png").is_file()
    assert (run_plots / "case_00.png").is_file()


def test_cstr_cli_plots_only_selected_channels_on_log_time(tmp_path, monkeypatch):
    """The mechanism saves ~1,270 channels; plots stay at T, P, NC7H16, O2, CO."""
    from matplotlib.figure import Figure
    from experiments.cstr import plot
    import matplotlib.pyplot as plt

    gas = plot.ct.Solution(plot.MECHANISM)
    selected = [f"X[{gas.species_index(name)}]" for name in ("NC7H16", "O2", "CO")]
    unselected = [f"X[{index}]" for index in range(gas.n_species)
                  if f"X[{index}]" not in selected][:2]
    labels = np.array(["T", "P", unselected[0], *selected, unselected[1]])
    reference = np.ones((1, 3, len(labels)))
    runs = [write_run(tmp_path, model) for model in ("deeponet", "pod_deeponet")]
    for run in runs:
        ArtifactStore(run).write_reconstructions(
            case_ids=np.array([0]), coordinates=np.array([[0., 0.5, 1.]]),
            reference=reference, prediction=reference * 0.9, labels=labels,
        )
    rows: dict[str, list[str]] = {}
    scales: dict[str, set[str]] = {}
    original_save = Figure.savefig

    def capture(figure, path, **kwargs):
        key = str(Path(path).relative_to(tmp_path / "plots"))
        labelled = [axis for axis in figure.axes if axis.get_ylabel()]
        rows[key] = [axis.get_ylabel() for axis in labelled]
        scales[key] = {axis.get_xscale() for axis in labelled}
        return original_save(figure, path, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    monkeypatch.setattr("sys.argv", ["plot.py", "--run", str(runs[0]), "--run", str(runs[1]),
                                     "--output-dir", str(tmp_path / "plots")])
    with plt.rc_context():
        plot.main()
    expected = ["T", "P", *selected]
    case_plot = f"deeponet/{runs[0].name}/case_00.png"
    overlay = "comparisons/group_1/reconstruction_overlay.png"
    assert rows[case_plot] == expected
    assert rows[overlay] == expected
    # Time starts at 0, so the log axis falls back to symlog.
    assert scales[case_plot] == scales[overlay] == {"symlog"}


def test_cstr_co_diagnostic_can_run_explicitly_without_run(tmp_path, monkeypatch):
    import h5py
    import matplotlib.pyplot as plt
    from experiments.cstr import plot

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
