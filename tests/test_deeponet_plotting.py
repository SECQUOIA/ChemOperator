"""Tests for plotting saved DeepONet artifacts."""

from __future__ import annotations

import csv
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
from chem_operator.plotting import plot_deeponet_artifacts, plot_deeponet_runs


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
