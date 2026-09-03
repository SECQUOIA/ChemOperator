"""Tests for plotting saved DeepONet artifacts."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from chem_operator.plotting import plot_deeponet_artifacts


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
