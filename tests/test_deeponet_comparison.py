"""Tests for headless DeepONet comparison and artifact writing."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import chem_operator._models.deeponet_comparison as comparison
from chem_operator.models import (
    CoordinateScaler,
    DeepONetComparisonResult,
    DeepONetTrainingConfig,
    DeepONetTrainingHistory,
    PODTransform,
    save_deeponet_comparison,
)
from chem_operator.normalization import IdentityNormalizer


class TinyModel(torch.nn.Module):
    """Small model exposing the branch/trunk attributes in artifact metadata."""

    def __init__(self, value: float, *, flattened: bool = False) -> None:
        super().__init__()
        self.branch = torch.nn.Linear(1, 1)
        self.trunk = None if flattened else torch.nn.Linear(1, 1)
        self.value = value
        self.flattened = flattened

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        branch, trunk = inputs
        shape = (
            (branch.shape[0], 3)
            if self.flattened
            else (branch.shape[0], trunk.shape[0])
        )
        return self.branch.weight.new_full(shape, self.value)


class FakeAdapter:
    format = "cartesian_product"

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index != 0:
            raise IndexError(index)
        return {"trunk": torch.tensor([[0.0], [1.0], [2.0]])}


class TinyTrainingAdapter:
    """In-memory Cartesian trajectories for a one-epoch integration test."""

    format = "cartesian_product"

    def __len__(self) -> int:
        return 4

    def __getitem__(self, index: int) -> dict[str, object]:
        value = float(index + 1)
        coordinate = torch.tensor([0.0, 0.5, 1.0])
        return {
            "branch": torch.tensor([value]),
            "trunk": coordinate[:, None],
            "target": torch.tensor(
                [[value], [1.5 * value], [2.0 * value]]
            ),
            "coordinate": coordinate,
            "labels": ("value",),
        }


def history() -> DeepONetTrainingHistory:
    return DeepONetTrainingHistory(
        steps=[1, 2],
        loss_train=[[1.0], [0.5]],
        loss_test=[[1.25], [0.75]],
        loss_name="relative_l2",
        best_epoch=2,
        best_valid_loss=0.75,
    )


def pod_transform() -> PODTransform:
    return PODTransform(
        mean=np.zeros(3, dtype=np.float32),
        basis=np.ones((3, 1), dtype=np.float32),
        explained_variance_ratio=np.ones(1, dtype=np.float32),
        cumulative_explained_variance=1.0,
        output_shape=(3, 1),
    )


def result_fixture() -> DeepONetComparisonResult:
    config = DeepONetTrainingConfig(epochs=2, batch_size=2)
    return DeepONetComparisonResult(
        direct_model=TinyModel(0.0),
        pod_model=TinyModel(1.0, flattened=True),
        direct_config=config,
        pod_config=config,
        direct_history=history(),
        pod_history=history(),
        coordinate_scaler=CoordinateScaler(
            minimum=np.zeros(1, dtype=np.float32),
            span=np.ones(1, dtype=np.float32),
        ),
        pod=pod_transform(),
        metrics={
            "deeponet": {
                "rmse": 2.0,
                "relative_l2": 1.0,
                "max_absolute_error": 2.0,
            },
            "pod_deeponet": {
                "rmse": 1.0,
                "relative_l2": 0.5,
                "max_absolute_error": 1.0,
            },
        },
        reconstructions=comparison.ReconstructionSamples(
            coordinates=np.array([[0.0, 1.0, 2.0]], dtype=np.float32),
            reference=np.full((1, 3, 1), 2.0, dtype=np.float32),
            predictions=np.array(
                [
                    [[[0.0], [0.0], [0.0]]],
                    [[[1.0], [1.0], [1.0]]],
                ],
                dtype=np.float32,
            ),
            labels=("value",),
        ),
    )


def test_comparison_is_headless() -> None:
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import chem_operator.datasets, chem_operator.models; "
            "assert 'matplotlib' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr


def test_run_comparison_streams_metrics_and_limits_samples(monkeypatch) -> None:
    fitted: list[object] = []
    pod = pod_transform()

    def fake_fit(dataset, **kwargs):
        fitted.append(dataset)
        assert kwargs["variance_threshold"] == pytest.approx(0.95)
        return pod

    def fake_train(train, validation, *, pod=None, **kwargs):
        del train, validation, kwargs
        model = TinyModel(0.0) if pod is None else TinyModel(1.0, flattened=True)
        return model, history()

    batch = {
        "branch": torch.ones((2, 1)),
        "trunk": torch.tensor([[-1.0], [0.0], [1.0]]),
        "target": torch.full((2, 3, 1), 2.0),
        "coordinates": (
            torch.tensor([0.0, 1.0, 2.0]),
            torch.tensor([0.0, 1.0, 2.0]),
        ),
        "labels": ("value",),
    }
    monkeypatch.setattr(comparison, "fit_incremental_pod_dataset", fake_fit)
    monkeypatch.setattr(comparison, "train_deeponet_lazy", fake_train)
    monkeypatch.setattr(
        comparison,
        "make_deeponet_dataloader",
        lambda *args, **kwargs: [batch],
    )
    adapter = FakeAdapter()
    config = DeepONetTrainingConfig(epochs=1, batch_size=2)
    result = comparison.run_deeponet_comparison(
        adapter,
        adapter,
        adapter,
        IdentityNormalizer(),
        direct_config=config,
        pod_variance_threshold=0.95,
        reconstruction_cases=1,
    )

    assert fitted == [adapter]
    assert result.coordinate_scaler.minimum.tolist() == [0.0]
    assert result.coordinate_scaler.span.tolist() == [2.0]
    assert result.metrics["deeponet"] == pytest.approx(
        {"rmse": 2.0, "relative_l2": 1.0, "max_absolute_error": 2.0}
    )
    assert result.metrics["pod_deeponet"] == pytest.approx(
        {"rmse": 1.0, "relative_l2": 0.5, "max_absolute_error": 1.0}
    )
    assert result.reconstructions.reference.shape == (1, 3, 1)
    assert result.reconstructions.predictions.shape == (2, 1, 3, 1)


def test_run_comparison_rejects_invalid_reconstruction_count() -> None:
    adapter = FakeAdapter()
    with pytest.raises(ValueError, match="reconstruction_cases"):
        comparison.run_deeponet_comparison(
            adapter,
            adapter,
            adapter,
            IdentityNormalizer(),
            direct_config=DeepONetTrainingConfig(epochs=1),
            reconstruction_cases=0,
        )


def test_run_comparison_trains_both_models_for_one_epoch(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    config = DeepONetTrainingConfig(
        epochs=1,
        batch_size=2,
        width=4,
        latent_width=2,
        branch_hidden_layers=1,
        trunk_hidden_layers=1,
        display_every=1,
    )
    adapter = TinyTrainingAdapter()
    result = comparison.run_deeponet_comparison(
        adapter,
        adapter,
        adapter,
        IdentityNormalizer(),
        direct_config=config,
        pod=pod_transform(),
        reconstruction_cases=1,
    )

    assert result.direct_history.steps == [1]
    assert result.pod_history.steps == [1]
    assert all(
        np.isfinite(value)
        for metrics in result.metrics.values()
        for value in metrics.values()
    )
    assert result.reconstructions.predictions.shape == (2, 1, 3, 1)


def test_save_comparison_writes_versioned_artifacts(tmp_path: Path) -> None:
    save_deeponet_comparison(
        result_fixture(),
        tmp_path,
        problem="synthetic_problem",
    )

    with (tmp_path / "metrics.json").open(encoding="utf-8") as file:
        metrics = json.load(file)
    assert metrics["schema_version"] == 1
    assert metrics["problem"] == "synthetic_problem"
    assert tuple(metrics["models"]) == ("deeponet", "pod_deeponet")
    assert metrics["models"]["deeponet"]["metrics"]["relative_l2"] == 1.0
    assert metrics["pod"] == {
        "components": 1,
        "cumulative_explained_variance": 1.0,
    }

    with (tmp_path / "history.csv").open(
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 8
    assert set(rows[0]) == {"model", "epoch", "split", "metric", "value"}
    assert {row["model"] for row in rows} == {"deeponet", "pod_deeponet"}

    with np.load(tmp_path / "reconstructions.npz", allow_pickle=False) as data:
        assert np.asarray(data["predictions"]).shape == (2, 1, 3, 1)
        assert np.asarray(data["model_ids"]).tolist() == [
            "deeponet",
            "pod_deeponet",
        ]
        assert np.asarray(data["labels"]).tolist() == ["value"]
    with np.load(tmp_path / "ipca_pod_matrix.npz", allow_pickle=False) as data:
        assert np.asarray(data["basis"]).shape == (3, 1)
        assert np.asarray(data["output_shape"]).tolist() == [3, 1]
