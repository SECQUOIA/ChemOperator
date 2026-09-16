"""Tests for single-variant DeepONet orchestration."""

from pathlib import Path

import pytest

from chem_operator.experiments import (
    DeepONetTuningSettings,
    ExperimentSpec,
    RunContext,
    run_deeponet_variant,
)


def spec(tmp_path: Path, model: str) -> ExperimentSpec:
    return ExperimentSpec(
        problem_id="synthetic",
        model_id=model,
        benchmark_protocol_id="test-v1",
        dataset_fingerprints={
            "train": "sha256:train",
            "validation": "sha256:validation",
            "test": "sha256:test",
        },
        fields=("u",),
        channels=("u",),
        units={"u": "-"},
        coordinates=("x",),
        selected_test_case_ids=(0,),
        project_root=tmp_path,
    )


def test_variant_can_skip_tuning_and_training(tmp_path: Path) -> None:
    context = RunContext.create(
        tmp_path,
        problem="synthetic",
        model="deeponet",
        run_id="run-1",
        seed=7,
    )

    result = run_deeponet_variant(
        context=context,
        spec=spec(tmp_path, "deeponet"),
        normalizer=object(),  # type: ignore[arg-type]
        pod=None,
        search_space={},
        tuning_data=lambda _context: None,  # type: ignore[arg-type]
        final_data=lambda: None,
        tuning_settings=DeepONetTuningSettings(max_epochs=1, num_samples=1),
        tune=False,
        train=False,
    )

    assert result is None


def test_pod_variant_requires_a_pod_transform(tmp_path: Path) -> None:
    context = RunContext.create(
        tmp_path,
        problem="synthetic",
        model="pod_deeponet",
        run_id="run-1",
        seed=7,
    )

    with pytest.raises(ValueError, match="requires a POD transform"):
        run_deeponet_variant(
            context=context,
            spec=spec(tmp_path, "pod_deeponet"),
            normalizer=object(),  # type: ignore[arg-type]
            pod=None,
            search_space={},
            tuning_data=lambda _context: None,  # type: ignore[arg-type]
            final_data=lambda: None,
            tuning_settings=DeepONetTuningSettings(max_epochs=1, num_samples=1),
            tune=False,
            train=False,
        )
