"""Tests for reusable DeepONet tuning configuration."""

from pathlib import Path

import pytest

from chem_operator.experiments import (
    DeepONetTuningSettings,
    RunContext,
    Tuner,
    deeponet_tuner,
)


def test_deeponet_tuner_builds_a_generic_tuner(tmp_path: Path) -> None:
    context = RunContext.create(
        tmp_path,
        problem="synthetic",
        model="deeponet",
        run_id="run-1",
        seed=7,
    )
    settings = DeepONetTuningSettings(
        max_epochs=12,
        num_samples=4,
        max_concurrent_trials=2,
        cpus_per_trial=3,
        gpus_per_trial=0,
    )
    dataset_factory = lambda _context: (object(), object())

    tuner = deeponet_tuner(
        search_space={"width": 64},
        pod=None,
        dataset_factory=dataset_factory,
        context=context,
        settings=settings,
    )

    assert isinstance(tuner, Tuner)
    assert tuner.search_space == {"width": 64}
    assert tuner.dataset_factory is dataset_factory
    assert tuner.config.metric == "best_valid_loss"
    assert tuner.config.max_epochs == 12
    assert tuner.config.num_samples == 4
    assert tuner.config.max_concurrent_trials == 2
    assert tuner.config.resources_per_trial == {"cpu": 3, "gpu": 0}
    assert tuner.config.optuna_seed == 7


@pytest.mark.parametrize(
    ("max_epochs", "num_samples"),
    [(0, 1), (1, 0)],
)
def test_deeponet_tuning_settings_require_positive_counts(
    max_epochs: int,
    num_samples: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        DeepONetTuningSettings(
            max_epochs=max_epochs,
            num_samples=num_samples,
        )
