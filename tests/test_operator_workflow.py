"""Trainer boundary of the operator workflow: one factory, library-set seeds and epochs."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ray import tune
from torch.utils.data import Dataset

from chem_operator.experiments import (
    ArtifactStore,
    EvaluationOutcome,
    ExperimentSpec,
    OperatorTrainer,
    Tuner,
    build_tuning_config,
    parse_operator_args,
    run_operator,
)


def spec(tmp_path: Path) -> ExperimentSpec:
    return ExperimentSpec(
        problem_id="synthetic",
        model_id="fno",
        benchmark_protocol_id="test-v1",
        dataset_fingerprints={"train": "a", "validation": "b", "test": "c"},
        fields=("temperature",),
        channels=("temperature",),
        units={"temperature": "K"},
        coordinates=("x", "y"),
        selected_test_case_ids=(0,),
        tuning_budget={"samples": 1, "epochs": 2},
        project_root=tmp_path,
    )


def evaluate(trained, data, context):
    prediction = trained.predict(data, context).numpy()
    reference = np.stack([data[index]["y"].numpy() for index in range(len(data))])
    return EvaluationOutcome(
        metrics={"relative_l2": float(np.linalg.norm(prediction - reference) / np.linalg.norm(reference))},
        reconstructions={"case_ids": np.asarray((0,)), "reference": reference[:1],
                         "prediction": prediction[:1]},
        inference_seconds=0.1,
    )


def test_tuning_and_final_training_share_the_factory_with_stage_seed_and_epochs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A spy factory sees the context seed in every call, --tune-epochs in trials and 2x in final."""
    calls = tmp_path / "factory_calls.jsonl"

    # Local classes, so Ray pickles them by value: workers cannot import this test module.
    class TinyFields(Dataset):
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            value = torch.full((1, 2, 2), float(index + 1))
            return {"x": value, "y": 2.0 * value}

    class Reader:
        def close(self) -> None:
            pass

    def trainer_factory(config, context):
        # Trials run in Ray workers, so record calls in a file rather than in memory.
        with calls.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "trial": "trial_id" in context.provenance,
                "seed": config["seed"],
                "context_seed": context.seed,
                "epochs": config["epochs"],
            }) + "\n")
        return OperatorTrainer(lambda _config: torch.nn.Conv2d(1, 1, kernel_size=1),
                               {"batch_size": 2, "learning_rate": 1.0e-2, **config})

    # Ray's socket paths must stay short, so its temporary directory is not under tmp_path.
    paths = SimpleNamespace(root=tmp_path, data=tmp_path,
                            ray=Path("/tmp") / f"co-ray-{os.getpid()}-factory")
    monkeypatch.setattr(sys, "argv", [
        "model.py", "--tune-seed", "7", "--tune-epochs", "2", "--samples", "1", "--plot-cases", "1",
        "--device", "cpu", "--runs-root", str(tmp_path / "runs"), "--run-id", "r",
    ])
    args = parse_operator_args(paths, tune_epochs=5, samples=3)
    context = run_operator(
        args, paths, spec(tmp_path), lambda split: (Reader(), TinyFields()),
        trainer_factory, {"learning_rate": tune.choice([1.0e-2])}, evaluate,
    )

    records = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    trials = [record for record in records if record["trial"]]
    finals = [record for record in records if not record["trial"]]
    assert trials and finals
    assert all(record["seed"] == record["context_seed"] == 7 for record in records)
    assert {record["epochs"] for record in trials} == {2}
    assert {record["epochs"] for record in finals} == {4}

    store = ArtifactStore(context.paths.run_dir)
    manifest = store.read_manifest()
    assert manifest["status"] == "completed"
    assert manifest["provenance"]["final_epochs"] == 4
    assert manifest["provenance"]["final_epochs_source"] == "2x_tune"
    best = store.read_best_config()
    assert (best["seed"], best["epochs"]) == (7, 4)


@pytest.mark.parametrize("reserved", ("seed", "epochs"))
def test_tuner_rejects_search_spaces_that_set_seed_or_epochs(reserved: str) -> None:
    with pytest.raises(ValueError, match="must not define"):
        Tuner(lambda config: None, {"width": 4, reserved: 1})


def test_tuner_objectives_may_keep_their_own_keys() -> None:
    """Only trainer factories get the library's seed and epochs."""
    tuner = Tuner.from_objective(lambda *args: None, {"seed": 1, "epochs": 2})
    assert tuner.search_space == {"seed": 1, "epochs": 2}


def test_build_tuning_config_derives_gpus_from_the_device(tmp_path: Path) -> None:
    cpu = build_tuning_config(metric="best_valid_loss", samples=10, epochs=6, seed=3,
                              device=torch.device("cpu"), temp_dir=tmp_path)
    assert cpu.resources_per_trial == {"cpu": 1, "gpu": 0}  # one CPU per trial by default
    assert (cpu.ray_runtime.num_cpus, cpu.ray_runtime.num_gpus) == (1, 0)
    assert (cpu.metric, cpu.mode, cpu.num_samples, cpu.max_epochs, cpu.optuna_seed) == (
        "best_valid_loss", "min", 10, 6, 3)
    assert (cpu.optuna_startup_trials, cpu.reduction_factor, cpu.best_scope) == (2, 3, "last")
    assert cpu.ray_runtime.temp_dir == tmp_path.resolve()
    assert cpu.ray_runtime.object_store_memory == 100 * 1024**2

    cuda = build_tuning_config(metric="m", samples=1, epochs=1, seed=0, device="cuda")
    assert cuda.resources_per_trial == {"cpu": 1, "gpu": 1}
    assert cuda.ray_runtime.num_gpus == 1


def test_build_tuning_config_keeps_ray_totals_separate_from_trial_resources() -> None:
    shared = build_tuning_config(metric="m", samples=1, epochs=1, seed=0, device="cuda",
                                 cpus_per_trial=1.5, max_concurrent_trials=3)
    assert shared.resources_per_trial == {"cpu": 1.5, "gpu": 1}
    assert (shared.ray_runtime.num_cpus, shared.ray_runtime.num_gpus) == (5, 3)

    explicit = build_tuning_config(metric="m", samples=1, epochs=1, seed=0, device="cuda",
                                   gpus_per_trial=0.5, ray_cpus=16, ray_gpus=2,
                                   runtime_env={"working_dir": "."})
    assert explicit.resources_per_trial == {"cpu": 1, "gpu": 0.5}
    assert (explicit.ray_runtime.num_cpus, explicit.ray_runtime.num_gpus) == (16, 2)
    assert explicit.ray_runtime.runtime_env == {"working_dir": "."}
