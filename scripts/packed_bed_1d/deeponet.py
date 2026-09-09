"""Tune and train direct and IPCA-POD DeepONets for packed-bed profiles."""

from __future__ import annotations

from datetime import datetime
import gc
import json
import os
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault("DDE_BACKEND", "pytorch")
# DeepXDE imports Matplotlib internally even though this runner does not plot.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

import optuna
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from chem_operator.datasets import (
    ChemOperatorDataset,
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    TargetTransformConfig,
)
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import (
    CoordinateScaler,
    DeepONetTrainingConfig,
    DeepXDEAdapter,
    PODTransform,
    fit_incremental_pod_dataset,
    fit_zscore_normalizer,
    run_deeponet_comparison,
    save_deeponet_comparison,
    tune_deeponet_hyperparameters,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = ExamplePaths.from_script(__file__, dataset="packed_bed_1D")
TUNE_STORAGE = PATHS.output / "ray_results"
FIELDS = ("T", "P", "X", "Z", "rhou", "velocity")
CONSTANTS = (
    "length",
    "radius",
    "porosity",
    "tortuosity",
    "particle_diameter",
    "specific_surface_area",
    "inlet_velocity",
    "wall_temperature",
    "heat_transfer_coefficient",
    "solve_energy",
    "membrane_present",
    "membrane_permeability",
    "membrane_thickness",
    "sweep_pressure",
    "inlet_nh3_mole_fraction",
)
COORDINATE_STRIDE = 1
RESAMPLE_POINTS = 128
SEED = 7
POD_VARIANCE_THRESHOLD = 0.999
RECONSTRUCTION_CASES = 2
MAX_TRAJECTORIES: int | None = None

MAX_EPOCHS = 40
NUM_SAMPLES = 10
MAX_CONCURRENT_TRIALS = 1
TIME_BUDGET_S = 5 * 60 * 60
CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0
PIN_MEMORY = bool(GPUS_PER_TRIAL)
RAY_OBJECT_STORE_BYTES = 100 * 1024**2
FINAL_DISPLAY_EVERY = 10


def search_space(model_kind: str, pod_components: int) -> dict[str, Any]:
    """Return fresh Ray Tune domains for one model's Optuna study."""

    return {
        "loss": "relative_l2",
        "width": tune.choice([128, 256, 512, 768]),
        "latent_width": (
            pod_components
            if model_kind == "pod"
            else tune.choice([8, 16, 32, 64])
        ),
        "branch_hidden_layers": tune.choice([2, 3, 4]),
        "trunk_hidden_layers": (
            0 if model_kind == "pod" else tune.choice([2, 3, 4])
        ),
        "activation": tune.choice(["gelu", "tanh"]),
        "learning_rate": tune.loguniform(5e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-6, 1e-4),
        "batch_size": tune.choice([16, 32]),
        "epochs": MAX_EPOCHS,
        "seed": SEED,
    }


def raw_dataset(data_dir: Path, split: str, stride: int) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        data_dir / f"packed_bed_1d_{split}.h5",
        task="operator_cartesian",
        coordinate_name="z",
        input_fields=FIELDS,
        output_fields=FIELDS,
        constant_inputs=CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        index_stride=stride,
        dtype=torch.float32,
    )


def limited(dataset: Dataset, maximum: int | None) -> Dataset:
    if maximum is None:
        return dataset
    return Subset(dataset, range(min(maximum, len(dataset))))


def processor(normalizer: ZScoreNormalizer) -> DataProcessor:
    return DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=FIELDS,
            constant_field_order=CONSTANTS,
        ),
        normalizer=normalizer,
        normalization_config=NormalizationConfig(enabled=True),
        target_transform=TargetTransformConfig(mode="state"),
    )


def adapter(dataset: Dataset, normalizer: ZScoreNormalizer) -> DeepXDEAdapter:
    return DeepXDEAdapter(
        dataset,
        processor(normalizer),
        format="cartesian_product",
        coordinate_name="z",
        include_constants=True,
        resample_points=RESAMPLE_POINTS,
        coordinate_mode="relative",
    )


def tune_packed_bed_deeponet(
    config: Mapping[str, Any],
    *,
    model_kind: str,
    normalizer: ZScoreNormalizer,
    pod: PODTransform,
) -> None:
    """Open lazy HDF5 datasets inside one Ray trial and close them afterward."""

    train_raw = raw_dataset(PATHS.data, "train", COORDINATE_STRIDE)
    valid_raw = raw_dataset(PATHS.data, "valid", COORDINATE_STRIDE)
    try:
        train = adapter(limited(train_raw, MAX_TRAJECTORIES), normalizer)
        validation = adapter(limited(valid_raw, MAX_TRAJECTORIES), normalizer)
        scaler = CoordinateScaler.fit(train[0]["trunk"].numpy())
        tune_deeponet_hyperparameters(
            config,
            train=train,
            validation=validation,
            coordinate_scaler=scaler,
            pod=pod if model_kind == "pod" else None,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
        )
    finally:
        train_raw.close()
        valid_raw.close()


def tune_model(
    name: str,
    *,
    normalizer: ZScoreNormalizer,
    pod: PODTransform,
    run_suffix: str,
) -> tuple[dict[str, Any], float, dict[str, int]]:
    """Tune one model family with Optuna search and ASHA early stopping."""

    optuna_search = OptunaSearch(
        metric="best_valid_loss",
        mode="min",
        sampler=optuna.samplers.TPESampler(
            seed=SEED,
            n_startup_trials=2,
            multivariate=True,
        ),
    )
    scheduler = ASHAScheduler(
        metric="best_valid_loss",
        mode="min",
        time_attr="training_iteration",
        max_t=MAX_EPOCHS,
        grace_period=max(1, MAX_EPOCHS // 3),
        reduction_factor=2,
    )
    parameterized = tune.with_parameters(
        tune_packed_bed_deeponet,
        model_kind=name,
        normalizer=normalizer,
        pod=pod,
    )
    trainable = tune.with_resources(
        parameterized,
        resources={"cpu": CPUS_PER_TRIAL, "gpu": GPUS_PER_TRIAL},
    )
    tuner = tune.Tuner(
        trainable,
        param_space=search_space(name, pod.n_components),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=NUM_SAMPLES,
            max_concurrent_trials=MAX_CONCURRENT_TRIALS,
            time_budget_s=TIME_BUDGET_S,
            reuse_actors=False,
        ),
        run_config=tune.RunConfig(
            name=f"packed_bed_1d_{name}_{run_suffix}",
            storage_path=str(TUNE_STORAGE.resolve()),
            verbose=1,
        ),
    )
    results = tuner.fit()
    best = results.get_best_result(
        metric="best_valid_loss",
        mode="min",
        scope="last",
    )
    best_config = dict(best.config)
    best_loss = float(best.metrics["best_valid_loss"])
    parameter_counts = {
        key: int(best.metrics[key])
        for key in ("n_params", "n_params_branch", "n_params_trunk")
    }
    print(f"Best {name} validation loss: {best_loss:.6e}")
    print(f"Best {name} configuration: {best_config}")
    print(f"Best {name} parameter counts: {parameter_counts}")
    return best_config, best_loss, parameter_counts


def training_config(
    config: Mapping[str, Any],
    epoch_multiplier: float = 1.0,
) -> DeepONetTrainingConfig:
    """Convert a resolved Ray configuration to the final training config."""

    return DeepONetTrainingConfig(
        loss=str(config.get("loss", "relative_l2")),
        epochs=int(config["epochs"] * epoch_multiplier),
        learning_rate=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
        batch_size=int(config["batch_size"]),
        width=int(config["width"]),
        latent_width=int(config["latent_width"]),
        branch_hidden_layers=int(config["branch_hidden_layers"]),
        trunk_hidden_layers=int(config["trunk_hidden_layers"]),
        activation=str(config["activation"]),
        display_every=FINAL_DISPLAY_EVERY,
        seed=int(config["seed"]),
    )


def json_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.item() if hasattr(value, "item") else value
        for key, value in config.items()
    }


def main() -> None:
    PATHS.output.mkdir(parents=True, exist_ok=True)
    PATHS.ray.mkdir(parents=True, exist_ok=True)
    stats_raw = raw_dataset(PATHS.data, "train", COORDINATE_STRIDE)
    try:
        print("Fitting packed-bed Z-score statistics from training trajectories ...")
        stats_loader = DataLoader(
            limited(stats_raw, MAX_TRAJECTORIES),
            batch_size=None,
            shuffle=False,
            num_workers=DATALOADER_WORKERS,
        )
        normalizer = fit_zscore_normalizer(stats_loader, FIELDS, CONSTANTS)
    finally:
        stats_raw.close()

    pod_raw = raw_dataset(PATHS.data, "train", COORDINATE_STRIDE)
    try:
        pod_dataset = adapter(
            limited(pod_raw, MAX_TRAJECTORIES),
            normalizer,
        )
        pod = fit_incremental_pod_dataset(
            pod_dataset,
            variance_threshold=POD_VARIANCE_THRESHOLD,
            num_workers=DATALOADER_WORKERS,
        )
        print(
            f"IPCA retained {pod.n_components} trajectory components for "
            f"{pod.cumulative_explained_variance:.6%} cumulative variance."
        )
    finally:
        pod_raw.close()
    del stats_loader, stats_raw, pod_dataset, pod_raw
    gc.collect()

    try:
        ray.init(
            ignore_reinit_error=True,
            include_dashboard=False,
            num_cpus=CPUS_PER_TRIAL,
            num_gpus=GPUS_PER_TRIAL,
            object_store_memory=RAY_OBJECT_STORE_BYTES,
            _temp_dir=str(PATHS.ray.resolve()),
        )
        run_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
        direct_config, direct_loss, direct_parameter_counts = tune_model(
            "direct",
            normalizer=normalizer,
            pod=pod,
            run_suffix=run_suffix,
        )
        pod_config, pod_loss, pod_parameter_counts = tune_model(
            "pod",
            normalizer=normalizer,
            pod=pod,
            run_suffix=run_suffix,
        )
    finally:
        ray.shutdown()

    tuning_summary = {
        "direct": {
            "best_valid_loss": direct_loss,
            "config": json_config(direct_config),
            "parameter_counts": direct_parameter_counts,
        },
        "pod": {
            "best_valid_loss": pod_loss,
            "config": json_config(pod_config),
            "parameter_counts": pod_parameter_counts,
        },
    }
    with (PATHS.output / "best_hyperparameters.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(tuning_summary, file, indent=2)

    train_raw = raw_dataset(PATHS.data, "train", COORDINATE_STRIDE)
    valid_raw = raw_dataset(PATHS.data, "valid", COORDINATE_STRIDE)
    test_raw = raw_dataset(PATHS.data, "test", COORDINATE_STRIDE)
    try:
        result = run_deeponet_comparison(
            adapter(limited(train_raw, MAX_TRAJECTORIES), normalizer),
            adapter(limited(valid_raw, MAX_TRAJECTORIES), normalizer),
            adapter(limited(test_raw, MAX_TRAJECTORIES), normalizer),
            normalizer,
            direct_config=training_config(direct_config, 2.0),
            pod_config=training_config(pod_config, 2.0),
            pod=pod,
            reconstruction_cases=RECONSTRUCTION_CASES,
            num_workers=DATALOADER_WORKERS,
            pin_memory=PIN_MEMORY,
        )
        save_deeponet_comparison(
            result,
            PATHS.output,
            problem="packed_bed_1d",
        )
        print(f"Artifacts written to {PATHS.output}")
    finally:
        train_raw.close()
        valid_raw.close()
        test_raw.close()


if __name__ == "__main__":
    main()
