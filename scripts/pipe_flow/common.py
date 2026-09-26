"""Shared pipe-flow dataset and DeepONet experiment configuration."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import os
from typing import Iterator

os.environ.setdefault("DDE_BACKEND", "pytorch")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("RAY_memory_monitor_refresh_ms", "0")

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
from chem_operator.experiments import (
    DeepONetTuningSettings,
    ExperimentSpec,
    RayRuntimeConfig,
    RunContext,
    WorkflowStages,
    add_run_arguments,
    add_workflow_arguments,
    fingerprint_path,
)
from chem_operator.models import (
    DeepONetAdapter,
    fit_zscore_normalizer,
)
from chem_operator.normalization import ZScoreNormalizer


PATHS = ExamplePaths.from_script(__file__, dataset="pipe_flow")
FILE_STEM = "hagen_poiseuille_pipe_flow"
PROBLEM_ID = "pipe_flow"
PROTOCOL_ID = "operator-cartesian-v1"
FIELDS = ("velocity",)
CONSTANTS = ("radius", "length", "dynamic_viscosity", "pressure_drop")
UNITS = {"velocity": "m/s"}

SEED = 42
RECONSTRUCTION_CASES = 3
MAX_TRAJECTORIES: int | None = None

MAX_EPOCHS = 20
NUM_SAMPLES = 8
MAX_CONCURRENT_TRIALS = 1
CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
DATALOADER_WORKERS = 0
PIN_MEMORY = bool(GPUS_PER_TRIAL)
RAY_OBJECT_STORE_BYTES = 100 * 1024**2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_workflow_arguments(parser)
    add_run_arguments(
        parser,
        default_runs_root=PATHS.root / "artifacts" / "runs",
    )
    return parser.parse_args()


def raw_dataset(split: str) -> ChemOperatorDataset:
    return ChemOperatorDataset(
        PATHS.data / f"{FILE_STEM}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="r",
        input_fields=FIELDS,
        output_fields=FIELDS,
        constant_inputs=CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )


def limited(dataset: Dataset) -> Dataset:
    if MAX_TRAJECTORIES is None:
        return dataset
    return Subset(dataset, range(min(MAX_TRAJECTORIES, len(dataset))))


def adapter(dataset: Dataset, normalizer: ZScoreNormalizer) -> DeepONetAdapter:
    processor = DataProcessor(
        field_packer=FieldPacker(
            channel_axis="last",
            variable_field_order=FIELDS,
            constant_field_order=CONSTANTS,
        ),
        normalizer=normalizer,
        normalization_config=NormalizationConfig(enabled=True),
        target_transform=TargetTransformConfig(mode="state"),
    )
    return DeepONetAdapter(
        dataset,
        processor,
        format="cartesian_product",
        coordinate_name="r",
        include_constants=True,
        include_input_state=False,
        coordinate_mode="relative",
    )


def prepare_normalizer() -> ZScoreNormalizer:
    raw = raw_dataset("train")
    try:
        normalizer = fit_zscore_normalizer(
            DataLoader(limited(raw), batch_size=None, shuffle=False),
            FIELDS,
            CONSTANTS,
        )
    finally:
        raw.close()
    return normalizer


@contextmanager
def tuning_data(
    context: RunContext,
    *,
    normalizer: ZScoreNormalizer,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter]]:
    del context
    train_raw, valid_raw = raw_dataset("train"), raw_dataset("valid")
    try:
        yield (
            adapter(limited(train_raw), normalizer),
            adapter(limited(valid_raw), normalizer),
        )
    finally:
        train_raw.close()
        valid_raw.close()


@contextmanager
def final_data(
    *,
    normalizer: ZScoreNormalizer,
) -> Iterator[tuple[DeepONetAdapter, DeepONetAdapter, DeepONetAdapter]]:
    raw = tuple(raw_dataset(split) for split in ("train", "valid", "test"))
    try:
        yield tuple(adapter(limited(item), normalizer) for item in raw)  # type: ignore[misc]
    finally:
        for item in raw:
            item.close()


def experiment_spec(model_id: str) -> ExperimentSpec:
    return ExperimentSpec(
        problem_id=PROBLEM_ID,
        model_id=model_id,
        benchmark_protocol_id=PROTOCOL_ID,
        dataset_fingerprints={
            name: fingerprint_path(PATHS.data / f"{FILE_STEM}_{split}.h5")
            for name, split in (
                ("train", "train"),
                ("validation", "valid"),
                ("test", "test"),
            )
        },
        fields=FIELDS,
        channels=FIELDS,
        units=UNITS,
        coordinates=("r",),
        selected_test_case_ids=range(RECONSTRUCTION_CASES),
        tuning_budget={"samples": NUM_SAMPLES, "epochs": MAX_EPOCHS},
        project_root=PATHS.root,
    )


def validate_stages(stages: WorkflowStages) -> None:
    if stages.generate:
        raise ValueError("Use the pipe-flow dataset-generation script first.")
    if stages.plot:
        raise ValueError("Use scripts/pipe_flow/plot.py for saved runs.")
    if stages.train_config != "best":
        raise ValueError("DeepONet training currently requires --train-config best.")


def tuning_settings() -> DeepONetTuningSettings:
    return DeepONetTuningSettings(
        max_epochs=MAX_EPOCHS,
        num_samples=NUM_SAMPLES,
        max_concurrent_trials=MAX_CONCURRENT_TRIALS,
        cpus_per_trial=CPUS_PER_TRIAL,
        gpus_per_trial=GPUS_PER_TRIAL,
        ray_runtime=RayRuntimeConfig(
            num_cpus=CPUS_PER_TRIAL,
            num_gpus=GPUS_PER_TRIAL,
            object_store_memory=RAY_OBJECT_STORE_BYTES,
            temp_dir=PATHS.ray.resolve(),
        ),
    )

# PhysicsNeMo study: reuse the same scientific run identity and field contract.
from dataclasses import dataclass, replace
from pathlib import Path
from torch.utils.data import TensorDataset
from chem_operator.experiments import operator_spec, parse_operator_args
BRANCH_NAMES = CONSTANTS
ALL_CONSTANTS = BRANCH_NAMES + ("density", "pressure_gradient")
@dataclass(frozen=True)
class Normalization:
    branch_mean: torch.Tensor
    branch_std: torch.Tensor
    velocity_mean: torch.Tensor
    velocity_std: torch.Tensor


@dataclass(frozen=True)
class PipeData:
    branch: torch.Tensor
    coordinates: torch.Tensor
    radius: torch.Tensor
    viscosity: torch.Tensor
    pressure_gradient: torch.Tensor
    target: torch.Tensor
    target_normalized: torch.Tensor | None = None

    def normalized(self, statistics: Normalization) -> "PipeData":
        return replace(
            self,
            branch=(self.branch - statistics.branch_mean)
            / statistics.branch_std,
            target_normalized=(self.target - statistics.velocity_mean)
            / statistics.velocity_std,
        )

    def dataset(self) -> TensorDataset:
        if self.target_normalized is None:
            raise RuntimeError("Normalize a split before constructing a loader.")
        return TensorDataset(
            self.branch,
            self.coordinates,
            self.radius,
            self.viscosity,
            self.pressure_gradient,
            self.target_normalized,
            self.target,
        )


def load_physics_split(data_dir: Path, split: str, maximum: int | None) -> PipeData:
    torch.set_default_device("cpu")
    dataset = ChemOperatorDataset(
        data_dir / f"hagen_poiseuille_pipe_flow_{split}.h5",
        task="operator_cartesian",
        coordinate_name="r",
        input_fields=("velocity",),
        output_fields=("velocity",),
        constant_inputs=ALL_CONSTANTS,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )
    count = len(dataset) if maximum is None else min(maximum, len(dataset))
    branch: list[torch.Tensor] = []
    coordinates: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    radii: list[torch.Tensor] = []
    viscosities: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    try:
        for index in range(count):
            sample = dataset[index]
            constants = sample["constant_inputs"]
            branch.append(torch.stack([constants[name] for name in BRANCH_NAMES]))
            radii.append(constants["radius"].reshape(1))
            viscosities.append(constants["dynamic_viscosity"].reshape(1))
            gradients.append(constants["pressure_gradient"].reshape(1))
            coordinates.append(
                torch.cat(
                    (
                        sample["input_coordinates"]["r"],
                        sample["output_coordinates"]["r"],
                    )
                ).reshape(-1, 1)
            )
            targets.append(
                torch.cat(
                    (
                        sample["input_fields"]["velocity"],
                        sample["output_fields"]["velocity"],
                    )
                ).reshape(-1, 1)
            )
    finally:
        dataset.close()
    return PipeData(
        branch=torch.stack(branch),
        coordinates=torch.stack(coordinates),
        radius=torch.stack(radii),
        viscosity=torch.stack(viscosities),
        pressure_gradient=torch.stack(gradients),
        target=torch.stack(targets),
    )


def fit_physics_normalization(data: PipeData) -> Normalization:
    return Normalization(
        branch_mean=data.branch.mean(dim=0),
        branch_std=data.branch.std(dim=0, unbiased=False).clamp_min(1.0e-8),
        velocity_mean=data.target.mean(),
        velocity_std=data.target.std(unbiased=False).clamp_min(1.0e-8),
    )

class PhysicsPipeDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.tensors = data.dataset()

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, index):
        values = self.tensors[index]
        return dict(zip(("branch", "coordinates", "radius", "viscosity", "pressure_gradient", "y", "reference"), values))

    def close(self):
        """This dataset owns in-memory tensors; its HDF5 reader is already closed."""


def physics_experiment_spec(args, model_id):
    return operator_spec(args, PATHS, PROBLEM_ID, model_id, FILE_STEM, FIELDS, UNITS, ("r",))
