"""Shared problem specification and data preparation."""

from __future__ import annotations
import os
import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
os.environ.setdefault("DDE_BACKEND", "pytorch")
from typing import Any, Callable, Mapping
import numpy as np
from physicsnemo.sym.eq.pde import PDE
import torch
from ray import tune
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from chem_operator._models.deeponet import DeepONetTrainingConfig
from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import FNOAdapter, FNOChannel, fit_fno_zscore_normalizer, fit_zscore_normalizer
from chem_operator.normalization import ZScoreNormalizer, normalizer_from_state_dict
from chem_operator.reactors.pipe_flow_transient.dataset_generator import TransientHagenPoiseuille
from chem_operator.experiments import CompositeLossTrainer, EvaluationOutcome, LossTerm, StreamingRegressionMetrics, parse_operator_args, operator_spec, run_operator
PATHS = ExamplePaths.from_script(__file__, dataset="pipe_flow_transient")

FIELD_NAMES = ("velocity",)
MODEL_CONSTANT_NAMES = (
    "radius",
    "length",
    "dynamic_viscosity",
    "pressure_drop",
)
PHYSICS_CONSTANT_NAMES = (
    "radius", "length", "dynamic_viscosity", "pressure_drop", "density",
)
INPUT_CHANNELS = tuple(
    FNOChannel(name, "constant", name) for name in MODEL_CONSTANT_NAMES
)
OUTPUT_CHANNELS = (FNOChannel("velocity", "field", "velocity"),)
FILE_STEM = "transient_hagen_poiseuille_pipe_flow"
METRIC = "best_valid_relative_l2"
SEED = 42

PDE_REGISTRY: dict[str, type[PDE]] = {
    TransientHagenPoiseuille.__name__: TransientHagenPoiseuille,
}


MAX_TRAIN_TRAJECTORIES = None
CONSTANT_NAMES = MODEL_CONSTANT_NAMES
ROLLOUT_CONSTANT_NAMES = PHYSICS_CONSTANT_NAMES
class PhysicsFNOAdapter(FNOAdapter):
    """Add PDE provenance and dimensional constants to an FNO sample."""

    def __getitem__(self, index: int) -> dict[str, Any]:
        # Build from physical_item directly so each lazy HDF5 case is read once.
        physical = self.physical_item(index)
        metadata = physical.get("metadata", {})
        if "physicsnemo_pde" not in metadata:
            raise KeyError(
                "Dataset metadata is missing required 'physicsnemo_pde'."
            )
        parameters = metadata.get("params", {})
        try:
            physics_constants = torch.stack(
                [
                    torch.as_tensor(parameters[name], dtype=physical["x"].dtype)
                    .reshape(())
                    for name in PHYSICS_CONSTANT_NAMES
                ]
            )
        except KeyError as exc:
            raise KeyError(
                f"Dataset metadata params are missing physics constant "
                f"{exc.args[0]!r}."
            ) from exc
        model_input = torch.stack(
            [
                self.normalizer.normalize(physical["x"][channel], definition.label)
                for channel, definition in enumerate(self.input_channels)
            ]
        )
        target = torch.stack(
            [
                self.normalizer.normalize(physical["y"][channel], definition.label)
                for channel, definition in enumerate(self.output_channels)
            ]
        )
        return {
            "x": model_input,
            "y": target,
            **{name: physical[name] for name in self.coordinate_names},
            "physics_constants": physics_constants,
            "physicsnemo_pde": str(metadata["physicsnemo_pde"]),
        }


def raw_dataset(data_dir: Path, split: str) -> ChemOperatorDataset:
    """Open one complete generated transient trajectory per HDF5 case."""
    return ChemOperatorDataset(
        data_dir / f"{FILE_STEM}_{split}.h5",
        task="operator_cartesian",
        coordinate_name="t",
        input_fields=FIELD_NAMES,
        output_fields=FIELD_NAMES,
        constant_inputs=PHYSICS_CONSTANT_NAMES,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )


def dataset_pde_name(
    dataset: ChemOperatorDataset,
    maximum: int | None = None,
) -> str:
    """Return the single supported PDE declared by a dataset subset."""
    count = len(dataset) if maximum is None else min(len(dataset), maximum)
    if count == 0:
        raise RuntimeError("Cannot resolve a PDE from an empty dataset.")
    names: set[str] = set()
    for index in range(count):
        metadata = dataset[index].get("metadata", {})
        try:
            names.add(str(metadata["physicsnemo_pde"]))
        except KeyError as exc:
            raise KeyError(
                "Dataset metadata is missing required 'physicsnemo_pde'."
            ) from exc
    if len(names) != 1:
        raise ValueError(
            "Dataset trajectories must declare exactly one physicsnemo_pde; "
            f"found {sorted(names)}."
        )
    name = names.pop()
    if name not in PDE_REGISTRY:
        raise ValueError(
            f"Unsupported metadata physicsnemo_pde {name!r}; "
            f"supported values are {sorted(PDE_REGISTRY)}."
        )
    return name


def make_adapter(
    data_dir: Path,
    split: str,
    normalizer: ZScoreNormalizer,
    maximum: int | None,
) -> tuple[ChemOperatorDataset, PhysicsFNOAdapter]:
    """Return an open raw split and its physics-aware FNO adapter."""
    dataset = raw_dataset(data_dir, split)
    adapter = PhysicsFNOAdapter(
        dataset,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=("t", "r"),
        max_trajectories=maximum,
    )
    return dataset, adapter


def fit_normalizer(data_dir: Path) -> ZScoreNormalizer:
    """Fit preprocessing statistics using only the training subset."""
    dataset = raw_dataset(data_dir, "train")
    count = len(dataset)
    if MAX_TRAIN_TRAJECTORIES is not None:
        count = min(count, MAX_TRAIN_TRAJECTORIES)
    try:
        dataset_pde_name(dataset, 1)
        return fit_fno_zscore_normalizer(
            Subset(dataset, range(count)),
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        )
    finally:
        dataset.close()


def adapter_radial_spacing(dataset: PhysicsFNOAdapter) -> float:
    """Return the first case's validated dimensionless radial spacing.

    Every training/evaluation batch is checked again by :func:`physics_losses`,
    avoiding a separate eager pass through a large lazy HDF5 split.
    """
    if len(dataset) == 0:
        raise RuntimeError("Cannot determine grid spacing from an empty adapter.")
    sample = dataset[0]
    radius = sample["physics_constants"][0]
    x = sample["r"] / radius
    differences = x[1:] - x[:-1]
    if not torch.allclose(
        differences,
        differences[0].expand_as(differences),
        rtol=1.0e-5,
        atol=1.0e-7,
    ):
        raise ValueError("Physics-informed FNO requires a uniform radial grid.")
    return float(differences[0])


def training_pde_name(data_dir: Path) -> str:
    """Resolve the governing PDE from the generated training data."""
    dataset = raw_dataset(data_dir, "train")
    try:
        return dataset_pde_name(dataset, 1)
    finally:
        dataset.close()

PROBLEM_ID = "pipe_flow_transient"

def experiment_spec(args, model_id):
    return operator_spec(args, PATHS, PROBLEM_ID, model_id, FILE_STEM,
                         FIELD_NAMES, {"velocity": "m/s"}, ("t", "r"))

def parse_args(*, epochs, tune_epochs, samples, parser=None):
    return parse_operator_args(
        PATHS,
        epochs=epochs,
        tune_epochs=tune_epochs,
        samples=samples,
        parser=parser,
    )


def rollout_raw_dataset(data_dir: Path, split: str, *, full_trajectory: bool = False) -> ChemOperatorDataset:
    """Read adjacent profile pairs or full trajectories for rollout scoring."""
    return ChemOperatorDataset(
        data_dir / f"{FILE_STEM}_{split}.h5",
        task="operator_cartesian" if full_trajectory else "rollout",
        coordinate_name="t",
        input_fields=FIELD_NAMES,
        output_fields=FIELD_NAMES,
        constant_inputs=ROLLOUT_CONSTANT_NAMES,
        n_steps_input=1,
        n_steps_output=1,
        dtype=torch.float32,
    )


def rollout_constants(sample: Mapping[str, Any]) -> torch.Tensor:
    return torch.stack([
        sample["constant_inputs"][name].reshape(()) for name in ROLLOUT_CONSTANT_NAMES
    ])


def fourier_times(t_now: torch.Tensor, t_next: torch.Tensor, constants: torch.Tensor):
    radius, _, viscosity, _, density = constants.unbind(dim=-1)
    scale = (viscosity / density) / radius.square()
    return t_now * scale, (t_next - t_now) * scale


class RolloutAdapter(Dataset):
    """Normalize adjacent radial profiles and retain physical metadata."""

    def __init__(self, dataset: ChemOperatorDataset, normalizer: ZScoreNormalizer, n_radial: int):
        self.dataset = dataset
        self.normalizer = normalizer
        self.n_radial = n_radial

    def __len__(self) -> int:
        return len(self.dataset)

    def branch(self, velocity: torch.Tensor, constants: torch.Tensor,
               t_now: torch.Tensor, t_next: torch.Tensor) -> torch.Tensor:
        profile = self.normalizer.normalize(velocity, "velocity")
        parameters = torch.stack([
            self.normalizer.normalize(constants[..., index], name)
            for index, name in enumerate(ROLLOUT_CONSTANT_NAMES)
        ], dim=-1)
        tau, dtau = fourier_times(t_now, t_next, constants)
        return torch.cat((profile, parameters, tau.unsqueeze(-1), dtau.unsqueeze(-1)), dim=-1)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.dataset[index]
        current = sample["input_fields"]["velocity"][0]
        following = sample["output_fields"]["velocity"][0]
        radius_grid = sample["output_coordinates"]["r"]
        if current.numel() != self.n_radial or following.numel() != self.n_radial or radius_grid.numel() != self.n_radial:
            raise ValueError("Rollout sample radial resolution differs from training resolution.")
        t_now = sample["input_coordinates"]["t"][0]
        t_next = sample["output_coordinates"]["t"][0]
        if t_next <= t_now:
            raise ValueError("Rollout times must increase.")
        constants = rollout_constants(sample)
        radius_fraction = radius_grid / constants[0]
        if not torch.all(radius_fraction[1:] > radius_fraction[:-1]):
            raise ValueError("Radial coordinates must increase.")
        target = self.normalizer.normalize(following, "velocity")
        return {
            "branch": self.branch(current, constants, t_now, t_next),
            "x": radius_fraction,
            "y": target,
            "target": target,
            "current": current,
            "t_now": t_now,
            "t_next": t_next,
            "r": radius_grid,
            "constants": constants,
            "physicsnemo_pde": str(sample["metadata"]["physicsnemo_pde"]),
        }


def fit_rollout_normalizer(data_dir: Path) -> ZScoreNormalizer:
    """Fit branch and target normalization on training trajectories only."""
    dataset = rollout_raw_dataset(data_dir, "train", full_trajectory=True)
    try:
        dataset_pde_name(dataset)
        return fit_zscore_normalizer(dataset, FIELD_NAMES, ROLLOUT_CONSTANT_NAMES)
    finally:
        dataset.close()


def make_rollout_adapter(data_dir: Path, split: str, normalizer: ZScoreNormalizer,
                         n_radial: int) -> tuple[ChemOperatorDataset, RolloutAdapter]:
    dataset = rollout_raw_dataset(data_dir, split)
    return dataset, RolloutAdapter(dataset, normalizer, n_radial)


def evaluate_rollout_run(trainer, data, context, normalizer: ZScoreNormalizer,
                         data_dir: Path, n_radial: int, *, cases: int = 2,
                         max_cases: int | None = None) -> EvaluationOutcome:
    """Score teacher-forced windows and recursive trajectories in physical units."""
    model = trainer.model.to(context.device).eval()
    one_step = StreamingRegressionMetrics()
    rollout = StreamingRegressionMetrics()
    arrays = {key: [] for key in ("case_ids", "reference", "prediction", "t", "r")}
    started = time.perf_counter()
    with torch.no_grad():
        for batch in DataLoader(data, batch_size=int(trainer.config["batch_size"])):
            predicted = model(batch["branch"].to(context.device), batch["x"].to(context.device))
            predicted = normalizer.denormalize(predicted.cpu(), "velocity")
            reference = normalizer.denormalize(batch["y"], "velocity")
            one_step.update(predicted, reference)

        raw = rollout_raw_dataset(data_dir, "test", full_trajectory=True)
        adapter = RolloutAdapter(raw, normalizer, n_radial)
        try:
            count = len(raw) if max_cases is None else min(len(raw), max_cases)
            for case_index in range(count):
                sample = raw[case_index]
                constants = rollout_constants(sample)
                r = sample["output_coordinates"]["r"]
                if r.numel() != n_radial:
                    raise ValueError("Test trajectory radial resolution differs from training resolution.")
                x = (r / constants[0]).to(context.device).unsqueeze(0)
                times = torch.cat((sample["input_coordinates"]["t"], sample["output_coordinates"]["t"]))
                reference = torch.cat((sample["input_fields"]["velocity"], sample["output_fields"]["velocity"]))
                current = reference[0]
                profiles = [current]
                for step in range(1, len(times)):
                    branch = adapter.branch(current, constants, times[step - 1], times[step]).to(context.device).unsqueeze(0)
                    predicted = model(branch, x)
                    current = normalizer.denormalize(predicted.squeeze(0).cpu(), "velocity")
                    profiles.append(current)
                prediction = torch.stack(profiles)
                rollout.update(prediction[1:], reference[1:])
                if len(arrays["case_ids"]) < cases:
                    arrays["case_ids"].append(sample["metadata"]["case_index"])
                    arrays["reference"].append(reference.unsqueeze(0).numpy())
                    arrays["prediction"].append(prediction.unsqueeze(0).numpy())
                    arrays["t"].append(times.numpy())
                    arrays["r"].append(r.numpy())
        finally:
            raw.close()
    elapsed = time.perf_counter() - started
    scores = {f"one_step_{key}": value for key, value in one_step.compute().items()}
    scores.update(rollout.compute())
    diagnostics = trainer.evaluate_losses(data, context)
    scores.update({key: value for key, value in diagnostics.items() if key.endswith("_loss")})
    return EvaluationOutcome(
        scores,
        {**{key: np.asarray(value) for key, value in arrays.items()},
         "labels": np.asarray(FIELD_NAMES), "coordinate_names": np.asarray(("t", "r"))},
        elapsed,
    )


class RolloutDeepXDEGrid(nn.Module):
    """Present a shared radial grid to a DeepXDE Cartesian network."""

    def __init__(self, network: nn.Module, n_radial: int):
        super().__init__()
        self.network = network
        self.n_radial = n_radial

    def forward(self, branch: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if branch.ndim != 2 or x.ndim != 2 or branch.shape[0] != x.shape[0]:
            raise ValueError("Expected branch [batch, features] and x [batch, radius].")
        if x.shape[1] != self.n_radial:
            raise ValueError("Rollout radial grid differs from the model resolution.")
        if not torch.allclose(x, x[0:1].expand_as(x), rtol=1e-5, atol=1e-7):
            raise ValueError("DeepXDE Cartesian batches require a shared radial grid.")
        trunk = (2.0 * x[0] - 1.0).unsqueeze(-1)
        return self.network((branch, trunk))


def rollout_batch(batch: Mapping[str, Any], device: torch.device, dtype: torch.dtype):
    branch = batch["branch"].to(device=device, dtype=dtype)
    x = batch["x"].to(device=device, dtype=dtype)
    target = batch["y"].to(device=device, dtype=dtype)
    return (branch, x), target, batch


def rollout_architecture(config: Mapping[str, Any]) -> DeepONetTrainingConfig:
    """Build the architecture options used by both DeepXDE rollout models."""
    return DeepONetTrainingConfig(
        width=int(config["width"]),
        latent_width=int(config["latent_width"]),
        branch_hidden_layers=int(config["branch_hidden_layers"]),
        trunk_hidden_layers=int(config["trunk_hidden_layers"]),
        activation=str(config["activation"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )


class RolloutDeepXDETrainerBase(CompositeLossTrainer):
    """Shared rollout losses, metrics, and checkpoint metadata."""

    def __init__(self, config: Mapping[str, Any], normalizer: ZScoreNormalizer,
                 n_radial: int, *, model_id: str,
                 model_builder: Callable[[Mapping[str, Any], int], nn.Module],
                 pod_metadata: Mapping[str, Any] | None = None):
        self.n_radial = n_radial
        self.model_id = model_id
        self.normalizer = normalizer
        self.model_builder = model_builder
        super().__init__(
            lambda cfg: self.model_builder(cfg, self.n_radial),
            config,
            batch_adapter=rollout_batch,
            loss_terms=[LossTerm("data_loss", lambda prediction, target, batch: F.mse_loss(prediction, target))],
            selection_metric="relative_l2",
            metric_adapter=lambda prediction, target, batch: (
                self.normalizer.denormalize(prediction, "velocity"),
                self.normalizer.denormalize(target, "velocity"),
            ),
            checkpoint_metadata={
                "normalizer": normalizer.state_dict(),
                "n_radial": n_radial,
                "model_id": model_id,
                "constant_names": list(ROLLOUT_CONSTANT_NAMES),
                "branch_time_features": ["fourier_time", "fourier_step"],
                "pod": pod_metadata,
            },
        )

    def _restore_model_metadata(self, metadata: Mapping[str, Any]) -> None:
        """Allow model-specific trainers to restore state before model creation."""

    def _model_from_payload(self, payload, *, device, dtype):
        metadata = payload["metadata"]
        if metadata["model_id"] != self.model_id:
            raise ValueError("Checkpoint model ID does not match this trainer.")
        self.n_radial = int(metadata["n_radial"])
        self.normalizer = normalizer_from_state_dict(metadata["normalizer"])
        self._restore_model_metadata(metadata)
        return super()._model_from_payload(payload, device=device, dtype=dtype)


def rollout_search_space() -> dict[str, Any]:
    """Hyperparameters shared by direct and POD DeepONet rollouts."""
    return {
        "width": tune.choice([64, 128]),
        "branch_hidden_layers": tune.choice([2, 3]),
        "activation": tune.choice(["gelu", "tanh"]),
        "learning_rate": tune.loguniform(1e-4, 5e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-4),
        "batch_size": tune.choice([4, 8, 16]),
    }


def prepare_rollout_run():
    """Parse common CLI options and fit training-only preprocessing."""
    args = parse_args(epochs=20, tune_epochs=10, samples=8)
    normalizer = fit_rollout_normalizer(args.data_dir)
    raw = rollout_raw_dataset(args.data_dir, "train")
    try:
        n_radial = int(raw[0]["input_fields"]["velocity"].shape[-1])
    finally:
        raw.close()
    return args, normalizer, n_radial


def run_rollout_experiment(args, normalizer: ZScoreNormalizer, n_radial: int,
                           model_id: str, trainer_factory, search_space) -> None:
    """Run the common tuning, training, and rollout evaluation workflow."""
    from dataclasses import replace

    spec = replace(
        experiment_spec(args, model_id),
        benchmark_protocol_id="rollout-one-step-v1",
    )
    run_operator(
        args, PATHS, spec,
        lambda split: make_rollout_adapter(args.data_dir, split, normalizer, n_radial),
        trainer_factory,
        search_space,
        lambda trainer, data, context: evaluate_rollout_run(
            trainer, data, context, normalizer, args.data_dir, n_radial,
            cases=args.plot_cases, max_cases=args.max_cases,
        ),
    )

__all__ = ['PATHS', 'FIELD_NAMES', 'MODEL_CONSTANT_NAMES', 'PHYSICS_CONSTANT_NAMES', 'INPUT_CHANNELS', 'OUTPUT_CHANNELS', 'FILE_STEM', 'METRIC', 'SEED', 'PDE_REGISTRY', 'MAX_TRAIN_TRAJECTORIES', 'CONSTANT_NAMES', 'PhysicsFNOAdapter', 'raw_dataset', 'dataset_pde_name', 'make_adapter', 'fit_normalizer', 'adapter_radial_spacing', 'training_pde_name', 'PROBLEM_ID', 'experiment_spec', 'parse_args']
