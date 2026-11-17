"""Shared problem specification and data preparation."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from functools import partial
import math
import os
import re
from typing import Any
import numpy as np
from scipy.interpolate import RegularGridInterpolator
import torch
from torch.utils.data import Dataset, Subset
from chem_operator.datasets import ChemOperatorDataset
from chem_operator.example_paths import ExamplePaths
from chem_operator.models import (
    FNOAdapter, FNOChannel, PODTransform, fit_fno_zscore_normalizer, relative_l2_loss,
)
from chem_operator.normalization import ZScoreNormalizer
from chem_operator.experiments import (
    DeepONetTrainer,
    DeepONetTuningSettings,
    EvaluationOutcome,
    OperatorTrainer,
    LossTerm,
    RunContext,
    deeponet_training_config,
    deeponet_tuner,
    evaluate_fields,
    operator_ray_runtime,
    operator_spec,
    parse_operator_args,
    run_operator,
    unwrap_dataset,
)
PATHS = ExamplePaths.from_script(__file__, dataset="q2d_cmr")

FILE_STEM = "q2d_cmr"
SEED = 42
METRIC = "best_valid_loss"

TUNE_SAMPLES = 20
TUNE_EPOCHS = 30
FINAL_EPOCHS = 100
EVALUATION_BATCH_SIZE = 2

CPUS_PER_TRIAL = 2
GPUS_PER_TRIAL = 1 if torch.cuda.is_available() else 0
MAX_CONCURRENT_TRIALS = 1

LATENCY_WARMUPS = 5
LATENCY_REPEATS = 20
RELATIVE_L2_EPS = 1.0e-8
AMORTIZED_EVALUATION_COUNTS = (1, 10, 100, 1000)
REPRESENTATIVE_RESOLUTION_COUNT = 4

MESH_FILE_PATTERN = re.compile(r"q2d_cmr_(\d+)_(\d+)_test\.h5")
COORDINATE_NAMES = ("z", "r")

INPUT_CHANNELS = (
    FNOChannel(
        "T0",
        "parameter",
        "T0",
        display_name="Inlet temperature",
        unit="K",
    ),
    FNOChannel(
        "SCCM",
        "constant",
        "SCCM",
        display_name="Inlet flow rate",
        unit="sccm",
    ),
)

OUTPUT_CHANNELS = (
    # FNOChannel(
    #     "velocity_axial",
    #     "field",
    #     "velocity_axial",
    #     display_name="Axial velocity",
    #     unit="m/s",
    # ),
    FNOChannel(
        "X_CH4",
        "species",
        "X",
        species="CH4",
        display_name="Methane",
        unit="-",
    ),
    FNOChannel(
        "X_H2",
        "species",
        "X",
        species="H2",
        display_name="Hydrogen",
        unit="-",
    ),
    FNOChannel(
        "theta_C(s)",
        "species",
        "theta",
        species="C(s)",
        display_name="Carbon Accumulation",
        unit="-",
    ),
)
OUTPUT_LABELS = tuple(channel.label for channel in OUTPUT_CHANNELS)


@dataclass(frozen=True)
class MeshFile:
    """One resolution-sweep HDF5 file."""

    path: Path
    n_z: int
    n_r: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.n_z, self.n_r

    @property
    def mesh_points(self) -> int:
        return self.n_z * self.n_r


def raw_dataset(path: Path) -> ChemOperatorDataset:
    """Open one complete steady Q2D field per HDF5 case."""
    field_names = FNOAdapter.required_field_names(
        INPUT_CHANNELS,
        OUTPUT_CHANNELS,
    )
    if not field_names:
        raise ValueError("At least one configured channel must read an HDF5 field.")
    return ChemOperatorDataset(
        path,
        task="field_map",
        coordinate_name="z",
        input_fields=field_names,
        output_fields=field_names,
        constant_inputs=FNOAdapter.required_constant_names(
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        ),
        n_steps_input=1,
        dtype=torch.float32,
    )


def field_adapter(
    dataset: Dataset,
    normalizer: ZScoreNormalizer,
    shape: tuple[int, int] | None = None,
) -> FNOAdapter:
    """View a raw dataset as normalized channel-first fields on one mesh."""
    return FNOAdapter(
        dataset,
        normalizer,
        input_channels=INPUT_CHANNELS,
        output_channels=OUTPUT_CHANNELS,
        coordinate_names=COORDINATE_NAMES,
        spatial_shape=shape,
    )


def _case_geometry(metadata: Mapping[str, Any]) -> tuple[float, float]:
    """Return physical axial length and lumen radius for domain validation."""
    inputs = metadata.get("input_values", {})
    try:
        return float(inputs["LENGTH_LUMEN"]), float(inputs["CHANNELRAD_LUMEN"])
    except KeyError as exc:
        raise KeyError("Q2D metadata is missing its physical domain geometry.") from exc


def _same_geometry(first: Sequence[float], second: Sequence[float]) -> bool:
    return bool(np.allclose(first, second, rtol=1.0e-10, atol=1.0e-12))


def fit_normalizer(
    path: Path,
) -> tuple[ZScoreNormalizer, tuple[float, float], tuple[int, int]]:
    """Fit training-only statistics and discover the shared domain and mesh."""
    dataset = raw_dataset(path)
    try:
        if len(dataset) == 0:
            raise RuntimeError("The training dataset contains no cases.")
        normalizer = fit_fno_zscore_normalizer(
            dataset,
            INPUT_CHANNELS,
            OUTPUT_CHANNELS,
        )
        first = field_adapter(dataset, normalizer).physical_item(0)
        geometry = _case_geometry(first["metadata"])
        shape = tuple(int(value) for value in first["y"].shape[-2:])
        training = field_adapter(dataset, normalizer, shape)
        for index in range(1, len(training)):
            physical = training.physical_item(index)
            if not _same_geometry(_case_geometry(physical["metadata"]), geometry):
                raise ValueError("Training cases do not share one physical domain.")
    finally:
        dataset.close()
    return normalizer, geometry, shape


def make_adapter(
    path: Path,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    shape: tuple[int, int] | None = None,
) -> tuple[ChemOperatorDataset, FNOAdapter]:
    """Open a raw dataset and its normalized Q2D adapter on the training domain."""
    raw = raw_dataset(path)
    try:
        current_geometry = _case_geometry(raw[0]["metadata"])
        if not _same_geometry(current_geometry, geometry):
            raise ValueError(
                f"Dataset geometry {current_geometry} differs from training "
                f"geometry {geometry}."
            )
        return raw, field_adapter(raw, normalizer, shape)
    except BaseException:
        raw.close()
        raise


def checkpoint_metadata(
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    shape: tuple[int, int],
    *,
    coordinates: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> dict[str, Any]:
    """Describe the data contract every Q2D checkpoint carries."""
    metadata: dict[str, Any] = {
        "normalizer": normalizer.state_dict(),
        "geometry": list(geometry),
        "training_shape": list(shape),
        "input_channels": [channel.label for channel in INPUT_CHANNELS],
        "output_channels": list(OUTPUT_LABELS),
        "coordinate_names": list(COORDINATE_NAMES),
    }
    if coordinates is not None:
        metadata["training_coordinates"] = {
            axis: value.clone() for axis, value in zip(COORDINATE_NAMES, coordinates)
        }
    return metadata


def relative_l2_per_case(
    prediction: torch.Tensor, reference: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-case relative L2 over all channels and per channel.

    Both arguments are channel-first ``(cases, channels, *spatial)`` tensors.
    """
    overall = relative_l2_loss(prediction, reference, reduction="none", eps=RELATIVE_L2_EPS)
    per_field = relative_l2_loss(
        prediction.flatten(0, 1), reference.flatten(0, 1),
        reduction="none", eps=RELATIVE_L2_EPS,
    ).reshape(reference.shape[:2])
    return overall, per_field


def interpolate_fields(
    fields: np.ndarray,
    source_z: np.ndarray,
    source_r: np.ndarray,
    target_z: np.ndarray,
    target_r: np.ndarray,
) -> np.ndarray:
    """Interpolate channel-first cell-centre fields on physical coordinates.

    Q2D meshes store cell centres whose axial extent depends on the mesh
    size, so resampling must use the stored coordinates rather than array
    indices. Target points just outside the source centres hold the nearest
    value instead of extrapolating.
    """
    z, r = np.meshgrid(target_z, target_r, indexing="ij")
    points = np.column_stack((
        np.clip(z.ravel(), source_z[0], source_z[-1]),
        np.clip(r.ravel(), source_r[0], source_r[-1]),
    ))
    interpolator = RegularGridInterpolator(
        (source_z, source_r), np.moveaxis(fields, 0, -1), method="linear"
    )
    return np.moveaxis(
        interpolator(points).reshape(len(target_z), len(target_r), -1), -1, 0
    )


def discover_mesh_files(data_dir: Path) -> list[MeshFile]:
    """Return all available ``q2d_cmr_{n_z}_{n_r}_test`` files."""
    files: list[MeshFile] = []
    for path in data_dir.glob("q2d_cmr_*_*_test.h5"):
        match = MESH_FILE_PATTERN.fullmatch(path.name)
        if match is not None:
            files.append(MeshFile(path, int(match.group(1)), int(match.group(2))))
    files.sort(key=lambda item: (item.n_z, item.n_r))
    discovered = {(item.n_z, item.n_r) for item in files}
    axial_points = {item.n_z for item in files}
    radial_points = {item.n_r for item in files}
    expected = {
        (n_z, n_r)
        for n_z in axial_points
        for n_r in radial_points
    }
    missing = sorted(expected - discovered)
    if missing:
        print(
            "Mesh benchmark files not present (continuing): "
            + ", ".join(f"{n_z}x{n_r}" for n_z, n_r in missing)
        )
    if not files:
        raise FileNotFoundError("No Q2D resolution-sweep test files were found.")
    return files


def case_controls(metadata: Mapping[str, Any]) -> tuple[float, float]:
    """Return the inlet temperature and flow rate that identify a case."""
    params = metadata.get("params", {})
    try:
        return float(params["T0"]), float(params["sccm"])
    except KeyError as exc:
        raise KeyError("Case metadata is missing T0 or sccm.") from exc


def _controls_match(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
) -> bool:
    first_t0, first_sccm = case_controls(first)
    second_t0, second_sccm = case_controls(second)
    return math.isclose(first_t0, second_t0, rel_tol=1.0e-10, abs_tol=1.0e-8) and (
        math.isclose(
            first_sccm,
            second_sccm,
            rel_tol=1.0e-10,
            abs_tol=1.0e-8,
        )
    )


def find_superresolution_pair(
    mesh_files: Sequence[MeshFile],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    training_shape: tuple[int, int],
) -> tuple[MeshFile, int, MeshFile, int]:
    """Find a training-mesh case with truth on the finest matching mesh."""
    try:
        base_file = next(
            item
            for item in mesh_files
            if item.shape == training_shape
        )
    except StopIteration as exc:
        raise FileNotFoundError(
            "No mesh-sweep test file matches the discovered training mesh "
            f"{training_shape[0]}x{training_shape[1]}."
        ) from exc
    base_raw, base_data = make_adapter(base_file.path, normalizer, geometry, base_file.shape)
    candidates = sorted(
        (
            item
            for item in mesh_files
            if item.mesh_points > math.prod(training_shape)
            and item.n_z >= training_shape[0]
            and item.n_r >= training_shape[1]
        ),
        key=lambda item: (item.mesh_points, item.n_r, item.n_z),
        reverse=True,
    )
    try:
        base_metadata = [
            base_data.physical_item(index)["metadata"]
            for index in range(len(base_data))
        ]
        for candidate in candidates:
            fine_raw, fine_data = make_adapter(
                candidate.path,
                normalizer,
                geometry,
                candidate.shape,
            )
            try:
                for base_index, base_case in enumerate(base_metadata):
                    for fine_index in range(len(fine_data)):
                        fine_case = fine_data.physical_item(fine_index)["metadata"]
                        if _controls_match(base_case, fine_case):
                            return base_file, base_index, candidate, fine_index
            finally:
                fine_raw.close()
    finally:
        base_raw.close()
    raise RuntimeError("No matched coarse/fine case exists for superresolution.")

PROBLEM_ID = "q2d_cmr"

def experiment_spec(args, model_id):
    return operator_spec(args, PATHS, PROBLEM_ID, model_id, FILE_STEM,
                         OUTPUT_LABELS, {channel.label: channel.unit for channel in OUTPUT_CHANNELS},
                         COORDINATE_NAMES)

def parse_args(*, model_name: str = "FNO", benchmark: bool = True):
    parser = argparse.ArgumentParser(description=f"Tune and train a Q2D {model_name} with canonical artifacts.")
    if benchmark:
        parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=True,
                            help="Evaluate available mesh-sweep files and save cost and superresolution artifacts.")
    return parse_operator_args(PATHS, epochs=FINAL_EPOCHS, tune_epochs=TUNE_EPOCHS, samples=TUNE_SAMPLES, parser=parser)

ModelFactory = Callable[[Mapping[str, Any], torch.device], torch.nn.Module]


def make_trainer(
    model_factory: ModelFactory,
    config: Mapping[str, Any],
    context: Any,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    shape: tuple[int, int],
) -> OperatorTrainer:
    """Use the same loss, physical metrics, and checkpoint contract for all models."""

    def physical(value: torch.Tensor) -> torch.Tensor:
        return torch.stack(
            [
                normalizer.denormalize(value[:, index], channel.label)
                for index, channel in enumerate(OUTPUT_CHANNELS)
            ],
            dim=1,
        )

    return OperatorTrainer(
        lambda cfg: model_factory(cfg, context.device),
        config,
        loss_terms=[LossTerm("data_loss", lambda prediction, target, batch: relative_l2_loss(
            prediction, target, eps=RELATIVE_L2_EPS,
        ))],
        selection_metric="data_loss",
        metric_adapter=lambda prediction, target, batch: (
            physical(prediction),
            physical(target),
        ),
        checkpoint_metadata=checkpoint_metadata(normalizer, geometry, shape),
    )


def run_model(
    args: Any, model_id: str, model_factory: ModelFactory, search_space: Mapping[str, Any],
    *, after_run: Callable[..., None] | None = None,
) -> None:
    """Run Q2D tuning, final training, and test evaluation.

    ``after_run`` receives the trainer and result, plus the prepared data
    directory, normalizer, geometry, and training shape as keyword arguments.
    """
    normalizer, geometry, shape = fit_normalizer(args.data_dir / f"{FILE_STEM}_train.h5")

    def evaluate_run(trainer: OperatorTrainer, data: FNOAdapter, context: Any):
        return evaluate_fields(
            trainer, data, context, labels=OUTPUT_LABELS, coordinate_names=COORDINATE_NAMES,
            cases=args.plot_cases, normalized_relative_l2=True,
        )

    run_operator(
        args,
        PATHS,
        experiment_spec(args, model_id),
        lambda split: make_adapter(
            args.data_dir / f"{FILE_STEM}_{split}.h5", normalizer, geometry, shape
        ),
        lambda config, context: make_trainer(
            model_factory, config, context, normalizer, geometry, shape
        ),
        search_space,
        evaluate_run,
        selection_metric="data_loss",
        after_run=partial(
            after_run, data_dir=args.data_dir, normalizer=normalizer,
            geometry=geometry, shape=shape,
        ) if after_run is not None else None,
    )


class Q2DDeepONetAdapter(Dataset):
    """View Q2D fields as scalar branches and a shared Cartesian trunk."""

    format = "cartesian_product"

    def __init__(
        self,
        dataset: FNOAdapter,
        geometry: tuple[float, float],
        coordinates: tuple[torch.Tensor, torch.Tensor],
        max_cases: int | None = None,
    ) -> None:
        if any(channel.source not in {"parameter", "constant"}
               for channel in dataset.input_channels):
            raise ValueError("Q2D DeepONet branch channels must be scalar controls.")
        self.dataset = dataset
        self.coordinates = tuple(value.detach().cpu().clone() for value in coordinates)
        self.shape = tuple(len(value) for value in self.coordinates)
        self.labels = tuple(channel.label for channel in dataset.output_channels)
        self.count = len(dataset) if max_cases is None else min(max_cases, len(dataset))
        if self.count < 1:
            raise ValueError("Q2D DeepONet requires at least one selected case.")
        self.trunk = torch.stack(
            torch.meshgrid(*self.coordinates, indexing="ij"), dim=-1,
        ).reshape(-1, 2)
        for index in range(self.count):
            physical = dataset.physical_item(index)
            if not _same_geometry(_case_geometry(physical["metadata"]), geometry):
                raise ValueError(f"Q2D case {index} differs from training geometry.")
            for axis, expected in zip(COORDINATE_NAMES, self.coordinates):
                actual = physical[axis]
                if actual.shape != expected.shape or not torch.allclose(
                    actual, expected, rtol=1e-6, atol=1e-8,
                ):
                    raise ValueError(
                        f"Q2D case {index} differs from the training coordinate grid ({axis})."
                    )

    def __len__(self) -> int:
        return self.count

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < self.count:
            raise IndexError(index)
        sample = self.dataset[index]
        return {
            "branch": sample["x"][:, 0, 0],
            "trunk": self.trunk,
            "coordinate": self.trunk,
            "target": sample["y"].movedim(0, -1).reshape(-1, len(self.labels)),
            "labels": self.labels,
        }

    def field_output(self, values: torch.Tensor) -> torch.Tensor:
        """Restore the channel-first axial/radial field layout."""
        return values.movedim(-1, -2).reshape(
            *values.shape[:-2], len(self.labels), *self.shape,
        )


def _deeponet_split(
    path: Path,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    coordinates: tuple[torch.Tensor, torch.Tensor],
    max_cases: int | None,
) -> tuple[ChemOperatorDataset, Q2DDeepONetAdapter]:
    """Open one split, validating only the selected cases."""
    shape = tuple(len(value) for value in coordinates)
    raw = raw_dataset(path)
    try:
        return raw, Q2DDeepONetAdapter(
            field_adapter(raw, normalizer, shape), geometry, coordinates, max_cases,
        )
    except BaseException:
        raw.close()
        raise


@contextmanager
def deeponet_data(
    data_dir: Path,
    splits: Sequence[str],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    coordinates: tuple[torch.Tensor, torch.Tensor],
    max_cases: int | None = None,
) -> Iterator[tuple[Q2DDeepONetAdapter, ...]]:
    """Open and validate selected cases, closing every HDF5 handle on exit."""
    with ExitStack() as stack:
        datasets = []
        for split in splits:
            raw, data = _deeponet_split(
                Path(data_dir) / f"{FILE_STEM}_{split}.h5",
                normalizer, geometry, coordinates, max_cases,
            )
            stack.callback(raw.close)
            datasets.append(data)
        yield tuple(datasets)


def prepare_deeponet_data(
    data_dir: Path,
) -> tuple[ZScoreNormalizer, tuple[float, float], tuple[torch.Tensor, torch.Tensor]]:
    """Fit training-only statistics and discover the physical training grid."""
    path = Path(data_dir) / f"{FILE_STEM}_train.h5"
    normalizer, geometry, shape = fit_normalizer(path)
    raw, dataset = make_adapter(path, normalizer, geometry, shape)
    try:
        sample = dataset[0]
        coordinates = (sample["z"].clone(), sample["r"].clone())
    finally:
        raw.close()
    return normalizer, geometry, coordinates


def deeponet_search_space() -> dict[str, Any]:
    """Shared branch architecture and optimizer search for both DeepONets."""
    from ray import tune

    return {
        "loss": "relative_l2",
        "width": tune.choice([64, 128, 256]),
        "branch_hidden_layers": tune.choice([2, 3, 4]),
        "activation": tune.choice(["gelu", "tanh"]),
        "learning_rate": tune.loguniform(1e-4, 3e-3),
        "weight_decay": tune.loguniform(1e-8, 1e-4),
        "batch_size": tune.choice([4, 8, 16]),
        "epochs": TUNE_EPOCHS,
        "seed": SEED,
    }


def evaluate_q2d_deeponet(
    trainer: DeepONetTrainer,
    data: Q2DDeepONetAdapter | Subset,
    context: RunContext,
    *,
    cases: int,
) -> EvaluationOutcome:
    """Score physical fields with the shared evaluator and Q2D plot layout."""
    adapter = unwrap_dataset(data)
    indices = list(data.indices) if isinstance(data, Subset) else list(range(len(data)))
    return evaluate_fields(
        trainer, Subset(adapter.dataset, indices), context,
        labels=adapter.labels, coordinate_names=COORDINATE_NAMES, cases=cases,
        batch_size=EVALUATION_BATCH_SIZE, normalized_relative_l2=True,
        predict=lambda trained, _, run_context: adapter.field_output(
            trained.predict(data, run_context)
        ),
    )


def run_deeponet(
    args: argparse.Namespace,
    model_id: str,
    search_space: Mapping[str, Any],
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    coordinates: tuple[torch.Tensor, torch.Tensor],
    *,
    pod: PODTransform | None = None,
) -> RunContext:
    """Run the fixed-mesh DeepXDE tuning and final training on the shared workflow."""
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    shape = tuple(len(value) for value in coordinates)
    metadata = checkpoint_metadata(normalizer, geometry, shape, coordinates=coordinates)

    def dataset_factory(split: str):
        return _deeponet_split(
            args.data_dir / f"{FILE_STEM}_{split}.h5",
            normalizer, geometry, coordinates, args.max_cases,
        )

    def trainer_factory(config: Mapping[str, Any], context: RunContext) -> DeepONetTrainer:
        config = dict(config, loss="relative_l2")
        if pod is not None:
            config.update(latent_width=pod.n_components, trunk_hidden_layers=0)
        return DeepONetTrainer(
            deeponet_training_config(config), pod=pod,
            pin_memory=context.device.type == "cuda", checkpoint_metadata=metadata,
        )

    def tuner_factory(_trial_trainer, space, trial_data, context):
        gpus = int(context.device.type == "cuda")
        return deeponet_tuner(
            search_space=space, pod=pod, dataset_factory=trial_data, context=context,
            settings=DeepONetTuningSettings(
                max_epochs=args.tune_epochs, num_samples=args.samples,
                max_concurrent_trials=MAX_CONCURRENT_TRIALS,
                cpus_per_trial=CPUS_PER_TRIAL, gpus_per_trial=gpus,
                ray_runtime=operator_ray_runtime(
                    PATHS, cpus=CPUS_PER_TRIAL, gpus=gpus, env={"DDE_BACKEND": "pytorch"},
                ),
            ),
            pin_memory=bool(gpus),
        )

    return run_operator(
        args, PATHS, experiment_spec(args, model_id), dataset_factory, trainer_factory,
        search_space, partial(evaluate_q2d_deeponet, cases=args.plot_cases),
        seed=SEED, cpus=CPUS_PER_TRIAL, tuner_factory=tuner_factory,
    )
