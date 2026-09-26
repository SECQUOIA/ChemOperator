"""Model definition, scientific losses, and canonical experiment entry point."""

from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from collections.abc import Mapping, Sequence
import json
import math
import time
from typing import Any
from neuralop.losses import LpLoss
from neuralop.models import FNO
import numpy as np
import pandas as pd
from ray import tune
import torch
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader
from chem_operator.models import FNOAdapter
from chem_operator.normalization import ZScoreNormalizer
from chem_operator.experiments import FNOTrainer, LossTerm
from chem_operator.experiments import run_operator, evaluate_fields
from scripts.q2d.common import (
    PATHS,
    FILE_STEM,
    SEED,
    METRIC,
    TUNE_SAMPLES,
    TUNE_EPOCHS,
    FINAL_EPOCHS,
    EVALUATION_BATCH_SIZE,
    CPUS_PER_TRIAL,
    GPUS_PER_TRIAL,
    MAX_CONCURRENT_TRIALS,
    LATENCY_WARMUPS,
    LATENCY_REPEATS,
    RELATIVE_L2_EPS,
    AMORTIZED_EVALUATION_COUNTS,
    REPRESENTATIVE_RESOLUTION_COUNT,
    MESH_FILE_PATTERN,
    INPUT_CHANNELS,
    OUTPUT_CHANNELS,
    MeshFile,
    raw_dataset,
    adapter_spatial_shape,
    fit_normalizer,
    normalizer_state,
    normalizer_from_state,
    make_adapter,
    discover_mesh_files,
    find_superresolution_pair,
    PROBLEM_ID,
    experiment_spec,
    parse_args,
)

from scripts.q2d.common import _case_controls
def model_from_config(
    config: Mapping[str, Any],
    device: torch.device,
) -> FNO:
    """Construct the configured two-dimensional NeuralOperator FNO."""
    return FNO(
        n_modes=(int(config["modes_z"]), int(config["modes_r"])),
        in_channels=len(INPUT_CHANNELS),
        out_channels=len(OUTPUT_CHANNELS),
        hidden_channels=int(config["hidden_channels"]),
        n_layers=int(config["n_layers"]),
        positional_embedding="grid",
        domain_padding=float(config.get("domain_padding", 0.0)),
    ).to(device)


def _relative_l2_per_sample(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=1)
    reference = target.flatten(start_dim=1)
    return torch.linalg.vector_norm(difference, dim=1) / torch.linalg.vector_norm(
        reference, dim=1
    ).clamp_min(RELATIVE_L2_EPS)


def _relative_l2_per_field(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    difference = (prediction - target).flatten(start_dim=2)
    reference = target.flatten(start_dim=2)
    return torch.linalg.vector_norm(difference, dim=2) / torch.linalg.vector_norm(
        reference, dim=2
    ).clamp_min(RELATIVE_L2_EPS)


def evaluate(
    model: FNO,
    dataset: FNOAdapter,
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate mean per-case normalized relative L2."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    global_values: list[torch.Tensor] = []
    field_values: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            target = batch["y"].to(device)
            prediction = model(batch["x"].to(device))
            global_values.append(
                _relative_l2_per_sample(prediction, target).detach().cpu()
            )
            field_values.append(
                _relative_l2_per_field(prediction, target).detach().cpu()
            )
    global_tensor = torch.cat(global_values)
    fields_tensor = torch.cat(field_values)
    metrics = {"normalized_relative_l2": float(global_tensor.mean())}
    for index, spec in enumerate(OUTPUT_CHANNELS):
        metrics[f"normalized_relative_l2_{spec.label}"] = float(
            fields_tensor[:, index].mean()
        )
    return metrics


def resize_model_input(
    model_input: torch.Tensor,
    shape: tuple[int, int],
) -> torch.Tensor:
    """Resize channel-first inputs to a requested FNO evaluation mesh.

    Broadcast scalar channels remain exactly constant. This interpolation also
    makes future continuous physical input fields configuration-compatible
    with zero-shot superresolution. Evaluating directly on the requested mesh
    also keeps NeuralOperator domain padding and unpadding resolution-consistent.
    """
    if tuple(model_input.shape[-2:]) == tuple(shape):
        return model_input
    return torch_functional.interpolate(
        model_input.unsqueeze(0),
        size=shape,
        mode="bilinear",
        align_corners=True,
    )[0]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _complete_fno_evaluation(
    model: FNO,
    physical_input: torch.Tensor,
    dataset: FNOAdapter,
    training_shape: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Run preprocessing, resampling, inference, and postprocessing once."""
    normalized_input = torch.stack(
        [
            dataset.normalizer.normalize(
                physical_input[index],
                channel.label,
            )
            for index, channel in enumerate(dataset.input_channels)
        ]
    )
    training_input = resize_model_input(normalized_input, training_shape)
    evaluation_input = resize_model_input(training_input, output_shape)
    normalized_output = model(evaluation_input.unsqueeze(0).to(device)).cpu()[0]
    return dataset.denormalize_output(normalized_output)


def inference_latency(
    model: FNO,
    physical_input: torch.Tensor,
    dataset: FNOAdapter,
    training_shape: tuple[int, int],
    output_shape: tuple[int, int],
    *,
    device: torch.device,
) -> float:
    """Return median end-to-end single-case FNO evaluation time."""
    model.eval()
    with torch.no_grad():
        for _ in range(LATENCY_WARMUPS):
            _complete_fno_evaluation(
                model,
                physical_input,
                dataset,
                training_shape,
                output_shape,
                device=device,
            )
        _synchronize(device)
        durations = []
        for _ in range(LATENCY_REPEATS):
            _synchronize(device)
            tic = time.perf_counter()
            _complete_fno_evaluation(
                model,
                physical_input,
                dataset,
                training_shape,
                output_shape,
                device=device,
            )
            _synchronize(device)
            durations.append(time.perf_counter() - tic)
    return float(np.median(durations))


def benchmark_meshes(  # pylint: disable=too-many-locals
    model: FNO,
    normalizer: ZScoreNormalizer,
    geometry: tuple[float, float],
    mesh_files: Sequence[MeshFile],
    training_shape: tuple[int, int],
    *,
    device: torch.device,
) -> pd.DataFrame:
    """Evaluate all runs in every discovered resolution-sweep file."""
    rows: list[dict[str, Any]] = []
    model.eval()
    for mesh_file in mesh_files:
        raw, dataset = make_adapter(mesh_file.path, normalizer, geometry)
        try:
            for case_index in range(len(dataset)):
                sample = dataset[case_index]
                target = sample["y"].unsqueeze(0).to(device)
                target_shape = tuple(int(value) for value in target.shape[-2:])
                if target_shape != (mesh_file.n_z, mesh_file.n_r):
                    raise ValueError(
                        f"{mesh_file.path.name} declares "
                        f"{mesh_file.n_z}x{mesh_file.n_r} but stores {target_shape}."
                    )
                training_input = resize_model_input(sample["x"], training_shape)
                prediction_input = resize_model_input(
                    training_input,
                    target_shape,
                )
                with torch.no_grad():
                    prediction = model(prediction_input.unsqueeze(0).to(device))
                global_l2 = float(
                    _relative_l2_per_sample(prediction, target).cpu()[0]
                )
                field_l2 = _relative_l2_per_field(prediction, target).cpu()[0]
                physical = dataset.physical_item(case_index)
                metadata = physical["metadata"]
                t0, sccm = _case_controls(metadata)
                row: dict[str, Any] = {
                    "dataset_file": mesh_file.path.name,
                    "case_index": case_index,
                    "n_z": mesh_file.n_z,
                    "n_r": mesh_file.n_r,
                    "mesh_points": mesh_file.mesh_points,
                    "T0_K": t0,
                    "SCCM": sccm,
                    "normalized_relative_l2": global_l2,
                    "solver_wall_time_s": float(metadata["wall_time"]),
                    "fno_wall_time_s": inference_latency(
                        model,
                        physical["x"],
                        dataset,
                        training_shape,
                        target_shape,
                        device=device,
                    ),
                }
                for field_index, spec in enumerate(OUTPUT_CHANNELS):
                    row[f"normalized_relative_l2_{spec.label}"] = float(
                        field_l2[field_index]
                    )
                rows.append(row)
        finally:
            raw.close()
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError("The mesh benchmark produced no rows.")
    return frame.sort_values(
        ["mesh_points", "n_z", "n_r", "case_index"],
        ignore_index=True,
    )


def dataset_generation_costs(data_dir) -> dict[str, float | int]:
    """Return numerical-simulation costs for training and validation data."""
    result: dict[str, float | int] = {}
    total_seconds = 0.0
    total_cases = 0
    for split in ("train", "valid"):
        dataset = raw_dataset(data_dir / f"{FILE_STEM}_{split}.h5")
        try:
            wall_times = [
                float(dataset[index]["metadata"]["wall_time"])
                for index in range(len(dataset))
            ]
        finally:
            dataset.close()
        if not wall_times or any(
            not math.isfinite(value) or value < 0.0 for value in wall_times
        ):
            raise ValueError(
                f"{split} data contain missing or invalid solver wall times."
            )
        split_seconds = sum(wall_times)
        result[f"{split}_data_generation_seconds"] = split_seconds
        result[f"{split}_data_cases"] = len(wall_times)
        total_seconds += split_seconds
        total_cases += len(wall_times)
    result["data_generation_seconds"] = total_seconds
    result["data_generation_cases"] = total_cases
    return result


def _resolution_cost_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Aggregate benchmark costs for each exact axial-radial resolution."""
    summary = (
        frame.groupby(["n_z", "n_r"], as_index=False, sort=True)
        .agg(
            mesh_points=("mesh_points", "first"),
            benchmark_cases=("case_index", "size"),
            numerical_solver_seconds=("solver_wall_time_s", "mean"),
            fno_evaluation_seconds=("fno_wall_time_s", "mean"),
            normalized_relative_l2=("normalized_relative_l2", "mean"),
        )
        .sort_values(["mesh_points", "n_z", "n_r"], ignore_index=True)
    )
    records: list[dict[str, Any]] = []
    for row in summary.itertuples(index=False):
        records.append(
            {
                "n_z": int(row.n_z),
                "n_r": int(row.n_r),
                "mesh_points": int(row.mesh_points),
                "benchmark_cases": int(row.benchmark_cases),
                "numerical_solver_seconds": float(
                    row.numerical_solver_seconds
                ),
                "fno_evaluation_seconds": float(row.fno_evaluation_seconds),
                "normalized_relative_l2": float(row.normalized_relative_l2),
            }
        )
    return records


def _representative_resolutions(
    records: Sequence[Mapping[str, Any]],
    training_shape: tuple[int, int],
) -> list[Mapping[str, Any]]:
    """Select dataset-derived resolutions spanning the benchmark range."""
    if len(records) <= REPRESENTATIVE_RESOLUTION_COUNT:
        selected = list(records)
    else:
        indices = np.linspace(
            0,
            len(records) - 1,
            REPRESENTATIVE_RESOLUTION_COUNT,
        ).round().astype(int)
        selected = [records[index] for index in dict.fromkeys(indices)]
    training_record = next(
        (
            record
            for record in records
            if (record["n_z"], record["n_r"]) == training_shape
        ),
        None,
    )
    if training_record is not None and training_record not in selected:
        if len(selected) < 3:
            selected.append(training_record)
        else:
            replacement = min(
                range(1, len(selected) - 1),
                key=lambda index: abs(
                    selected[index]["mesh_points"]
                    - training_record["mesh_points"]
                ),
            )
            selected[replacement] = training_record
    return sorted(
        selected,
        key=lambda record: (
            record["mesh_points"],
            record["n_z"],
            record["n_r"],
        ),
    )


def build_cost_model(
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
    *,
    ray_tuning_seconds: float,
    final_training_seconds: float,
    data_dir: Path,
) -> dict[str, Any]:
    """Build the persisted offline, online, and break-even cost model."""
    data_costs = dataset_generation_costs(data_dir)
    offline_seconds = (
        float(data_costs["data_generation_seconds"])
        + ray_tuning_seconds
        + final_training_seconds
    )
    records = _resolution_cost_records(frame)
    for record in records:
        difference = (
            record["numerical_solver_seconds"]
            - record["fno_evaluation_seconds"]
        )
        record["break_even_evaluations"] = (
            offline_seconds / difference if difference > 0.0 else None
        )
        record["asymptotic_speedup"] = (
            record["numerical_solver_seconds"]
            / record["fno_evaluation_seconds"]
        )
    representatives = _representative_resolutions(records, training_shape)
    return {
        "definitions": {
            "fno_total": (
                "T_data + T_tune + T_final_train + "
                "N * t_FNO(resolution)"
            ),
            "numerical_total": "N * t_num(resolution)",
            "fno_amortized": (
                "T_offline / N + t_FNO(resolution)"
            ),
            "speedup": "T_num(N, resolution) / T_FNO(N, resolution)",
            "data_generation_source": (
                "sum of per-case solver wall_time metadata for the complete "
                "training and validation datasets"
            ),
            "ray_tuning_scope": (
                "elapsed wall time around all Ray Tune trials, including "
                "pruned or unsuccessful trials"
            ),
            "resolution_aggregation": (
                "mean per-case wall time for each exact (n_z, n_r) mesh"
            ),
            "fno_evaluation_includes": [
                "input normalization",
                "interpolation or resampling",
                "device transfer",
                "model inference",
                "output denormalization",
            ],
        },
        **data_costs,
        "ray_tuning_seconds": float(ray_tuning_seconds),
        "final_training_seconds": float(final_training_seconds),
        "offline_seconds": float(offline_seconds),
        "amortized_evaluation_counts": list(AMORTIZED_EVALUATION_COUNTS),
        "representative_resolutions": [
            {
                "n_z": int(record["n_z"]),
                "n_r": int(record["n_r"]),
                "mesh_points": int(record["mesh_points"]),
            }
            for record in representatives
        ],
        "resolutions": records,
    }
def search_space():
    return {'modes_z': tune.choice([4, 5, 6]), 'modes_r': tune.choice([2, 3, 4]), 'hidden_channels': tune.choice([16, 20, 24, 28, 32]), 'n_layers': tune.choice([6, 7, 8, 9]), 'learning_rate': tune.loguniform(0.0001, 0.004), 'weight_decay': tune.loguniform(1e-08, 0.0001), 'batch_size': 2, 'domain_padding': tune.choice([0.0, 0.05, 0.1, 0.15])}

MODEL_ID = "fno"

def make_trainer(config, context, normalizer, geometry, shape):
    loss = LpLoss(d=2,p=2,reduction="mean")
    def physical(value):
        return torch.stack([normalizer.denormalize(value[:,i],channel.label)
                            for i,channel in enumerate(OUTPUT_CHANNELS)],dim=1)
    return FNOTrainer(lambda cfg: model_from_config(cfg,context.device),config,
        loss_terms=[LossTerm("data_loss", lambda p,t,b: loss(p,t))], selection_metric="data_loss",
        metric_adapter=lambda p,t,b: (physical(p),physical(t)),
        checkpoint_metadata={"normalizer": normalizer.state_dict(), "geometry": list(geometry),
                             "training_shape": list(shape), "input_channels": [c.label for c in INPUT_CHANNELS],
                             "output_channels": [c.label for c in OUTPUT_CHANNELS]})

def save_benchmarks(trainer, result, *, data_dir, normalizer, geometry, shape):
    from chem_operator.experiments import ArtifactStore, fingerprint_path
    output = result.context.paths.run_dir
    mesh_files = discover_mesh_files(data_dir)
    frame = benchmark_meshes(trainer.model,normalizer,geometry,mesh_files,shape,device=result.context.device)
    frame.to_csv(output / "mesh_benchmark.csv",index=False)
    store = ArtifactStore(result.context)
    timings = store.read_manifest()["timings"]
    costs = build_cost_model(frame,shape,ray_tuning_seconds=timings["tuning_seconds"],
                            final_training_seconds=timings["final_training_seconds"],data_dir=data_dir)
    (output / "cost_model.json").write_text(json.dumps(costs,indent=2,allow_nan=False)+"\n")
    provenance = {item.path.name: fingerprint_path(item.path) for item in mesh_files}
    coarse,ci,fine,fi = find_superresolution_pair(mesh_files,normalizer,geometry,shape)
    arrays = {}
    for label,item,index in (("coarse",coarse,ci),("fine",fine,fi)):
        raw,data = make_adapter(item.path,normalizer,geometry)
        try:
            sample = data[index]
            with torch.no_grad():
                x = resize_model_input(resize_model_input(sample["x"],shape),(item.n_z,item.n_r))
                pred = trainer.model(x[None].to(result.context.device)).cpu()[0]
            arrays.update({f"{label}_{name}": value.numpy() for name,value in
                (("reference",data.denormalize_output(sample["y"])),("prediction",data.denormalize_output(pred)),
                 ("z",sample["z"]),("r",sample["r"]))})
        finally: raw.close()
    arrays["labels"] = np.asarray([c.label for c in OUTPUT_CHANNELS])
    np.savez_compressed(output / "superresolution.npz",**arrays)
    store.update_manifest(benchmark_metadata={"dataset_fingerprints": provenance, "training_shape": list(shape),
                          "coarse_case": [coarse.path.name,ci], "fine_case": [fine.path.name,fi]})

def main():
    args = parse_args()
    normalizer,geometry,shape = fit_normalizer(args.data_dir / f"{FILE_STEM}_train.h5")
    def evaluate_run(trainer,data,context):
        outcome = evaluate_fields(trainer,data,context,labels=tuple(c.label for c in OUTPUT_CHANNELS),
            coordinate_names=("z","r"),cases=args.plot_cases,
            extra_metrics=lambda t,d,c: evaluate(t.model,d,batch_size=EVALUATION_BATCH_SIZE,device=c.device))
        return outcome
    run_operator(args,PATHS,experiment_spec(args,MODEL_ID),
        lambda split: make_adapter(args.data_dir / f"{FILE_STEM}_{split}.h5",normalizer,geometry),
        lambda config,context: make_trainer(config,context,normalizer,geometry,shape), search_space(),evaluate_run,
        selection_metric="data_loss",after_run=(lambda trainer,result: save_benchmarks(trainer,result,
            data_dir=args.data_dir,normalizer=normalizer,geometry=geometry,shape=shape)) if args.benchmark else None)

if __name__ == "__main__":
    main()
