"""Plot Q2D run artifacts and optional tolerance-sweep comparisons."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import math
from typing import Any, Mapping, Sequence
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter
from matplotlib.colors import LogNorm
from scipy.interpolate import RegularGridInterpolator
from chem_operator.plotting import plot_operator_runs, plot_operator_fields
from chem_operator.experiments import load_run
SLIDE_DPI = 220


def _interpolate_fields(
    fields: np.ndarray,
    source_z: np.ndarray,
    source_r: np.ndarray,
    target_z: np.ndarray,
    target_r: np.ndarray,
) -> np.ndarray:
    """Interpolate channel-first cell-centre fields on physical coordinates."""
    z, r = np.meshgrid(target_z, target_r, indexing="ij")
    # The meshes store cell centres, so edge centres can fall just outside the
    # other mesh. Hold the nearest value there instead of extrapolating it.
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


def _relative_field_errors(prediction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Return physical relative L2 errors, one per output channel."""
    difference = (prediction - reference).reshape(len(reference), -1)
    truth = reference.reshape(len(reference), -1)
    return np.linalg.norm(difference, axis=1) / np.maximum(
        np.linalg.norm(truth, axis=1), 1.0e-8
    )


def plot_tolerance_superresolution(
    path: Path,
    run_path: Path,
    data_dir: Path,
    *,
    target_shape: tuple[int, int] = (14, 20),
) -> None:
    """Compare coarse interpolation and FNO against the 1e-6 fine solution."""
    import torch

    from chem_operator.normalization import normalizer_from_state_dict
    from scripts.q2d.common import OUTPUT_CHANNELS, make_adapter
    from scripts.q2d.fno import (
        _complete_fno_evaluation,
        inference_latency,
        model_from_config,
    )

    checkpoint = torch.load(run_path / "checkpoint.pt", map_location="cpu", weights_only=True)
    metadata = checkpoint["metadata"]
    training_shape = tuple(metadata["training_shape"])
    if training_shape != (14, 6):
        raise ValueError(f"Tolerance comparison requires a 14x6 FNO, got {training_shape}.")
    normalizer = normalizer_from_state_dict(metadata["normalizer"])
    geometry = tuple(metadata["geometry"])
    fine_path = data_dir / "q2d_cmr_tolerance_20_10_test.h5"
    coarse_path = data_dir / "q2d_cmr_tolerance_14_6_test.h5"
    fine_raw, fine = make_adapter(fine_path, normalizer, geometry)
    try:
        coarse_raw, coarse = make_adapter(coarse_path, normalizer, geometry)
    except Exception:
        fine_raw.close()
        raise
    try:
        fine_cases = {
            float(fine.physical_item(index)["metadata"]["params"]["rtol_ss"]): index
            for index in range(len(fine))
        }
        if 1.0e-6 not in fine_cases:
            raise ValueError(f"{fine_path.name} has no rtol=1e-6 reference case.")
        reference_item = fine.physical_item(fine_cases[1.0e-6])
        fine_sample = fine[fine_cases[1.0e-6]]
        target_z = (np.arange(target_shape[0]) + 0.5) * geometry[0] / target_shape[0]
        target_r = (np.arange(target_shape[1]) + 0.5) * geometry[1] / target_shape[1]
        reference = _interpolate_fields(
            reference_item["y"].numpy(),
            fine_sample["z"].numpy(), fine_sample["r"].numpy(),
            target_z, target_r,
        )
        rows = []
        for index in range(len(coarse)):
            item = coarse.physical_item(index)
            sample = coarse[index]
            case_metadata = item["metadata"]
            if any(
                not math.isclose(float(case_metadata["params"][key]),
                                 float(reference_item["metadata"]["params"][key]))
                for key in ("T0", "P0", "sccm")
            ):
                raise ValueError("Tolerance sweep cases do not share physical controls.")
            interpolated = _interpolate_fields(
                item["y"].numpy(), sample["z"].numpy(), sample["r"].numpy(),
                target_z, target_r,
            )
            rows.append((
                float(case_metadata["params"]["rtol_ss"]),
                float(case_metadata["wall_time"]),
                _relative_field_errors(interpolated, reference),
            ))
        model = model_from_config(checkpoint["config"], torch.device("cpu"))
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        physical_input = coarse.physical_item(0)["x"]
        with torch.no_grad():
            prediction = _complete_fno_evaluation(
                model, physical_input, coarse,
                training_shape, target_shape, device=torch.device("cpu"),
            ).numpy()
        fno_seconds = inference_latency(
            model, physical_input, coarse,
            training_shape, target_shape, device=torch.device("cpu"),
        )
    finally:
        coarse_raw.close()
        fine_raw.close()

    rows.sort(key=lambda row: row[0])
    rtols = np.asarray([row[0] for row in rows])
    seconds = np.asarray([row[1] for row in rows])
    errors = np.stack([row[2] for row in rows])
    fno_errors = _relative_field_errors(prediction, reference)
    figure, axes = plt.subplots(1, len(OUTPUT_CHANNELS), figsize=(17, 5),
                               constrained_layout=True)
    colors = plt.get_cmap("viridis")
    norm = LogNorm(vmin=rtols.min(), vmax=rtols.max())
    for index, (axis, channel) in enumerate(zip(axes, OUTPUT_CHANNELS)):
        axis.scatter(seconds, errors[:, index], c=rtols, cmap=colors, norm=norm,
                     s=85, edgecolor="black", linewidth=0.5, label="14×6 + interpolation")
        axis.scatter([fno_seconds], [fno_errors[index]], marker="*", s=210,
                     color="tab:red", edgecolor="black", linewidth=0.5,
                     label="FNO superresolution")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(channel.display_name)
        axis.set_xlabel("Wall time per case [s]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Relative L2 error on target grid")
    axes[-1].legend(loc="best")
    figure.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=colors), ax=axes,
                    label="Solver rtol")
    figure.suptitle(
        f"Tolerance and FNO superresolution on {target_shape[0]}×{target_shape[1]} grid "
        "(20×10, rtol=1e-6 reference)"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=SLIDE_DPI)
    # import matplotlib
    # matplotlib.use('Qt5Agg')
    # plt.show()
    plt.close(figure)


def _grouped_summary(
    frame: pd.DataFrame,
    column: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grouped = frame.groupby("mesh_points", sort=True)[column]
    return (
        grouped.mean().index.to_numpy(dtype=float),
        grouped.mean().to_numpy(dtype=float),
        grouped.min().to_numpy(dtype=float),
        grouped.max().to_numpy(dtype=float),
    )


def plot_cumulative_break_even(
    path: Path,
    cost_model: Mapping[str, Any],
) -> None:
    """Plot cumulative numerical and FNO costs with break-even markers."""
    representative_keys = {
        (item["n_z"], item["n_r"])
        for item in cost_model["representative_resolutions"]
    }
    representatives = [
        record
        for record in cost_model["resolutions"]
        if (record["n_z"], record["n_r"]) in representative_keys
    ]
    finite_break_evens = [
        float(record["break_even_evaluations"])
        for record in representatives
        if record["break_even_evaluations"] is not None
    ]
    maximum_cases = max(
        cost_model["amortized_evaluation_counts"][-1],
        1.1 * max(finite_break_evens, default=0.0),
    )
    requested_cases = np.linspace(0.0, maximum_cases, 500)
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(16, 9),
        squeeze=False,
        constrained_layout=True,
    )
    offline_seconds = float(cost_model["offline_seconds"])
    for axis, record in zip(axes.flat, representatives):
        numerical = requested_cases * record["numerical_solver_seconds"]
        fno = (
            offline_seconds
            + requested_cases * record["fno_evaluation_seconds"]
        )
        axis.plot(requested_cases, numerical, label="Numerical solver")
        axis.plot(requested_cases, fno, linestyle="--", label="FNO")
        break_even = record["break_even_evaluations"]
        if break_even is not None:
            intersection_time = (
                break_even * record["numerical_solver_seconds"]
            )
            axis.scatter(
                [break_even],
                [intersection_time],
                color="black",
                marker="X",
                s=90,
                zorder=4,
                label=f"Break-even: N={break_even:.1f}",
            )
        axis.set_title(
            f"{record['n_z']}×{record['n_r']} "
            f"({record['mesh_points']} points)"
        )
        axis.set_xlabel("Number of new cases, N")
        axis.set_ylabel("Cumulative wall time [s]")
        axis.grid(alpha=0.28)
        axis.legend()
    for axis in axes.flat[len(representatives):]:
        axis.set_visible(False)
    figure.suptitle(
        "Cumulative cost and FNO break-even "
        f"(offline cost = {offline_seconds:.1f} s)"
    )
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def _pareto_indices(
    times: np.ndarray,
    mesh_points: np.ndarray,
) -> np.ndarray:
    """Return indices not dominated in lower-time/higher-resolution space."""
    keep = []
    for index, (wall_time, resolution) in enumerate(
        zip(times, mesh_points)
    ):
        dominated = np.any(
            (times <= wall_time)
            & (mesh_points >= resolution)
            & ((times < wall_time) | (mesh_points > resolution))
        )
        if not dominated:
            keep.append(index)
    return np.asarray(
        sorted(keep, key=lambda item: times[item]),
        dtype=int,
    )


def plot_amortized_pareto(
    path: Path,
    cost_model: Mapping[str, Any],
) -> None:
    """Plot amortized cost-resolution Pareto frontiers."""
    records = cost_model["resolutions"]
    mesh_points = np.asarray(
        [record["mesh_points"] for record in records],
        dtype=float,
    )
    numerical = np.asarray(
        [record["numerical_solver_seconds"] for record in records],
        dtype=float,
    )
    fno_online = np.asarray(
        [record["fno_evaluation_seconds"] for record in records],
        dtype=float,
    )
    offline_seconds = float(cost_model["offline_seconds"])
    figure, axes = plt.subplots(
        1,
        len(cost_model["amortized_evaluation_counts"]),
        figsize=(21, 6),
        sharey=True,
        constrained_layout=True,
    )
    numerical_frontier = _pareto_indices(numerical, mesh_points)
    for axis, evaluations in zip(axes, cost_model["amortized_evaluation_counts"]):
        fno_amortized = offline_seconds / evaluations + fno_online
        fno_frontier = _pareto_indices(fno_amortized, mesh_points)
        axis.scatter(numerical, mesh_points, color="tab:blue", alpha=0.25)
        axis.scatter(fno_amortized, mesh_points, color="tab:orange", alpha=0.25)
        axis.plot(
            numerical[numerical_frontier],
            mesh_points[numerical_frontier],
            color="tab:blue",
            marker="o",
            label="Numerical frontier",
        )
        axis.plot(
            fno_amortized[fno_frontier],
            mesh_points[fno_frontier],
            color="tab:orange",
            marker="o",
            label="FNO frontier",
        )
        axis.set_xscale("log")
        axis.xaxis.set_major_locator(LogLocator(base=10.0))
        axis.xaxis.set_major_formatter(LogFormatterMathtext(base=10.0))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.set_title(f"N = {evaluations}")
        axis.set_xlabel("Amortized wall time per case [s]")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Requested resolution [mesh points]")
    axes[-1].legend()
    figure.suptitle("Amortized cost-resolution Pareto frontiers")
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def plot_break_even_map(
    path: Path,
    cost_model: Mapping[str, Any],
) -> None:
    """Plot speedup across exact resolutions and evaluation counts."""
    records = cost_model["resolutions"]
    numerical = np.asarray(
        [record["numerical_solver_seconds"] for record in records],
        dtype=float,
    )
    fno_online = np.asarray(
        [record["fno_evaluation_seconds"] for record in records],
        dtype=float,
    )
    finite_break_evens = [
        float(record["break_even_evaluations"])
        for record in records
        if record["break_even_evaluations"] is not None
    ]
    maximum_cases = max(
        cost_model["amortized_evaluation_counts"][-1],
        2.0 * max(finite_break_evens, default=0.0),
    )
    evaluation_counts = np.geomspace(1.0, maximum_cases, 240)
    offline_seconds = float(cost_model["offline_seconds"])
    speedup = (
        evaluation_counts[:, None] * numerical[None, :]
        / (
            offline_seconds
            + evaluation_counts[:, None] * fno_online[None, :]
        )
    )
    resolution_index = np.arange(len(records))
    positive_minimum = max(float(speedup.min()), np.finfo(float).tiny)
    maximum = float(speedup.max())
    figure, axis = plt.subplots(figsize=(16, 9), constrained_layout=True)
    axis.set_yscale("log")
    image = axis.pcolormesh(
        resolution_index,
        evaluation_counts,
        speedup,
        shading="nearest",
        cmap="coolwarm",
        norm=LogNorm(vmin=positive_minimum, vmax=maximum),
    )
    if positive_minimum <= 1.0 <= maximum:
        contour = axis.contour(
            resolution_index,
            evaluation_counts,
            speedup,
            levels=[1.0],
            colors="black",
            linewidths=2,
        )
        label_candidates = [
            (float(index), float(record["break_even_evaluations"]))
            for index, record in enumerate(records)
            if record["break_even_evaluations"] is not None
        ]
        label_location = label_candidates[len(label_candidates) // 3]
        axis.clabel(
            contour,
            fmt={1.0: "break-even"},
            inline=True,
            inline_spacing=8,
            manual=[label_location],
        )
    tick_step = max(1, math.ceil(len(records) / 12))
    ticks = resolution_index[::tick_step]
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [
            f"{records[index]['n_z']}×{records[index]['n_r']}"
            for index in ticks
        ],
        rotation=45,
        ha="right",
    )
    axis.set_xlabel("Requested axial × radial resolution")
    axis.set_ylabel("Number of new evaluations, N")
    axis.set_title("FNO-to-numerical cumulative-cost speedup")
    figure.colorbar(image, ax=axis, label="Speedup")
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def _resolution_plot_summary(
    frame: pd.DataFrame,
    column: str,
) -> pd.DataFrame:
    """Aggregate one benchmark quantity by exact axial-radial resolution."""
    return (
        frame.groupby(["n_z", "n_r"], as_index=False, sort=True)
        .agg(
            mesh_points=("mesh_points", "first"),
            mean=(column, "mean"),
            minimum=(column, "min"),
            maximum=(column, "max"),
        )
        .sort_values(["mesh_points", "n_z", "n_r"], ignore_index=True)
    )


def _set_resolution_ticks(
    axis: plt.Axes,
    summary: pd.DataFrame,
    *,
    maximum_ticks: int = 14,
) -> None:
    """Label an ordered categorical axial-by-radial resolution axis."""
    step = max(1, math.ceil(len(summary) / maximum_ticks))
    ticks = np.arange(0, len(summary), step)
    if ticks[-1] != len(summary) - 1:
        ticks = np.append(ticks, len(summary) - 1)
    axis.set_xticks(ticks)
    axis.set_xticklabels(
        [
            f"{int(summary.iloc[index]['n_z'])}×"
            f"{int(summary.iloc[index]['n_r'])}"
            for index in ticks
        ],
        rotation=42,
        ha="right",
    )
    axis.set_xlim(-0.6, len(summary) - 0.4)
    axis.set_xlabel("Axial × radial resolution")


def _mark_training_resolution(
    axis: plt.Axes,
    summary: pd.DataFrame,
    training_shape: tuple[int, int],
) -> None:
    """Mark the categorical resolution used to train the FNO."""
    matches = summary.index[
        (summary["n_z"] == training_shape[0])
        & (summary["n_r"] == training_shape[1])
    ].tolist()
    if not matches:
        raise ValueError(
            "The training resolution is absent from the mesh benchmark."
        )
    axis.axvline(
        matches[0],
        color="black",
        linestyle="--",
        linewidth=2.0,
        label=(
            "FNO training resolution "
            f"({training_shape[0]}×{training_shape[1]})"
        ),
    )


def plot_mesh_l2(
    path: Path,
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
) -> None:
    """Plot normalized relative L2 against exact requested resolution."""
    summary = _resolution_plot_summary(
        frame,
        "normalized_relative_l2",
    )
    x = np.arange(len(summary))
    figure, axis = plt.subplots(figsize=(15, 7), constrained_layout=True)
    axis.plot(x, summary["mean"], marker="o", label="FNO mean")
    axis.fill_between(
        x,
        summary["minimum"],
        summary["maximum"],
        alpha=0.22,
        label="Case min–max",
    )
    axis.set_ylabel("Normalized relative L2")
    axis.set_yscale("log")
    axis.set_title("FNO error across requested mesh resolutions")
    _set_resolution_ticks(axis, summary)
    _mark_training_resolution(axis, summary, training_shape)
    axis.grid(alpha=0.28)
    axis.legend(ncols=3, loc="upper left")
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)


def plot_mesh_wall_time(
    path: Path,
    frame: pd.DataFrame,
    training_shape: tuple[int, int],
    cost_model: Mapping[str, Any],
) -> None:
    """Plot online times and grouped offline-cost bars on a shared axis."""
    solver = _resolution_plot_summary(frame, "solver_wall_time_s")
    fno = _resolution_plot_summary(frame, "fno_wall_time_s")
    if not solver[["n_z", "n_r"]].equals(fno[["n_z", "n_r"]]):
        raise ValueError("Solver and FNO resolution summaries do not align.")
    x = np.arange(len(solver))
    figure, (axis, offline_axis) = plt.subplots(
        1,
        2,
        figsize=(16, 12),
        gridspec_kw={"width_ratios": (3.2, 2.3)},
        sharey=True,
        constrained_layout=True,
    )
    for summary, label, color in (
        (solver, "Numerical solver per case", "tab:blue"),
        (fno, "FNO per case", "tab:orange"),
    ):
        axis.plot(x, summary["mean"], marker="o", color=color, label=label)
        axis.fill_between(
            x,
            summary["minimum"],
            summary["maximum"],
            color=color,
            alpha=0.18,
        )
    ray_tuning_seconds = float(cost_model["ray_tuning_seconds"])
    final_training_seconds = float(cost_model["final_training_seconds"])
    axis.set_ylabel("Wall time [s]")
    positive_minimum = float(fno["minimum"].min())
    axis.set_yscale("symlog", linthresh=max(positive_minimum / 2.0, 1.0e-6))
    axis.set_title("Online evaluation cost by resolution")
    _set_resolution_ticks(axis, solver)
    _mark_training_resolution(axis, solver, training_shape)
    axis.grid(alpha=0.28)
    axis.legend()
    # axis.legend(ncols=2, loc="upper left")

    components = (
        (
            float(cost_model["data_generation_seconds"]),
            r"Data Generation",
            "tab:blue",
        ),
        (
            ray_tuning_seconds,
            r"Tuning Model",
            "tab:purple",
        ),
        (
            final_training_seconds,
            r"Final training",
            "tab:green",
        ),
    )
    for seconds, label, _ in components:
        if not math.isfinite(seconds) or seconds < 0.0:
            raise ValueError(f"{label} wall time must be nonnegative and finite.")

    bar_positions = np.array((-0.34, 0.0, 0.34))
    bars = offline_axis.bar(
        bar_positions,
        [component[0] for component in components],
        color=[component[2] for component in components],
        width=0.28,
        edgecolor="white",
        linewidth=1.5,
    )
    for bar, (seconds, label, _) in zip(bars, components):
        offline_axis.annotate(
            f"{seconds:.1f} s",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=16,
        )
    total_offline_seconds = sum(component[0] for component in components)
    offline_axis.text(
        0.5,
        0.05,
        rf"Total offline time = {total_offline_seconds:.1f} s",
        transform=offline_axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=20,
        bbox={
            "boxstyle": "round,pad=0.5",
            "facecolor": "white",
            "edgecolor": "0.35",
            "alpha": 0.94,
        },
    )
    offline_axis.set_xlim(-0.65, 0.65)
    offline_axis.set_xticks(
        bar_positions,
        [component[1] for component in components],
        fontsize=14
    )
    offline_axis.set_title("One-time offline cost components")
    offline_axis.grid(axis="y", alpha=0.28)
    offline_axis.tick_params(axis="y", labelleft=False)
    figure.savefig(path, dpi=SLIDE_DPI)
    plt.close(figure)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Canonical run directory; repeat to compare compatible models.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "fno")
    parser.add_argument("--cases", type=int, default=2)
    parser.add_argument("--tolerance-comparison", action="store_true",
                        help="Plot coarse interpolation and FNO error versus wall time.")
    parser.add_argument("--data-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / "datasets" / "q2d_cmr")
    args = parser.parse_args()
    plot_operator_runs(args.run,args.output_dir,cases=args.cases)
    for run_path in args.run:
        run = load_run(run_path)
        output_dir = (args.output_dir if len(args.run) == 1 else
                      args.output_dir / run.manifest["model_id"] / run.path.name)
        if args.tolerance_comparison:
            plot_tolerance_superresolution(
                output_dir / "tolerance_superresolution_error_vs_time.png",
                run.path, args.data_dir,
            )
        benchmark = run.path / "mesh_benchmark.csv"
        if not benchmark.exists():
            continue
        frame = pd.read_csv(benchmark)
        cost = json.loads((run.path / "cost_model.json").read_text())
        shape = tuple(run.manifest["benchmark_metadata"]["training_shape"])
        plot_mesh_l2(output_dir / "mesh_l2_vs_points.png",frame,shape)
        plot_mesh_wall_time(output_dir / "mesh_wall_time_vs_points.png",frame,shape,cost)
        plot_cumulative_break_even(output_dir / "cumulative_break_even.png",cost)
        plot_amortized_pareto(output_dir / "amortized_pareto_frontiers.png",cost)
        plot_break_even_map(output_dir / "break_even_speedup_map.png",cost)
        with np.load(run.path / "superresolution.npz",allow_pickle=False) as arrays:
            for prefix in ("coarse","fine"):
                plot_operator_fields(output_dir / f"{prefix}_superresolution.png",
                    arrays[f"{prefix}_reference"],arrays[f"{prefix}_prediction"],arrays["labels"],
                    [arrays[f"{prefix}_z"],arrays[f"{prefix}_r"]],["z","r"])

if __name__ == "__main__":
    main()
