"""Plot Q2D run artifacts without loading a model or dataset."""
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
from chem_operator.plotting import plot_operator_runs, plot_operator_fields
from chem_operator.experiments import load_run
SLIDE_DPI = 220
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
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "results" / "fno")
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()
    run = load_run(args.run)
    plot_operator_runs([args.run],args.output_dir,cases=args.cases)
    benchmark = args.run / "mesh_benchmark.csv"
    if benchmark.exists():
        frame = pd.read_csv(benchmark)
        cost = json.loads((args.run / "cost_model.json").read_text())
        shape = tuple(run.manifest["benchmark_metadata"]["training_shape"])
        plot_mesh_l2(args.output_dir / "mesh_l2_vs_points.png",frame,shape)
        plot_mesh_wall_time(args.output_dir / "mesh_wall_time_vs_points.png",frame,shape,cost)
        plot_cumulative_break_even(args.output_dir / "cumulative_break_even.png",cost)
        plot_amortized_pareto(args.output_dir / "amortized_pareto_frontiers.png",cost)
        plot_break_even_map(args.output_dir / "break_even_speedup_map.png",cost)
        with np.load(args.run / "superresolution.npz",allow_pickle=False) as arrays:
            for prefix in ("coarse","fine"):
                plot_operator_fields(args.output_dir / f"{prefix}_superresolution.png",
                    arrays[f"{prefix}_reference"],arrays[f"{prefix}_prediction"],arrays["labels"],
                    [arrays[f"{prefix}_z"],arrays[f"{prefix}_r"]],["z","r"])

if __name__ == "__main__":
    main()
