"""Validate bundled and Docker-backed Q2D cases and plot their fields."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

# Plotting environment variables must be set before importing Matplotlib.
# pylint: disable=wrong-import-position,too-many-locals
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np

from chem_operator.datasets import SimulationRecord
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.q2d.dataset_generator import (
    CMRSim,
    compare_record_to_legacy_csv,
    default_docker_solver_command,
)


PATHS = ExamplePaths.from_script(__file__)
DEFAULT_RUN_DIR = PATHS.example / "runs"


def _field_for_plot(record: SimulationRecord, name: str) -> tuple[np.ndarray, str]:
    if name in record.fields:
        return np.asarray(record.fields[name], dtype=float), name

    for group, species_names in record.metadata.get("field_species", {}).items():
        prefix = f"{group}_"
        if not name.startswith(prefix):
            continue

        species = name[len(prefix) :]
        if species not in species_names:
            break
        data = np.asarray(record.fields[group], dtype=float)
        return data[:, :, species_names.index(species)], name

    raise KeyError(f"{name!r} is not a field in this record.")


def plot_record_fields(
    record: SimulationRecord,
    output_dir: str | Path,
    case_name: str,
    field_names: Sequence[str] = (
        "gas_temperature",
        "velocity_axial",
        "Y_CH4",
        "Y_H2",
    ),
) -> list[Path]:
    """Plot axial profiles and z-r field maps for selected record fields."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z = np.asarray(record.coordinates["z"], dtype=float)
    r = np.asarray(record.coordinates["r"], dtype=float)
    r_min = float(r.min())
    r_max = float(r.max())
    if np.isclose(r_min, r_max):
        padding = max(abs(r_min) * 0.1, 1e-6)
        r_min -= padding
        r_max += padding
    paths: list[Path] = []

    fig, axes = plt.subplots(
        len(field_names),
        1,
        figsize=(7, 2.5 * len(field_names)),
    )
    axes = np.atleast_1d(axes)
    for ax, field_name in zip(axes, field_names):
        data, label = _field_for_plot(record, field_name)
        for i, radius in enumerate(r):
            ax.plot(z, data[:, i], label=f"r={radius:.4g} m")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("z [m]")
    fig.tight_layout()
    profile_path = output_dir / f"{case_name}_z_profiles.png"
    fig.savefig(profile_path, dpi=180)
    plt.close(fig)
    paths.append(profile_path)

    fig, axes = plt.subplots(
        1,
        len(field_names),
        figsize=(4 * len(field_names), 3.6),
    )
    axes = np.atleast_1d(axes)
    for ax, field_name in zip(axes, field_names):
        data, label = _field_for_plot(record, field_name)
        image = ax.imshow(
            data.T,
            origin="lower",
            aspect="auto",
            extent=[z.min(), z.max(), r_min, r_max],
        )
        ax.set_title(label)
        ax.set_xlabel("z [m]")
        ax.set_ylabel("r [m]")
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.tight_layout()
    zr_path = output_dir / f"{case_name}_z_r_fields.png"
    fig.savefig(zr_path, dpi=180)
    plt.close(fig)
    paths.append(zr_path)

    return paths


def run_tutorial_examples(
    output_dir: str | Path = PATHS.output,
) -> dict[str, Any]:
    """Parse both methane CMR tutorial cases, validate, and make plots."""
    results: dict[str, Any] = {}

    for annular_channel, case_name in ((False, "cmr"), (True, "cmr_annular")):
        sim = CMRSim(annular_channel=annular_channel)
        case = sim.make_case({})
        record = sim.run_case(case)

        template_dir = Path(case.mechanism_parameters["template_case_dir"])
        reference_csv = (
            template_dir / "solution_files" / "solution_1173K_10Bar_500sccm.csv"
        )
        max_abs_diff = compare_record_to_legacy_csv(record, reference_csv)
        figure_paths = plot_record_fields(record, output_dir, case_name)

        y_species = record.metadata["field_species"].get("Y", [])
        ch4_out = np.nan
        h2_out = np.nan
        if "Y" in record.fields and "CH4" in y_species:
            ch4_out = float(record.fields["Y"][-1, 0, y_species.index("CH4")])
        if "Y" in record.fields and "H2" in y_species:
            h2_out = float(record.fields["Y"][-1, 0, y_species.index("H2")])

        results[case_name] = {
            "record": record,
            "reference_csv": reference_csv,
            "figures": figure_paths,
            "max_abs_diff": max_abs_diff,
            "z_end": float(record.coordinates["z"][-1]),
            "ch4_out": ch4_out,
            "h2_out": h2_out,
            "annular_velocity_out": float(record.fields["velocity_axial"][-1, -1])
            if annular_channel
            else None,
        }

    return results


def run_radial_grid_examples(
    output_dir: str | Path = PATHS.output,
    solver_command: str | Sequence[str] | None = None,
    lumen_points: int = 4,
    work_root: str | Path = DEFAULT_RUN_DIR,
) -> dict[str, Any]:
    """Run non-tutorial CMR cases with more radial points and make plots."""
    results: dict[str, Any] = {}
    command = solver_command or default_docker_solver_command()

    for annular_channel, case_name in (
        (False, f"cmr_radial{lumen_points}"),
        (True, f"cmr_annular_radial{lumen_points}"),
    ):
        sim = CMRSim(
            annular_channel=annular_channel,
            solver_command=command,
            use_reference_if_no_solver=False,
            keep_case_dirs=True,
            work_root=work_root,
        )
        case = sim.make_case(
            {
                "lumen_points": lumen_points,
                "solve_support": 0,
                "support_points": 0,
                "refine": 0,
                "max_grid_points": 2000,
                "max_time_steps": 1000,
            }
        )
        record = sim.run_case(case)
        figure_paths = plot_record_fields(record, output_dir, case_name)

        y_species = record.metadata["field_species"].get("Y", [])
        ch4_out = np.nan
        h2_out = np.nan
        if "Y" in record.fields and "CH4" in y_species:
            ch4_out = float(record.fields["Y"][-1, 0, y_species.index("CH4")])
        if "Y" in record.fields and "H2" in y_species:
            h2_out = float(record.fields["Y"][-1, 0, y_species.index("H2")])

        results[case_name] = {
            "record": record,
            "figures": figure_paths,
            "z_end": float(record.coordinates["z"][-1]),
            "r_points": int(len(record.coordinates["r"])),
            "ch4_out_center": ch4_out,
            "h2_out_center": h2_out,
            "case_dir": record.metadata.get("case_dir"),
            "solver_mode": record.metadata.get("format"),
        }

    return results


def parse_args() -> argparse.Namespace:
    """Parse the validation workflow selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--radial-grid",
        action="store_true",
        help="Run Docker-backed radial-grid cases instead of bundled references.",
    )
    parser.add_argument(
        "--lumen-points",
        type=int,
        default=3,
        help="Radial lumen point count used with --radial-grid (default: 3).",
    )
    return parser.parse_args()


def main() -> None:
    """Run and report the selected validation workflow."""
    args = parse_args()
    if args.radial_grid:
        results = run_radial_grid_examples(lumen_points=args.lumen_points)
        for name, result in results.items():
            print(
                f"{name}: z_end={result['z_end']:.7g}, "
                f"r_points={result['r_points']}, "
                f"CH4_out_center={result['ch4_out_center']:.7g}, "
                f"H2_out_center={result['h2_out_center']:.7g}, "
                f"case_dir={result['case_dir']}"
            )
            wall_time = result["record"].metadata.get("wall_time")
            print(f"{wall_time = }")
            for figure in result["figures"]:
                print(f"{name}: wrote {figure}")
        return

    results = run_tutorial_examples()
    for name, result in results.items():
        max_diff = max(result["max_abs_diff"].values())
        print(
            f"{name}: z_end={result['z_end']:.7g}, "
            f"CH4_out={result['ch4_out']:.7g}, "
            f"H2_out={result['h2_out']:.7g}, "
            f"max_abs_tutorial_diff={max_diff:.3g}"
        )
        if result["annular_velocity_out"] is not None:
            print(
                f"{name}: annular_velocity_out="
                f"{result['annular_velocity_out']:.7g}"
            )
        for figure in result["figures"]:
            print(f"{name}: wrote {figure}")


if __name__ == "__main__":
    main()
