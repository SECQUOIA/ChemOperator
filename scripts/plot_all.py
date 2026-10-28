"""Run every documented plotting entry point for the newest available runs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def latest_run(parent: str) -> Path | None:
    path = ROOT / parent
    runs = [item for item in path.iterdir() if item.is_dir()] if path.is_dir() else []
    return max(runs, key=lambda item: item.stat().st_mtime) if runs else None


def run_plot(name: str, script: str, *runs: Path | None, **options: str) -> None:
    selected = [run for run in runs if run is not None]
    if not selected:
        print(f"Skipping {name}: no runs were found.", file=sys.stderr)
        return

    command = ["uv", "run", "python", str(ROOT / script)]
    for run in selected:
        command.extend(("--run", str(run)))
    for option, value in options.items():
        command.extend((f"--{option.replace('_', '-')}", value))
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> None:
    run_plot(
        "CSTR",
        "scripts/cstr/plot.py",
        latest_run("artifacts/runs/cstr_non_isothermal/deeponet"),
        latest_run("artifacts/runs/cstr_non_isothermal/pod_deeponet"),
    )
    run_plot(
        "PFR",
        "scripts/pfr/plot.py",
        latest_run("artifacts/runs/pfr_chain/deeponet"),
        latest_run("artifacts/runs/pfr_chain/pod_deeponet"),
    )
    run_plot(
        "packed bed",
        "scripts/packed_bed_1d/plot.py",
        latest_run("artifacts/runs/packed_bed_1d/deeponet"),
        latest_run("artifacts/runs/packed_bed_1d/pod_deeponet"),
    )

    direct = latest_run("artifacts/runs/pipe_flow/deeponet")
    pod = latest_run("artifacts/runs/pipe_flow/pod_deeponet")
    if direct is not None or pod is not None:
        run_plot(
            "pipe flow",
            "scripts/pipe_flow/plot.py",
            direct,
            pod,
        )
    else:
        run_plot(
            "pipe flow",
            "scripts/pipe_flow/plot.py",
            latest_run("artifacts/runs/pipe_flow/physics_deeponet_data"),
            latest_run("artifacts/runs/pipe_flow/physics_deeponet"),
        )

    run_plot(
        "transient pipe flow",
        "scripts/pipe_flow_transient/plot.py",
        latest_run("artifacts/runs/pipe_flow_transient/transient_fno"),
        latest_run("artifacts/runs/pipe_flow_transient/transient_nemo_fno"),
    )
    run_plot(
        "PFR heat",
        "scripts/pfr_heat/plot.py",
        latest_run("artifacts/runs/pfr_heat/fno_physics"),
        latest_run("artifacts/runs/pfr_heat/deeponet_fno_physics"),
        latest_run("artifacts/runs/pfr_heat/fno_data"),
        latest_run("artifacts/runs/pfr_heat/deeponet_fno_data"),
    )
    run_plot("Q2D", "scripts/q2d/plot.py", latest_run("artifacts/runs/q2d_cmr/fno"))


if __name__ == "__main__":
    main()
