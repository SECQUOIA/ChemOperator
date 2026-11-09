"""Plot canonical pipe-flow direct/POD DeepONet runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from chem_operator.example_paths import ExamplePaths
from chem_operator.experiments import load_run
from chem_operator.plotting import plot_deeponet_run_set, plot_operator_runs


PATHS = ExamplePaths.from_script(__file__)
PLOT_CASES = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="Canonical run directory; repeat to compare compatible models.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.example / "results" / "deeponet_comparison",
    )
    parser.add_argument("--cases", type=int, default=PLOT_CASES)
    args = parser.parse_args()
    model_ids = {load_run(path).manifest["model_id"] for path in args.run}
    if not model_ids.issubset({"deeponet", "pod_deeponet"}):
        for path in plot_operator_runs(args.run, args.output_dir, cases=args.cases):
            print(path)
        return
    paths = plot_deeponet_run_set(
        args.run,
        selected_labels=("velocity",),
        coordinate_label="Radius r [m]",
        output_dir=args.output_dir,
        cases=args.cases,
    )
    print("Plots written to " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
