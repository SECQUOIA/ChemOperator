"""Plot canonical pipe-flow direct/POD DeepONet runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import plot_deeponet_runs, plot_operator_runs


PATHS = ExamplePaths.from_script(__file__)
PLOT_CASES = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", help="Canonical physics or other run; repeat to compare.")
    parser.add_argument("--deeponet-run", type=Path)
    parser.add_argument("--pod-deeponet-run", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.example / "results" / "deeponet_comparison",
    )
    parser.add_argument("--cases", type=int, default=PLOT_CASES)
    args = parser.parse_args()
    if args.run:
        if args.deeponet_run or args.pod_deeponet_run:
            parser.error("Use --run or the paired DeepONet arguments.")
        for path in plot_operator_runs(args.run, args.output_dir, cases=args.cases):
            print(path)
        return
    if not args.deeponet_run or not args.pod_deeponet_run:
        parser.error("Provide --run or both DeepONet run paths.")
    paths = plot_deeponet_runs(
        args.deeponet_run,
        args.pod_deeponet_run,
        selected_labels=("velocity",),
        coordinate_label="Radius r [m]",
        output_dir=args.output_dir,
        cases=args.cases,
    )
    print("Plots written to " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
