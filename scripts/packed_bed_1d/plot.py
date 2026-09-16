"""Plot canonical packed-bed direct/POD DeepONet runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import plot_deeponet_runs


PATHS = ExamplePaths.from_script(__file__)
PLOT_CASES = 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deeponet-run", type=Path, required=True)
    parser.add_argument("--pod-deeponet-run", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.example / "results" / "deeponet_comparison",
    )
    parser.add_argument("--cases", type=int, default=PLOT_CASES)
    args = parser.parse_args()
    paths = plot_deeponet_runs(
        args.deeponet_run,
        args.pod_deeponet_run,
        selected_labels=("T", "velocity", "X[0]", "Z[0]"),
        coordinate_label="Axial position z [m]",
        output_dir=args.output_dir,
        cases=args.cases,
    )
    print("Plots written to " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
