"""Plot canonical pipe-flow direct/POD DeepONet runs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import add_plot_run_arguments, select_plot_runs, plot_selected_runs, print_plot_paths


PATHS = ExamplePaths.from_script(__file__)
PLOT_CASES = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_run_arguments(parser, root=PATHS.root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PATHS.example / "results" / "deeponet_comparison",
    )
    parser.add_argument("--cases", type=int, default=PLOT_CASES)
    args = parser.parse_args()
    runs = select_plot_runs(args, parser, problem_id="pipe_flow")
    paths = plot_selected_runs(runs, args.output_dir, cases=args.cases)
    print_plot_paths(paths)


if __name__ == "__main__":
    main()
