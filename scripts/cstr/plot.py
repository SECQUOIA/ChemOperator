"""Plot saved CSTR runs, with optional training-set CO diagnostics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cantera as ct
import h5py
import numpy as np

from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import (
    add_plot_run_arguments, plot_selected_runs, print_plot_paths,
    select_plot_runs, set_log_xscale, use_publication_style,
)

PATHS = ExamplePaths.from_script(__file__)
PROBLEM_ID = "cstr_non_isothermal"
MECHANISM = "n-heptane-NUIG-2016.yaml"
# Species drawn alongside temperature and pressure in reconstruction plots.
PLOT_SPECIES = ("NC7H16", "O2", "CO")
DATASET = PATHS.root / "datasets" / "cstr" / "cstr_non_isothermal_train.h5"
OUTPUT = PATHS.plots / PROBLEM_ID / "diagnostics" / "train_set_CO.png"


def plot_co_histories(dataset: Path = DATASET, output: Path = OUTPUT) -> Path:
    """Plot CO mole-fraction histories for every case in the dataset."""
    import matplotlib.pyplot as plt

    co_index = ct.Solution(MECHANISM).species_index("CO")
    with h5py.File(dataset, "r") as archive:
        cases = archive["cases"]
        case_names = sorted(cases)
        if not case_names:
            raise ValueError(f"Dataset contains no cases: {dataset}")

        colors = plt.colormaps["viridis"](
            np.linspace(0.0, 1.0, len(case_names))
        )
        figure, axis = plt.subplots(figsize=(8.0, 5.0))
        times = []
        for color, case_name in zip(colors, case_names):
            case = cases[case_name]
            time = np.asarray(case["coordinates/t"])
            co = np.asarray(case["fields/X"][:, co_index])
            axis.plot(time, co, color=color, linewidth=0.9, alpha=0.75)
            times.append(time)

    set_log_xscale(axis, times)
    axis.set(
        xlabel="Time [s]",
        ylabel="CO mole fraction",
        title=f"CSTR training-set CO histories ({len(case_names)} cases)",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output)
    plt.close(figure)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_plot_run_arguments(parser, root=PATHS.root, problem_id=PROBLEM_ID,
                           cases=3, required=False)
    parser.add_argument("--co-histories", action="store_true",
                        help="Also plot raw training-set CO histories; may be used without --run.")
    parser.add_argument("--dataset", type=Path, default=DATASET,
                        help="Training archive for --co-histories.")
    parser.add_argument("--co-output", type=Path, default=OUTPUT,
                        help="Output image for --co-histories.")
    args = parser.parse_args()
    if not (args.run or args.recent or args.co_histories):
        parser.error("Provide --run, --recent, or --co-histories.")
    use_publication_style(use_tex=args.use_tex)
    if args.co_histories:
        path = plot_co_histories(args.dataset, args.co_output)
        print(f"Plot written to {path}")
    if not (args.run or args.recent):
        return
    runs = select_plot_runs(args, parser, problem_id=PROBLEM_ID)
    # The mechanism saves every species; plot only T, P, and a few key species.
    # A log time axis keeps the early ignition transient visible.
    gas = ct.Solution(MECHANISM)
    labels = ("T", "P", *(f"X[{gas.species_index(name)}]" for name in PLOT_SPECIES))
    paths = plot_selected_runs(runs, args.output_dir, cases=args.cases,
                               labels=labels, xlog=True)
    print_plot_paths(paths)


if __name__ == "__main__":
    main()
