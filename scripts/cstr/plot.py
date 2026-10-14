"""Plot the CSTR training-set CO histories on a logarithmic time axis."""

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
from chem_operator.plotting import plot_deeponet_run_set

PATHS = ExamplePaths.from_script(__file__)
MECHANISM = "n-heptane-NUIG-2016.yaml"
PLOT_CASES = 2
DATASET = PATHS.root / "datasets" / "cstr" / "cstr_non_isothermal_train.h5"
OUTPUT = PATHS.example / "results" / "train_set_CO.png"


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
        all_positive = True
        minimum_nonzero = np.inf
        for color, case_name in zip(colors, case_names):
            case = cases[case_name]
            time = np.asarray(case["coordinates/t"])
            co = np.asarray(case["fields/X"][:, co_index])
            axis.plot(time, co, color=color, linewidth=0.9, alpha=0.75)
            all_positive = all_positive and bool(np.all(time > 0.0))
            nonzero = np.abs(time[time != 0.0])
            if nonzero.size:
                minimum_nonzero = min(minimum_nonzero, float(nonzero.min()))

    if all_positive:
        axis.set_xscale("log")
    else:
        linthresh = minimum_nonzero / 2.0 if np.isfinite(minimum_nonzero) else 1.0e-12
        axis.set_xscale("symlog", linthresh=linthresh)
    axis.set(
        xlabel="Time [s]",
        ylabel="CO mole fraction",
        title=f"CSTR training-set CO histories ({len(case_names)} cases)",
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def main() -> None:
    path = plot_co_histories()
    print(f"Plot written to {path}")

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
    fuel_index = ct.Solution(MECHANISM).species_index("NC7H16")
    paths = plot_deeponet_run_set(
        args.run,
        selected_labels=("T", "P", f"X[{fuel_index}]"),
        coordinate_label="Time [s]",
        output_dir=args.output_dir,
        cases=args.cases,
    )
    print("Plots written to " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
