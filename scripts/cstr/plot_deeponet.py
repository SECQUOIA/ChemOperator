"""Plot the saved CSTR direct/POD DeepONet comparison."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cantera as ct

from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import plot_deeponet_artifacts


PATHS = ExamplePaths.from_script(__file__)
ARTIFACTS = PATHS.example / "results" / "deeponet"
MECHANISM = "n-heptane-NUIG-2016.yaml"
PLOT_CASES = 2


def main() -> None:
    fuel_index = ct.Solution(MECHANISM).species_index("NC7H16")
    paths = plot_deeponet_artifacts(
        ARTIFACTS,
        selected_labels=("T", "P", f"X[{fuel_index}]"),
        coordinate_label="Time [s]",
        cases=PLOT_CASES,
    )
    print("Plots written to " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
