"""Plot the saved packed-bed direct/POD DeepONet comparison."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

from chem_operator.example_paths import ExamplePaths
from chem_operator.plotting import plot_deeponet_artifacts


PATHS = ExamplePaths.from_script(__file__)
ARTIFACTS = PATHS.example / "results" / "deeponet"
PLOT_CASES = 2


def main() -> None:
    paths = plot_deeponet_artifacts(
        ARTIFACTS,
        selected_labels=("T", "velocity", "X[0]", "Z[0]"),
        coordinate_label="Axial position z [m]",
        cases=PLOT_CASES,
    )
    print("Plots written to " + ", ".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
