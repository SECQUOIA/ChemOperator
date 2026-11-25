"""Experiment scripts for chem_operator (unpublished uv workspace member).

Run a script as a module, for example ``python -m experiments.pipe_flow.fno --help``.
Data and outputs live under the project root (see :mod:`experiments.paths`).
"""

import os

# Scripts choose the plotting environment; the library never does. The package
# is imported before any script module body runs, so these apply before Matplotlib.
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
