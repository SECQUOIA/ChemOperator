from __future__ import annotations
import os
from pathlib import Path
# from collections.abc import Iterable, Sequence, Callable
# from typing import Any, Protocol, Literal
# from io import StringIO
# import time
# from dataclasses import dataclass
import warnings
from importlib.resources import files
from typing import TYPE_CHECKING

# from tqdm import tqdm, trange

import cantera as ct
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from matplotlib.axes import Axes

os.environ["DDE_BACKEND"] = "pytorch"
# import deepxde as dde

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

datasets_path = Path(__file__).resolve().parent.parent.parent / "datasets"

def get_mechanism_file(file_name: str) -> Path:
    """
    Deprecated
    Return the path to an example mechanism or data file from https://github.com/Cantera/cantera-example-data.
    """
    warnings.warn(
        "get_mechanism_file() is deprecated. cantera-example-data is added use with ct.add_data_directory() in __init__.py",
        category=DeprecationWarning,
        stacklevel=2
    )
    return files("chem_operator.external.example_data") / file_name

def get_species_names(sample, gas=None):
    if "species_names" in sample.get("metadata", {}):
        return list(sample["metadata"]["species_names"])

    if gas is not None:
        return list(gas.species_names)

    mechanism = sample.get("metadata", {}).get("mechanism")
    if mechanism is None:
        raise ValueError("Need `gas=ct.Solution(...)` or metadata['mechanism'].")

    gas = ct.Solution(mechanism)
    return list(gas.species_names)

def to_numpy(x):
    if hasattr(x, "detach"):  # torch tensor
        return x.detach().cpu().numpy()
    return np.asarray(x)

def add_filtered_handles(ax: "Axes") -> None:
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
