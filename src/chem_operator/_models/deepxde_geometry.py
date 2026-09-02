"""Geometry type contracts used by the DeepXDE adapter."""

from collections.abc import Callable
from typing import Literal

import numpy as np

ArrayTransform = Callable[[np.ndarray], np.ndarray]
DeepXDEFormat = Literal["cartesian_product", "pointwise"]

