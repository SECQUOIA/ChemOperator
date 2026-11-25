"""Locate the project directory that holds ``datasets/``, ``artifacts/`` and ``.ray/``."""

from __future__ import annotations

import os
from os import PathLike
from pathlib import Path
import tomllib

from chem_operator.example_paths import ExamplePaths

PROJECT_ROOT_ENV = "EXPERIMENTS_PROJECT_ROOT"


def project_root() -> Path:
    """Return ``$EXPERIMENTS_PROJECT_ROOT``, or else the uv workspace root above this package.

    The workspace root is the nearest directory whose ``pyproject.toml`` declares
    ``[tool.uv.workspace]``. A member's own ``pyproject.toml`` (such as
    ``experiments/pyproject.toml``) does not count.
    """
    configured = os.environ.get(PROJECT_ROOT_ENV)
    if configured:
        root = Path(configured).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"{PROJECT_ROOT_ENV}={configured!r} is not a directory.")
        return root
    package = Path(__file__).resolve().parent
    for directory in package.parents:
        if _declares_workspace(directory / "pyproject.toml"):
            return directory
    raise FileNotFoundError(
        f"No uv workspace root above {package}. Set {PROJECT_ROOT_ENV} to the directory "
        "that holds datasets/ and artifacts/."
    )


def example_paths(script_file: str | PathLike[str],
                  dataset: str | PathLike[str] | None = None) -> ExamplePaths:
    """Return :class:`ExamplePaths` for *script_file*, rooted at :func:`project_root`."""
    return ExamplePaths.from_script(script_file, dataset, root=project_root())


def _declares_workspace(pyproject: Path) -> bool:
    if not pyproject.is_file():
        return False
    with pyproject.open("rb") as stream:
        return "workspace" in tomllib.load(stream).get("tool", {}).get("uv", {})
