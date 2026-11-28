"""Tests for the experiments package's project root and entry-point defaults."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pytest

import experiments.paths as experiment_paths
from experiments.paths import PROJECT_ROOT_ENV, example_paths, project_root

REPO = Path(__file__).resolve().parents[1]


def fake_package(tmp_path: Path, *, workspace: bool) -> Path:
    """Lay out ``<ws>/experiments/src/experiments/paths.py`` and return that file."""
    workspace_root = tmp_path / "ws"
    member = workspace_root / "experiments"
    module = member / "src" / "experiments" / "paths.py"
    module.parent.mkdir(parents=True)
    module.touch()
    # The member's own pyproject.toml must not count as the project root.
    (member / "pyproject.toml").write_text('[project]\nname = "experiments"\n')
    table = '[tool.uv.workspace]\nmembers = ["experiments"]\n' if workspace else ""
    (workspace_root / "pyproject.toml").write_text(f'[project]\nname = "lib"\n{table}')
    return module


def test_default_root_is_this_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    assert project_root() == REPO


def test_empty_setting_falls_back_to_the_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, "")
    assert project_root() == REPO


def test_workspace_root_skips_the_member_pyproject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    monkeypatch.setattr(experiment_paths, "__file__", str(fake_package(tmp_path, workspace=True)))
    assert project_root() == (tmp_path / "ws").resolve()


def test_missing_workspace_asks_for_the_setting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(PROJECT_ROOT_ENV, raising=False)
    monkeypatch.setattr(experiment_paths, "__file__", str(fake_package(tmp_path, workspace=False)))
    with pytest.raises(FileNotFoundError, match=PROJECT_ROOT_ENV):
        project_root()


def test_setting_overrides_the_workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path))
    assert project_root() == tmp_path.resolve()

    script = REPO / "experiments" / "src" / "experiments" / "pfr" / "fno.py"
    paths = example_paths(script, "pfr")
    assert paths.root == tmp_path.resolve()
    assert paths.data == tmp_path.resolve() / "datasets" / "pfr"
    assert paths.ray == tmp_path.resolve() / ".ray"
    assert paths.example == script.parent


def test_setting_must_be_a_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(PROJECT_ROOT_ENV, str(tmp_path / "missing"))
    with pytest.raises(NotADirectoryError, match=PROJECT_ROOT_ENV):
        project_root()


@pytest.mark.parametrize("preset", (None, "pdf"))
def test_package_import_defaults_matplotlib_backend(preset: str | None) -> None:
    env = {key: value for key, value in os.environ.items()
           if key not in {"MPLBACKEND", "MPLCONFIGDIR"}}
    if preset is not None:
        env["MPLBACKEND"] = preset
    result = subprocess.run(
        [sys.executable, "-c",
         "import os, experiments; print(os.environ['MPLBACKEND'], os.environ['MPLCONFIGDIR'])"],
        cwd=REPO, env=env, capture_output=True, text=True, check=True,
    )
    assert result.stdout.split() == [preset or "Agg", "/tmp/matplotlib"]
