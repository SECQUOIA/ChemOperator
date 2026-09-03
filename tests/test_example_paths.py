"""Tests for conventional research-example paths."""

from pathlib import Path

import pytest

from chem_operator.example_paths import ExamplePaths


def project_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create the minimum directory shape needed for root discovery."""
    root = tmp_path / "project"
    script = root / "scripts" / "pipe_flow_transient" / "transient_fno.py"
    script.parent.mkdir(parents=True)
    (root / "pyproject.toml").touch()
    script.touch()
    return root, script


def test_from_script_discovers_conventional_paths(tmp_path: Path) -> None:
    root, script = project_tree(tmp_path)

    paths = ExamplePaths.from_script(script, dataset="pipe_flow_transient")

    assert paths.root == root
    assert paths.datasets == root / "datasets"
    assert paths.data == root / "datasets" / "pipe_flow_transient"
    assert paths.example == root / "scripts" / "pipe_flow_transient"
    assert paths.output == paths.example / "results" / "transient_fno"
    assert paths.ray == root / ".ray"


def test_absolute_dataset_and_resolve_paths_are_preserved(tmp_path: Path) -> None:
    root, script = project_tree(tmp_path)
    external = tmp_path / "external-data"
    paths = ExamplePaths.from_script(script, dataset=external)

    assert paths.data == external
    assert paths.resolve(external) == external
    assert paths.resolve("relative/output") == root / "relative/output"


def test_dataset_is_optional(tmp_path: Path) -> None:
    _, script = project_tree(tmp_path)

    assert ExamplePaths.from_script(script).data is None


def test_missing_project_marker_raises(tmp_path: Path) -> None:
    script = tmp_path / "scripts" / "example.py"
    script.parent.mkdir()
    script.touch()

    with pytest.raises(FileNotFoundError, match="pyproject.toml"):
        ExamplePaths.from_script(script)
