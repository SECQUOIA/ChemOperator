"""Tests for conventional research-example paths."""

from pathlib import Path

import pytest

from chem_operator.example_paths import ExamplePaths


def project_tree(tmp_path: Path) -> tuple[Path, Path]:
    """Create a project root and a script inside an experiments package below it."""
    root = tmp_path / "project"
    package = root / "experiments" / "src" / "experiments"
    script = package / "pipe_flow_transient" / "transient_fno.py"
    script.parent.mkdir(parents=True)
    script.touch()
    return root, script


def test_from_script_builds_conventional_paths_below_root(tmp_path: Path) -> None:
    root, script = project_tree(tmp_path)

    paths = ExamplePaths.from_script(script, dataset="pipe_flow_transient", root=root)

    assert paths.root == root
    assert paths.datasets == root / "datasets"
    assert paths.data == root / "datasets" / "pipe_flow_transient"
    assert paths.example == script.parent
    assert paths.output == paths.example / "results" / "transient_fno"
    assert paths.plots == root / "artifacts" / "plots"
    assert paths.ray == root / ".ray"


def test_absolute_dataset_and_resolve_paths_are_preserved(tmp_path: Path) -> None:
    root, script = project_tree(tmp_path)
    external = tmp_path / "external-data"
    paths = ExamplePaths.from_script(script, dataset=external, root=root)

    assert paths.data == external
    assert paths.resolve(external) == external
    assert paths.resolve("relative/output") == root / "relative/output"


def test_dataset_is_optional(tmp_path: Path) -> None:
    root, script = project_tree(tmp_path)

    assert ExamplePaths.from_script(script, root=root).data is None


def test_root_is_explicit_and_need_not_contain_the_script(tmp_path: Path) -> None:
    """The library never infers a project layout from where the script lives."""
    _, script = project_tree(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()

    with pytest.raises(TypeError, match="root"):
        ExamplePaths.from_script(script)  # pylint: disable=missing-kwoa

    paths = ExamplePaths.from_script(script, dataset="pfr", root=elsewhere)
    assert paths.root == elsewhere
    assert paths.data == elsewhere / "datasets" / "pfr"
    assert paths.example == script.parent
