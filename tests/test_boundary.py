"""chem_operator stands alone: no ``experiments`` imports and no repo-layout assumptions."""

from __future__ import annotations

import ast
from pathlib import Path
import re

import chem_operator

PACKAGE = Path(chem_operator.__file__).resolve().parent
SOURCES = sorted(PACKAGE.rglob("*.py"))

# ``Path(__file__)`` followed by ``.parent`` hops or ``.parents[k]``.
_PARENT_HOPS = re.compile(r"__file__\)(?:\.resolve\(\))?((?:\.parent)+)\b")
_PARENTS_INDEX = re.compile(r"__file__\)(?:\.resolve\(\))?\.parents\[(\d+)\]")


def _label(path: Path) -> str:
    return path.relative_to(PACKAGE.parent).as_posix()


def test_library_never_imports_experiments() -> None:
    found = []
    for path in SOURCES:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules = [node.module]
            else:
                continue
            found += [f"{_label(path)}:{node.lineno} imports {module}"
                      for module in modules if module.split(".")[0] == "experiments"]
    assert not found, "\n".join(found)


def test_library_assumes_no_repo_layout() -> None:
    """Paths stay inside the installed package; no project files or PYTHONPATH edits."""
    found = []
    for path in SOURCES:
        # Directories between this module and the package root's parent: leaving the
        # package means reaching file.parents[k] with k >= depth.
        depth = len(path.relative_to(PACKAGE).parts)
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            where = f"{_label(path)}:{lineno}"
            for match in _PARENT_HOPS.finditer(line):
                if match.group(1).count(".parent") - 1 >= depth:
                    found.append(f"{where} climbs out of the package: {line.strip()}")
            for match in _PARENTS_INDEX.finditer(line):
                if int(match.group(1)) >= depth:
                    found.append(f"{where} climbs out of the package: {line.strip()}")
            if "pyproject.toml" in line:
                found.append(f"{where} looks for pyproject.toml: {line.strip()}")
            if "PYTHONPATH" in line:
                found.append(f"{where} touches PYTHONPATH: {line.strip()}")
    assert not found, "\n".join(found)
