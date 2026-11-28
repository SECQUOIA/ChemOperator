"""The experiments package and its tests use only chem_operator's public modules."""

from __future__ import annotations

import ast
from pathlib import Path

import experiments

PACKAGE = Path(experiments.__file__).resolve().parent
TESTS = Path(__file__).resolve().parent

# Private names the pipe_flow_transient rollout scripts still import. Steps 7-8 decide
# whether chem_operator gets public builders for them; remove an entry once it is fixed.
ALLOWED = {
    ("pipe_flow_transient/rollout_deeponet.py", "chem_operator._models.deeponet._network"),
    ("pipe_flow_transient/rollout_pod_deeponet.py",
     "chem_operator._models.deeponet._PODOnlyDeepONet"),
}


def _is_private(dotted: str) -> bool:
    parts = dotted.split(".")
    return parts[0] == "chem_operator" and any(part.startswith("_") for part in parts[1:])


def _private_imports() -> set[tuple[str, str]]:
    files = [(path, path.relative_to(PACKAGE).as_posix()) for path in PACKAGE.rglob("*.py")]
    files += [(path, f"tests/{path.relative_to(TESTS).as_posix()}") for path in TESTS.rglob("*.py")]
    found = set()
    for path, label in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            found |= {(label, name) for name in names if _is_private(name)}
    return found


def test_experiments_import_only_public_chem_operator_modules() -> None:
    found = _private_imports()
    unexpected = sorted(found - ALLOWED)
    assert not unexpected, "\n".join(f"{label} imports {name}" for label, name in unexpected)
    fixed = sorted(ALLOWED - found)
    assert not fixed, "No longer imported; remove from ALLOWED:\n" + "\n".join(
        f"{label} {name}" for label, name in fixed)
