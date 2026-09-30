"""CLI contracts shared by model and plotting scripts."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest

from chem_operator.experiments import add_workflow_arguments


ROOT = Path(__file__).resolve().parents[1]


def test_model_workflow_help_omits_removed_flags() -> None:
    parser = argparse.ArgumentParser()
    add_workflow_arguments(parser)
    help_text = parser.format_help()
    assert "--tune" in help_text
    assert "--train" in help_text
    assert "--plot-cases" in help_text
    assert "--generate" not in help_text
    assert "--plot " not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(["--generate"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--plot"])


@pytest.mark.parametrize(
    "script",
    (
        "scripts/pfr/plot.py",
        "scripts/q2d/plot.py",
    ),
)
def test_plot_scripts_require_run_paths(script: str) -> None:
    command = [sys.executable, str(ROOT / script)]
    help_result = subprocess.run(
        [*command, "--help"], cwd=ROOT, capture_output=True, text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--run RUN" in help_result.stdout
    assert "--deeponet-run" not in help_result.stdout

    missing_result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert missing_result.returncode != 0
    assert "--run" in missing_result.stderr


def test_plot_all_uses_repeatable_run_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    path = ROOT / "scripts/plot_all.py"
    spec = importlib.util.spec_from_file_location("plot_all_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[list[str]] = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, **_: calls.append(command))

    module.run_plot(
        "PFR", "scripts/pfr/plot.py", Path("direct"), Path("pod")
    )

    assert len(calls) == 1
    assert calls[0][-4:] == ["--run", "direct", "--run", "pod"]
