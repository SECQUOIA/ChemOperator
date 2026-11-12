"""CLI contracts shared by model and plotting scripts."""

from __future__ import annotations

import argparse
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
        "scripts/cstr/plot.py",
        "scripts/packed_bed_1d/plot.py",
        "scripts/pfr/plot.py",
        "scripts/pfr_heat/plot.py",
        "scripts/pipe_flow/plot.py",
        "scripts/pipe_flow_transient/plot.py",
        "scripts/q2d/plot.py",
        "scripts/cstr/plot.py",
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
    assert "--recent" in help_result.stdout
    assert "--runs-root RUNS_ROOT" in help_result.stdout
    assert "--output-dir OUTPUT_DIR" in help_result.stdout
    assert "--cases CASES" in help_result.stdout
    assert "--deeponet-run" not in help_result.stdout

    missing_result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, check=False,
    )
    assert missing_result.returncode != 0
    assert "--run" in missing_result.stderr
