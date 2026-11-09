"""CLI contracts shared by model and plotting scripts."""

from __future__ import annotations

import argparse
import importlib.util
import json
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


def test_plot_all_runs_child_with_current_python_without_uv(tmp_path, monkeypatch):
    from scripts import plot_all

    script = tmp_path / "child.py"
    script.write_text(
        "import json, sys\nfrom pathlib import Path\n"
        "Path('child.json').write_text(json.dumps([sys.executable, sys.argv[1:]]))\n"
    )
    monkeypatch.setattr(plot_all, "ROOT", tmp_path)
    monkeypatch.setenv("PATH", str(tmp_path))
    plot_all.run_plot("child", "child.py", Path("saved-run"), cases="1")
    interpreter, arguments = json.loads((tmp_path / "child.json").read_text())
    assert interpreter == sys.executable
    assert arguments == ["--run", "saved-run", "--cases", "1"]


def test_plot_all_discovers_transient_physicsnemo_variants(tmp_path, monkeypatch):
    from scripts import plot_all

    models = ("transient_nemo_fno_data", "transient_nemo_fno_physics")
    paths = [tmp_path / "artifacts/runs/pipe_flow_transient" / model / "run-1" for model in models]
    for path in paths:
        path.mkdir(parents=True)
    monkeypatch.setattr(plot_all, "ROOT", tmp_path)
    calls = {}
    monkeypatch.setattr(plot_all, "run_plot", lambda name, script, *runs: calls.update({name: runs}))
    plot_all.main()
    assert tuple(run for run in calls["transient pipe flow"] if run is not None) == tuple(paths)
