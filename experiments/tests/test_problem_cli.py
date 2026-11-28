"""Every problem's model and plot scripts follow the library's CLI contracts."""

from __future__ import annotations

from functools import partial
import importlib
import subprocess
import sys

import pytest

from chem_operator.experiments import final_epochs, tuning_resources
from experiments.paths import project_root

DEFAULT_RESOURCES = {
    "cpus_per_trial": 1, "gpus_per_trial": None, "max_concurrent_trials": 1, "time_budget_s": None,
}


@pytest.mark.parametrize("problem", ("cstr", "packed_bed_1d", "pfr", "pipe_flow"))
def test_model_parsers_validate_before_resolving_cache(problem, monkeypatch, capsys) -> None:
    common = importlib.import_module(f"experiments.{problem}.common")

    def unexpected_cache_resolution(*args, **kwargs):
        pytest.fail("Invalid workflow reached cache resolution")

    monkeypatch.setattr(common, "resolve_data_cache", unexpected_cache_resolution)
    monkeypatch.setattr(sys, "argv", ["model.py", "--no-tune", "--no-train"])
    with pytest.raises(SystemExit) as error:
        common.parse_args()
    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "At least one of tuning or training" in stderr
    assert "Traceback" not in stderr


@pytest.mark.parametrize("problem", ("cstr", "packed_bed_1d", "pfr", "pipe_flow"))
def test_1d_parsers_use_the_standard_operator_cli(problem, monkeypatch, tmp_path) -> None:
    common = importlib.import_module(f"experiments.{problem}.common")
    factories = []
    monkeypatch.setattr(common, "resolve_data_cache",
                        lambda args, factory: factories.append(factory))

    monkeypatch.setattr(sys, "argv", ["model.py"])
    args = common.parse_args()
    assert args.data_dir == common.PATHS.data
    assert (args.epochs, args.max_cases) == (None, None)
    assert (args.tune_epochs, args.samples) == (common.MAX_EPOCHS, common.NUM_SAMPLES)
    assert args.plot_cases == 2
    assert final_epochs(args) == 2 * common.MAX_EPOCHS

    monkeypatch.setattr(sys, "argv", [
        "model.py", "--data-dir", str(tmp_path), "--epochs", "3", "--tune-epochs", "2",
        "--samples", "4", "--max-cases", "5", "--plot-cases", "1",
    ])
    args = common.parse_args()
    assert (args.data_dir, args.epochs, args.tune_epochs, args.samples, args.max_cases,
            args.plot_cases) == (tmp_path, 3, 2, 4, 5, 1)
    assert final_epochs(args) == 3
    # The --data-cache estimate opens the splits under --data-dir.
    assert isinstance(factories[-1], partial) and factories[-1].args == (tmp_path,)


@pytest.mark.parametrize(
    "problem", ("cstr", "pfr", "packed_bed_1d", "pipe_flow", "pfr_heat", "pipe_flow_transient", "q2d"),
)
def test_every_script_inherits_the_resource_defaults(problem, monkeypatch) -> None:
    common = importlib.import_module(f"experiments.{problem}.common")
    monkeypatch.setattr(common, "resolve_data_cache", lambda args, factory: None)
    monkeypatch.setattr(sys, "argv", ["model.py"])
    assert tuning_resources(common.parse_args()) == DEFAULT_RESOURCES


@pytest.mark.parametrize(
    "script",
    (
        "experiments.cstr.plot",
        "experiments.packed_bed_1d.plot",
        "experiments.pfr.plot",
        "experiments.pfr_heat.plot",
        "experiments.pipe_flow.plot",
        "experiments.pipe_flow_transient.plot",
        "experiments.q2d.plot",
        "experiments.cstr.plot",
    ),
)
def test_plot_scripts_require_run_paths(script: str) -> None:
    command = [sys.executable, "-m", script]
    help_result = subprocess.run(
        [*command, "--help"], cwd=project_root(), capture_output=True, text=True,
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
        command, cwd=project_root(), capture_output=True, text=True, check=False,
    )
    assert missing_result.returncode != 0
    assert "--run" in missing_result.stderr
