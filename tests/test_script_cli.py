"""CLI contracts shared by model and plotting scripts."""

from __future__ import annotations

import argparse
from functools import partial
import importlib
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from chem_operator.experiments import (
    WorkflowStages,
    add_workflow_arguments,
    final_epochs,
    parse_operator_args,
    parse_workflow_args,
)


ROOT = Path(__file__).resolve().parents[1]


def test_workflow_stages_require_an_enabled_stage() -> None:
    with pytest.raises(ValueError, match="At least one of tuning or training"):
        WorkflowStages(tune=False, train=False)


@pytest.mark.parametrize(
    ("argv", "tune", "train"),
    (([], True, True), (["--no-train"], True, False), (["--no-tune"], False, True)),
)
def test_parse_workflow_args_preserves_enabled_stages(argv, tune, train) -> None:
    parser = argparse.ArgumentParser()
    add_workflow_arguments(parser)
    parser.add_argument("--model-option", type=int, default=7)
    args = parse_workflow_args(parser, argv)
    assert (args.tune, args.train) == (tune, train)
    assert args.model_option == 7


@pytest.mark.parametrize(
    ("argv", "message"),
    (
        (["--no-tune", "--no-train"], "At least one of tuning or training"),
        (["--no-train", "--seeds", "1"], "--seeds requires --train"),
        (["--seeds", "1", "1"], "Seeds must be unique"),
        (["--tune-seed", "-1"], "--tune-seed cannot be negative"),
        (["--plot-cases", "0"], "plot_cases must be positive"),
        (["--train-config", ""], "train_config must be non-empty"),
    ),
)
def test_workflow_parsers_report_validation_as_cli_errors(
    argv, message, tmp_path, monkeypatch, capsys,
) -> None:
    # Exercise the helper and the operator parser used by the other examples.
    parser = argparse.ArgumentParser()
    add_workflow_arguments(parser)
    paths = SimpleNamespace(root=tmp_path, data=tmp_path)
    monkeypatch.setattr(sys, "argv", ["model.py", *argv])
    for parse in (
        lambda: parse_workflow_args(parser),
        lambda: parse_operator_args(paths, epochs=1, tune_epochs=1, samples=1),
    ):
        with pytest.raises(SystemExit) as error:
            parse()
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        assert "usage:" in stderr
        assert message in stderr
        assert "Traceback" not in stderr


@pytest.mark.parametrize("problem", ("cstr", "packed_bed_1d", "pfr", "pipe_flow"))
def test_model_parsers_validate_before_resolving_cache(problem, monkeypatch, capsys) -> None:
    import importlib

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


def _operator_args(monkeypatch, tmp_path, argv, *, epochs=None, tune_epochs=5):
    monkeypatch.setattr(sys, "argv", ["model.py", *argv])
    paths = SimpleNamespace(root=tmp_path, data=tmp_path)
    return parse_operator_args(paths, epochs=epochs, tune_epochs=tune_epochs, samples=1)


@pytest.mark.parametrize(
    ("argv", "expected"),
    (([], 10), (["--tune-epochs", "3"], 6), (["--epochs", "7"], 7),
     (["--tune-epochs", "3", "--epochs", "1"], 1)),
)
def test_final_epochs_is_twice_tune_epochs_unless_overridden(
    argv, expected, monkeypatch, tmp_path,
) -> None:
    args = _operator_args(monkeypatch, tmp_path, argv)
    assert final_epochs(args) == expected


def test_explicit_epochs_default_is_still_final(monkeypatch, tmp_path) -> None:
    args = _operator_args(monkeypatch, tmp_path, [], epochs=40, tune_epochs=20)
    assert args.epochs == 40
    assert final_epochs(args) == 40


@pytest.mark.parametrize("value", ("0", "-2"))
def test_non_positive_epochs_override_is_rejected(value, monkeypatch, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        _operator_args(monkeypatch, tmp_path, ["--epochs", value])
    assert error.value.code == 2
    assert "Epochs and samples must be positive" in capsys.readouterr().err
    with pytest.raises(ValueError, match="--epochs must be positive"):
        final_epochs(SimpleNamespace(epochs=int(value), tune_epochs=5))


def test_optional_epochs_help_names_the_default(monkeypatch, tmp_path, capsys) -> None:
    with pytest.raises(SystemExit):
        _operator_args(monkeypatch, tmp_path, ["--help"])
    assert "2 x --tune-epochs" in " ".join(capsys.readouterr().out.split())


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
