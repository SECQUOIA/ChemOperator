"""Library CLI contracts: workflow stages, operator arguments and final epochs."""

from __future__ import annotations

import argparse
import sys
from types import SimpleNamespace

import pytest

from chem_operator.experiments import (
    WorkflowStages,
    add_workflow_arguments,
    final_epochs,
    parse_operator_args,
    parse_workflow_args,
    run_context_from_namespace,
)


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
        lambda: parse_operator_args(paths, tune_epochs=1, samples=1),
    ):
        with pytest.raises(SystemExit) as error:
            parse()
        assert error.value.code == 2
        stderr = capsys.readouterr().err
        assert "usage:" in stderr
        assert message in stderr
        assert "Traceback" not in stderr


def _operator_args(monkeypatch, tmp_path, argv, *, tune_epochs=5):
    monkeypatch.setattr(sys, "argv", ["model.py", *argv])
    paths = SimpleNamespace(root=tmp_path, data=tmp_path)
    return parse_operator_args(paths, tune_epochs=tune_epochs, samples=1)


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


@pytest.mark.parametrize(
    ("argv", "expected"),
    (([], {"final_epochs": 10, "final_epochs_source": "2x_tune"}),
     (["--epochs", "3"], {"final_epochs": 3, "final_epochs_source": "override"})),
)
def test_run_provenance_records_final_epochs_and_source(
    argv, expected, monkeypatch, tmp_path,
) -> None:
    args = _operator_args(monkeypatch, tmp_path, ["--runs-root", str(tmp_path), *argv])
    context = run_context_from_namespace(args, problem="p", model="m", seed=1)
    assert {key: context.provenance[key] for key in expected} == expected


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
