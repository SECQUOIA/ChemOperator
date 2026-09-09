"""Tests for the quasi-2D CMR solver adapter and dataset configuration."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from chem_operator.reactors.q2d.dataset_generator import (
    CMRSim,
    DEFAULT_TUTORIAL_CASES,
    compare_record_to_legacy_csv,
    default_docker_solver_command,
    parse_q2d_grid_csv,
    read_input_dat,
    write_input_dat,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "q2d" / "generate_dataset.py"
)
SCRIPT_SPEC = importlib.util.spec_from_file_location(
    "q2d_generate_dataset",
    SCRIPT_PATH,
)
assert SCRIPT_SPEC is not None and SCRIPT_SPEC.loader is not None
GENERATE_DATASET = importlib.util.module_from_spec(SCRIPT_SPEC)
SCRIPT_SPEC.loader.exec_module(GENERATE_DATASET)


def test_input_dat_round_trip_preserves_overrides(tmp_path) -> None:
    """Known fields can be changed without losing unrelated template values."""
    template = DEFAULT_TUTORIAL_CASES[False] / "input.dat"
    values = read_input_dat(template)
    values.update({"INLETTEMP": 1200.0, "REFINE": False})
    output = tmp_path / "input.dat"

    write_input_dat(template, output, values)
    rendered = read_input_dat(output)

    assert rendered["INLETTEMP"] == 1200.0
    assert rendered["REFINE"] is False
    assert rendered["GASCTIFILE_LUMEN"] == values["GASCTIFILE_LUMEN"]


def test_tutorial_reference_is_parsed_without_external_solver() -> None:
    """The default simulator remains usable without Docker."""
    simulator = CMRSim()
    case = simulator.make_case({})
    record = simulator.run_case(case)
    reference = (
        DEFAULT_TUTORIAL_CASES[False]
        / "solution_files"
        / "solution_1173K_10Bar_500sccm.csv"
    )

    differences = compare_record_to_legacy_csv(record, reference)

    assert record.metadata["format"] == "legacy_wide_csv"
    assert record.coordinates["z"].ndim == 1
    assert record.fields["velocity_axial"].shape[1] == 1
    assert max(differences.values()) == 0.0


def test_grid_csv_parser_builds_coordinate_and_species_axes(tmp_path) -> None:
    """Long-form solver exports become dense z-r fields and species groups."""
    csv_path = tmp_path / "q2d_grid_example.csv"
    csv_path.write_text(
        "axial_index,radial_index,region,z,r,pressure,Y_CH4,Y_H2\n"
        "0,0,lumen,0.0,0.001,10.0,0.8,0.2\n"
        "0,1,support,0.0,0.002,9.0,0.7,0.3\n"
        "1,0,lumen,0.1,0.001,8.0,0.6,0.4\n"
        "1,1,support,0.1,0.002,7.0,0.5,0.5\n",
        encoding="utf-8",
    )
    manifest_path = tmp_path / "q2d_grid_manifest_example.json"
    manifest_path.write_text(json.dumps({"dimensions": {"axial_points": 2}}))

    record = parse_q2d_grid_csv(csv_path, manifest_path=manifest_path)

    np.testing.assert_allclose(record.coordinates["z"], [0.0, 0.1])
    np.testing.assert_allclose(record.coordinates["r"], [0.001, 0.002])
    np.testing.assert_allclose(record.fields["pressure"], [[10.0, 9.0], [8.0, 7.0]])
    assert record.fields["Y"].shape == (2, 2, 2)
    assert record.metadata["field_species"]["Y"] == ["CH4", "H2"]
    assert record.metadata["region_labels"] == {"lumen": 0, "support": 1}
    assert record.metadata["manifest"]["dimensions"]["axial_points"] == 2


def test_dataset_script_exports_shared_simulator_configuration() -> None:
    """Model workflows can reuse the dataset runner's Q2D configuration."""
    q2d_parameter_space = GENERATE_DATASET.q2d_parameter_space
    q2d_simulator = GENERATE_DATASET.q2d_simulator

    assert q2d_simulator.name == "q2d_cmr"
    assert q2d_simulator.parameter_space is not q2d_parameter_space
    assert set(q2d_simulator.parameter_space) == set(q2d_parameter_space)
    assert q2d_simulator.solver_command == default_docker_solver_command()
    assert q2d_simulator.use_reference_if_no_solver is False
