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


def test_q2d_canonical_training_benchmark_and_plotting(tmp_path, monkeypatch):
    """A small synthetic grid verifies the whole FNO artifact and mesh contract."""
    from argparse import Namespace
    import torch
    from chem_operator.datasets import SimulationDatasetGenerator, SimulationRecord
    from chem_operator.experiments import ExperimentRunner, RunContext, load_run
    from scripts.q2d import fno, common, plot

    def record(nz, nr, offset=0.):
        z,r = np.linspace(0.,1.,nz),np.linspace(0.,.5,nr)
        base = z[:,None] + r[None,:] + offset
        return SimulationRecord(
            coordinates={"z": z,"r": r}, fields={"X": np.stack((.2+.1*base,.4-.1*base),axis=-1),
                                                      "theta": (.1+.01*base)[...,None]},
            constants={"SCCM": 500.+offset}, metadata={"params": {"T0": 1173.+offset,"sccm": 500.+offset},
                "field_species": {"X": ["CH4","H2"],"theta": ["C(s)"]},"wall_time": .01,
                "input_values": {"LENGTH_LUMEN": 1.,"CHANNELRAD_LUMEN": .5}})

    generator = SimulationDatasetGenerator(CMRSim(),tmp_path / "data")
    for split in ("train","valid","test"):
        generator.save_split(split,[record(6,4,0.),record(6,4,.1)])
    for nz,nr in ((6,4),(8,6)):
        generator.save_split(f"{nz}_{nr}_test",[record(nz,nr)])
    normalizer,geometry,shape = common.fit_normalizer(tmp_path / "data" / "q2d_cmr_train.h5")
    raw,data = common.make_adapter(tmp_path / "data" / "q2d_cmr_test.h5",normalizer,geometry)
    context = RunContext.create(tmp_path / "runs",problem="q2d_cmr",model="fno",run_id="smoke",seed=42)
    args = Namespace(data_dir=tmp_path / "data",samples=1,tune_epochs=1,plot_cases=1,max_cases=None)
    config = dict(modes_z=2,modes_r=2,hidden_channels=4,n_layers=1,learning_rate=1e-3,
                  weight_decay=0.,batch_size=1,epochs=1)
    trainer = fno.make_trainer(config,context,normalizer,geometry,shape)
    try:
        result = ExperimentRunner(context,common.experiment_spec(args,"fno")).run(trainer,data,data,data,config=config,
            evaluator=lambda t,d,c: fno.evaluate_fields(t,d,c,labels=tuple(ch.label for ch in common.OUTPUT_CHANNELS),
                                                       coordinate_names=("z","r"),cases=1))
    finally:
        raw.close()
    monkeypatch.setattr(fno,"LATENCY_WARMUPS",0)
    monkeypatch.setattr(fno,"LATENCY_REPEATS",1)
    fno.save_benchmarks(trainer,result,data_dir=tmp_path / "data",normalizer=normalizer,geometry=geometry,shape=shape)
    manifest = load_run(context.paths.run_dir).manifest
    assert len(manifest["benchmark_metadata"]["dataset_fingerprints"]) == 2
    checkpoint = torch.load(context.paths.checkpoint(),weights_only=True)
    assert any(value.is_complex() for value in checkpoint["state_dict"].values())
    # Plotting must continue working when datasets are inaccessible.
    (tmp_path / "data").rename(tmp_path / "hidden-data")
    monkeypatch.setattr("sys.argv",["plot.py","--run",str(context.paths.run_dir),"--output-dir",str(tmp_path / "plots")])
    plot.main()
    assert (tmp_path / "plots" / "fine_superresolution.png").is_file()
    assert (tmp_path / "plots" / "break_even_speedup_map.png").is_file()
