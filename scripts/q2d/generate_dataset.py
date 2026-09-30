"""Generate canonical, resolution-sweep, and tolerance-sweep Q2D datasets."""

from collections.abc import Mapping

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.q2d.dataset_generator import (
    CMRSim,
    default_docker_solver_command,
)
from chem_operator.sampling import Constant, Grid, ParameterSpec, Uniform


RESOLUTIONS = ((14, 6), (14, 10), (14, 14), (14, 20))
STEADY_RTOLS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)

q2d_parameter_space = {
    "annular_channel": Constant(False),
    "T0": Uniform(1150.0, 1240.0),
    "sccm": Uniform(470.0, 530.0),
    "P0": Constant(1.0e6),
    "lumen_points": Constant(6),
    "mesh_points": Constant(14),
    "refine": Constant(False),
    "rtol_ss": Constant(1e-3),
}

q2d_multiresolution_parameter_space = {
    resolution: q2d_parameter_space
    | {
        "mesh_points": Constant(resolution[0]),
        "lumen_points": Constant(resolution[1]),
    }
    for resolution in RESOLUTIONS
}

q2d_tolerance_parameter_space = {
    "T0": Constant(1173.0),
    "P0": Constant(1e6),
    "sccm": Constant(500.0),
    "mesh_points": Constant(14),
    "lumen_points": Constant(20),
    "refine": Constant(False),
    "rtol_ss": Grid(STEADY_RTOLS),
}

q2d_tolerance_interp_parameter_space = {
    "T0": Constant(1173.0),
    "P0": Constant(1e6),
    "sccm": Constant(500.0),
    "mesh_points": Constant(6),
    "lumen_points": Constant(14),
    "refine": Constant(False),
    "rtol_ss": Grid(STEADY_RTOLS),
}


def _simulator(parameter_space: Mapping[str, ParameterSpec], name: str) -> CMRSim:
    """Construct one Docker-backed simulator with a unique dataset name."""
    simulator = CMRSim(
        parameter_space=parameter_space,
        solver_command=default_docker_solver_command(),
        use_reference_if_no_solver=False,
    )
    simulator.name = name
    return simulator


q2d_simulator = _simulator(q2d_parameter_space, "q2d_cmr")


def _dataset_path():
    return ExamplePaths.from_script(__file__).datasets / "q2d_cmr"


def generate_dataset(n_cases: int = 100) -> None:
    """Generate and overwrite the canonical 14-by-6 dataset splits."""
    q2d_dataset_generator = SimulationDatasetGenerator(q2d_simulator, _dataset_path())
    records_splits = q2d_dataset_generator.generate_splits(n_cases=n_cases)
    q2d_dataset_generator.save_splits(records_splits, overwrite=True)


def generate_multiresolution_datasets(n_cases: int = 3) -> None:
    """Generate one matched test split at each fixed mesh resolution."""
    for (n_z, n_r), parameter_space in q2d_multiresolution_parameter_space.items():
        simulator = _simulator(parameter_space, f"q2d_cmr_{n_z}_{n_r}")
        generator = SimulationDatasetGenerator(simulator, _dataset_path())
        records = generator.generate_split("test", n_cases, generator.seed + 3)
        generator.save_split("test", records, overwrite=True)


def _generate_tolerance_test(
    parameter_space: Mapping[str, ParameterSpec],
    name: str,
) -> None:
    """Generate one physical tutorial case at every configured tolerance."""
    simulator = _simulator(parameter_space, name)
    generator = SimulationDatasetGenerator(simulator, _dataset_path())
    records = generator.generate_split("test", 1, generator.seed + 2)
    generator.save_split("test", records, overwrite=True)


def generate_tolerance_datasets() -> None:
    """Generate paired target-grid and coarse-grid tolerance test files."""
    _generate_tolerance_test(
        q2d_tolerance_parameter_space,
        "q2d_cmr_tolerance_14_20",
    )
    _generate_tolerance_test(
        q2d_tolerance_interp_parameter_space,
        "q2d_cmr_tolerance_6_14",
    )


def generate_all_datasets(n_cases: int = 80) -> None:
    """Generate the canonical, resolution-sweep, and tolerance datasets."""
    generate_dataset(n_cases=n_cases)
    generate_multiresolution_datasets()
    generate_tolerance_datasets()


if __name__ == "__main__":
    generate_all_datasets()
