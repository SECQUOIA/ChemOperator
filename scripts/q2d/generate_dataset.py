"""Generate quasi-2D catalytic membrane reactor datasets."""

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.q2d.dataset_generator import (
    CMRSim,
    default_docker_solver_command,
)
from chem_operator.sampling import Constant, Uniform


q2d_parameter_space = {
    "annular_channel": Constant(False),
    "T0": Uniform(1150.0, 1240.0),
    "sccm": Uniform(470.0, 530.0),
    "P0": Constant(1.0e6),
    "lumen_points": Constant(6),
    "mesh_points": Constant(14),
    "refine": Constant(False),
}

q2d_simulator = CMRSim(
    parameter_space=q2d_parameter_space,
    solver_command=default_docker_solver_command(),
    use_reference_if_no_solver=False,
)


def generate_dataset(n_cases: int = 100) -> None:
    """Generate and overwrite all base dataset splits."""
    paths = ExamplePaths.from_script(__file__)
    dataset_path = paths.datasets / "q2d_cmr"

    q2d_dataset_generator = SimulationDatasetGenerator(
        q2d_simulator,
        dataset_path,
    )
    records_splits = q2d_dataset_generator.generate_splits(n_cases=n_cases)
    q2d_dataset_generator.save_splits(records_splits, overwrite=True)


if __name__ == "__main__":
    generate_dataset()

    # This resolution sweep is intentionally disabled until the base dataset
    # workflow has been exercised independently.
    if False:  # pylint: disable=using-constant-test
        for n_z in range(12, 21):
            print(f"{n_z=}")
            for n_r in range(4, 11):
                test_parameter_space = q2d_parameter_space | {
                    "mesh_points": Constant(n_z),
                    "lumen_points": Constant(n_r),
                }
                test_q2d_simulator = CMRSim(
                    parameter_space=test_parameter_space,
                    solver_command=default_docker_solver_command(),
                    use_reference_if_no_solver=False,
                )
                test_q2d_simulator.name += f"_{n_z}_{n_r}"
                test_q2d_dataset_generator = SimulationDatasetGenerator(
                    test_q2d_simulator,
                    dataset_path,
                )
                records_test_split = test_q2d_dataset_generator.generate_split(
                    "test",
                    3,
                    test_q2d_dataset_generator.seed + 3,
                )
                test_q2d_dataset_generator.save_split(
                    "test",
                    records_test_split,
                    overwrite=True,
                )
