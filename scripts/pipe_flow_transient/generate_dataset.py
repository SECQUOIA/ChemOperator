from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.pipe_flow_transient.dataset_generator import TransientHagenPoiseuillePipeFlowSim
from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.sampling import Constant, Uniform

pipe_flow_transient_simulator = TransientHagenPoiseuillePipeFlowSim(
    parameter_space={
        "radius": Uniform(0.5e-3, 1.5e-3),
        "length": Uniform(0.5, 2.0),
        "dynamic_viscosity": Uniform(0.8e-3, 1.2e-3),
        "pressure_drop": Uniform(10.0, 100.0),
        "density": Constant(1000.0),
        "n_time_points": Constant(128),
        "n_radial_points": Constant(128),
        "max_fourier_number": Constant(2.0),
    },
)

if __name__ == "__main__":
    paths = ExamplePaths.from_script(__file__)
    dataset_path = paths.datasets / "pipe_flow_transient"

    dataset_generator = SimulationDatasetGenerator(
        pipe_flow_transient_simulator, dataset_path
    )
    record_splits = dataset_generator.generate_splits(n_cases=10000)
    dataset_generator.save_splits(record_splits, overwrite=True)
