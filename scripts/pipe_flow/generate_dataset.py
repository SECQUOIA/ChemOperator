from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.pipe_flow.dataset_generator import HagenPoiseuillePipeFlowSim
from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.sampling import Constant, Uniform

pipe_flow_simulator = HagenPoiseuillePipeFlowSim(
    parameter_space={
        "radius": Uniform(1e-4, 5.0e-3),
        "length": Uniform(1.0, 5.0),
        "dynamic_viscosity": Uniform(0.8e-3, 1.2e-3),
        "pressure_drop": Uniform(1.0, 100.0),
        "density": Constant(1000.0),
        "n_radial_points": Constant(128),
    }
)

if __name__ == "__main__":
    paths = ExamplePaths.from_script(__file__)
    dataset_path = paths.datasets / "pipe_flow"

    dataset_generator = SimulationDatasetGenerator(
        pipe_flow_simulator, dataset_path
    )
    record_splits = dataset_generator.generate_splits(n_cases=2000)
    dataset_generator.save_splits(record_splits, overwrite=True)
