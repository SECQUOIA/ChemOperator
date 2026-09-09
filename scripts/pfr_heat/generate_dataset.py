"""Generate coupled three-species PFR/cylindrical-wall datasets."""

from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.pfr_heat.dataset_generator import PFRHeatSim
from chem_operator.sampling import Constant, Uniform


pfr_heat_simulator = PFRHeatSim(
    parameter_space={
        "inlet_flow_a": Uniform(50.0, 100.0),
        "inlet_concentration_a": Uniform(0.05, 0.10),
        "inlet_temperature": Uniform(350.0, 450.0),
        "outer_temperature": Uniform(300.0, 350.0),
        "volumetric_heat_transfer": Constant(4000.0),
        "wall_aspect_ratio_sq": Constant(0.20),
        "interface_biot": Constant(1.5),
        "inner_radius": Constant(1.0),
        "outer_radius": Constant(2.0),
        "n_axial_points": Constant(101),
        "n_radial_points": Constant(21),
        "tolerance": Constant(1.0e-4),
        "max_nodes": Constant(5000),
    }
)

if __name__ == "__main__":
    paths = ExamplePaths.from_script(__file__)
    generator = SimulationDatasetGenerator(pfr_heat_simulator, paths.datasets / "pfr_heat")
    splits = generator.generate_splits(n_cases=20)
    generator.save_splits(splits, overwrite=True)
