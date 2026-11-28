"""Generate coupled three-species PFR/cylindrical-wall datasets."""

from chem_operator.datasets import SimulationDatasetGenerator
from experiments.paths import example_paths
from experiments.pfr_heat.dataset_generator import PFRHeatSim
from chem_operator.sampling import Constant, Uniform


pfr_heat_simulator = PFRHeatSim(
    parameter_space={
        "inlet_flow_a": Uniform(30.0, 80.0),
        "inlet_concentration_a": Uniform(0.05, 0.10),
        "inlet_temperature": Uniform(375.0, 425.0),
        "outer_temperature": Uniform(300.0, 325.0),
        "volumetric_heat_transfer": Constant(4000.0),
        "wall_aspect_ratio_sq": Constant(0.20),
        "interface_biot": Constant(1.5),
        "inner_radius": Constant(1.0),
        "outer_radius": Constant(1.25),
        "n_axial_points": Constant(101),
        "n_radial_points": Constant(21),
        "tolerance": Constant(1.0e-4),
        "max_nodes": Constant(10000),
    }
)

def generate_dataset(n_cases: int = 1000) -> None:
    """Generate and overwrite all dataset splits."""
    paths = example_paths(__file__)
    generator = SimulationDatasetGenerator(
        pfr_heat_simulator,
        paths.datasets / "pfr_heat",
    )
    splits = generator.generate_splits(n_cases=n_cases)
    generator.save_splits(splits, overwrite=True)


if __name__ == "__main__":
    generate_dataset()
