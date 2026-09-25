from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.packed_bed_1D.dataset_generator import PackedBed1DSim
from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.sampling import Constant, Uniform

# This is run when imported so this can be accessed for simulations during testing
packed_bed_simulator = PackedBed1DSim(
    parameter_space={
        # sampled inlet and wall parameters, centered near the tutorial case
        "T0": Uniform(660.0, 690.0),
        "P0": Uniform(4.75e5, 5.25e5),
        "inlet_velocity": Uniform(8e-4, 1.2e-3),
        "wall_temperature": Uniform(705.0, 740.0),
        "inlet_nh3_mole_fraction": Uniform(0.975, 0.995),
        # constants from the tutorial
        "diluent_species": Constant("AR"),
        "length": Constant(5e-2),
        "radius": Constant(5e-3),
        "porosity": Constant(0.5),
        "tortuosity": Constant(2.0),
        "particle_diameter": Constant(3.37e-4),
        "specific_surface_area": Constant(3.5e6),
        "heat_transfer_coefficient": Constant(1e2),
        "solve_energy": Constant(True),
        "membrane_present": Constant(True),
        "membrane_permeability": Constant(1e-15),
        "membrane_thickness": Constant(3e-6),
        "membrane_species": Constant("H2"),
        "sweep_pressure": Constant(1e5),
        # solver controls
        "first_step_size": Constant(1e-15),
        "atol": Constant(1e-14),
        "rtol": Constant(1e-6),
        "max_steps": Constant(8000),
    }
)

if __name__ == "__main__":
    paths = ExamplePaths.from_script(__file__)
    dataset_path = paths.datasets / "packed_bed_1D"

    packed_bed_dataset_generator = SimulationDatasetGenerator(
        packed_bed_simulator, dataset_path
    )
    records_splits = packed_bed_dataset_generator.generate_splits(
        n_cases=1000,
    )
    packed_bed_dataset_generator.save_splits(records_splits, overwrite=True)
