import cantera as ct
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.cstr.dataset_generator import CSTRSim, NonIsothermalCSTRSim
from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.sampling import Constant, Uniform

cstr_simulator = CSTRSim(
    parameter_space={
        # sampled physical parameters
        "phi": Uniform(0.5, 2.0),
        "reactive_fraction": Uniform(0.02, 0.08),
        # constants
        "T0": Constant(925.0), # Kelvin
        "P0": Constant(1.046138 * ct.one_atm), # atm -> Pa. equals 1.06 bars
        "reactor_volume": Constant(30.5 * (1e-2) ** 3), # m^3
        "residence_time": Constant(2.0), # s
        "pressure_controller_K": Constant(1e-6),
        # solver controls
        "t_final": Constant(50.0), # s
        "dt": Constant(2e-2), # Not used when adaptive = True
        "adaptive": Constant(False),
    }
)

cstr_non_isothermal_simulator = NonIsothermalCSTRSim(
    parameter_space={
        # sampled physical parameters
        "phi": Uniform(0.5, 2.0),
        "reactive_fraction": Uniform(0.02, 0.08),
        # constants
        "T0": Constant(925.0),
        "P0": Constant(1.046138 * ct.one_atm),
        "reactor_volume": Constant(30.5 * (1e-2) ** 3),
        "residence_time": Constant(2.0),
        "pressure_controller_K": Constant(1e-6),
        # thermal controls
        "solve_energy": Constant(True), # controls isothermal or non-isothermal
        "ambient_temperature": Constant(600.0),
        "wall_area": Constant(1.0e-2),
        "heat_transfer_coefficient": Uniform(0.001, 2.0),
        # solver controls
        "t_final": Constant(50.0),
        "dt": Constant(2e-2),
        "adaptive": Constant(False),
    }
)

if __name__ == "__main__":
    paths = ExamplePaths.from_script(__file__)
    dataset_path = paths.datasets / "cstr"

    cstr_dataset_generator = SimulationDatasetGenerator(
        cstr_simulator, dataset_path
    )
    records_splits = cstr_dataset_generator.generate_splits(n_cases=50)
    cstr_dataset_generator.save_splits(records_splits, overwrite=True)

    cstr_non_isothermal_dataset_generator = SimulationDatasetGenerator(
        cstr_non_isothermal_simulator,
        dataset_path,
    )
    records_splits = cstr_non_isothermal_dataset_generator.generate_splits(n_cases=50)
    cstr_non_isothermal_dataset_generator.save_splits(
        records_splits,
        overwrite=False,
    )
