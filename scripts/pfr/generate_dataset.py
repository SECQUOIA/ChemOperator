import cantera as ct
from chem_operator.example_paths import ExamplePaths
from chem_operator.reactors.pfr.dataset_generator import (
    PFRLagrangianParticleSim, PFRChainOfReactorsSim, PFRNonIsothermalChainOfReactorsSim
)
from chem_operator.datasets import SimulationDatasetGenerator
from chem_operator.sampling import Constant, Uniform

pfr_param_space = {
    # sampled inlet parameters
    "T0": Uniform(1200.0, 1800.0),
    "u0": Uniform(0.004, 0.008),
    "phi": Uniform(0.5, 2.0),
    "ar_o2_ratio": Uniform(0.0, 0.2),
    # constants
    "P0": Constant(ct.one_atm),
    "length": Constant(1.5e-7),
    "area": Constant(1.0e-4),
    # solver controls
    "n_steps": Constant(200),
}
pfr_lagrangian_simulator = PFRLagrangianParticleSim(pfr_param_space)

pfr_chain_param_space = pfr_param_space | {
    "pressure_controller_K": Constant(1e-12),
    "max_time_step": Constant(1e4),
}
pfr_chain_simulator = PFRChainOfReactorsSim(pfr_chain_param_space)

pfr_non_isothermal_param_space = (
    pfr_chain_param_space
    | {
        "solve_energy": Constant(True),
        "ambient_temperature": Constant(1000.0),
        "wall_area_per_volume": Constant(400.0), # m^-1
        "heat_transfer_coefficient": Uniform(0.0, 200.0), # W / m^2 / K
    }
)
pfr_simulator = PFRNonIsothermalChainOfReactorsSim(
    pfr_non_isothermal_param_space
)

if __name__ == "__main__":
    paths = ExamplePaths.from_script(__file__)
    dataset_path = paths.datasets / "pfr"

    pfr_lagrangian_dataset_generator = SimulationDatasetGenerator(
        pfr_lagrangian_simulator, dataset_path,
    )
    records_splits = pfr_lagrangian_dataset_generator.generate_splits(n_cases=1000)
    pfr_lagrangian_dataset_generator.save_splits(records_splits, overwrite=True)

    pfr_chain_dataset_generator = SimulationDatasetGenerator(
        pfr_chain_simulator, dataset_path,
    )
    records_splits = pfr_chain_dataset_generator.generate_splits(n_cases=300)
    pfr_chain_dataset_generator.save_splits(records_splits, overwrite=True)

    pfr_dataset_generator = SimulationDatasetGenerator(
        pfr_simulator, dataset_path,
    )
    records_splits = pfr_dataset_generator.generate_splits(n_cases=300)
    pfr_dataset_generator.save_splits(records_splits, overwrite=True)
