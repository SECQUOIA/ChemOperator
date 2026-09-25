# ChemOperator

ChemOperator is a research library for generating chemical-reactor simulation
datasets and preparing them for neural-operator models. It provides a common
record format for reactor simulations, reproducible parameter sampling, lazy
HDF5 datasets, tensor preprocessing, and adapters for neural-operator workflows.

The repository includes Cantera-based CSTR and plug-flow reactor models,
a heterogeneous 1D packed-bed model, analytical steady and transient pipe-flow
models, a wrapper around an external quasi-2D catalytic membrane reactor
solver, and more on the way.

> [!IMPORTANT]
> ChemOperator is currently a research codebase, not a stable public release.
> APIs and the HDF5 schema may change. The Python import package is
> `chem_operator`.

## Features

- A shared `CaseSimulator` interface and `SimulationRecord` output schema
- Reproducible random, categorical, constant, and Cartesian-grid sampling
- Train/validation/test generation with one HDF5 file per split
- Lazy, worker-safe PyTorch dataset access with temporal or spatial windowing
- Field packing, state/delta targets, and reversible normalization
- DeepXDE, NeuralOperator, FNO, autoencoder, and POD utilities
- Headless direct and POD-DeepONet comparison and artifact helpers
- PhysicsNeMo PDE definitions for steady and transient Hagen–Poiseuille flow
- Packaged Cantera example mechanisms

## Supported simulators

| System | Simulator | Coordinates | Main fields | Solver |
| --- | --- | --- | --- | --- |
| Continuous stirred-tank reactor | `CSTRSim` | `t` | `T`, `P`, `X` | Cantera reactor network |
| Non-isothermal CSTR | `NonIsothermalCSTRSim` | `t` | `T`, `P`, `X` | Cantera with optional wall heat transfer |
| Lagrangian plug-flow reactor | `PFRLagrangianParticleSim` | `t` | `z`, `T`, `P`, `X`, `velocity` | Cantera constant-pressure reactor |
| Chain-of-reactors PFR | `PFRChainOfReactorsSim` | `z` | `t`, `T`, `P`, `X`, `velocity`, `residence_time` | Cantera steady reactor chain |
| Non-isothermal reactor-chain PFR | `PFRNonIsothermalChainOfReactorsSim` | `z` | Same as reactor-chain PFR | Cantera with optional wall heat transfer |
| Heterogeneous packed bed | `PackedBed1DSim` | `z` | `rhou`, `P`, `T`, `Y`, `X`, `Z`, `velocity` | IDA DAE solver and Cantera surface kinetics |
| Steady circular-pipe flow | `HagenPoiseuillePipeFlowSim` | `r` | `velocity` | Analytical |
| Startup circular-pipe flow | `TransientHagenPoiseuillePipeFlowSim` | `t`, `r` | `velocity`, `flow_rate` | Analytical Fourier–Bessel series |
| Quasi-2D catalytic membrane reactor | `CMRSim` | `z`, `r` | Thermochemical and flow fields | External Q2D executable or bundled tutorial output |

## Installation

### Requirements

- Python 3.12 or newer
- [uv](https://docs.astral.sh/uv/) for the locked development environment
- A platform supported by the scientific and NVIDIA packages in
  [`pyproject.toml`](pyproject.toml)
- System SUNDIALS/IDA support may be needed when building `scikits-odes`

Full install from fresh WSL:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential gfortran cmake pkg-config git curl python3-dev libsundials-dev libopenblas-dev liblapack-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
git clone https://github.com/SECQUOIA/ChemOperator.git
cd ChemOperator
uv sync
```

Run commands inside the environment with `uv run`:

```bash
uv run python -c "import chem_operator; print(chem_operator.__file__)"
uv run pytest -v
```

The project uses a `src/` package layout. `uv sync` installs the working tree
in editable form, so imports resolve to `src/chem_operator`.

### Cantera mechanisms

Importing `chem_operator` registers the packaged mechanism directory with
Cantera. Repository simulators can therefore use mechanism names such as
`ammonia-Ru-Ba-YSZ-CSM-2019.yaml` without an absolute path:

```python
import cantera as ct
import chem_operator  # Registers ChemOperator's packaged Cantera data.

gas = ct.Solution("ammonia-CO-H2-Alzueta-2023.yaml")
```

## Quick start

This example generates a small analytical pipe-flow dataset, saves it as HDF5,
and reads one complete radial profile.

```python
from pathlib import Path

import torch

from chem_operator.datasets import (
    ChemOperatorDataset,
    SimulationDatasetGenerator,
)
from chem_operator.reactors.pipe_flow.dataset_generator import (
    HagenPoiseuillePipeFlowSim,
)
from chem_operator.sampling import Constant, Uniform

output_dir = Path("datasets/quickstart")

simulator = HagenPoiseuillePipeFlowSim(
    parameter_space={
        "radius": Uniform(0.8e-3, 1.2e-3),
        "length": Uniform(0.8, 1.2),
        "dynamic_viscosity": Uniform(0.9e-3, 1.1e-3),
        "pressure_drop": Uniform(40.0, 80.0),
        "density": Constant(1000.0),
        "n_radial_points": Constant(64),
    }
)

generator = SimulationDatasetGenerator(simulator, output_dir, seed=7)
splits = generator.generate_splits(n_cases=10)
generator.save_splits(splits)

dataset = ChemOperatorDataset(
    output_dir / "hagen_poiseuille_pipe_flow_train.h5",
    task="field_map",
    coordinate_name="r",
    input_fields=("velocity",),
    output_fields=("velocity",),
    constant_inputs=(
        "radius",
        "length",
        "dynamic_viscosity",
        "density",
        "pressure_drop",
        "pressure_gradient",
    ),
    dtype=torch.float32,
)

try:
    sample = dataset[0]
    print(sample["input_fields"]["velocity"].shape)   # torch.Size([1])
    print(sample["output_fields"]["velocity"].shape)  # torch.Size([63])
    print(sample["output_coordinates"]["r"].shape)    # torch.Size([63])
    print(sample["metadata"]["reynolds_number"])
finally:
    dataset.close()
```

`save_splits` refuses to replace existing files by default. Pass
`overwrite=True` when replacement is intentional.

## Data pipeline

The main workflow separates simulation, storage, data selection, and
model-specific transformations:

```text
parameter specs
      │
      ▼
CaseSimulator.make_case(params) ──► CaseParameters
      │
      ▼
CaseSimulator.run_case(case)   ──► SimulationRecord
      │
      ▼
SimulationDatasetGenerator     ──► <simulator>_{train,valid,test}.h5
      │
      ▼
ChemOperatorDataset            ──► raw dictionaries of PyTorch tensors
      │
      ▼
DataProcessor / model adapter  ──► model-ready x, y, constants, coordinates
```

### Parameter sampling

Parameter spaces map parameter names to specifications from
`chem_operator.sampling`:

| Specification | Behavior |
| --- | --- |
| `Constant(value)` | Returns the same value for every base case |
| `Uniform(low, high)` | Samples uniformly on the linear interval |
| `LogUniform(low, high)` | Samples uniformly in log space |
| `Normal(mean, std, clip=None)` | Samples a normal distribution with optional clipping |
| `Choice(values)` | Samples one discrete value |
| `CallableSample(fn)` | Calls `fn(rng)` for custom sampling |
| `Grid(values)` | Expands every base case over every Cartesian grid combination |

Random parameters use NumPy generators seeded independently for the three
splits (`seed`, `seed + 1`, and `seed + 2`). Grid parameters do not consume a
random draw. If a parameter space contains multiple `Grid` values, every
sampled base case produces their Cartesian product.

`generate_splits` defaults to fractions `0.8 / 0.1 / 0.1`. Fractions must sum
to one; integer rounding is assigned to train and validation first, with the
remainder placed in test.

### Simulation records

Every simulator returns:

```python
SimulationRecord(
    coordinates={"t": ...},
    fields={"T": ..., "P": ..., "X": ...},
    constants={"residence_time": ...},
    metadata={"mechanism": ..., "units": ...},
)
```

- `coordinates` contains independent axes such as time, axial distance, or
  radius.
- `fields` contains arrays defined on one or more coordinate axes.
- `constants` contains scalar or array-valued values fixed for the record.
  Nested dictionaries are supported.
- `metadata` contains JSON-serializable provenance and descriptive data.

For time-dependent Cantera records with `T`, `P`, and either `X` or `Y`,
`SimulationRecord.to_SolutionArray()` reconstructs a Cantera
`SolutionArray`.

### HDF5 layout

Each split uses schema `simulation-record-split-v0`:

```text
<simulator>_<split>.h5
├── field_names
├── constant_names
└── cases
    └── 000000
        ├── coordinates
        ├── fields
        ├── constants
        └── metadata (JSON attribute)
```

The generator adds the simulator name, split, seed, sampled parameters, and
case/grid indices to each record's metadata. Failed simulation cases are
reported and skipped.

## Loading datasets

`ChemOperatorDataset` scans the file once to build a sample index, then opens
the HDF5 handle lazily. Its state drops the open handle during serialization,
so each PyTorch `DataLoader` worker opens its own handle.

Every item has this structure:

```python
{
    "input_fields": {"T": tensor, "X": tensor},
    "output_fields": {"T": tensor, "X": tensor},
    "constant_inputs": {"residence_time": tensor},
    "input_coordinates": {"t": tensor},
    "output_coordinates": {"t": tensor},
    "metadata": {...},
}
```

The `task` controls how a record becomes samples:

| Task | Selection |
| --- | --- |
| `next_step` | Sliding input/output windows; typically one output step |
| `rollout` | Sliding windows intended for multi-step targets |
| `operator_pointwise` | First input window paired separately with each later output window |
| `operator_cartesian` | One sample per case: first input window to all later points |
| `steady_map` | First input window to the final requested output window |
| `field_map` | One sample per case: first grid point to the rest of a steady field |

Useful windowing options are:

- `n_steps_input` and `n_steps_output` for input and fixed output lengths
- `index_stride` for subsampling the selected coordinate
- `prediction_horizon` to choose the first output by coordinate distance
  instead of index distance
- `full_trajectory_mode=True` to return the remaining trajectory
- `dtype` to cast all numeric tensors

For `operator_cartesian` and `field_map`, the reader returns the complete
remaining trajectory regardless of `n_steps_output`. Coordinates unrelated
to `coordinate_name` are returned whole instead of being sliced. This is what
preserves the radial grid in a transient `(t, r)` velocity field.

Close datasets explicitly after use, especially before replacing an HDF5
file:

```python
dataset.close()
```

The reader also exposes lightweight discovery metadata without loading field
arrays. `inspect_fields()`, `inspect_constants()`, and
`inspect_coordinates()` return JSON-safe descriptors. Field, species, and
channel references can be resolved with `resolve_field()`,
`resolve_species()`, `resolve_channel()`, and
`resolve_channel_reference()`. Use `manifest()` or `fingerprint()` to record
the dataset view used by a checkpoint.

## Preprocessing

`DataProcessor` converts raw dictionaries into packed tensors while preserving
coordinates and metadata:

```python
import torch

from chem_operator.dataset_processing import (
    DataProcessor,
    FieldPacker,
    NormalizationConfig,
    ProcessedDataset,
    TargetTransformConfig,
)
from chem_operator.datasets import ChemOperatorDataset
from chem_operator.models import fit_zscore_normalizer

fields = ("velocity",)
constants = (
    "radius",
    "length",
    "dynamic_viscosity",
    "pressure_drop",
)

raw = ChemOperatorDataset(
    "datasets/pipe_flow_transient/"
    "transient_hagen_poiseuille_pipe_flow_train.h5",
    task="operator_cartesian",
    coordinate_name="t",
    input_fields=fields,
    output_fields=fields,
    constant_inputs=constants,
    dtype=torch.float32,
)

normalizer = fit_zscore_normalizer(raw, fields, constants)
processor = DataProcessor(
    field_packer=FieldPacker(
        channel_axis="last",
        variable_field_order=fields,
        constant_field_order=constants,
    ),
    normalizer=normalizer,
    normalization_config=NormalizationConfig(enabled=True),
    target_transform=TargetTransformConfig(mode="state"),
)
processed = ProcessedDataset(raw, processor)

item = processed[0]
print(item["x"].shape, item["y"].shape, item["constants"].shape)
processed.close()
```

### Packing and targets

`FieldPacker` flattens each field's feature dimensions into channels and
concatenates fields in the configured order. It supports channel-last and
channel-first tensors and can unpack predictions using its recorded layout.

`TargetTransformConfig` supports:

- `mode="state"` for absolute future states (`"absolute"` and `"identity"`
  are equivalent aliases)
- `mode="delta", multi_step_delta="direct"` for every future state minus the
  final input state
- `mode="delta", multi_step_delta="incremental"` for sequential increments
  (`"sequential"` is an alias)

Use `processor.inverse_reconstruct(prediction, model_input)` to denormalize a
prediction and, for delta targets, add it back to the appropriate reference
state.

### Normalization

The normalization module provides:

- `IdentityNormalizer`
- `ZScoreNormalizer`
- `RMSNormalizer`
- `MinMaxNormalizer`

Each supports field-wise and already-packed tensors, plus separate statistics
for state deltas. `fit_zscore_normalizer` streams trajectories rather than
loading an entire HDF5 split into memory. Fit statistics on the training split
only, then reuse them for validation, testing, and inference.

Every normalizer provides a versioned, JSON-safe `state_dict()`. Restore it
with `normalizer_from_state_dict()`; the same representation is also safe to
store with `torch.save` and load with `weights_only=True`.

## Model adapters

Adapters in `chem_operator.models` keep framework-specific shapes separate
from HDF5 ingestion:

| Adapter or utility | Purpose |
| --- | --- |
| `DeepXDEAdapter` | Builds DeepXDE pointwise or Cartesian-product operator arrays |
| `NeuralOperatorAdapter` | Applies channel conventions and optionally broadcasts constants |
| `FNOAdapter` | Builds complete channel-first 2D grids for NeuralOperator FNOs |
| `AutoencoderAdapter` | Concatenates complete trajectories into reconstruction pairs |

`FNOAdapter` channels can come from:

- sampled parameters in `sample["metadata"]["params"]`
- stored constants
- complete scalar fields
- one named species from a grouped field such as `X` or `Y`

For example:

```python
from chem_operator.models import FNOChannel

input_channels = (
    FNOChannel("T0", "parameter", "T0", unit="K"),
    FNOChannel("pressure_drop", "constant", "pressure_drop", unit="Pa"),
)
output_channels = (
    FNOChannel("velocity", "field", "velocity", unit="m/s"),
    FNOChannel("X_H2", "species", "X", species="H2"),
)
```

Use `fit_fno_zscore_normalizer` for this configurable interface. It computes
one scalar mean and standard deviation per channel, so the same statistics
broadcast to a different grid resolution. Only configure a solution field as
an input when it is available at inference time; otherwise it leaks target
information.

## Included experiments

The scripts are research experiments rather than a unified command-line
interface. Direct and POD-DeepONets have separate entrypoints and write
canonical run directories; pipe-flow's physics-informed study uses Hydra.

| Script | Workflow |
| --- | --- |
| `scripts/{cstr,packed_bed_1d,pfr,pipe_flow}/deeponet.py` | Tune and train one direct DeepONet |
| `scripts/{cstr,packed_bed_1d,pfr,pipe_flow}/pod_deeponet.py` | Tune and train one POD-DeepONet |
| `scripts/{cstr,packed_bed_1d,pfr,pipe_flow}/plot.py` | Compare explicit canonical direct and POD run directories |
| [`scripts/pipe_flow/physics_deeponet.py`](scripts/pipe_flow/physics_deeponet.py) | Hydra-configured pipe-flow data/physics-loss study |
| [`scripts/pipe_flow_transient/transient_fno.py`](scripts/pipe_flow_transient/transient_fno.py) | Neuraloperator FNO tuning, training, evaluation, checkpointing, and plots |
| [`scripts/pipe_flow_transient/transient_nemo_fno.py`](scripts/pipe_flow_transient/transient_nemo_fno.py) | PhysicsNeMo transient pipe-flow FNO workflow |
| [`scripts/pfr_heat/fno.py`](scripts/pfr_heat/fno.py) | Coupled PhysicsNeMo reactor/wall FNO workflow |
| [`scripts/pfr_heat/deeponet_fno.py`](scripts/pfr_heat/deeponet_fno.py) | Coupled PhysicsNeMo-MLP reactor DeepONet and wall FNO workflow |
| [`scripts/q2d/generate_dataset.py`](scripts/q2d/generate_dataset.py) | Quasi-2D CMR dataset generation and disabled mesh-resolution sweep |
| [`scripts/q2d/validate_solver.py`](scripts/q2d/validate_solver.py) | Bundled-reference and Docker-backed Q2D validation plots |
| [`scripts/q2d/fno.py`](scripts/q2d/fno.py) | Quasi-2D FNO training, superresolution, and break-even analysis |
| [`scripts/diagnostics/processing.py`](scripts/diagnostics/processing.py) | Visual smoke test for preprocessing and inverse reconstruction |

Examples:

```bash
uv run scripts/cstr/deeponet.py
uv run scripts/cstr/pod_deeponet.py
uv run scripts/cstr/plot.py \
  --deeponet-run artifacts/runs/cstr_non_isothermal/deeponet/<run-id> \
  --pod-deeponet-run artifacts/runs/cstr_non_isothermal/pod_deeponet/<run-id>
uv run scripts/pipe_flow_transient/transient_fno.py
uv run scripts/pipe_flow/physics_deeponet.py final.epochs=5
uv run scripts/pfr_heat/deeponet_fno.py
uv run scripts/q2d/validate_solver.py
```

The physics-informed FNO entry points accept
`--variant {data,physics,both}` and default to `physics`. This applies to
`pipe_flow_transient/transient_nemo_fno.py`, `pfr_heat/fno.py`, and
`pfr_heat/deeponet_fno.py`. The `physics` variant combines supervised data
loss with PDE and constraint losses; it is not physics-only training. The
`both` option runs independent data-only and physics-informed searches, writing
artifacts under model IDs suffixed with `_data` and `_physics`, respectively.

The DeepONet trainers do not import plotting code. Each model writes a
versioned canonical run containing its manifest, best configuration, history,
metrics, checkpoint, tuning trials, and bounded reconstructions. Run the
corresponding `plot.py` with the two run paths afterward to generate loss and
reconstruction figures.

Review each script's data paths, run-mode flags, trajectory limits, and compute
settings before launching it. The tuning scripts can be long-running and use a
GPU when PyTorch reports one as available. Examples are grouped by physical
case, with configuration beside their runners. Training runners write below
`artifacts/runs/<problem>/<model>/<run-id>` by default. Project-level Ray state
is written below `.ray/`.

The experiment tuning layer defaults `RAY_ENABLE_UV_RUN_RUNTIME_ENV=0` before
Ray is imported, so local workers reuse the installed environment under
`uv run` instead of copying the project and reinstalling dependencies into
Ray's temporary directory. Set this variable to `1` before launching if you
need Ray's automatic uv environment replication for a distributed deployment.

## Quasi-2D packed-bed solver

`CMRSim` is a file-based wrapper for the separately maintained solver vendored
under [`Quasi-2D-packbed-experiement/`](Quasi-2D-packbed-experiement). Without
a solver command, its default tutorial cases parse the bundled reference CSV
files. New parameterized cases require a compiled executable or the provided
Docker image.

Build the image from the repository root:

```bash
docker build \
  -f Quasi-2D-packbed-experiement/docker/quasi2d-packbed/Dockerfile \
  -t chem-operator-q2d \
  Quasi-2D-packbed-experiement
```

Then configure the simulator:

```python
from chem_operator.reactors.q2d.dataset_generator import (
    CMRSim,
    default_docker_solver_command,
)

simulator = CMRSim(
    solver_command=default_docker_solver_command(),
    use_reference_if_no_solver=False,
    keep_case_dirs=True,
)
```

Generate the base Q2D dataset with the same simulator configuration used by
the FNO experiment:

```bash
uv run python scripts/q2d/generate_dataset.py
```

The mesh-resolution sweep in that script is retained behind an `if False`
guard. The base generator and the sweep require the Docker image. Bundled
tutorial outputs can be checked without Docker; pass `--radial-grid` to run
the Docker-backed validation cases instead:

```bash
uv run python scripts/q2d/validate_solver.py
uv run python scripts/q2d/validate_solver.py --radial-grid --lumen-points 3
```

Alternatively, set `CHEM_OPERATOR_Q2D_SOLVER_COMMAND`. The command may contain
`{case_dir}`, which is replaced with the absolute generated case directory.
See the [vendored solver README](Quasi-2D-packbed-experiement/external/Quasi-2D-packbed/README.md)
for its compilation, citation, and license information.

## Creating a simulator

Custom simulators implement the `CaseSimulator` protocol:

```python
from collections.abc import Mapping
from typing import Any

from chem_operator.datasets import CaseParameters, SimulationRecord
from chem_operator.sampling import ParameterSpec


class MySimulator:
    name = "my_simulator"

    def __init__(self, parameter_space):
        self._parameter_space = dict(parameter_space)

    @property
    def parameter_space(self) -> Mapping[str, ParameterSpec]:
        return self._parameter_space

    def make_case(self, params: Mapping[str, Any]) -> CaseParameters:
        return CaseParameters(
            initial_conditions={"state": params["initial_state"]},
            solver_parameters={"n_steps": params["n_steps"]},
        )

    def run_case(self, case: CaseParameters) -> SimulationRecord:
        # Run the physical or numerical model here.
        return SimulationRecord(
            coordinates={"t": ...},
            fields={"state": ...},
            constants={"control": ...},
            metadata={"solver": "my solver", "units": {"t": "s"}},
        )
```

Choose fields so that arrays varying along the selected coordinate use that
coordinate as their first dimension. Record constant values in `constants`,
not by repeating them along a field axis.

## Repository layout

```text
.
├── src/chem_operator/
│   ├── datasets.py              # Public dataset compatibility facade
│   ├── dataset_processing.py    # Public preprocessing compatibility facade
│   ├── normalization.py         # Public normalization compatibility facade
│   ├── models.py                # Public model compatibility facade
│   ├── _datasets/               # Dataset records, generation, and HDF5 reader
│   ├── _dataset_processing/     # Packing and reversible target transforms
│   ├── _normalization/          # Tensor normalizer implementations
│   ├── _models/                 # Adapters, POD, and training implementations
│   ├── sampling.py              # Parameter specifications
│   ├── example_data/            # Packaged Cantera mechanisms
│   └── reactors/                # Built-in physical systems
├── tests/                       # Unit and integration tests
├── scripts/                     # Dataset/model experiments
├── nemo-examples/               # PhysicsNeMo prototypes
├── datasets/                    # Local generated datasets
├── artifacts/                   # Benchmark artifacts
└── Quasi-2D-packbed-experiement/ # External Q2D solver integration
```

Generated datasets, checkpoints, plots, and benchmark artifacts can be large.
Treat the checked-in examples as research outputs rather than package data
required to import `chem_operator`.

## Development

Run the test suite:

```bash
uv run pytest -v
pytest --cov=chem_operator
```

Run one module or one test:

```bash
uv run pytest tests/test_pipe_flow.py -v
uv run pytest tests/test_fno_adapter.py::test_normalizer_broadcasts_on_a_different_resolution -v
```

Run static analysis:

```bash
uv run pylint src/chem_operator tests
```

Generate package class diagrams (requires the Graphviz executable in addition
to the Python package):

```bash
uv run pyreverse -o png -p ChemOperator src/chem_operator
```

Tests cover the analytical solutions, HDF5 round trips, Cantera tutorial
comparisons, non-isothermal reactor behavior, PhysicsNeMo residuals, FNO
normalization across resolutions, and a CPU FNO training smoke test.


## Current limitations and roadmap

- Dataset generation currently holds all records for a split in memory before
  writing them.
- The HDF5 schema is versioned as `v0` and has no migration layer.
- The experiment scripts do not yet share one CLI or configuration system.
- FNO physics-informed loss and superresolution comparisons remain active
  research areas.
- Accuracy-constrained break-even analysis still needs matched tutorial cases
  solved at multiple resolutions.

Q2D-specific work is tracked in
[`scripts/TODO_q2d_fno.md`](scripts/TODO_q2d_fno.md).
