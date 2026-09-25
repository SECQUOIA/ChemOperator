# Model usage

Run commands from the repository root. The examples below use `uv` so that
the project environment and dependencies are selected automatically:

```bash
uv run python scripts/<folder>/<script>.py [options]
```

Pass `--help` to a model or plotting entry point to see its effective CLI.
Generated datasets are written below `datasets/`; canonical model runs are
written below `artifacts/runs/` unless `--runs-root` is supplied.

## Common model options

The direct and POD DeepONet scripts accept these options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--tune`, `--no-tune` | tune | Enable or skip hyperparameter tuning. |
| `--train`, `--no-train` | train | Enable or skip final training and evaluation. |
| `--runs-root PATH` | `artifacts/runs` | Select the parent directory for run artifacts. |
| `--run-id ID` | UTC timestamp | Select the final run-directory component. |
| `--device DEVICE` | `auto` | Select `auto`, `cpu`, `cuda`, or another PyTorch device. |
| `--plot-cases N` | `2` | Number of test reconstructions to store. |
| `--train-config CONFIG` | `best` | Select the final configuration; these scripts currently require `best`. |

They also expose `--generate` and `--plot` compatibility switches, but reject
them when enabled. Use each folder's `generate_dataset.py` and `plot.py`
instead.

The FNO and physics workflow scripts additionally accept:

| Option | Purpose |
| --- | --- |
| `--data-dir PATH` | Override the dataset directory. |
| `--epochs N` | Set final-training epochs. |
| `--tune-epochs N` | Set epochs per tuning trial. |
| `--samples N` | Set the number of hyperparameter trials. |
| `--max-cases N` | Limit the number of cases per split for a smoke test. |

At least one of tuning or training must remain enabled. When training with
`--no-tune`, use the same `--run-id` and `--runs-root` as a completed tuning
run so the workflow can read its saved best configuration.

For example:

```bash
# Tune and train.
uv run python scripts/pfr/deeponet.py --run-id pfr_direct_01

# Tune only, then train that run in a later invocation.
uv run python scripts/pfr/deeponet.py --run-id pfr_direct_02 --no-train
uv run python scripts/pfr/deeponet.py --run-id pfr_direct_02 --no-tune

# Small FNO smoke run.
uv run python scripts/pipe_flow_transient/transient_fno.py \
  --device cpu --max-cases 8 --samples 1 --tune-epochs 1 --epochs 1 \
  --plot-cases 1
```

## `scripts/cstr`

Generate the isothermal and non-isothermal CSTR splits. This generates 80
cases for each simulator and overwrites existing splits:

```bash
uv run python scripts/cstr/generate_dataset.py
```

Tune and train the direct DeepONet for non-isothermal CSTR temperature,
pressure, and composition trajectories:

```bash
uv run python scripts/cstr/deeponet.py
```

Fit the training-only POD basis, then tune and train POD-DeepONet:

```bash
uv run python scripts/cstr/pod_deeponet.py
```

Compare saved direct and POD runs:

```bash
uv run python scripts/cstr/plot.py \
  --deeponet-run artifacts/runs/cstr_non_isothermal/deeponet/<run-id> \
  --pod-deeponet-run artifacts/runs/cstr_non_isothermal/pod_deeponet/<run-id> \
  --output-dir scripts/cstr/results/deeponet_comparison \
  --cases 2
```

`common.py` is support code for dataset loading, normalization, experiment
metadata, tuning resources, and the shared CLI; it is not an entry point.

## `scripts/pfr`

Generate the PFR datasets. The script generates 1,000 Lagrangian cases, 300
chain-of-reactors cases, and 300 non-isothermal chain cases, overwriting the
corresponding splits:

```bash
uv run python scripts/pfr/generate_dataset.py
```

Tune and train the direct or POD-compressed chain-of-reactors model:

```bash
uv run python scripts/pfr/deeponet.py
uv run python scripts/pfr/pod_deeponet.py
```

Both predict temperature, pressure, composition, and velocity along `z`.
Compare their saved runs with:

```bash
uv run python scripts/pfr/plot.py \
  --deeponet-run artifacts/runs/pfr_chain/deeponet/<run-id> \
  --pod-deeponet-run artifacts/runs/pfr_chain/pod_deeponet/<run-id> \
  --output-dir scripts/pfr/results/deeponet_comparison \
  --cases 2
```

`common.py` contains the shared PFR data and experiment configuration.

## `scripts/packed_bed_1d`

Generate 1,000 one-dimensional ammonia packed-bed/membrane cases, overwriting
existing splits:

```bash
uv run python scripts/packed_bed_1d/generate_dataset.py
```

Tune and train the direct or POD-compressed DeepONet:

```bash
uv run python scripts/packed_bed_1d/deeponet.py
uv run python scripts/packed_bed_1d/pod_deeponet.py
```

Compare saved runs:

```bash
uv run python scripts/packed_bed_1d/plot.py \
  --deeponet-run artifacts/runs/packed_bed_1d/deeponet/<run-id> \
  --pod-deeponet-run artifacts/runs/packed_bed_1d/pod_deeponet/<run-id> \
  --output-dir scripts/packed_bed_1d/results/deeponet_comparison \
  --cases 2
```

`common.py` contains the shared packed-bed data and experiment configuration.

## `scripts/pipe_flow`

Generate 10,000 steady Hagen--Poiseuille pipe-flow cases, overwriting existing
splits:

```bash
uv run python scripts/pipe_flow/generate_dataset.py
```

Tune and train the standard direct or POD DeepONet:

```bash
uv run python scripts/pipe_flow/deeponet.py
uv run python scripts/pipe_flow/pod_deeponet.py
```

Run the PhysicsNeMo DeepONet comparison:

```bash
# Tune a data-only baseline and then its physics-informed counterpart.
uv run python scripts/pipe_flow/physics_deeponet.py --variant both

# Run only one variant.
uv run python scripts/pipe_flow/physics_deeponet.py --variant data
uv run python scripts/pipe_flow/physics_deeponet.py --variant physics

# Use another experiment configuration.
uv run python scripts/pipe_flow/physics_deeponet.py \
  --config path/to/physics_deeponet.yaml \
  --variant both --physics-samples 4
```

Its additional options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--config PATH` | `scripts/pipe_flow/physics_deeponet.yaml` | Architecture choices, search ranges, seed, and physics-weight settings. |
| `--variant {both,data,physics}` | `both` | Select the data-only baseline, physics-informed model, or sequential comparison. |
| `--physics-samples N` | YAML budget | Override physics-weight trials when running the physics variant. |

The YAML does not replace the Ray search space: the script converts its
`search` values into `tune.choice` and `tune.loguniform` distributions. With
`--variant both`, the physics run reuses the best data-only architecture and
tunes the physics-loss weight.

Plot one or more canonical physics/operator runs:

```bash
uv run python scripts/pipe_flow/plot.py \
  --run artifacts/runs/pipe_flow/physics_deeponet_data/<run-id> \
  --run artifacts/runs/pipe_flow/physics_deeponet/<run-id> \
  --output-dir scripts/pipe_flow/results/comparison --cases 3
```

Or compare the standard direct/POD pair:

```bash
uv run python scripts/pipe_flow/plot.py \
  --deeponet-run artifacts/runs/pipe_flow/deeponet/<run-id> \
  --pod-deeponet-run artifacts/runs/pipe_flow/pod_deeponet/<run-id> \
  --cases 3
```

Do not combine `--run` with the paired DeepONet options. `common.py` provides
both the standard and physics-specific data contracts.

## `scripts/pipe_flow_transient`

Generate 10,000 transient Hagen--Poiseuille cases on time/radius grids,
overwriting existing splits:

```bash
uv run python scripts/pipe_flow_transient/generate_dataset.py
```

Tune and train the data-only NeuralOperator FNO:

```bash
uv run python scripts/pipe_flow_transient/transient_fno.py
```

Tune and train the PhysicsNeMo FNO with data, PDE-residual, and constraint losses:

```bash
uv run python scripts/pipe_flow_transient/transient_nemo_fno.py
```

Both models predict the velocity field over time and radius and accept all common FNO options listed above.

Compare any number of saved runs by repeating `--run`:

```bash
uv run python scripts/pipe_flow_transient/plot.py \
  --run artifacts/runs/pipe_flow_transient/transient_fno/<run-id> \
  --run artifacts/runs/pipe_flow_transient/transient_nemo_fno/<run-id> \
  --output-dir scripts/pipe_flow_transient/results/comparison \
  --cases 2
```

`common.py` owns the shared FNO channels, normalization, PDE provenance, and
dataset adapters.

## `scripts/pfr_heat`

Generate 1,000 coupled three-species PFR/cylindrical-wall cases, overwriting
existing splits:

```bash
uv run python scripts/pfr_heat/generate_dataset.py
```

Tune and train the coupled PhysicsNeMo model:

```bash
uv run python scripts/pfr_heat/fno.py
```

The model uses a 1-D FNO for species flows and gas temperature, then feeds the
predicted wall-side gas temperature into a 2-D cylindrical-wall FNO. Its
defaults are 6 trials, 40 tuning epochs, and 75 final epochs. It accepts all
common FNO options.

Tune and train the coupled reactor-DeepONet/wall-FNO model:

```bash
uv run python scripts/pfr_heat/deeponet_fno.py
```

This variant uses PhysicsNeMo MLPs for a four-output DeepONet over the axial
reactor coordinate and feeds its predicted gas temperature into the same
PhysicsNeMo wall FNO. Both components train end to end with the data, reactor,
wall, and boundary losses.

Plot one or more saved runs:

```bash
uv run python scripts/pfr_heat/plot.py \
  --run artifacts/runs/pfr_heat/fno/<run-id> \
  --run artifacts/runs/pfr_heat/deeponet_fno/<run-id> \
  --output-dir scripts/pfr_heat/results/comparison \
  --cases 2
```

`common.py` validates the coupled grid and PDE metadata and supplies adapters,
normalizers, and experiment configuration.

## `scripts/q2d`

Generate 100 quasi-two-dimensional catalytic membrane reactor cases:

```bash
uv run python scripts/q2d/generate_dataset.py
```

This invokes the configured external/Docker solver, overwrites the base
splits, and leaves the source-code resolution sweep disabled.

Tune, train, and benchmark the axial/radial NeuralOperator FNO:

```bash
uv run python scripts/q2d/fno.py
```

It accepts the common FNO options and one additional Boolean option:

| Option | Default | Purpose |
| --- | --- | --- |
| `--benchmark`, `--no-benchmark` | benchmark | Enable or skip mesh-sweep timing, cost, break-even, and super-resolution artifacts. |

For example, train without the post-training benchmark:

```bash
uv run python scripts/q2d/fno.py --no-benchmark
```

Plot a saved run and, when present, its benchmark artifacts:

```bash
uv run python scripts/q2d/plot.py \
  --run artifacts/runs/q2d_cmr/fno/<run-id> \
  --output-dir scripts/q2d/results/fno \
  --cases 1
```

Validate the bundled tutorial/reference cases and create field plots:

```bash
uv run python scripts/q2d/validate_solver.py
```

Run Docker-backed radial-grid validation instead:

```bash
uv run python scripts/q2d/validate_solver.py \
  --radial-grid --lumen-points 4
```

`--lumen-points` defaults to `3` and is used only with `--radial-grid`.
`common.py` supplies Q2D channels, geometry handling, normalization, mesh-file
discovery, and experiment configuration. `fields.ipynb` is an interactive
field-inspection notebook, and `TODO_fno.md` contains development notes.

## `scripts/diagnostics`

Smoke-test one processed batch and target-reconstruction round trip for the
configured reactor datasets, then write `scripts/diagnostics/processing.png`:

```bash
uv run python scripts/diagnostics/processing.py
```

This script has no command-line options and does not train a model.
