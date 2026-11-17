# Model Usage

Be sure to run all commands from the repository root directory.

## Environment Setup

Activate your virtual environment before executing any commands:
* **Bash / Zsh:**
```bash
source .venv/bin/activate
```

* **PowerShell:**
```PowerShell
.venv\Scripts\Activate.ps1
```

> You can use `uv run` instead of `python` to automatically handle environment selection and dependencies.

## Running Scripts

Execute your scripts using the following standard format:
```bash
python scripts/<folder>/<script>.py [options]
```

Pass `--help` to a model or plotting entry point to see its effective CLI.
Generated datasets are written below `datasets/`; canonical model runs are
written below `artifacts/runs/` unless `--runs-root` is supplied, and plots are
written below `artifacts/plots/<problem_id>/` unless `--output-dir` is supplied.

Each workflow keeps separate entry points for dataset generation, model runs,
and artifact plotting:

| Task | Command |
| --- | --- |
| Generate data | `python scripts/<folder>/generate_dataset.py` |
| Tune and train | `python scripts/<folder>/<model>.py` |
| Plot saved runs | `python scripts/<folder>/plot.py --run PATH [--run PATH ...]` |

Plotters require `--run` or `--recent`, which are mutually exclusive. Repeat
`--run` to compare compatible runs. Use `--recent` instead of `--run` to plot
the newest completed run for every model, or `bash scripts/plot_all.sh` to do
so across workflows. Every plotter shares the same output options:

| Option | Default | Purpose |
| --- | --- | --- |
| `--output-dir PATH` | `artifacts/plots/<problem_id>` | Plot directory; mirrors `artifacts/runs/<problem_id>`. |
| `--cases N` | per workflow (1 to 3) | Number of test cases to plot; must be positive. |

Within the plot directory, each run is written to `<model_id>/<run-id>/` and
compatible runs are overlaid under `comparisons/group_N/`.

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
| `--train-config CONFIG` | `best` | Hyperparameters for final training: `best` (this run's `best_config.json`), `recent` (the newest tuned run of this model), a completed run directory, or a JSON file. |
| `--seeds SEED [SEED ...]` | unset | Train the final model once per seed under `<run>/seeds/<seed>/`; requires `--train`. |
| `--tune-seed N` | problem `SEED` | Seed for the Optuna study (and for training when `--seeds` is omitted). |

The FNO and physics workflow scripts additionally accept:

| Option | Purpose |
| --- | --- |
| `--data-dir PATH` | Override the dataset directory. |
| `--epochs N` | Set final-training epochs. |
| `--tune-epochs N` | Set epochs per tuning trial. |
| `--samples N` | Set the number of hyperparameter trials. |
| `--max-cases N` | Limit the number of cases per split for a smoke test. |

At least one of tuning or training must remain enabled. When training with
`--no-tune --train-config best`, use the same `--run-id` and `--runs-root` as a
completed tuning run so the workflow can read its saved best configuration;
pass another run's directory to `--train-config` to reuse its hyperparameters
under a new `--run-id`. `--train-config recent` does the same with the newest
completed run of the same problem and model that ran a tuning study (a
non-empty `tuning_trials.csv`, not a `--max-cases` smoke run); runs trained
from a saved configuration are never chosen.

For example:

```bash
# Tune and train.
python scripts/pfr/deeponet.py --run-id pfr_direct_01

# Tune only, then train that run in a later invocation.
python scripts/pfr/deeponet.py --run-id pfr_direct_02 --no-train
python scripts/pfr/deeponet.py --run-id pfr_direct_02 --no-tune

# Small FNO smoke run.
python scripts/pipe_flow_transient/transient_fno.py \
  --device cpu --max-cases 8 --samples 1 --tune-epochs 1 --epochs 1 \
  --plot-cases 1
```

### Multi-seed runs

Without `--seeds` a run directory holds one trained model (the "flat"
layout). With `--seeds`, tuning artifacts stay in the run directory and each
seed is trained from the same best configuration into a complete child run at
`<run>/seeds/<seed>/`:

```
artifacts/runs/<problem>/<model>/<run-id>/
  manifest.json          seeds: [42, 43, 44], tuning_seed, status
  best_config.json       tuning_trials.csv   ray_results/
  seeds/42/              manifest.json history.csv metrics.csv checkpoint.pt reconstructions.npz
  seeds/43/  seeds/44/
```

A run's *members* are the root directory when it holds `metrics.csv` (every
flat run) plus every completed `seeds/*` child, so a completed flat run can be
extended with more seeds under the same `--run-id`, and completed children are
skipped when a run is resumed. Plotters and the paper tables average over
members at read time; nothing aggregated is stored.

```bash
# Tune once, then train seeds 42-44 into the same run.
python scripts/cstr/deeponet.py --run-id cstr_do_01 --seeds 42 43 44

# Add two more seeds to that run later.
python scripts/cstr/deeponet.py --run-id cstr_do_01 --no-tune --seeds 45 46

# Train five seeds from whichever run of this model was tuned most recently.
python scripts/cstr/deeponet.py --no-tune --train-config recent --seeds 1 2 3 4 5

# Reuse a specific completed run's hyperparameters under a new run id.
python scripts/cstr/deeponet.py --run-id cstr_do_02 --no-tune \
  --train-config artifacts/runs/cstr_non_isothermal/deeponet/cstr_do_01 \
  --seeds 1 2 3 4 5

# Plot the aggregate (mean ± std bands) or one seed.
python scripts/cstr/plot.py --run artifacts/runs/cstr_non_isothermal/deeponet/cstr_do_02
python scripts/cstr/plot.py --run artifacts/runs/cstr_non_isothermal/deeponet/cstr_do_02/seeds/3
```

For a multi-seed run, `training_validation_loss.png` and the comparison
`validation_history.png` show the mean curve with a ±1 standard-deviation
band over the epochs all seeds reached. Reconstruction plots are written per
seed under `<plots>/<model>/<run-id>/seeds/<seed>/`, with the first member's
copies kept at the usual top-level paths. `paper/export_model_tables.py`
reports the test error as mean ± std with a `Seeds` column.

## `scripts/cstr`

Generate the isothermal and non-isothermal CSTR splits. This generates 100
cases for each simulator and overwrites existing splits:

```bash
python scripts/cstr/generate_dataset.py
```

Tune and train the direct DeepONet for non-isothermal CSTR temperature,
pressure, and composition trajectories:

```bash
python scripts/cstr/deeponet.py
```

Tune and train the NeuralOperator FNO on the same log-time grid,
initial state, and full output state:

```bash
python scripts/cstr/fno.py
```
Use `--no-train` and `--no-tune`
with the same `--run-id` to split the two stages across invocations.

Fit the training-only POD basis, then tune and train POD-DeepONet:

```bash
python scripts/cstr/pod_deeponet.py
```

Compare saved direct DeepONet, POD-DeepONet, and FNO runs:

```bash
python scripts/cstr/plot.py \
  --run artifacts/runs/cstr_non_isothermal/deeponet/<run-id> \
  --run artifacts/runs/cstr_non_isothermal/pod_deeponet/<run-id> \
  --run artifacts/runs/cstr_non_isothermal/fno/<run-id> \
  --cases 2
```

This comparison uses saved artifacts and mean per-case validation errors;
the original training archive is not required. Raw CO histories are a separate
opt-in diagnostic:

```bash
python scripts/cstr/plot.py --co-histories \
  --dataset datasets/cstr/cstr_non_isothermal_train.h5 \
  --co-output artifacts/plots/cstr_non_isothermal/diagnostics/train_set_CO.png
```

The diagnostic may also be combined with `--run`. Text rendering uses
Matplotlib by default; add `--use-tex` only when an external LaTeX installation
is available.

`common.py` is support code for dataset loading, normalization, experiment
metadata, tuning resources, and the shared CLI; it is not an entry point.

## `scripts/pfr`

Generate the PFR datasets. The script generates 1,000 Lagrangian cases, 300
chain-of-reactors cases, and 300 non-isothermal chain cases, overwriting the
corresponding splits:

```bash
python scripts/pfr/generate_dataset.py
```

Tune and train the direct DeepONet, POD-DeepONet, or supervised FNO on the
chain-of-reactors dataset:

```bash
python scripts/pfr/deeponet.py
python scripts/pfr/pod_deeponet.py
python scripts/pfr/fno.py
```

All three predict temperature, pressure, composition, and velocity along `z`.
Compare their saved runs with:

```bash
python scripts/pfr/plot.py \
  --run artifacts/runs/pfr_chain/deeponet/<run-id> \
  --run artifacts/runs/pfr_chain/pod_deeponet/<run-id> \
  --run artifacts/runs/pfr_chain/fno/<run-id> \
  --cases 2
```

`common.py` contains the shared PFR data and experiment configuration.

## `scripts/packed_bed_1d`

Generate 1,000 one-dimensional ammonia packed-bed/membrane cases, overwriting
existing splits:

```bash
python scripts/packed_bed_1d/generate_dataset.py
```

Tune and train the direct DeepONet, POD-compressed DeepONet, or supervised FNO:

```bash
python scripts/packed_bed_1d/deeponet.py
python scripts/packed_bed_1d/pod_deeponet.py
python scripts/packed_bed_1d/fno.py
```

Compare saved runs:

```bash
python scripts/packed_bed_1d/plot.py \
  --run artifacts/runs/packed_bed_1d/deeponet/<run-id> \
  --run artifacts/runs/packed_bed_1d/pod_deeponet/<run-id> \
  --run artifacts/runs/packed_bed_1d/fno/<run-id> \
  --cases 2
```

`common.py` contains the shared packed-bed data and experiment configuration.
The FNO resamples every profile to 128 fixed physical positions from `0` to
`0.05 m`. The DeepONet adapters instead resample to 128 relative positions
through each case's recorded endpoint, which may extend slightly beyond
`0.05 m` because the solver overshoots the nominal bed length.

## `scripts/pipe_flow`

Generate 10,000 steady Hagen--Poiseuille pipe-flow cases, overwriting existing
splits:

```bash
python scripts/pipe_flow/generate_dataset.py
```

Tune and train the standard direct or POD DeepONet:

```bash
python scripts/pipe_flow/deeponet.py
python scripts/pipe_flow/pod_deeponet.py
```

Run the PhysicsNeMo DeepONet comparison:

```bash
# Tune a data-only baseline and then its physics-informed counterpart.
python scripts/pipe_flow/physics_deeponet.py --variant both

# Run only one variant.
python scripts/pipe_flow/physics_deeponet.py --variant data
python scripts/pipe_flow/physics_deeponet.py --variant physics
```

Its additional options are:

| Option | Default | Purpose |
| --- | --- | --- |
| `--variant {both,data,physics}` | `both` | Select the data-only baseline, physics-informed model, or sequential comparison. |

The search space, seed, and physics-weight range are module constants in the
script, as in `fno.py`. Both variants search the same architecture and
optimizer space; the physics variant additionally tunes the physics-loss weight.

Plot one or more canonical physics/operator runs:

```bash
python scripts/pipe_flow/plot.py \
  --run artifacts/runs/pipe_flow/physics_deeponet_data/<run-id> \
  --run artifacts/runs/pipe_flow/physics_deeponet_physics/<run-id> \
  --cases 3
```

Or compare the standard direct/POD pair:

```bash
python scripts/pipe_flow/plot.py \
  --run artifacts/runs/pipe_flow/deeponet/<run-id> \
  --run artifacts/runs/pipe_flow/pod_deeponet/<run-id> \
  --cases 3
```

`common.py` provides both the standard and physics-specific data contracts.

## `scripts/pipe_flow_transient`

Generate 10,000 transient Hagen--Poiseuille cases on time/radius grids,
overwriting existing splits:

```bash
python scripts/pipe_flow_transient/generate_dataset.py
```

Tune and train the data-only NeuralOperator FNO:

```bash
python scripts/pipe_flow_transient/transient_fno.py
```

Tune and train the PhysicsNeMo FNO with data, PDE-residual, and constraint losses:

```bash
python scripts/pipe_flow_transient/transient_nemo_fno.py
```

Both models predict the velocity field over time and radius and accept all common FNO options listed above.

Tune and train data-only DeepXDE models that predict one radial profile at a time:

```bash
python scripts/pipe_flow_transient/deeponet.py
python scripts/pipe_flow_transient/pod_deeponet.py
```

The POD basis is fitted on training next-step profiles. Both runs report
teacher-forced one-step and recursive rollout metrics and can be compared with
the existing PhysicsNeMo rollout DeepONet using `plot.py` and repeated `--run`.

Compare any number of saved runs by repeating `--run`:

```bash
python scripts/pipe_flow_transient/plot.py \
  --run artifacts/runs/pipe_flow_transient/transient_fno/<run-id> \
  --run artifacts/runs/pipe_flow_transient/transient_nemo_fno_data/<run-id> \
  --run artifacts/runs/pipe_flow_transient/transient_nemo_fno_physics/<run-id> \
  --cases 2
```

`common.py` owns the shared FNO channels, normalization, PDE provenance, and
dataset adapters.

## `scripts/pfr_heat`

Generate 1,000 coupled three-species PFR/cylindrical-wall cases, overwriting
existing splits:

```bash
python scripts/pfr_heat/generate_dataset.py
```

Tune and train the coupled PhysicsNeMo model:

```bash
python scripts/pfr_heat/fno.py
```

The model uses a 1-D FNO for species flows and gas temperature, then feeds the
predicted wall-side gas temperature into a 2-D cylindrical-wall FNO. Its
defaults are 10 trials, 50 tuning epochs, and 60 final epochs. It accepts all
common FNO options.

Tune and train the coupled reactor-DeepONet/wall-FNO model:

```bash
python scripts/pfr_heat/deeponet_fno.py
```

This variant uses PhysicsNeMo MLPs for a four-output DeepONet over the axial
reactor coordinate and feeds its predicted gas temperature into the same
PhysicsNeMo wall FNO. Both components train end to end with the data, reactor,
wall, and boundary losses.

Plot one or more saved runs:

```bash
python scripts/pfr_heat/plot.py \
  --run artifacts/runs/pfr_heat/fno_physics/<run-id> \
  --run artifacts/runs/pfr_heat/deeponet_fno_physics/<run-id> \
  --cases 2
```

`common.py` validates the coupled grid and PDE metadata and supplies adapters,
normalizers, and experiment configuration.

## `scripts/q2d`

Generate 80 canonical quasi-two-dimensional catalytic membrane reactor cases,
matched resolution-sweep test cases, and paired tolerance-sweep test cases:

```bash
python scripts/q2d/generate_dataset.py
```

This invokes the configured external/Docker solver and overwrites all three
dataset suites. The six resolution meshes combine axial sizes 14, 16, 18
with radial sizes 6, 8. Select an individual suite explicitly when needed:

```bash
python scripts/q2d/generate_dataset.py --suite base --n-cases 80
python scripts/q2d/generate_dataset.py --suite resolution
python scripts/q2d/generate_dataset.py --suite tolerance
```

`--n-cases` controls the canonical case count for `--suite all` (the default)
and `--suite base`; the resolution and tolerance suites retain their own
configured sample counts.

Tune, train, and benchmark the axial/radial NeuralOperator FNO:

```bash
python scripts/q2d/fno.py
```

It accepts the common FNO options and one additional Boolean option:

| Option | Default | Purpose |
| --- | --- | --- |
| `--benchmark`, `--no-benchmark` | benchmark | Enable or skip mesh-sweep timing, cost, break-even, and super-resolution artifacts. |

For example, train without the post-training benchmark:

```bash
python scripts/q2d/fno.py --no-benchmark
```

Tune and train fixed-mesh direct and POD-DeepONets on the same Q2D channels:

```bash
python scripts/q2d/deeponet.py
python scripts/q2d/pod_deeponet.py
```

Both use `T0` and `SCCM` to predict methane, hydrogen, and surface carbon fields.
They accept the common run, tuning, and training options, including `--data-dir`,
`--epochs`, `--tune-epochs`, `--samples`, `--max-cases`, and `--plot-cases`.
They require the same geometry and coordinate grid across selected cases and do
not expose `--benchmark`. POD is fitted only on selected training cases, requires
at least two cases, and retains 99.9% of normalized output variance.

Train directly from a JSON configuration, or reuse a run's saved best configuration:

```bash
python scripts/q2d/deeponet.py --no-tune --train-config config.json --epochs 100
python scripts/q2d/pod_deeponet.py --run-id q2d_pod_01 --no-train
python scripts/q2d/pod_deeponet.py --run-id q2d_pod_01 --no-tune
```

The JSON configuration uses `width`, `latent_width`, `branch_hidden_layers`,
`trunk_hidden_layers`, `activation`, `learning_rate`, `weight_decay`, and
`batch_size`. For POD, the fitted basis determines `latent_width` and disables the
learned trunk. Epochs and seed follow the Q2D CLI and shared defaults.

Compare their saved fields and metrics with another Q2D model:

```bash
python scripts/q2d/plot.py \
  --run artifacts/runs/q2d_cmr/deeponet/<run-id> \
  --run artifacts/runs/q2d_cmr/pod_deeponet/<run-id> \
  --run artifacts/runs/q2d_cmr/fno/<run-id>
```

Plot a saved run and, when present, its benchmark artifacts:

```bash
python scripts/q2d/plot.py \
  --run artifacts/runs/q2d_cmr/fno/<run-id> \
  --cases 1
```

Validate the bundled tutorial/reference cases and create field plots:

```bash
python scripts/q2d/validate_solver.py
```

Run Docker-backed radial-grid validation instead:

```bash
python scripts/q2d/validate_solver.py \
  --radial-grid --lumen-points 4
```

`--lumen-points` defaults to `3` and is used only with `--radial-grid`.
`common.py` supplies Q2D channels, geometry handling, normalization, mesh-file
discovery, and experiment configuration. `fields.ipynb` is an interactive
field-inspection notebook, and `TODO_fno.md` contains development notes.

Compare completed Q2D FNO, TFNO, and UNO runs with:

```bash
scripts/q2d/plot.py \
  --run artifacts/runs/q2d_cmr/fno/<fno-run-id> \
  --run artifacts/runs/q2d_cmr/tfno/<tfno-run-id> \
  --run artifacts/runs/q2d_cmr/uno/<uno-run-id>
```

The TFNO and UNO entry points save training, evaluation, and checkpoint artifacts.
Mesh benchmarks, superresolution artifacts, and cost models are produced by the FNO
entry point only. The Q2D plotter's `--tolerance-comparison` option supports FNO runs only.


## `scripts/diagnostics`

Smoke-test one processed batch and target-reconstruction round trip for the
configured reactor datasets, then write `scripts/diagnostics/processing.png`:

```bash
python scripts/diagnostics/processing.py
```

This script has no command-line options and does not train a model.
