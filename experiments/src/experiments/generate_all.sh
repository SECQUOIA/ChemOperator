#!/bin/bash

# Generate datasets under <project root>/datasets (EXPERIMENTS_PROJECT_ROOT, or else
# the uv workspace root). Each generator overwrites its files in place. Only q2d
# reads command-line arguments; the others run in full even when given --help.
# Uncomment the datasets to regenerate.

set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# python "$SCRIPT_DIR/cstr/generate_dataset.py"
# python "$SCRIPT_DIR/packed_bed_1d/generate_dataset.py"
# python "$SCRIPT_DIR/pfr/generate_dataset.py"
# python "$SCRIPT_DIR/pfr_heat/generate_dataset.py"
# python "$SCRIPT_DIR/pipe_flow/generate_dataset.py"
# python "$SCRIPT_DIR/pipe_flow_transient/generate_dataset.py"

python "$SCRIPT_DIR/q2d/generate_dataset.py" --suite base
# python "$SCRIPT_DIR/q2d/generate_dataset.py" --suite resolution
# python "$SCRIPT_DIR/q2d/generate_dataset.py" --suite tolerance
