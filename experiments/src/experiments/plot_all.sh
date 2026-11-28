#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# Usage: plot_all.sh [extra plot.py args], e.g. plot_all.sh --use-tex (requires LaTeX)

set -- "$@" --recent

python "$SCRIPT_DIR/cstr/plot.py" "$@"
python "$SCRIPT_DIR/packed_bed_1d/plot.py" "$@"
python "$SCRIPT_DIR/pfr/plot.py" "$@"
python "$SCRIPT_DIR/pfr_heat/plot.py" --plot-cases 3 "$@"
python "$SCRIPT_DIR/pipe_flow/plot.py" "$@"
python "$SCRIPT_DIR/pipe_flow_transient/plot.py" "$@"
python "$SCRIPT_DIR/q2d/plot.py" "$@"
