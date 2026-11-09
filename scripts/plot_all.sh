#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# python "$SCRIPT_DIR/cstr/plot.py" --recent
python "$SCRIPT_DIR/packed_bed_1d/plot.py" --recent
python "$SCRIPT_DIR/pfr/plot.py" --recent
python "$SCRIPT_DIR/pfr_heat/plot.py" --recent
python "$SCRIPT_DIR/pipe_flow/plot.py" --recent
python "$SCRIPT_DIR/pipe_flow_transient/plot.py" --recent
python "$SCRIPT_DIR/q2d/plot.py" --recent