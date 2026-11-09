#!/bin/bash

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# python "$SCRIPT_DIR/cstr/deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/cstr/pod_deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/cstr/fno.py" --tune --train "$@"

# python "$SCRIPT_DIR/packed_bed_1d/deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/packed_bed_1d/pod_deeponet.py" --tune --train "$@"
python "$SCRIPT_DIR/packed_bed_1d/fno.py" --tune --train "$@"

# python "$SCRIPT_DIR/pfr/deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/pfr/pod_deeponet.py" --tune --train "$@"
python "$SCRIPT_DIR/pfr/fno.py" --tune --train "$@"

python "$SCRIPT_DIR/pfr_heat/deeponet_fno.py" --tune --train --variant both "$@"
python "$SCRIPT_DIR/pfr_heat/fno.py" --tune --train --variant both "$@"

# python "$SCRIPT_DIR/pipe_flow/deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/pipe_flow/pod_deeponet.py" --tune --train "$@"
python "$SCRIPT_DIR/pipe_flow/fno.py" --tune --train "$@"

# python "$SCRIPT_DIR/pipe_flow_transient/deeponet.py" --tune --train "$@"
python "$SCRIPT_DIR/pipe_flow_transient/pod_deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/nemo_deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/transient_fno.py" --tune --train "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/transient_nemo_fno.py" --tune --train "$@"

python "$SCRIPT_DIR/q2d/deeponet.py" --tune --train "$@"
python "$SCRIPT_DIR/q2d/pod_deeponet.py" --tune --train "$@"
# python "$SCRIPT_DIR/q2d/fno.py" --tune --train "$@"
# python "$SCRIPT_DIR/q2d/tfno.py" --tune --train "$@"
# python "$SCRIPT_DIR/q2d/uno.py" --tune --train "$@"
