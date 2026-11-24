#!/bin/bash

# Passing --device cuda explicitly will warn if no GPU avaliable.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# tune and train all
set -- "$@" --tune --train --seeds 1 2 3 4 5

# train all
# --no-tune --train --seeds 1 2 3 4 5 --train-config recent

# python "$SCRIPT_DIR/cstr/deeponet.py" "$@"
# python "$SCRIPT_DIR/cstr/pod_deeponet.py" "$@"
# python "$SCRIPT_DIR/cstr/fno.py" "$@"

# python "$SCRIPT_DIR/packed_bed_1d/deeponet.py" "$@"
# python "$SCRIPT_DIR/packed_bed_1d/pod_deeponet.py" "$@"
# python "$SCRIPT_DIR/packed_bed_1d/fno.py" "$@"

# python "$SCRIPT_DIR/pfr/deeponet.py" "$@"
# python "$SCRIPT_DIR/pfr/pod_deeponet.py" "$@"
# python "$SCRIPT_DIR/pfr/fno.py" "$@"

# python "$SCRIPT_DIR/pfr_heat/deeponet_fno.py" --variant both --plot-cases 3 "$@"
# python "$SCRIPT_DIR/pfr_heat/pod_deeponet_fno.py" --variant both --plot-cases 3 "$@"
# python "$SCRIPT_DIR/pfr_heat/fno.py" --variant both --plot-cases 3 "$@"

# python "$SCRIPT_DIR/pipe_flow/deeponet.py" "$@"
# python "$SCRIPT_DIR/pipe_flow/pod_deeponet.py" "$@"
# python "$SCRIPT_DIR/pipe_flow/fno.py" "$@"

# rollout
# python "$SCRIPT_DIR/pipe_flow_transient/rollout_deeponet.py" "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/rollout_pod_deeponet.py" "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/rollout_fno.py" "$@"

# rollout physics
# python "$SCRIPT_DIR/pipe_flow_transient/rollout_nemo_deeponet.py" "$@"

# field
# python "$SCRIPT_DIR/pipe_flow_transient/field_fno.py" "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/field_nemo_fno.py" "$@"

# python "$SCRIPT_DIR/q2d/deeponet.py" "$@"
# python "$SCRIPT_DIR/q2d/pod_deeponet.py" "$@"
# python "$SCRIPT_DIR/q2d/fno.py" "$@"
# python "$SCRIPT_DIR/q2d/tfno.py" "$@"
# python "$SCRIPT_DIR/q2d/uno.py" "$@"
