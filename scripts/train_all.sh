#!/bin/bash

# Passing --device cuda explicitly will warn if no GPU avaliable.

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)

# python "$SCRIPT_DIR/cstr/deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/cstr/pod_deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/cstr/fno.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"

# python "$SCRIPT_DIR/packed_bed_1d/deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/packed_bed_1d/pod_deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/packed_bed_1d/fno.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"

# python "$SCRIPT_DIR/pfr/deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pfr/pod_deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pfr/fno.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"

# python "$SCRIPT_DIR/pfr_heat/deeponet_fno.py" --no-tune --train --variant both --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pfr_heat/fno.py" --no-tune --train --variant both --seeds 1 2 3 4 5 --train-config recent "$@"

# python "$SCRIPT_DIR/pipe_flow/deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pipe_flow/pod_deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pipe_flow/fno.py" --no-tune --train --variant both --seeds 1 2 3 4 5 --train-config recent "$@"
# No completed tuned runs yet:
# python "$SCRIPT_DIR/pipe_flow/physics_deeponet.py" --no-tune --train --variant both --seeds 1 2 3 4 5 --train-config recent "$@"

# python "$SCRIPT_DIR/pipe_flow_transient/nemo_deeponet.py" --no-tune --train --variant both --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/transient_nemo_fno.py" --no-tune --train --variant both --seeds 1 2 3 4 5 --train-config recent "$@"
# No completed tuned runs yet:
# python "$SCRIPT_DIR/pipe_flow_transient/deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/pod_deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/pipe_flow_transient/transient_fno.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"

python "$SCRIPT_DIR/q2d/deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
python "$SCRIPT_DIR/q2d/pod_deeponet.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
python "$SCRIPT_DIR/q2d/fno.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/q2d/tfno.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
# python "$SCRIPT_DIR/q2d/uno.py" --no-tune --train --seeds 1 2 3 4 5 --train-config recent "$@"
