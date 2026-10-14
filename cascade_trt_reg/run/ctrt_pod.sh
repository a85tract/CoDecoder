#!/bin/bash
# ctrt_pod.sh STEP... (CASCADE-TRT-REG, on the H100 pod, root $Q = /workspace/ctrt as code/env.sh sets it): the H100 session,
# ctrt_session.sh's steps in code/env.sh's environment, with B's seeds and beacon record; records in $Q/rec/H100.
source "$(dirname "$(readlink -f "$0")")/../code/env.sh"
export PY P PL OUT=${OUT:-$Q/rec/H100} CARD=h100 SRT_SEEDS=$P/cascade_trt_reg/seeds_ctrt.json \
       SRT_BEACON=$P/cascade_trt_reg/beacon_ctrt.json
unset SRT_DRY
exec bash $P/cascade_trt_reg/run/ctrt_session.sh "$@"
