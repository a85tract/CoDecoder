#!/bin/bash
# ctrt_pod_rtx.sh STEP... (CASCADE-TRT-REG, written after B for the move of the RTX PRO 6000 part from the cluster's partition
# artxpro6000, which had no free card, to a RunPod RTX PRO 6000 Blackwell Server Edition pod; root $Q = /workspace/ctrt as
# code/env.sh sets it): ctrt_pod.sh with CARD=rtx, so ctrt_session.sh takes the RTX PRO 6000's branch (the study's d = 13 lock,
# no H100 session record) and srt_cas.py / srt_d13.py the card's own guards; records in $Q/rec/RTX.
source "$(dirname "$(readlink -f "$0")")/../code/env.sh"
export PY P PL OUT=${OUT:-$Q/rec/RTX} CARD=rtx SRT_CARD=rtx SRT_SEEDS=$P/cascade_trt_reg/seeds_ctrt.json \
       SRT_BEACON=$P/cascade_trt_reg/beacon_ctrt.json
unset SRT_DRY
exec bash $P/cascade_trt_reg/run/ctrt_session.sh "$@"
