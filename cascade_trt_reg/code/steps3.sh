#!/bin/bash
# steps3.sh STEP... (CASCADE-TRT, on the pod, after steps2.sh pairs): env.sh's environment, then
#   checkbb     the BB72 decision check's compiled process again (batch 1 first; trt_check.py) and the comparison with
#               the TensorRT process of steps2.sh checksplit
#   alone_pipe pairs2   as pairs.sh
#   opt5        sensitivity: TensorRT plans at builder optimisation level 5 (plans_o5/), Cascade alone on them,
#               3 reps per network (rows to out/alone_o5.jsonl)
set -uo pipefail
source /workspace/ctrt/code/env.sh
mkdir -p $Q/logs $O
cd $BBT/ising_compare
run() { echo "$(date +%T) $*"; "$@" || echo "$(date +%T) FAILED($?): $*"; }
for STEP in "$@"; do
  exec > >(tee -a $Q/logs/$STEP.log) 2>&1
  echo "== $STEP $(date -Is)"
  case $STEP in
    checkbb)
      run $PY $C/trt_check.py --net bb --engine compiled --plans $PL --ckpt $CK --save $O/check_bb_compiled.npz
      run $PY $C/trt_check.py --net bb --compare $O/check_bb_compiled.npz $O/check_bb_trt.npz --out $O/check_bb_split.json ;;
    alone_pipe|pairs2)
      bash $C/pairs.sh $STEP ;;
    opt5)
      P5=$Q/plans_o5; mkdir -p $P5
      run $PY $C/trt_build.py --net bb --ckpt $CK --batch 1 --out $P5 --opt-level 5
      run $PY $C/trt_build.py --net bb --ckpt $CK --batch 4096 --out $P5 --opt-level 5
      run $PY $C/trt_build.py --net d13 --batch 1 --out $P5 --opt-level 5 --conv-prec
      run $PY $C/trt_build.py --net d13 --batch 16 --out $P5 --opt-level 5 --conv-prec
      for rep in 1 2 3; do for net in bb d13; do
        run $PY $C/trt_alone.py --net $net --engine trt --rep $((rep + 20)) --plans $P5 --ckpt $CK --out $O/alone_o5.jsonl
      done; done ;;
  esac
  echo "== $STEP end $(date -Is)"
done
