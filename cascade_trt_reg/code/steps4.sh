#!/bin/bash
# steps4.sh STEP... (CASCADE-TRT, on the pod, after steps3.sh): env.sh's environment, then
#   alone_seq   Cascade alone again with one Cascade graph at a time (trt_alone.py --seq: batch 1 built, timed and
#               freed before the batch's graph is built), 3 reps x (BB72, d = 13) x (compiled, TensorRT), interleaved:
#               the first alone step held both compiled graphs in one process, which made the compiled batch-1 path
#               slower than in the pipeline harness (and its graph asserted in the check)
#   checkbb2    the BB72 check's compiled side with one graph per process (--only b1, --only bB), then the comparison
set -uo pipefail
source /workspace/ctrt/code/env.sh
mkdir -p $Q/logs $O
cd $BBT/ising_compare
run() { echo "$(date +%T) $*"; "$@" || echo "$(date +%T) FAILED($?): $*"; }
for STEP in "$@"; do
  exec > >(tee -a $Q/logs/$STEP.log) 2>&1
  echo "== $STEP $(date -Is)"
  case $STEP in
    alone_seq)
      for rep in 1 2 3; do for net in bb d13; do for eng in compiled trt; do
        run $PY $C/trt_alone.py --net $net --engine $eng --rep $((rep + 30)) --plans $PL --ckpt $CK --seq --out $O/alone_seq.jsonl
      done; done; done ;;
    checkbb2)
      run $PY $C/trt_check.py --net bb --engine compiled --only b1 --plans $PL --ckpt $CK --save $O/check_bb_compiled_b1.npz
      run $PY $C/trt_check.py --net bb --engine compiled --only bB --plans $PL --ckpt $CK --save $O/check_bb_compiled_bB.npz
      run $PY $C/trt_check.py --net bb --compare $O/check_bb_compiled_b1.npz $O/check_bb_trt.npz \
        --merge $O/check_bb_compiled_bB.npz --out $O/check_bb_split.json ;;
  esac
  echo "== $STEP end $(date -Is)"
done
