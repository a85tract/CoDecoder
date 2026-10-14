#!/bin/bash
# steps2.sh STEP... (CASCADE-TRT, on the pod, after run_steps.sh build/check/alone): env.sh's environment, then
#   d13v2       rebuild the d = 13 plans with every convolution held to BF16 (--conv-prec; FP16 only if BF16 fails):
#               TensorRT's BF16 flag alone gave an FP32/TF32 engine (plans_d13_tf32/ keeps those plans)
#   inspect     layer summary of every plan
#   alone13     d = 13 Cascade alone again, compiled and TensorRT interleaved, 3 reps (rows to out/alone.jsonl, tagged
#               by the plan's sha256 in "trt")
#   checksplit  the decision check with one process per engine (BB72 and d = 13), then the comparison
#   pairs pairs2 alone_pipe   as pairs.sh
set -uo pipefail
source /workspace/ctrt/code/env.sh
mkdir -p $Q/logs $O
cd $BBT/ising_compare
run() { echo "$(date +%T) $*"; "$@" || echo "$(date +%T) FAILED($?): $*"; }
for STEP in "$@"; do
  exec > >(tee -a $Q/logs/$STEP.log) 2>&1
  echo "== $STEP $(date -Is)"
  case $STEP in
    d13v2)
      mkdir -p $Q/plans_d13_tf32 && mv $PL/d13_* $Q/plans_d13_tf32/ 2>/dev/null
      grep '"net": "d13"' $PL/build.jsonl > $Q/plans_d13_tf32/build.jsonl
      run $PY $C/trt_build.py --net d13 --batch 1 --out $PL --conv-prec
      run $PY $C/trt_build.py --net d13 --batch 16 --out $PL --conv-prec ;;
    inspect)
      run $PY $C/inspect_layers.py $PL 6 ;;
    alone13)
      for rep in 1 2 3; do for eng in compiled trt; do
        run $PY $C/trt_alone.py --net d13 --engine $eng --rep $((rep + 10)) --plans $PL --out $O/alone.jsonl
      done; done ;;
    checksplit)
      for net in bb d13; do
        for eng in compiled trt; do
          run $PY $C/trt_check.py --net $net --engine $eng --plans $PL --ckpt $CK --save $O/check_${net}_$eng.npz
        done
        run $PY $C/trt_check.py --net $net --compare $O/check_${net}_compiled.npz $O/check_${net}_trt.npz \
          --out $O/check_${net}_split.json
      done ;;
    pairs|pairs2|alone_pipe|smoke)
      bash $C/pairs.sh $STEP ;;
  esac
  echo "== $STEP end $(date -Is)"
done
