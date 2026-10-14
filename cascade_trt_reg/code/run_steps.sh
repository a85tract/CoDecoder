#!/bin/bash
# run_steps.sh STEP... (CASCADE-TRT, on the pod): the harnesses' environment (speed_rt/cascade/common.sh and h100_pod.sh's
# d = 13 part, on the public tree), then each step in order; output in $O, one log per step in $Q/logs.
#   build   TensorRT plans: bb at 1 and 4,096, d13 at 1 and 16 (trt_build.py)
#   check   the decision check (trt_check.py), BB72 then d = 13
#   alone   Cascade alone, 3 reps x (BB72, d = 13) x (compiled, trt), interleaved, one fresh process each (trt_alone.py)
#   pairs   the headline pairs through the harnesses (trt_launch.py), engines interleaved per rep
#   pairs2  the first registered test's pairs (phase C, S55p -> C14p), if cheap
set -uo pipefail
Q=${Q:-/workspace/ctrt}; P=$Q/pub; BBT=$P/cascade/bb72; T13=$P/cascade/d13; C=$Q/code; PL=$Q/plans; O=$Q/out
PC=$BBT/cascade_compare/phase_c; PD=$BBT/cascade_compare/phase_d; BO=$BBT/cascade_compare/batchopt
mkdir -p $Q/logs $O $PL $Q/cache/triton $Q/cache/inductor
export PYTHONPATH=$BBT/ising_compare:$BBT/qldpc_neural:$PC:$PD:$BO:$T13:$C OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       PYTHONDONTWRITEBYTECODE=1 QND_RELAY=$Q/relay BB_PIDNS=1 TRITON_CACHE_DIR=$Q/cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$Q/cache/inductor CASCADE_ROOT=$Q/d13root CONVLINES_RELEASE=$P/evaluation \
       NCCL_IB_DISABLE=1 PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
       CTRT_PUB=$P CTRT_PLANS=$PL
PY=$Q/venv/bin/python; CK=$P/weights/cascade_bb72/C14p.pt
cd $BBT/ising_compare
run() { echo "$(date +%T) $*"; "$@" || echo "$(date +%T) FAILED($?): $*"; }
for STEP in "$@"; do
  exec > >(tee -a $Q/logs/$STEP.log) 2>&1
  echo "== $STEP $(date -Is)"
  case $STEP in
    build)
      run $PY $C/trt_build.py --net bb --ckpt $CK --batch 1 --out $PL
      run $PY $C/trt_build.py --net bb --ckpt $CK --batch 4096 --out $PL
      run $PY $C/trt_build.py --net d13 --batch 1 --out $PL
      run $PY $C/trt_build.py --net d13 --batch 16 --out $PL ;;
    check)
      run $PY $C/trt_check.py --net bb --plans $PL --ckpt $CK --out $O/check_bb.json
      run $PY $C/trt_check.py --net d13 --plans $PL --out $O/check_d13.json ;;
    alone)
      for rep in 1 2 3; do for net in bb d13; do for eng in compiled trt; do
        run $PY $C/trt_alone.py --net $net --engine $eng --rep $rep --plans $PL --ckpt $CK --out $O/alone.jsonl
      done; done; done ;;
    pairs|pairs2|alone_pipe|smoke)
      bash $C/pairs.sh $STEP ;;
    rebuild)
      [ -d $PL ] && mv $PL ${PL}_ws24 && mkdir -p $PL
      run $PY $C/trt_build.py --net bb --ckpt $CK --batch 1 --out $PL
      run $PY $C/trt_build.py --net bb --ckpt $CK --batch 4096 --out $PL
      run $PY $C/trt_build.py --net d13 --batch 1 --out $PL
      run $PY $C/trt_build.py --net d13 --batch 16 --out $PL ;;
  esac
  echo "== $STEP end $(date -Is)"
done
