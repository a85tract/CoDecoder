#!/bin/bash
# ARB3: the joint step's cost measured directly on this card (checks/bench_joint.py, the build stage's benchmark): Cascade's
# frozen paths with and without the joint step on the same inputs, single (host round trips and CUDA-event GPU time per
# replay) and batched (CUDA-event time per replay at 4,096 / b* = 16). Untimed by the gate; after the timing jobs.
# Caller: B (the dir with pod/ and checks/), R (release root), PY, W4 (C4b map dir), RELAY, D13ROOT, DLOCK (a d = 13 lock
# with the frozen paths), OUTB (new dir), KB (cache dir for this job).
set -euo pipefail
: "${B:?} ${R:?} ${PY:?} ${W4:?} ${RELAY:?} ${D13ROOT:?} ${DLOCK:?} ${OUTB:?} ${KB:?}"
BB=$R/cascade/bb72; PC=$BB/cascade_compare/phase_c; PD=$BB/cascade_compare/phase_d; BO=$BB/cascade_compare/batchopt
mkdir -p $OUTB $KB/triton $KB/inductor
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 TRITON_CACHE_DIR=$KB/triton \
  TORCHINDUCTOR_CACHE_DIR=$KB/inductor POD_RELEASE=$R POD_MAP_W4=$W4 QND_RELAY=$RELAY \
  PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
  SV_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; hostname
cd $BB/ising_compare
for P in 0.002 0.003; do
  WJ=$([ $P = 0.002 ] && echo 6.309573444801933 || echo 3.758374042884441)
  FR=$([ $P = 0.002 ] && echo 0.041 || echo 0.055)        # the differing share of the routed rows (TEST.md 4)
  PYTHONPATH=$BB/ising_compare:$BB/qldpc_neural:$PC:$PD:$BO $PY $B/pod/podrun.py $B/checks/bench_joint.py bb --p $P \
    --w $WJ --frac $FR --out $OUTB/bb_$P.json > $OUTB/bb_$P.log 2>&1 && echo "bench bb $P ok" || echo "bench bb $P FAILED"
done
cd $R/cascade/d13
for P in 0.006 0.0035; do
  WJ=$([ $P = 0.006 ] && echo 7.86574441929562 || echo 13.842311471679352)
  PYTHONPATH=$R/cascade/d13 CONVLINES_RELEASE=$R/evaluation CASCADE_ROOT=$D13ROOT NCCL_IB_DISABLE=1 \
    $PY $B/pod/podrun13.py $B/checks/bench_joint.py d13 --p $P --w $WJ --lock $DLOCK --out $OUTB/d13_$P.json \
    > $OUTB/d13_$P.log 2>&1 && echo "bench d13 $P ok" || echo "bench d13 $P FAILED"
done
date -Is; echo BENCHDONE
