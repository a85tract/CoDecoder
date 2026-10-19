#!/bin/bash
# ARB3's timing of the joint decision on BB72 (unregistered; TIMING.md): one drive.py r1 job on this machine's card with
# --joint-both --engines base: per repetition and p (0.2 %, 0.3 %), Cascade alone (single, batched), the composition
# S55'-deep -> C14' on the BB engine (Table I's), and the same composition with the joint step, each one fresh process,
# 3 repetitions, 20,000 single / 500,000 batched shots, gate on, no burn. Shots: the stage-1 beacon record's seeds under
# new set names (sets_arb3.json), so new seeds. Before the clock, untimed: C4b's read for m2.py's in-process S check.
# Caller: R (the release root, patched), PY, OUT (new dir), CARD (rtx|h100), JC (job-local cache dir), W4 (the W4 map
# dir), RELAY (QND_RELAY), WB (the joint w per p).
set -euo pipefail
: "${R:?} ${PY:?} ${OUT:?} ${CARD:?} ${JC:?} ${W4:?} ${RELAY:?} ${WB:?}"
BB=$R/cascade/bb72; PC=$BB/cascade_compare/phase_c; PD=$BB/cascade_compare/phase_d; BO=$BB/cascade_compare/batchopt
CAS=$R/speed_rt/cascade; WTS=$R/weights/cascade_bb72
[ ! -e "$OUT" ] || { echo "$OUT exists"; exit 7; }
mkdir -p $JC/triton $JC/inductor
cd $BB/ising_compare
export PYTHONPATH=$BB/ising_compare:$BB/qldpc_neural:$PC:$PD:$BO OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 QND_RELAY=$RELAY TRITON_CACHE_DIR=$JC/triton TORCHINDUCTOR_CACHE_DIR=$JC/inductor \
  PC_TREE="$(cat $BB/DEPLOYED)" SRT_BURN_S=0 \
  PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.default_limit --format=csv,noheader
echo "host $(hostname) cpu $(lscpu | grep 'Model name' | sed 's/.*: *//') nproc $(nproc) tree $PC_TREE out $OUT $(date -Is)"
C4B=$W4/out/check/depspam_fm/read
if [ ! -f $C4B/p0.00300.npz ]; then
  t0=$(date +%s)
  mkdir -p $C4B $W4/out/speed && cp $PC/out/speed_components.json $W4/out/speed/
  POD_RELEASE=$R POD_MAP_W4=$W4 $PY $R/../pod/podrun.py $PC/pc_eval.py --family depspam_fm --ckpt $WTS/C14p.pt,$WTS/S55p_deep.pt \
    --prec bf16 --kind 12 --p 0.002,0.003 --chunks 10:20 --procs 8 --save $C4B --save-det --out $C4B/rows_0.002,0.003.json \
    > $W4/c4b_read.log 2>&1 || { echo "C4b read FAILED"; tail -20 $W4/c4b_read.log; exit 1; }
  echo "C4b read in $(( $(date +%s) - t0 )) s"
fi
export SRT_RELMAP=1 SRT_RELMAP_RELEASE=$R SRT_RELMAP_W4=$W4 SRT_SETS_FILE=$CAS/sets_arb3.json \
  SRT_SEEDS=$CAS/seeds_arb3.json SRT_BEACON=$CAS/beacon.json
unset SRT_DRY
$PY $CAS/drive.py r1 --tree $BB --out $OUT --card $CARD --joint-w "$WB" --joint-both --engines base ${DRIVE_ARGS:-}
date -Is; echo JOBDONE
