#!/bin/bash
# ARB3 d = 13 checks at one p on the pod: bash run_d13.sh P W
# C5 (sv_speed pipeline, the pod's dry lock: kind 13, chunks 0-49, single 20,000 and batched 500,000) with the joint step
# (--joint-w W) and without it (smoke timing baseline); the reads of the same shots (sv_read --dry --stage speed: S1, B1 on
# chunks 0-1, SB, BQ5 on chunks 0-49); sv_joint check of each joint pipeline file.
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
P=$1; WJ=$2
T=$R/cascade/d13
O=$W/out/d13/p$P; mkdir -p $O $W/out/logs $W/d13root/ckpt/CS13
exec > >(tee -a $W/out/logs/d13_$P.log) 2>&1
st() { echo "$(date -Is) $*" | tee -a $W/out/status.txt; }
ln -sf $R/weights/cascade_d13/CS13/final_ema.pt $W/d13root/ckpt/CS13/final_ema.pt
export PYTHONPATH=$T CONVLINES_RELEASE=$R/evaluation CASCADE_ROOT=$W/d13root NCCL_IB_DISABLE=1
cd $T
DL=$W/out/d13/dry_lock.json
[ -f $DL ] || $PY $W/pod/podrun13.py --derive $T/out/locked_d13.json $DL || { st "FAILED d13 lock"; exit 1; }
RUN="$PY $W/pod/podrun13.py"
st "d13 $P start (w $WJ)"
for J in joint base; do
  for mode in single batched; do
    x=""; [ $J = joint ] && x="--joint-w $WJ"
    $RUN surface.sv_speed pipeline --config C5 --p $P --mode $mode --rep 1 --lock $DL --stage c7 --dry \
      --out $O/rows_$J.jsonl --save $O/pipes_$J $x || { st "FAILED d13 pipeline $J $mode"; exit 1; }
    st "d13 pipeline $J $mode done"
  done
done
[ -d $O/read/single ] || $RUN surface.sv_read --dry --stage speed --p $P --chunks 0-2 --comp S1,B1 --lock $DL \
  --out $O/read/single || { st "FAILED d13 read single"; exit 1; }
[ -d $O/read/batched ] || $RUN surface.sv_read --dry --stage speed --p $P --chunks 0-50 --comp SB,BQ5 --lock $DL \
  --out $O/read/batched || { st "FAILED d13 read batched"; exit 1; }
st "d13 reads done"
for f in $O/pipes_joint/C5_*.npz; do
  $PY -m surface.sv_joint check --lock $DL --pipe $f --read $O/read --out $O/check_$(basename ${f%.npz}).json \
    || st "d13 check FAILED $f"
done
st "d13 $P end"
