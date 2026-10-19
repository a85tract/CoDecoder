#!/bin/bash
# ARB3 BB72 checks at one p on the pod: bash run_bb.sh P W   (P = 0.002 or 0.003, W = the joint weight)
# 1. the timed pipelines (d1_m2.py pipeline, D1's configuration at P, dry kind 13): single 20,000 shots and batched 500,000,
#    S on the base and the new engine, with the joint step (--joint-w W, --save) and without it (smoke timing baseline)
# 2. the reads of the same shots (chunks 0-49 of kind 13): eager (joint_read.py), compiled@1 and compiled@4096
#    (d1_compiled.py read --joint --dry), the class scores (joint_scores.py)
# 3. joint_count.py compare (each joint pipeline against the read on its path) and count
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
P=$1; WJ=$2
CFG=$([ $P = 0.002 ] && echo 0 || echo 1)
O=$W/out/bb/p$P; mkdir -p $O $W/out/logs
exec > >(tee -a $W/out/logs/bb_$P.log) 2>&1
st() { echo "$(date -Is) $*" | tee -a $W/out/status.txt; }
export PYTHONPATH=$BBPATH
RUN="$PY $W/pod/podrun.py"
cd $BB/ising_compare
st "bb $P start (w $WJ)"
C4B=$POD_MAP_W4/out/check/depspam_fm/read
if [ ! -f $C4B/p0.00300.npz ]; then
  mkdir -p $C4B $POD_MAP_W4/out/speed && cp $PC/out/speed_components.json $POD_MAP_W4/out/speed/
  $RUN $PC/pc_eval.py --family depspam_fm --ckpt $WTS/C14p.pt,$WTS/S55p_deep.pt --prec bf16 --kind 12 --p 0.002,0.003 \
    --chunks 10:20 --procs 8 --save $C4B --save-det --out $C4B/rows_0.002,0.003.json || { st "FAILED C4b read"; exit 1; }
fi
for J in joint base; do
  for mode in single batched; do
    for eng in base new; do
      n=$([ $mode = single ] && echo 20000 || echo 500000)
      x=""; [ $J = joint ] && x="--joint-w $WJ --save $O/pipe_${mode}_${eng}.npz"
      $RUN $PD/d1_m2.py pipeline --setting depspam_fm --config $CFG --p $P --mode $mode --engine $eng --tail-b 0 --rep 1 \
        --n $n --kind 13 $x --out $O/rows_$J.jsonl || { st "FAILED pipeline $J $mode $eng"; exit 1; }
      st "pipeline $J $mode $eng done"
    done
  done
done
[ -f $O/read/done ] || { $RUN $PD/joint_read.py --family depspam_fm --ckpt $WTS/C14p.pt,$WTS/S55p_deep.pt --prec bf16 \
  --kind 13 --p $P --chunks 0:50 --procs 8 --save $O/read --save-det --out $O/read/rows.json && touch $O/read/done; } \
  || { st "FAILED eager read"; exit 1; }
st "eager read done"
[ -f $O/cread/done_$P ] || $RUN $PD/d1_compiled.py read --read $O/read --p $P --out $O/cread --joint --dry \
  || { st "FAILED compiled read"; exit 1; }
st "compiled read done"
$RUN $PD/joint_scores.py --read $O/read --compiled $O/cread --p $P --ckpt $WTS/S55p_deep.pt || st "scores check FAILED"
for f in $O/pipe_*.npz; do
  $RUN $PD/joint_count.py compare --pipe $f --read $O/read --compiled $O/cread --p $P --w $WJ \
    --out ${f%.npz}_compare.json || st "compare FAILED $f"
done
$RUN $PD/joint_count.py count --read $O/read --compiled $O/cread --p $P --w $WJ --out $O/count.json
st "bb $P end"
