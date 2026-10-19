#!/bin/bash
# Replication of the d = 13 single-mode smoke timing at p = 0.006 (reps 2, 3; base and joint alternating), and a dry test of
# srt_d13.py's session and joint-lock commands (H100 card).
source "$(dirname "$0")/pod_env.sh"
T=$R/cascade/d13; O=$W/out/d13/p0.006; DL=$W/out/d13/dry_lock.json
export PYTHONPATH=$T CONVLINES_RELEASE=$R/evaluation CASCADE_ROOT=$W/d13root NCCL_IB_DISABLE=1
cd $T
for rep in 2 3; do
  $PY $W/pod/podrun13.py surface.sv_speed pipeline --config C5 --p 0.006 --mode single --rep $rep --lock $DL --stage c7 \
    --dry --out $O/rows_rep_base.jsonl --save $O/pipes_rep_base
  $PY $W/pod/podrun13.py surface.sv_speed pipeline --config C5 --p 0.006 --mode single --rep $rep --lock $DL --stage c7 \
    --dry --out $O/rows_rep_joint.jsonl --save $O/pipes_rep_joint --joint-w 1.0
done
echo "$(date -Is) d13 rep done" >> $W/out/status.txt
cd $T && SRT_CARD=h100 $PY $R/speed_rt/cascade/srt_d13.py session --out $W/out/d13/srt_session.json \
  && SRT_CARD=h100 $PY $R/speed_rt/cascade/srt_d13.py joint-lock --lock $T/out/locked_d13.json \
     --session $W/out/d13/srt_session.json --out $W/out/d13/srt_joint_lock.json
echo "$(date -Is) srt_d13 test rc $?" >> $W/out/status.txt
