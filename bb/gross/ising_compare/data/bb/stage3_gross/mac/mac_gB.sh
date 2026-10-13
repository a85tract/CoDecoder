#!/bin/bash
# gross stage 3, Mac mini quiet session B (STAGE3_GROSS_PLAN.md 4.2): warm-up, the CPU term on the first 1,000 routed
# test shots, sector Relay-BP and BP-OSD on the first 1,000 test shots; three fresh processes each
W=~/work/gross_stage3; C=$W/code
export PYTHONPATH=$C/ising_compare:$C/qldpc_neural BB_SHOTS=$W/shots BB_PREDS=$W/preds QND_RELAY=~/work/relay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
export BB_SESSION="M4 session gB"
PY=~/work/.venv/bin/python; R=$W/out/rows_cpu_B.csv; L=$W/out/load_B.log
cd $C/ising_compare
load() { echo "$1 $(date +%T) $(uptime | sed 's/.*load averages*: //') | $(ps -Ao pcpu,comm -r | sed -n '2,4p' | awk '{printf "%s %s; ", $1, $NF}')" >> $L; }
echo "== warm-up $(date +%T)"
$PY -m bb.stage3_gross cputerm --split test --chunks 0-29 --first 100 --proc 0 --out $W/out/warmup_B.csv > /dev/null
for d in relay bposd; do $PY -m bb.timing opponent --code bb144 --p 0.004 --dec $d --split test --chunks 0 --first 100 --out $W/out/warmup_B.csv > /dev/null; done
echo "== CPU term $(date +%T)"
for k in 1 2 3; do load "before cpu_term proc$k"; $PY -m bb.stage3_gross cputerm --split test --chunks 0-29 --first 1000 --proc $k --out $R > /dev/null; load "after"; done
echo "== opponents $(date +%T)"
for d in relay bposd; do for k in 1 2 3; do
  load "before $d proc$k"; $PY -m bb.timing opponent --code bb144 --p 0.004 --dec $d --split test --chunks 0 --first 1000 --proc $k --out $R > /dev/null; load "after"
done; done
echo "SESSION_B_END $(date +%T)"
