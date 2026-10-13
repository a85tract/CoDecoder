#!/bin/bash
# gross stage 4, the Mac mini's quiet session (STAGE4_GROSS_PLAN.md 4.2): one single-threaded process at a time, each after
# an idle check (ps: no Python process at >= 1 % CPU, no process at >= 50 %); load before and after each. Warm-up (untimed),
# then three fresh processes each: sector Relay-BP and BP-OSD on the first 1,000 test4 shots, full-DEM Relay-BP on the
# first 300 (G-4F's bar), G-4R's CPU term (sector Relay-BP on its first 1,000 routed test4 shots) and G-4F's (full-DEM
# Relay-BP on its first 100 routed test4 shots).
W=~/work/gross_stage4; C=$W/code
export PYTHONPATH=$C/ising_compare:$C/qldpc_neural BB_SHOTS=$W/shots BB_PREDS=$W/preds QND_RELAY=~/work/relay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
export BB_SESSION=M4 session s4
PY=~/work/.venv/bin/python; R=$W/out/rows_cpu.csv; U=$W/out/warmup.csv; L=$W/out/load.log
cd $C/ising_compare
load() { echo "$1 $(date +%T) $(uptime | sed 's/.*load averages*: //') | $(ps -Ao pcpu,comm -r | sed -n '2,4p' | awk '{printf "%s %s; ", $1, $NF}')" >> $L; }
idle() {
  for t in $(seq 1 60); do
    busy=$(ps -Ao pcpu=,comm= | awk '($1 >= 1.0 && tolower($0) ~ /python/) || $1 >= 50.0' | head -3 | tr '\n' ';')
    [ -z "$busy" ] && return 0
    echo "busy $(date +%T): $busy" >> $L; sleep 30
  done
  echo "IDLE_FAIL $(date +%T)" >> $L; return 1
}
run() { idle || { echo "IDLE_FAIL $(date +%T)"; exit 3; }; load "before $1"; shift; "$@" > /dev/null || echo "FAILED $*"; load "after"; }
OPP="$PY -m bb.timing opponent --code bb144 --p 0.004 --split test4 --chunks 0"
CPU="$PY -m bb.stage4_gross cputerm --split test4"
echo "== warm-up $(date +%T)"
for d in relay bposd; do run "warm-up $d" $OPP --dec $d --first 100 --out $U; done
run "warm-up relay_full" $OPP --dec relay_full --first 10 --out $U
run "warm-up G-4R" $CPU --cell G-4R --first 100 --proc 0 --out $U
run "warm-up G-4F" $CPU --cell G-4F --first 10 --proc 0 --out $U
echo "== opponents $(date +%T)"
for d in relay bposd; do for k in 1 2 3; do run "$d proc$k" $OPP --dec $d --first 1000 --proc $k --out $R; done; done
for k in 1 2 3; do run "relay_full proc$k" $OPP --dec relay_full --first 300 --proc $k --out $R; done
echo "== CPU terms $(date +%T)"
for k in 1 2 3; do run "G-4R cpu_term proc$k" $CPU --cell G-4R --first 1000 --proc $k --out $R; done
for k in 1 2 3; do run "G-4F cpu_term proc$k" $CPU --cell G-4F --first 100 --proc $k --out $R; done
echo "SESSION_END $(date +%T)"
