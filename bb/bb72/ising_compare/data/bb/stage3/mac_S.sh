#!/bin/bash
# BB72 stage 3, Amendment S: Mac mini quiet session S, the CPU term of BB72-2S and BB72-3S (amendment_S.json)
W=~/work/bb72_stage3; C=$W/code
export PYTHONPATH=$C/ising_compare:$C/qldpc_neural BB_SHOTS=$W/shots BB_PREDS=$W/preds QND_RELAY=~/work/relay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1
export BB_SESSION="M4 session S"
PY=~/work/.venv/bin/python; R=$W/out/rows_cpu_S.csv; L=$W/out/load_S.log; A=data/bb/stage3/amendment_S.json
cd $C/ising_compare
top() { ps -Ao pcpu,comm -r | sed -n '2,4p' | awk '{printf "%s %s; ", $1, $NF}'; }
# idle = no process at >= 20 % CPU in ps, no python process at >= 1 % CPU, 1-min load < 3; wait (15-s steps) until it
# holds. (Registered as "no python process"; a system Python 3.9 idle since boot, 0.0 % CPU, kept that false.)
idle() {
  local t0=$SECONDS
  until [ "$(ps -Ao pcpu= | awk '$1>=20' | wc -l | tr -d ' ')" = 0 ] && [ "$(ps -Ao pcpu=,comm= | awk 'tolower($0) ~ /python/ && $1 >= 1' | wc -l | tr -d ' ')" = 0 ] \
        && awk -v l="$(sysctl -n vm.loadavg | awk '{print $2}')" 'BEGIN{exit !(l<3)}'; do sleep 15; done
  echo "before $1 $(date +%T) idle (no process >= 20 % CPU, no python process >= 1 % CPU, load < 3; waited $((SECONDS-t0)) s) load $(sysctl -n vm.loadavg) | $(top)" >> $L
}
after() { echo "after $1 $(date +%T) load $(sysctl -n vm.loadavg) | $(top)" >> $L; }
echo "== session S start $(date +%T)"
echo "python processes at the start: $(ps -Ao pid=,pcpu=,etime=,comm= | grep -i python | awk '{printf "pid %s %s %% CPU up %s %s; ", $1, $2, $3, $4}')" >> $L
idle warm-up; $PY -m bb.timing opponent --code bb72 --p 0.003 --dec relay --split test --chunks 1 --first 1000 --out $W/out/warmup_S.csv > /dev/null; after warm-up
for k in 1 2 3; do idle "BB72-2S proc$k"; $PY -m bb.stage3 cputerm --amend $A --cell BB72-2S --split test --chunks 0-29 --tag r12 --first 1000 --proc $k --out $R > /dev/null; after "BB72-2S proc$k"; done
for k in 1 2 3; do idle "BB72-3S proc$k"; $PY -m bb.stage3 cputerm --amend $A --cell BB72-3S --split test --chunks 0-19 --tag r12 --first 1000 --proc $k --out $R > /dev/null; after "BB72-3S proc$k"; done
echo "SESSION_S_END $(date +%T)"
