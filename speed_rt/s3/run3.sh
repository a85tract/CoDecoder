#!/bin/bash
# SPEED-RT stage 3 (STAGE3_batched.md sections 2 and 5): one pod's GPU work, one process at a time.
#   run3.sh decide POINT...   the decision pass of each point on its timing shots ($W/shots/<point>) -> $W/out/decide
#   run3.sh time POINT...     each point's timed processes, rep 1..5 (gross: 1..3), in the order route (each tau),
#                             alone; a process that fails (exit != 0: window rejected or outputs differ) is replaced at
#                             once, at most three attempts per slot (C5) -> $W/out/time
# The core is the pre-screen's ($W/out/prescreen/prescreen.json). Heartbeat: $W/out/progress.txt.
set -u
W=${S3_W:-/workspace/s3}; PY=$W/venv/bin/python; cd "$W/rel" || exit 2
export PYTHONPATH=$W/rel
CORE=$($PY -c "import json; print(json.load(open('$W/out/prescreen/prescreen.json'))['core'])")
what=$1; shift
for pt in "$@"; do
  if [ "$what" = decide ]; then
    $PY -m speed_rt.s3.gpu3 decide --point $pt --shots $W/shots/$pt --out $W/out/decide; echo "decide $pt exit $?" >> $W/out/progress.txt
    continue
  fi
  reps="1 2 3 4 5"; taus="q"; [ $pt = gross-p4 ] && { reps="1 2 3"; taus="bal acc"; }
  for rep in $reps; do
    for slot in $(for t in $taus; do echo "route:$t"; done) alone; do
      mode=${slot%%:*}; tau=${slot#*:}; [ $mode = alone ] && tau=q
      for att in 1 2 3; do
        $PY -m speed_rt.s3.gpu3 time --point $pt --mode $mode --tau $tau --shots $W/shots/$pt --decide $W/out/decide \
          --core $CORE --out $W/out/time --rep $rep --attempt $att > $W/out/time_log_${pt}_${slot/:/_}_${rep}_a$att.txt 2>&1
        rc=$?; echo "$(date +%T) time $pt $slot rep $rep attempt $att exit $rc" >> $W/out/progress.txt
        [ $rc = 0 ] && break
      done
    done
  done
done
echo "$(date +%T) $what $* done" >> $W/out/progress.txt
