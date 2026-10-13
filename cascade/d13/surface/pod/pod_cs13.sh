#!/bin/bash
# pod_cs13.sh DEADLINE: CS13 on one rented H200. Runs on the pod, detached:
#   setsid nohup /workspace/cascade_w5/pod_cs13.sh DEADLINE > /workspace/cascade_w5/logs/runner.out 2>&1 < /dev/null &
# jobs/cs13.sh's contract for one card: the tree 917ad20b at /workspace/cascade_w5/code (hashes checked against the
# tree the Alpine chain pins), the venv /workspace/venv_cu128 (the pins of Alpine's venv_cu128), CASCADE_ROOT
# /workspace/cascade_w5, gpu.sbatch's environment. One segment with jobs/cs13.sh's arguments and --micro 4 (world 1,
# four micro-batches of 832), from scratch, or with --resume after an intentional pause. --max-minutes ends the segment
# with last.pt at the epoch DEADLINE (before the pod's spend cap) if the run has not ended by then. The ends, as in
# jobs/cs13.sh, in the run directory $D:
#   DONE       completed (120,000 steps) or no_improvement, exit 0 and final_ema.pt: the status, the checkpoints' sha256
#   HANDOVER   paused at DEADLINE (exit 3, progress, last.pt): last.pt's sha256; the run continues elsewhere with
#              world 1 and --micro 4 (surface_train refuses another world size or micro-batch count on resume)
#   ATTENTION  every other end (a non-finite loss, a crash, no status): the reason, the status, the log's last lines
# runner.state: "running" while the trainer runs, then "ended <exit code> <status>"; runner.pid, trainer.pid.
set -uo pipefail
DEADLINE=${1:?DEADLINE (epoch seconds) missing}
R=/workspace/cascade_w5; T=$R/code; D=$R/ckpt/CS13; PY=/workspace/venv_cu128/bin/python
export PYTHONPATH=$T OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CONVLINES_RELEASE=$T/release CASCADE_ROOT=$R \
       TRITON_CACHE_DIR=$R/triton_cache TORCHINDUCTOR_CACHE_DIR=$R/inductor_cache
cd $T || exit 1
mkdir -p $D $R/logs
N=$(( $(grep -c ' start: ' $D/segments.log 2>/dev/null || true) + 1 ))
SEG=pod$N                                        # the segment's name
JLOG=$R/logs/CS13_$SEG.out                       # the trainer's output
echo $$ > $R/runner.pid

sfield() {                                       # a field of status.json (empty if the file is missing)
  if [ -f $D/status.json ]; then sed -nE "s/.*\"$1\": \"?([A-Za-z0-9_.:+-]+)\"?.*/\1/p" $D/status.json; fi
}
note() { echo "$(date -Is) $SEG $*" >> $D/segments.log; echo "pod_cs13: $*"; }
halt() {                                         # ATTENTION with the reason (the first one stays)
  if [ ! -f $D/ATTENTION ]; then
    { echo "ATTENTION $(date -Is) run CS13 segment $SEG host $(hostname)"
      echo "reason: $1"
      echo "status.json: $(cat $D/status.json 2>/dev/null || echo missing)"
      echo "last lines of $JLOG:"
      tail -n 12 $JLOG 2>/dev/null | cut -c1-300 || true; } > $D/ATTENTION.tmp
    mv $D/ATTENTION.tmp $D/ATTENTION
  fi
  note "ATTENTION: $1"
  echo "ended attention" > $R/runner.state
  exit 1
}

if [ -f $D/DONE ]; then
  note "DONE exists: nothing to run"; exit 0
elif [ -f $D/ATTENTION ]; then
  note "refused: ATTENTION exists ($(sed -n 2p $D/ATTENTION | cut -c1-200))"; exit 1
fi
if [ -f $D/last.pt ]; then
  prev=$(sfield status)
  [ "$prev" = paused ] ||
    halt "last.pt exists but the previous segment did not end in an intentional pause (status.json: ${prev:-missing})"
  RES=--resume
  rm -f $D/HANDOVER
elif [ -f $D/status.json ] || grep -q ' start: ' $D/segments.log 2>/dev/null; then
  halt "an earlier segment ran but left no last.pt"
else
  RES=""
fi
MM=$(( (DEADLINE - $(date +%s)) / 60 ))          # minutes before the pause at the deadline
[ $MM -ge 15 ] || halt "only $MM minutes left before the deadline $DEADLINE"
if [ -f $D/status.json ]; then                    # the previous end, kept in the history; this end writes a new one
  { cat $D/status.json; echo; } >> $D/status.history
  rm $D/status.json
fi
note "start: host $(hostname), $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1), W 1, MM $MM (deadline $(date -d @$DEADLINE -Is)), tree $(cat $T/DEPLOYED), ${RES:-fresh}"
A="--run CS13 --run-id 1 --d 13 --L 13 --H 256 --res skip --mup-base 0 --compile --workers 2 --log-every 200 \
   --max-steps 120000 --monitor 0.006:100:130,0.007:100:110 --monitor-every 1e7 --monitor-kind 1 --monitor-pm \
   --pm-procs 12 --raw-until 8e7 --no-improvement 3e7:0.02 --max-minutes $MM $RES"
echo running > $R/runner.state
$PY -m surface.surface_train $A --micro 4 > $JLOG 2>&1 &
TP=$!
echo $TP > $R/trainer.pid
rc=0
wait $TP || rc=$?
st=$(sfield status); s0=$(sfield start); s1=$(sfield step)
note "end: exit $rc, status ${st:-none}, steps ${s0:-?} -> ${s1:-?}"
case "$st" in
  paused)
    [ $rc = 3 ] && [ -f $D/last.pt ] && [ "${s1:-0}" -gt "${s0:-0}" ] ||
      halt "paused with exit code $rc at steps ${s0:-?} -> ${s1:-?}: no progress or an inconsistent end"
    { echo "HANDOVER $(date -Is) run CS13 segment $SEG: paused at step $s1 at the deadline $(date -d @$DEADLINE -Is)"
      echo "continue with world 1 (--micro 4) and --resume from this last.pt"
      cat $D/status.json; echo
      echo "$(sha256sum $D/last.pt | cut -c1-64)  last.pt"; } > $D/HANDOVER.tmp
    mv $D/HANDOVER.tmp $D/HANDOVER
    note "HANDOVER: paused at step $s1" ;;
  completed|no_improvement)
    [ $rc = 0 ] && [ -f $D/final_ema.pt ] || halt "status $st with exit code $rc; final_ema.pt: $(ls $D/final_ema.pt 2>&1)"
    { echo "DONE $(date -Is) run CS13 segment $SEG: status $st at step $s1"
      cat $D/status.json; echo
      for f in final_ema.pt budget_ema.pt last.pt log.json; do
        if [ -f $D/$f ]; then echo "$(sha256sum $D/$f | cut -c1-64)  $f"; fi
      done; } > $D/DONE.tmp
    mv $D/DONE.tmp $D/DONE
    note "DONE: $st at step $s1" ;;
  nonfinite)
    halt "non-finite loss or weights by step $s1 (segment from step $s0); last.pt keeps the last finite checkpoint" ;;
  *)
    halt "the trainer ended without a status (exit code $rc): a crash, a kill or a failed start" ;;
esac
echo "ended $rc $st" > $R/runner.state
true
