#!/bin/bash
# watch_cs13.sh NEXT_DEADLINE: the watcher on the pod, detached:
#   setsid nohup /workspace/cascade_w5/watch_cs13.sh NEXT_DEADLINE >> /workspace/cascade_w5/logs/watch_cs13.out 2>&1 < /dev/null &
# It waits for the end of the running segment of pod_cs13.sh and starts one more segment if, and only if, the trainer
# paused: runner.state "ended 3 paused", status.json "paused", HANDOVER naming last.pt's sha256, no DONE or ATTENTION,
# the runner and the trainer gone, and at least 20 minutes before NEXT_DEADLINE. The next segment is
#   setsid nohup /workspace/cascade_w5/pod_cs13.sh NEXT_DEADLINE >> /workspace/cascade_w5/logs/runner.out 2>&1 < /dev/null &
# which resumes from last.pt with the same arguments and --max-minutes up to NEXT_DEADLINE (pod_cs13.sh's own checks
# apply). Every other end is left as it is: ATTENTION (the pull loop's FAILED), DONE, or a pause too close to
# NEXT_DEADLINE (the pull loop's HANDOVER). It starts at most one segment and then ends.
# watcher.state: "armed ...", then "started SEG ..." or "declined ..."; watcher.pid; one line per event on stdout.
# WATCH_DRY=DIR: one dry pass on copies in DIR (runner.state, runner.pid, trainer.pid and ckpt/CS13/ with status.json,
# HANDOVER, last.pt, segments.log, DONE, ATTENTION): prints the decision and the command; writes and starts nothing.
set -uo pipefail
NEXT=${1:?NEXT_DEADLINE (epoch seconds) missing}
R=/workspace/cascade_w5; X=${WATCH_DRY:-$R}          # X: where the run's records are read (DIR for a dry pass)
D=$X/ckpt/CS13

log() { echo "$(date -Is) $*"; }
put() {                                          # the state, also in watcher.state (not in a dry pass)
  log "$*"
  if [ -z "${WATCH_DRY:-}" ]; then echo "$*" > $R/watcher.state.tmp && mv $R/watcher.state.tmp $R/watcher.state; fi
}
sfield() {                                       # a field of status.json (as in pod_cs13.sh)
  if [ -f $D/status.json ]; then sed -nE "s/.*\"$1\": \"?([A-Za-z0-9_.:+-]+)\"?.*/\1/p" $D/status.json; fi
}
alive() {                                        # the pid in file $1 runs pod_cs13.sh or the trainer
  local p; p=$(cat "$1" 2>/dev/null)
  [ -n "$p" ] && grep -qa -e pod_cs13.sh -e surface.surface_train /proc/$p/cmdline 2>/dev/null
}
decide() {                                       # wait, start or "decline: REASON", from the records in $X
  local rs st hs ls mm
  if [ -f $D/DONE ]; then echo "decline: DONE exists (the run has ended)"; return; fi
  if [ -f $D/ATTENTION ]; then echo "decline: ATTENTION exists ($(sed -n 2p $D/ATTENTION | cut -c1-150))"; return; fi
  rs=$(cat $X/runner.state 2>/dev/null)
  if [ "$rs" = running ]; then
    if alive $X/runner.pid; then echo wait; else echo "decline: runner.state says running but the runner is gone"; fi
    return
  fi
  if [ "$rs" != "ended 3 paused" ]; then echo "decline: runner.state '$rs' is not the end of a pause"; return; fi
  if alive $X/runner.pid || alive $X/trainer.pid; then echo wait; return; fi     # the runner is finishing
  st=$(sfield status)
  if [ "$st" != paused ]; then echo "decline: status.json says '${st:-missing}', not paused"; return; fi
  if [ ! -f $D/HANDOVER ] || [ ! -f $D/last.pt ]; then echo "decline: HANDOVER or last.pt is missing"; return; fi
  hs=$(awk '$2 == "last.pt" {print $1}' $D/HANDOVER); ls=$(sha256sum $D/last.pt | cut -c1-64)
  if [ -z "$hs" ] || [ "$hs" != "$ls" ]; then echo "decline: last.pt's sha256 $ls is not HANDOVER's ${hs:-none}"; return; fi
  mm=$(( (NEXT - $(date +%s)) / 60 ))
  if [ $mm -lt 20 ]; then echo "decline: only $mm minutes before the next deadline"; return; fi
  echo start
}

n=$(( $(grep -c ' start: ' $D/segments.log 2>/dev/null || true) + 1 ))      # the next segment, as pod_cs13.sh counts
if [ -n "${WATCH_DRY:-}" ]; then
  d=$(decide)
  log "dry pass on $X: $d"
  if [ "$d" = start ]; then
    log "would run: setsid nohup $R/pod_cs13.sh $NEXT >> $R/logs/runner.out 2>&1 < /dev/null &"
    log "segment pod$n from step $(sfield step), MM $(( (NEXT - $(date +%s)) / 60 )) now (deadline $(date -d @$NEXT -Is)), --resume"
  fi
  exit 0
fi

[ "$NEXT" -gt $(( $(date +%s) + 1200 )) ] || { log "NEXT_DEADLINE $NEXT is less than 20 minutes away"; exit 1; }
exec 8> $R/watcher.lock
flock -n 8 || { log "another watcher holds $R/watcher.lock"; exit 1; }
echo $$ > $R/watcher.pid
put "armed $(date -Is): after a pause, the next segment resumes with the deadline $(date -d @$NEXT -Is)"
while true; do
  d=$(decide)
  case "$d" in
    wait) sleep 20 ;;
    start) break ;;
    *) put "declined $(date -Is): ${d#decline: }"; exit 0 ;;
  esac
done
n=$(( $(grep -c ' start: ' $D/segments.log 2>/dev/null || true) + 1 ))
s=$(sfield step)
log "paused at step $s: $(sed -n 1p $D/HANDOVER | cut -c1-200)"
setsid nohup $R/pod_cs13.sh $NEXT >> $R/logs/runner.out 2>&1 < /dev/null 8>&- &
put "started pod$n $(date -Is): pod_cs13.sh $NEXT (pid $!) resumes from step $s"
for i in $(seq 24); do                           # the runner's start, as segments.log and runner.state show it
  sleep 5
  if [ -f $D/ATTENTION ]; then log "pod$n: ATTENTION: $(sed -n 2p $D/ATTENTION | cut -c1-200)"; exit 0; fi
  if [ "$(cat $R/runner.state 2>/dev/null)" = running ] && grep -q " pod$n start: " $D/segments.log; then
    log "pod$n runs: $(grep " pod$n start: " $D/segments.log | tail -1 | cut -c1-250)"; exit 0
  fi
done
log "pod$n: runner.state is '$(cat $R/runner.state 2>/dev/null)' 2 minutes after the start"
