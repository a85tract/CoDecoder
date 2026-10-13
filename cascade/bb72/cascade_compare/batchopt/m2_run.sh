#!/bin/bash
# Milestone 2 (batchopt/M2.md): m2.py's steps, each in a fresh process, in the order of STEPS; the first step that exits
# non-zero (or an unknown step, or a missing setting) ends the run with a non-zero exit (review finding 1). The
# environment comes from m2.sbatch or m2_pod_run.sh (PYTHONPATH, the caches, RUN, PYBIN, O, REFS).
#   STEPS   any of: dev refs screen ptx nets pipe tailnets tail report gate (in order; gate: the run exits 1 unless
#           the screen and the tail checks in O passed, and each of them is there if this run's STEPS made it)
#   DEV_ARGS, SCREEN_ARGS, PTX_ARGS, NETS_ARGS, PIPE_ARGS, TAIL_ARGS, REPORT_ARGS   extra arguments of those steps;
#   NETS_EXTRA, PIPE_EXTRA   arguments the driver appends to every timed process
#   REPS    fresh processes per timed configuration (default 3); TAIL_B lever 2's small batch; TAIL_NETS, TAIL_PATHS,
#           TAIL_SETTINGS the tail steps' networks, paths and settings
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd)/m2.py
RUN=${RUN:-}; O=${O:?output directory}; PYBIN=${PYBIN:?the venv python}; REFS=${REFS:?refs directory}
PY="$RUN $PYBIN"
mkdir -p "$O"
log() { echo "== $* $(date +%T)"; }
step() {  # step NAME CMD...: a non-zero exit ends the run with it
  local n=$1; shift
  log "$n start"; local t0; t0=$(date +%s)
  "$@"; local rc=$?
  log "$n exit=$rc $(( $(date +%s) - t0 )) s"
  [ $rc -eq 0 ] || { log "run stopped: $n failed"; exit $rc; }
}
for s in ${STEPS:?steps}; do
  case $s in
    dev) step dev $PY $P dev --out $O ${DEV_ARGS:-} ;;
    refs) step refs $PY $P refs --out $REFS ;;
    screen) step screen $PY $P screen --refs $REFS --out $O ${SCREEN_ARGS:-} ;;
    ptx) step ptx $PY $P ptx --out $O ${PTX_ARGS:-} ;;
    nets) step nets $PY $P drive --what nets --rows $O/nets.jsonl --reps ${REPS:-3} --extra "${NETS_EXTRA:-}" \
            ${NETS_ARGS:-} ;;
    pipe) step pipe $PY $P drive --what pipe --rows $O/pipe.jsonl --reps ${REPS:-3} --tail-b ${TAIL_B:?} \
            --extra "${PIPE_EXTRA:-}" ${PIPE_ARGS:-} ;;
    tailnets) for t in ${TAIL_NETS:-C14 C14p}; do
            step "tailnets $t" $PY $P nets --net $t --paths ${TAIL_PATHS:?} --out $O/tailnets.jsonl; done ;;
    tail) for x in ${TAIL_SETTINGS:-uniform depspam_fm}; do
            step "tail $x" $PY $P tail --setting $x --tail-b ${TAIL_B:?} --out $O ${TAIL_ARGS:-}; done ;;
    report) step report $PY $P report --dirs ${REPORT_DIRS:-$O} --out $O ${REPORT_ARGS:-} ;;
    gate) need=""                    # what this run's own steps must have left in O
          case " $STEPS " in *" screen "*) need="--need-screen" ;; esac
          case " $STEPS " in *" tail "*) need="$need --need-tail $(echo ${TAIL_SETTINGS:-uniform depspam_fm} | tr ' ' ,)" ;;
          esac
          step gate $PY $P gate --out $O $need ;;
    *) log "unknown step $s"; log "run stopped: $s failed"; exit 2 ;;
  esac
done
ls -la $O | tail -n +2 | awk '{print $5, $9}' | tail -40
exit 0
