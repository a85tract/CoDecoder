#!/bin/bash
# The Linux host's fetcher (STAGE2_common C1, partial results): copies a session's saved records from the local save
# folder SAVE/<SESSION>/ (cfg_box.json save_to; harness.py copies each finished window there) to DEST/<SESSION>/ (a
# local path, or HOST:PATH over ssh) every 15 minutes while a batch runs, and once more at its end (--once). It reads
# only the save folder, never the tmpfs, and runs pinned to the harness's CPUs (the CCD without slots). Each fetch's
# start, end and progress (windows saved / windows planned) go to SAVE/fetch_<SESSION>.log, copied to DEST/fetch.log.
# The fetched records serve safekeeping and the progress count only.
#   bash fetch_box.sh SESSION DEST [--once] [--every 900] [--save ~/srt_box_io/out] [--cpus 8-15,24-31]
set -uo pipefail
S=$1; DEST=$2; shift 2
ONCE=0; EVERY=900; SAVE=$HOME/srt_box_io/out; CPUS=8-15,24-31
while [ $# -gt 0 ]; do
  case $1 in --once) ONCE=1 ;; --every) EVERY=$2; shift ;; --save) SAVE=$2; shift ;; --cpus) CPUS=$2; shift ;;
             *) echo "unknown option $1"; exit 2 ;; esac
  shift
done
LOG=$SAVE/fetch_$S.log
mkdir -p "$SAVE/$S"
case $DEST in *:*) ;; *) mkdir -p "$DEST/$S" ;; esac
while :; do
  t0=$(date -Is)
  if timeout 600 taskset -c "$CPUS" rsync -a --mkpath "$SAVE/$S/" "$DEST/$S/"; then
    n=$(find "$SAVE/$S" -name window.json | wc -l)
    planned=$(grep -o '"id": "w[0-9]*"' "$SAVE/$S/plan.json" 2>/dev/null | wc -l)
    echo "$t0 $(date -Is) ok $S: $n windows saved ($planned planned, repeats r* extra), $(du -sk "$SAVE/$S" | cut -f1) kB" >> "$LOG"
  else
    echo "$t0 $(date -Is) FAILED $S (destination unreachable; the records stay in $SAVE/$S)" >> "$LOG"
  fi
  timeout 120 taskset -c "$CPUS" rsync -a --mkpath "$LOG" "$DEST/fetch.log" 2>/dev/null
  tail -1 "$LOG"
  [ "$ONCE" = 1 ] && break
  sleep "$EVERY"
done
