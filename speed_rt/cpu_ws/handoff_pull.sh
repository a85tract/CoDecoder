#!/bin/bash
# The Linux host's hand-off (STAGE2_common C2): copies the stage's hand-off units from the folder the GPU passes fill
# (SRC: a local path, or HOST:PATH over ssh; one folder per unit, each with SHA256SUMS and its marker DONE, or
# DRAW_DONE for bb/draw) into the harness's local hand-off folder DEST (cfg_box.json handoff_from). Each round lists
# the markers first, then copies every file but the markers, then the markers listed, so a marker never arrives before
# its unit's files. harness.py run copies a unit into the tmpfs and checks its checksums at a window boundary. Runs
# pinned to the harness's CPUs; loops every EVERY seconds (default 60) until --once. Log: DEST/../handoff_pull.log.
#   bash handoff_pull.sh SRC DEST [--once] [--every 60] [--cpus 8-15,24-31]
set -uo pipefail
SRC=$1; DEST=$2; shift 2
ONCE=0; EVERY=60; CPUS=8-15,24-31
while [ $# -gt 0 ]; do
  case $1 in --once) ONCE=1 ;; --every) EVERY=$2; shift ;; --cpus) CPUS=$2; shift ;; *) echo "unknown option $1"; exit 2 ;; esac
  shift
done
mkdir -p "$DEST"
LOG=$(dirname "$DEST")/handoff_pull.log
R="timeout 1800 taskset -c $CPUS rsync"
while :; do
  t0=$(date -Is); list=$(mktemp)
  $R -r --list-only --include='*/' --include=DONE --include='*_DONE' --exclude='*' "$SRC/" 2>/dev/null \
    | awk '$1 !~ /^d/ {print $NF}' > "$list"
  if $R -a --exclude=DONE --exclude='*_DONE' "$SRC/" "$DEST/" && { [ ! -s "$list" ] || $R -a --files-from="$list" "$SRC/" "$DEST/"; }; then
    echo "$t0 $(date -Is) ok: markers $(tr '\n' ' ' < "$list")" >> "$LOG"
  else
    echo "$t0 $(date -Is) FAILED (source unreachable or a copy failed; tried again in $EVERY s)" >> "$LOG"
  fi
  rm -f "$list"
  tail -1 "$LOG"
  [ "$ONCE" = 1 ] && break
  sleep "$EVERY"
done
