#!/bin/bash
# The fetcher (STAGE2_common C1, partial results): pulls a session's saved records from the Windows CPU host's folder
# %USERPROFILE%\speedrt_out\<SESSION>\ (harness.py copies each finished window there) to DEST/<SESSION>/ on the Linux
# workstation, every 15 minutes while a batch runs, and once more at its end (--once). It reads only that Windows
# folder, with the host's own tar.exe over ssh, and never starts anything inside the WSL distro, so a fetch adds no
# block-device read in the VM. Each fetch's start, end and progress (windows saved / windows planned) go to
# DEST/fetch.log. The fetched records serve safekeeping and the progress count only.
#   WIN_SSH: the ssh command that runs a cmd.exe line on the Windows host (one hop: "ssh HOST"; through a jump host
#            whose own ssh holds the key: "ssh HOP ssh HOST"); the session's folder name has no spaces.
#   WIN_SSH="ssh HOST" bash fetch.sh SESSION DEST [--once] [--every 900]
set -uo pipefail
S=$1; DEST=$2; shift 2
ONCE=0; EVERY=900
while [ $# -gt 0 ]; do
  case $1 in --once) ONCE=1 ;; --every) EVERY=$2; shift ;; *) echo "unknown option $1"; exit 2 ;; esac
  shift
done
: "${WIN_SSH:?set WIN_SSH to the ssh command that reaches the Windows host}"
mkdir -p "$DEST"
while :; do
  t0=$(date -Is); tmp=$DEST/.fetch_$$.tgz
  if timeout 600 $WIN_SSH "tar.exe -C speedrt_out -czf - $S" > "$tmp" 2>/dev/null && gzip -t "$tmp" 2>/dev/null \
     && tar -xzf "$tmp" -C "$DEST"; then
    n=$(find "$DEST/$S" -name window.json | wc -l)
    planned=$(grep -o '"id": "w[0-9]*"' "$DEST/$S/plan.json" 2>/dev/null | wc -l)
    echo "$t0 $(date -Is) ok $S: $n windows saved ($planned planned, repeats r* extra), $(stat -c %s "$tmp") bytes" >> "$DEST/fetch.log"
  else
    echo "$t0 $(date -Is) FAILED $S (link or host down; the records stay in the Windows folder)" >> "$DEST/fetch.log"
  fi
  rm -f "$tmp"
  tail -1 "$DEST/fetch.log"
  [ "$ONCE" = 1 ] && break
  sleep "$EVERY"
done
