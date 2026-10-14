#!/bin/bash
# collect.sh SRC... DEST: puts the BB stage's sessions into the layout that speed_rt/bb/analysis_bb.py reads (cp only, no network).
# Each SRC is recognised by its files and may be given once:
#   DEST/gpu/        the GPU pod's pod_bb.sh output directory (after pod2alpine): session.json (its "role": "gpu"), rows.csv, attempts.json,
#                    with watch/, logs/, prescreen/, pod_env.txt, preds.sha256, gpu_bposd_detail.csv, nvidia_smi_q_timed.txt and freeze_*.txt
#                    beside them. preds/ (the routing reads, listed in preds.sha256) is not copied: it is the hand-off to the CPU worker.
#   DEST/cpu/        the CPU worker's winserver session: rows.csv, accepted.json, session.json, grace.json, host.json (and whatever else the
#                    worker publishes: windows, reference segments, logs); recognised by accepted.json and grace.json beside rows.csv
#   DEST/draw/       a draw.sh output directory (manifest_bb72.csv, seedcheck.json): everything but shots/
# A part is given once: when a session was run again (the failure rule's whole-session rerun), pass only the one that counts; the
# other keeps its own directory outside DEST and is published beside the records. DEST is created; a part that exists is refused.
# DEST/COLLECTED.sha256 lists every file copied. Absolute directory prefixes in the copied text records are dropped first (see the code).
set -euo pipefail
[ $# -ge 2 ] || { sed -n 2,13p "$0"; exit 2; }
DEST=${!#}; mkdir -p "$DEST"; DEST=$(cd "$DEST" && pwd)
for SRC in "${@:1:$#-1}"; do
  [ -d "$SRC" ] || { echo "$SRC: not a directory"; exit 2; }
  if [ -f "$SRC/session.json" ] && [ -f "$SRC/rows.csv" ] && [ -f "$SRC/attempts.json" ]; then
    part=$(grep -o '"role": "[a-z0-9]*"' "$SRC/session.json" | head -1 | cut -d'"' -f4)
    case "$part" in gpu) ;; *) echo "$SRC: session.json names no GPU pod role"; exit 2 ;; esac
  elif [ -f "$SRC/rows.csv" ] && [ -f "$SRC/accepted.json" ] && [ -f "$SRC/grace.json" ]; then part=cpu
  elif [ -f "$SRC/manifest_bb72.csv" ] && [ -f "$SRC/seedcheck.json" ]; then part=draw
  else echo "$SRC: not a pod_bb.sh or draw.sh output"; exit 2; fi
  [ ! -e "$DEST/$part" ] || { echo "$DEST/$part exists"; exit 2; }
  mkdir -p "$DEST/$part"
  for f in "$SRC"/* "$SRC"/.[!.]*; do
    [ -e "$f" ] || continue
    case "$part:$(basename "$f")" in draw:shots|*:preds|*:.dry_*) continue ;; esac
    cp -r "$f" "$DEST/$part/"
  done
  echo "$SRC -> $DEST/$part"
done
# Text records name files relative (publication then changes no hash): an absolute directory prefix (the pod's /workspace/..., the cluster's /scratch/...) in a
# copied log, json, csv or txt is dropped and the file name stays; files without one are not touched. COLLECTED.sha256 lists the files as copied here.
find "$DEST" -type f \( -name '*.log' -o -name '*.json' -o -name '*.csv' -o -name '*.txt' \) -print0 | while IFS= read -r -d '' f; do
  if grep -qE '(^|[^[:alnum:]_./-])/(workspace|scratch|home|root|tmp|curc)/' "$f"; then
    sed -i -E 's#(^|[^[:alnum:]_./-])/(workspace|scratch|home|root|tmp|curc)/([^[:space:]"'"'"',:;)]*/)?#\1#g' "$f"
  fi
done
( cd "$DEST" && find . -type f ! -name COLLECTED.sha256 | sort | xargs sha256sum > COLLECTED.sha256 )
echo "collected $(wc -l < "$DEST/COLLECTED.sha256") files into $DEST"
