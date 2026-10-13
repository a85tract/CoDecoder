#!/bin/bash
# collect.sh CARD SRC... DEST: copies the output directories of the Cascade jobs of one card (drive.py's OUT of R1, R2, R3, R4;
# CARD = RTX for Alpine's RTX PRO 6000, H100 for the pod, whose directories were copied off the pod first: the folder that
# holds attempts.json) into the layout that analysis_cas.from_records reads: DEST/CARD/R1 ... DEST/CARD/R4. The round of a SRC is
# read from its attempts.json. cp only; a DEST/CARD/<round> that exists is never overwritten (the script stops), so every round
# of a card is copied once.
set -euo pipefail
[ $# -ge 3 ] || { echo "usage: collect.sh CARD SRC... DEST"; exit 2; }
CARD=$1; DEST=${!#}
case $CARD in RTX|H100) ;; *) echo "card '$CARD' is not RTX or H100"; exit 6 ;; esac
mkdir -p "$DEST/$CARD"
for SRC in "${@:2:$#-2}"; do
  [ -f "$SRC/attempts.json" ] || { echo "$SRC has no attempts.json"; exit 3; }
  R=$(grep -m1 -E '^ "round": "' "$SRC/attempts.json" | sed -E 's/.*"round": "([a-z0-9]+)".*/\1/' | tr a-z A-Z)
  case $R in R1|R2|R3|R4) ;; *) echo "$SRC: round '$R' is not one of r1 r2 r3 r4"; exit 4 ;; esac
  [ ! -e "$DEST/$CARD/$R" ] || { echo "$DEST/$CARD/$R exists: stop"; exit 5; }
  cp -r "$SRC" "$DEST/$CARD/$R"
  echo "$SRC -> $DEST/$CARD/$R ($(find "$DEST/$CARD/$R" -type f | wc -l) files)"
done
