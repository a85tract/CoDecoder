#!/bin/bash
# One ARB read at P over N chunks (from c0) into O: the read through the shim (trace on), the seed check (through the
# shim, trace off), the score export. Exit status: 0 when all three pass.
#   bash pod_read.sh P N O
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
P=$1; N=$2; O=$3
mkdir -p $O/trace_$P
cd $R/ising_compare
echo "== read $P chunks $C0:$((C0 + N)) -> $O start $(date -Is)"
if [ ! -f $O/done_$P ]; then
  ARB_TRACE=$O/trace_$P $PY $CODE/arb_shim.py $CODE/arb_read.py --family depspam_fm --ckpt $WTS/C14p.pt,$WTS/S55p_deep.pt \
    --prec bf16 --kind 14 --p $P --chunks $C0:$((C0 + N)) --procs 8 --save $O --save-det --out $O/rows_$P.json \
    && touch $O/done_$P || { echo "== read $P FAILED $(date -Is)"; exit 1; }
fi
echo "== seedcheck $P $(date -Is)"
$PY $CODE/arb_shim.py $CODE/arb_seedcheck.py --read $O --p $P --release $W/release || { echo "== seedcheck $P FAILED"; exit 2; }
echo "== scores $P $(date -Is)"
$PY $CODE/arb_scores.py --read $O --p $P --ckpt $WTS/S55p_deep.pt --lock $LOCK || { echo "== scores $P FAILED"; exit 3; }
echo "== read $P done $(date -Is)"
