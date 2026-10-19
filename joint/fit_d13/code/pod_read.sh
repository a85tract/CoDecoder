#!/bin/bash
# One ARB-d13 read at P over chunks 0 .. N-1 into O: the read through the shim (trace on; continued with --resume if
# a journal exists and the read is not done), then the seed check through the shim (trace off; --old-all when
# OLD_ALL=1). Exit 0 when both pass.   bash pod_read.sh P N O BPATH
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
P=$1; N=$2; O=$3; BPATH=$4
mkdir -p $O/trace_$P
cd $W
echo "== read $P chunks 0-$N b-path $BPATH -> $O start $(date -Is)"
if [ ! -f $O/done_$P ]; then
  RES=""; [ -f $O/p$(printf %.4f $P)/progress.jsonl ] && RES=--resume
  ARB2_TRACE=$O/trace_$P $PY $CODE/arb2_shim.py $CODE/arb2_read.py --p $P --chunks 0-$N --out $O --b-path $BPATH $RES \
    && touch $O/done_$P || { echo "== read $P FAILED $(date -Is)"; exit 1; }
fi
echo "== seedcheck $P $(date -Is)"
OLD=""; [ "${OLD_ALL:-0}" = 1 ] && OLD="--old-all --procs ${OLD_PROCS:-8}"
$PY $CODE/arb2_shim.py $CODE/arb2_seedcheck.py --read $O --p $P --release $REL $OLD || { echo "== seedcheck $P FAILED"; exit 2; }
echo "== read $P done $(date -Is)"
