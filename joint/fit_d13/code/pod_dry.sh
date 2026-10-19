#!/bin/bash
# ARB-d13 dry run on the pod: chunks 0-1 of each p (ARB's seeds), each chunk its own part, Cascade through the CUDA graph
# of its eager module with every dry shot also run by direct eager calls (bitwise check, and the eager rate); the seed
# check on these rows (both chunks sampled: digests against the comparison's, the re-test's and the development block's
# chunks 0 and 1, and the cluster's digests of the comparison's chunks); the analysis on these rows. Never reported.
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
O=$W/out/dry
mkdir -p $O/read $W/out/logs
exec > >(tee -a $W/out/logs/dry.log) 2>&1
cd $W
for P in 0.006 0.0035; do
  mkdir -p $O/read/trace_$P
  echo "== dry read $P $(date -Is)"
  ARB2_TRACE=$O/read/trace_$P $PY $CODE/arb2_shim.py $CODE/arb2_read.py --p $P --chunks 0-2 --out $O/read --part 1 \
    --b-path graph --check-eager 10000 || { echo "DRY_FAILED read $P"; exit 1; }
  echo "== dry seedcheck $P $(date -Is)"
  $PY $CODE/arb2_shim.py $CODE/arb2_seedcheck.py --read $O/read --p $P --release $REL --sample 2 \
    || { echo "DRY_FAILED seedcheck $P"; exit 1; }
done
$PY $CODE/arb2_analysis.py --read $O/read --lock $LOCK --out $O/analysis --allow-partial || { echo "DRY_FAILED analysis"; exit 1; }
echo "DRY_OK $(date -Is)"
