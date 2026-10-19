#!/bin/bash
# ARB dry run on the pod: 2 chunks per p (the first two of each block, ARB's seeds), the whole chain (read, seed check,
# score export) and the analysis on these rows, into out/dry. Nothing from it is reported as a result.
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
O=$W/out/dry
mkdir -p $O $W/out/logs
exec > >(tee -a $W/out/logs/dry.log) 2>&1
for P in 0.002 0.003; do bash $CODE/pod_read.sh $P 2 $O/read || { echo "DRY_FAILED at $P"; exit 1; }; done
cd $R/ising_compare
$PY $CODE/arb_analysis.py --read $O/read --lock $LOCK --out $O/analysis || { echo "DRY_FAILED analysis"; exit 1; }
echo "DRY_OK $(date -Is)"
