#!/bin/bash
# ARB's full run on the pod (PLAN.md steps 1-5): per p the read of D1's block size (1,100 and 1,700 chunks), its seed check
# and score export; the dry rows against the full read's first chunks; the analysis; the outputs' sha256 manifest.
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
O=$W/out/read
mkdir -p $O $W/out/logs
exec > >(tee -a $W/out/logs/run.log) 2>&1
st() { echo "$(date -Is) $*" | tee -a $W/out/status.txt; }
st "run start"
for P in 0.002 0.003; do
  N=$([ $P = 0.002 ] && echo 1100 || echo 1700)
  st "read $P start ($N chunks)"
  bash $CODE/pod_read.sh $P $N $O || { st "FAILED read $P (status $?)"; exit 1; }
  $PY $CODE/arb_drycompare.py --dry $W/out/dry/read --full $O --p $P || st "dry/full comparison at $P differs (see log)"
  st "read $P done"
done
cd $R/ising_compare
st "analysis start"
$PY $CODE/arb_analysis.py --read $O --lock $LOCK --out $W/out/analysis || { st "FAILED analysis"; exit 1; }
st "analysis done"
cd $W/out && find . -type f ! -name manifest.sha256 ! -name status.txt ! -name '*.log' -print0 | sort -z \
  | xargs -0 sha256sum > manifest.sha256
st "ALL_DONE"
