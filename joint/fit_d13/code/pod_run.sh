#!/bin/bash
# ARB-d13's full run on the pod (PLAN.md steps 2-5): per p the read (400 and 3,000 chunks) and its seed check, the dry
# rows against the full read's; the analysis; the outputs' sha256 manifest.   bash pod_run.sh BPATH
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
BPATH=${1:?b-path (graph or eager)}
O=$W/out/read
mkdir -p $O $W/out/logs
exec > >(tee -a $W/out/logs/run.log) 2>&1
st() { echo "$(date -Is) $*" | tee -a $W/out/status.txt; }
st "run start (b-path $BPATH, OLD_ALL=${OLD_ALL:-0})"
for P in 0.006 0.0035; do
  N=$([ $P = 0.006 ] && echo 400 || echo 3000)
  st "read $P start ($N chunks)"
  bash $CODE/pod_read.sh $P $N $O $BPATH || { st "FAILED read $P (status $?)"; exit 1; }
  $PY $CODE/arb2_drycompare.py --dry $W/out/dry/read --full $O --p $P || st "dry/full comparison at $P differs (see log)"
  st "read $P done"
done
cd $W
st "analysis start"
$PY $CODE/arb2_analysis.py --read $O --lock $LOCK --out $W/out/analysis || { st "FAILED analysis"; exit 1; }
st "analysis done"
cd $W/out && find . -type f ! -name manifest.sha256 ! -name status.txt ! -name '*.log' -print0 | sort -z \
  | xargs -0 sha256sum > manifest.sha256
st "ALL_DONE"
