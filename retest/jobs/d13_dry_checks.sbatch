#!/bin/bash
#SBATCH -J rt_d13_drychecks
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH --nice=1000
#SBATCH -t 01:00:00
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# Checks on the dry reads of d13_e5.sbatch / d13_e6.sbatch (dry seeds, RT_DRY_CHUNKS), CPU only. DRY_OUT = the RT_OUT
# of the dry reads (default .../retest/dry/d13_out); DRY_VERIFY = more read directories to verify (space separated).
#   1. verify: the chunk digests the reads stored (obs_*.npz) against a direct Stim sample of the dry seeds;
#   2. confirm and report as they run after the beacon, on the dry read: they refuse (the parts cover 1 chunk, the lock
#      asks for 100 and 3,000);
#   0. the refusals of d13_read.sh: RT_DRY_CHUNKS without dry seeds, dry seeds without RT_DRY_CHUNKS;
#   3. confirm on the dry read with RT_DRY_CONFIRM=0-1 (p = 0.006 only, 20 hypotheses), then report, which refuses
#      (the C7 rows of both p) -- the statistics, Holm and label steps run on a read written through the shim.
set -euo pipefail
RT=${RT:-<cluster-scratch>/retest/code/dry}; export RT
OUT=${OUT:-<cluster-scratch>/retest/replay/d13}
DRYD=<cluster-scratch>/retest/dry
DRY_OUT=${DRY_OUT:-$DRYD/d13_out}
source $RT/retest/jobs/d13_env.sh
PY="apptainer exec -B <cluster-scratch-root> $I/pyfull.sif $V/bin/python"
export RT_BEACON=$DRYD/beacon_dry.json RT_SEEDS=$DRYD/seeds_dry.json
mkdir -p $OUT; st=0
echo "== 0: refusals of d13_read.sh (each must exit 1 before anything runs)"
mkdir -p $OUT/nondry
$PY -c "import json; b = json.load(open('$RT_BEACON')); b.pop('dry'); json.dump(b, open('$OUT/nondry/beacon.json', 'w'))"
refusals=0
for scn in "nondry-with-RT_DRY_CHUNKS" "dry-without-RT_DRY_CHUNKS"; do
  rc=0
  if [ $scn = nondry-with-RT_DRY_CHUNKS ]; then
    RT_BEACON=$OUT/nondry/beacon.json RT_OUT=$OUT/nondry RT_DRY_CHUNKS=0-1 PS=0.006 bash $RTD/jobs/d13_e5.sbatch > $OUT/refusal_$scn.log 2>&1 || rc=$?
  else
    env -u RT_DRY_CHUNKS RT_OUT=$OUT/nondry PS=0.006 bash $RTD/jobs/d13_e6.sbatch > $OUT/refusal_$scn.log 2>&1 || rc=$?
  fi
  echo "$scn: rc $rc: $(tail -n 1 $OUT/refusal_$scn.log | cut -c1-200)"
  [ $rc = 1 ] || { st=1; refusals=1; }
done
for d in $DRY_OUT ${DRY_VERIFY:-}; do
  n=$(echo $d | tr '/' '_' | tail -c 20)
  echo "== 1: verify the digests of $d"
  PYTHONPATH=$PYTHONPATH:$RTD/replay/d13 $PY $SHIM d13_replay verify $d/d13/conf $OUT/dry_verify$n.json \
    > $OUT/dry_verify$n.log 2>&1 || st=1
  tail -n 2 $OUT/dry_verify$n.log | cut -c1-600
done
echo "== 2: confirm and report on the dry read, as after the beacon"
rc=0; RT_OUT=$OUT RT_CONF=$DRY_OUT/d13/conf RT_ROUTE=$OUT/route_dry bash $RTD/jobs/d13_analysis.sh > $OUT/dry_analysis.log 2>&1 || rc=$?
echo "rc $rc (nonzero expected)"; grep -v -E "UserWarning|warnings.warn" $OUT/dry_analysis.log | tail -n 4 | cut -c1-600
echo "== 3: confirm with RT_DRY_CONFIRM=0-1, then report"
rc=0; RT_DRY_CONFIRM=0-1 RT_OUT=$OUT RT_CONF=$DRY_OUT/d13/conf RT_ROUTE=$OUT/route_dry_confirm \
  bash $RTD/jobs/d13_analysis.sh > $OUT/dry_confirm.log 2>&1 || rc=$?
echo "rc $rc"; grep -v -E "UserWarning|warnings.warn" $OUT/dry_confirm.log | tail -n 12 | cut -c1-700
date; echo "JOBDONE st=$st"; exit $st
