#!/bin/bash
#SBATCH -J rt_d13_replay
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --nice=1000
#SBATCH -t 01:30:00
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# Replay checks of the d = 13 shim (RETEST.md section d = 13), CPU only:
#   1. patch off: sv_route confirm and report (d13_analysis.sh, through the shim) on the original e5 / e6 reads
#      ($R/eval/conf) into $OUT/route_orig, compared with the original e8's outputs in $R/eval;
#   2. patch on, the dry seeds: the main process, surface_data's spawned sampling pool and sv_read's spawned M workers
#      draw the intended seeds; the resume path's draw; 3. every other kind unchanged, kind 11 outside the sets refused;
#   the same probe with the patch off (the pinned function object; kind 11 = its formula);
#   negative controls: the probe with the patch in the main process only and with no patch (RT_SEEDS set) must fail
#   where the workers or the main process draw other seeds (the checks are not vacuous).
# RT = the tree with retest/ (default: the dry tree); results in OUT (default .../replay/d13).
set -euo pipefail
RT=${RT:-<cluster-scratch>/retest/code/dry}; export RT
OUT=${OUT:-<cluster-scratch>/retest/replay/d13}
DRYD=<cluster-scratch>/retest/dry
source $RT/retest/jobs/d13_env.sh
PY="apptainer exec -B <cluster-scratch-root> $I/pyfull.sif $V/bin/python"
X=$RTD/replay/d13/d13_replay.py
[ ! -e $OUT/route_orig ] || { echo "$OUT/route_orig exists: the analysis writes once; set OUT to a new directory"; exit 1; }
mkdir -p $OUT; st=0
echo "== 2/3: patch ON, dry seeds"
RT_BEACON=$DRYD/beacon_dry.json RT_SEEDS=$DRYD/seeds_dry.json PYTHONPATH=$PYTHONPATH:$RTD/replay/d13 \
  $PY $SHIM d13_replay seeds $OUT/replay_seeds_on.json > $OUT/seeds_on.log 2>&1 || st=1
tail -n 3 $OUT/seeds_on.log | cut -c1-800
echo "== 3: patch OFF"
env -u RT_SEEDS -u RT_BEACON PYTHONPATH=$PYTHONPATH:$RTD/replay/d13 \
  $PY $SHIM d13_replay seeds $OUT/replay_seeds_off.json > $OUT/seeds_off.log 2>&1 || st=1
tail -n 3 $OUT/seeds_off.log | cut -c1-800
echo "== controls: patch in the main process only / no patch"
for how in main-only none; do
  RT_BEACON=$DRYD/beacon_dry.json RT_SEEDS=$DRYD/seeds_dry.json PYTHONPATH=$PYTHONPATH:$RTD/replay/d13 \
    $PY $X seeds-control $OUT/replay_seeds_control_$how.json $how > $OUT/seeds_control_$how.log 2>&1 || st=1
  tail -n 1 $OUT/seeds_control_$how.log | cut -c1-800
done
echo "== 1: analysis on the original reads, patch OFF"
env -u RT_SEEDS -u RT_BEACON RT_OUT=$OUT RT_CONF=$R/eval/conf RT_ROUTE=$OUT/route_orig bash $RTD/jobs/d13_analysis.sh || st=1
$PY $X compare $R/eval $OUT/route_orig $OUT/replay_analysis.json || st=1
date; echo "JOBDONE st=$st"; exit $st
