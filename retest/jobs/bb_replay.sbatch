#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 8 --mem=32G -t 01:30:00 -J rt_bbreplay
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# Replay checks of the BB shim (RETEST.md section 6), patch off: regenerate the original test chunks from the
# published beacons, unpack the published per-shot outputs, derive the CPU-term rows and parameter copies, run the
# three frozen analyses through the shim and compare with the published results. T = the tree (bb72/, gross_stage4/,
# retest/), R = output dir. Then the patch-on seed checks with the seeds of SEEDS_DIR (dry: a past round).
set -euo pipefail
T=${T:-<cluster-scratch>/retest/code/dry}
R=${R:-<cluster-scratch>/retest/replay/bb}
SEEDS_DIR=${SEEDS_DIR:-<cluster-scratch>/retest/dry}
PY=<cluster-scratch>/retest/bin/pycpu
export QND_RELAY=${QND_RELAY:-<cluster-scratch>/bb/relay}   # the relay clone at d185194 (bb/code.py)
S=$T/retest/shim/rt_bb.py; X=$T/retest/replay/bb_replay.py
unset RT_SEEDS RT_BEACON RT_MANIFEST RT_STAGE RT_CELLS
rm -rf $R && mkdir -p $R && export BB_SHOTS=$R/shots BB_PREDS=$R/preds
st=0; chk() { "$@" || { echo "FAILED: $*"; st=1; }; }

cd $T/bb/bb72/ising_compare && export PYTHONPATH=.:../qldpc_neural
$PY $S bb.shots --code bb72 --p 0.003 --split test --chunks 0-19 --rows $R/rows_bb72.csv > /dev/null
$PY $S bb.shots --code bb72 --p 0.002 --split test --chunks 0-29 --rows $R/rows_bb72.csv > /dev/null
export RT_MANIFEST=$R/rows_bb72.csv
chk $PY $X shots $R/shots_bb72.json
$PY $X unpack $R/unpack_bb72.json
export RT_STAGE=$R/stage3 RT_CELLS=$R/stage3/cells.json
$PY $S --prepare $R/stage3
$PY $S bb.stage3 analyze --out $R/stage3 > $R/analysis.log
$PY $S bb.stage3 analyze --amend $R/stage3/amendment_S.json --out $R/stage3 > $R/analysis_amendment_S.log
O=data/bb/stage3
for f in results.json results_amendment_S.json RESULTS_stage3.md RESULTS_stage3_amendment_S.md; do
  chk $PY $X compare $O/$f $R/stage3/$f $R/compare_${f%.*}.json; done
chk $PY $X rows $O/rows_cpu.csv $R/stage3/rows_cpu_retest.csv $R/rows_cpu_equal.json
chk $PY $X rows $O/rows_cpu_S.csv $R/stage3/rows_cpu_S_retest.csv $R/rows_cpu_S_equal.json
unset RT_STAGE RT_CELLS RT_MANIFEST

cd $T/bb/gross_stage4/ising_compare
$PY $S bb.shots --code bb144 --p 0.004 --split test4 --chunks 0-14 --rows $R/rows_gross.csv > /dev/null
export RT_MANIFEST=$R/rows_gross.csv
chk $PY $X shots $R/shots_gross.json
$PY $X unpack $R/unpack_gross.json
export RT_STAGE=$R/stage4 RT_CELLS=$R/stage4/cell.json
$PY $S --prepare $R/stage4
$PY $S bb.stage4_gross analyze --out $R/stage4 > $R/analysis_gross.log
O=data/bb/stage4_gross
for f in results.json RESULTS_stage4_gross.md; do chk $PY $X compare $O/$f $R/stage4/$f $R/compare_gross_${f%.*}.json; done
chk $PY $X rows $O/rows_cpu.csv $R/stage4/rows_cpu_retest.csv $R/rows_cpu_gross_equal.json
unset RT_STAGE RT_CELLS RT_MANIFEST

export RT_BEACON=$SEEDS_DIR/beacon_dry.json RT_SEEDS=$SEEDS_DIR/seeds_dry.json
cd $T/bb/bb72/ising_compare && chk $PY $T/retest/replay/bb_seedcheck.py $R/seedcheck_bb72.json
cd $T/bb/gross_stage4/ising_compare && chk $PY $T/retest/replay/bb_seedcheck.py $R/seedcheck_gross.json
echo "replay exit $st"; exit $st
