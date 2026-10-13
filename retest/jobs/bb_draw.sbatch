#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 4 --mem=16G -t 00:45:00 -J rt_bbdraw
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# The BB test sets (RETEST.md section 2): host check (the original test chunks regenerated on this node from the
# published beacons equal the published arrays, as stage 3 / 4's host checks), then BB72-3 (bb72 p 0.003 test 0-19),
# BB72-2 (bb72 p 0.002 test 0-29) and gross (bb144 p 0.004 test4 0-14) from the re-test's seeds, then the seed checks.
set -euo pipefail
source ${T:-<cluster-scratch>/retest/code/freeze}/retest/jobs/bb_env.sh
H=$OUT/hostcheck; mkdir -p $H
( unset RT_SEEDS RT_BEACON RT_DRY_SHOTS; export BB_SHOTS=$H/shots RT_MANIFEST=$H/rows_bb72.csv
  cd $T/bb/bb72/ising_compare && export PYTHONPATH=.:../qldpc_neural
  $PYC $S bb.shots --code bb72 --p 0.003 --split test --chunks 0-19 --rows $H/rows_bb72.csv > /dev/null
  $PYC $S bb.shots --code bb72 --p 0.002 --split test --chunks 0-29 --rows $H/rows_bb72.csv > /dev/null
  $PYC $X shots $OUT/hostcheck_bb72.json
  export RT_MANIFEST=$H/rows_gross.csv; cd $T/bb/gross_stage4/ising_compare
  $PYC $S bb.shots --code bb144 --p 0.004 --split test4 --chunks 0-14 --rows $H/rows_gross.csv > /dev/null
  $PYC $X shots $OUT/hostcheck_gross.json ) || { echo HOSTCHECK_FAIL; exit 1; }
rm -rf $H/shots
bb72
$PYC $S bb.shots --code bb72 --p 0.003 --split test --chunks 0-19 --rows $OUT/manifest_bb72.csv > /dev/null
$PYC $S bb.shots --code bb72 --p 0.002 --split test --chunks 0-29 --rows $OUT/manifest_bb72.csv > /dev/null
$PYC $T/retest/replay/bb_seedcheck.py $OUT/seedcheck_bb72.json > /dev/null
gross
$PYC $S bb.shots --code bb144 --p 0.004 --split test4 --chunks 0-14 --rows $OUT/manifest_gross.csv > /dev/null
$PYC $T/retest/replay/bb_seedcheck.py $OUT/seedcheck_gross.json > /dev/null
echo "DRAW_END rows $(($(wc -l < $OUT/manifest_bb72.csv) - 1)) + $(($(wc -l < $OUT/manifest_gross.csv) - 1))"
