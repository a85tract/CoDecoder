# Common environment of the BB re-test jobs on Alpine (RETEST.md section 3.1), sourced by every bb_*.sbatch.
# MODE=run (default): the freeze tree, the re-test's beacon and seeds; MODE=dry: a past round, chunks of RT_DRY_SHOTS.
T=${T:-<cluster-scratch>/retest/code/freeze}
MODE=${MODE:-run}
if [ "$MODE" = dry ]; then
  RTD=<cluster-scratch>/retest/dry; export RT_BEACON=$RTD/beacon_dry.json RT_SEEDS=$RTD/seeds_dry.json
  export RT_DRY_SHOTS=${RT_DRY_SHOTS:-100}; OUT=${OUT:-$RTD/bb}
else
  RTD=<cluster-scratch>/retest/run; export RT_BEACON=$RTD/beacon.json RT_SEEDS=$RTD/seeds.json
  unset RT_DRY_SHOTS; OUT=${OUT:-<cluster-scratch>/retest/out/bb}
fi
export BB_SHOTS=$OUT/shots BB_PREDS=$OUT/preds QND_RELAY=<cluster-scratch>/bb/relay PYTHONUNBUFFERED=1
mkdir -p $OUT/shots $OUT/preds $OUT/rows
S=$T/retest/shim/rt_bb.py; X=$T/retest/replay/bb_replay.py
PYC=<cluster-scratch>/retest/bin/pycpu; PYG=<cluster-scratch>/retest/bin/pycu128
RTX="NVIDIA RTX PRO 6000 Blackwell Server Edition"
bb72() { cd $T/bb/bb72/ising_compare && export PYTHONPATH=.:../qldpc_neural RT_MANIFEST=$OUT/manifest_bb72.csv; }
gross() { cd $T/bb/gross_stage4/ising_compare && export PYTHONPATH=.:../qldpc_neural RT_MANIFEST=$OUT/manifest_gross.csv; }
