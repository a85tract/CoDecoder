# The replay checks of the output-head re-test's shim (review finding 10), one job: an args file of head_job.sbatch run with
# the DRY seeds (RT_DRY=1, RT_BEACON/RT_SEEDS = retest/dry/{beacon,seeds}_dry.json) and RT_OUT = <results>/out, so that no
# real directory is touched. The results are <results>/replay{1_analysis,2_seeds,3_kinds}.json, <results> = dirname RT_OUT.
#   1  a1.analyze through the shim, patch off (RT_SEEDS unset), on A1's original reads, into a new directory, against A1's
#      own output files (bytes, or JSON equal), the 18 taus, the verdicts of A1_RESULT, the published files' sha256 prefixes
#   2  with dry seeds: the main process and spawned workers see the new test seeds (module run by rt_a1.py with a spawn pool);
#      the draw (2 chunks per code, through the production command) stores the rule's seeds and the arrays equal stim sampled
#      directly with them; the resume path (KEPT, RECOVERED; patch off refuses the chunk): the Relay-BP fallback's per-shot
#      decoder seeds (200 shots, 4 workers) are the shots' indices in their set, 0..99 and 10,000..10,099; the shim's guards
#   3  every other seed kind of a1.sets.seed_of equals the original function, the test kinds with the patch off equal A1's
#      seeds and the rows of A1's rows.csv
#   sbatch -p atesting --qos=cpu-testing -c 8 --mem=32G -t 01:00:00 -J head_replay --export=ALL,... CMD=head_replay.sh jobs/head_job.sbatch
source $RT/retest/jobs/head_common.sh
set -x
REP=$(dirname $RT_OUT); W=$REP/work; RP=$RT/retest/replay/head; HR=$RP/head_replay.py; TAGS=$RT/retest/tags.json
S=$OUT/sets
export PYTHONPATH=$PYTHONPATH:$RP
PYR="apptainer exec -B <cluster-scratch-root> $I/pyfull.sif <cluster-scratch>/bb/venv/bin/python"
[ -n "${RT_DRY:-}" ] || { echo "the replay runs with the dry seeds only"; exit 1; }
mkdir -p $W
st=0; chk() { "$@" || { echo "FAILED: $*"; st=1; }; }
sums() { (cd $S && sha256sum a1_surface_test_00[01].npz a1_bb72_a1test_00[01].npz); }

# ---- check 1: the original analysis through the shim, patch off
for code in surface bb72; do
  env -u RT_SEEDS A1_SETS=$A/sets A1_SETS_ROWS=$A/sets/rows_dry.csv:$A/sets/rows.csv \
    $PY $SHIM a1.analyze --config data/a1/analysis_$code.json --reads $A/reads/$code --checks $A/reads/$code/checks.json \
    --out $W/analysis_orig_$code
  chk $PY $HR compare-analysis $A/analysis/$code $W/analysis_orig_$code $code $A/reads/$code $W/r1_$code.json
done

# ---- check 2: seeds (dry seeds): main process and spawned workers
$PY $SHIM head_replay probe $W/probe.json
chk $PY $HR check-probe $W/probe.json $RT_BEACON $RT_SEEDS $TAGS $W/r2a.json
# the set-generation main process (the production command), 2 chunks per code
$PYS a1.sets draw --code surface --kind test --chunks 0-1 --freeze $PIN --rows $S/rows.csv
$PYS a1.sets draw --code bb72 --kind a1test --chunks 0-1 --freeze $PIN --rows $S/rows.csv
chk $PY $HR check-draw $S $RT_BEACON $RT_SEEDS $TAGS $W/r2b.json
sums > $W/chunks_before.txt
# the resume path: rows present (KEPT), rows lost (RECOVERED from the bytes, seed checked), rows lost with the patch off
# (refused: the chunk is not the original seed's), rows restored
set +e
$PYS a1.sets draw --code surface --kind test --chunks 0-1 --freeze $PIN --rows $S/rows.csv > $W/draw_kept.log 2>&1
$PYS a1.sets draw --code bb72 --kind a1test --chunks 0-1 --freeze $PIN --rows $S/rows.csv >> $W/draw_kept.log 2>&1
rm -f $S/rows.csv
$PYS a1.sets draw --code surface --kind test --chunks 0-1 --freeze $PIN --rows $S/rows.csv > $W/draw_recovered.log 2>&1
$PYS a1.sets draw --code bb72 --kind a1test --chunks 0-1 --freeze $PIN --rows $S/rows.csv >> $W/draw_recovered.log 2>&1
rm -f $S/rows.csv
env -u RT_SEEDS A1_SETS=$S A1_SETS_ROWS=$S/rows.csv $PY $SHIM a1.sets draw --code surface --kind test --chunks 0-1 \
  --freeze $PIN --rows $S/rows.csv > $W/draw_off.log 2>&1
echo "patch-off draw exit $?"
set -e
$PYS a1.sets draw --code surface --kind test --chunks 0-1 --freeze $PIN --rows $S/rows.csv
$PYS a1.sets draw --code bb72 --kind a1test --chunks 0-1 --freeze $PIN --rows $S/rows.csv
sums > $W/chunks_after.txt
chk $PY $HR check-resume $W/draw_kept.log $W/draw_recovered.log $W/draw_off.log $W/chunks_before.txt $W/chunks_after.txt $W/r2c.json
# the Relay-BP fallback's per-shot seeds (bb's venv: relay-bp), 4 workers, a recorder in every process
rm -rf $W/relay_rec; mkdir -p $W/relay_rec
RT_REPLAY_RECORD=$W/relay_rec $PYR $RP/relay_launcher.py
chk $PY $HR check-relay $W/relay_rec $W/r2d.json

# ---- the shim's guards: each refusal must come from the shim (rt_a1:), the controls must not
$PY -c "
import json
b = json.load(open('$RT_BEACON')); b['dry'] = False; json.dump(b, open('$W/beacon_nondry.json', 'w'))
c = json.load(open('data/a1/analysis_surface.json')); c['q_registered'] = 0.02; c['q'] = [0, 0.01, 0.02]; json.dump(c, open('$W/config_changed.json', 'w'))
json.dump([{'ckpt': 'x', 'arm': 'S-VA', 'seed': 1, 'kind': 'cal', 'chunks': '0-19', 'out': '$OUT/reads/surface/x_cal.npz'}], open('$OUT/job_cal.json', 'w'))
json.dump([{'ckpt': 'x', 'arm': 'S-VA', 'seed': 1, 'kind': 'test', 'chunks': '0-99', 'out': '$A/reads/surface/x_test.npz'}], open('$OUT/job_out_of_tree.json', 'w'))"
: > $W/guards.list
g() { n=$1; shift; set +e; "$@" > $W/guard_$n.log 2>&1; echo "$n:$?:$W/guard_$n.log" >> $W/guards.list; set -e; }
D="--code surface --kind test --chunks 0-0 --freeze $PIN"
g kind_cal $PYS a1.sets draw --code surface --kind cal --chunks 0-0 --freeze $PIN --rows $S/rows.csv
g rows_outside_the_re_test $PYS a1.sets draw $D --rows $A/sets/rows_retest_x.csv
g sets_dir_overridden env A1_SETS=$A/sets $PYS a1.sets draw $D --rows $S/rows.csv
g dry_beacon_without_RT_DRY env -u RT_DRY $PYS a1.sets draw $D --rows $S/rows.csv
g RT_DRY_with_a_real_beacon env RT_BEACON=$W/beacon_nondry.json $PYS a1.sets draw $D --rows $S/rows.csv
g cal_read_job $PYS a1.read --code surface --many $OUT/job_cal.json
g read_out_under_A1 $PYS a1.read --code surface --many $OUT/job_out_of_tree.json
g sets_dry_preflight $PYS a1.preflight --code surface --decisions $A/rule --ckpt-root $ICMP_CKPT --reads $OUT/reads/surface --out $OUT/x.json --sets dry
g config_not_frozen env -u RT_DRY RT_BEACON=$W/beacon_nondry.json $PYS a1.analyze --config $W/config_changed.json --reads $OUT/reads/surface --out $OUT/x
g control_frozen_config env -u RT_DRY RT_BEACON=$W/beacon_nondry.json $PYS a1.analyze --config data/a1/analysis_surface.json --reads $OUT/reads/surface --out $OUT/ctrl_analysis
g control_valid_draw $PYS a1.sets draw $D --rows $S/rows.csv
chk $PY $HR guards $(cat $W/guards.list) $W/r2f.json

# ---- check 3: every other kind
$PY $SHIM head_replay kinds $W/kinds_on.json
env -u RT_SEEDS $PY $SHIM head_replay kinds $W/kinds_off.json
chk $PY $HR check-kinds $W/kinds_on.json $W/kinds_off.json $A/sets/rows.csv $W/r3.json $RT_BEACON $RT_SEEDS $TAGS
chk $PY $HR assemble $W $REP
echo "REPLAY DONE status $st"
[ $st = 0 ]
