# D1's one analysis for the re-test (RETEST.md test D1): D1's jobs/d1_confirm.sh (d1_confirm.py with D1's lock and F, on
# the reads and compiled reads, the timing rows of D1's job CJ73) run from D1's deployed tree through the seed shim,
# so that its check "every chunk's seed is eval_seed(kind, p, j, family)" holds for the re-test's seeds. Sourced by
# d1_cpu_job.sh (STEP=d1_confirm_rt.sh). Needs RT, RT_BEACON, RT_SEEDS; optional RT_OUT (default
# <cluster-scratch>/retest/out, the root of all tests; D1's directory is RT_OUT/d1 =: D). Reads D/<p>/{read,compiled}
# (both p done), links them into D/confirm_in/{read,compiled} (the directory pair d1_confirm.py takes), checks them against the
# sha256 each read job recorded, writes D/confirm/npz_sha256.txt (the committed record X2's adapter takes the read hashes from)
# and D/confirm/run_<k>/{d1_result.json,d1_tables.md}. One run: a second needs RT_CONFIRM_RERUN=1.
: "${RT:?RT, the freeze tree root}" "${RT_BEACON:?}" "${RT_SEEDS:?}"
export RT_BEACON RT_SEEDS
F=310ce8e0f8bfb9d3b97b27186547a836d29fc9ca
LOCK=$PD/d1_locked_depspam_fm.json; LOCK_SHA=3ab39941df356883be395ef7bbf39672fdb546729187bd89fa523be22714f96c
SPEED=<cluster-scratch>/cascade_d1/out/speed/rows.jsonl; SPEED_SHA=77889d11e194352f9a3fefaf1386810500f3566f0e4b0b144523d0ad9f52edd8
ROOT=${RT_OUT:-<cluster-scratch>/retest/out}/d1; CIN=$ROOT/confirm_in; OUT=$ROOT/confirm
PYS="$PY $RT/retest/shim/rt_d1.py"
[ "$(sha256sum $LOCK | cut -c1-64)" = "$LOCK_SHA" ] || { echo "the lock is not D1's"; exit 1; }
[ "$(sha256sum $SPEED | cut -c1-64)" = "$SPEED_SHA" ] || { echo "the timing rows are not D1-S's (job CJ73)"; exit 1; }
if ls $OUT/run_* > /dev/null 2>&1 && [ -z "${RT_CONFIRM_RERUN:-}" ]; then echo "an analysis run exists in $OUT"; exit 1; fi
mkdir -p $CIN/read $CIN/compiled $OUT
for P in 0.002 0.003; do
  t=$(printf 'p%.5f' $P)
  for part in read compiled; do
    [ -f $ROOT/$P/$part/done_$P ] || { echo "the $part read at $P is not done"; exit 1; }
    for ext in npz json; do ln -sfn $ROOT/$P/$part/$t.$ext $CIN/$part/$t.$ext; done
  done
  ( cd $ROOT/$P && sha256sum -c --quiet sha256.txt ) || { echo "a file of the read at $P changed since its job"; exit 1; }
done
( cd $CIN && sha256sum read/p*.npz compiled/p*.npz ) > $OUT/npz_sha256.txt
$PYS $PD/d1_confirm.py --lock $LOCK --F $F --read $CIN/read --compiled $CIN/compiled --speed $SPEED --out $OUT
