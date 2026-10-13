# X2's BB72 analysis (hub ising-rev-x2 20179987: bb_d1.py, bb_join.py) on the re-test's records, through its adapter
# retest/shim/rt_x2.py; exploratory, run after the confirmatory d1_confirm.py and reported apart from its verdicts.
# Sourced by d1_x2.sbatch. Needs RT; optional RT_OUT (default <cluster-scratch>/retest/out, the root of all tests; D1's
# directory is RT_OUT/d1 =: D), RT_X2_RUN (the confirm run to take the outcome counts from; default the first). Reads D/confirm_in
# (the reads), D/confirm/npz_sha256.txt and D/confirm/<run>/d1_result.json (the committed records the adapter's input checks come
# from); writes D/x2/bb/{p0.00200.npz,p0.00300.npz,summary.json,fig.npz,adapter.json}. RT_X2_LOCK_SHA: the dry run only.
: "${RT:?RT, the freeze tree root}"
ROOT=${RT_OUT:-<cluster-scratch>/retest/out}/d1
RUN=${RT_X2_RUN:-$(ls -d $ROOT/confirm/run_* | sort -V | head -1)}
[ -f $RUN/d1_result.json ] || { echo "no confirmatory result: X2 runs after d1_confirm.py"; exit 1; }
$PYX $RT/retest/shim/rt_x2.py run --inputs $ROOT/confirm_in --result $RUN/d1_result.json \
     --npz-sha $ROOT/confirm/npz_sha256.txt --out $ROOT/x2 ${RT_X2_LOCK_SHA:+--lock-sha $RT_X2_LOCK_SHA}
