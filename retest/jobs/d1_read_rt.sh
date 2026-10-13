# D1's read at P for the re-test (RETEST.md test D1): the steps of D1's jobs/d1_read.sh, run from D1's deployed tree
# ($T, F = 310ce8e0), with D1's lock unchanged, the eager read's python calls going through the seed shim
# retest/shim/rt_d1.py. Sourced by d1_R1.sbatch / d1_R2.sbatch through d1_gpu_job.sh, which sets T PC PD PY.
# Needs: P (0.002 or 0.003), RT (the freeze tree root: RT/retest/...), RT_BEACON and RT_SEEDS (exported; seeds.py),
# optional RT_OUT (default <cluster-scratch>/retest/out, the root of all tests; the read writes RT_OUT/d1/<P>/{read,compiled}) and
# RT_DRY_CHUNKS (the dry run's number of chunks, allowed only with the dry seeds: d1_guard_rt.sh).
# Differences from d1_read.sh, and nothing else:
#   1. the guards are d1_guard_rt.sh (no kind-14 freshness search; see there);
#   2. F is fixed to D1's freeze commit and the lock to D1's (sha256 checked); the lock is not overridable;
#   3. the output directory is RT_OUT/d1/<P>; the block record also names the seeds' and beacon's sha256 (resume safety);
#   4. the eager read's `$PY .../pc_eval.py` calls (curves.sh) are `$PY rt_d1.py .../pc_eval.py`;
#   5. the shim writes a trace of every seed the main process and the sampling workers use (RT_D1_TRACE, read/trace/); after the
#      eager read, d1_seedcheck_rt.py checks the trace and the recorded seeds against the beacon and re-draws a sample of the
#      chunks from them; a failure stops the job before the compiled read;
#   6. no STREAMS / ROUNDS override (the lock's values), RT_DRY_CHUNKS replaces the lock's chunk count in the dry run;
#   7. after the compiled read, the sha256 of the four files the analyses read go to RT_OUT/d1/<P>/sha256.txt.
# d1_compiled.py read runs as in D1 (no shim): it draws no shots, the saved detector rows are its input.
: "${P:?P, the claim point}" "${RT:?RT, the freeze tree root}" "${RT_BEACON:?}" "${RT_SEEDS:?}"
export RT_BEACON RT_SEEDS
F=310ce8e0f8bfb9d3b97b27186547a836d29fc9ca
LOCK=$PD/d1_locked_depspam_fm.json; LOCK_SHA=3ab39941df356883be395ef7bbf39672fdb546729187bd89fa523be22714f96c
O=${RT_OUT:-<cluster-scratch>/retest/out}/d1/$P
PYS="$PY $RT/retest/shim/rt_d1.py"
source $RT/retest/jobs/d1_guard_rt.sh
d1_guard_rt "$F" "$P" "$T" "$O" || exit 1
LSHA=$(sha256sum $LOCK | cut -c1-64)
[ "$LSHA" = "$LOCK_SHA" ] || { echo "the lock $LOCK is not D1's (sha256 $LSHA)"; exit 1; }
read -r KIND N < <($PY -c "import json; L = json.load(open('$LOCK')); print(L['kind_seed'], L['chunks']['$P'])")
[ "$KIND" = 14 ] || { echo "the lock's seed kind is $KIND, not 14"; exit 1; }
NLOCK=$N
if [ -n "${RT_DRY_CHUNKS:-}" ]; then N=$RT_DRY_CHUNKS; echo "DRY RUN: $N chunks instead of the lock's $NLOCK"; fi
C0=$(( 16#${F:0:7} % 9000000 ))
mkdir -p $O/read $O/compiled
SSHA=$(sha256sum $RT_SEEDS | cut -c1-64); BSHA=$(sha256sum $RT_BEACON | cut -c1-64)
[ -f $O/read/block_$P.json ] || echo "{\"F\": \"$F\", \"c0\": $C0, \"n\": $N, \"p\": $P, \"lock\": \"$LOCK\", \"lock_sha256\": \"$LSHA\", \"seeds_file\": \"$RT_SEEDS\", \"seeds_sha256\": \"$SSHA\", \"beacon_file\": \"$RT_BEACON\", \"beacon_sha256\": \"$BSHA\", \"dry_chunks\": \"${RT_DRY_CHUNKS:-}\"}" \
  > $O/read/block_$P.json
echo "read: F $F p $P seed kind $KIND chunks $C0:$((C0 + N)) lock ${LSHA:0:16} seeds ${SSHA:0:16} beacon ${BSHA:0:16} -> $O"
if [ ! -f $O/read/done_$P ]; then
  rm -rf $O/read/trace; mkdir -p $O/read/trace
  ( export RT_D1_TRACE=$O/read/trace; PY="$PYS"; SETTING=depspam_fm; PGROUPS="$P@$C0:$((C0 + N))"; OUT=$O/read; TAGS="C14p S55p_deep"; source $PC/jobs/curves.sh )
  [ -f $O/read/done_$P ] || { echo "the eager read at $P did not finish"; exit 1; }
fi
$PYS $RT/retest/jobs/d1_seedcheck_rt.py --read $O/read --p $P || { echo "the seed check of the read at $P failed"; exit 1; }
if [ ! -f $O/compiled/done_$P ]; then
  $PY $PD/d1_compiled.py read --read $O/read --p $P --out $O/compiled --lock $LOCK || true
  [ -f $O/compiled/done_$P ] || { echo "the compiled read at $P did not finish"; exit 1; }
fi
( cd $O && sha256sum read/p*.npz read/p*.json compiled/p*.npz compiled/p*.json > sha256.txt )
echo "read at $P done: $(ls $O/read $O/compiled | tr '\n' ' ')"
