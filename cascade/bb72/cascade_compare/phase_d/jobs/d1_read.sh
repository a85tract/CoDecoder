# D1-R1 / D1-R2 (PHASE_D1_PLAN.md 2.2-2.6, 5.3): D1's read at P, in a job of gpu.sbatch on a full RTX PRO 6000.
# Needs F (the freeze commit's full sha) and P (0.002 or 0.003). Optional: LOCK (default the tree's lock), O (default
# $A/out; the dry runs give their own), SEARCH (the guard's directories), STREAMS and ROUNDS (default the lock's).
#   1. the guards (d1_guard.sh)
#   2. c0 = int(F[:7], 16) mod 9,000,000; the chunks c0 .. c0 + n - 1, n the lock's block at P; the lock's seed kind
#   3. the eager read, unless O/read/done_P: phase C's curves.sh (pc_eval.py: C14' on bb.cascade_eval's evaluation
#      backend under bf16 autocast, S55'-deep on the fused engine, the detector rows kept) -> O/read
#   4. the compiled reads, unless O/compiled/done_P: d1_compiled.py read -> O/compiled
LOCK=${LOCK:-$PD/d1_locked_depspam_fm.json}; O=${O:-$A/out}
: "${F:?F, the freeze commit}" "${P:?P, the claim point}"
source $PD/jobs/d1_guard.sh
d1_guard "$F" "$P" "$T" "$O" || exit 1
read -r KIND N < <($PY -c "import json; L = json.load(open('$LOCK')); print(L['kind_seed'], L['chunks']['$P'])")
C0=$(( 16#${F:0:7} % 9000000 ))
mkdir -p $O/read $O/compiled
LSHA=$(sha256sum $LOCK | cut -c1-64)
echo "{\"F\": \"$F\", \"c0\": $C0, \"n\": $N, \"p\": $P, \"lock\": \"$LOCK\", \"lock_sha256\": \"$LSHA\"}" \
  > $O/read/block_$P.json
echo "read: F $F p $P seed kind $KIND chunks $C0:$((C0 + N)) lock ${LSHA:0:16} -> $O"
if [ ! -f $O/read/done_$P ]; then
  ( SETTING=depspam_fm; PGROUPS="$P@$C0:$((C0 + N))"; OUT=$O/read; TAGS="C14p S55p_deep"; source $PC/jobs/curves.sh )
  [ -f $O/read/done_$P ] || { echo "the eager read at $P did not finish"; exit 1; }
fi
if [ ! -f $O/compiled/done_$P ]; then
  $PY $PD/d1_compiled.py read --read $O/read --p $P --out $O/compiled --lock $LOCK ${STREAMS:+--streams $STREAMS} \
      ${ROUNDS:+--rounds $ROUNDS} || true
  [ -f $O/compiled/done_$P ] || { echo "the compiled read at $P did not finish"; exit 1; }
fi
echo "read at $P done: $(ls $O/read $O/compiled | tr '\n' ' ')"
