# D1-S (PHASE_D1_PLAN.md 4.1), in a job of gpu.sbatch on a full RTX PRO 6000: the twelve runs of both claim points
# (Cascade alone single and batched; D single and batched on today's engine, base, and on milestone 2's, new), each in a
# fresh d1_m2.py pipeline process, rep-major (all twelve, then all twelve again, REPS times). It stops at the first
# process that fails; no process is rerun or replaced. Optional: O (default $A/out/speed), REPS (3), N1 (20,000
# single-mode shots), NB (500,000 batched-mode shots), D1_LOCK (exported; default the tree's lock).
O=${O:-$A/out/speed}; REPS=${REPS:-3}; N1=${N1:-20000}; NB=${NB:-500000}
mkdir -p $O
[ -f $O/rows.jsonl ] && { echo "$O/rows.jsonl exists: D1-S does not run twice into one file"; exit 1; }
for rep in $(seq 1 $REPS); do
  for pi in "0.002 0" "0.003 1"; do
    read -r p i <<< "$pi"
    for run in "-1 single base" "-1 batched base" "$i single base" "$i single new" "$i batched base" "$i batched new"; do
      read -r cfg mode eng <<< "$run"
      n=$([ $mode = single ] && echo $N1 || echo $NB)
      echo "== rep $rep p $p config $cfg $mode $eng $(date +%T)"
      $PY $PD/d1_m2.py pipeline --setting depspam_fm --config $cfg --p $p --mode $mode --engine $eng --tail-b 0 \
          --rep $rep --n $n --out $O/rows.jsonl || { echo "STOP: rep $rep p $p config $cfg $mode $eng failed"; exit 1; }
    done
  done
done
echo "D1-S done: $(wc -l < $O/rows.jsonl) rows"
