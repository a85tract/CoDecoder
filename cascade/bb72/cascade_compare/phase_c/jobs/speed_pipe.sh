# Phase C speed, the routed pipelines end to end (PHASE_C_PLAN.md section 5.3): every locked configuration at the p of
# RELAY_P, both modes, each in REPS fresh processes, interleaved (rep-major; pc_report takes the slowest), on phase C's
# GPU, with the networks on the paths frozen in the lock; beside them, in the same rep loop, Cascade alone at each claim
# point in both modes (config -1), the value pc_report compares the pipelines with. Optional: REPS (3), NS (single-mode
# shots, 20,000), NB (batched-mode shots, 500,000), SKIND (seed kind, default 12; 13 in dry runs), LOCKD, SP, SETTINGS_.
SP=${SP:-$R/out/speed}
for rep in $(seq 1 ${REPS:-3}); do
  for S in ${SETTINGS_:-uniform depspam_fm}; do
    L=${LOCKD:-$R/out}/locked_$S.json
    for p in 0.002 0.003; do
      for i in -1 $($PY -c "import json; L = json.load(open('$L')); print(' '.join(str(i) for i, c in enumerate(L['configs']) if c['p'] == $p))"); do
        for m in single batched; do
          [ $m = single ] && N=${NS:-20000} || N=${NB:-500000}
          $PY $PC/pc_speed.py pipeline --locked $L --config $i --p $p --mode $m --rep $rep --n $N ${SKIND:+--kind $SKIND} \
              --speed $SP/speed_components.json --out $SP/rows.jsonl || echo "pipeline $S $i $p $m rep $rep: exit $?"
        done
      done
    done
  done
done
