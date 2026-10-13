# Phase C speed, the networks (PHASE_C_PLAN.md section 5): every network in REPS fresh processes, interleaved
# (rep-major), then the components the routing rule reads (each network's frozen paths). Optional: REPS (3), TAGS
# (every network present, LATE ones included once their weights exist), FLAGS (e.g. --quick --batches 1,1024,4096 for
# dry runs: Cascade on its graph path, ConvLines on its fused paths), SKIND (the screen's seed kind, default 12; 13 in
# dry runs), SP (output directory; $R/out/speed_late for a LATE network timed after the lock, which must not change
# the frozen file).
REPS=${REPS:-3}; SP=${SP:-$R/out/speed}
TAGS=${TAGS:-$($PY $PC/pc_common.py tags) $($PY $PC/pc_common.py tags late)}
mkdir -p $SP
nvidia-smi -q -d CLOCK,PERFORMANCE | grep -E -A3 'Clocks Event Reasons|Max Clocks' | head -20 > $SP/clocks_$SLURM_JOB_ID.txt || true
for rep in $(seq 1 $REPS); do
  for t in $TAGS; do
    $PY $PC/pc_speed.py nets --net $t --rep $rep ${FLAGS:-} ${SKIND:+--kind $SKIND} --out $SP/rows.jsonl \
        || echo "speed $t rep $rep: exit $?"
  done
done
$PY $PC/pc_speed.py summarize --rows $SP/rows.jsonl --out $SP/speed_components.json
