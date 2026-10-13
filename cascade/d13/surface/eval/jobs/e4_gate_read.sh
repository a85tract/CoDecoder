# The bitwise gate's reads (item 8), in a job other than e3's: S1, B1, MS on the single mode's shots and SB, Bb, BQ1,
# BQ5, MB on the batched mode's, then the check (every output of the gate plan, finding 5) and the gate file beside the
# lock. If the gate passes, commit and push the lock and the gate file (scripts/lockpair.sh, which writes the receipt).
source $T/surface/eval/jobs/common.sh
if [ -n "${DRY:-}" ]; then G1=200-201; GB=200-205; else G1=200-202; GB=200-250; fi
for p in 0.006 0.0035; do
  run $PY -m surface.sv_read $D --stage speed --p $p --chunks $G1 --comp S1,B1,MS --lock $LOCK --out $E/gate_read/single
  run $PY -m surface.sv_read $D --stage speed --p $p --chunks $GB --comp SB,Bb,BQ1,BQ5,MB --lock $LOCK \
      --out $E/gate_read/batched
done
run $PY -m surface.sv_speed bitwise --expect gate --lock $LOCK --pipes $E/gate_pipes --read $E/gate_read \
    --out $E/bitwise_gate.json
run $PY -m surface.sv_route gate --lock $LOCK --check $E/bitwise_gate.json --out $GATE
