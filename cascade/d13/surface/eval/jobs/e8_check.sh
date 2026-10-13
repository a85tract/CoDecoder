# After C7, in another job: the read computations on the pipelines' shots and the bitwise check of all 48 outputs
# (SURFACE_PLAN 4.4, finding 5), then the one confirmation analysis and the report, whose timings are those of the
# checked outputs.
source $T/surface/eval/jobs/common.sh
if [ -n "${DRY:-}" ]; then C1=0-1; CB=0-5; else C1=0-2; CB=0-50; fi
for p in 0.006 0.0035; do
  run $PY -m surface.sv_read $D --stage speed --p $p --chunks $C1 --comp S1,B1,MS --lock $LOCK --out $E/check_read/single
  run $PY -m surface.sv_read $D --stage speed --p $p --chunks $CB --comp SB,Bb,BQ1,BQ5,MB --lock $LOCK \
      --out $E/check_read/batched
done
run $PY -m surface.sv_speed bitwise --expect c7 --lock $LOCK --pipes $E/pipes --read $E/check_read \
    --out $E/bitwise_c7.json
run $PY -m surface.sv_route confirm --lock $LOCK --gate $GATE --conf $E/conf --out $E/confirm_d13.json
run $PY -m surface.sv_route report --lock $LOCK --confirm $E/confirm_d13.json --speed $E/speed.json \
    --pipes $E/pipes.jsonl --bitwise $E/bitwise_c7.json --out $E/SURFACE_RESULT_draft.md
