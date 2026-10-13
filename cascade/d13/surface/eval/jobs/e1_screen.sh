# Step C3 (SURFACE_PLAN 5, 9.6), before the development reads: the unit checks, the speed screen of S and B (three
# fresh processes each, interleaved; kind 12, SCREEN_CHUNKS) and M alone (C3: three processes per p and mode; kind 12,
# the pipelines' chunks), then the summary that freezes the paths. One full RTX PRO 6000.
source $T/surface/eval/jobs/common.sh
if [ -n "${DRY:-}" ]; then SA="--n-agree 50000,10000 --n-rt 2000 --secs 1"; MC=300-305; NS=2000; NB=50000
else SA=""; MC=0-50; NS=20000; NB=500000; fi
run $PY -m surface.sv_test
for rep in 1 2 3; do
  for net in S B; do
    BB=; [ $net = B ] && [ -n "${SV_B_BATCHES:-}" ] && BB="--batches $SV_B_BATCHES"    # a MIG slice's code check only
    run $PY -m surface.sv_speed screen $D --net $net --rep $rep $SA $BB --out $E/screen.jsonl --save $E/screen
  done
  for p in 0.006 0.0035; do for m in single batched; do
    n=$NS; [ $m = batched ] && n=$NB
    run $PY -m surface.sv_speed pipeline $D --config C3 --p $p --mode $m --rep $rep --chunks $MC --n $n \
        --out $E/screen.jsonl --save $E/screen_m
  done; done
done
run $PY -m surface.sv_speed summarize --rows $E/screen.jsonl --out $E/speed.json
