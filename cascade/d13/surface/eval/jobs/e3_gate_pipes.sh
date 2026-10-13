# The bitwise gate's pipelines (item 8 of the independent check; kind 12): C1, C4, C5 and C2 at both p in both modes,
# one fresh process each, with the lock's thresholds, on the chunks and sizes of the lock's gate plan (sv_common.PIPES:
# GATE_CHUNKS 200-249, 20,000 single and 500,000 batched shots). e4 reads the same shots in another job.
source $T/surface/eval/jobs/common.sh
for p in 0.006 0.0035; do for c in C1 C4 C5 C2; do for m in single batched; do
  run $PY -m surface.sv_speed pipeline $D --config $c --p $p --mode $m --rep 1 --lock $LOCK --stage gate \
      --out $E/gate_pipes.jsonl --save $E/gate_pipes
done; done; done
