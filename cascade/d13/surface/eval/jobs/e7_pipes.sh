# Step C7 (SURFACE_PLAN 5): the pipelines C1, C4, C5 and Cascade alone (C2) at both p in both modes, three fresh
# processes each, rep-major in one job, on the lock's C7 plan (kind 12, PIPE_CHUNKS 0-49: 20,000 shots single, 500,000
# batched). e8's bitwise check requires all 48 outputs, from this one job.
source $T/surface/eval/jobs/common.sh
for rep in ${E7_REPS:-1 2 3}; do for p in 0.006 0.0035; do for c in C1 C4 C5 C2; do for m in single batched; do
  run $PY -m surface.sv_speed pipeline $D --config $c --p $p --mode $m --rep $rep --lock $LOCK --stage c7 \
      --out $E/pipes.jsonl --save $E/pipes
done; done; done; done
