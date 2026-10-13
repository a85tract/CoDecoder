# The confirmation's batched computations (kind 11, the lock's chunks 0 .. n-1 at each p): SB, Bb, BQ1, BQ5 and MB,
# each p in one process, so that the batches are the pipelines' (SURFACE_PLAN 4.4), and plain PM (MP, context only).
# After the lock commit: each read refuses to draw a shot without the committed lock and gate (finding 2). A read that
# stopped continues where it stopped with RESUME=1 in the job file (finding 6), once the stopped job has ended.
source $T/surface/eval/jobs/common.sh
for p in ${PS:-0.006 0.0035}; do
  run $PY -m surface.sv_read $D --stage conf --p $p --chunks $(lockchunks $p) --comp SB,Bb,BQ1,BQ5,MB,MP --lock $LOCK \
      --gate $GATE ${RESUME:+--resume} --out $E/conf
done
