# The confirmation's batch-1 computations S1, B1 and MS (kind 11): at both p over the lock's chunks, or, to split B1
# over two cards, at P over CH (two jobs, CH split at a multiple of 100 chunks). After the lock commit, as e5; it may
# run beside e5 (the observable files are published once, finding 7). RESUME=1 continues a read that stopped.
source $T/surface/eval/jobs/common.sh
if [ -n "${P:-}" ]; then LIST="$P:$CH"; else LIST="0.006:$(lockchunks 0.006) 0.0035:$(lockchunks 0.0035)"; fi
for x in $LIST; do
  run $PY -m surface.sv_read $D --stage conf --p ${x%%:*} --chunks ${x#*:} --comp S1,B1,MS --lock $LOCK --gate $GATE \
      ${RESUME:+--resume} --out $E/conf
done
