#!/bin/bash
# BB72 stage 3: cudaq-qec's batch-1 timing on the first test shots, 3 accepted fresh processes per split (STAGE3_PLAN 4.1)
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out; R=$O/rows_gpu_post.csv
export BB_SESSION="H100 session post"
idle() { for i in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
cnt() { [ -f "$R" ] || { echo 0; return; }; $PY -c "
import csv, json, sys
print(sum(1 for r in csv.DictReader(open('$R')) if r['role'] == 'opponent' and r['name'] == 'gpu_bposd' and float(r['p']) == $1 and json.loads(r['note']).get('accepted')))"; }
for p in 0.003 0.002; do
  for k in 1 2 3 4 5; do
    [ "$(cnt $p)" -ge 3 ] && break
    idle && $PY -m bb.gpu_bposd time --code bb72 --p $p --split test --chunks 0 --batches 1 --reps1 300 --out $O/gpu_bposd_post.csv --timing-out $R --proc $k 2>&1 | grep -E "^timing" | cut -c1-80
  done
  echo "p $p accepted $(cnt $p)"
done
echo "CQTIME_END $(date +%T)"
