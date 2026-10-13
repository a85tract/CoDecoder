#!/bin/bash
# gross stage 3, pod after sampling (STAGE3_GROSS_PLAN.md step 8): cudaq-qec's batch-1 timing on the first test shots
# (quiet pod, 3 accepted fresh processes), then the network's test read and cudaq-qec's test read (blind)
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out; CK=/workspace/data/ckpt/g2_D12k113c10/final_ema.pt; R=$O/rows_gpu_post.csv
export BB_SESSION=h100-gpu-pod-post
idle() { for i in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
nacc() { [ -f $R ] || { echo 0; return; }; $PY - "$R" <<'PY'
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == "opponent" and r["name"] == "gpu_bposd"
          and json.loads(r["note"]).get("accepted")))
PY
}
echo "== cudaq-qec batch-1 timing $(date +%T)"
for k in 1 2 3 4 5; do
  [ "$(nacc)" -ge 3 ] && break
  idle && $PYQ -m bb.gpu_bposd time --code bb144 --p 0.004 --split test --chunks 0 --batches 1 --reps1 300 \
    --out $O/gpu_bposd_post.csv --timing-out $R --proc $k 2>&1 | grep -E "Error|Traceback" | cut -c1-160
done
echo "gpu_bposd accepted $(nacc)"
unset BB_SESSION
echo "== network test read $(date +%T)"
$PY -m bb.stage3_gross infer --split test --chunks 0-29 --ckpt $CK | tail -1
echo "NETREAD_END $(date +%T)"
echo "== cudaq-qec test read $(date +%T)"
$PYQ -m bb.gpu_bposd read --code bb144 --p 0.004 --split test --chunks 0-29 --out $O/gpu_bposd_test_reads.csv 2>&1 | tail -1 | cut -c1-160
echo "READS_END $(date +%T)"
