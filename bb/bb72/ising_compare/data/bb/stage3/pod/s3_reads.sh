#!/bin/bash
# BB72 stage 3: the H100 reads on the test splits and cudaq-qec's batch-1 timing (STAGE3_PLAN.md steps 8)
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out; CK=/workspace/data/ckpt; POL=/workspace/repo/ising_compare/data/bb/policies; GPU="NVIDIA H100 80GB HBM3"
echo "== network reads $(date +%T)"
$PY -m bb.route infer --split test --code bb72 --p 0.003 --tag r11 --chunks 0-19 --policy $POL/bb72_p0.003_rep1.json --ckpt $CK/r11/final_ema.pt --gpu "$GPU" | tail -1 | cut -c1-90
$PY -m bb.route infer --split test --code bb72 --p 0.003 --tag r12 --chunks 0-19 --policy $POL/bb72_p0.003_rep2.json --ckpt $CK/r12/final_ema.pt --gpu "$GPU" | tail -1 | cut -c1-90
$PY -m bb.route infer --split test --code bb72 --p 0.002 --tag r12 --chunks 0-29 --policy $POL/bb72_p0.002_rep2.json --ckpt $CK/r12/final_ema.pt --gpu "$GPU" | tail -1 | cut -c1-90
echo "NETREADS_END $(date +%T)"
echo "== cudaq-qec batch-1 timing $(date +%T)"
export BB_SESSION="H100 session post"
R=$O/rows_gpu_post.csv
idle() { for i in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
for p in 0.003 0.002; do
  for k in 1 2 3 4 5; do
    n=$([ -f $R ] && $PY - "$R" "$p" <<'PY' || echo 0
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == "opponent" and r["name"] == "gpu_bposd" and float(r["p"]) == float(sys.argv[2]) and json.loads(r["note"]).get("accepted")))
PY
)
    [ "$n" -ge 3 ] && break
    idle && $PY -m bb.gpu_bposd time --code bb72 --p $p --split test --chunks 0 --batches 1 --reps1 300 --out $O/gpu_bposd_post.csv --timing-out $R --proc $k 2>&1 | grep -E "^timing" | cut -c1-80
  done
done
echo "== cudaq-qec test reads $(date +%T)"
unset BB_SESSION
$PY -m bb.gpu_bposd read --code bb72 --p 0.003 --split test --chunks 0-19 --out $O/gpu_bposd_test_reads.csv 2>&1 | tail -1 | cut -c1-120
$PY -m bb.gpu_bposd read --code bb72 --p 0.002 --split test --chunks 0-29 --out $O/gpu_bposd_test_reads.csv 2>&1 | tail -1 | cut -c1-120
echo "READS_END $(date +%T)"
