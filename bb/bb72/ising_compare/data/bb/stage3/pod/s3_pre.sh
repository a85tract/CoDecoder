#!/bin/bash
# BB72 stage 3, pod steps before the freeze (STAGE3_PLAN.md step 4)
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out; CK=/workspace/data/ckpt; GPU="NVIDIA H100 80GB HBM3"
nvidia-smi -q > $O/nvidia_smi_q.txt; echo "pod $RUNPOD_POD_ID gpus $(nvidia-smi -L | wc -l)" > $O/allocation.txt
idle() { for i in $(seq 1 20); do $PY -m bb.timing idle --out "$1" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
acc() { $PY - "$1" "$2" <<'PY'
import csv, json, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["role"] == sys.argv[2]]
print("yes" if rows and json.loads(rows[-1]["note"]).get("accepted") else "no")
PY
}
echo "== validation $(date +%T)"
for t in r11 r12; do for p in 0.003 0.002; do
  for k in va vb; do $PY -m bb.route infer --split cal --chunks 0 --code bb72 --p $p --tag $t$k --engine fused --ckpt $CK/$t/final_ema.pt | tail -1 | cut -c1-120; done
  $PY -m bb.route infer --split cal --chunks 0 --code bb72 --p $p --tag ${t}vf --engine fp32 --ckpt $CK/$t/final_ema.pt | tail -1 | cut -c1-120
  $PY -m bb.validate check --split cal --code bb72 --p $p --a ${t}va --b ${t}vb --ref ${t}vf --gpu H100 --out $O/validation.jsonl | cut -c1-200
done; done
echo "== cal reads $(date +%T)"
for t in r11 r12; do for p in 0.003 0.002; do
  $PY -m bb.route infer --split cal --chunks 0-19 --code bb72 --p $p --tag ${t}h --engine fused --ckpt $CK/$t/final_ema.pt | tail -1 | cut -c1-160
done; done
echo "== preflight $(date +%T)"
export BB_SESSION="H100 session preflight"
idle $O/rows_preflight.csv; $PY -m bb.latency --ckpt $CK/r12/final_ema.pt --engine fused --p 0.003 --proc 1 --out $O/rows_preflight.csv | tail -1 | cut -c1-300
idle $O/rows_preflight.csv; $PY -m bb.gpu_bposd time --code bb72 --p 0.002 --split cal --chunks 0 --batches 1 --reps1 300 --out $O/gpu_bposd_preflight.csv --timing-out $O/rows_preflight.csv --proc 1 2>&1 | grep -E "^timing|Error" | cut -c1-400
echo "PREFLIGHT_END $(date +%T)"
