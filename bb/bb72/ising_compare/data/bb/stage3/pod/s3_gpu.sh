#!/bin/bash
# BB72 stage 3: the H100 GPU-term session and cudaq-qec's batch-1 rows on cal chunk 0 (STAGE3_PLAN.md 4.1), before the freeze
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out; CK=/workspace/data/ckpt; R=$O/rows_gpu_pre.csv
export BB_SESSION="H100 session pre"
idle() { for i in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
nacc() { $PY - "$R" "$1" "$2" <<'PY'
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == sys.argv[2] and r["name"] == sys.argv[3]
          and r["session"] and json.loads(r["note"]).get("accepted") and (sys.argv[2] != "opponent" or json.loads(r["note"]).get("split") == "cal")))
PY
}
for t in r11 r12; do
  sha=$(sha256sum $CK/$t/final_ema.pt | cut -c1-12)
  for k in 1 2 3 4 5 6 7; do
    [ "$(nacc gpu_term fused:$sha)" -ge 5 ] && break
    idle && $PY -m bb.latency --ckpt $CK/$t/final_ema.pt --engine fused --p 0.003 --proc $k --out $R | cut -c1-80
  done
  echo "$t accepted $(nacc gpu_term fused:$sha)"
done
for p in 0.003 0.002; do
  for k in 1 2 3 4 5; do
    [ "$($PY - "$R" "$p" <<'PY'
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == "opponent" and r["name"] == "gpu_bposd" and float(r["p"]) == float(sys.argv[2]) and json.loads(r["note"]).get("accepted")))
PY
)" -ge 3 ] && break
    idle && $PY -m bb.gpu_bposd time --code bb72 --p $p --split cal --chunks 0 --batches 1 --reps1 300 --out $O/gpu_bposd_pre.csv --timing-out $R --proc $k 2>&1 | grep -E "^timing" | cut -c1-80
  done
done
echo "GPU_PRE_END $(date +%T)"
