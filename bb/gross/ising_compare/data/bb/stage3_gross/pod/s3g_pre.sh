#!/bin/bash
# gross stage 3, pod steps before the freeze (STAGE3_GROSS_PLAN.md step 5): GPU record, host check, engine check,
# the GPU term and handoff (5 accepted bb.latency processes), cudaq-qec's batch-1 time on cal chunk 0 (3 accepted)
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out; CK=/workspace/data/ckpt/g2_D12k113c10/final_ema.pt; R=$O/rows_gpu_pre.csv
nvidia-smi -q > $O/nvidia_smi_q.txt
echo "pod gpu-pod machine gpu-machine gpus $(nvidia-smi -L | wc -l) code $(cat /workspace/repo/COMMIT)" > $O/allocation.txt
echo "== host check $(date +%T)"; $PY -m bb.stage3_gross hostcheck --out $O/hostcheck.json > /dev/null; echo "hostcheck rc $?"
echo "== engine check $(date +%T)"
$PY -m bb.stage3_gross infer --split cal --chunks 0-9 --ckpt $CK --tag g2s3cal | tail -1
$PY -m bb.stage3_gross engcheck --ref /workspace/data/ref --tag g2s3cal --out $O/engcheck.json > /dev/null; echo "engcheck rc $?"
echo "== GPU side $(date +%T)"
export BB_SESSION=h100-gpu-pod-pre
idle() { for i in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
nacc() { [ -f $R ] || { echo 0; return; }; $PY - "$R" "$1" "$2" <<'PY'
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == sys.argv[2] and r["name"] == sys.argv[3]
          and json.loads(r["note"]).get("accepted")))
PY
}
sha=$(sha256sum $CK | cut -c1-12)
for k in 1 2 3 4 5 6 7; do
  [ "$(nacc gpu_term graph32:$sha)" -ge 5 ] && break
  idle && $PY -m bb.latency --ckpt $CK --engine graph32 --p 0.004 --proc $k --out $R > /dev/null
done
echo "gpu_term accepted $(nacc gpu_term graph32:$sha)"
for k in 1 2 3 4 5; do
  [ "$(nacc opponent gpu_bposd)" -ge 3 ] && break
  idle && $PYQ -m bb.gpu_bposd time --code bb144 --p 0.004 --split cal --chunks 0 --batches 1 --reps1 300 \
    --out $O/gpu_bposd_pre.csv --timing-out $R --proc $k 2>&1 | grep -E "Error|Traceback" | cut -c1-160
done
echo "gpu_bposd accepted $(nacc opponent gpu_bposd)"
echo "PRE_END $(date +%T)"
