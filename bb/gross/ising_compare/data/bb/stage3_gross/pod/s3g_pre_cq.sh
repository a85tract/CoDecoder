#!/bin/bash
# gross stage 3, pod before the freeze: cudaq-qec's batch-1 time on cal chunk 0, 3 accepted fresh processes (rerun after
# the relay testdata module, which builds the X-sector DEM, was added to the pod; the first attempts failed at import)
source /workspace/env.sh
cd /workspace/repo/ising_compare
O=/workspace/out; R=$O/rows_gpu_pre.csv
export BB_SESSION=h100-gpu-pod-pre
idle() { for i in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
nacc() { $PY - "$R" <<'PY'
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == "opponent" and r["name"] == "gpu_bposd"
          and json.loads(r["note"]).get("accepted")))
PY
}
for k in 1 2 3 4 5; do
  [ "$(nacc)" -ge 3 ] && break
  idle && $PYQ -m bb.gpu_bposd time --code bb144 --p 0.004 --split cal --chunks 0 --batches 1 --reps1 300 \
    --out $O/gpu_bposd_pre.csv --timing-out $R --proc $k 2>&1 | grep -E "Error|Traceback" | cut -c1-160
done
echo "gpu_bposd accepted $(nacc)"
echo "PRE_CQ_END $(date +%T)"
