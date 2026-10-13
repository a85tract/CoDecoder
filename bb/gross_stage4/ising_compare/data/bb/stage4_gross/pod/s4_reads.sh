#!/bin/bash
# gross stage 4, pod after the manifest commit (STAGE4_GROSS_PLAN.md 6, step 8): the network's test4 read through fusedt
# (blind), cudaq-qec's batch-1 timing on the first test4 shots (quiet pod, 3 accepted fresh processes), cudaq-qec's test4
# read (one whole read, blind)
source <pod-home>/g7/env.sh
cd $W/code/ising_compare
O=$W/out; CK=$W/ckpt/final_ema.pt; R=$O/rows_gpu.csv
echo "== network test4 read $(date +%T)"
$PY -m bb.stage4_gross infer --split test4 --chunks 0-14 --ckpt $CK | tail -1
echo "NETREAD_END $(date +%T)"
export BB_SESSION=h100-$POD-s4
idle() { for t in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
nacc() { [ -f $R ] || { echo 0; return; }; $PY - "$R" <<'PY'
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == "opponent" and r["name"] == "gpu_bposd"
          and json.loads(r["note"]).get("accepted")))
PY
}
echo "== cudaq-qec batch-1 timing $(date +%T)"
for k in 1 2 3 4 5; do
  [ "$(nacc)" -ge 3 ] && break
  idle && $PYQ -m bb.gpu_bposd time --code bb144 --p 0.004 --split test4 --chunks 0 --batches 1 --reps1 300 \
    --out $O/gpu_bposd_time.csv --timing-out $R --proc $k 2>&1 | grep -E "Error|Traceback" | cut -c1-160
done
echo "gpu_bposd accepted $(nacc)"
unset BB_SESSION
echo "== cudaq-qec test4 read $(date +%T)"
$PYQ -m bb.gpu_bposd read --code bb144 --p 0.004 --split test4 --chunks 0-14 --out $O/gpu_bposd_test4_reads.csv 2>&1 | tail -1 | cut -c1-160
echo "READS_END $(date +%T)"
