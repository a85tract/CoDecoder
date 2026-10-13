#!/bin/bash
# gross stage 4, pod: fusedt's GPU term and handoff (bb.latency's protocol through bb.fast lat), 3 accepted fresh processes,
# each after bb.timing's idle check (STAGE4_GROSS_PLAN.md 4.1); nothing else runs on the pod meanwhile
source <pod-home>/g7/env.sh
cd $W/code/ising_compare
O=$W/out; CK=$W/ckpt/final_ema.pt; R=$O/rows_gpu.csv
export BB_SESSION=h100-$POD-s4
idle() { for t in $(seq 1 20); do $PY -m bb.timing idle --out "$R" > /dev/null && return 0; sleep 3; done; echo IDLE_FAIL; return 1; }
nacc() { [ -f $R ] || { echo 0; return; }; $PY - "$R" "$1" "$2" <<'PY'
import csv, json, sys
print(sum(1 for r in csv.DictReader(open(sys.argv[1])) if r["role"] == sys.argv[2] and r["name"] == sys.argv[3]
          and json.loads(r["note"]).get("accepted")))
PY
}
name=fusedt:$(sha256sum $CK | cut -c1-12)
for k in 1 2 3 4 5; do
  [ "$(nacc gpu_term $name)" -ge 3 ] && break
  idle && $PY -m bb.fast lat --ckpt $CK --kind fusedt --proc $k --out $R > $O/lat_$k.log 2>&1
done
echo "gpu_term accepted $(nacc gpu_term $name) GPU_END $(date +%T)"
