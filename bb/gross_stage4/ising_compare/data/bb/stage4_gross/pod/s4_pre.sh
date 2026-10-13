#!/bin/bash
# gross stage 4, pod steps before the freeze (STAGE4_GROSS_PLAN.md 6, step 3): GPU record, host check, fusedt's CAL read
# (bb.fast eq: the 10 CAL chunks through fusedt, compared with stage 2's graph32 read; tag g2_D12k113c10_fusedt_s4)
source <pod-home>/g7/env.sh
cd $W/code/ising_compare
O=$W/out; CK=$W/ckpt/final_ema.pt
nvidia-smi -q > $O/nvidia_smi_q.txt
echo "pod $POD gpus $(nvidia-smi -L | wc -l) code $(cat $W/code/COMMIT) cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2)" > $O/allocation.txt
echo "== host check $(date +%T)"; $PY -m bb.stage3_gross hostcheck --out $O/hostcheck.json > /dev/null; echo "hostcheck rc $?"
echo "== fusedt CAL read $(date +%T)"
$PY -m bb.fast eq --ckpt $CK --kind fusedt --label s4 --ref g2_D12k113c10 --out $O/eq_fusedt_s4.json > $O/eq.log 2>&1; echo "eq rc $?"
echo "PRE_END $(date +%T)"
