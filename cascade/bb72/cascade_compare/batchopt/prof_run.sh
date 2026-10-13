#!/bin/bash
# The profile (batchopt/PROFILE.md): the steps of prof_batched.py on whatever card this host has, each in a fresh process,
# in the order below; a failed step is logged and the next one runs. The same file runs on Alpine (prof.sbatch sets the
# variables) and on a pod (the H100 profile of PROFILE.md). The environment (PYTHONPATH with the tree's ising_compare,
# qldpc_neural and cascade_compare/phase_c; QND_RELAY; TRITON_CACHE_DIR, TORCHINDUCTOR_CACHE_DIR) comes from the caller.
#   PYBIN  the venv's python                RUN    a prefix that enters its container ("" on a pod)
#   O      the output directory             NSYS, NCU  optional: Nsight Systems / Compute, run through $RUN
#   STEPS  optional: the steps (default: shots env work s55 c14 pipe report ncu nsys report)
set -uo pipefail
P=$(cd "$(dirname "$0")" && pwd)/prof_batched.py
RUN=${RUN:-}; O=${O:?output directory}; PYBIN=${PYBIN:?the venv python}
PY="$RUN $PYBIN"
mkdir -p "$O"
log() { echo "== $* $(date +%T)"; }
step() {  # step NAME CMD...
  local n=$1; shift
  log "$n start"; local t0; t0=$(date +%s)
  "$@"; local rc=$?
  log "$n exit=$rc $(( $(date +%s) - t0 )) s"
}
ncu_all() {  # Nsight Compute, if the card's performance counters are open to this user
  [ -n "${NCU:-}" ] || { log "ncu: no executable"; return 0; }
  $RUN $NCU --version | tail -1
  $RUN $NCU --target-processes all --metrics sm__cycles_elapsed.avg \
    $PYBIN -c "import torch; x = torch.ones(8, device='cuda'); print(float((x + 1).sum()))" > $O/ncu_check.txt 2>&1
  if grep -q '==ERROR==' $O/ncu_check.txt || ! grep -q 'sm__cycles_elapsed' $O/ncu_check.txt; then
    log "ncu: no performance counters here"; grep -m3 -E 'ERR|ERROR' $O/ncu_check.txt; return 0
  fi
  local M=dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum
  step ncu_s55 timeout 900 $RUN $NCU --target-processes all --nvtx --nvtx-include "prof/" --graph-profiling node \
    --set full --metrics $M -f -o $O/ncu_s55 $PYBIN $P ncuwork --what s55 --batch 4096 --out $O
  step ncu_c14 timeout 1200 $RUN $NCU --target-processes all --nvtx --nvtx-include "prof/" --graph-profiling node \
    --launch-count 120 --section SpeedOfLight --section LaunchStats --section Occupancy --section WarpStateStats \
    --metrics $M \
    -f -o $O/ncu_c14 $PYBIN $P ncuwork --what c14 --batch 4096 --out $O
  for t in s55 c14; do
    [ -f $O/ncu_$t.ncu-rep ] || continue
    $RUN $NCU --import $O/ncu_$t.ncu-rep --page raw --csv > $O/ncu_${t}_raw.csv 2> /dev/null
    $PY $P ncuparse --csv $O/ncu_${t}_raw.csv --out $O/ncu_$t.json
  done
}
nsys_pipe() {  # Nsight Systems over the batched pipeline at q1 = 5 % (kernels inside the CUDA graphs, NVTX, copies)
  [ -n "${NSYS:-}" ] || { log "nsys: no executable"; return 0; }
  step nsys_pipe timeout 900 $RUN $NSYS profile -t cuda,nvtx,osrt --cuda-graph-trace=node --sample=none \
    --cpuctxsw=none -f true -o $O/nsys_pipe $PYBIN $P pipe --out $O --tag _nsys --q1 0.05 --n 200000 --no-trace \
    --no-overlap
  [ -f $O/nsys_pipe.nsys-rep ] && $RUN $NSYS stats -r cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_sum -f csv \
    -o $O/nsys_pipe $O/nsys_pipe.nsys-rep > /dev/null 2>&1 || log "nsys stats: none"
}
for s in ${STEPS:-shots env work s55 c14 pipe report ncu nsys report}; do
  case $s in
    ncu) ncu_all ;;
    nsys) nsys_pipe ;;
    *) step $s $PY $P $s --out $O ;;
  esac
done
ls -la $O | tail -n +2 | awk '{print $5, $9}' | column -t
