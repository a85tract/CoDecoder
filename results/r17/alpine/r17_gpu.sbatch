#!/bin/bash
#SBATCH -p ah200
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:h200:1
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem-per-cpu=3840
#SBATCH -t 03:00:00
#SBATCH -J r17_gpu
#SBATCH -o <cluster-scratch>/ising_r17/logs/%x_%j.out
# R17's GPU reads (PLAN_R17.md sections 3-4) on one whole H200 (sm_90), in one job so that the checks and the reads
# run on the same device and software:
#   1. r17.py pins (every pinned file, the 8 registered TensorRT plans included);
#   2. ours on all 200,000 V17b shots against the development record (H100), bit for bit: the frozen
#      speed2.s19.t4ours val, unchanged;
#   3. r17.py hopper: every Ising part on T5 chunks 0-19, + plain and + correlated PM, against S4-R3's H100 decisions;
#      M1 / M4 torch residuals of V17b chunk 0 against the read pod's files;
#   4. only if 2 and 3 passed: r17.py read of T6 (ours, plain and correlated PM, every Ising part).
#   sbatch r17/alpine/r17_gpu.sbatch
# DRY=1: steps 1-3 as above but dry (the plans are read from $B/standin, which the rehearsal filled with the 8
# registered plans), step 4 on the dry stand-in T6dry into $B/out/dry/preds_<job>. No T6 shot is read. If no whole
# H200 is free, the run (and a dry run) may take an H200 MIG slice instead (--gres=gpu:h200_3g.71gb:1): the same
# checks then decide on that slice.
# POD=1 (the H100 fallback, PLAN_R17.md section 7.3): this script run by bash on a RunPod H100 whose paths mirror
# Alpine's (r17/pod.sh push: $B, $R/ext/ising_nv, the frozen tree, V17b; $R/venv_trt is the frozen pod recipe's venv),
# without apptainer; r17/pod.sh start sets SLURM_JOB_ID and SLURM_CPUS_ON_NODE.
set -uo pipefail
B=<cluster-scratch>/ising_r17; R=<cluster-scratch>/ising
DRY=${DRY:-0}; CORE=${CORE:-$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status | cut -d, -f1 | cut -d- -f1)}; PROCS=$(( ${SLURM_CPUS_ON_NODE:-16} - 2 ))
O=$B/out/gpu_$SLURM_JOB_ID; mkdir -p $O $B/cache/triton $B/cache/nv
cat > $O/run.sh <<EOS
set -uo pipefail
export PATH=$R/venv_trt/bin:\$PATH PYTHONPATH=.:../neural_emulation OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  PYTHONDONTWRITEBYTECODE=1 ICMP_NV=$R/ext/ising_nv/code ICMP_SHOTS=$R/shots TRITON_CACHE_DIR=$B/cache/triton \
  CUDA_CACHE_PATH=$B/cache/nv
cd $B/code/ising_compare || exit 1
A="--freeze data/study4/freeze.json --record $B/r17/r17.json"
X="\$([ $DRY = 1 ] && echo --dry --standin $B/standin)"
st() { local n=\$1; shift; "\$@" > $O/\$n.log 2>&1; local rc=\$?; echo "== \$n exit \$rc \$(date -Is)"; tail -2 $O/\$n.log | cut -c1-300
       [ \$rc = 0 ] || { echo "FAILED at \$n (exit \$rc)"; exit \$rc; }; }
nvidia-smi --query-gpu=name,uuid,driver_version,memory.total,mig.mode.current --format=csv,noheader
st pins python $B/r17/r17.py pins \$A --bundle $B/bundle --s4r3 $B/s4r3 --out $O/pins.json \$([ $DRY = 1 ] && echo --allow-missing)
st ours_val taskset -c $CORE python -m speed2.s19.t4ours val --d 17 --freeze data/study4/freeze.json --root $B/bundle \
  --lib $B/bundle/lib/libs7router.so --devrec $B/bundle/devrec/d17.npz --core $CORE --out $O/ours_val --chunks 0-19
st hopper python $B/r17/r17.py hopper \$A --bundle $B/bundle --s4r3 $B/s4r3 --reg $B/reg --t5 $B/T5 --out $O \
  --procs $PROCS \$X
if [ $DRY = 1 ]; then
  st read python $B/r17/r17.py read \$A --manifest $B/shots/t6dry_manifest.json --shots $B/shots/T6dry --bundle $B/bundle \
    --s4r3 $B/s4r3 --hopper $O/hopper.json --root $B/bundle --lib $B/bundle/lib/libs7router.so \
    --devrec $B/bundle/devrec/d17.npz --procs $PROCS --out $B/out/dry/preds_$SLURM_JOB_ID \$X
else
  st read python $B/r17/r17.py read \$A --manifest $B/shots/t6_manifest.json --shots $B/shots/T6 --bundle $B/bundle \
    --s4r3 $B/s4r3 --hopper $O/hopper.json --root $B/bundle --lib $B/bundle/lib/libs7router.so \
    --devrec $B/bundle/devrec/d17.npz --procs $PROCS --out $B/out/preds
fi
EOS
hostname; date -Is
if [ "${POD:-0}" = 1 ]; then bash $O/run.sh; else apptainer exec --nv -B <cluster-scratch> $R/pyfull.sif bash $O/run.sh; fi; rc=$?
echo "exit $rc"; date -Is; echo JOBDONE; exit $rc
