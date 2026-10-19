#!/bin/bash
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1 -c 16 --mem=96G -t 01:10:00
#SBATCH -J arbcp_bb72
#SBATCH -o <cluster-scratch>/qec_arbcp/logs/%x_%j.out
# ARBCP, BB72: cp_run.py at p = 0.2 % and 0.3 % on one full RTX PRO 6000 (D1's card). The environment of the re-test's
# compiled reads (retest/jobs/d1_gpu_job.sh) rebuilt: D1's venv bb/venv_cu128 (torch 2.11.0+cu128, triton 3.6.0) on
# Python 3.12.14 (uv's build; D1's container image is gone), the release tree of the re-test's freeze, D1's seed caches
# (speedrt/cascade/caches/r1_*.tar, copied from cascade_d1/cache on 13 Oct) unpacked into the job.
set -euo pipefail
Q=<cluster-scratch>/qec_arbcp
F=$Q/keep/freeze/cascade/bb72
PC=$F/cascade_compare/phase_c
PD=$F/cascade_compare/phase_d
PY=$Q/uvpy/cpython-3.12.14-linux-x86_64-gnu/bin/python3.12
JC=/tmp/arbcp_kc_${SLURM_JOB_ID:-manual}
mkdir -p $JC/triton $JC/inductor
tar -C $JC/triton -xf $Q/keep/caches/r1_triton.tar
tar -C $JC/inductor -xf $Q/keep/caches/r1_inductor.tar
trap 'rm -rf $JC' EXIT
export PYTHONPATH=$F/ising_compare:$F/qldpc_neural:$PC:$PD:$Q/code:$Q/keep/venv_cu128/lib/python3.12/site-packages \
       QND_RELAY=$Q/keep/relay OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CUDA_DEVICE_MAX_CONNECTIONS=32 \
       TRITON_CACHE_DIR=$JC/triton TORCHINDUCTOR_CACHE_DIR=$JC/inductor \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total --format=csv,noheader
echo "node $(hostname) job ${SLURM_JOB_ID:-} $(date -Is)"
cd $F/ising_compare
for P in ${PS:-0.002 0.003}; do
  $PY $Q/code/cp_run.py --p $P --data $Q/data/$P --out $Q/out/bb72 --c14 $Q/keep/freeze/weights/cascade_bb72/C14p.pt \
    --s55 $Q/keep/freeze/weights/cascade_bb72/S55p_deep.pt --lock $PD/d1_locked_depspam_fm.json
  touch $Q/out/bb72/done_$P
done
date; echo JOBDONE
