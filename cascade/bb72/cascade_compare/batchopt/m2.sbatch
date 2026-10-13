#!/bin/bash
#SBATCH -J b2m2
#SBATCH -p artxpro6000
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 00:45:00
#SBATCH --nice=200
#SBATCH -o <cluster-scratch>/cascade_batchopt/logs/%x_%j.out
# Milestone 2 (batchopt/M2.md) on a full RTX PRO 6000: m2_run.sh in the tree $A/code (batchopt/deploy.sh), with bb's
# venv_cu128 in the pyfull container as phase C's gpu.sbatch runs it, the outputs in $A/out/m2_<job> (refs in
# $A/out/m2_refs). The steps come from the submitting environment (m2_run.sh). Triton's and Inductor's caches are
# $A/cache_m2/{triton,inductor}, seeded once from the formal run's copies ($A/cache), so that today's kernels and
# Cascade's compiled path are the binaries and the autotuning choices phase C ran; the formal run's directory is only
# read. Override -t, -c and --mem on the sbatch line (screens and timings at most 2 hours, development 45 minutes).
set -uo pipefail
A=<cluster-scratch>/cascade_batchopt
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=${M2_TREE:-$(readlink $A/code)}                   # M2_TREE pins a deployed tree (code.<time>) for queued jobs
for c in triton inductor; do
  [ -d $A/cache_m2/$c ] || { mkdir -p $A/cache_m2 && cp -r $A/cache/$c $A/cache_m2/$c; }
done
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c:$T/cascade_compare/batchopt \
       OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$B/relay TRITON_CACHE_DIR=$A/cache_m2/triton \
       TORCHINDUCTOR_CACHE_DIR=$A/cache_m2/inductor PC_TREE="$(cat $T/DEPLOYED)" \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
export RUN="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif" PYBIN=$B/venv_cu128/bin/python \
       O=$A/out/m2_${SLURM_JOB_ID:-manual} REFS=$A/out/m2_refs
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm --format=csv,noheader
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $PC_TREE out $O steps ${STEPS:-?}"
cd $T/ising_compare
bash $T/cascade_compare/batchopt/m2_run.sh; rc=$?
date; echo JOBDONE rc=$rc
exit $rc
