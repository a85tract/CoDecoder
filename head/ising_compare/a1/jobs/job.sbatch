#!/bin/bash
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/rev_a1/logs/%x_%j.out
# One job of the readout ablation A1 (revision/REVISION_PLAN.md 2.1): the commands in a1/jobs/args/$CMD, run from the
# immutable deployment $A/code_at/$PIN that a1/jobs/deploy.sh made (gate finding 3 on revision/A1_READ_PLAN.md). Before
# anything runs, pin_check proves the tree is PIN's: DEPLOYED is PIN, the sha256 of SHA256SUMS is PINSUMS (both printed
# by deploy.sh), every file has its sha256, no file is missing or added, and nothing is writable. A later deployment of
# another commit never changes a queued or running job. $PY = ising_compare's torch venv in the pyfull container
# (venv_gpu: torch 2.11.0, CUDA 12.6; on a Blackwell RTX PRO 6000 or its slices the same torch for CUDA 12.8).
# Partition, GPU, cores, memory and time come from sbatch, e.g.
#   sbatch -p acpu --qos=cpu-normal -c 2 --mem=16G -t 00:15:00 -J a1_barrier --export=ALL,CMD=barrier_check.txt,PIN=<commit>,PINSUMS=<sha256> a1/jobs/job.sbatch
set -euo pipefail
export A=<cluster-scratch>/rev_a1
I=<cluster-scratch>/ising
# pin_check TREE PIN PINSUMS: TREE is PIN's deployment, untouched (tests/a1_pin_check.sh runs this function on altered
# copies of a deployment, each of which it must refuse)
pin_check() {
  local t=$1 pin=$2 sums=$3
  [ "$(cat $t/DEPLOYED 2>/dev/null)" = "$pin" ] || { echo "PIN CHECK $t: DEPLOYED is not $pin"; return 1; }
  [ "$(sha256sum < $t/SHA256SUMS | cut -c1-64)" = "$sums" ] || { echo "PIN CHECK $t: SHA256SUMS is not $sums"; return 1; }
  (cd $t && sha256sum --quiet --strict -c SHA256SUMS) || { echo "PIN CHECK $t: a file differs from SHA256SUMS"; return 1; }
  [ "$(cd $t && find . -type f ! -path ./SHA256SUMS ! -path ./DEPLOYED | sed 's|^\./||' | LC_ALL=C sort)" = \
    "$(sed 's/^[0-9a-f]*  //' $t/SHA256SUMS | LC_ALL=C sort)" ] || { echo "PIN CHECK $t: files other than SHA256SUMS lists"; return 1; }
  [ -z "$(find $t -perm /222 -print -quit)" ] || { echo "PIN CHECK $t: writable"; return 1; }
}
[ -n "${PIN:-}" ] && [ -n "${PINSUMS:-}" ] || { echo "PIN and PINSUMS are not set (a1/jobs/deploy.sh prints them)"; exit 1; }
T=$A/code_at/$PIN
pin_check $T $PIN $PINSUMS || exit 1
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/neural_emulation:$T/qldpc_neural OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       PYTHONDONTWRITEBYTECODE=1 \
       ICMP_NV=$I/ext/ising_nv/code ICMP_SHOTS=$I/shots ICMP_CKPT=$A/ckpt/surface ICMP_OUT=$A/out/surface \
       BB_SHOTS=<cluster-scratch>/bb/shots QND_RELAY=<cluster-scratch>/bb/relay BB_CKPT=$A/ckpt/bb72 \
       BB_OUT=$A/out/bb72 BB_PREDS=$A/preds_bb A1_SETS=$A/sets A1_SETS_ROWS=$A/sets/rows_dry.csv:$A/sets/rows.csv \
       TRITON_CACHE_DIR=$A/triton_cache TORCHINDUCTOR_CACHE_DIR=$A/inductor_cache
mkdir -p $A/logs $A/ckpt $A/out $A/sets $A/reads $A/trunk $A/analysis
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
case "$GPU" in *Blackwell*) V=<cluster-scratch>/bb/venv_cu128 ;; *) V=$I/venv_gpu ;; esac
echo "node $(hostname) gpu '$GPU' mig '${CUDA_VISIBLE_DEVICES:-}' tree $T PIN $PIN PINSUMS $PINSUMS: checked; venv $V cmd $CMD cpus ${SLURM_CPUS_PER_TASK:-?}"
nvidia-smi -L 2>/dev/null || true
PY="apptainer exec --nv -B <cluster-scratch> $I/pyfull.sif $V/bin/python"
for c in ${CMD//:/ }; do eval "$(cat a1/jobs/args/$c)"; done         # CMD: one args file, or several joined by ':'
date; echo JOBDONE
