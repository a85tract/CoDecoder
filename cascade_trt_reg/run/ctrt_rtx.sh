#!/bin/bash
#SBATCH -J ctrt_rtx
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=96G
#SBATCH -t 02:30:00
#SBATCH --nice=1000
# ctrt_rtx.sh STEP... (CASCADE-TRT-REG, one Slurm job on one whole RTX PRO 6000 of the cluster's partition artxpro6000):
# stage 1's job form (speed_rt/cascade/r1.sh, common.sh and their staging: the venv unpacked on the node's local disk and
# bound over its own path in the cluster's container, the kernel caches in a job-local directory, the harness
# environment of code/env.sh), then ctrt_session.sh STEP... with CARD=rtx. The caller sets (the cluster's paths):
#   VENV_TAR, VENV_PATH   the TensorRT venv's tar.zst (of a directory named as VENV_PATH's last part) and its path
#   SIF, BIND             the container image and the file system to bind (the scratch root)
#   P, PL, OUT            the public tree at B, this card's plans, the records (RECORDS/RTX)
#   QND_RELAY, CASCADE_ROOT   as code/env.sh (the Relay-BP checkout; ckpt/CS13/final_ema.pt -> the public CS13)
#   CACHE_SEED, CACHE_OUT optional: a tar of Triton / Inductor caches (triton/, inductor/) to start from, and one to
#                         write at the end (a dry job's caches, compiled on the dry shots, for the timing jobs to start from)
#   SRT_SEEDS, SRT_BEACON default to B's seeds and beacon record in P; DRIVE_ARGS, CHECK_ARGS, SRT_DRY for dry runs
set -uo pipefail
: "${VENV_TAR:?} ${VENV_PATH:?} ${SIF:?} ${BIND:?} ${P:?} ${PL:?} ${OUT:?} ${QND_RELAY:?} ${CASCADE_ROOT:?}"
VL=/tmp/ctrt_${SLURM_JOB_ID:-manual}; mkdir -p $VL/cache/triton $VL/cache/inductor; trap 'rm -rf $VL' EXIT
t0=$(date +%s)
zstd -dcq $VENV_TAR | tar -C $VL -xf - && cp $SIF $VL/c.sif || { echo "staging failed"; exit 1; }
[ -z "${CACHE_SEED:-}" ] || tar -C $VL/cache -xf $CACHE_SEED || { echo "cache seed failed"; exit 1; }
export PY="apptainer exec --nv -B $BIND -B $VL/$(basename $VENV_PATH):$VENV_PATH $VL/c.sif $VENV_PATH/bin/python"
echo "staged venv and container in $(( $(date +%s) - t0 )) s on $(hostname)"
BBT=$P/cascade/bb72; T13=$P/cascade/d13; PC=$BBT/cascade_compare/phase_c; PD=$BBT/cascade_compare/phase_d
export PYTHONPATH=$BBT/ising_compare:$BBT/qldpc_neural:$PC:$PD:$BBT/cascade_compare/batchopt:$T13:$P/cascade_trt_reg/code \
       OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 NCCL_IB_DISABLE=1 \
       TRITON_CACHE_DIR=$VL/cache/triton TORCHINDUCTOR_CACHE_DIR=$VL/cache/inductor CONVLINES_RELEASE=$P/evaluation \
       QND_RELAY CASCADE_ROOT CTRT_PUB=$P CARD=rtx SRT_CARD=rtx P PL OUT \
       SRT_SEEDS=${SRT_SEEDS:-$P/cascade_trt_reg/seeds_ctrt.json} SRT_BEACON=${SRT_BEACON:-$P/cascade_trt_reg/beacon_ctrt.json}
bash $P/cascade_trt_reg/run/ctrt_session.sh "$@"
rc=$?
[ -z "${CACHE_OUT:-}" ] || tar -C $VL/cache -cf $CACHE_OUT triton inductor
echo "JOBDONE rc=$rc $(date -Is)"
exit $rc
