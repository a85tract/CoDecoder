#!/bin/bash
#SBATCH -J srt_r4
#SBATCH -p artxpro6000
#SBATCH --qos=gpu-normal
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH -N 1
#SBATCH -c 24
#SBATCH --mem=100G
#SBATCH -t 00:45:00
#SBATCH --nice=1000
# R4: Cascade at d = 13 on the surface code (e7): 48 pipelines of the d = 13 tree (C1, C4, C5, C2 at p 0.6 % and 0.35 %,
# single and batched, 3 repetitions), rep-major as e7_pipes.sh runs them, under the gate. The caller sets:
#   SRT_ROOT, PY, SRT_SEEDS, SRT_BEACON, OUT   as common.sh (PY runs bb's venv_cu128, usually staged on the node's local disk
#                 and bound over its own path, as the tree's eval/jobs/common.sh does; the staging belongs to the caller)
#   D13_TREE      the pinned d = 13 tree (DEPLOYED = its commit and the release's; surface/ and release/ in it)
#   D13_ROOT      the run directory of the d = 13 study (CASCADE_ROOT: ckpt/CS13/final_ema.pt)
#   D13_LOCK      the study's lock (locked_d13.json); D13_KCACHE: the kernel-cache directory the lock was made with. The
#                 lock's cache digest contains this PATH, so the cache must be visible at it: the caller binds a
#                 writable copy over it in the container (PY) or uses the directory itself.
# Estimate on the RTX PRO 6000 (dry job rtx_r4, scaled; the original e7 took 20 min): about 1.5 min venv and kernel-cache staging
# + 23 min = 25 min, plus 0.5-1 min per replaced process.
set -euo pipefail
: "${SRT_ROOT:?} ${D13_TREE:?} ${D13_ROOT:?} ${D13_LOCK:?} ${D13_KCACHE:?} ${PY:?} ${SRT_SEEDS:?} ${SRT_BEACON:?} ${OUT:?}"
CAS=$SRT_ROOT/speed_rt/cascade
T=$D13_TREE
[ ! -e "$OUT" ] || { echo "$OUT exists: a job writes into a fresh directory"; exit 7; }
case "$(cat $T/DEPLOYED)" in *+dirty*) echo "$T was deployed from a dirty checkout"; exit 1 ;; esac
cd $T
export PYTHONPATH=$T OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 CONVLINES_RELEASE=$T/release \
       CASCADE_ROOT=$D13_ROOT NCCL_IB_DISABLE=1 TRITON_CACHE_DIR=$D13_KCACHE/triton \
       TORCHINDUCTOR_CACHE_DIR=$D13_KCACHE/inductor SRT_BURN_S=${SRT_BURN_S:-0} \
       SV_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm --format=csv,noheader
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $(cat $T/DEPLOYED) driver $SV_GPU_DRIVER burn $SRT_BURN_S s out $OUT"
$PY $CAS/drive.py r4 --tree $T --out $OUT --lock $D13_LOCK ${DRIVE_ARGS:-}
date; echo JOBDONE
