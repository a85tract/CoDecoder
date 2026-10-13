# common.sh: the environment of the BB72 Cascade jobs R1-R3, sourced by r1.sh, r2.sh and r3.sh. It is the body of D1's
# scripts/gpu.sbatch (the retest's d1_gpu_job.sh), with the pinned tree, job-local copies of the Triton / Inductor caches,
# no bytecode written into the tree, and the launcher's variables. The caller sets (all read only except OUT and JOBTMP):
#   SRT_ROOT      the deployed public tree (the folder that holds speed_rt/)
#   D1_TREE       the pinned D1 tree (its DEPLOYED file names the commit; ising_compare, qldpc_neural, cascade_compare/ in it)
#   PY            the command that runs the BB venv's Python 3.12 (torch 2.11, CUDA 12.8) in its container, with the scratch
#                 file system bound, e.g. "apptainer exec --nv -B SCRATCH IMAGE.sif VENV/bin/python"
#   RELAY_ROOT    the Relay-BP checkout whose test circuits the harnesses read (QND_RELAY)
#   TRITON_SEED, INDUCTOR_SEED   the seed caches of the round, each a directory or a tar file of it (tar -C DIR -cf SEED.tar .),
#                 copied or unpacked into JOBTMP; the originals are only read. A tar file reads in seconds where a directory of
#                 thousands of small files takes minutes on the cluster's file system.
#   SRT_SEEDS, SRT_BEACON (SRT_DRY=1 for a dry record)   the seeds file and the beacon record (srtlib.py)
#   OUT           the job's output directory, which must not exist (drive.py)
#   DRIVE_ARGS    extra arguments of drive.py (dry runs: --reps 1 --n-single 2000 --n-batched 20000 --only ... --force-bad ...)
#   SRT_BURN_S    seconds of burn before each timed window (default 0; tests only)
#   JOBTMP        a job-local scratch directory (default /tmp/srt_cas_<job id>)
set -euo pipefail
: "${SRT_ROOT:?} ${D1_TREE:?} ${PY:?} ${RELAY_ROOT:?} ${TRITON_SEED:?} ${INDUCTOR_SEED:?} ${SRT_SEEDS:?} ${SRT_BEACON:?} ${OUT:?}"
CAS=$SRT_ROOT/speed_rt/cascade
T=$D1_TREE
PC=$T/cascade_compare/phase_c
PD=$T/cascade_compare/phase_d
BO=$T/cascade_compare/batchopt
JC=${JOBTMP:-/tmp/srt_cas_${SLURM_JOB_ID:-manual}}
[ ! -e "$OUT" ] || { echo "$OUT exists: a job writes into a fresh directory"; exit 7; }
case "$(cat $T/DEPLOYED)" in *+dirty*) echo "$T was deployed from a dirty checkout"; exit 1 ;; esac
mkdir -p $JC/triton $JC/inductor
seed() { if [ -f "$1" ]; then tar -C "$2" -xf "$1"; else cp -r "$1/." "$2"; fi; }
seed $TRITON_SEED $JC/triton && seed $INDUCTOR_SEED $JC/inductor
trap 'rm -rf $JC' EXIT
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC:$PD:$BO OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       QND_RELAY=$RELAY_ROOT TRITON_CACHE_DIR=$JC/triton TORCHINDUCTOR_CACHE_DIR=$JC/inductor PC_TREE="$(cat $T/DEPLOYED)" \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
       SRT_BURN_S=${SRT_BURN_S:-0}
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm --format=csv,noheader
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $PC_TREE driver $PC_GPU_DRIVER burn $SRT_BURN_S s out $OUT"
