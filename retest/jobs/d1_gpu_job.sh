# The environment of D1's GPU jobs for the re-test: the body of D1's scripts/gpu.sbatch (cascade_d1/code.20261011_152939/
# cascade_compare/phase_d/scripts/gpu.sbatch, sha256 bb96c95e...), with D1's tree pinned, the Triton / Inductor caches
# copied from D1's seed caches (read only) into this job's own directory, no bytecode written into D1's tree, and
# the command file replaced by d1_read_rt.sh. Sourced by d1_R1.sbatch and d1_R2.sbatch (full RTX PRO 6000, one job per p).
set -euo pipefail
A=<cluster-scratch>/cascade_d1                     # D1's tree and caches: read only
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$A/code.20261011_152939                                 # D1's deployed tree: DEPLOYED = "net 64b2e25d d1 F"
PC=$T/cascade_compare/phase_c
PD=$T/cascade_compare/phase_d
V=$B/venv_cu128
JC=<cluster-scratch>/retest/jobcache/d1/${SLURM_JOB_ID:-manual}
mkdir -p $JC && cp -r $A/cache/triton $JC/triton && cp -r $A/cache/inductor $JC/inductor
trap 'rm -rf $JC' EXIT
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC:$PD:$T/cascade_compare/batchopt OMP_NUM_THREADS=1 \
       PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 QND_RELAY=$B/relay TRITON_CACHE_DIR=$JC/triton \
       TORCHINDUCTOR_CACHE_DIR=$JC/inductor PC_TREE="$(cat $T/DEPLOYED)" \
       PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm --format=csv,noheader
PY="apptainer exec --nv -B <cluster-scratch-root> $I/pyfull.sif $V/bin/python"
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $PC_TREE venv $V driver $PC_GPU_DRIVER read P=$P RT=$RT"
source $RT/retest/jobs/${STEP:-d1_read_rt.sh}
date; echo JOBDONE
