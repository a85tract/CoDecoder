# Sourced inside a Slurm job by d13_read.sh, d13_analysis.sh and the d = 13 replay job. It gives the pinned tree's
# command files the environment that scripts/gpu.sbatch of the evaluation gave them (R, B, I, T, V, PYTHONPATH,
# CONVLINES_RELEASE, CASCADE_ROOT, OMP_NUM_THREADS, NCCL_IB_DISABLE), with the tree fixed to the lock's. RT = the deployed
# freeze tree (contains retest/); the Blackwell node's venv is bb/venv_cu128, as gpu.sbatch chose.
R=<cluster-scratch>/cascade_w5
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$R/code.20261012_060258
TREE=$T; export TREE
V=$B/venv_cu128
RTD=${RT:?RT: the deployed freeze tree (contains retest/)}/retest
SHIM=$RTD/shim/rt_surface.py
unset DRY RUN                      # common.sh would switch to the evaluation's dry mode
cd $T
export PYTHONPATH=$T OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 CONVLINES_RELEASE=$T/release CASCADE_ROOT=$R \
       TRITON_CACHE_DIR=$R/triton_cache TORCHINDUCTOR_CACHE_DIR=$R/inductor_cache \
       NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-} job ${SLURM_JOB_ID:-} tree $(cat $T/DEPLOYED) venv $V RT=$RT"
