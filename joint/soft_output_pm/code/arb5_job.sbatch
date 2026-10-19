#!/bin/bash
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/qec_arb5/logs/%x_%j.out
# ARB5 (exploratory): arb5_alpine.py at d = 7, 11, 13, 15, 17 (sv2test) and d = 9 (d9stest), one process per distance,
# each pinned to its own core of the job's cpuset, in the release's container and venv, with the deployed SURF-v2 tree.
#   sbatch -p acpu -c 8 --mem=32G -t 01:30:00 -J arb5 arb5_job.sbatch
set -euo pipefail
S=/scratch/alpine/${USER:?}; W=$S/qec_arb5; E=$S/surf_v2/eval; I=$S/ising
T=$E/code_at/ce8780e7357877a91ede59e1ef7ed7bdfd02280e
PY="apptainer exec -B /scratch/alpine $I/pyfull.sif $S/release_r/venv_ising/bin/python"
export PYTHONPATH=$T OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
O=$W/out; mkdir -p $O $W/shots_d9s
echo "node $(hostname) cpus ${SLURM_CPUS_PER_TASK:-?} tree $T"; date -Is
cd $W/code
pids=()
k=0
for d in 17 15 13 11 7; do
  $PY arb5_alpine.py sv2 --d $d --core $k --out $O > $W/logs/d$d.log 2>&1 & pids+=($!); k=$((k+1))
done
$PY arb5_alpine.py d9s --routed $W/in/d9s_routed.npz --manifest $W/in/d9s_test.json --shots $W/shots_d9s --core $k \
  --out $O > $W/logs/d9.log 2>&1 & pids+=($!)
rc=0
for p in "${pids[@]}"; do wait $p || rc=$?; done
tail -n 3 $W/logs/d*.log
date -Is; echo "JOBDONE rc=$rc"
