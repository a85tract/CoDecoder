#!/bin/bash
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/cascade_w2/logs/%x_%j.out
# The curve sets (bb/cascade/CASCADE_CURVES.md): one CPU job. Evaluates bb/cascade/args/$CMD with $PY = the BB sub-route's CPU venv
# (stim 1.16.0, relay-bp 0.2.2, ldpc 2.4.1; read only) in the pyfull container and $P = the CPUs allocated.
#   sbatch -p acpu --qos=cpu-normal -c 16 --mem=32G -J NAME -t HH:MM:SS --export=ALL,CMD=NAME.txt bb/cascade/cpu.sbatch
set -euo pipefail
W=<cluster-scratch>/cascade_w2
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$(readlink $W/code)                    # pin this job to the tree deployed at its start
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$B/relay W2_DIR=$W \
       BB_CKPT=$B/ckpt
P=${SLURM_CPUS_PER_TASK:-$SLURM_CPUS_ON_NODE}
lscpu | grep 'Model name'; echo "node $(hostname) cpus $P tree $(cat ../DEPLOYED) cmd $CMD"
PY="apptainer exec -B <cluster-scratch> $I/pyfull.sif $B/venv/bin/python"
eval "$(cat bb/cascade/args/$CMD)"
date; echo JOBDONE
