#!/bin/bash
#SBATCH -J w6prof_check
#SBATCH -p acompile
#SBATCH --qos=compile
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o <cluster-scratch>/cascade_batchopt/logs/%x_%j.out
# The profile's CPU smoke test before the GPU job: the script compiles, its parsing and summary code passes selftest on
# synthetic inputs, and the shots and work steps (no GPU: bb.fused's tables and the FLOP counts on the CPU) run.
set -uo pipefail
A=<cluster-scratch>/cascade_batchopt
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$(readlink $A/code)
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       QND_RELAY=$B/relay TRITON_CACHE_DIR=$A/cache/triton_cpu TORCHINDUCTOR_CACHE_DIR=$A/cache/inductor_cpu
export RUN="apptainer exec -B <cluster-scratch> $I/pyfull.sif" PYBIN=$B/venv_cu128/bin/python \
       O=$A/out/check_${SLURM_JOB_ID:-manual} STEPS="shots work report"
cd $T/ising_compare
$RUN $PYBIN -m py_compile $T/cascade_compare/batchopt/prof_batched.py && echo compiled
$RUN $PYBIN $T/cascade_compare/batchopt/prof_batched.py selftest
bash $T/cascade_compare/batchopt/prof_run.sh
date; echo JOBDONE
