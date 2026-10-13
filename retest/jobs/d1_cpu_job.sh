# The environment of D1's CPU analysis job for the re-test: the body of D1's scripts/cpu.sbatch (cascade_d1/code.20261011_152939/
# cascade_compare/phase_d/scripts/cpu.sbatch, sha256 ac6ffd13...), D1's tree pinned, no bytecode written into it, the command
# file given by STEP (a script of retest/jobs). Sourced by d1_confirm.sbatch and the replay jobs. PY: the torch CPU venv.
set -euo pipefail
A=<cluster-scratch>/cascade_d1                     # D1's tree: read only
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=$A/code.20261011_152939
PC=$T/cascade_compare/phase_c
PD=$T/cascade_compare/phase_d
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC:$PD:$T/cascade_compare/batchopt OMP_NUM_THREADS=1 \
       PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 QND_RELAY=$B/relay PC_TREE="$(cat $T/DEPLOYED)"
PY="apptainer exec -B <cluster-scratch-root> $I/pyfull.sif $I/venv_gpu/bin/python"
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-?} tree $PC_TREE step $STEP RT=$RT"
source $RT/retest/jobs/$STEP
date; echo JOBDONE
