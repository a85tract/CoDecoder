#!/bin/bash
#SBATCH -J b2m2rep
#SBATCH -p acpu
#SBATCH --qos=cpu-testing
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem=8G
#SBATCH -t 00:15:00
#SBATCH -o <cluster-scratch>/cascade_batchopt/logs/%x_%j.out
# Milestone 2 (batchopt/M2.md): the combined report of several job directories on a CPU node (m2.py report reads the rows
# and phase C's files and needs no GPU; the login node has neither apptainer nor Python 3.12), with the gate's verdict
# on each directory that holds a screen or tail checks. In the tree $A/code (or M2_TREE), as m2.sbatch runs it. DIRS
# may be separated by spaces, colons or commas; pass it in the environment, not in --export (which splits at commas):
#   DIRS="$A/out/m2_<job> $A/out/m2_<job> ..." OUT=$A/out/m2_<name> sbatch m2_report.sbatch
set -uo pipefail
A=<cluster-scratch>/cascade_batchopt
B=<cluster-scratch>/bb
I=<cluster-scratch>/ising
T=${M2_TREE:-$(readlink $A/code)}
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c:$T/cascade_compare/batchopt \
       PYTHONUNBUFFERED=1 PC_TREE="$(cat $T/DEPLOYED)"
PY="apptainer exec -B <cluster-scratch> $I/pyfull.sif $B/venv_cu128/bin/python $T/cascade_compare/batchopt/m2.py"
DIRS=$(echo ${DIRS:?} | tr ' :' ',,')
echo "node $(hostname) tree $PC_TREE dirs $DIRS out ${OUT:?}"
cd $T/ising_compare
mkdir -p "$OUT"
$PY report --dirs "$DIRS" --out "$OUT"; rc=$?
for d in ${DIRS//,/ }; do
  if [ -e $d/screen.jsonl ] || ls $d/tail_*.json > /dev/null 2>&1; then echo "gate $d: $($PY gate --out $d | tail -1)"; fi
done
date; echo JOBDONE rc=$rc
exit $rc
