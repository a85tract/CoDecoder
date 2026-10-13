#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -J r17_analyze
#SBATCH -o <cluster-scratch>/ising_r17/logs/%x_%j.out
# R17's analysis (PLAN_R17.md section 5.3). WHAT=merge: the BP-OSD shares joined. WHAT=inventory: the pre-commit, the
# sha256 of every input (committed and pushed before the analysis). WHAT=analyze: the one-time analysis into
# $B/out/final (refused if it exists or if any input differs from the committed inventory INV).
#   sbatch --export=ALL,WHAT=analyze,INV=$B/r17/run/final_inputs.sha256,CPU=<cpu_term.json>,HOP=<hopper.json> ...
# DRY=1: the dry stand-in T6dry. merge: SHARES (a dry BP-OSD job's directory) into $B/out/dry/bposd_merged_<job>;
# inventory into $B/out/dry/final_inputs_<job>.sha256; analyze: PREDS, BPOSD (a merged directory), CPU, HOP and,
# if given, INV (then checked as in the run) into $B/out/dry/final_<job>.
set -uo pipefail
B=<cluster-scratch>/ising_r17; R=<cluster-scratch>/ising
DRY=${DRY:-0}; WHAT=${WHAT:?merge, inventory or analyze}
PY="apptainer exec -B <cluster-scratch> $R/pyfull.sif $R/venv_gpu/bin/python"
export PYTHONPATH=.:../neural_emulation OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
cd $B/code/ising_compare || exit 1
A="--freeze data/study4/freeze.json --record $B/r17/r17.json"
if [ "$DRY" = 1 ]; then T="--manifest $B/shots/t6dry_manifest.json --shots $B/shots/T6dry"; P=${PREDS:-}; BP=${BPOSD:-}
  SH=${SHARES:-}; MO=$B/out/dry/bposd_merged_$SLURM_JOB_ID
else T="--manifest $B/shots/t6_manifest.json --shots $B/shots/T6"; P=$B/out/preds; BP=$B/out/bposd
  SH=$B/out/bposd_shares; MO=$B/out/bposd; fi
case $WHAT in
merge) mkdir -p $MO; $PY $B/r17/r17.py bposd-merge $A $T --dir ${SH:?} --out $MO ;;
inventory|analyze)
  I="$A $T --s4r3 $B/s4r3 --preds ${P:?} --bposd ${BP:?} --cpu-term ${CPU:?} --hopper ${HOP:?}"
  if [ $WHAT = inventory ]; then
    IO=$B/out/final_inputs.sha256; [ "$DRY" = 1 ] && IO=$B/out/dry/final_inputs_$SLURM_JOB_ID.sha256
    $PY $B/r17/r17.py inventory $I --out $IO && sha256sum $IO
  elif [ "$DRY" = 1 ]; then
    $PY $B/r17/r17.py analyze $I $([ -n "${INV:-}" ] && echo --inventory $INV) --out $B/out/dry/final_$SLURM_JOB_ID
  else $PY $B/r17/r17.py analyze $I --inventory ${INV:?} --out $B/out/final; fi ;;
esac
rc=$?; echo "exit $rc"; date -Is; echo JOBDONE; exit $rc
