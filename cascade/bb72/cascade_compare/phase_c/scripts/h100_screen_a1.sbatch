#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J h100b_screen
#SBATCH -o <cluster-scratch>/cascade_h100b/logs/%x_%j.out
# The second H100 pod's screen (phase_c/H100_TIMING.md, "Amendment 1"), CPU only, in place of the first pod's job CJ36 (whose tree
# had no pc_h100.py): both H100 runs' outputs on C4b's check shots against C4b's reads on the RTX PRO 6000
# (pc_h100.py screen, per setting), each against the lock it was made with, then each run's summary with its verdicts
# (pc_h100.py summarize):
#   first   the first pod's outputs and rows ($A0/out) against the first lock ($R/out/locked_<setting>.json)
#   amend1  the second pod's outputs and rows ($A/out) against Amendment 1's lock ($R/out/amend1/locked_<setting>.json);
#           the second C4b reuses the first C4b's reads, so these are the same reads
# Reads the primary run's locks, frozen paths and reads and the first pod's results in place, read-only; writes only under
# $A/screen. Its tree is $A/code (h100_deploy.sh with TO=alpine A=$A, from ising-cascade-h100t2); the second pod's results
# are $A/out (uploaded by h100b_pull.sh once the pod's run is copied and verified).
set -euo pipefail
A=<cluster-scratch>/cascade_h100b; A0=<cluster-scratch>/cascade_h100t
R=<cluster-scratch>/cascade_w4; B=<cluster-scratch>/bb; I=<cluster-scratch>/ising
T=$(readlink $A/code); PC=$T/cascade_compare/phase_c
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$B/relay
PYR="apptainer exec -B <cluster-scratch> $I/pyfull.sif $B/venv/bin/python"
SP=$R/out/speed/speed_components.json
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-} tree $(cat $T/DEPLOYED) $(date -Is)"
for d in $A0/out $A/out; do
  [ -f $d/END ] || { echo "no H100 results in $d"; exit 1; }
  (cd $d && sha256sum --quiet --strict -c SHA256SUMS) || { echo "$d does not verify"; exit 1; }
done
for S in uniform depspam_fm; do
  [ -f $R/out/check_$S.json ] || { echo "C4b's check_$S.json is missing: its read may be incomplete"; exit 1; }
done
rm -rf $A/screen; mkdir -p $A/screen/first $A/screen/amend1
for run in first amend1; do
  if [ $run = first ]; then H=$A0/out; LK=$R/out; else H=$A/out; LK=$R/out/amend1; fi
  for S in uniform depspam_fm; do
    $PYR $PC/pc_h100.py screen --outputs $H/outputs/$S --read $R/out/check/$S/read --locked $LK/locked_$S.json \
        --speed $SP --out $A/screen/$run/screen_$S.json
  done
  $PYR $PC/pc_h100.py summarize --rows $H/rows.jsonl --speed $SP --env $H/env.json \
      --locked $LK/locked_uniform.json $LK/locked_depspam_fm.json \
      --screen $A/screen/$run/screen_uniform.json $A/screen/$run/screen_depspam_fm.json \
      --out $A/screen/$run/h100_speed.json | tee $A/screen/$run/summary.txt
done
(cd $A/screen && find . -type f ! -name SHA256SUMS ! -name END -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo "$(date -Is) job ${SLURM_JOB_ID:-}" > $A/screen/END
date; echo JOBDONE
