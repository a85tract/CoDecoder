#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -J h100t_screen
#SBATCH -o <cluster-scratch>/cascade_h100t/logs/%x_%j.out
# The first H100 pod's screen (phase_c/H100_TIMING.md), CPU only, after C4b (submitted with --dependency=afterok:<C4b>): the
# H100's outputs on C4b's check shots against C4b's reads on the RTX PRO 6000 (pc_h100.py screen, per setting), then
# the summary with the screen's verdicts (pc_h100.py summarize). Reads the primary run's lock, frozen paths and reads
# in place, read-only; writes only under $A/screen. Its tree is $A/code (h100_deploy.sh with TO=alpine); the H100's
# results are $A/out (uploaded by h100_pull.sh once the pod's run is copied and verified).
set -euo pipefail
A=<cluster-scratch>/cascade_h100t; R=<cluster-scratch>/cascade_w4
B=<cluster-scratch>/bb; I=<cluster-scratch>/ising
T=$(readlink $A/code); PC=$T/cascade_compare/phase_c
cd $T/ising_compare
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 QND_RELAY=$B/relay
PYR="apptainer exec -B <cluster-scratch> $I/pyfull.sif $B/venv/bin/python"
SP=$R/out/speed/speed_components.json
echo "node $(hostname) cpus ${SLURM_CPUS_ON_NODE:-} tree $(cat $T/DEPLOYED) $(date -Is)"
[ -f $A/out/END ] || { echo "no H100 results in $A/out (h100_pull.sh uploads them after the pod's run)"; exit 1; }
(cd $A/out && sha256sum --quiet --strict -c SHA256SUMS) || { echo "$A/out does not verify"; exit 1; }
mkdir -p $A/screen
for S in uniform depspam_fm; do
  [ -f $R/out/check_$S.json ] || { echo "C4b's check_$S.json is missing: its read may be incomplete"; exit 1; }
  $PYR $PC/pc_h100.py screen --outputs $A/out/outputs/$S --read $R/out/check/$S/read --locked $R/out/locked_$S.json \
      --speed $SP --out $A/screen/screen_$S.json
done
$PYR $PC/pc_h100.py summarize --rows $A/out/rows.jsonl --speed $SP --env $A/out/env.json \
    --locked $R/out/locked_uniform.json $R/out/locked_depspam_fm.json \
    --screen $A/screen/screen_uniform.json $A/screen/screen_depspam_fm.json --out $A/screen/h100_speed.json \
  | tee $A/screen/summary.txt
(cd $A/screen && sha256sum screen_uniform.json screen_depspam_fm.json h100_speed.json summary.txt > SHA256SUMS)
echo "$(date -Is) job ${SLURM_JOB_ID:-}" > $A/screen/END
date; echo JOBDONE
