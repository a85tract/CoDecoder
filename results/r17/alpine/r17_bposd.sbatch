#!/bin/bash
#SBATCH -p aa100
#SBATCH --qos=gpu-normal
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -J r17_bposd
#SBATCH -o <cluster-scratch>/ising_r17/logs/%x_%j.out
# One share of R17's BP-OSD read (PLAN_R17.md sections 4 and 7.1): ARMS on T6 chunks CHUNKS (a part of 0-19), study 4's
# decoder and settings (icmp.s4_bposd), WORKERS decoder processes, worker i on GPU i mod GPUS, each with its own
# 1/WORKERS of the job's cores. The GPU count, its type and the cores are given on the command line, e.g. one A100
# 80GB with four workers, both arms on chunks 0-6 (PLAN_R17.md section 7.4 has the run's three shares):
#   sbatch --gres=gpu:a100_80gb:1 -c 20 --mem=80640M --export=ALL,GPUS=1,WORKERS=4,CHUNKS=0-6 r17/alpine/r17_bposd.sbatch
# A worker needs about 16 GB of host memory at bp_batch_size 2048, and Alpine grants at most 80,640 MiB per A100 80GB:
# five workers at --mem=76G lost one to SIGKILL in the rehearsal, and the frozen decode_parallel4 then waits for it
# forever. Such a job stops printing; cancel it and resubmit with ARMS set to the arms whose output is missing.
# An A100 40GB holds two workers of about 13.5 GB of GPU memory.
# DRY=1: the dry stand-in T6dry, LIMIT shots per arm (default 2,500, to measure the rate; LIMIT=0 decodes every shot
# of the share), into SHARES (default $B/out/dry/bposd_<job>; a rehearsal's share jobs name one directory, as the
# run's write into $B/out/bposd_shares); nothing counts.
set -uo pipefail
B=<cluster-scratch>/ising_r17; R=<cluster-scratch>/ising
DRY=${DRY:-0}; GPUS=${GPUS:?}; WORKERS=${WORKERS:?}; ARMS=${ARMS:-minllr bp60}; CHUNKS=${CHUNKS:-}
if [ "$DRY" = 1 ]; then M=$B/shots/t6dry_manifest.json; S=$B/shots/T6dry; O=${SHARES:-$B/out/dry/bposd_$SLURM_JOB_ID}
else M=$B/shots/t6_manifest.json; S=$B/shots/T6; O=$B/out/bposd_shares; fi
mkdir -p $O
cd $B/code/ising_compare || exit 1
hostname; date -Is; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
apptainer exec --nv -B <cluster-scratch> $R/pyfull.sif env PYTHONPATH=.:../neural_emulation OMP_NUM_THREADS=1 \
  PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MINLLR_GPUS=$GPUS $R/venv_cudaq/bin/python $B/r17/r17.py bposd \
  --freeze data/study4/freeze.json --record $B/r17/r17.json --manifest $M --shots $S --out $O --arms $ARMS \
  --procs $WORKERS $([ -n "$CHUNKS" ] && echo --chunks $CHUNKS) $([ "$DRY" = 1 ] && echo --limit ${LIMIT:-2500})
rc=$?; echo "exit $rc"; date -Is; echo JOBDONE; exit $rc
