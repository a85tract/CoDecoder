#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:40:00
#SBATCH -J r17_sample
#SBATCH -o <cluster-scratch>/ising_r17/logs/%x_%j.out
# T5 chunks 0-19 drawn again for the hopper check (each file equal to S4-R3's manifest), then T6 (PLAN_R17.md
# section 2): 100 chunks of 10,000 shots, seeds 91,170,000 + j, into $B/shots/T6 (which must not exist), manifest
# $B/shots/t6_manifest.json, all read-only. An Alpine AMD EPYC node (x86-64 with AVX2; r17.py sample refuses others).
# DRY=1: the stand-in instead of T6 (V17b chunks 0-1 under their own seeds, checked against V17b) into $B/shots/T6dry.
#   sbatch [--export=ALL,DRY=1] r17/alpine/r17_sample.sbatch
set -uo pipefail
B=<cluster-scratch>/ising_r17; R=<cluster-scratch>/ising
DRY=${DRY:-0}
PY="apptainer exec -B <cluster-scratch> $R/pyfull.sif $R/venv_gpu/bin/python"
export PYTHONPATH=.:../neural_emulation OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       ICMP_NV=$R/ext/ising_nv/code ICMP_SHOTS=$R/shots
cd $B/code/ising_compare || exit 1
A="--freeze data/study4/freeze.json --record $B/r17/r17.json"
st() { local n=$1; shift; "$@" > $B/logs/sample_${SLURM_JOB_ID}_$n.log 2>&1; local rc=$?; echo "== $n exit $rc $(date -Is)"
       tail -2 $B/logs/sample_${SLURM_JOB_ID}_$n.log | cut -c1-300; [ $rc = 0 ] || { echo "FAILED at $n"; echo JOBDONE; exit $rc; }; }
hostname; grep -m1 "model name" /proc/cpuinfo
st pins $PY $B/r17/r17.py pins $A --bundle $B/bundle --s4r3 $B/s4r3 --allow-missing
st t5regen $PY $B/r17/r17.py t5regen $A --s4r3 $B/s4r3 --chunks 0-19 --out $B/T5
mkdir -p $B/shots
if [ "$DRY" = 1 ]; then st sample_dry $PY $B/r17/r17.py sample $A --out $B/shots --dry
else st sample $PY $B/r17/r17.py sample $A --out $B/shots; fi
sha256sum $B/shots/t6*_manifest.json; date -Is; echo JOBDONE
