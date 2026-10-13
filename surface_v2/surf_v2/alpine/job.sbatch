#!/bin/bash
#SBATCH -N 1
#SBATCH -o /scratch/alpine/%u/surf_v2/eval/logs/%x_%j.out
# One SURF-v2 evaluation job on Alpine: the commands of surf_v2/args/$CMD (several joined by ':'), run from the
# immutable deployment $E/code_at/$PIN that surf_v2/alpine/deploy.sh made. Before anything runs, pin_check proves the
# tree is PIN's (sv2/jobs/job.sbatch's check): DEPLOYED is PIN, SHA256SUMS hashes to PINSUMS, every file has its
# sha256, no file is missing or added, nothing is writable. Every job is submitted with --nice=1000. $PY is the release
# reproduction's venv in the pyfull container (release_r/venv_ising: requirements.txt's pins, Python 3.12.14, plus the
# Ising extras), $PYCQ CUDA-Q QEC's (BP-OSD; numpy 2.5.3, no torch).
#   sbatch -p acpu --qos=cpu-normal -c 16 --mem=64G -t 01:00:00 -J sv2e_dry --export=ALL,CMD=alpine_dry_cpu.txt,PIN=<commit>,PINSUMS=<sha256> surf_v2/alpine/job.sbatch
set -euo pipefail
S=/scratch/alpine/${USER:?}; E=$S/surf_v2/eval; I=$S/ising
pin_check() {
  local t=$1 pin=$2 sums=$3
  [ "$(cat $t/DEPLOYED 2>/dev/null)" = "$pin" ] || { echo "PIN CHECK $t: DEPLOYED is not $pin"; return 1; }
  [ "$(sha256sum < $t/SHA256SUMS | cut -c1-64)" = "$sums" ] || { echo "PIN CHECK $t: SHA256SUMS is not $sums"; return 1; }
  (cd $t && sha256sum --quiet --strict -c SHA256SUMS) || { echo "PIN CHECK $t: a file differs from SHA256SUMS"; return 1; }
  [ "$(cd $t && find . -type f ! -path ./SHA256SUMS ! -path ./DEPLOYED | sed 's|^\./||' | LC_ALL=C sort)" = \
    "$(sed 's/^[0-9a-f]*  //' $t/SHA256SUMS | LC_ALL=C sort)" ] || { echo "PIN CHECK $t: files other than SHA256SUMS lists"; return 1; }
  [ -z "$(find $t -perm /222 -print -quit)" ] || { echo "PIN CHECK $t: writable"; return 1; }
}
[ -n "${PIN:-}" ] && [ -n "${PINSUMS:-}" ] || { echo "PIN and PINSUMS are not set (surf_v2/alpine/deploy.sh prints them)"; exit 1; }
T=$E/code_at/$PIN
pin_check $T $PIN $PINSUMS || exit 1
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
V=$S/release_r/venv_ising
PY="apptainer exec --nv -B /scratch/alpine $I/pyfull.sif $V/bin/python"
PYCQ="apptainer exec --nv -B /scratch/alpine $I/pyfull.sif $I/venv_cudaq/bin/python"
O=$E/out/${SLURM_JOB_NAME}_$SLURM_JOB_ID; mkdir -p $O $E/tmp/mpl $E/tmp/cache
# every cache on scratch: the account's home (2 GB) was full on 12 October
export PYTHONPATH=$T OMP_NUM_THREADS=${OMP:-1} PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=$E/tmp/mpl \
       ISING_DECODING_DIR=$I/ext/ising_nv XDG_CACHE_HOME=$E/tmp/cache CUDA_CACHE_PATH=$E/tmp/cache/nv \
       TRITON_CACHE_DIR=$E/tmp/cache/triton
echo "node $(hostname) gpu '$GPU' tree $T PIN $PIN: checked; venv $V cmd $CMD cpus ${SLURM_CPUS_PER_TASK:-?} out $O"
cd $T
for c in ${CMD//:/ }; do echo "== $c"; eval "$(cat surf_v2/args/$c)"; done
date -Is; echo JOBDONE
