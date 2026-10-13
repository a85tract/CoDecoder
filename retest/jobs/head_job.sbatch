#!/bin/bash
#SBATCH --qos=gpu-normal
#SBATCH --nice=1000
#SBATCH -N 1
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# One job of the re-test of the output-head comparison (A1; paper Tables head and headcmp): A1's own job.sbatch
# (a1/jobs/job.sbatch of the pinned tree) with the paths of the re-test. The commands in $RT/retest/jobs/$CMD (head_*.sh,
# one file or several joined by ':') run from the immutable deployment $A/code_at/$PIN that A1's jobs ran from; before
# anything runs, pin_check proves the tree is PIN's (A1's function, unchanged): DEPLOYED is PIN, the sha256 of SHA256SUMS
# is PINSUMS, every file has its sha256, no file is missing or added, nothing is writable. Every a1 command goes
# through the shim ($PYS = $PY + $RT/retest/shim/rt_a1.py), which gives the test kinds their new seeds and points the sets
# at $RT_OUT/head/sets. $A (A1's tree) is read only here: pinned code, checkpoints, the rule decisions, the original
# calibration chunks and reads. Everything the re-test writes lies under $RT_OUT/head.
# Environment (set by jobs/head_submit.sh): RT (the deployed freeze tree, which contains retest/), RT_BEACON, RT_SEEDS,
# RT_OUT; RT_DRY=1 only with the dry beacon and seeds (the shim refuses otherwise). Partition, GPU, cores, memory and time
# come from sbatch (head_submit.sh); every job has --nice=1000.
set -euo pipefail
export A=<cluster-scratch>/rev_a1
I=<cluster-scratch>/ising
PIN=2e69738e74be71c5cfbcb6661ecfa28a54048877                     # A1's freeze G: the tree its draw and reads ran from
PINSUMS=0098f4b42c153b69d34d7594c85898da96ab1c89285d0442c767a458a5d4b94e
# pin_check TREE PIN PINSUMS (a1/jobs/job.sbatch, unchanged)
pin_check() {
  local t=$1 pin=$2 sums=$3
  [ "$(cat $t/DEPLOYED 2>/dev/null)" = "$pin" ] || { echo "PIN CHECK $t: DEPLOYED is not $pin"; return 1; }
  [ "$(sha256sum < $t/SHA256SUMS | cut -c1-64)" = "$sums" ] || { echo "PIN CHECK $t: SHA256SUMS is not $sums"; return 1; }
  (cd $t && sha256sum --quiet --strict -c SHA256SUMS) || { echo "PIN CHECK $t: a file differs from SHA256SUMS"; return 1; }
  [ "$(cd $t && find . -type f ! -path ./SHA256SUMS ! -path ./DEPLOYED | sed 's|^\./||' | LC_ALL=C sort)" = \
    "$(sed 's/^[0-9a-f]*  //' $t/SHA256SUMS | LC_ALL=C sort)" ] || { echo "PIN CHECK $t: files other than SHA256SUMS lists"; return 1; }
  [ -z "$(find $t -perm /222 -print -quit)" ] || { echo "PIN CHECK $t: writable"; return 1; }
}
for v in RT RT_BEACON RT_SEEDS RT_OUT CMD; do [ -n "${!v:-}" ] || { echo "$v is not set (jobs/head_submit.sh sets it)"; exit 1; }; done
[ -f "$RT/retest/shim/rt_a1.py" ] && [ -f "$RT_BEACON" ] && [ -f "$RT_SEEDS" ] || { echo "RT, RT_BEACON or RT_SEEDS: no such file"; exit 1; }
case "$(readlink -f $RT_OUT)/" in "$(readlink -f $A)/"*) echo "RT_OUT is under A1's tree"; exit 1 ;; esac
T=$A/code_at/$PIN
pin_check $T $PIN $PINSUMS || exit 1
cd $T/ising_compare
OUT=$RT_OUT/head
mkdir -p $OUT/sets $OUT/reads/surface $OUT/reads/bb72 $OUT/unused
export RT RT_BEACON RT_SEEDS RT_OUT OUT PIN
[ -z "${RT_DRY:-}" ] || export RT_DRY
# A1's environment (a1/jobs/job.sbatch) with the writable names moved under $OUT: the sets (rows: the original calibration
# chunks' rows, then the new test chunks' rows), the caches, and the training-side output directories no a1 command writes to
export PYTHONPATH=$T/ising_compare:$T/neural_emulation:$T/qldpc_neural OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       PYTHONDONTWRITEBYTECODE=1 \
       ICMP_NV=$I/ext/ising_nv/code ICMP_SHOTS=$I/shots ICMP_CKPT=$A/ckpt/surface ICMP_OUT=$OUT/unused/icmp_out \
       BB_SHOTS=<cluster-scratch>/bb/shots QND_RELAY=<cluster-scratch>/bb/relay BB_CKPT=$A/ckpt/bb72 \
       BB_OUT=$OUT/unused/bb_out BB_PREDS=$OUT/unused/preds_bb A1_SETS=$OUT/sets A1_SETS_ROWS=$RT/retest/head/cal_chunks.csv:$OUT/sets/rows.csv \
       TRITON_CACHE_DIR=$OUT/triton_cache TORCHINDUCTOR_CACHE_DIR=$OUT/inductor_cache
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
case "$GPU" in *Blackwell*) V=<cluster-scratch>/bb/venv_cu128 ;; *) V=$I/venv_gpu ;; esac
echo "node $(hostname) gpu '$GPU' mig '${CUDA_VISIBLE_DEVICES:-}' tree $T PIN $PIN PINSUMS $PINSUMS: checked; venv $V cmd $CMD cpus ${SLURM_CPUS_PER_TASK:-?} dry '${RT_DRY:-}'"
echo "shim sha256 $(sha256sum < $RT/retest/shim/rt_a1.py | cut -c1-64) RT $RT RT_OUT $RT_OUT"
nvidia-smi -L 2>/dev/null || true
PY="apptainer exec --nv -B <cluster-scratch-root> $I/pyfull.sif $V/bin/python"
SHIM=$RT/retest/shim/rt_a1.py
PYS="$PY $SHIM"
for c in ${CMD//:/ }; do eval "$(cat $RT/retest/jobs/$c)"; done         # CMD: one args file, or several joined by ':'
date; echo JOBDONE
