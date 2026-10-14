#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH --nice=1000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 00:45:00
#SBATCH -J srt_bbdraw
# SPEED-RT's draw of the three BB timing sets on an Alpine CPU node (x86-64 with AVX2), once, after the beacon record exists:
# srt-bb72-p3 (bb72, p 0.003, split "test", 20 chunks of 10,000 shots), srt-bb72-p2 (0.002, "test", 30), srt-gross-p4 (bb144,
# 0.004, "test4", 15), each chunk sampled by the frozen bb.shots from its seed (speed_rt.sets) through the shim
# (speed_rt/bb/srt_bb.py; the sets are those of speed_rt/bb/sets_bb.json, the BB stage's own sets file), inside Alpine's pycpu (bb/venv: Stim 1.16.0, numpy 2.2.6). Steps: the record loads; the host is
# x86-64 with AVX2 and the versions above; the host check (the original test sets, regenerated here with their frozen
# seeds, equal the published arrays: retest/replay/bb_replay.py shots, as RETEST's draw); the draw; the seed check (every
# manifest seed recomputed from the record, every file's sha256, every set complete). A non-zero exit means nothing may be timed
# on these shots. The output directory must not exist.
# Environment, set by the caller: SRT_REL = the deployed copy of the public tree (needed under sbatch, which runs a copy of this
# script); SRT_ROOT = a directory for caches (XDG_CACHE_HOME, MPLCONFIGDIR); PYCPU = the command that runs the BB CPU venv's
# Python (on Alpine the pycpu wrapper: apptainer exec of the container with bb/venv: Stim 1.16.0, numpy 2.2.6); QND_RELAY = the
# Relay-BP clone at d185194 whose tests/testdata holds the circuits.
#   sbatch -o LOG --export=ALL,SRT_REL=...,SRT_ROOT=...,PYCPU=...,QND_RELAY=... draw.sh BEACON OUTDIR [dry]
#   bash draw.sh BEACON OUTDIR [dry]          (inside a job)
# dry: BEACON is a dry record (python -m speed_rt.beacon dry --round R --out DRY.json); chunks of $DRY_SHOTS (default 200) shots.
# Outputs in OUTDIR: shots/ (BB_SHOTS), manifest_bb72.csv, manifest_gross.csv (bb.shots rows: file, sha256, shots, seed, code, p,
# split, chunk, host, cpu, stim), env_check.json, hostcheck_bb72.json, hostcheck_gross.json, seedcheck.json, SHA256SUMS, draw.txt.
set -euo pipefail
# realpath -s, not readlink -f: the container binds the logical scratch path, not its resolved target
BEACON=$(realpath -s "${1:?usage: draw.sh BEACON OUTDIR [dry]}"); OUT=${2:?usage: draw.sh BEACON OUTDIR [dry]}; MODE=${3:-run}
case "$MODE" in run|dry) ;; *) echo "mode is run or dry"; exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REL=${SRT_REL:-$(cd "$HERE/../.." && pwd)}
[ -d "$REL/bb/bb72/ising_compare" ] || { echo "no deployed tree: set SRT_REL"; exit 2; }
: "${SRT_ROOT:?SRT_ROOT: a directory for caches}" "${PYCPU:?PYCPU: the command that runs the BB CPU venv python}" "${QND_RELAY:?QND_RELAY: the Relay-BP clone}"
PYC=$PYCPU
SHIM=$REL/speed_rt/bb/srt_bb.py
[ ! -e "$OUT" ] || { echo "$OUT exists: the draw writes into a fresh directory"; exit 2; }
mkdir -p "$OUT/shots" "$OUT/hostcheck" && OUT=$(cd "$OUT" && pwd)
mkdir -p "$SRT_ROOT/cache/xdg" "$SRT_ROOT/cache/mpl"
export QND_RELAY PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
       MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 XDG_CACHE_HOME=$SRT_ROOT/cache/xdg MPLCONFIGDIR=$SRT_ROOT/cache/mpl
export SRT_BEACON=$BEACON BB_SHOTS=$OUT/shots
if [ "$MODE" = dry ]; then export SRT_DRY=1 SRT_DRY_SHOTS=${DRY_SHOTS:-200}; else unset SRT_DRY SRT_DRY_SHOTS; fi
treedir() { case "$1" in bb72) echo $REL/bb/bb72/ising_compare ;; bb144) echo $REL/bb/gross_stage4/ising_compare ;; esac; }
mfile() { case "$1" in bb72) echo $OUT/manifest_bb72.csv ;; bb144) echo $OUT/manifest_gross.csv ;; esac; }
step() { echo "== $1 $(date -Is)"; }
echo "draw $MODE rel $REL commit $(cat $REL/DEPLOYED 2>/dev/null || echo none) node $(hostname) job ${SLURM_JOB_ID:-none} out $OUT"
cd "$REL/bb/bb72/ising_compare"; export PYTHONPATH=.:../qldpc_neural

step record
$PYC $SHIM --check-beacon
step host
$PYC $SHIM --env-check $OUT/env_check.json > /dev/null
step "host check: the original test sets regenerated here equal the published arrays"
H=$OUT/hostcheck
( unset SRT_BEACON SRT_DRY SRT_DRY_SHOTS; export SRT_ORIGINAL_SEEDS=1 BB_SHOTS=$H/shots
  cd "$(treedir bb72)" || exit 1; export SRT_MANIFEST=$H/rows_bb72.csv
  $PYC $SHIM bb.shots --code bb72 --p 0.003 --split test --chunks 0-19 --rows $H/rows_bb72.csv > /dev/null < /dev/null || exit 1
  $PYC $SHIM bb.shots --code bb72 --p 0.002 --split test --chunks 0-29 --rows $H/rows_bb72.csv > /dev/null < /dev/null || exit 1
  $PYC $SHIM $REL/retest/replay/bb_replay.py shots $OUT/hostcheck_bb72.json < /dev/null | cut -c1-300; [ "${PIPESTATUS[0]}" = 0 ] || exit 1
  cd "$(treedir bb144)" || exit 1; export SRT_MANIFEST=$H/rows_gross.csv
  $PYC $SHIM bb.shots --code bb144 --p 0.004 --split test4 --chunks 0-14 --rows $H/rows_gross.csv > /dev/null < /dev/null || exit 1
  $PYC $SHIM $REL/retest/replay/bb_replay.py shots $OUT/hostcheck_gross.json < /dev/null | cut -c1-300; [ "${PIPESTATUS[0]}" = 0 ] || exit 1
) || { echo HOSTCHECK_FAIL; exit 1; }
rm -rf "$H/shots"

step draw
while read -r name code p split n; do
  cd "$(treedir "$code")"; export SRT_MANIFEST=$(mfile "$code")
  $PYC $SHIM bb.shots --code "$code" --p "$p" --split "$split" --chunks 0-$((n - 1)) --rows "$SRT_MANIFEST" < /dev/null | tail -1 | cut -c1-120
  echo "$name drawn ($n chunks)"
done < <(cd "$REL/bb/bb72/ising_compare" && $PYC $SHIM --list-sets)

step "seed check"
cd "$REL/bb/bb72/ising_compare"; unset SRT_MANIFEST
$PYC $SHIM --check-manifest $OUT/manifest_bb72.csv $OUT/manifest_gross.csv --out $OUT/seedcheck.json || { echo SEEDCHECK_FAIL; exit 1; }
( cd $OUT && cp "$BEACON" ./beacon_record.json && sha256sum manifest_bb72.csv manifest_gross.csv seedcheck.json env_check.json beacon_record.json \
    hostcheck_bb72.json hostcheck_gross.json > SHA256SUMS )
{ echo "mode $MODE commit $(cat $REL/DEPLOYED 2>/dev/null || echo none) job ${SLURM_JOB_ID:-none} node $(hostname) $(date -Is)"
  echo "files $(ls $OUT/shots | wc -l) bytes $(du -sb $OUT/shots | cut -f1)"; } > $OUT/draw.txt
cat $OUT/SHA256SUMS | cut -c1-90
echo "DRAW_END $(date -Is) files $(ls $OUT/shots | wc -l)"
