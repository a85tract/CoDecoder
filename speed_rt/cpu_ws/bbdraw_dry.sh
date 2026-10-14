#!/bin/bash
# DRY RUNS ONLY (speed_rt/cpu_ws): speed_rt/bb/draw.sh in dry mode without its host check (the regeneration of the
# ORIGINAL test sets compared with the published arrays). That check belongs to the registered draw on the cluster; the CPU host
# only receives copies of those shots. Steps kept: the record, the env check, the dry draw, the seed check.
#   SRT_REL=... SRT_ROOT=... PYCPU=... QND_RELAY=... DRY_SHOTS=N bbdraw_dry.sh BEACON OUTDIR
set -euo pipefail
# realpath -s, not readlink -f: the container binds the logical scratch path, not its resolved target
BEACON=$(realpath -s "${1:?usage: draw.sh BEACON OUTDIR [dry]}"); OUT=${2:?usage: draw.sh BEACON OUTDIR [dry]}; MODE=dry
case "$MODE" in run|dry) ;; *) echo "mode is run or dry"; exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REL=${SRT_REL:-$(cd "$HERE/../.." && pwd)}
[ -d "$REL/bb/bb72/ising_compare" ] || { echo "no deployed tree: set SRT_REL"; exit 2; }
: "${SRT_ROOT:?SRT_ROOT: a directory for caches}" "${PYCPU:?PYCPU: the command that runs the BB CPU venv python}" "${QND_RELAY:?QND_RELAY: the Relay-BP clone}"
PYC=$PYCPU
SHIM=$REL/speed_rt/bb/srt_bb.py
[ ! -e "$OUT" ] || { echo "$OUT exists: the draw writes into a fresh directory"; exit 2; }
mkdir -p "$OUT/shots" && OUT=$(cd "$OUT" && pwd)
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
echo "host check skipped (dry run on the CPU host)"

step draw
while read -r name code p split n; do
  cd "$(treedir "$code")"; export SRT_MANIFEST=$(mfile "$code")
  $PYC $SHIM bb.shots --code "$code" --p "$p" --split "$split" --chunks 0-$((n - 1)) --rows "$SRT_MANIFEST" < /dev/null | tail -1 | cut -c1-120
  echo "$name drawn ($n chunks)"
done < <(cd "$REL/bb/bb72/ising_compare" && $PYC $SHIM --list-sets)

step "seed check"
cd "$REL/bb/bb72/ising_compare"; unset SRT_MANIFEST
$PYC $SHIM --check-manifest $OUT/manifest_bb72.csv $OUT/manifest_gross.csv --out $OUT/seedcheck.json || { echo SEEDCHECK_FAIL; exit 1; }
( cd $OUT && cp "$BEACON" ./beacon_record.json && sha256sum manifest_bb72.csv manifest_gross.csv seedcheck.json env_check.json beacon_record.json > SHA256SUMS )
{ echo "mode dry (no host check) commit $(cat $REL/DEPLOYED 2>/dev/null || echo none) job ${SLURM_JOB_ID:-none} node $(hostname) $(date -Is)"
  echo "files $(ls $OUT/shots | wc -l) bytes $(du -sb $OUT/shots | cut -f1)"; } > $OUT/draw.txt
cat $OUT/SHA256SUMS | cut -c1-90
echo "DRAW_END $(date -Is) files $(ls $OUT/shots | wc -l)"
