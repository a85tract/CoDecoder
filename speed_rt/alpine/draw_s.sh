#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH --nice=1000
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -J srt_surfdraw
# speed_rt/alpine/draw_s.sh BEACON OUTDIR [dry]: the surface stage's draw (the stage's section, "The timing shots"), once, on an
# CPU node (x86-64 with AVX2; the study ran it on a cluster node), after the stage's beacon record exists: the six timing sets srt-surf-d7 ... d17 (20 chunks
# of 10,000 shots each) and the twelve reference sets srt-ref-d<d>-p<3|6> (one chunk each) of speed_rt/sets_surf.json, sampled by
# Stim 1.16.0 (numpy 2.2.6) inside the surface environment (PYSURF below: Stim 1.16.0, numpy 2.2.6,
# PyMatching 2.4.0). Steps: the record loads (speed_rt.beacon check; a dry record by the dry loader); the draw of both kinds
# (speed_rt/draw.py checks the host: x86-64 with AVX2, and the versions); the check (every file's sha256 against its manifest and
# every seed recomputed from the record). A non-zero exit means nothing may be timed on these shots. The output directory must
# not exist.
# The caller sets SRT_REL (the deployed copy of the public tree at the draw's commit: surface_v2/, speed_rt/ and evaluation/ are
# needed) and may set PYSURF (the command that runs the surface environment's Python).
#   PYSURF="..." sbatch -o LOG --export=ALL,SRT_REL=... alpine/draw_s.sh BEACON OUTDIR [dry]     (the folder is named for the cluster of the run)
# dry: BEACON is a dry record (python -m speed_rt.beacon dry --round R --out DRY.json); the output is never a timing input.
# Outputs in OUTDIR: shots/ (the surface chunks), refshots/ (the reference chunks), srt-surf.json and srt-ref.json (the manifests:
# every chunk's set, index, seed and sha256, per d the digest of the whole set, the circuits' sha256), SHA256SUMS, draw.txt.
set -euo pipefail
BEACON=$(realpath -s "${1:?usage: draw_s.sh BEACON OUTDIR [dry]}"); OUT=${2:?usage: draw_s.sh BEACON OUTDIR [dry]}; MODE=${3:-run}
case "$MODE" in run|dry) ;; *) echo "mode is run or dry"; exit 2 ;; esac
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REL=${SRT_REL:-$(cd "$HERE/../.." && pwd)}
[ -d "$REL/surface_v2" ] && [ -d "$REL/speed_rt" ] || { echo "no deployed tree: set SRT_REL"; exit 2; }
# The surface environment's Python (Stim 1.16.0, numpy 2.2.6, PyMatching 2.4.0): PYSURF, a command; on a cluster that runs Python in a
# container, e.g. PYSURF="apptainer exec -B <cluster-scratch> <image> <venv>/bin/python" in the submitting shell's environment.
PY=${PYSURF:-python3}
[ ! -e "$OUT" ] || { echo "$OUT exists: the draw writes into a fresh directory"; exit 2; }
mkdir -p "$OUT" && OUT=$(cd "$OUT" && pwd)
export PYTHONPATH=$REL/surface_v2:$REL PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
DRY=""; [ "$MODE" = dry ] && DRY="--dry"
cd "$REL"
exec > >(tee "$OUT/draw.txt") 2>&1
# nothing below prints or records an absolute path: the log, the manifests and SHA256SUMS name files relative to the output directory
echo "draw $MODE commit $(cat $REL/DEPLOYED 2>/dev/null || echo none) node $(hostname) job ${SLURM_JOB_ID:-none} $(date -Is)"
if [ "$MODE" = run ]; then $PY -m speed_rt.beacon check --beacon "$BEACON"; fi
cd "$OUT"
$PY -m speed_rt.draw --beacon "$BEACON" $DRY --kind surface --out shots --manifest srt-surf.json
$PY -m speed_rt.draw --beacon "$BEACON" $DRY --kind ref --out refshots --manifest srt-ref.json
$PY -m speed_rt.draw check --dir shots --manifest srt-surf.json --beacon "$BEACON" $DRY
$PY -m speed_rt.draw check --dir refshots --manifest srt-ref.json --beacon "$BEACON" $DRY
find shots refshots -type f | sort | xargs sha256sum > SHA256SUMS
echo "DRAW DONE $(date -Is): $(wc -l < SHA256SUMS) chunk files"
