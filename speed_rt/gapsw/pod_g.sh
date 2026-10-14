#!/bin/bash
# pod_g.sh STEP...: the Decoder Switching stage's session on its H100 pod's Xeon (CPU work only; the GPU stays idle), in order:
#   setup      the venv (uv's CPython 3.12.3 with the three registered pins: numpy 2.2.6, Stim 1.16.0, PyMatching 2.4.0) and the
#              checks of the deployed tree, the draw (check every chunk against the manifest) and the probe's shots
#   prescreen  speed_rt.gapsw.drive_gs prescreen -> $W/prescreen (the timed core, the /proc/stat self-test, the host probe and the
#              session's reference); exit 4 when the pod fails: the pod is replaced
#   masks      untimed, before any timed window: the front end's batch decode of every d's timing shots in a fork pool -> $W/masks
#   timed      speed_rt.gapsw.drive_gs run -> $W/session (each process on the quietest physical core at its start; three interleaved repetitions of the 12 gap-switching
#              slots (no Grace reference slots: the stage's fit is the surface family's); one process at a time)
#   pack       $W/session, $W/prescreen and $W/masks (json and mask files) into $W/gs_records.tgz for the analysis
# Layout: $W/rel the deployed tree (the public tree of speed_rt/gapsw/pack.sh; $W/rel/DEPLOYED its commit), $W/venv, $W/shots/srt the
# drawn chunk files and $W/shots/srt-gs.json their manifest, $W/probe/d9_p0.006_X_val_000.npz SURF-v2's validation chunk 0 of d = 9.
# PAR=1 (NW, CPUS): the parallel driver, as on winserver. DRY=1: dry runs (a dry draw; REPS repetitions (default 1), CH: only the first CH chunks of the DS sets (default all), FORCE=SUBSTR=N,...:
# the first N attempts of the slots named are forced to be rejected, ONLY="SUBSTR ...": only the slots whose ids contain one of them; nothing registered). Run long steps detached and poll.
set -euo pipefail
W=${W:-/workspace/gs}; REL=$W/rel; PY=${PY:-$W/venv/bin/python}
export PYTHONPATH=$REL:$REL/surface_v2 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SH=$W/shots/srt; MAN=$W/shots/srt-gs.json; PF=$W/probe/d9_p0.006_X_val_000.npz
PAR=${PAR:-0}                                    # PAR=1: the parallel driver (drive_par.py): NW (default 8) timed processes at once, each on its own physical core
cpus() { [ -n "${CPUS:-}" ] || CPUS=$($PY -m speed_rt.gapsw.drive_par cpus --n ${NW:-8}); }     # the listed CPUs (needs the venv)
DRYARGS=""; CHARGS=""
[ "${DRY:-0}" = 1 ] && { DRYARGS="--dry --reps ${REPS:-1}${CH:+ --chunks $CH}${FORCE:+ --force-bad $FORCE}${ONLY:+ --only $ONLY}"; CHARGS="${CH:+--chunks $CH}"; }
cd $REL
echo "pod $(hostname) tree $(cat $REL/DEPLOYED) cpu $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs) allowed $(nproc --all) online"
for step in "$@"; do
  echo "== $step $(date -Is)"
  case $step in
  setup)
    command -v uv > /dev/null || python3 -m pip install -q uv
    [ -d $W/venv ] || uv venv -q --seed --python 3.12.3 $W/venv
    $PY -c "import sys; assert sys.version_info[:3] == (3, 12, 3), sys.version"
    $PY -m pip install -q numpy==2.2.6 stim==1.16.0 PyMatching==2.4.0
    $PY -c "import numpy, stim, pymatching; assert (numpy.__version__, stim.__version__, pymatching.__version__) == ('2.2.6', '1.16.0', '2.4.0'); print('venv ok', numpy.__version__, stim.__version__, pymatching.__version__)"
    (cd $REL && sha256sum --quiet -c SUMS && echo "tree ok: $(wc -l < SUMS) files")
    $PY -m speed_rt.gapsw.draw_gs check --dir $SH --manifest $MAN
    $PY -c "
import numpy as np
from speed_rt.gapsw import gslib as G
from convlines import circuits
det, obs = G.load_chunk('$PF', circuits.circuit(9).num_detectors)
assert G.shots_sha256(np.ascontiguousarray(det), obs) == G.PROBE_SHOTS_SHA256, 'the probe shots are not the registered chunk'
print('probe shots ok')" ;;
  prescreen)
    if [ "$PAR" = 1 ]; then cpus; echo "parallel: CPUS $CPUS"; $PY -m speed_rt.gapsw.drive_par prescreen --cpus $CPUS --probe-file $PF --out $W/prescreen
    else $PY -m speed_rt.gapsw.drive_gs prescreen --probe-file $PF --out $W/prescreen; fi ;;
  masks)
    NC=$($PY -c "import os; print(len(os.sched_getaffinity(0)))"); WK=${WORKERS:-$(( NC > 24 ? 16 : NC - 4 ))}
    for d in 7 9 11 13 15 17; do
      $PY -m speed_rt.gapsw.gs_time mask --d $d --shots $SH --manifest $MAN --workers $WK --out $W/masks $CHARGS
    done ;;
  timed)
    if [ "$PAR" = 1 ]; then cpus; DRV="drive_par run --cpus $CPUS"; else DRV="drive_gs run"; fi
    $PY -m speed_rt.gapsw.$DRV --shots $SH --manifest $MAN --masks $W/masks --probe-file $PF \
      --probe-ref $(cat $W/prescreen/PROBE_REF) --out $W/session $DRYARGS ;;
  pack)
    (cd $W && tar czf gs_records.tgz session prescreen masks/*.json && sha256sum gs_records.tgz) ;;
  *) echo "unknown step $step"; exit 1 ;;
  esac
done
date -Is; echo JOBDONE
