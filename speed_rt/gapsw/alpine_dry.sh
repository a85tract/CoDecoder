#!/bin/bash
# alpine_dry.sh STEP...: the Decoder Switching stage's functional dry runs on Alpine's CPU (non-timing work: dry shots from a dry
# beacon, never a timing shot). Run inside an acpu job (see the sbatch line); needs $T the deployed tree (speed_rt/gapsw/pack.sh,
# unpacked), $G the work directory, $DRYBEACON the dry beacon record (speed_rt/beacon.py dry), $PROBE SURF-v2's validation
# chunk 0 of d = 9, $PY the release venv in the pyfull container (CPU only).
#   sbatch -p acpu --qos=cpu-normal --nice=1000 -c 8 --mem=32G -t 01:00:00 -J gs_dry --wrap 'bash alpine_dry.sh draw masks ...'
#   draw      the dry draw: every set, the first CH chunks of the DS sets (default 2) -> $G/shots
#   masks     the front end's batch decode and the routed mask of every d -> $G/masks
#   smoke     (after pre) one front end and one routed-PM term (a quick functional test)
#   pre       the pre-screen -> $G/prescreen$TAG
#   session   the dry session: drive_gs run --dry (REPS repetitions, default 1; FORCE=slot=N,... forces rejections) -> $G/session$TAG
#   analysis  analysis_gs --dry --fit $FIT twice -> $G/an1, $G/an2, and their diff ($FIT: the stage's Grace fit file, a synthetic one in a dry)
set -euo pipefail
: ${T:?} ${G:?} ${DRYBEACON:?} ${PROBE:?} ${PY:?}
export PYTHONPATH=$T:$T/surface_v2 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
CH=${CH-2}; REPS=${REPS:-1}; TAG=${TAG:-}          # CH unset: 2 chunks of the DS sets; CH set but empty: all of them (the draw's 5 chunks)
SH=$G/shots/srt; MAN=$G/shots/srt-gs.json
cd $T
CORE() { awk '/Cpus_allowed_list/{print $2}' /proc/self/status | cut -d, -f1 | cut -d- -f1; }
for step in "$@"; do
  echo "== $step $(date -Is)"
  case $step in
  draw)
    $PY -m speed_rt.gapsw.draw_gs --beacon $DRYBEACON --dry ${CH:+--chunks $CH} ${SETS:+--sets $SETS} --out $SH --manifest $MAN
    $PY -m speed_rt.gapsw.draw_gs check --dir $SH --manifest $MAN ;;
  masks)
    for d in 7 9 11 13 15 17; do
      $PY -m speed_rt.gapsw.gs_time mask --d $d --shots $SH --manifest $MAN ${CH:+--chunks $CH} --workers ${WORKERS:-6} --out $G/masks
    done ;;
  smoke)
    C=$(cat $G/prescreen$TAG/CORE)
    A="--shots $SH --manifest $MAN ${CH:+--chunks $CH} --core $C --rep 1 --probe-file $PROBE --probe-ref $(cat $G/prescreen$TAG/PROBE_REF)"
    $PY -m speed_rt.gapsw.gs_time fe --d 7 --masks $G/masks $A --out $G/smoke/fe_d7
    $PY -m speed_rt.gapsw.gs_time term --d 7 --masks $G/masks $A --out $G/smoke/term_d7 ;;
  pre)
    $PY -m speed_rt.gapsw.drive_gs prescreen --probe-file $PROBE --out $G/prescreen$TAG ;;
  session)
    $PY -m speed_rt.gapsw.drive_gs run --dry ${CH:+--chunks $CH} --reps $REPS --shots $SH --manifest $MAN --masks $G/masks \
      --probe-file $PROBE --probe-ref $(cat $G/prescreen$TAG/PROBE_REF) --force-bad "${FORCE:-}" \
      --out $G/session$TAG ;;
  analysis)
    for k in 1 2; do
      $PY -m speed_rt.gapsw.analysis_gs --dry --fit ${FIT:?} --session $G/session$TAG --prescreen $G/prescreen$TAG --manifest $MAN --masks $G/masks \
        --out $G/an$k$TAG
    done
    diff -r $G/an1$TAG $G/an2$TAG && echo "BYTE-IDENTICAL: the two runs' $(ls $G/an1$TAG | wc -l) files" ;;
  *) echo "unknown step $step"; exit 1 ;;
  esac
done
date -Is; echo JOBDONE
