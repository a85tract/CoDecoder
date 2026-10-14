#!/bin/bash
# SPEED-RT CPU half on the Windows CPU host: everything a timed process reads goes into the tmpfs /dev/shm/srt first
# (untimed).
# Runs inside the WSL2 distro as root; every step logs to $QRT/logs/prep_<step>.log (QRT: ${QRT:-$HOME/qrt}) and stops
# at the first error.
# On the Linux host (README "Linux host"), as the harness's user, with SRT_TMPFS=/dev/shm/srt_box (the tmpfs root) and
# SRT_CFG=cfg_box.json (the config `shots` imports with).
#   prep.sh env                 the pinned environment (setup_env.sh restore, else build), the clock probe, cpu_ws
#   prep.sh code TGZ [COMMIT]   the code: TGZ holds surface_v2/ evaluation/ bb/ retest/replay/ speed_rt/ of the frozen
#                               commit (git archive); unpacked to /dev/shm/srt/code (an existing tree is kept aside);
#                               COMMIT (the full 40-hex hash; a dry label without it) goes into code/DEPLOYED, which
#                               the records name
#   prep.sh relay TGZ           Relay-BP's tests/testdata at d185194 (the three BB circuits and the helper) -> /dev/shm/srt/relay
#   prep.sh val                 validation chunk 0 at d = 9 and 13 (convlines.testset val): the reference segment, the
#                               filler and Decoder Switching's probe file
#   prep.sh surface BEACON [--dry]   the surface timing sets and the 18 reference points (speed_rt.draw) and the
#                               view; with --dry also the synthetic routing masks (mkmask_dry.py, dry runs only)
#   prep.sh gs BEACON [--dry] [CHUNKS]   Decoder Switching's sets (draw_gs; CHUNKS per set, dry only) and its masks
#                               (gs_time mask, 8 workers, untimed)
#   prep.sh bb BEACON [--dry] [SHOTS]   the BB sets (speed_rt/bb/draw.sh; dry: SHOTS per chunk) and, dry only,
#                               synthetic predictions (bbpreds_dry.py)
#   prep.sh shots RUN [FROM]    the registered shots from the hand-off: units surface_shots (shots/ = the timing
#                               chunks, refshots/ = the reference chunks, srt-surf.json, srt-ref.json; or the same
#                               as surf/ and ref/ subtrees of the data folder) and gs (shots/, srt-gs.json, or gs/...;
#                               the registered probe chunk d9_p0.006_X_val_000.npz at the top or in probe/, if sent),
#                               copied into the tmpfs and checked by `harness.py import` (records in RUN/handoff/);
#                               linked into the data folder; the probe chunk also as the reference segment's d = 9
#                               chunk; the surface view; Decoder Switching's masks (8 workers)
#   prep.sh handoff_dry DIR     DRY ONLY: the dry inputs laid out as the hand-off folder DIR (a "speedrt_in_dry" folder,
#                               normally the Windows one through /mnt/c), which `harness.py run --handoff-from DIR` imports
# For the registered session the shots come from the stage's registered draws (copied, not redrawn: npz bytes differ),
# and the masks, residuals and predictions from the GPU pass, through the hand-off folder that the harness copies into
# the tmpfs at a window boundary (STAGE2_common C2); then only env, code, relay and val run here.
set -euo pipefail
R=${SRT_TMPFS:-/dev/shm/srt}; C=$R/code; D=$R/data; PY=$R/venv/bin/python; HERE=$(dirname "$(readlink -f "$0")")
QRT=${QRT:-$HOME/qrt}
mkdir -p "$R" "$D" "$QRT/logs"
step=$1; shift
exec > >(tee -a "$QRT/logs/prep_$step.log") 2>&1
echo "=== prep $step $* $(date -Is)"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
case "$step" in
env)
  bash "$HERE/setup_env.sh" restore 2>/dev/null || bash "$HERE/setup_env.sh" build
  mkdir -p "$R/cpu_ws" && cp -a "$HERE"/. "$R/cpu_ws/"
  gcc -O2 -static -o "$R/cpu_ws/clk" "$R/cpu_ws/clk.c" 2>/dev/null || gcc -O2 -o "$R/cpu_ws/clk" "$R/cpu_ws/clk.c"
  echo "clock probe: $("$R/cpu_ws/clk" 50000000) MHz on one vCPU"
  # the C libraries in the tmpfs ($R/lib, the interpreter's loader too); cfg_ws.json gives every process LD_LIBRARY_PATH
  bash "$R/cpu_ws/tmpfs_libs.sh" "$R"
  # the shared objects the timed processes load from the disk, locked in memory (pinlibs.py), once per distro boot
  if ! { [ -e "$R/pinlibs.pid" ] && kill -0 "$(cat "$R/pinlibs.pid")" 2>/dev/null; }; then
    setsid -f bash -c "echo \$\$ > $R/pinlibs.pid; exec $PY $R/cpu_ws/pinlibs.py $R > $QRT/logs/pinlibs.log 2>&1 < /dev/null"
    sleep 8
  fi
  echo "pinlibs pid $(cat "$R/pinlibs.pid"): $(cut -c1-60 $QRT/logs/pinlibs.log)"
  ;;
code)
  [ -d "$C" ] && mv "$C" "$C.old.$(date +%s)"
  mkdir -p "$C" && tar -C "$C" -xzf "$1"
  echo "${2:-dry-$(sha256sum "$1" | cut -c1-12)}" > "$C/DEPLOYED"     # no git call from a timed process
  sha256sum "$1"; du -sh "$C"; cat "$C/DEPLOYED" 2>/dev/null || true
  ;;
relay)
  rm -rf "$R/relay"; mkdir -p "$R/relay"; tar -C "$R/relay" -xzf "$1"
  ls "$R/relay/tests/testdata/bicycle_bivariate" | wc -l
  ;;
val)
  rm -rf "$D/val" "$D/probe"; mkdir -p "$D/val" "$D/probe"
  (cd "$C/surface_v2" && PYTHONPATH=. "$PY" -m convlines.testset val --d 9 --chunks 0 --out "$D/val" \
     && PYTHONPATH=. "$PY" -m convlines.testset val --d 13 --chunks 0 --out "$D/val")
  cp "$D/val/d9_p0.006_X_val_000.npz" "$D/probe/"
  sha256sum "$D"/val/*.npz
  ;;
surface)
  B=$1; DRY=${2:-}
  cd "$C"
  export PYTHONPATH=$C/surface_v2:$C
  "$PY" -m speed_rt.draw --beacon "$B" $DRY --kind surface --out "$D/surf/shots" --manifest "$D/surf/srt-surf.json"
  "$PY" -m speed_rt.draw --beacon "$B" $DRY --kind ref --out "$D/ref/shots" --manifest "$D/ref/srt-ref.json"
  "$PY" -m speed_rt.draw view --shots "$D/surf/shots" --out "$D/surf/view"
  if [ "$DRY" = --dry ]; then "$PY" "$R/cpu_ws/mkmask_dry.py" --view "$D/surf/view" --out "$D/surf/masks"; fi
  du -sh "$D/surf" "$D/ref"
  ;;
gs)
  B=$1; DRY=${2:-}; CH=${3:-}
  cd "$C"
  export PYTHONPATH=$C:$C/surface_v2
  "$PY" -m speed_rt.gapsw.draw_gs --beacon "$B" $DRY --out "$D/gs/shots" --manifest "$D/gs/srt-gs.json" \
     --sets srt-gs-d7 srt-gs-d9 srt-gs-d11 srt-gs-d13 srt-gs-d15 srt-gs-d17 ${CH:+--chunks $CH}
  for d in 7 9 11 13 15 17; do
    "$PY" -m speed_rt.gapsw.gs_time mask --d $d --shots "$D/gs/shots" --manifest "$D/gs/srt-gs.json" --workers 8 \
       --out "$D/gs/masks" ${CH:+--chunks $CH}
  done
  du -sh "$D/gs"
  ;;
bb)
  B=$1; DRY=${2:-}; N=${3:-10000}
  export SRT_REL=$C SRT_ROOT=$R/bbrun PYCPU=$PY QND_RELAY=$R/relay DRY_SHOTS=$N
  if [ "$DRY" = --dry ]; then bash "$R/cpu_ws/bbdraw_dry.sh" "$B" "$D/bb"; else bash "$C/speed_rt/bb/draw.sh" "$B" "$D/bb"; fi
  if [ "$DRY" = --dry ]; then "$PY" "$R/cpu_ws/bbpreds_dry.py" --tree "$C" --draw "$D/bb"; fi
  du -sh "$D/bb"
  ;;
shots)
  RUN=$1; FROM=${2:-}; H=$R/handoff
  (cd "$R/cpu_ws" && LD_LIBRARY_PATH=$R/lib "$PY" harness.py import --cfg "${SRT_CFG:-cfg_ws.json}" --units surface_shots,gs \
     --log "$RUN/handoff" ${FROM:+--handoff-from "$FROM"})
  rm -rf "$D/surf/shots" "$D/surf/view" "$D/ref/shots" "$D/gs/shots" "$D/gs/masks"
  mkdir -p "$D/surf" "$D/ref" "$D/gs" "$D/probe" "$D/val"
  U=$H/surface_shots; G=$H/gs
  if [ -d "$U/surf/shots" ]; then                     # the data folder's own layout
    ln -s "$U/surf/shots" "$D/surf/shots"; cp "$U/surf/srt-surf.json" "$D/surf/"
    ln -s "$U/ref/shots" "$D/ref/shots"; cp "$U/ref/srt-ref.json" "$D/ref/"
  else                                                # flat: shots/, refshots/, srt-surf.json, srt-ref.json
    ln -s "$U/shots" "$D/surf/shots"; cp "$U/srt-surf.json" "$D/surf/"
    ln -s "$U/refshots" "$D/ref/shots"; cp "$U/srt-ref.json" "$D/ref/"
  fi
  if [ -d "$G/gs/shots" ]; then ln -s "$G/gs/shots" "$D/gs/shots"; cp "$G/gs/srt-gs.json" "$D/gs/"
  else ln -s "$G/shots" "$D/gs/shots"; cp "$G/srt-gs.json" "$D/gs/"; fi
  for f in "$G/probe/d9_p0.006_X_val_000.npz" "$G/d9_p0.006_X_val_000.npz"; do    # the registered probe chunk, if sent
    if [ -f "$f" ]; then cp "$f" "$D/probe/"; cp "$f" "$D/val/"; break; fi
  done
  sha256sum "$D/probe/d9_p0.006_X_val_000.npz" "$D/val/d9_p0.006_X_val_000.npz" || true
  (cd "$C" && PYTHONPATH=$C/surface_v2:$C "$PY" -m speed_rt.draw view --shots "$D/surf/shots" --out "$D/surf/view")
  for d in 7 9 11 13 15 17; do
    (cd "$C" && PYTHONPATH=$C:$C/surface_v2 "$PY" -m speed_rt.gapsw.gs_time mask --d $d --shots "$D/gs/shots" \
       --manifest "$D/gs/srt-gs.json" --workers 8 --out "$D/gs/masks")
  done
  du -shL "$D/surf/shots" "$D/ref/shots" "$D/gs/shots" "$D/gs/masks"
  ;;
handoff_dry)
  H=$1; S=$R/handoff_dry_src
  case "$H" in *speedrt_in_dry*) ;; *) echo "refusing $H: not a speedrt_in_dry folder"; exit 2 ;; esac
  rm -rf "$S" && mkdir -p "$S/surface/low" "$S/surface/high" "$S/bb/draw"
  for f in "$D"/surf/masks/val_*.npz; do
    d=$(basename "$f" | sed 's/^val_d\([0-9]*\)_.*/\1/'); cp "$f" "$S/surface/$([ "$d" -le 11 ] && echo low || echo high)/"
  done
  (cd "$C/surface_v2" && PYTHONPATH=. "$PY" "$R/cpu_ws/mkres_dry.py" --view "$D/surf/view" --out "$S/ising")
  cp -r "$D/bb/shots" "$D"/bb/manifest_*.csv "$S/bb/draw/" && cp -r "$D/bb/preds" "$S/bb/preds"
  sums() { (cd "$1" && find . -type f ! -name "$2" | sort | xargs -r sha256sum > "$2"); }
  sums "$S/surface" SHA256SUMS; sums "$S/ising" SHA256SUMS; sums "$S/bb/draw" draw_SHA256SUMS; sums "$S/bb" SHA256SUMS
  rm -rf "$H" && mkdir -p "$H" && cp -r "$S"/. "$H"/
  touch "$H/surface/DONE" "$H/ising/DONE" "$H/bb/draw/DRAW_DONE" "$H/bb/DONE"     # the markers last
  du -sh "$H"/*
  ;;
*) echo "unknown step $step"; exit 2 ;;
esac
echo "=== done $step $(date -Is)"
