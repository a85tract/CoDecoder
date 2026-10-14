#!/bin/bash
# ctrt_session.sh STEP... (CASCADE-TRT-REG, on the card's host): the timing rounds and the decision check, each a step:
#   r1 | r2 | r4   ctrt_drive.py ROUND into $OUT/R1 | R2 | R4 (stage 1's arguments: r2 the phase C lock and speed
#                  components, r4 the study's lock on the RTX PRO 6000 and, on the H100, stage 1's session record and the
#                  lock derived from it, both published under speed_rt/cascade/records/H100/R4/)
#   check          the decision check (ctrt_check.py: TensorRT, then the compiled path with one BB72 graph per process,
#                  then the comparisons) into $OUT/check
#   build          the build records ($PL/build.jsonl and the sha256 of every ONNX file and plan) into $OUT/build
#   prescreen      the H100 pod's pre-screen, stage 1's (h100_pod.sh prescreen), on the idle pod before anything is timed
# The caller sets PY (the venv's Python), P (the public tree at B), PL (this card's plans), OUT (this card's records:
# RECORDS/RTX or RECORDS/H100), CARD (rtx or h100), SRT_SEEDS and SRT_BEACON (B's seeds and beacon record), the harnesses'
# environment of code/env.sh (PYTHONPATH, caches, QND_RELAY, CASCADE_ROOT, CONVLINES_RELEASE), and DRIVE_ARGS and
# CHECK_ARGS for dry runs (with SRT_DRY=1 and a dry beacon record: speed_rt/beacon.py dry).
set -uo pipefail
: "${PY:?} ${P:?} ${PL:?} ${OUT:?} ${CARD:?} ${SRT_SEEDS:?} ${SRT_BEACON:?}"
R=$P/cascade_trt_reg/run; BBT=$P/cascade/bb72; T13=$P/cascade/d13; PC=$BBT/cascade_compare/phase_c
CK=$P/weights/cascade_bb72/C14p.pt
DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
export CTRT_PLANS=$PL PC_GPU_DRIVER="$DRV" SV_GPU_DRIVER="$DRV" SRT_BURN_S=0
if [ "$CARD" = h100 ]; then
  L13=$P/speed_rt/cascade/records/H100/R4/derived_lock.json
  export SRT_SESSION=$P/speed_rt/cascade/records/H100/R4/session.json
else
  L13=$T13/out/locked_d13.json
fi
mkdir -p $OUT
run() { echo "$(date +%T) $*"; "$@"; local rc=$?; [ $rc = 0 ] || echo "$(date +%T) FAILED($rc): $*"; return $rc; }
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.limit --format=csv,noheader
for STEP in "$@"; do
  echo "== $STEP $(date -Is)"
  case $STEP in
    r1) cd $BBT/ising_compare && run $PY $R/ctrt_drive.py r1 --tree $BBT --out $OUT/R1 --card $CARD ${DRIVE_ARGS:-} ;;
    r2) cd $BBT/ising_compare && run $PY $R/ctrt_drive.py r2 --tree $BBT --out $OUT/R2 --card $CARD \
          --w4-lock $PC/out/locked_depspam_fm.json --speed $PC/out/speed_components.json ${DRIVE_ARGS:-} ;;
    r4) cd $T13 && run $PY $R/ctrt_drive.py r4 --tree $T13 --out $OUT/R4 --card $CARD --lock $L13 ${DRIVE_ARGS:-} ;;
    check)
      D=$OUT/check; mkdir -p $D; cd $BBT/ising_compare
      ck() { local tag=$1; shift; SRT_TRACE=$D/trace_$tag.jsonl run $PY $R/ctrt_check.py "$@" ${CHECK_ARGS:-} > $D/log_$tag.txt 2>&1; }
      ck bb_trt --net bb --engine trt --plans $PL --ckpt $CK --save $D/check_bb_trt.npz
      ck bb_compiled_b1 --net bb --engine compiled --only b1 --plans $PL --ckpt $CK --save $D/check_bb_compiled_b1.npz
      ck bb_compiled_bB --net bb --engine compiled --only bB --plans $PL --ckpt $CK --save $D/check_bb_compiled_bB.npz
      ck d13_trt --net d13 --engine trt --plans $PL --save $D/check_d13_trt.npz
      ck d13_compiled --net d13 --engine compiled --plans $PL --save $D/check_d13_compiled.npz
      ck bb_compare --net bb --compare $D/check_bb_compiled_b1.npz $D/check_bb_trt.npz \
        --merge $D/check_bb_compiled_bB.npz --out $D/check_bb.json
      ck d13_compare --net d13 --compare $D/check_d13_compiled.npz $D/check_d13_trt.npz --out $D/check_d13.json ;;
    build)
      mkdir -p $OUT/build && cp $PL/build.jsonl $OUT/build/ && (cd $PL && sha256sum *.onnx *.plan > $OUT/build/plans.sha256) ;;
    prescreen)   # the H100 pod's pre-screen, stage 1's own (speed_rt/cascade/h100_pod.sh prescreen) -> $OUT/prescreen_<stamp>
      Q=$OUT SRT_ROOT=$P PY=$PY run bash $P/speed_rt/cascade/h100_pod.sh prescreen ;;
    *) echo "unknown step $STEP"; exit 2 ;;
  esac
  echo "== $STEP end $(date -Is)"
done
