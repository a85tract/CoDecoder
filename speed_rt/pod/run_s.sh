#!/bin/bash
# speed_rt/pod/run_s.sh STEP...: one pod session of the surface stage on its H100 pod (the stage's section), in order. The pods time GPU
# work only (CURRENT_RULES): no CPU part runs here. ROLE=low (pod S-low, d = 7, 9, 11) or ROLE=high (pod S-high, d = 13, 15, 17) is required:
#   setup      the environment (the frozen pod recipe), the d = 7 cubin (nvcc 12.1.105, sha256 checked), the studies' validation shots
#              (their digests in surface_v2/routing/manifest.json), and trtexec 10.14.1.48
#   prescreen  speed_rt.gpu_prescreen (stage 1's rule: five GPU checks, CPU steal at most 1 % over 5 s, quota at least 20 CPUs) -> $W/prescreen
#              (prescreen.json); the pod is replaced when it fails (exit 4; writes $W/CORE, the timed core, when it passes)
#   build      Ising's ONNX (ising_trt prep on the validation chunk 0: the FP8 calibration rows) and plans of the role's d: the eight registered
#              per d (ising_trt build) and D5's four official ones (speed_rt.ising_d5 official), then every plan's check on the validation
#              chunk 0 (ising_trt check, ising_d5 check); no timing shot; then the builds' record $W/work/BUILD_INPUTS.json
#              (speed_rt/build_inputs.py), which the session copies into its records
#   shots      the role's timing chunks (copied from the draw to $W/shots/srt) checked against the draw's manifest (srt-surf.json in
#              $TESTSETS, default speed_rt/testsets), then the view $W/view of the role's distances (speed_rt/draw.py view --ds)
#   prep | packm | timed | handoff | residuals | pack   speed_rt.pod_s's steps -> $W/masks, $W/handoff/masks, $W/session (the GPU block),
#              $W/res, $W/handoff/res and DONE.json (see pod_s.py; the hand-off for the CPU worker is $W/handoff)
# Layout: $W/rel the deployed public tree (DEPLOYED = its commit); $W/venv the main venv (torch 2.11.0+cu126, TensorRT 10.14.1.48, ModelOpt
# 0.47.0) and PYCQ the CUDA-Q venv (cudaq-qec 0.8.0); $W/in/ising_nv NVIDIA's Ising-Decoding at 3ce4379 with its checkpoints; $W/shots/{val,srt};
# $W/build/d7. DRY=1: dry runs (nothing replaced, --limit LIMIT, GPU rounds DRY_ROUNDS, BP-OSD shots BPOSD_SHOTS, plans of the distances DRY_DS
# only, DRY_INJECT=ITEM:KEY[:proc|all],...: forced failures of round 1's slots, see pod_s.py).
set -euo pipefail
: "${ROLE:?ROLE=low or ROLE=high}"
case $ROLE in low) DS="7 9 11" ;; high) DS="13 15 17" ;; *) echo "ROLE is low or high"; exit 1 ;; esac
DSC=${DS// /,}
W=${W:-$PWD}; REL=$W/rel; PY=${PY:-$W/venv/bin/python}; B=$W/build/d7; TS=${TESTSETS:-$REL/speed_rt/testsets}
export PYCQ=${PYCQ:-$W/venv_cudaq/bin/python}
export PYTHONPATH=$REL/surface_v2:$REL ISING_DECODING_DIR=$W/in/ising_nv PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
DRYARGS=""; [ "${DRY:-0}" = 1 ] && DRYARGS="--dry --limit ${LIMIT:-20000} --bposd-shots ${BPOSD_SHOTS:-20} --rounds ${DRY_ROUNDS:-5}"; [ -n "${DRY_INJECT:-}" ] && DRYARGS="$DRYARGS --dry-inject $DRY_INJECT"
cd $REL/surface_v2
echo "role $ROLE (d $DS) pod $(hostname) commit $(cat $REL/DEPLOYED) gpu $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
S="--role $ROLE --view $W/view --val $W/shots/val --masks $W/masks --onnx $W/onnx --work $W/work --inputs $W/inputs --res $W/res --out $W/session --build $B --prescreen $W/prescreen/prescreen.json --handoff $W/handoff"
for step in "$@"; do
  echo "== $step $(date -Is)"
  case $step in
  setup)
    $PY -c "import torch, stim, pymatching, numpy, tensorrt, modelopt; assert (torch.__version__.split('+')[0], stim.__version__, pymatching.__version__, numpy.__version__, tensorrt.__version__.split('.post')[0], modelopt.__version__) == ('2.11.0', '1.16.0', '2.4.0', '2.2.6', '10.14.1.48', '0.47.0'), 'not the frozen recipe'"
    $PYCQ -c "import cudaq_qec, stim, numpy; print('cudaq venv ok', stim.__version__, numpy.__version__)"
    [ -e $B/kernel_sm90a.cubin ] || NVCC=${NVCC:-/usr/local/cuda-12.1/bin/nvcc} bash timing/d7/build.sh $B > $W/d7build.log 2>&1
    [ "$(sha256sum < $B/kernel_sm90a.cubin | cut -c1-64)" = 8f4180d7ffe607c0f8ac023afdb6ec2fbce9f9a59a5ff873e71908e067ea2c2d ] ||
      { echo "d = 7 cubin: not the studies'"; exit 1; }
    $PY -c "
import os, numpy as np
from convlines import decoder, testset
from timing import deployed
for d in (7, 9, 11, 13, 15, 17):
    sh = [testset.load(os.path.join('$W/shots/val', testset.name('val', d, j)), d) for j in range(20)]
    assert deployed.shots_sha256(np.concatenate([s[0] for s in sh]), np.concatenate([s[1] for s in sh])) == decoder.routing_manifest()['val'][str(d)]['shots_sha256'], d
print('validation shots: the studies at every d')"
    # D5: NVIDIA's trtexec of TensorRT 10.14.1.48 (the plans' version), from NVIDIA's CUDA apt repository, every
    # TensorRT package pinned to it (the repository's newer releases are refused)
    V=10.14.1.48-1+cuda12.9
    [ -x /usr/src/tensorrt/bin/trtexec ] || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends \
      $(for p in libnvinfer-bin libnvinfer10 libnvinfer-plugin10 libnvonnxparsers10 libnvinfer-lean10 libnvinfer-vc-plugin10 libnvinfer-dispatch10; do echo "$p=$V"; done) > $W/trtexec_apt.log 2>&1
    /usr/src/tensorrt/bin/trtexec --help > $W/trtexec_help.txt 2>&1 || true            # (a pipe into head would end it by SIGPIPE: pipefail)
    head -1 $W/trtexec_help.txt | grep -q "TensorRT v101401" || { echo "trtexec is not TensorRT 10.14.1"; exit 1; }
    ;;
  prescreen)
    [ "${DRY:-0}" = 1 ] && rm -rf $W/prescreen                       # a dry run may repeat it; a registered one refuses a second record
    $PY -m speed_rt.gpu_prescreen --out $W/prescreen || { [ "${DRY:-0}" = 1 ] || exit 4; }     # a dry run goes on with a failed pre-screen
    $PY -c "import json; print(json.load(open('$W/prescreen/prescreen.json'))['core'])" > $W/CORE ;;
  build)
    D="$DS"; [ "${DRY:-0}" = 1 ] && D=${DRY_DS:-$DS}                 # dry runs may build fewer distances
    $PY -m timing.ising_trt prep --d $D --val $W/shots/val --onnx $W/onnx
    $PY -m timing.ising_trt build --d $D --onnx $W/onnx --work $W/work
    $PY -m speed_rt.ising_d5 official --d $D --onnx $W/onnx --work $W/work
    # the plans' validation, on the validation chunk 0 (no timing shot): a plan that fails is never timed
    $PY -m timing.ising_trt check --d $D --val $W/shots/val --onnx $W/onnx --work $W/work || echo "some registered plans FAIL"
    $PY -m speed_rt.ising_d5 check --d $D --view $W/shots/val --onnx $W/onnx --work $W/work || echo "some official plans FAIL"
    # the builds' record: every tree file they read, NVIDIA's code, the validation chunks 0, trtexec, every file built (the analysis
    # checks the tree files against F_2's; the session copies it into its records)
    $PY -m speed_rt.build_inputs --rel $REL --onnx $W/onnx --work $W/work --val $W/shots/val --nv $W/in/ising_nv \
      --trtexec-help $W/trtexec_help.txt --out $W/work/BUILD_INPUTS.json ;;
  shots)
    $PY -m speed_rt.draw check --dir $W/shots/srt --manifest $TS/srt-surf.json --ds $DSC
    [ -e $W/view ] || $PY -m speed_rt.draw view --shots $W/shots/srt --out $W/view --ds $DSC ;;
  prep|packm|timed|handoff|residuals|pack)
    $PY -m speed_rt.pod_s $step $S --core $(cat $W/CORE) $DRYARGS ;;
  *) echo "unknown step $step"; exit 1 ;;
  esac
done
date -Is; echo JOBDONE
