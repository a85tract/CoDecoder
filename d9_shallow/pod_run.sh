#!/bin/bash
# D9-SHALLOW on its one H100 pod (PLAN.md section 4): bash pod_run.sh STEP... from W (default /workspace/d9s), with the
# tree in $W/rel (the commit's `git archive`: surface_v2, speed_rt, d9_shallow; DEPLOYED = the commit), in this order:
#   setup      the pod recipe (CPython 3.12.3 by uv; torch 2.11.0 from the cu126 index with Triton 3.6.0; numpy 2.2.6,
#              scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, safetensors 0.8.0: d9_shallow/env/pod.txt), checked; the tree's
#              sha256 lists (d9_shallow/SHA256SUMS, d9_shallow/PUBLIC.sha256); B1's weights and config entry overlaid on
#              the run tree (surface_v2/weights/, configs_sv2.json); the compiled router (timing/d7/build.sh, nvcc 12.1,
#              the studies' cubin sha256)
#   prescreen  pod_d9s.py prescreen (exit 4: the pod is released and replaced)
#   draw       after the beacon: draw_d9s.py --beacon $W/rel/d9_shallow/testsets/beacon_d9s.json -> $W/shots
#   masks | timed | handoff | pack     pod_d9s.py's steps on the view $W/view (after commit B is public)
#   reads      reads_d9s.py: B1 and d9_b3_sv2s1 (the deployed engine, batch one), then plain and correlated PM (the pod's CPUs)
set -euo pipefail
W=${W:-/workspace/d9s}; REL=$W/rel; V=$W/venv; PY=$V/bin/python; B=$W/build/d7
export PYTHONPATH=$REL/surface_v2:$REL PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
cd $REL/surface_v2
for step in "$@"; do
  echo "== $step $(date -Is) $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  case $step in
  setup)
    [ -e $W/OVERLAID ] || (cd $REL && sha256sum --quiet -c d9_shallow/SHA256SUMS && sha256sum --quiet -c d9_shallow/PUBLIC.sha256) || { echo "the tree fails its sha256 lists"; exit 1; }
    command -v uv > /dev/null || python3 -m pip install -q uv
    [ -x $PY ] || uv venv -q --seed --python 3.12.3 $V
    $V/bin/pip install -q torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
    $V/bin/pip install -q --extra-index-url https://download.pytorch.org/whl/cu126 -r $REL/d9_shallow/env/pod.txt
    $PY -c "import sys, torch, triton, numpy, scipy, stim, pymatching, safetensors
v = (sys.version.split()[0], torch.__version__, triton.__version__, numpy.__version__, scipy.__version__, stim.__version__, pymatching.__version__, safetensors.__version__)
print('env', v, torch.cuda.get_device_name()); assert v == ('3.12.3', '2.11.0+cu126', '3.6.0', '2.2.6', '1.18.1', '1.16.0', '2.4.0', '0.8.0'), v"
    $V/bin/pip freeze > $W/pip_freeze.txt
    cp $REL/d9_shallow/weights/d9_b1w64_sh.safetensors $REL/surface_v2/weights/
    $PY -c "
import json
f = 'weights/configs_sv2.json'; c = json.load(open(f))
c['networks']['d9_b1w64_sh'] = json.load(open('../d9_shallow/weights/d9_b1w64_sh.config.json'))
json.dump(c, open(f, 'w'), indent=1)
from convlines import decoder, provenance
assert decoder.weights_sha256('d9_b1w64_sh') == json.load(open('../d9_shallow/freeze_d9s.json'))['networks']['d9_b1w64_sh']['sha256']
assert provenance.code_digest() == json.load(open('../d9_shallow/freeze_d9s.json'))['code_digest'], provenance.code_digest()
print('tree ok: B1 overlaid, code digest', provenance.code_digest())
open('$W/OVERLAID', 'w').write('overlaid')"
    [ -e $B/libs7router.so ] || NVCC=${NVCC:-/usr/local/cuda-12.1/bin/nvcc} bash timing/d7/build.sh $B 16 > $W/d7build.log 2>&1 || { tail -5 $W/d7build.log; exit 1; }
    [ "$(sha256sum < $B/kernel_sm90a.cubin | cut -c1-64)" = 8f4180d7ffe607c0f8ac023afdb6ec2fbce9f9a59a5ff873e71908e067ea2c2d ] ||
      { echo "d = 7 cubin: not the studies'"; exit 1; }
    echo "SETUP OK" ;;
  prescreen) $PY ../d9_shallow/pod_d9s.py prescreen --view $W/view --out $W/session --build $B ;;
  draw)
    $PY ../d9_shallow/draw_d9s.py --beacon $REL/d9_shallow/testsets/beacon_d9s.json --out $W/shots
    mkdir -p $W/view
    for j in $(seq -w 0 19); do ln -s $W/shots/d9stiming/d9_p0.006_X_d9stiming_0$j.npz $W/view/d9_p0.006_X_val_0$j.npz; done ;;
  masks|timed|handoff|pack) $PY ../d9_shallow/pod_d9s.py $step --view $W/view --out $W/session --build $B ;;
  reads)
    (cd $REL && cmp d9_shallow/testsets/d9s_test.json $W/shots/manifests/d9s_test.json) || { echo "the test manifest is not commit B's"; exit 1; }
    for n in d9_b1w64_sh d9_b3_sv2s1; do
      $PY ../d9_shallow/reads_d9s.py net --net $n --shots $W/shots --manifest $REL/d9_shallow/testsets/d9s_test.json --out $W/reads --build $B
    done
    $PY ../d9_shallow/reads_d9s.py base --shots $W/shots --manifest $REL/d9_shallow/testsets/d9s_test.json --out $W/reads --procs 16 ;;
  *) echo "unknown step $step"; exit 1 ;;
  esac
done
