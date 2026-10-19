#!/bin/bash
# SPEED-RT stage 3 (STAGE3_batched.md): one GPU pod's set-up and pre-screen.
#   pod3.sh setup ROLE    one environment for every role and card (ROLE names the log only): stage 2's BB72 pod lock
#                         (bb/bb72/ising_compare/data/bb/stage3/pod/pod_pip_freeze.txt: torch 2.11.0+cu128, Triton 3.6.0,
#                         numpy 2.2.6, Stim 1.16.0) plus safetensors 0.8.0 and PyMatching 2.4.0; its freeze is saved
#   pod3.sh overlay       D9-SHALLOW's network on the surface tree (as d9_shallow/pod_run.sh)
#   pod3.sh prescreen     python -m speed_rt.gpu_prescreen (stage 1's rule, unchanged) -> $W/prescreen/prescreen.json
# Layout: $W/rel the release tree at the freeze (its payload), $W/venv, $W/out.
set -uo pipefail
W=${S3_W:-/workspace/s3}; REL=$W/rel; PY=$W/venv/bin/python
case "${1:-}" in
setup)
  ROLE=$2
  LOCK=$REL/bb/bb72/ising_compare/data/bb/stage3/pod/pod_pip_freeze.txt; IDX=cu128
  command -v uv > /dev/null || pip install -q uv || exit 2
  uv python install 3.12.14 > /dev/null 2>&1 || exit 2
  rm -rf "$W/venv" && uv venv -q --python 3.12.14 "$W/venv" || exit 2
  grep -v '^#' "$LOCK" | grep -v '^\s*$' | grep -v '^-e ' > "$W/lock.txt"
  uv pip install -q -p "$PY" --no-deps --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/$IDX \
    --extra-index-url https://pypi.org/simple -r "$W/lock.txt" || exit 3
  for p in safetensors==0.8.0 pymatching==2.4.0; do "$PY" -c "import ${p%%=*}" 2> /dev/null || uv pip install -q -p "$PY" --no-deps $p; done
  mkdir -p "$W/out" && uv pip freeze -p "$PY" > "$W/out/freeze_$ROLE.txt"
  "$PY" -c "import torch, triton, numpy, stim; print('venv ok', torch.__version__, triton.__version__, numpy.__version__, stim.__version__, torch.cuda.get_device_name(0))"
  ;;
overlay)   # D9-SHALLOW's network on the surface tree, as d9_shallow/pod_run.sh overlays it
  cp "$REL/d9_shallow/weights/d9_b1w64_sh.safetensors" "$REL/surface_v2/weights/"
  cd "$REL/surface_v2" && "$PY" -c "
import json
f = 'weights/configs_sv2.json'; c = json.load(open(f))
c['networks']['d9_b1w64_sh'] = json.load(open('../d9_shallow/weights/d9_b1w64_sh.config.json'))
json.dump(c, open(f, 'w'), indent=1)
from convlines import decoder
assert decoder.weights_sha256('d9_b1w64_sh') == json.load(open('../d9_shallow/freeze_d9s.json'))['networks']['d9_b1w64_sh']['sha256']
print('overlay ok')"
  ;;
prescreen)
  cd "$REL" && PYTHONPATH=$REL/surface_v2:$REL "$PY" -m speed_rt.s3.prescreen3 --out "$W/out/prescreen"; echo "prescreen exit $?"
  ;;
*) echo "usage: $0 setup ROLE | prescreen"; exit 2 ;;
esac
