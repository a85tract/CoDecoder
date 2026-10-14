#!/bin/bash
# setup_pod.sh venv|unpack (CASCADE-TRT, on the pod, root $Q):
#   venv    the BB harness venv as the speed re-test's pods had it (speed_rt/cascade/h100_pod.sh setup: CPython 3.12.3 by
#           uv, the D1 tree's req_cu128.txt pins: torch 2.11.0+cu128, triton 3.6.0), plus pip TensorRT 10.x (cu12), onnx
#           and onnxscript, installed with the pins as constraints; writes $Q/VENV_OK. Needs $Q/code (code.tgz).
#   unpack  the public tree and weights (bundle.tgz) into $Q/pub, the Relay-BP checkout of the BB72 circuits at the pinned
#           revision into $Q/relay (qnd.circuits; QND_RELAY), CASCADE_ROOT's ckpt link; writes $Q/UNPACK_OK.
set -euo pipefail
Q=${Q:-/workspace/ctrt}; PYVER=3.12.3; MODE=${1:?venv or unpack}
mkdir -p $Q/setup
cd $Q
if [ $MODE = venv ]; then
  { nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.limit --format=csv,noheader
    lscpu | grep -E '^(Model name|CPU\(s\))'; nproc; } | tee $Q/setup/host.txt
  REQ=$Q/code/req_cu128.txt
  command -v uv > /dev/null || python3 -m pip install -q uv
  uv venv -q --seed --python $PYVER $Q/venv
  PIP="$Q/venv/bin/pip --disable-pip-version-check"
  $PIP install -q pip==25.0.1
  $PIP install -q --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r $REQ
  grep -v '^torch==' $REQ > $Q/setup/constraints.txt
  $PIP install -q -c $Q/setup/constraints.txt "tensorrt-cu12>=10,<11" onnx onnxscript
  $Q/venv/bin/pip freeze > $Q/setup/pip_freeze.txt
  $Q/venv/bin/python - $REQ $Q/setup/pip_freeze.txt <<'PY'
import re, sys
n = lambda s: re.sub(r"[-_.]+", "-", s.split("==")[0].strip().lower()) + "==" + s.split("==")[1].strip().split("+")[0]
req, fr = ({n(x) for x in open(f) if "==" in x and not x.lstrip().startswith("#")} for f in sys.argv[1:3])
miss = sorted(req - fr)
import platform, torch, triton, numpy, stim, tensorrt, onnx                          # noqa: E401
print(f"python {platform.python_version()} torch {torch.__version__} triton {triton.__version__} numpy {numpy.__version__} "
      f"stim {stim.__version__} tensorrt {tensorrt.__version__} onnx {onnx.__version__} card {torch.cuda.get_device_name(0)}")
print("pins installed:", "ok" if not miss else "MISSING " + " ".join(miss))
sys.exit(0 if not miss and torch.cuda.is_available() else 1)
PY
  date -Is > $Q/VENV_OK
  echo "venv ok"
elif [ $MODE = unpack ]; then
  mkdir -p $Q/pub $Q/d13root/ckpt/CS13
  tar -xzf bundle.tgz -C $Q/pub
  [ -d $Q/relay ] || git clone -q https://github.com/trmue/relay.git $Q/relay
  git -C $Q/relay checkout -q d185194ba0cb4101ced4340d82b2ee6d42f225f0
  ln -sf $Q/pub/weights/cascade_d13/CS13/final_ema.pt $Q/d13root/ckpt/CS13/final_ema.pt
  date -Is > $Q/UNPACK_OK
  echo "unpack ok"
fi
