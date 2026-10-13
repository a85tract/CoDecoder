#!/bin/bash
# The first H100 pod: the venv of the secondary H100 timing, with the pins of Alpine's bb/venv_cu128 (req_cu128.txt) plus
# relay-bp 0.2.2 (Alpine's bb/venv, which the GPU jobs put on sys.path). Runs detached on the pod.
# req_cu128.txt is scripts/h100_req_cu128.txt, copied to $Q on the pod; the run used this script as is. The second pod's run ran it
# with Q=/opt/h100b (batchopt/req_cu128.txt, the same pins).
Q=${Q:-/opt/h100t}; exec >> $Q/setup_venv.log 2>&1
set -euo pipefail
trap 'echo "== venv FAILED (line $LINENO) $(date -Is)"' ERR
echo "== venv start $(date -Is)"
[ -x $Q/venv/bin/python ] || python3.12 -m venv $Q/venv
PIP="$Q/venv/bin/pip --disable-pip-version-check"
$PIP install -q pip==25.0.1
$PIP install --progress-bar off --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
  -r $Q/req_cu128.txt
$PIP install --progress-bar off --no-deps relay-bp==0.2.2
$Q/venv/bin/pip freeze > $Q/venv_freeze.txt
$Q/venv/bin/python - <<'PY'
import sys, torch, triton, stim, numpy, scipy, relay_bp, importlib.metadata as md
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda, "triton", triton.__version__,
      "stim", stim.__version__, "numpy", numpy.__version__, "scipy", scipy.__version__, "relay-bp", md.version("relay-bp"))
print("cuda available", torch.cuda.is_available(), "device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
x = torch.randn(256, 256, device="cuda"); print("matmul ok", float((x @ x).sum()) == float((x @ x).sum()))
PY
echo "== venv done $(date -Is)"
