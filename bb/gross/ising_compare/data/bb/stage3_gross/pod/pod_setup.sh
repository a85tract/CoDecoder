#!/bin/bash
# gross stage 3: pod environment. <pod-home>/g2venv = gross2's lock exactly (the frozen engine's torch 2.11.0+cu126);
# <pod-home>/cqvenv = cudaq-qec 0.8.0 (as on BB72's stage-3 pod) for bb.gpu_bposd.
set -x
export UV_CACHE_DIR=/workspace/uvcache
L=/workspace/env.lock; V=<pod-home>/g2venv; Q=<pod-home>/cqvenv
pkgs() { grep -v '^#' "$1" | sort; }
uv python install 3.12.14 || exit 2
rm -rf $V && uv venv -q --python 3.12.14 $V || exit 2
uv pip install -q -p $V/bin/python --no-deps --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r <(pkgs $L) || exit 3
[ "$(uv pip freeze -p $V/bin/python | sort)" = "$(pkgs $L)" ] && echo LOCK_OK || echo LOCK_DIFF
rm -rf $Q && uv venv -q --python 3.12.14 $Q || exit 4
uv pip install -q -p $Q/bin/python "cudaq-qec==0.8.0" "numpy==2.2.6" "stim==1.16.0" "scipy==1.18.1" || echo CUDAQ_FAIL
uv pip freeze -p $V/bin/python > /workspace/out/freeze_g2venv.txt; uv pip freeze -p $Q/bin/python > /workspace/out/freeze_cqvenv.txt
$V/bin/python - <<'PY'
import sys, torch, triton, numpy, stim, scipy
x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all()
print("ENV g2venv", sys.version.split()[0], "torch", torch.__version__, torch.version.cuda, torch.backends.cudnn.version(),
      "triton", triton.__version__, "numpy", numpy.__version__, "stim", stim.__version__, "scipy", scipy.__version__,
      "gpu", torch.cuda.get_device_name(0), "stim modules", sorted(m for m in sys.modules if m.startswith("stim._")))
PY
$Q/bin/python -c "import sys, importlib.metadata as m, cudaq_qec, numpy, stim; print('ENV cqvenv', sys.version.split()[0], 'cudaq-qec', m.version('cudaq-qec'), 'numpy', numpy.__version__, 'stim', stim.__version__)"
echo SETUP_END $(date +%T)
# Added on the pod after the first run (the cudaq-qec meta package installed no cu12 payload, and the relay testdata
# module that builds the X-sector DEM needs BB72's pod packages); the resulting venv is freeze_cqvenv.txt:
#   uv pip install -p $Q/bin/python "cudaq-qec-cu12==0.8.0" "cuda-quantum-cu12==0.16.0.post1" "numpy==2.2.6" "stim==1.16.0" "scipy==1.18.1"
#   uv pip install -p $Q/bin/python "sinter==1.16.0"
#   uv pip install -p $Q/bin/python "relay-bp==0.2.2" "ldpc==2.4.1" "numpy==2.2.6" "stim==1.16.0" "scipy==1.18.1"
#   uv pip install -p $Q/bin/python "beliefmatching==0.2.0" "pymatching==2.4.0" "matplotlib==3.11.2" "cuquantum-python-cu12==26.6.0" \
#     "nvmath-python==0.9.0" "cupy-cuda12x==13.6.0" "relay-bp==0.2.2" "ldpc==2.4.1" "numpy==2.2.6" "stim==1.16.0" "sinter==1.16.0" "scipy==1.18.1"
# Data on the pod: the checkpoint, cal chunks 0-9 and D-bank chunk 10 (sha256 = the manifest), stage 2's H100 cal reads
# (data/ref, for the engine check), the 14 relay circuit files and the relay testdata module (__init__, circuits, utils;
# sha256 equal to the Mac mini's and Derecho's copies). /workspace/env.sh: PY=<pod-home>/g2venv/bin/python,
# PYQ=<pod-home>/cqvenv/bin/python, PYTHONPATH=repo/ising_compare:repo/qldpc_neural, BB_SHOTS, BB_PREDS, QND_RELAY, BB_PIDNS=1.
