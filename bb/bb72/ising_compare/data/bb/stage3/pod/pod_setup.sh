#!/bin/bash
# BB72 stage 3: pod environment (Python 3.12.14 venv; torch 2.11.0+cu128 as in the policies' engine signature)
set -x
cd /workspace
export UV_CACHE_DIR=/workspace/uvcache
uv python install 3.12.14 && uv venv --clear -p 3.12.14 /workspace/venv || exit 2
uv pip install --python /workspace/venv/bin/python --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "torch==2.11.0+cu128" || exit 3
uv pip install --python /workspace/venv/bin/python "numpy==2.2.6" "stim==1.16.0" "sinter==1.16.0" scipy || exit 4
uv pip install --python /workspace/venv/bin/python "cudaq-qec==0.8.0" || echo CUDAQ_FAIL
/workspace/venv/bin/python - <<'PY'
import sys, torch, triton, numpy, stim, scipy
print("PY", sys.version.split()[0], "torch", torch.__version__, torch.version.cuda, torch.backends.cudnn.version(),
      "triton", triton.__version__, "numpy", numpy.__version__, "stim", stim.__version__, "scipy", scipy.__version__)
print("gpu", torch.cuda.get_device_name(0))
print("stim modules", sorted(m for m in sys.modules if m.startswith("stim._")))
try:
    import cudaq_qec, importlib.metadata as m; print("cudaq_qec", m.version("cudaq-qec"))
except Exception as e: print("cudaq_qec ERR", e)
PY
echo SETUP_END
