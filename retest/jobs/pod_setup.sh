#!/bin/bash
# The rented H100 of the re-test (RETEST.md section 3.1): stage 4's two environments from its locks, as its pod set them
# up (gross_stage4/.../stage4_gross/pod/s4_setup.sh): g2venv (env.lock; torch 2.11.0+cu126, the gross network's fusedt
# engine) and cqvenv (cqvenv.lock; cudaq-qec 0.8.0 and cuda-quantum 0.16.0.post1, the versions of stage 3's and 4's
# pods). Inputs, unpacked before: <pod-home>/rt/code (the freeze tree: bb72/, gross_stage4/, retest/, a git archive) and
# <pod-home>/rt/relay/tests/testdata (the relay clone's circuits at d185194). Run as root on the pod: bash pod_setup.sh
set -x
export UV_CACHE_DIR=<pod-home>/uvcache
W=<pod-home>/rt; V=<pod-home>/g2venv; Q=<pod-home>/cqvenv
L=$W/code/bb/gross_stage4/ising_compare/data/bb/stage4_gross/pod
pkgs() { grep -v '^#' "$1" | sort; }
[ -d $W/code/bb/gross_stage4 ] && [ -d $W/code/bb/bb72 ] && [ -d $W/code/retest ] && [ -d $W/relay/tests/testdata ] || exit 1
command -v uv > /dev/null || pip install -q uv
command -v gcc > /dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential; }
uv python install 3.12.14 || exit 2
rm -rf $V && uv venv -q --python 3.12.14 $V || exit 2
uv pip install -q -p $V/bin/python --no-deps --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r <(pkgs $L/env.lock) || exit 3
[ "$(uv pip freeze -p $V/bin/python | sort)" = "$(pkgs $L/env.lock)" ] && echo LOCK_OK || echo LOCK_DIFF
rm -rf $Q && uv venv -q --python 3.12.14 $Q || exit 4
uv pip install -q -p $Q/bin/python --no-deps -r <(pkgs $L/cqvenv.lock) || echo CUDAQ_FAIL
[ "$(uv pip freeze -p $Q/bin/python | sort)" = "$(pkgs $L/cqvenv.lock)" ] && echo CQLOCK_OK || echo CQLOCK_DIFF
uv pip freeze -p $V/bin/python > $W/freeze_g2venv.txt; uv pip freeze -p $Q/bin/python > $W/freeze_cqvenv.txt
cat > $W/env.sh <<EOS
export W=$W PY=$V/bin/python PYQ=$Q/bin/python QND_RELAY=$W/relay BB_PIDNS=1 PYTHONUNBUFFERED=1 \
  S=$W/code/retest/shim/rt_bb.py X=$W/code/retest/replay/bb_replay.py
EOS
source $W/env.sh
sha256sum $W/code/weights/bb/g2_D12k113c10_final_ema.pt
$PY - <<'PY'
import sys, torch, triton, numpy, stim
x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all()
print("ENV g2venv", sys.version.split()[0], "torch", torch.__version__, torch.version.cuda, torch.backends.cudnn.version(),
      "triton", triton.__version__, "numpy", numpy.__version__, "stim", stim.__version__, "gpu", torch.cuda.get_device_name(0))
PY
$PYQ -c "import sys, importlib.metadata as m, cudaq_qec, numpy, stim; print('ENV cqvenv', sys.version.split()[0], 'cudaq-qec', m.version('cudaq-qec-cu12'), 'numpy', numpy.__version__, 'stim', stim.__version__)"
echo SETUP_END $(date +%T)
