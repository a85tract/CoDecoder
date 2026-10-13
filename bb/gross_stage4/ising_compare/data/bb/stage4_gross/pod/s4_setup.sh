#!/bin/bash
# gross stage 4 pod (STAGE4_GROSS_PLAN.md 4.1): <pod-home>/g2venv = gross2's lock exactly (Python 3.12.14, torch 2.11.0+cu126,
# Triton 3.6.0: fusedt's environment in GROSS_FAST_ENGINE.md and stage 3's); <pod-home>/cqvenv = stage 3's cudaq-qec 0.8.0 venv
# exactly (its freeze, cqvenv.lock). /workspace/g7_code.tgz: git archive of the code commit; /workspace/g7_data.tgz: the
# checkpoint, CAL chunks 0-9 and D-bank chunk 10, graph32's stage-2 CAL read, sector Relay-BP's CAL read, the relay circuits
# and testdata module. Work tree <pod-home>/g7 (not /workspace: its MooseFS refuses tar's chown).
set -x
export UV_CACHE_DIR=<pod-home>/uvcache
W=<pod-home>/g7; V=<pod-home>/g2venv; Q=<pod-home>/cqvenv
pkgs() { grep -v '^#' "$1" | sort; }
mkdir -p $W/out $W/preds && tar xzf /workspace/g7_code.tgz --no-same-owner -C $W && tar xzf /workspace/g7_data.tgz --no-same-owner -C $W || exit 1
command -v uv > /dev/null || pip install -q uv
command -v gcc > /dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential; }
uv python install 3.12.14 || exit 2
rm -rf $V && uv venv -q --python 3.12.14 $V || exit 2
uv pip install -q -p $V/bin/python --no-deps --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu126 --extra-index-url https://pypi.org/simple -r <(pkgs /workspace/env.lock) || exit 3
[ "$(uv pip freeze -p $V/bin/python | sort)" = "$(pkgs /workspace/env.lock)" ] && echo LOCK_OK || echo LOCK_DIFF
rm -rf $Q && uv venv -q --python 3.12.14 $Q || exit 4
uv pip install -q -p $Q/bin/python --no-deps -r <(pkgs /workspace/cqvenv.lock) || echo CUDAQ_FAIL
[ "$(uv pip freeze -p $Q/bin/python | sort)" = "$(pkgs /workspace/cqvenv.lock)" ] && echo CQLOCK_OK || echo CQLOCK_DIFF
uv pip freeze -p $V/bin/python > $W/out/freeze_g2venv.txt; uv pip freeze -p $Q/bin/python > $W/out/freeze_cqvenv.txt
cat > $W/env.sh <<EOS
export W=$W PY=$V/bin/python PYQ=$Q/bin/python PYTHONPATH=$W/code/ising_compare:$W/code/qldpc_neural BB_SHOTS=$W/shots \
  BB_PREDS=$W/preds QND_RELAY=$W/relay BB_PIDNS=1 PYTHONUNBUFFERED=1
EOS
source $W/env.sh
sha256sum $W/ckpt/final_ema.pt
$PY - <<'PY'
import sys, torch, triton, numpy, stim
x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all()
print("ENV g2venv", sys.version.split()[0], "torch", torch.__version__, torch.version.cuda, torch.backends.cudnn.version(),
      "triton", triton.__version__, "numpy", numpy.__version__, "stim", stim.__version__, "gpu", torch.cuda.get_device_name(0))
PY
$PYQ -c "import sys, importlib.metadata as m, cudaq_qec, numpy, stim; print('ENV cqvenv', sys.version.split()[0], 'cudaq-qec', m.version('cudaq-qec-cu12'), 'numpy', numpy.__version__, 'stim', stim.__version__)"
echo SETUP_END $(date +%T)
