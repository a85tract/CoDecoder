#!/bin/bash
# ARB3 pod setup: the D1 tree's pins (phase_c/scripts/h100_req_cu128.txt: torch 2.11.0+cu128, triton 3.6.0) on Python
# 3.12.14 in a uv venv, as ARB's pod_setup.sh; then the environment record out/env.json.
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
mkdir -p $W/out/logs
exec > >(tee -a $W/out/logs/setup.log) 2>&1
echo "== setup start $(date -Is)"
command -v uv > /dev/null || pip install -q uv || exit 2
uv python install 3.12.14 || exit 2
[ -x $PY ] || uv venv -q --python 3.12.14 $W/venv || exit 2
uv pip install -q -p $PY --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r $PC/scripts/h100_req_cu128.txt || exit 3
uv pip freeze -p $PY > $W/out/logs/venv_freeze.txt
diff <(grep -v '^#' $PC/scripts/h100_req_cu128.txt | sort) <(sort $W/out/logs/venv_freeze.txt) && echo PINS_EQUAL || echo PINS_DIFFER
( cd $R/weights && sha256sum -c --ignore-missing MANIFEST.sha256 2>&1 | grep -E "cascade_bb72/(C14p|S55p_deep)|cascade_d13|d13_b5" )
$PY -c "import torch, triton, numpy, stim; print('torch', torch.__version__, 'triton', triton.__version__, 'numpy', numpy.__version__, 'stim', stim.__version__, torch.cuda.get_device_name(0)); x = torch.randn(256, 256, device='cuda'); assert torch.isfinite(x @ x).all()"
echo "== setup end $(date -Is)"
