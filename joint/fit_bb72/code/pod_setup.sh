#!/bin/bash
# ARB pod setup: D1's venv (the pins of bb/venv_cu128: release phase_c/scripts/h100_req_cu128.txt; torch 2.11.0+cu128,
# triton 3.6.0) on Python 3.12.14, then the environment record out/env.json. Inputs unpacked under $W before.
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
mkdir -p $W/out/logs
exec > >(tee -a $W/out/logs/setup.log) 2>&1
echo "== setup start $(date -Is)"
command -v uv > /dev/null || pip install -q uv || exit 2
command -v gcc > /dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential; } || exit 2
uv python install 3.12.14 || exit 2
[ -x $PY ] || uv venv -q --python 3.12.14 $W/venv || exit 2
uv pip install -q -p $PY --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r $PC/scripts/h100_req_cu128.txt || exit 3
uv pip freeze -p $PY > $W/out/logs/venv_freeze.txt
diff <(grep -v '^#' $PC/scripts/h100_req_cu128.txt | sort) <(sort $W/out/logs/venv_freeze.txt) && echo PINS_EQUAL || echo PINS_DIFFER
( cd $W/release/weights && sha256sum -c --ignore-missing MANIFEST.sha256 2>&1 | grep cascade_bb72 )
cd $R/ising_compare
$PY - <<'PY'
import json, os, platform, subprocess, sys
import numpy, scipy, stim, torch, triton
from bb import cascade_data as cd
import hashlib
x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all()
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
circ = {p: hashlib.sha256(str(cd.circuit_at("bb72", p, "depspam_fm")).encode()).hexdigest() for p in (0.002, 0.003)}
env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
       "cudnn": torch.backends.cudnn.version(), "triton": triton.__version__, "numpy": numpy.__version__,
       "scipy": scipy.__version__, "stim": stim.__version__, "gpu": torch.cuda.get_device_name(0),
       "capability": list(torch.cuda.get_device_capability(0)), "nvidia_smi": smi, "driver": os.environ["PC_GPU_DRIVER"],
       "cpu": cd._cpu_model(), "nproc": os.cpu_count(), "host": platform.node(), "circuit_sha256": circ,
       "stim_env": cd.env_record("bb72"),
       "mem_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1)}
json.dump(env, open(os.path.join(os.environ["W"], "out", "env.json"), "w"), indent=1)
print("ENV", json.dumps(env))
PY
echo "== setup end $(date -Is)"
