#!/bin/bash
# ARB-d13 pod setup: the comparison's venv (release cascade/d13/surface/pod/requirements_cu128.txt, the pins of bb's
# venv_cu128: torch 2.11.0+cu128, triton 3.6.0, stim 1.16.0) on Python 3.12.14, the checkpoint where sv_common looks for
# it (CASCADE_ROOT/ckpt/CS13/final_ema.pt), then the environment record out/env.json. Inputs unpacked under $W before.
set -uo pipefail
source "$(dirname "$0")/pod_env.sh"
mkdir -p $W/out/logs $W/triton_cache $W/inductor_cache $CASCADE_ROOT/ckpt/CS13
exec > >(tee -a $W/out/logs/setup.log) 2>&1
echo "== setup start $(date -Is)"
command -v uv > /dev/null || pip install -q uv || exit 2
command -v zstd > /dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q zstd; } || exit 2
command -v gcc > /dev/null || { apt-get update -q && DEBIAN_FRONTEND=noninteractive apt-get install -y -q build-essential; } || exit 2
uv python install 3.12.14 || exit 2
[ -x $PY ] || uv venv -q --python 3.12.14 $W/venv || exit 2
REQ=$REL/cascade/d13/surface/pod/requirements_cu128.txt
uv pip install -q -p $PY --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r $REQ || exit 3
uv pip freeze -p $PY > $W/out/logs/venv_freeze.txt
diff <(grep -v '^#' $REQ | grep -iv '^pip==' | tr 'A-Z_' 'a-z-' | sort) <(tr 'A-Z_' 'a-z-' < $W/out/logs/venv_freeze.txt | sort) \
  && echo PINS_EQUAL || echo PINS_DIFFER
ln -sfn $REL/weights/cascade_d13/CS13/final_ema.pt $CASCADE_ROOT/ckpt/CS13/final_ema.pt
( cd $REL/weights && sha256sum -c --ignore-missing MANIFEST.sha256 2>&1 | grep -E 'cascade_d13|d13_b5_r775|grid_d13' )
cd $W
$PY - <<'PY'
import hashlib, json, os, platform, subprocess, sys
import numpy, scipy, stim, torch, triton
from surface import surface_data as sdata, surface_layout as sl, sv_common as sc
x = torch.randn(256, 256, device="cuda"); assert torch.isfinite(x @ x).all()
smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
circ = {str(p): hashlib.sha256(str(sl.circuit_at(13, p)).encode()).hexdigest() for p in (0.006, 0.0035)}
envr = sdata.env_record(13)
env = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.version.cuda,
       "cudnn": torch.backends.cudnn.version(), "triton": triton.__version__, "numpy": numpy.__version__,
       "scipy": scipy.__version__, "stim": stim.__version__, "gpu": torch.cuda.get_device_name(0),
       "gpu_is_the_comparisons": torch.cuda.get_device_name(0) == sc.GPU,
       "capability": list(torch.cuda.get_device_capability(0)), "nvidia_smi": smi, "driver": os.environ["SV_GPU_DRIVER"],
       "cpu": sc.cpu_model(), "nproc": os.cpu_count(), "host": platform.node(),
       "circuit_pinned_sha256": sl.check_circuit_at(13), "circuit_at_sha256": circ,
       "stim_env": envr, "stim_fingerprint_equals_releases": envr["stim_fingerprint"] == "21335d99f1b86f0b",
       "mem_gb": round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2 ** 30, 1),
       "identity": sc.env()}
json.dump(env, open(os.path.join(os.environ["W"], "out", "env.json"), "w"), indent=1)
print("ENV", json.dumps(env))
PY
echo "== setup end $(date -Is)"
