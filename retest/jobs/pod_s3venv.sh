#!/bin/bash
# Fallback for BB72's network reads (RETEST.md section 3.1): stage 3's own platform, an H100 with stage 3's pod
# environment rebuilt from its record (bb/bb72/.../stage3/pod/pod_pip_freeze.txt; torch 2.11.0+cu128, cuDNN 9.19.0.56,
# Triton 3.6.0), so that the read's engine signature differs from the policies' in "gpu" only, as stage 3's did.
set -x
export UV_CACHE_DIR=<pod-home>/uvcache
W=<pod-home>/rt; V=<pod-home>/s3venv
F=$W/code/bb/bb72/ising_compare/data/bb/stage3/pod/pod_pip_freeze.txt
pkgs() { grep -v '^#' "$1" | grep -v "^-e \|@ file" | sort; }
rm -rf $V && uv venv -q --python 3.12.14 $V || exit 2
uv pip install -q -p $V/bin/python --no-deps --index-strategy unsafe-best-match \
  --index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple -r <(pkgs $F) || exit 3
[ "$(uv pip freeze -p $V/bin/python | sort)" = "$(pkgs $F)" ] && echo S3LOCK_OK || echo S3LOCK_DIFF
uv pip freeze -p $V/bin/python > $W/freeze_s3venv.txt
echo "export PY3=$V/bin/python" >> $W/env.sh
$V/bin/python -c "import sys, torch, triton; print('ENV s3venv', sys.version.split()[0], torch.__version__, torch.version.cuda, torch.backends.cudnn.version(), triton.__version__, torch.cuda.get_device_name(0))"
echo S3SETUP_END $(date +%T)
