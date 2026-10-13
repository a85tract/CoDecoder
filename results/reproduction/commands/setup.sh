#!/bin/bash
set -x
export DEBIAN_FRONTEND=noninteractive
echo "== cuda dirs"; ls -d /usr/local/cuda* 2>/dev/null; ls /usr/local/cuda/bin 2>/dev/null | head; cat /etc/apt/sources.list.d/* 2>/dev/null | head
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq rsync git-lfs numactl >/dev/null 2>&1 && echo "apt ok"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1; export PATH=$HOME/.local/bin:$PATH
uv python install 3.12.14
cd /workspace/repo && uv venv --python 3.12.14 .venv && . .venv/bin/activate
uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126 && uv pip install -r requirements.txt
python -c "import torch,triton,stim,pymatching,numpy,scipy; print(torch.__version__, triton.__version__, stim.__version__, numpy.__version__, scipy.__version__, torch.cuda.is_available())"
cd /workspace && git clone -q https://github.com/NVIDIA/Ising-Decoding && cd Ising-Decoding && git checkout -q 3ce4379fa6cf475789007bde4b6a8206e8622992 && git lfs install --skip-repo && git lfs pull && sha256sum models/*.pt
cd /workspace/repo && . .venv/bin/activate && uv pip install hydra-core==1.3.7 omegaconf==2.3.1 ldpc==2.4.1 beliefmatching==0.2.0
echo "== SETUP DONE"
