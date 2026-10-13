cd /workspace/repo && . .venv/bin/activate
mkdir -p out/host
nvidia-smi > out/host/nvidia-smi.txt; nvidia-smi -q > out/host/nvidia-smi-q.txt
lscpu > out/host/lscpu.txt; cat /proc/meminfo | head -3 > out/host/meminfo.txt; cat /sys/fs/cgroup/cpu.max > out/host/cgroup_cpu.max
cat /etc/os-release > out/host/os-release.txt; uname -a > out/host/uname.txt
python --version > out/host/python.txt; pip freeze > out/host/pip-freeze.txt 2>/dev/null || python -m pip freeze > out/host/pip-freeze.txt
/usr/local/cuda-12.1/bin/nvcc --version > out/host/nvcc.txt; g++ --version | head -1 >> out/host/nvcc.txt
echo "runpod pod <pod-1>, H100 SXM secure cloud, image runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04, started $(date -u -d @$(( $(date +%s) - $(cut -d. -f1 /proc/uptime) )) 2>/dev/null) " > out/host/pod.txt
(cd /workspace/Ising-Decoding && git rev-parse HEAD) > out/host/ising_commit.txt
