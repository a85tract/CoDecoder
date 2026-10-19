# Sourced by every ARB3 step on the pod (W = the pod's work directory).
W=${W:-/workspace/arb3}
R=$W/release
BB=$R/cascade/bb72
export W R BB PY=$W/venv/bin/python PC=$BB/cascade_compare/phase_c PD=$BB/cascade_compare/phase_d \
  BO=$BB/cascade_compare/batchopt WTS=$R/weights/cascade_bb72 \
  OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 UV_CACHE_DIR=$W/uvcache \
  TRITON_CACHE_DIR=$W/cache/triton TORCHINDUCTOR_CACHE_DIR=$W/cache/inductor \
  PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
  SV_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
  QND_RELAY=$W/relay_min CUDA_DEVICE_MAX_CONNECTIONS=32 POD_MAP_W4=$W/w4 POD_TREE="arb3 pod (release 94aaaa7 + arb-prod)"
export BBPATH=$BB/ising_compare:$BB/qldpc_neural:$PC:$PD:$BO:$W/pod
LOCK=$PD/d1_locked_depspam_fm.json
