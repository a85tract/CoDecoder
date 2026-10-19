# Sourced by every ARB-d13 step on the pod (W = the pod's work directory).
W=${W:-/workspace/arb2}
export W PY=$W/venv/bin/python CODE=$W/code REL=$W/release \
  ARB2_SEEDS=$W/seeds/arb2_seeds.json ARB2_RELEASE=$W/release CASCADE_ROOT=$W/cascade_root \
  PYTHONPATH=$W/release/cascade/d13:$W/code OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
  TRITON_CACHE_DIR=$W/triton_cache TORCHINDUCTOR_CACHE_DIR=$W/inductor_cache UV_CACHE_DIR=$W/uvcache \
  SV_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
  LOCK=$W/release/cascade/d13/out/locked_d13.json
