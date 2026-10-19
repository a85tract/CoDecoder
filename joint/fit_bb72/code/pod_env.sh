# Sourced by every ARB step on the pod (W = the pod's work directory).
W=${W:-/workspace/arb}
R=$W/release/cascade/bb72
export W R PY=$W/venv/bin/python CODE=$W/code PC=$R/cascade_compare/phase_c PD=$R/cascade_compare/phase_d \
  QND_RELAY=$W/relay_min ARB_SEEDS=$W/seeds/arb_seeds.json ARB_RELEASE=$W/release \
  PYTHONPATH=$R/ising_compare:$R/qldpc_neural:$R/cascade_compare/phase_c:$R/cascade_compare/phase_d:$W/code \
  OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 TRITON_CACHE_DIR=$W/triton_cache UV_CACHE_DIR=$W/uvcache \
  PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
  WTS=$W/release/weights/cascade_bb72 LOCK=$R/cascade_compare/phase_d/d1_locked_depspam_fm.json C0=6433102
