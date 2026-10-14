# env.sh (CASCADE-TRT, on the pod): source for the harnesses' environment (the same as run_steps.sh sets).
Q=${Q:-/workspace/ctrt}; P=$Q/pub; BBT=$P/cascade/bb72; T13=$P/cascade/d13; C=$Q/code; PL=$Q/plans; O=$Q/out
PC=$BBT/cascade_compare/phase_c; PD=$BBT/cascade_compare/phase_d; BO=$BBT/cascade_compare/batchopt
export PYTHONPATH=$BBT/ising_compare:$BBT/qldpc_neural:$PC:$PD:$BO:$T13:$C OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       PYTHONDONTWRITEBYTECODE=1 QND_RELAY=$Q/relay BB_PIDNS=1 TRITON_CACHE_DIR=$Q/cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$Q/cache/inductor CASCADE_ROOT=$Q/d13root CONVLINES_RELEASE=$P/evaluation \
       NCCL_IB_DISABLE=1 PC_GPU_DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)" \
       CTRT_PUB=$P CTRT_PLANS=$PL CTRT_CKPT=$P/weights/cascade_bb72/C14p.pt
PY=$Q/venv/bin/python; CK=$P/weights/cascade_bb72/C14p.pt
