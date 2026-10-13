# Sourced on a pod (bash): the environment of a1/jobs/job.sbatch with the pod's paths. /workspace/a1 holds the code
# tree (code -> code.<commit>), the inputs from Alpine (in/: the Ising-Decoding code, the BB72 circuit files of the
# Relay clone, the trunk files, the d = 9 val_legacy and BB72 dev / monitor shots), the venv and the outputs.
export A=/workspace/a1
T=$(readlink -f $A/code)
export PYTHONPATH=$T/ising_compare:$T/neural_emulation:$T/qldpc_neural OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       ICMP_NV=$A/in/nv ICMP_SHOTS=$A/in/shots ICMP_CKPT=$A/ckpt/surface ICMP_OUT=$A/out/surface \
       BB_SHOTS=$A/in/bbshots QND_RELAY=$A/in/relay BB_CKPT=$A/ckpt/bb72 BB_OUT=$A/out/bb72 BB_PREDS=$A/preds_bb \
       A1_SETS=$A/sets A1_SETS_ROWS=$A/sets/rows.csv TRITON_CACHE_DIR=$A/triton_cache TORCHINDUCTOR_CACHE_DIR=$A/inductor_cache \
       PATH=$A/code/ising_compare/a1/pod/shim:$PATH
PY=/workspace/venv_cu128/bin/python
mkdir -p $A/logs $A/ckpt $A/out $A/sets $A/val $A/trunk $A/done
# a1/jobs/args/*.txt read the trunk files from $A/trunk and the job's end from squeue (shim/squeue answers on the pod)
for f in $A/in/trunk/*.pt; do [ -e $A/trunk/$(basename $f) ] || ln -s $f $A/trunk/; done
cd $T/ising_compare
