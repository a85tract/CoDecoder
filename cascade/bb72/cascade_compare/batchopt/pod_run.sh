#!/bin/bash
# The profile on a pod (the H100 profile of PROFILE.md), after pod_deploy.sh: the venv once, then prof_run.sh with the pod's
# paths, detached so that a dropped ssh session does not stop it:
#   setsid nohup bash /opt/batchopt/code/cascade_compare/batchopt/pod_run.sh < /dev/null > /opt/batchopt/prof.log 2>&1 &
# PYBIN may name an existing venv with the same pins (the first H100 pod's /opt/h100t/venv has them); otherwise $Q/venv is built
# from req_cu128.txt the first time (Python 3.12). NSYS and NCU default to the pod's nsys and ncu if it has them. The
# outputs go to $Q/out/<time>, with SHA256SUMS and END last.
set -uo pipefail
Q=${Q:-/opt/batchopt}; T=$(readlink -f $Q/code)
if [ -z "${PYBIN:-}" ]; then
  PYBIN=$Q/venv/bin/python
  if [ ! -x $PYBIN ]; then
    PIP="$Q/venv/bin/pip --disable-pip-version-check"
    python3.12 -m venv $Q/venv && $PIP install -q pip==25.0.1 && $PIP install -q --index-url \
      https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.org/simple \
      -r $T/cascade_compare/batchopt/req_cu128.txt || { echo "venv failed"; exit 1; }
  fi
fi
export PYBIN RUN="" O=$Q/out/$(date +%Y%m%d_%H%M%S) NSYS=${NSYS:-$(command -v nsys || true)} \
       NCU=${NCU:-$(command -v ncu || true)}
export PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$T/cascade_compare/phase_c OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
       QND_RELAY=<cluster-scratch>/bb/relay TRITON_CACHE_DIR=$Q/cache/triton \
       TORCHINDUCTOR_CACHE_DIR=$Q/cache/inductor PC_TREE="$(cat $T/DEPLOYED)"
mkdir -p $O $Q/cache/triton $Q/cache/inductor
nvidia-smi --query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.limit --format=csv,noheader
echo "tree $PC_TREE out $O python $PYBIN nsys ${NSYS:-none} ncu ${NCU:-none}"
cd $T/ising_compare
bash $T/cascade_compare/batchopt/prof_run.sh
(cd $O && find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS)
echo done > $O/END
