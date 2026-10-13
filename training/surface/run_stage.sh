#!/bin/bash
# One training stage of a released surface-code network with the frozen program (training/TRAINING.md, step 5):
#   bash training/surface/run_stage.sh STAGE [more arch.train arguments]      from the repository root
# STAGE is an argument file of training/surface/ising_compare/scripts/alpine_arch/args/ without .txt (the stages and
# their order: training/LINEAGE.md). Its arch.train line runs with three changes:
# - the parent checkpoint is read from $ICMP_CKPT/<parent run>/final_ema.pt;
# - --init-sha is dropped, because it pins the original parent's sha256 and a retrained parent has its own;
# - --no-val is added. The validation monitor reads stored shots that are not in this release, and it does not change
#   the training.
# Needs a CUDA GPU, ISING_DECODING_DIR (README section 3) and absolute ICMP_CKPT and ICMP_OUT; PY defaults to python.
# Checkpoints go to $ICMP_CKPT/<run>/ (final_ema.pt is the read-out).
set -euo pipefail
STAGE=$1
shift
LINE=$(grep -m1 -- "-m arch.train" "training/surface/ising_compare/scripts/alpine_arch/args/$STAGE.txt")
LINE=$(sed -E -e 's#\$A/ckpt/([^ ]+)/final_ema\.pt#$ICMP_CKPT/\1/final_ema.pt#g' \
              -e 's#\$D/init/([^ ]+)_final_ema\.pt#$ICMP_CKPT/\1/final_ema.pt#g' \
              -e 's# --init-sha [0-9a-f]+##' <<< "$LINE")
[[ $LINE == *--no-val* ]] || LINE="$LINE --no-val"
cd training/surface/ising_compare
export PYTHONPATH=. ICMP_NV=${ICMP_NV:-$ISING_DECODING_DIR/code}
PY=${PY:-python}
mkdir -p "$ICMP_CKPT" "$ICMP_OUT"
echo "$LINE $*"
eval "$LINE $*"
