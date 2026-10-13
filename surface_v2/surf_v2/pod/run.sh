#!/bin/bash
# surf_v2/pod/run.sh ARGS...: SURF-v2's evaluation commands on an evaluation pod (E0, the dry pass; E1, the reads and
# the timing session): the commands of surf_v2/args/<ARGS> (several in order), run from the release checkout $REL at
# the pinned commit, every output under $W. The pod's layout:
#   $W/rel        the release at the pinned commit (git archive; DEPLOYED = the commit); the exported weights in
#                 rel/weights, uncommitted until the freeze
#   $W/venv       the frozen pod recipe (R17's: torch 2.11.0+cu126, Triton 3.6.0, Stim 1.16.0, PyMatching 2.4.0,
#                 numpy 2.2.6) plus requirements.txt's safetensors and scipy and the Ising extras
#   $W/ising_nv   NVIDIA Ising-Decoding at 3ce4379 with its two checkpoints (ISING_DECODING_DIR)
#   $W/shots      sv2cal, sv2dry, sv2test (once drawn) and val (surf_v2.valshots's, copied from Alpine)
#   $W/build/d7   timing/d7/build.sh's output (the d = 7 kernel and the router library)
#   $W/reads, $W/masks, $W/session, $W/logs
set -euo pipefail
W=${W:-/workspace/sv2eval}; REL=$W/rel; PY=${PY:-$W/venv/bin/python}; B=$W/build/d7
export PYTHONPATH=$REL ISING_DECODING_DIR=$W/ising_nv PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
mkdir -p $W/reads $W/masks $W/session $W/logs
cd $REL
echo "pod $(hostname) commit $(cat DEPLOYED 2>/dev/null || git rev-parse HEAD) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
for c in "$@"; do echo "== $c $(date -Is)"; eval "$(cat surf_v2/args/$c)"; done
date -Is; echo JOBDONE
