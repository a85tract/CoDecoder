#!/bin/bash
# The smoke test of the training release (training/TRAINING.md, step 0): a tiny surface-code front end trained from a
# Stim circuit for a few minutes on a CPU, then its threshold tuned with PyMatching as the strong decoder.
#   bash training/smoke.sh [OUT]     from the repository root; PY (default python), SAMPLES (1e6), BATCH (256)
# The threshold follows the paper's rule with M = 0.9 x PyMatching's own failures on the 100,000 calibration shots.
# Writes OUT/d5_p003.stim (Stim's rotated surface code, d = 5, 5 rounds, X basis, uniform circuit noise p = 0.3 %),
# OUT/train/ (final_ema.pt, log.jsonl, run.json), OUT/tune.json and OUT/tune_frontier.csv, and the logs.
set -euo pipefail
OUT=${1:-runs/smoke}
PY=${PY:-python}
ROOT=$(cd "$(dirname "$0")/.." && pwd)                      # the repository root
export PYTHONPATH="$ROOT/evaluation:$ROOT/training${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
$PY - "$OUT/d5_p003.stim" <<'PYEOF'
import sys
import stim
stim.Circuit.generated("surface_code:rotated_memory_x", distance=5, rounds=5, after_clifford_depolarization=0.003,
                       before_round_data_depolarization=0.003, before_measure_flip_probability=0.003,
                       after_reset_flip_probability=0.003).to_file(sys.argv[1])
PYEOF
$PY -m convlines_train.train --circuit "$OUT/d5_p003.stim" --out "$OUT/train" --samples "${SAMPLES:-1e6}" --seed 1 \
    --batch "${BATCH:-256}" --eval-shots 20000 --eval-every 2.5e5 --workers 2 2>&1 | tee "$OUT/train.log"
$PY -m convlines_train.tune --frontend "$OUT/train/final_ema.pt" --shots 100000 --seed 2 \
    --strong convlines_train.strong:pymatching --max-fraction 0.2 --step 0.01 --rule paper --target-strong 0.9 \
    --out "$OUT/tune.json" 2>&1 | tee "$OUT/tune.log"
