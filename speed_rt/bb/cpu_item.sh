#!/bin/bash
# SPEED-RT BB stage: ONE CPU process of the BB comparisons, for the CPU worker's session on winserver (CURRENT RULES, 13 October 15:45: every
# CPU part is timed there, on the Grace basis; the pods time GPU work only). The harness (slots, concurrency, tmpfs, reference segments,
# windows, acceptance) is the CPU worker's; this script is what one slot runs, with the frozen BB code and the registered arguments:
#     cpu_item.sh ITEM PROC OUT.csv        ITEM from `cpu_item.sh --list`, PROC = the round (1, 2, 3; 0 for a warm-up), OUT.csv = the rows file
#     cpu_item.sh --list                   the item names, one per line: "ITEM KIND SHOTS"
#     cpu_item.sh --warm                   the untimed warm-up processes (one per CPU decoder), one "ITEM PROC" line each, to run with PROC 0 into their own CSV
# One process = one decoder built once, the shots read, then a timed loop around decode() only (bb.timing opponent) or the routed fraction x
# the mean decode time over the first routed shots in shot order (bb.stage3 cputerm, bb.stage4_gross cputerm). Single-threaded (OMP,
# OPENBLAS, MKL, NUMEXPR and RAYON threads 1). Pin it to one core (taskset) in the harness; nothing here pins.
# Inputs (paths on winserver, set by the caller):
#   BBW_REL     the deployed tree of the stage's freeze commit (bb/, speed_rt/, weights/bb not needed)
#   BBW_DRAW    the draw: shots/ (the 65 timing-shot files), manifest_bb72.csv, manifest_gross.csv (from the stage's B commit and Alpine)
#   BBW_PREDS   the GPU pod's routing reads (the hand-off): the 65 npz files of out_gpu/preds, checked against preds.sha256
#   BBW_RELAY   the Relay-BP clone at d185194 whose tests/testdata holds the circuits
#   BBW_PY      the CPU environment's python: cpuvenv.lock (Python 3.12.14, relay-bp 0.2.2, ldpc 2.4.1, numpy 2.2.6, scipy 1.18.1, stim 1.16.0)
# Dry runs only: BBW_DRY_FIRST=N caps every shot count at N. For the rows' session label set BB_SESSION. Every input the process reads should sit in tmpfs (the harness's rule).
set -uo pipefail
if [ "${1:-}" = "--list" ]; then
  cat <<'EOF2'
relay_p0.003 baseline 10000
bposd_p0.003 baseline 10000
relay_p0.002 baseline 10000
bposd_p0.002 baseline 10000
relay_full_p0.003 baseline 2000
relay_full_p0.002 baseline 2000
cpu_BB72-3S term 1000
cpu_BB72-2S term 1000
cpu_BB72-3F term 1000
cpu_BB72-2F term 5000
relay_p0.004 baseline 1000
bposd_p0.004 baseline 500
relay_full_p0.004 baseline 75
cpu_G-4R term 300
cpu_G-4F term 25
EOF2
  exit 0
fi
if [ "${1:-}" = "--warm" ]; then
  cat <<'EOF2'
relay_p0.003:warm 0
bposd_p0.003:warm 0
relay_full_p0.003:warm 0
relay_p0.004:warm 0
bposd_p0.004:warm 0
cpu_G-4R:warm 0
relay_full_p0.004:warm 0
cpu_G-4F:warm 0
EOF2
  exit 0
fi
ITEM=${1:?ITEM}; PROC=${2:?PROC}; CSV=${3:?OUT.csv}
: "${BBW_REL:?}" "${BBW_DRAW:?}" "${BBW_PREDS:?}" "${BBW_RELAY:?}" "${BBW_PY:?}"
T72=$BBW_REL/bb/bb72/ising_compare; TG=$BBW_REL/bb/gross_stage4/ising_compare; SHIM=$BBW_REL/speed_rt/bb/srt_bb.py
export QND_RELAY=$BBW_RELAY BB_SHOTS=$BBW_DRAW/shots BB_PREDS=$BBW_PREDS BB_PIDNS=1 BB_SESSION=${BB_SESSION:-srt-bb-cpu} \
       PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
unset SRT_BEACON SRT_DRY SRT_DRY_SHOTS SRT_ORIGINAL_SEEDS          # a timing step never draws shots
WARM=0; case "$ITEM" in *:warm) WARM=1; ITEM=${ITEM%:warm} ;; esac
w() { [ "$WARM" = 1 ] && echo "$1" || echo "$2"; }                  # a warm-up's shot count, or the registered one
cap() { if [ -n "${BBW_DRY_FIRST:-}" ] && [ "$BBW_DRY_FIRST" -lt "$1" ]; then echo "$BBW_DRY_FIRST"; else echo "$1"; fi; }   # dry runs only: every shot count at most N
TREE=$T72; MAN=$BBW_DRAW/manifest_bb72.csv
case "$ITEM" in
  relay_p0.00[23]|bposd_p0.00[23]|relay_full_p0.00[23])
    dec=${ITEM%%_p0*}; p=0.${ITEM##*_p0.}; chunk=0; first=10000; [ "$dec" = relay_full ] && first=2000
    [ "$WARM" = 1 ] && { chunk=1; first=1000; }
    ARGS=("$SHIM" bb.timing opponent --code bb72 --p "$p" --dec "$dec" --split test --chunks $chunk --first "$(cap "$first")" --proc "$PROC" --out "$CSV") ;;
  cpu_BB72-3S) ARGS=("$SHIM" bb.stage3 cputerm --amend data/bb/stage3/amendment_S.json --cell BB72-3S --split test --chunks 0-19 --tag r12 --first "$(cap 1000)" --proc "$PROC" --out "$CSV") ;;
  cpu_BB72-2S) ARGS=("$SHIM" bb.stage3 cputerm --amend data/bb/stage3/amendment_S.json --cell BB72-2S --split test --chunks 0-29 --tag r12 --first "$(cap 1000)" --proc "$PROC" --out "$CSV") ;;
  cpu_BB72-3F) ARGS=("$SHIM" bb.stage3 cputerm --cell BB72-3F --split test --chunks 0-19 --tag r12 --first "$(cap 1000)" --proc "$PROC" --out "$CSV") ;;
  cpu_BB72-2F) ARGS=("$SHIM" bb.stage3 cputerm --cell BB72-2F --split test --chunks 0-29 --tag r12 --first "$(cap 5000)" --proc "$PROC" --out "$CSV") ;;
  relay_p0.004|bposd_p0.004|relay_full_p0.004)
    TREE=$TG; MAN=$BBW_DRAW/manifest_gross.csv; dec=${ITEM%%_p0*}; first=1000; [ "$dec" = bposd ] && first=500; [ "$dec" = relay_full ] && first=75   # gross counts cut (STAGE2_bb 1.3)
    [ "$WARM" = 1 ] && { first=100; [ "$dec" = relay_full ] && first=10; }
    ARGS=("$SHIM" bb.timing opponent --code bb144 --p 0.004 --dec "$dec" --split test4 --chunks 0 --first "$(cap "$first")" --proc "$PROC" --out "$CSV") ;;
  cpu_G-4R) TREE=$TG; MAN=$BBW_DRAW/manifest_gross.csv; ARGS=("$SHIM" bb.stage4_gross cputerm --cell G-4R --split test4 --first "$(cap "$(w 100 300)")" --proc "$PROC" --out "$CSV") ;;
  cpu_G-4F) TREE=$TG; MAN=$BBW_DRAW/manifest_gross.csv; ARGS=("$SHIM" bb.stage4_gross cputerm --cell G-4F --split test4 --first "$(cap "$(w 10 25)")" --proc "$PROC" --out "$CSV") ;;
  *) echo "unknown item $ITEM"; exit 2 ;;
esac
cd "$TREE" && export PYTHONPATH=.:../qldpc_neural SRT_MANIFEST=$MAN
exec "$BBW_PY" "${ARGS[@]}"
