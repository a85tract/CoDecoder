#!/bin/bash
# pairs.sh WHAT (CASCADE-TRT, on the pod, called by run_steps.sh with its environment): the pairs through the harnesses'
# pipeline subcommands (trt_launch.py), each one fresh process; per rep and setting the compiled and the TensorRT Cascade
# alternate. Rows: $O/pairs_<harness>.jsonl (each row's "ctrt" names the engine and the launcher's patches).
#   pairs        D1 pairs S55p_deep -> C14p (d1_m2, depspam_fm, configs 0 / 1 = p 0.2 / 0.3 %, S on the multi-shot
#                engine "new"), and d = 13 C5 (sv_speed, the H100 run's derived lock, p 0.6 / 0.35 %)
#   pairs2       phase C pairs S55p -> C14p (pc_speed, amend1 depspam_fm lock, configs 2 / 7)
#   alone_pipe   Cascade alone in the same harnesses (d1_m2 config -1 at 0.2 %, sv_speed C2 at 0.6 %): cross-check
#   smoke        one short process per harness and engine (dry: shot counts cut, rows to $O/smoke_*.jsonl)
set -uo pipefail
WHAT=${1:?}
Q=${Q:-/workspace/ctrt}; P=$Q/pub; C=$Q/code; O=$Q/out; PY=$Q/venv/bin/python
PC=$P/cascade/bb72/cascade_compare/phase_c; PD=$P/cascade/bb72/cascade_compare/phase_d
L13=$P/speed_rt/cascade/records/H100/R4/derived_lock.json
L=$C/trt_launch.py
run() { echo "$(date +%T) $*"; "$@" || echo "$(date +%T) FAILED($?): $*"; }
n_of() { [ $1 = single ] && echo 20000 || echo 500000; }
d1() {  # eng cfg p mode rep [n] [out]
  CTRT_ENGINE=$1 run $PY $L $PD/d1_m2.py pipeline --setting depspam_fm --config $2 --p $3 --mode $4 --engine new \
    --tail-b 0 --rep $5 --n ${6:-$(n_of $4)} --out ${7:-$O/pairs_d1.jsonl}; }
d13() { # eng cfg p mode rep [out]
  CTRT_ENGINE=$1 run $PY $L surface.sv_speed pipeline --config $2 --p $3 --mode $4 --rep $5 --lock $L13 --stage c7 \
    --out ${6:-$O/pairs_d13.jsonl} --save $O/save_d13/$1_$2_$3_$4_r$5_$(date +%H%M%S); }
pcs() { # eng cfg p mode rep
  CTRT_ENGINE=$1 run $PY $L $PC/pc_speed.py pipeline --locked $PC/out/locked_depspam_fm.json --config $2 --p $3 \
    --mode $4 --rep $5 --n $(n_of $4) --speed $PC/out/speed_components.json --out $O/pairs_pc.jsonl; }
case $WHAT in
  smoke)
    for e in compiled trt; do
      d1 $e 0 0.002 single 1 2000 $O/smoke_d1.jsonl
      d1 $e 0 0.002 batched 1 20000 $O/smoke_d1.jsonl
      d13 $e C5 0.006 single 1 $O/smoke_d13.jsonl
    done ;;
  pairs)
    for rep in 1 2 3; do
      for cp in "0 0.002" "1 0.003"; do set -- $cp; for m in single batched; do for e in compiled trt; do
        d1 $e $1 $2 $m $rep; done; done; done
      for p in 0.006 0.0035; do for m in single batched; do for e in compiled trt; do
        d13 $e C5 $p $m $rep; done; done; done
    done ;;
  pairs2)
    for rep in 1 2 3; do
      for cp in "2 0.002" "7 0.003"; do set -- $cp; for m in single batched; do for e in compiled trt; do
        pcs $e $1 $2 $m $rep; done; done; done
    done ;;
  alone_pipe)
    for rep in 1 2 3; do for m in single batched; do for e in compiled trt; do
      d1 $e -1 0.002 $m $rep "" $O/alone_pipe_d1.jsonl
      d13 $e C2 0.006 $m $rep $O/alone_pipe_d13.jsonl
    done; done; done ;;
esac
