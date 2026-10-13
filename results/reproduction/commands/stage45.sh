#!/bin/bash
cd /workspace/repo && export ISING_DECODING_DIR=/workspace/Ising-Decoding
PT=/workspace/venv-trt/bin/python; PB=/workspace/venv-bposd/bin/python
mkdir -p logs trt ising_eval out
# stage 4a: Ising TensorRT timed plans: build, check, time (quiet GPU)
for d in 7 9 11 13 15; do $PT -m timing.ising_trt build --d $d --onnx onnx --work trt > logs/trt_build_d$d.log 2>&1; done
for d in 7 9 11 13 15; do $PT -m timing.ising_trt check --d $d --val shots/val --onnx onnx --work trt > logs/trt_check_d$d.log 2>&1; echo "check d$d exit $?" >> logs/trt_check.log; done
for d in 7 9 11 13 15; do
  $PT -m timing.ising_trt time --d $d --val shots/val --onnx onnx --work trt --order fwd --out out/ising_b1.csv > logs/trt_time_fwd_d$d.log 2>&1
  $PT -m timing.ising_trt time --d $d --val shots/val --onnx onnx --work trt --order rev --out out/ising_b1.csv > logs/trt_time_rev_d$d.log 2>&1
done
echo STAGE4A DONE > logs/stage4a.done
# stage 4b: the registered parts, built and evaluated on the test sets; stage 5: BP-OSD on the test sets, in parallel
( for d in 7 9 11 13 15; do $PT -m timing.ising_trt build --d $d --onnx onnx --work trt --parts registered > logs/trt_build_parts_d$d.log 2>&1; done
  $PT -m timing.ising_trt eval --d 7 --onnx onnx --work trt --parts registered --set T2 --shots shots/T2 --chunks 0-199 --out ising_eval > logs/trt_eval_d7.log 2>&1
  for d in 9 11 13; do $PT -m timing.ising_trt eval --d $d --onnx onnx --work trt --parts registered --set T2 --shots shots/T2 --chunks 0-99 --out ising_eval > logs/trt_eval_d$d.log 2>&1; done
  $PT -m timing.ising_trt eval --d 15 --onnx onnx --work trt --parts registered --set T3 --shots shots/T3 --chunks 0-99 --out ising_eval > logs/trt_eval_d15.log 2>&1
  echo STAGE4B DONE > logs/stage4b.done ) &
B=("min-LLR BP-OSD" "BP 60 BP-OSD")
( $PB -m convlines.evaluate --set T2 --d 7 --chunks 0-199 --shots shots/T2 --routing published --device cpu --opponents "${B[@]}" --out out/eval_bposd_d7.json > logs/eval_bposd_d7.log 2>&1 ) &
for d in 9 11 13; do ( $PB -m convlines.evaluate --set T2 --d $d --chunks 0-99 --shots shots/T2 --routing published --device cpu --opponents "${B[@]}" --out out/eval_bposd_d$d.json > logs/eval_bposd_d$d.log 2>&1 ) & done
( $PB -m convlines.evaluate --set T3 --d 15 --chunks 0-19 --shots shots/T3 --routing published --device cpu --opponents "${B[@]}" --out out/eval_bposd_d15.json > logs/eval_bposd_d15.log 2>&1 ) &
wait
echo STAGE45 DONE > logs/stage45.done
