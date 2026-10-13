#!/bin/bash
cd /workspace/repo && . .venv/bin/activate && export ISING_DECODING_DIR=/workspace/Ising-Decoding PATH=$HOME/.local/bin:$PATH
mkdir -p logs out masks build
OPP=("plain PM" "correlated PM" "Ising M1 + PM" "Ising M4 + PM")
# a. re-hash the regenerated sets
( python -m convlines.testset check --set T2 --dir shots/T2; python -m convlines.testset check --set T3 --dir shots/T3; echo CHECK DONE ) > logs/testset_check.log 2>&1 &
# b. circuits from NVIDIA's code
python scripts/make_circuits.py > logs/make_circuits.log 2>&1 &
# c. d = 7 build with nvcc 12.1.105, then deployed d7 and the full d7 evaluation
( NVCC=/usr/local/cuda-12.1/bin/nvcc bash timing/d7/build.sh build/d7 12 && echo BUILD DONE && \
  python -m timing.deployed --d 7 --val shots/val --out masks --build build/d7 > logs/deployed_d7.log 2>&1 && \
  python -m timing.d7.run --chunks shots/T2/d7_p0.006_X_test2_000.npz shots/T2/d7_p0.006_X_test2_199.npz > logs/d7run.log 2>&1 && \
  python -m convlines.evaluate --set T2 --d 7 --chunks 0-199 --shots shots/T2 --routing deployed --build build/d7 --opponents "${OPP[@]}" --out out/eval_d7.json > logs/eval_d7.log 2>&1 && echo EVAL D7 DONE ) > logs/build_d7.log 2>&1 &
# d. deployed logits and masks at d = 9-15
( for d in 9 11 13 15; do python -m timing.deployed --d $d --val shots/val --out masks > logs/deployed_d$d.log 2>&1; done; echo DEPLOYED DONE ) > logs/deployed.log 2>&1 &
# e. smoke test, then the full evaluations at d = 9-15
( python -m convlines.evaluate --set T2 --d 9 --chunks 0 --shots shots/T2 --routing deployed --opponents "${OPP[@]}" --out out/smoke_d9.json > logs/smoke_d9.log 2>&1 && echo SMOKE OK && \
  for d in 9 11 13; do python -m convlines.evaluate --set T2 --d $d --chunks 0-99 --shots shots/T2 --routing deployed --opponents "${OPP[@]}" --out out/eval_d$d.json > logs/eval_d$d.log 2>&1 & done; \
  python -m convlines.evaluate --set T3 --d 15 --chunks 0-99 --shots shots/T3 --routing deployed --opponents "${OPP[@]}" --out out/eval_d15.json > logs/eval_d15.log 2>&1 & wait; echo EVAL 9-15 DONE ) > logs/eval_chain.log 2>&1 &
# f. BP-OSD env
( cd /workspace/repo && uv venv --python 3.12.14 /workspace/venv-bposd && . /workspace/venv-bposd/bin/activate && uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu && uv pip install numpy==2.5.3 scipy==1.18.1 stim==1.16.0 PyMatching==2.4.0 safetensors==0.8.0 cudaq-qec-cu12==0.8.0 && python -c "import cudaq_qec, numpy; print('bposd env ok', numpy.__version__)"; echo BPOSD ENV DONE ) > logs/env_bposd.log 2>&1 &
# g. TensorRT env
( cd /workspace/repo && uv venv --python 3.12.14 /workspace/venv-trt && . /workspace/venv-trt/bin/activate && uv pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126 && uv pip install -r requirements.txt hydra-core==1.3.7 omegaconf==2.3.1 ldpc==2.4.1 beliefmatching==0.2.0 && uv pip install tensorrt_cu12==10.14.1.48.post1 nvidia-modelopt==0.47.0 onnx==1.21.0 onnxruntime-gpu==1.24.4 onnxscript==0.7.2 onnx_graphsurgeon==0.6.1 polygraphy==0.53.6 onnxslim==0.1.97 onnxconverter-common==1.16.0 onnx-ir==1.0.0 cupy-cuda12x==14.2.0 ml_dtypes==0.6.0 lief==1.0.0 cppimport==26.4.17 && python -c "import tensorrt, modelopt; print('trt env ok', tensorrt.__version__, modelopt.__version__)"; echo TRT ENV DONE ) > logs/env_trt.log 2>&1 &
wait
echo STAGE2 ALL DONE
