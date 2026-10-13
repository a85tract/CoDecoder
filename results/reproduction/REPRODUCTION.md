# Reproduction on a rented H100 SXM (RunPod), 2026-10-06

A re-run of the main README's Reproduction section (sections 1-5) from the working tree at commit 13e85cf, on one
rented RunPod secure-cloud pod with a whole NVIDIA H100 80GB HBM3 (SXM). Everything that the release says can be
reproduced was reproduced: the test sets bit for bit, every accuracy cell of ours against plain PM, correlated PM
and the official Ising numerics (failures, p-values and verdicts equal to the registered rows), the d = 7 cubin byte
for byte, the deployed routing, the figure, the composite check, and the timing harnesses end to end with all 30
speed cells holding on the reproduced inputs. The timing session is not a qualified session in the registered sense
(a shared cloud host, see "Caveats"), so its numbers check the harness and the margins, not the registered values.
The BP-OSD opponents were read in full at d = 7 and 9 and on the registered 200,000-shot subset at d = 15 (every
cell as registered), and on 200,000-shot subsets at d = 11 and 13; every Ising TensorRT part the studies read was
rebuilt and read on the full test sets; the two post hoc controls added upstream during the run reproduce exactly.

The main README meant here is the earlier release's; its Reproduction section is now `results/surface/SURFACE.md`.
Paths below are the earlier release's: its `results/`, `weights/`, `testsets/`, `timing/` and `convlines/` are
`results/surface/`, `weights/surface/`, `evaluation/testsets/`, `evaluation/timing/` and `evaluation/convlines/` here.

Raw records are in `raw/` (every JSON and log the commands wrote, the host description in `raw/out/host/`), the
scripts that ran them in `commands/`, and `composite_repro.py` recomputed the composite from `raw/` on the pod (section 4).

## Host and environment

| item | this run | the studies |
|---|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, max SM clock 1,980 MHz | one H100 80GB HBM3 (SXM), rented cloud node |
| CPU | Intel Xeon Platinum 8480+ (2 sockets, 224 logical CPUs on the host); the pod's cgroup allows 23.8 CPUs | timing host of the speed work: Xeon Platinum 8480+; CPU terms: AMD EPYC 7713 / 7443, exclusive nodes |
| memory | 251 GB | |
| OS, image | Ubuntu 22.04.5, `runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04` (its torch was not used) | |
| Python and packages | 3.12.14 (uv); torch 2.11.0+cu126, Triton 3.6.0, numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, safetensors 0.8.0, matplotlib 3.11.2 (`raw/out/host/pip-freeze.txt`) | the same pins |
| compilers | nvcc 12.1.105 (`cuda-nvcc-12-1` from NVIDIA's apt repository), g++ 11.4.0 | nvcc 12.1.105 |
| Ising | NVIDIA Ising-Decoding at 3ce4379, checkpoints with sha256 08ce4b1e… (M1) and f49df7fc… (M4); hydra-core 1.3.7, omegaconf 2.3.1, ldpc 2.4.1, beliefmatching 0.2.0 | the same |
| Ising TensorRT | a second venv: tensorrt_cu12 10.14.1.48.post1, nvidia-modelopt 0.47.0, onnx 1.21.0, onnxruntime-gpu 1.24.4 and the rest of section 4's list (`raw/out/host/pip-freeze-trt.txt`) | the same |
| BP-OSD | a third venv: cudaq-qec-cu12 0.8.0, numpy 2.5.3, torch 2.11.0+cpu (`raw/out/host/pip-freeze-bposd.txt`) | cudaq-qec 0.8.0, numpy 2.5.3 |

The repository was copied to the pod (`/workspace/repo`) without `.git`; `commands/setup.sh` and `commands/envs.sh`
built the three environments.

## 2. Circuits and test sets

- `python -m convlines.testset regen` for every chunk: T2 at d = 7 (chunks 0-199), 9, 11, 13 (0-99) and T3 at
  d = 15 (0-99). All 600 files have the manifest's sha256 (`raw/logs/regen_d*.log`, one `== manifest` line per
  chunk), and `python -m convlines.testset check` re-hashed them: "500 of 500 T2 chunk files present, 0 differ"
  and "100 of 100 T3 chunk files present, 0 differ" (`raw/logs/testset_check.log`). The pod's CPU has AVX2 (and
  AVX-512); the whole regeneration took a few minutes with five processes.
- `python scripts/make_circuits.py` (NVIDIA's code at 3ce4379): all five circuits and grids "equal"
  (`raw/logs/make_circuits.log`). `convlines.circuits` accepts all five files; `sha256sum -c weights/MANIFEST.sha256`
  passes (`raw/out/manifest.log`).
- `python -m convlines.testset val --d D` for D = 7-15 (chunks 0-19). `timing.deployed` recognises the d = 15 shots
  as the studies' (`studies_shots: true`, digest 7b2cb1ef…); the d = 7-13 shots differ from the studies' ARM64 ones,
  as the README says.

## 3. Accuracy

`python -m convlines.evaluate --set T2|T3 --d D --chunks <all> --shots shots/T2|T3 --routing deployed
--opponents "plain PM" "correlated PM" "Ising M1 + PM" "Ising M4 + PM"` on every test shot, our c from the deployed
executables (the Triton engine at d = 9-15, the resident kernel of `build/d7` at d = 7). Results in
`raw/out/eval_d*.json`; the registered values are the `pm`, `pm_corr`, `M1_torch` and `M4_torch` rows of
`results/s2_test_eval.csv` and `results/s3_test_eval.csv`.

| d | shots | ours: this run / registered | routed (this run) | plain PM | correlated PM | Ising M1 + PM (torch) | Ising M4 + PM (torch) |
|---|---|---|---|---|---|---|---|
| 7 | 2,000,000 | **33,582** / 33,582 | 81,229 (4.06 %) | 60,138 / 60,138 | 35,291 / 35,291 | 48,875 / 48,875 | 38,095 / 38,095 |
| 9 | 1,000,000 | **10,789** / 10,789 | 9,910 (0.99 %) | 27,317 / 27,317 | 12,917 / 12,917 | 22,900 / 22,900 | 17,016 / 17,016 |
| 11 | 1,000,000 | **7,215** / 7,215 | 9,944 (0.99 %) | 24,358 / 24,358 | 9,325 / 9,325 | 20,280 / 20,280 | 14,871 / 14,871 |
| 13 | 1,000,000 | **5,627** / 5,627 | 19,405 (1.94 %) | 21,501 / 21,501 | 6,549 / 6,549 | 17,491 / 17,491 | 12,422 / 12,422 |
| 15 | 1,000,000 | **3,911** / 3,911 | 60,078 (6.01 %) | 18,442 / 18,442 | 4,311 / 4,311 | 15,288 / 15,288 | 10,081 / 10,081 |

Every cell is "better" with the McNemar p and the 97.5 % Newcombe interval of the registered row (the p-values agree
to three digits, for example d = 7 against correlated PM 5.48·10⁻⁴⁸ and d = 15 against correlated PM 7.0·10⁻³²).
At every chunk of every distance, 0 shots were routed differently from the published masks in `routing/`. (The
Ising columns of the main README's table are the fewest failures over all of Ising's parts, TensorRT plans included;
these are the PyTorch rows.)

The deployed executables themselves:

- `timing/d7/build.sh` with nvcc 12.1.105 and g++ 11.4.0 built `kernel_sm90a.cubin` with sha256 8f4180d7…, byte for
  byte the studies' cubin (`raw/logs/build_d7.log`; the router library's sha256 is 47facaf9…, g++ 11.4 against the
  release's g++ 12.2).
- `python -m timing.d7.run` on T2 chunks 0 and 199: 186 and 186 failures, 403 and 412 routed shots, all 20,000
  decisions equal to the fp32 reference's, the routed set equal to |c| < τ, every routed decision equal to the
  PyMatching wheel's, the largest |c| difference from the reference 0.025 and 0.020 (`raw/logs/d7run.log`).
- `python -m timing.deployed` on the 200,000 validation shots: 7,955 / 1,980 / 1,945 / 3,879 / 11,822 routed at
  d = 7 / 9 / 11 / 13 / 15, the README's verification counts; at d = 15 the mask equals the studies' published mask
  (11,822 routed, 0 different) (`raw/logs/deployed_d*.log`).
- The smoke test of the README's verification, `convlines.evaluate --set T2 --d 9 --chunks 0` with the four
  opponents: ours 94, plain PM 233, correlated PM 124, Ising M1 233, Ising M4 152 (`raw/out/smoke_d9.json`).

BP-OSD (min-LLR and BP 60, `cudaq-qec-cu12` 0.8.0 on the H100, `--routing published --device cpu` for ours,
`raw/out/eval_bposd*.json`, `raw/logs/eval_bposd*.log`): on this pod the two arms are CPU-bound (the OSD
post-processing runs on the host; `nvidia-smi` showed the GPU at 0-20 %), one chunk of 10,000 shots costing 16 s at
d = 7, 90 s at d = 9, about 4 min at d = 11 and 10 min at d = 13 with one process holding the pod's whole CPU quota;
five processes in parallel starved each other (`raw/logs/bposd_parallel_attempt/`), so the arms ran one distance at
a time. The full T2 read at d = 11 and 13 (100 chunks each) was out of reach in this session; the arms were read on the
first 20 chunks (200,000 shots) at every distance, the registered subset itself at d = 15, then the full sets at d = 7 and 9.

| d | shots read here | ours | opponent | failures here | McNemar p | verdict | registered: shots, ours, opponent, p, verdict |
|---|---|---|---|---|---|---|---|
| 7 | 2,000,000 (0-199) | 33,582 | min-LLR BP-OSD | 34,229 | 0.000224 | better | 2,000,000, 33,582, 34,227, 0.000234, better |
| 7 | 2,000,000 (0-199) | 33,582 | BP 60 BP-OSD | 41,124 | 0 | better | 2,000,000, 33,582, 41,124, 0, better |
| 9 | 1,000,000 (0-99) | 10,789 | min-LLR BP-OSD | 16,332 | 0 | better | 1,000,000, 10,789, 16,332, 0, better |
| 9 | 1,000,000 (0-99) | 10,789 | BP 60 BP-OSD | 19,873 | 0 | better | 1,000,000, 10,789, 19,873, 0, better |
| 11 | 200,000 (0-19) | 1,390 | min-LLR BP-OSD | 3,021 | 9.43e-216 | better | 1,000,000, 7,215, 15,078, 0, better |
| 11 | 200,000 (0-19) | 1,390 | BP 60 BP-OSD | 3,747 | 0 | better | 1,000,000, 7,215, 18,881, 0, better |
| 13 | 200,000 (0-19) | 1,150 | min-LLR BP-OSD | 2,599 | 1.27e-183 | better | 1,000,000, 5,627, 13,317, 0, better |
| 13 | 200,000 (0-19) | 1,150 | BP 60 BP-OSD | 3,489 | 0 | better | 1,000,000, 5,627, 17,623, 0, better |
| 15 | 200,000 (0-19) | 768 | min-LLR BP-OSD | 2,320 | 1.63e-230 | better | 200,000, 768, 2,318, 9.55e-231, better |
| 15 | 200,000 (0-19) | 768 | BP 60 BP-OSD | 3,247 | 0 | better | 200,000, 768, 3,247, 0, better |

At d = 15 the registered read *is* this subset, so the comparison is exact: ours 768 failures as registered; BP 60
BP-OSD 3,247 as registered, with the same 459 shots that only ours fails and 2,938 that only BP 60 fails; min-LLR
BP-OSD 2,320 against the registered 2,318 (459 against 456 shots only ours fails), two decisions per 200,000 apart,
which is within the GPU belief propagation's floating-point run-to-run variation; both cells "better" with p-values
equal to the registered ones to within a factor of two. The d = 7-13 subsets agree with the registered direction at every distance where the margin is large (d = 9-13: ours fails
34-67 % less often, every cell "better"). At d = 7, the registered weakest cell (ours 33,582 against min-LLR's
34,227 on 2,000,000 shots, p = 2.3·10⁻⁴), 200,000 shots are not enough: 3,402 against 3,483, p = 0.15, "not shown",
as the registered interval width predicts. The full d = 7 read (2,000,000 shots) then reproduces the registered
cell: min-LLR BP-OSD 34,229 failures against the registered 34,227 (14,997 shots only ours fails against 14,994;
p = 2.2·10⁻⁴ against 2.3·10⁻⁴; interval [−5.2·10⁻⁴, −1.3·10⁻⁴] as registered), "better"; BP 60 BP-OSD 41,124
exactly, with the registered 14,818 / 22,360 discordant counts. The full d = 9 read (1,000,000 shots) reproduces both cells exactly (16,332 and 19,873
failures; discordant counts 4,853 / 10,396 against 4,852 / 10,397 for min-LLR, 4,814 / 13,898 as registered for BP
60). At d = 11 and 13 only the 200,000-shot subsets were read.

Ising's TensorRT parts (`timing.ising_trt build --parts registered`, then `eval` on every test shot, exactly B
shots per call; `raw/logs/trt_eval_d*.log`, per-shot outputs in `ising_eval/` on the pod): every part the studies read
was rebuilt on this H100 and read on the same test sets. Most FP16 plans reproduce the registered counts exactly (every
registered-best FP16 part does), the others within 43 shots per million; the FP8 plans at d = 7-13 differ by 31-99 per million because their
calibration rows (validation chunk 0) are the x86-64 regeneration, not the studies' ARM64 shots (the README's warning):
at d = 15, where the regenerated calibration chunk is the studies' own, every one of the twelve FP8 parts reproduces
its registered count exactly, and the FP16 parts are within 7.

| d | part | failures: this run / registered | difference |
|---|---|---|---|
| 7 | M1_fp16_B1 | 48,878 / 48,879 | -1 |
| 7 | M1_fp16_B4 | 48,877 / 48,877 | +0 |
| 7 | M1_fp16_B32 | 48,866 / 48,880 | -14 |
| 7 | M1_fp16_B64 | 48,878 / 48,878 | +0 |
| 7 | M1_fp8_B1 | 48,808 / 48,890 | -82 |
| 7 | M1_fp8_B32 | 48,800 / 48,893 | -93 |
| 7 | M4_fp16_B1 | 38,088 / 38,088 | +0 |
| 7 | M4_fp16_B32 | 38,087 / 38,076 | +11 |
| 7 | M4_fp16_B64 | 38,088 / 38,088 | +0 |
| 7 | M4_fp8_B1 | 38,036 / 38,000 | +36 |
| 7 | M4_fp8_B32 | 38,036 / 38,004 | +32 |
| 7 | M1_fp16_L3_B1 | 48,877 / 48,877 | +0 |
| 7 | M1_fp16_L5_B1 | 48,885 / 48,885 | +0 |
| 7 | M1_fp8_L3_B1 | 48,808 / 48,890 | -82 |
| 7 | M1_fp8_L5_B1 | 48,808 / 48,890 | -82 |
| 7 | M4_fp16_L3_B1 | 38,088 / 38,088 | +0 |
| 7 | M4_fp16_L5_B1 | 38,096 / 38,096 | +0 |
| 7 | M4_fp8_L3_B1 | 38,036 / 38,000 | +36 |
| 7 | M4_fp8_L5_B1 | 38,036 / 38,000 | +36 |
| 9 | M1_fp16_B1 | 22,888 / 22,888 | +0 |
| 9 | M1_fp16_B4 | 22,888 / 22,883 | +5 |
| 9 | M1_fp16_B16 | 22,898 / 22,898 | +0 |
| 9 | M1_fp16_B32 | 22,905 / 22,905 | +0 |
| 9 | M1_fp16_B64 | 22,904 / 22,904 | +0 |
| 9 | M1_fp8_B1 | 22,908 / 22,841 | +67 |
| 9 | M1_fp8_B16 | 22,932 / 22,850 | +82 |
| 9 | M1_fp8_B64 | 22,922 / 22,854 | +68 |
| 9 | M4_fp16_B1 | 17,002 / 17,005 | -3 |
| 9 | M4_fp16_B16 | 17,001 / 17,001 | +0 |
| 9 | M4_fp16_B32 | 17,002 / 17,002 | +0 |
| 9 | M4_fp16_B64 | 17,002 / 17,002 | +0 |
| 9 | M4_fp8_B1 | 16,972 / 17,071 | -99 |
| 9 | M4_fp8_B16 | 16,975 / 17,071 | -96 |
| 9 | M4_fp8_B64 | 16,975 / 17,067 | -92 |
| 9 | M1_fp16_L3_B1 | 22,888 / 22,888 | +0 |
| 9 | M1_fp16_L5_B1 | 22,884 / 22,884 | +0 |
| 9 | M1_fp8_L3_B1 | 22,908 / 22,841 | +67 |
| 9 | M1_fp8_L5_B1 | 22,908 / 22,841 | +67 |
| 9 | M4_fp16_L3_B1 | 17,005 / 17,005 | +0 |
| 9 | M4_fp16_L5_B1 | 17,017 / 17,017 | +0 |
| 9 | M4_fp8_L3_B1 | 16,972 / 17,071 | -99 |
| 9 | M4_fp8_L5_B1 | 16,972 / 17,071 | -99 |
| 11 | M1_fp16_B1 | 20,270 / 20,270 | +0 |
| 11 | M1_fp16_B8 | 20,270 / 20,269 | +1 |
| 11 | M1_fp16_B16 | 20,287 / 20,287 | +0 |
| 11 | M1_fp16_B32 | 20,280 / 20,280 | +0 |
| 11 | M1_fp16_B64 | 20,280 / 20,269 | +11 |
| 11 | M1_fp8_B1 | 20,202 / 20,281 | -79 |
| 11 | M1_fp8_B8 | 20,202 / 20,288 | -86 |
| 11 | M1_fp8_B32 | 20,199 / 20,290 | -91 |
| 11 | M4_fp16_B1 | 14,868 / 14,859 | +9 |
| 11 | M4_fp16_B2 | 14,868 / 14,850 | +18 |
| 11 | M4_fp16_B8 | 14,868 / 14,871 | -3 |
| 11 | M4_fp16_B16 | 14,859 / 14,859 | +0 |
| 11 | M4_fp16_B32 | 14,858 / 14,858 | +0 |
| 11 | M4_fp16_B64 | 14,858 / 14,865 | -7 |
| 11 | M4_fp8_B1 | 14,879 / 14,819 | +60 |
| 11 | M4_fp8_B32 | 14,875 / 14,817 | +58 |
| 11 | M1_fp16_L3_B1 | 20,270 / 20,270 | +0 |
| 11 | M1_fp16_L5_B1 | 20,261 / 20,261 | +0 |
| 11 | M1_fp8_L3_B1 | 20,202 / 20,281 | -79 |
| 11 | M1_fp8_L5_B1 | 20,202 / 20,281 | -79 |
| 11 | M4_fp16_L3_B1 | 14,859 / 14,859 | +0 |
| 11 | M4_fp16_L5_B1 | 14,866 / 14,866 | +0 |
| 11 | M4_fp8_L3_B1 | 14,879 / 14,819 | +60 |
| 11 | M4_fp8_L5_B1 | 14,879 / 14,819 | +60 |
| 13 | M1_fp16_B1 | 17,491 / 17,491 | +0 |
| 13 | M1_fp16_B4 | 17,491 / 17,497 | -6 |
| 13 | M1_fp16_B8 | 17,489 / 17,489 | +0 |
| 13 | M1_fp16_B16 | 17,488 / 17,531 | -43 |
| 13 | M1_fp16_B32 | 17,496 / 17,488 | +8 |
| 13 | M1_fp8_B1 | 17,458 / 17,489 | -31 |
| 13 | M1_fp8_B16 | 17,452 / 17,498 | -46 |
| 13 | M4_fp16_B1 | 12,427 / 12,427 | +0 |
| 13 | M4_fp16_B4 | 12,411 / 12,411 | +0 |
| 13 | M4_fp16_B8 | 12,414 / 12,414 | +0 |
| 13 | M4_fp16_B16 | 12,418 / 12,421 | -3 |
| 13 | M4_fp16_B32 | 12,411 / 12,411 | +0 |
| 13 | M4_fp8_B1 | 12,448 / 12,403 | +45 |
| 13 | M4_fp8_B16 | 12,444 / 12,399 | +45 |
| 13 | M1_fp16_L3_B1 | 17,491 / 17,491 | +0 |
| 13 | M1_fp16_L5_B1 | 17,478 / 17,478 | +0 |
| 13 | M1_fp8_L3_B1 | 17,458 / 17,489 | -31 |
| 13 | M1_fp8_L5_B1 | 17,458 / 17,489 | -31 |
| 13 | M4_fp16_L3_B1 | 12,427 / 12,427 | +0 |
| 13 | M4_fp16_L5_B1 | 12,425 / 12,425 | +0 |
| 13 | M4_fp8_L3_B1 | 12,448 / 12,403 | +45 |
| 13 | M4_fp8_L5_B1 | 12,448 / 12,403 | +45 |

A consequence worth stating: the README's Ising columns take each model's fewest failures over its parts, and at
d = 7 (M4), 9 (M1) and 15 (M1) that minimum is an FP8 plan. Rebuilt here, the FP8 plans move both ways: at d = 7
no M4 plan reaches the registered 38,000 (the rebuilt FP8 plans give 38,036), at d = 9 the M1 FP8 plans give 22,908
against the registered 22,841 while the M4 FP8 plans give 16,972, below the registered best M4 part (17,001). So the
Ising columns of the README's accuracy table are tied to the studies' own FP8 plans at the ±100-per-million level;
the cells are unaffected (ours is thousands below).

## 4. Speed

The whole chain of section 4 ran on the pod after the accuracy runs, with nothing else on the GPU: the gate's
measurable parts, (a) five fresh `timing.gpu_term` processes per network interleaved across the networks, (c) five
d = 7 handoff processes and one router process per network at d = 9-15, (b) three interleaved fresh-process
repetitions of `timing.cpu_term term` and `ref` per network, then `convert` (`commands/stage3.sh`; records in
`raw/timing_out/`, `raw/cpu_out/`, logs in `raw/logs/`).

The timed core was chosen by the session rule (the GPU's NUMA node is node 1; the least busy physical core over 1 s,
outside core 0's: core 63, SMT sibling 175). Gate-like measurements on it (`raw/out/gate_core.json`,
`raw/cpu_out_plain/`): busy 1.4 % and steal 0 over 5 s; plain PM per block 36.15 / 36.35 / 36.05 µs at d = 9
(limit 1.10 × 35.895 = 39.5; spread 0.8 %) and 124.56 / 123.44 µs at d = 13 (limit 147.0; spread 0.9 %); the
router's floor 4.75-4.97 µs (limit 1.15 × 4.986 = 5.73); every SM-clock sample of every process at 100 % of the
maximum with no throttle reason. No separate 1 GiB / fp16 load process ran, and the 10-minute quiet-host screen of
study 3 was not run (the pod shares its host with other tenants, under a CPU quota).

| d | (a) GPU, five processes (µs) | slowest, registered | (c) handoff (µs) | registered | (b) node, three reps (µs) | ratio | (b) Grace, registered | composite, registered | fastest opponent (margin) |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 8.547, 8.558, 8.565, 8.558, 8.575 | **8.575**, 8.443 | measured 2.091, 2.135, 2.132, 1.687, 1.670 → **2.135** | 2.240 | 2.448, 2.450, 2.511 | 0.9488 | **2.383**, 2.240 | **13.092**, 12.924 | plain PM 17.16 (4.07) |
| 9 | 28.583, 28.404, 28.425, 28.601, 28.412 | **28.601**, 27.907 | estimate 0.547 → allowance **1.100** | 1.100 (est. 0.586) | 1.342, 1.348, 1.343 | 0.7605 | **1.025**, 1.042 | **30.726**, 30.049 | plain PM 36.99 (6.26) |
| 11 | 40.987, 41.111, 41.098, 41.098, 41.097 | **41.111**, 40.613 | estimate 0.581 → **1.100** | 1.100 (est. 0.589) | 2.573, 2.640, 2.581 | 0.8481 | **2.239**, 2.212 | **44.450**, 43.925 | Ising M1 + PM 68.63 (24.18) |
| 13 | 49.755, 49.676, 49.767, 49.782, 49.778 | **49.782**, 49.153 | estimate 0.253 → **1.100** | 1.100 (est. 0.299) | 8.768, 8.789, 8.763 | 0.8434 | **7.413**, 7.705 | **58.295**, 57.957 | Ising M1 + PM 90.26 (31.97) |
| 15 | 49.961, 49.967, 50.053, 49.912, 50.039 | **50.053**, 50.885 | estimate 0.164 → **1.100** | 1.335 (est. 1.335) | 41.160, 41.090, 41.341 | 0.8242 | **34.075**, 33.860 | **85.228**, 86.080 | Ising M1 + PM 122.85 (37.62) |

`composite_repro.py` printed this table from `raw/` on the pod (its `raw/timing_out/` records were not archived, so it
cannot be rerun here) and checked the 30 cells against
the registered opponents' times: all 30 hold (ours strictly below every opponent at every d). The composites are
within 2.3 % of the registered ones (above at d = 7-13 by 0.6-2.3 %, below at d = 15 by 1.0 %), the (a) terms within 2.5 % (above by 1.2-2.5 % at d = 7-13, below by 1.6 % at d = 15).

Details worth knowing:

- Every process's outputs are bit for bit equal to `timing.deployed`'s logits (`outputs_equal_deployed: true`; at
  d = 7 every one of the 200,000 shots, and every device interval shorter than its host round trip).
- The router processes at d = 9-15 (`raw/timing_out/*_handoff_p0.json`): complete path 33.462 / 47.519 / 62.425 /
  96.425 µs, device only 31.447 / 44.129 / 52.683 / 53.064 µs, routed 1,980 × 148.2 / 1,945 × 288.8 / 3,879 × 489.3 /
  11,822 × 730.8 µs; the routed set is |c| < τ, the unrouted decisions the sign of c, the routed decisions the
  PyMatching wheel's (`routing: true`).
- The node's correlated PM per block (median of three): 37.64 / 88.88 / 183.47 / 331.54 / 564.09 µs at d = 7-15
  (the studies' nodes: 82.79 µs at d = 9 on the EPYC 7713). `timing.cpu_term term --d 15 --mask routing/val_d15.npz`
  (the studies' own mask, accepted because the d = 15 shots are the studies') gave 41.667 µs per shot, 11,822 routed,
  conditional mean 704.9 µs (`raw/cpu_out_pub/`).
- **Acceptance verdicts.** `timing.gpu_term` writes `accepted: false` for every process here, for one reason only:
  its "GPU alone" check looks for the process's own pid in nvidia-smi's compute-app list, and inside this container
  nvidia-smi reports host-namespace pids (`[Not Found]` names), which never match. The lists themselves show the GPU
  alone: `apps_before` is empty and `apps_after` has exactly one entry (the process itself) in all 34 process records,
  and an independent sampler during the GPU term saw at most one compute process at every sample
  (`raw/logs/gpu_alone_samples.txt`). Every other check (clocks, outputs, d = 7's device-within-host, the router's
  routing checks) is true.

Ising's TensorRT plans at batch 1 (`timing.ising_trt build / check / time`, a second venv, `raw/logs/trt_*.log`,
`raw/out/ising_b1.csv`): `prep` exported both models at every distance and quantised them (FP8 Q/DQ on the
convolutions, calibrated on validation chunk 0); `build` made all 40 timed plans (M1 / M4 × FP16 / FP8 × level 3 / 5
at five distances); `check` passed every one of them against the official PyTorch numerics + PM on validation chunk 0
(agreement 99.73-100 %, no "worse" verdict, densities within 0.4 %), with FP8 execution confirmed on all 20 FP8 plans
(`raw/logs/trt_check_d*.log`); `time` ran each distance's eight plans in forward and reversed order, five
repetitions of 100 CUDA-graph replays per plan. The fastest validated plan per model, as the registered rule reads it
(the fastest repetition; the registered plan and time from `results/`):

| d | Ising M1: fastest plan, this run | registered | Ising M4: fastest plan, this run | registered | Ising M1 + PM: this run / registered | Ising M4 + PM: this run / registered |
|---|---|---|---|---|---|---|
| 7 | 42.53 µs (FP8 L3) | 43.39 (FP8 L3) | 55.61 µs (FP8 L5) | 56.48 (FP8 L5) | 47.90 / 48.76 | 60.54 / 61.41 |
| 9 | 43.68 µs (FP8 L3) | 44.77 (FP8 L3) | 57.50 µs (FP8 L5) | 58.29 (FP8 L5) | 55.31 / 56.40 | 67.29 / 68.08 |
| 11 | 45.70 µs (FP8 L3) | 45.94 (FP8 L3) | 59.41 µs (FP8 L5) | 59.37 (FP8 L3) | 68.39 / 68.63 | 78.71 / 78.67 |
| 13 | 50.57 µs (FP8 L3) | 50.61 (FP8 L3) | 65.54 µs (FP8 L5) | 65.19 (FP8 L5) | 90.22 / 90.26 | 98.69 / 98.34 |
| 15 | 55.80 µs (FP8 L5) | 57.35 (FP8 L5) | 73.32 µs (FP8 L3) | 74.71 (FP8 L5) | 121.30 / 122.85 | 128.24 / 129.63 |

(The totals add the registered Grace-basis PM share of `timing/grace_basis.csv`.) The plan times are within 2.7 % of
the registered ones, FP8 is the fastest precision at every distance as registered, and our reproduced composites
stay below the reproduced Ising totals at every distance (margins 24.2-37.6 µs at d = 11-15).

## Audit: looking for errors in the release

Beyond re-running the commands, the records and claims of the release were checked independently. Nothing changes a
registered cell. Found:

1. **Two different Apple M4 measurements are quoted for the same quantities, without saying so.** The Grace
   extrapolations in `timing/grace_basis.csv` start from Apple M4 core times of 6.62 / 14.26 / 27.76 µs (plain PM at
   d = 7 / 9 / 11), 15.04 / 65.54 µs (correlated PM at d = 7 / 11) and ratios 2.594 and 2.374, which imply M4 d = 13
   times of 49.97 µs (plain) and 117.79 µs (correlated). The M4-basis figure and the README's "What is not
   established" use 6.68 / 14.16 / 27.77 / 49.47 µs (plain) and 15.07 / 33.66 / 66.46 / 118.61 µs (correlated) for the
   same decoders on the same core. The two sets differ by up to 1.4 %; no speed cell is close enough to be affected
   (re-deriving the Grace values from the figure's M4 numbers moves plain PM at d = 7 from 17.16 to 17.50 µs and
   correlated PM at d = 11 from 155.61 to 156.7 µs). The README should say which M4 session each number comes from.
2. **The acceptance records of the registered timing processes are not in the release.** `results/s2_composite.json`
   carries one `a_gpu` per network (the slowest of five, per the rule) but not the five values, and no clock samples
   or acceptance verdicts for the (a) processes; the handoff records (`c_beside`) at d = 9-13 carry clock windows
   whose minimum SM clock is 345 / 345 / 825 MHz with the idle throttle reason seen, which the released harness's own
   rule (`sm_min_frac >= 0.97` over the sampled window) would reject, and the d = 15 handoff record
   (`c_measured_detail`), whose estimate (1.335 µs) is the one that counted, has no clock samples at all. So the
   README's rule 4 ("every timing process afterwards is accepted or replaced") cannot be checked from the release for
   any registered process. (In this run, every process's sampled window stayed at 100 % of the maximum clock.)
3. **The harness's "GPU alone" check cannot pass inside a container.** `timing.gpu_term` looks for its own pid in
   nvidia-smi's compute-app list; with a pid namespace (Docker, RunPod) nvidia-smi reports host pids, so every
   process is marked `accepted: false` even when it was alone. Comparing the list's length, or the GPU's used memory
   before the process starts, would make the check portable.
4. **Nothing else was found wrong.** Checked and consistent: all 144 rows of `results/s*_test_eval.csv` recomputed
   from their counts with an independent implementation of the exact McNemar test and Newcombe's method 8 (p-values,
   intervals, phi and verdicts equal to 1e-12); the Ising columns of the README's accuracy table are the minimum over
   each model's parts in the CSVs, with the part names as stated (ties at d = 7 M4, d = 9 M1 and d = 15 M1 noted);
   the speed table's Ising totals are the registered plan times plus the Grace PM shares; the composite table's
   products and sums; the handoff table against `c_beside` and `c_measured_detail`; the "what it means" margins;
   the routed fractions on the test and validation sets; the calibration counts in `weights/configs.json`; the
   parameter counts (the safetensors files hold 2,352 / 6,480 / 14,520 / 28,392 / 50,400 more values than the README's
   counts, exactly the non-trainable buffer A of detectors × lines, which `configs.json` says is included in the
   file); the drand beacon of both manifests (randomness = sha256(signature), and the round's signature equal to
   what `api.drand.sh` serves today for rounds 6,520,833 and 6,524,587); all 600 seeds in [2⁶², 2⁶³) and distinct;
   every routing mask's sha256, shot count and routed count against `routing/manifest.json`; the samplers' CPUs; the
   test sets' sizes on disk; the README's verification counts (ours on chunk 0 and the last chunk at every d, Ising
   M1 / M4 on chunk 0 at every d, 768 failures of ours on T3 chunks 0-19, 7,955 / 1,980 / 1,945 / 3,879 routed
   validation shots at d = 7-13, 11,822 at d = 15 equal to the studies' mask); and the d = 7 cubin's sha256.
Also checked, on this pod (`raw/out/audit/`): `timing.deployed --logits fp32` against the deployed masks on the
   200,000 validation shots routes 7 / 3 / 4 / 13 / 59 shots differently at d = 7 / 9 / 11 / 13 / 15 (the README
   states 3 at d = 9 and 59 at d = 15), with no unrouted shot's sign differing; `convlines.evaluate --routing fp32`
   on test chunk 0 routes 0 / 1 / 0 / 0 / 7 shots differently from the published masks with the same failures
   (the README's 1 and 7), and on the ten chunks of the README's verification 11 shots in all, as it states; the published-routing failures equal the deployed-routing ones on T2 chunks 0-19 at
   d = 7 and 9. An observation, not an error: on the validation shots the largest |c_fp32 − c_deployed| is 0.05 /
   1.77 / 0.35 / 0.90 / 3.73 at d = 7-15, larger than the 0.02 / 0.6-1.0 / 0.1-0.2 / 0.3 / 1.2-1.5 the README's
   verification reports on its ten test chunks; at d = 15 that gap is close to τ = 3.95, so the "up to rounding"
   phrase of the `--routing fp32` row understates how far the fp32 reference can sit from the deployed c on a rare
   shot (the decisions still agreed on every shot checked here).
A wording nit: the README says the larger networks were "initialised from smaller-distance or shallower
   networks", but `weights/configs.json` records the d = 11 network's three-block ancestor as "transferred from
   d = 13" (a larger distance); the network table itself (blocks, width, parameters, q, τ, grids, detector counts,
   calibration failures, training sample counts) matches `configs.json` exactly.
Hygiene: the README's own commands write `masks/`, `cpu_out/`, `onnx/`, `trt/`, `ising_eval/` and result JSON
   files into the repository root, but `.gitignore` lists only `shots/`, `timing_out/` and `build/`.

## 5. The figure and the composite check

`python figure/acc_speed.py` rebuilt both PNGs byte for byte equal to the committed files (sha256 95f81852… and
3c7447e3…; `raw/out/figure.log`), and `python -m timing.composite` reproduced the five registered composites and all
30 speed cells from `results/` (`raw/out/composite.log`).

## 6. The two post hoc controls (added upstream during this run)

Commits 89dc0c8 and 952b265 arrived on `main` while this run was in progress, adding section 6: Ising M1 / M4
followed by correlated PM, not registered claims. The pod's tree was synced to commit 926be27 (into a second copy
sharing the shots, masks, build and venv) and section 6's steps 2-5 and 7 were run (`raw/controls/`,
`commands/stage6.sh`). Step 6 (an Apple M4 core) was not run.

Step 3, every control on every test shot with `--routing deployed`, against `results/controls_test_eval.csv` and
`results/controls_decisions.csv`:

| d | ours | control | failures: this run / registered | only ours / only control: this run (registered) | p | verdict | decisions sha256 equal | observables sha256 equal |
|---|---|---|---|---|---|---|---|---|
| 7 | 33,582 | Ising M1 + correlated PM | 49,061 / 49,061 | 12,198 / 27,677 (12,198 / 27,677) | 0 | better / better | True | True |
| 7 | 33,582 | Ising M4 + correlated PM | 38,154 / 38,154 | 13,382 / 17,954 (13,382 / 17,954) | 1.52e-147 | better / better | True | True |
| 9 | 10,789 | Ising M1 + correlated PM | 22,851 / 22,851 | 3,689 / 15,751 (3,689 / 15,751) | 0 | better / better | True | True |
| 9 | 10,789 | Ising M4 + correlated PM | 16,972 / 16,972 | 3,987 / 10,170 (3,987 / 10,170) | 0 | better / better | True | True |
| 11 | 7,215 | Ising M1 + correlated PM | 19,989 / 19,989 | 2,651 / 15,425 (2,651 / 15,425) | 0 | better / better | True | True |
| 11 | 7,215 | Ising M4 + correlated PM | 14,829 / 14,829 | 2,802 / 10,416 (2,802 / 10,416) | 0 | better / better | True | True |
| 13 | 5,627 | Ising M1 + correlated PM | 17,105 / 17,105 | 2,423 / 13,901 (2,423 / 13,901) | 0 | better / better | True | True |
| 13 | 5,627 | Ising M4 + correlated PM | 12,255 / 12,255 | 2,571 / 9,199 (2,571 / 9,199) | 0 | better / better | True | True |
| 15 | 3,911 | Ising M1 + correlated PM | 14,845 / 14,845 | 1,954 / 12,888 (1,954 / 12,888) | 0 | better / better | True | True |
| 15 | 3,911 | Ising M4 + correlated PM | 9,916 / 9,916 | 2,122 / 8,127 (2,122 / 8,127) | 0 | better / better | True | True |

All ten rows reproduce exactly, and each control's decisions on all 12,000,000 shots have the registered sha256, as
do the observables. Step 2 (`timing.cpu_term residuals`): at d = 15 both models' residual detectors and logical frames have the registered content hashes (`results/controls_speed.json`); at d = 7-13 the hashes differ, as
`results/controls_speed.json` says they must (the studies' validation chunks there were sampled on ARM64).
Steps 4-5 (the CPU timing on the pinned core 63 after the BP-OSD reads had finished, three interleaved
fresh-process repetitions, then `convert --share`; `raw/controls/ctrl_cpu/`, `raw/controls/ctrl_cpu.log`), against
`results/controls_speed.json`; a control's speed adds the registered plan time, as the rule says:

| d | model | node's correlated-PM share on Ising's residuals, median of 3 (µs) | share on the Grace basis: this run / registered (µs) | control's speed: this run / registered (µs) | ours (this run's composite, µs) |
|---|---|---|---|---|---|
| 7 | M1 | 14.40 | 9.44 / 9.96 | 52.83 / 53.34 | 13.09 |
| 7 | M4 | 12.24 | 8.04 / 8.59 | 64.52 / 65.07 | 13.09 |
| 9 | M1 | 39.25 | 25.72 / 23.68 | 70.49 / 68.45 | 30.73 |
| 9 | M4 | 33.11 | 21.74 / 19.94 | 80.03 / 78.23 | 30.73 |
| 11 | M1 | 84.67 | 55.49 / 51.10 | 101.44 / 97.05 | 44.45 |
| 11 | M4 | 70.41 | 46.22 / 41.60 | 105.59 / 100.97 | 44.45 |
| 13 | M1 | 159.84 | 104.76 / 93.41 | 155.38 / 144.02 | 58.30 |
| 13 | M4 | 132.43 | 86.94 / 74.65 | 152.13 / 139.84 | 58.30 |
| 15 | M1 | 271.07 | 177.67 / 158.55 | 235.02 / 215.90 | 85.23 |
| 15 | M4 | 223.22 | 146.54 / 128.58 | 221.25 / 203.29 | 85.23 |

The shares on the Grace basis are within 16 % of the registered ones (this host's factor 0.6554405869574128 against the
registered node's), and ours stays below both controls at every distance. Step 7: `python -m timing.composite` at
commit 926be27 reproduces the controls' composites and counts ours below the controls in 40 of 40 cells, and
`python figure/acc_speed.py` rebuilds all four PNGs (the Grace and M4 bases, each with the matched-boundary view)
byte for byte equal to the committed files (`raw/controls/composite_head.log`, `raw/controls/figure_head.log`).

## Caveats

- The timing session is not a qualified one: the pod shares a 224-CPU host with other tenants under a 23.8-CPU
  quota, the quiet-host screen and the 1 GiB load process of the gate were not run, and the CPU term ran on the
  GPU host's Xeon core rather than an exclusive EPYC node (its Grace conversion uses the same node's correlated-PM
  speed, as the rule says). The numbers above test the harness, the bitwise equality of the engine's outputs and the
  margins to the opponents, not the registered values.
- The d = 7-13 validation shots are x86-64 regenerations and differ from the studies' ARM64 ones; the timing uses
  their statistics only (the README's note).
- nvidia-smi's pid namespace makes the harness's own "GPU alone" verdict uninformative here (above).
- The RunPod account was used through its API; the pod id was `<pod-1>`.
