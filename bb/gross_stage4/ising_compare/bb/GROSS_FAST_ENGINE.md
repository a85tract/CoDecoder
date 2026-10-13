# BB144 (gross) fast engine for the frozen network g2_D12k113c10

2026-10-07. Exploratory: nothing here is registered. The network is g2_D12k113c10 (`final_ema.pt`,
sha256 eb380295...) at p = 0.004, decoded one shot per call (batch 1). The reference is the frozen `graph32` engine
(bb.engine: the fp32 PyTorch module replayed as one captured CUDA graph, TF32 off). Its stage-2 H100 read of CAL set
tau = 0.5629201531410217 (q = 0.5 %), and stage 3 (cell G-4) uses it. The new code is `bb/fast.py`. `engine.py`,
`net.py` and `fused.py` are unchanged, so every frozen engine signature still holds.

Platform: one RunPod H100 80GB HBM3 (driver 580.126.09, MIG off), set up the way the gross
stage-3 pod was: gross2's lock (Python 3.12.14, torch 2.11.0+cu126, cuDNN 91002, Triton 3.6.0). TensorRT 10.16.1.11
and ONNX 1.23.2 sit in a second venv built from the same lock and are used by trt16 only. On this pod, graph32 re-read
CAL chunk 0 bit for bit (decisions and margins), and its timing rows carry stage 3's binding 50648cb2afd5efad. There
are no A100 numbers. The study lead put Derecho's GPUs off limits at 17:01 PDT because the QEC cap of 192 A100-hours
since 09-26 is used up, so all development and every number below come from the H100.

## Candidates

| kind | code | plain meaning |
|---|---|---|
| graph32 | bb.engine, frozen | the reference: fp32, one CUDA graph (GPU kernels recorded once and replayed in one launch) |
| graph | bb.engine, frozen | (a) the same module under bf16 autocast: convolutions and matmuls in bfloat16 (8-bit mantissa); vote and margin in fp32 |
| graph16 | bb.fast | (a) the same under fp16 autocast (11-bit mantissa) |
| compile32, compile16 | bb.fast | (b) torch.compile, mode max-autotune (kernels picked by benchmarking), with inductor freezing (weights folded in as constants), on the fp32 or fp16-autocast call. The CUDA graph is captured by hand, as bb.engine does for its inductor rung, because max-autotune's own CUDA graphs cannot run inside a captured graph |
| trt16 | bb.fast | (c) TensorRT FP16 kernels (LayerNorm kept in fp32), built from an ONNX export of the trunk and read-out. The detector parity, vote and margin run in fp32 PyTorch, and everything is in one CUDA graph |
| fused | bb.fused, frozen settings | (d) BB72's engine port: 16 Triton kernels (stem, 12 residual blocks, pooling, vote, top-2) with fp16 matmul inputs and fp32 accumulation; the residual stream, LayerNorm, vote and margin are fp32 |
| fusedt | bb.fused, tuned settings | (d) the same kernels and source, launched with 16 taps per iteration, 4 pipeline stages and no PDL (programmatic dependent launch, a Hopper feature that starts the next kernel early) |

The network fits bb.fused as it is, since it has no time blocks and no read-out branch R. The earlier "no fused rung
on gross" was a limit of the Blackwell card: one kernel needs 128 KB of shared memory, more than sm_120's 99 KB but
well within the H100's 227 KB.

## Equality on CAL (100,000 shots, batch 1, against the frozen graph32 H100 read)

The margin is the score gap between the best and second-best logical class. Its CAL median is 53.7, and tau routes
the 500 shots with the smallest margins to sector Relay-BP. "Hybrid" means the network on kept shots and sector
Relay-BP on routed ones (graph32: 409 failures). p is the exact two-sided McNemar test on the shots where only one of
the two reads fails.

| kind | decisions that differ | routing changes at tau (in / out) | network alone: candidate / graph32 failures (only cand : only g32, p) | hybrid: candidate / graph32 failures (only cand : only g32, p) | margin difference: median / 99.9th pct / max | plain meaning |
|---|---|---|---|---|---|---|
| fused, fusedt | 3 | 6 (3 / 3) | 717 / 717 (0 : 0, 1) | 406 / 409 (0 : 3, 0.25) | 0.024 / 0.97 / 7.1 | the closest fast kind; fusedt's read is bitwise equal to fused's |
| graph16 | 9 | 14 (8 / 6) | 716 / 717 (0 : 1, 1) | 405 / 409 (2 : 6, 0.29) | 0.045 / 1.24 / 19.0 | twice fused's margin error |
| graph (bf16) | 91 | 86 (47 / 39) | 713 / 717 (10 : 14, 0.54) | 402 / 409 (21 : 28, 0.39) | 0.33 / 9.6 / 36.8 | ten times fp16's changes |
| compile16 | 12 | 17 (7 / 10) | 717 / 717 (1 : 1, 1) | 409 / 409 (6 : 6, 1) | 0.044 / 1.25 / 12.1 | like graph16 |
| compile32 | 0 | 1 (1 / 0) | 717 / 717 (0 : 0, 1) | 408 / 409 (0 : 1, 1) | 0.000 / 0.001 / 0.01 | fp32 summed in another order: almost bitwise |
| trt16 | 36 | 33 (14 / 19) | 718 / 717 (3 : 2, 1) | 405 / 409 (9 : 13, 0.52) | 0.13 / 4.8 / 35.6 | TensorRT may run every layer but LayerNorm in fp16 |

Every candidate leaves both failure counts within noise (p >= 0.25). The changed decisions come from rounding on
shots whose top two scores are nearly tied. One CAL margin lies 1.4e-6 from tau, so any engine that is not bitwise
equal to graph32 moves a few shots across the threshold.

## Speed on the H100

This is bb.latency's protocol: 200 warm-up graph replays, then 5 x 100 timed replays with CUDA events; a process's
value is the mean. Each kind ran in 3 fresh processes, interleaved with the other kinds, each after bb.timing's idle
check. All 24 processes were accepted (SM clock at 100 % of its maximum throughout, one compute app on the card).

| kind | GPU term, slowest of 3 (ms) | the 3 processes (ms) | speed-up over graph32 | handoff, slowest (ms) | plain meaning |
|---|---|---|---|---|---|
| graph32 | 3.433 | 3.433, 3.409, 3.430 | 1.0x | 0.019 | stage 3 measured 3.430 ms |
| compile32 | 3.459 | 3.459, 3.458, 3.459 | 0.99x | 0.018 | no gain: the autotuner kept cuDNN's fp32 convolutions |
| graph | 1.502 | 1.502, 1.502, 1.502 | 2.29x | 0.017 | convolutions in bf16 |
| graph16 | 1.497 | 1.496, 1.497, 1.491 | 2.29x | 0.016 | convolutions in fp16 |
| compile16 | 0.806 | 0.805, 0.805, 0.806 | 4.26x | 0.017 | |
| trt16 | 0.780 | 0.778, 0.779, 0.780 | 4.40x | 0.017 | |
| fused | 0.385 | 0.384, 0.384, 0.385 | 8.93x | 0.017 | BB72's launch settings |
| fusedt | 0.254 | 0.254, 0.254, 0.254 | 13.5x | 0.019 | the recommendation |

graph32 runs each tap convolution as a dense conv3d over the taps' bounding box, with a 0/1 mask zeroing the rest. For
a block that box is 3 x 11 x 6 = 198 kernel positions for 87 taps, computed in fp32 without tensor cores. compile32's
autotuner timed that cuDNN convolution at 0.264 ms per call (`compile32_conv_autotune.txt`), so the 12 blocks take
about 3.2 of graph32's 3.43 ms. The fused engine gathers only the real taps and multiplies them on tensor cores, in
16 kernels.

Tuning (`fused_sweep_h100.txt`): 24 launch settings (4, 8 or 16 taps per iteration; 1 to 4 stages; PDL on or off), in
one process, each timed as 3 x 1,000 replays. BB72's frozen settings (8 taps, 2 stages, PDL on) were the slowest at a
359.7 us median. The fastest were (16, 3, off) at 242.8 us and (16, 4, off) at 243.5 us, equal within run noise.
fusedt uses (16, 4, off), the best in an earlier run of the same sweep. PDL slowed most settings, by about 90 us at
BB72's 8 taps and 2 stages.

Where fusedt's time goes (`fused_breakdown_h100.txt`, each part replayed alone in its own CUDA graph): 252.5 us in all.
The 12 residual blocks take 200.4 us (17.3 us each), and the stem 11.8, pooling 20.7, vote 7.8 and top-2 4.0 us. One
block does about 0.18 GFLOP, so it runs at about 1 % of the H100's fp16 tensor peak. Latency sets its time: 59 tiles
of 16 sites run on 132 SMs, and each tile works through 6 dependent gather-and-multiply steps. Roughly halving
the block time would need a different block kernel, for example splitting the taps of a tile over two programs and
summing in a fixed order. That would save about 0.1 ms, roughly 2 % of stage 3's composite with fusedt (about 5.2 ms,
below), so it was not done.

## Recommendation

Use fusedt for the next registration. It cuts the GPU term from 3.433 ms to 0.254 ms (3.18 ms less per shot, 13.5x)
and changes 3 of 100,000 CAL decisions and 6 routings relative to graph32. The network-alone failures stay at 717 and
the hybrid's go from 409 to 406, both within noise. It adds no kernel code: it is bb.fused's source as frozen for BB72
with three launch arguments (`bb.fast.FUSEDT`). Its reads are deterministic, since there are no atomics and the launch
shapes are fixed; fused and fusedt gave bitwise-equal CAL reads.

- Stage 3 (G-4) is not affected. Its frozen engine is graph32, and its signature, code and reads are unchanged. Its
  test composite was 3.4296 + 0.0226 + 4.9547 = 8.407 ms (7d06cd80); with fusedt's GPU term and handoff from this pod
  it would be about 0.254 + 0.019 + 4.955 = 5.23 ms, an estimate that mixes sessions.
- A policy on fusedt needs its own CAL read, a tau fitted on that read's margins, and bb.validate's engine check
  against the fp32 module, all before the test it is frozen for.
- With stage 2's CAL frontier, which uses Relay-BP's recorded Alpine times, the faster GPU term makes q = 0.6 %
  admissible under the 15.8 ms limit: 345 failures at an estimated 14.48 ms composite, against 409 at q = 0.5 % and
  15.12 ms with graph32. CAL is in-sample for this choice, so only a fresh test could confirm it.
- graph, graph16, compile16 and trt16 are all slower than fusedt and change more decisions, and compile32 is no faster
  than graph32.

## Records

`ising_compare/data/bb/fast_engine/` holds the following.
- `eq_<kind>_h100.json`: each comparison, with both engine signatures.
- `rows_h100.csv`: bb.timing rows, 24 gpu_term, 24 handoff and 30 idle checks (6 failed and were retried; every
  process started after one that passed).
- `fused_sweep.py` with `fused_sweep_h100.txt`, and `fused_breakdown.py` with `fused_breakdown_h100.txt`.
- `compile32_conv_autotune.txt` (inductor's convolution choices), `graph32_recheck_chunk0.json`, `gpu.txt`,
  `freeze_g2venv.txt`, `freeze_trtvenv.txt` and `eval.log`.

The scripts are `bb/scripts/pod/fast_setup.sh` and `fast_eval.sh`, run with EQ="fused fusedt graph16 graph compile16
compile32 trt16" and LAT="graph32 fused fusedt graph16 graph compile16 compile32 trt16". During the run,
fast_eval.sh's idle helper reused the loop variable, so the rows' proc column is not the process index; the helper has
been fixed since, and the process column still separates each kind's 3 processes. BB_SESSION was not set, so the
session column is empty.
The pod was created about 17:02 PDT and released at 17:36, about 35 minutes (about $2 at the
stage-3 pod's $3.49 per hour; this pod's rate was not given).

Reproduce: `python -m bb.fast eq --ckpt final_ema.pt --kind fusedt --label h100 --out E.json` and
`python -m bb.fast lat --ckpt final_ema.pt --kind fusedt --proc 1 --out T.csv` (3 fresh processes; the slowest counts).
