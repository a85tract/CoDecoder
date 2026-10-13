# Batched decoding of the BB72 comparison: profile on the RTX PRO 6000 and the H100

11 October 2026, branch `ising-cascade-batchopt`.

This report reads two runs of `prof_batched.py` (this directory). The first is Alpine job CJ45 on a full RTX PRO 6000
Blackwell Server Edition (03:22-03:29 PDT on 11 October). The second is the run of the same steps on the second H100
pod, a rented H100 80GB HBM3 (04:06-04:10 PDT; `H100_TIMING.md`, "Amendment 1", on `ising-cascade-h100t2`). It also
quotes phase C's pipeline timings at the claim points: step C7 under Amendment 1 on the RTX PRO 6000 (job CJ39) and the
second H100 pod's timing of the same configurations. Nothing here changes phase C or enters its report.

Terms used throughout:

- A shot is one BB72 X-memory experiment with 6 syndrome rounds, so a time per round is the time per shot divided by 6.
- S55 is the small ConvLines network (483k parameters) on its fused engine: five Triton kernels with FP16 matmul inputs
  and FP32 accumulation, bitwise equal to the registered batch-1 engine at any batch size (`phase_c/pc_fused.py`).
- C14 is our re-implementation of Cascade (14 layers of width 512, 19.7M parameters), compiled with torch.compile and
  run in BF16.
- S → B is the routed pipeline: S55 decodes every shot and hands its q1 least confident shots to C14.
- A time per shot is GPU time with the input already on the GPU, unless marked host to host. aS and aB are the times
  per shot of S55 and C14 alone at batch 4,096.

## Summary

- S55 alone takes 4.09 µs per shot on the RTX PRO 6000 and 5.61 µs on the H100 at batch 4,096 (0.68 and 0.94 µs per
  round), within 4 % of that at every batch from 256 to 16,384. Per shot it executes 256 MFLOP on FP16 tensor cores and
  needs 1.0 MB of DRAM traffic, but its programs request 28.0 MB from L2. Three quarters of that is weights and tables
  that every program re-reads to process 16 rows of one shot. On-chip traffic sets its time. On the RTX PRO 6000,
  Nsight Compute finds the three residual blocks at 84-86 % of the L1/shared-memory pipe's peak and the read-out's pool
  and vote at 75 % and 92 % of L2's peak, with the tensor pipe busy 14-17 % of the cycles. On the H100, S55's tensor
  work runs at 6 % of the FP16 GEMM rate and its DRAM traffic at 6 % of the copy rate.
- C14 alone takes 71.6 µs per shot on the RTX PRO 6000 and 43.4 µs on the H100. It executes 7.7 GFLOP and moves about
  90 MB through DRAM per shot, so DRAM bandwidth and the BF16 tensor rate set its time. It reaches 86 % of the DRAM
  copy rate and 28 % of the BF16 GEMM rate on the RTX PRO 6000, and 69 % and 23 % on the H100. On the H100 the SM clock
  falls to 1,500-1,860 MHz at the 700 W power cap while C14 runs.
- The H100 has 2.1 times the RTX PRO 6000's DRAM bandwidth and 2.0 times its BF16 rate, which C14 uses: C14 runs 1.65
  times faster there. The RTX PRO 6000 has 1.75 times the H100's SM count × clock (188 SMs at 2.43 GHz against 132 at
  1.98 GHz). S55's residual blocks are bound inside the SM, so they run 1.64 times slower on the H100, and S55 as a
  whole 1.37 times slower.
- S → B costs aS + q1 · aB plus 0.05-0.60 µs per shot on the RTX PRO 6000 and 0.07-0.35 µs on the H100. Almost all of
  the excess is the padded rows of the last C14 batch. Copies, routing, the queue and the GPU's idle gaps add at most
  0.05 µs per shot, and S and C14 on two CUDA streams gain nothing.
- At the claim points S → B costs 0.888 µs per round at p = 0.2 % (q1 = 1 %) and 1.376 µs at 0.3 % (q1 = 5 %) on the
  RTX PRO 6000 (C7), and 1.067 and 1.355 µs on the H100. S55 is 77 % and 50 % of that on the RTX PRO 6000, 89 % and
  70 % on the H100.

[The summary's ranking of the levers and its target for the next stage are not part of this release (see the note at
section 8).]

## 1. What was measured

`prof_batched.py` loads each network with `bb.cascade_eval.Predictor`, as `pc_speed.py` does, and builds its frozen
batched path with `pc_speed`'s own builders:

- S55: `pc_speed.fused_batch` (`pc_fused.Batched` in a CUDA graph with programmatic dependent launch (PDL), as frozen:
  `fused@4096`), at batch 256, 1,024, 4,096 and 16,384. Every batch size was checked bitwise against the reads' engine
  (`pc_fused.Batched` at 4,096) on 32,768 rows, every decision and margin, and passed on both cards.
- C14: `torch.compile(pc_speed.make_fwd(pr), dynamic=False)` in a `pc_speed.Graph`, as `pc_speed.build(pr, "compiled",
  B)` builds it (frozen: `compiled@4096`), at 4,096 and 8,192.

Per path and batch size it records the time per shot as C3 measures it (`pc_speed.amortized`, input resident and host
to host), the GPU time per replay from CUDA events, and the time per kernel from torch.profiler. For S55 the kernel
times come twice: from graph replays with PDL, and from eager launches without PDL, so that each kernel's time is its
own. It also records Kineto's launch fields (registers, shared memory, estimated occupancy), the PTX of the Triton
kernels, and nvidia-smi samples of SM clock, power and temperature every 100 ms.

The work per shot comes from the model definitions. S55's FLOPs and bytes per kernel follow `pc_fused.Batched`'s
launch geometry, padding included. "L2-requested" bytes are everything its programs load and store, including the
weights and tables every program re-reads. "DRAM-compulsory" bytes write and read each per-shot buffer once. C14's
FLOPs come from `torch.utils.flop_counter` on the gather-matmul form of its convolution and equal
`bb.cascade_net.macs_per_shot`; its DRAM bytes come from a model of the compiled forward (`cascade_bytes`).

The ceilings were measured on the same card: cuBLAS GEMMs at 8192³ in BF16, FP16 (FP32 accumulation), FP8, TF32 and
FP32, a 1 GiB device copy, a copy that stays in L2, and pinned PCIe copies. Both copy rates count the bytes read plus
the bytes written.

The pipeline step reruns `pc_speed.pipeline`'s batched loop without Relay-BP: S over batches of 4,096, the routed rows
queued on the GPU, C14 whenever 4,096 wait and once at the end, padded. It runs 500,000 shots: the 100,000 kind-12 shots
of the uniform setting at p = 0.3 % (chunks 0-9), cycled five times. tau1 is the q1-quantile of S55's margins on those
shots, so exactly q1 × 500,000 shots are routed. CUDA events surround every GPU segment and host clocks every host step.
One run at q1 = 5 % ran under torch.profiler and one (200,000 shots) under Nsight Systems. An overlap test replays S's
and C14's graphs serially and on two streams.

Nsight Compute 2025.4.1 ran on the RTX PRO 6000 only: S55's seven launches at batch 4,096 with the full set, and C14's
first 120 launches at batch 4,096 (about four of its 14 layers) with the speed-of-light, launch, occupancy and
warp-state sections. Nsight Compute locks the clocks at base, so its durations are about 1.3 times the unprofiled ones;
its percentages are what this report uses. The H100's container has no access to the GPU's performance counters
(ERR_NVGPUCTRPERM), so the H100 has no Nsight Compute data. Nsight Systems 2025.5.2.266 was installed on the pod after
the timing and traced the pipeline there.

The RTX PRO 6000 ran the kernels phase C compiled: the formal run's Triton and Inductor caches were copied read-only,
including the `_gmm` autotuning choice of its C3 timing. The H100 compiled its own for sm_90.

This is a profile, not a phase-C measurement: one process per step, no repetitions, no screen. Its pipeline values
agree with phase C's formal timings to within 0.3 % (section 6.4).

## 2. The two cards

| | RTX PRO 6000 Blackwell Server Edition | H100 80GB HBM3 (SXM) | H100 / RTX PRO 6000 |
|---|---|---|---:|
| architecture | sm_120 | sm_90 | |
| SMs × maximum SM clock | 188 × 2,430 MHz | 132 × 1,980 MHz | 0.57 |
| L2 | 128 MiB | 50 MiB | |
| shared memory per block, maximum | 99 KiB | 227 KiB | |
| power limit | 600 W | 700 W | |
| driver, host | 590.48.01, Alpine gpu-node-08 (AMD EPYC 9555) | 580.126.09, RunPod (Intel Xeon Platinum 8470) | |
| BF16 GEMM, TFLOP/s (nominal) | 390 (504) | 777 (989) | 1.99 |
| FP16 GEMM with FP32 accumulation, TFLOP/s | 306 (504) | 746 (989) | 2.44 |
| FP8 GEMM, TFLOP/s | 687 (1,008) | 1,400 (1,979) | 2.04 |
| TF32 GEMM, TFLOP/s | 218 (252) | 395 (495) | 1.81 |
| FP32 GEMM on the SIMT units, TFLOP/s | 78.8 (126) | 51.2 (67) | 0.65 |
| DRAM copy, GB/s read + write (nominal) | 1,461 (1,792) | 3,032 (3,350) | 2.08 |
| copy within L2, GB/s read + write | 5,459 | 4,387 | 0.80 |
| pinned PCIe copy, GB/s to / from the GPU | 57.6 / 57.3 | 55.5 / 55.4 | |
| replay of a one-kernel CUDA graph, µs | 2.19 | 3.43 | |

Both ran torch 2.11.0+cu128 (CUDA 12.8) and Triton 3.6.0.

## 3. S55 alone

### 3.1 Time per shot

| batch | RTX PRO 6000, resident (µs) | RTX PRO 6000, host to host | H100, resident | H100, host to host |
|---:|---:|---:|---:|---:|
| 256 | 4.19 | 4.27 | 5.83 | 5.92 |
| 1,024 | 4.14 | 4.17 | 5.66 | 5.70 |
| 4,096 | 4.09 | 4.11 | 5.61 | 5.63 |
| 16,384 | 4.12 | 4.13 | 5.60 | 5.61 |

The SM clock stayed at its maximum on both cards (2,392-2,422 MHz on the RTX PRO 6000, 1,980 MHz on the H100). At 256
shots each kernel already launches 4,096 or more programs. From 256 to 4,096 the time per shot falls by 2 % on the RTX
PRO 6000 and 4 % on the H100, and above 4,096 it does not fall. PDL changes nothing at these batch sizes: the graph's
kernel times equal the eager ones within about 1 %.

### 3.2 Per kernel

S55 is five kernels in seven launches per batch. Each program handles one tile of one shot: 16 nodes in the stem and
the blocks (16 programs per shot), 16 of the 512 read-out queries in the pool (32 programs per shot, 432 queries used),
and 128 of the 4,096 classes in the vote (32 programs per shot; BB72 has 12 logical qubits, so 4,096 classes).

| kernel | what each program does | RTX PRO 6000, µs/shot | share | H100, µs/shot | share | H100 / RTX PRO 6000 |
|---|---|---:|---:|---:|---:|---:|
| `_k_stem` | multiplies the gathered detector bits of its 16 nodes by a 256 × 128 weight; LayerNorm, down-projection to 32 channels | 0.25 | 6 % | 0.27 | 5 % | 1.08 |
| `_k_block` (3 launches) | 80-tap convolution over the neighbours' 32-channel activations (a 16 × 2,560 by 2,560 × 32 product), SiLU, up-projection to 128, residual add; then the next block's LayerNorm and down-projection, or in the last block LayerNorm, gate and value into U | 2.22 | 55 % | 3.64 | 66 % | 1.64 |
| `_k_pool` | for each of 4 heads, a fixed pooling of the shot's 256 rows of U onto its 16 queries; output projection, LayerNorm, MLP to one logit; the parity of the query's detectors | 1.07 | 26 % | 1.11 | 20 % | 1.04 |
| `_k_vote` | for each of its 128 classes, a 512-term dot product of the psi table with the pool's outputs, plus a bias; the partial top 2 | 0.51 | 13 % | 0.52 | 9 % | 1.02 |
| `_k_top2` | the final top 2: 12 decision bits and the margin | 0.00 | 0 % | 0.00 | 0 % | |
| all | | 4.05 | | 5.54 | | 1.37 |

These are eager launches without PDL at batch 4,096. The sums are 1 % below the CUDA-event times (4.09 and 5.61 µs).

Nsight Compute on the RTX PRO 6000, at batch 4,096:

| launch | duration (ms, base clock) | measured L2 traffic (MB/shot) | SM % | L1 % | L2 % | tensor pipe busy % | occupancy achieved / possible % | top stalls (cycles per instruction) |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `_k_stem` | 1.49 | 1.70 | 51 | 52 | 69 | 15.0 | 33 / 33 | long scoreboard 7.5, barrier 2.3 |
| `_k_block` 0 | 3.87 | 3.49 | 46 | 86 | 54 | 13.9 | 33 / 33 | long scoreboard 7.9, MIO throttle 5.9 |
| `_k_block` 1 | 3.91 | 3.49 | 46 | 85 | 54 | 13.7 | 33 / 33 | long scoreboard 8.1, MIO throttle 5.9 |
| `_k_block` 2 | 4.28 | 4.38 | 45 | 84 | 62 | 16.5 | 33 / 33 | long scoreboard 9.7, MIO throttle 4.9 |
| `_k_pool` | 5.61 | 7.02 | 38 | 47 | 75 | 17.3 | 25 / 25 | long scoreboard 9.1, wait 2.0 |
| `_k_vote` | 2.74 | 4.19 | 42 | 57 | 92 | 0 | 33 / 33 | long scoreboard 4.1, barrier 3.7 |
| `_k_top2` | 0.005 | 0.00 | 14 | 39 | 9 | 0 | 43 / 50 | |

Every kernel waits mostly on memory: "long scoreboard" is a warp waiting for a load through L1. The three blocks run the
L1/shared-memory pipe at 84-86 % of its peak, and "MIO throttle" (the queue of shared-memory instructions is full) is
their second stall. They are bound inside the SM. The pool and the vote run L2 at 75 % and 92 % of its peak and are
bound by L2. No kernel keeps the tensor pipe busy more than 17 % of the time. The measured L2 traffic is 24.3 MB per
shot, 87 % of the 28.0 MB the programs request; the rest hits in L1.

### 3.3 Work per shot

| kernel | programs per shot | tensor-core MFLOP | other MFLOP | L2-requested MB: weights and tables | L2-requested MB: the shot's own data | DRAM-compulsory MB |
|---|---:|---:|---:|---:|---:|---:|
| `_k_stem` | 16 | 18.9 | 0.38 | 1.59 | 0.21 | 0.15 |
| `_k_block` 0 and 1, each | 16 | 46.1 | 0.46 | 2.99 | 1.59 | 0.29 |
| `_k_block` 2 | 16 | 60.8 | 0.61 | 3.93 | 1.51 | 0.21 |
| `_k_pool` | 32 | 83.9 | 1.84 | 5.08 | 2.23 | 0.07 |
| `_k_vote` | 32 | 0 | 4.29 | 4.20 | 0.07 | 0.00 |
| `_k_top2` | 1 | 0 | 0.00 | 0.00 | 0.00 | 0.00 |
| all | | 255.9 (236.6 useful) | 8.0 | 20.8 | 7.2 | 1.02 |

The weights and tables (20.8 MB per shot) are the same for every shot. Each of a shot's 16 block programs reads its
block's tap weights (80 × 32 × 32 in FP16, 160 KiB), and each of its 32 vote programs reads a 128-row slice of the psi
table (4,096 × 512 in FP16, 4 MiB in all). The shot's own data (7.2 MB) is mostly re-reads. Each block program gathers
80 taps × 32 channels for its 16 nodes from the shot's 16 KB of activations, 1.3 MB per block and shot. Each of the
pool's 32 programs reads the shot's whole U, 2.1 MB per shot. Only 1.0 MB per shot must cross DRAM: the FP32 residual
stream (252 × 128 × 4 bytes = 129 KB), read and written by every layer, and the smaller activations. The "useful" FLOPs
exclude the padding of 252 nodes to 256 rows, 432 queries to 512 and 78 taps to 80.

### 3.4 Against the ceilings

| kernel | RTX PRO 6000: tensor TFLOP/s (% of FP16 GEMM) | RTX PRO 6000: requested L2 TB/s | H100: tensor TFLOP/s (% of FP16 GEMM) | H100: requested L2 TB/s (% of L2 copy) |
|---|---:|---:|---:|---:|
| `_k_stem` | 75.5 (25 %) | 7.2 | 69.9 (9 %) | 6.7 (152 %) |
| `_k_block` ×3 | 69.0 (23 %) | 6.6 | 42.1 (6 %) | 4.0 (91 %) |
| `_k_pool` | 78.4 (26 %) | 6.8 | 75.6 (10 %) | 6.6 (150 %) |
| `_k_vote` | 8.4 on the SIMT units (11 % of FP32 GEMM) | 8.4 | 8.3 (16 % of FP32 GEMM) | 8.2 (187 %) |
| S55 | 62.5 (20 %) | 6.8 | 45.6 (6 %) | 5.0 (114 %) |

S55's DRAM traffic runs at 249 GB/s on the RTX PRO 6000 (17 % of the copy rate) and 182 GB/s on the H100 (6 %). DRAM
and the FP16 tensor rate alone would allow 0.84 µs per shot on the RTX PRO 6000 (the tensor rate binds) and 0.34 µs on
the H100. S55 is far from that because its real traffic is on chip. `summary.md`'s roofline bound, 5.13 and 6.38 µs
per shot, divides the requested bytes by the L2 copy rate. S55 runs faster than that bound (0.80 and 0.88 of it),
because part of the requests hit in L1 and because the copy understates what L2 delivers to reads.

The copy within L2 understates what L2 delivers to reads. On the RTX PRO 6000 the vote moves 8.2 TB/s of measured L2
traffic at 92 % of L2's peak, 1.5 times the copy rate. Against that, S55's 10.5 tensor FLOP per byte of measured L2
traffic allow 86 TFLOP/s, and it reaches 62.5, 72 %. On the H100 the pool and the vote request 1.5-1.9 times the L2
copy rate, so many of their re-reads must hit in L1 (256 KiB per SM on the H100 against 128 KiB on the RTX PRO 6000).
The H100 has no counters to confirm it.

## 4. C14 alone

### 4.1 Time per shot

| batch | RTX PRO 6000, resident (µs) | RTX PRO 6000, host to host | H100, resident | H100, host to host |
|---:|---:|---:|---:|---:|
| 4,096 | 71.59 | 71.63 | 43.36 | 43.41 |
| 8,192 | 71.44 | 71.51 | 43.24 | 43.25 |

That is 11.9 and 7.2 µs per round at 4,096. Batch 8,192 is 0.2-0.3 % faster and needs twice the memory (31.7 against
15.9 GiB). On the RTX PRO 6000 the SM clock stayed at 2,392-2,400 MHz (520-567 W of 600). On the H100 it ran at
1,500-1,860 MHz, median 1,725, at 691-702 W of its 700 W limit: C14 is power-capped there.

### 4.2 By kernel

| kernels | RTX PRO 6000, µs/shot | share | H100, µs/shot | share | H100 speed-up |
|---|---:|---:|---:|---:|---:|
| `_gmm`, the bipartite convolution (Triton gather-matmul, 56 launches) | 19.66 | 27 % | 16.44 | 39 % | 1.20 |
| Inductor's fused batch-norm and SiLU kernels with the residual update, the concatenations and the casts | 36.44 | 51 % | 18.70 | 45 % | 1.95 |
| cuBLAS GEMMs: the down- and up-projections and the read-out | 15.59 | 22 % | 6.82 | 16 % | 2.29 |
| other | 0.03 | 0 % | 0.04 | 0 % | |
| sum of kernel time | 71.72 | | 42.01 | | 1.71 |

`summary.md` counts the batch-norm and SiLU kernel fused after `_gmm` (3.68 µs on the RTX PRO 6000, 1.91 on the H100) in
the convolution's category, which is why it gives the convolution 33 % and 44 %. The H100's profiled kernel sum is 3 %
below its event time; its clock moves under this load.

Every matrix product runs on BF16 tensor cores. `_gmm` was autotuned to 64-row blocks: on the RTX PRO 6000 with 8 warps
and mma.sync (m16n8k16), on the H100 with 4 warps, 4 stages and wgmma (m64n128k16). cuBLAS picks CUTLASS sm80 kernels
on the RTX PRO 6000 and its nvjet kernels on the H100.

Nsight Compute on the RTX PRO 6000 (C14's first 120 launches at batch 4,096). "Memory %" is the busiest of DRAM, L2 and
L1; where L2 and L1 are far lower, it is DRAM.

| kernels (launches profiled) | SM % | memory % | L2 % | L1 % |
|---|---:|---:|---:|---:|
| fused batch-norm, SiLU, residual (10) | 30 | 90 | 34 | 27 |
| concatenation (5) | 23 | 92 | 27 | 16 |
| batch-norm, concatenation and SiLU (4) | 48 | 91 | 26 | 34 |
| CUTLASS BF16 GEMMs (18) | 58 | 77 | 55 | 38 |
| `_gmm` (18) | 64 | 81 | 81 | 69 |

`_gmm`'s top stalls are long scoreboard (9.8 cycles per instruction, the gathered operands) and math pipe throttle (5.2).

### 4.3 Work per shot and the ceilings

C14 executes 7.71 GFLOP per shot on BF16 tensor cores: 5.55 in `_gmm` (72 %), 1.85 in the projections (24 %) and 0.31
in the read-out (4 %). The bytes model gives 90.1 MB of DRAM traffic per shot: 6.19 MB per layer, plus 3.4 MB for the
embedding and the read-out. Per layer, 3.10 MB is the residual update with batch-norm, SiLU and cast (the residual
stream is FP32), 1.29 MB the two projections, 0.77 MB the convolution, 0.52 MB the two concatenations and 0.52 MB two
more batch-norm and SiLU passes. At 85.6 FLOP per DRAM byte C14 sits below both cards' ridge points (267 and 256 FLOP
per byte), so DRAM bounds it.

| | RTX PRO 6000 | H100 |
|---|---:|---:|
| TFLOP/s (% of the BF16 GEMM rate) | 107.6 (28 %) | 177.9 (23 %) |
| DRAM GB/s by the model (% of the copy rate) | 1,258 (86 %) | 2,078 (69 %) |
| time at the DRAM copy rate, µs/shot | 61.7 | 29.7 |
| measured / that bound | 1.16 | 1.46 |
| `_gmm` alone, TFLOP/s (% of the BF16 GEMM rate) | 282 (72 %) | 338 (43 %) |

The projections are thin GEMMs (102 FLOP per byte) and run close to the DRAM copy rate. Two cuBLAS kernels are launched
28-29 times per forward and the others once each (the read-out). The first two take 13.5 µs per shot on the RTX PRO
6000 and 6.3 µs on the H100. Taken as the projections (18.1 MB per shot), that is 1.3 and 2.9 TB/s, 92 % and 94 % of
the copy rate.

## 5. Why S55 is slower on the H100 and C14 faster

The two networks sit at opposite ends of the roofline. Per shot, S55 executes 256 MFLOP and needs 1.0 MB of DRAM
traffic, while C14 executes 7.7 GFLOP and moves about 90 MB. What S55 needs in quantity is on-chip bandwidth: 24-28 MB
per shot through L2 and more through L1 and shared memory, because each of its programs handles 16 rows of one shot and
re-reads its layer's weights to do so.

The two cards trade these resources (section 2). The H100 has 2.1 times the DRAM bandwidth and 2.0 times the BF16 rate,
which C14 needs. The RTX PRO 6000 has 1.75 times the SM count × clock, 1.24 times the L2 copy rate and 1.5 times the
FP32 rate on the SIMT units, which S55 needs.

Kernel by kernel, S55's slowdown follows the resource that binds it. The residual blocks, which the RTX PRO 6000's
counters show bound by the L1/shared-memory pipe inside the SM, take 1.64 times longer on the H100, close to the 1.75
ratio of SM count × clock. They are 66 % of S55's time there. The pool, the vote and the stem, which are L2-bound on
the RTX PRO 6000, take only 1.02-1.08 times longer. The H100's L2 copies more slowly, but its larger L1 probably absorbs
more of their table re-reads (section 3.4). S55 cannot use the H100's Hopper tensor path either: its 16-row tiles
compile to the warp-level mma.sync on both cards, not to wgmma.

C14 runs 1.65 times faster on the H100 (71.6 against 43.4 µs per shot). Its DRAM-bound parts follow the bandwidth
ratio: the fused pointwise kernels run 1.95 times faster and the cuBLAS projections 2.29 times. Its convolution `_gmm`
gains only 1.20 times. On the RTX PRO 6000 `_gmm` already runs at 72 % of the BF16 GEMM rate, while on the H100 it
reaches 43 %. No counters say why; the gathered operands and the power-capped clock (1,500-1,860 MHz while C14 runs)
are the likely reasons.

For the routed pipeline, the RTX PRO 6000 is cheaper per shot below q1 = (5.61 − 4.10) / (71.46 − 43.13) = 5.3 % and
the H100 above it (section 6).

## 6. S55 → C14 batched

### 6.1 By routed fraction

| q1 | RTX PRO 6000: µs/shot | aS + q1 · aB | excess | S's share of the time | H100: µs/shot | aS + q1 · aB | excess | S's share of the time |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 4.15 | 4.10 | 0.05 | 100 % | 5.68 | 5.61 | 0.07 | 100 % |
| 1 % | 5.33 | 4.81 | 0.52 | 77 % | 6.39 | 6.04 | 0.35 | 89 % |
| 2 % | 5.92 | 5.53 | 0.39 | 70 % | 6.72 | 6.47 | 0.25 | 84 % |
| 5 % | 8.27 | 7.67 | 0.60 | 50 % | 8.11 | 7.77 | 0.34 | 70 % |
| 10 % | 11.80 | 11.25 | 0.55 | 35 % | 10.17 | 9.92 | 0.25 | 56 % |
| 40 % | 32.96 | 32.68 | 0.28 | 13 % | 23.02 | 22.86 | 0.16 | 26 % |

The times are wall time per shot over 500,000 shots. aS and aB were measured alone in the same process: 4.100 and
71.462 µs on the RTX PRO 6000, 5.610 and 43.130 µs on the H100.

### 6.2 The excess is the padded last batch

| q1 | routed shots | C14 batches (full + last) | real rows in the last batch | padded rows | their cost, RTX PRO 6000 (µs/shot) | their cost, H100 |
|---:|---:|---|---:|---:|---:|---:|
| 1 % | 5,000 | 1 + 1 | 904 | 3,192 | 0.46 | 0.28 |
| 2 % | 10,000 | 2 + 1 | 1,808 | 2,288 | 0.33 | 0.20 |
| 5 % | 25,000 | 6 + 1 | 424 | 3,672 | 0.52 | 0.32 |
| 10 % | 50,000 | 12 + 1 | 848 | 3,248 | 0.46 | 0.28 |
| 40 % | 200,000 | 48 + 1 | 3,392 | 704 | 0.10 | 0.06 |

S's own last batch adds 3,808 padded rows: 0.03 µs per shot on the RTX PRO 6000, 0.04 on the H100. With both counted,
0.03-0.05 µs per shot of the excess remains on the RTX PRO 6000 at q1 up to 10 %, and 0.14 at 40 %, where C14's 49
batches ran 0.3 % slower than alone. On the H100 the remainder is within ±0.07 µs per shot, because C14's speed there
moves by a few per cent with the clock. The padded rows cost the same in a longer run, but per shot they shrink in
proportion to its length.

### 6.3 The timeline

At q1 = 5 %, per shot, the copies in and out take 0.01-0.02 µs, the routing and the queue 0.01 µs, and the GPU's idle
gaps 0.01-0.02 µs, on either card. The host issues each S batch in about 0.01 µs per shot and then waits on the GPU. The
GPU's idle gaps come before S's input copies (123 per run, mean 10 µs on the RTX PRO 6000 and 22 µs on the H100) and
before queue operations (130 per run, mean 32 and 52 µs). Together they stay below 0.02 µs per shot, so the host loop
has nothing worth optimizing.

S's and C14's graphs replayed on two streams instead of one, with the input resident, at the S:B replay ratios of
q1 = 1, 5, 10 and 40 % (100:1, 20:1, 10:1, 5:2), took the same time: 1.00 times on the RTX PRO 6000 and 0.99-1.01 times
on the H100, also with S's stream at high priority. Each kernel fills the GPU, so there is nothing to overlap.

### 6.4 At the claim points

Phase C's locked S → B configurations at the claim points (Amendment 1) are S55 → C14 in `uniform` and S55′ → C14′ in
`depspam_fm`, with q1 = 1 % at p = 0.2 % and q1 = 5 % at 0.3 %. S55′ and C14′ have S55's and C14's architectures, and
their times match to 0.1 %.

| µs per round | RTX PRO 6000 (C7, job CJ39) | H100 (second pod) |
|---|---:|---:|
| `uniform` 0.2 %: S55 → C14, q1 1 % | 0.888 | 1.067 |
| `uniform` 0.3 %: S55 → C14, q1 5 % | 1.376 | 1.355 |
| `depspam_fm` 0.2 %: S55′ → C14′, q1 1 % | 0.887 | 1.067 |
| `depspam_fm` 0.3 %: S55′ → C14′, q1 5 % | 1.375 | 1.354 |
| Cascade alone (C14, C14′) | 11.98 | 7.28-7.30 |

Each value is the slowest of three fresh processes over 500,000 shots of kind 12. The RTX PRO 6000's rows are C7's
under Amendment 1, which finished at 05:37 MDT on 11 October. All its S → B processes kept their SM clock at or above
98.7 % of the maximum; 3 of its 12 Cascade processes fell below the 97 % rule. C8 has not judged any of them yet. The
H100's rows are from `H100_TIMING.md`, where most are marked because the H100 lowers its clock under these loads. The
profile's own pipeline gives 0.889 and 1.379 µs per round on the RTX PRO 6000 and 1.064 and 1.351 on the H100 at the
same q1, within 0.3 % of these.

Where a round's time goes at the claim points, from the profile's pipeline at the same q1 (sections 6.1-6.2):

| µs per round | RTX PRO 6000, 0.2 % | RTX PRO 6000, 0.3 % | H100, 0.2 % | H100, 0.3 % |
|---|---:|---:|---:|---:|
| S55, with its padded last batch | 0.689 (77 %) | 0.689 (50 %) | 0.943 (89 %) | 0.943 (70 %) |
| C14 on the routed shots | 0.119 (13 %) | 0.596 (43 %) | 0.072 (7 %) | 0.359 (27 %) |
| C14 on the padded rows of its last batch | 0.076 (9 %) | 0.087 (6 %) | 0.046 (4 %) | 0.053 (4 %) |
| copies, routing, queue, idle | 0.005 | 0.005 | 0.007 | 0.008 |
| measured total | 0.889 | 1.379 | 1.064 | 1.351 |

The shares are of the measured total. C14's time is split between the routed shots and the padded rows by their row
counts; on the H100, C14 ran 3 % faster in the pipeline than alone, so its two rows add up to slightly more than it took.

Even with S55 free, S → B would cost C14's routed share: 0.12 and 0.60 µs per round on the RTX PRO 6000, 0.07 and
0.36 on the H100.

## 7. The second H100 pod's headline findings against the files

| the second pod's statement | what the files show | verdict |
|---|---|---|
| S55 at batch 4,096: 5.61 µs/shot on the H100, 4.09 on the RTX PRO 6000 | 5.61 and 4.09 | confirmed |
| S55 is limited by L2 traffic | On the RTX PRO 6000 the pool and the vote are L2-bound (75 and 92 % of L2's peak, 39 % of S55's time), but the residual blocks (55 % of the time, 66 % on the H100) are bound by the L1/shared-memory pipe (84-86 %), with L2 at 54-62 % | partly: on-chip traffic, L1 and L2 |
| tensor cores at about 6 % of peak | 6.1 % of the FP16 GEMM rate on the H100; 20 % on the RTX PRO 6000, whose tensor pipe is busy 14-17 % of the cycles | the H100's value |
| C14: 43.4 µs/shot on the H100, 71.6 on the RTX PRO 6000 | 43.36 and 71.59 | confirmed |
| C14 at 69 % of DRAM bandwidth and 23 % of BF16 peak | 68.5 % and 22.9 % on the H100; 86 % and 28 % on the RTX PRO 6000 | the H100's values |
| `_gmm` 44 % of C14's time, fused batch-norm and SiLU 40 % | `summary.md`'s categories on the H100. By kernel, `_gmm` is 39 % and the fused pointwise kernels 45 % (27 % and 51 % on the RTX PRO 6000) | the H100's categories |
| a routed pipeline costs aS + q1 · aB plus at most 0.35 µs, mostly the padded last batch | 0.07-0.35 µs on the H100; 0.05-0.60 µs on the RTX PRO 6000, where the padded rows alone cost 0.52 µs at q1 = 5 % | holds on the H100 only |
| two CUDA streams gain nothing | 0.99-1.01 times and 1.00 times | confirmed |
| at q1 = 5 % the pipeline costs about the same on both cards, 8.11 against 8.27 µs | 8.105 and 8.271 µs per shot; equal cost at q1 = 5.3 % | confirmed |

[Sections 8 and 9, the ranking of the levers and the plan for the optimization stage with its targets, are not part of
this release; M2.md describes the two levers that were built and the targets it measured against.]

## 10. Files and reruns

- RTX PRO 6000: `<cluster-scratch>/cascade_batchopt/out/CJ45/` on Alpine: `summary.md` (its tables),
  `env.json`, `work.json`, `s55.json`, `c14.json`, `pipe.json`, `pipe_nsys.json`, `ncu_s55.*` and `ncu_c14.*` (JSON,
  raw CSV and report), `nsys_pipe.*`, the torch.profiler traces, and the job log
  `<cluster-scratch>/cascade_batchopt/logs/w6prof_CJ45.out`.
- H100: `~/work/cascade_h100b/results/out/prof/` on the Linux workstation and
  `<cluster-scratch>/cascade_h100b/out/prof/` on Alpine, the same files without Nsight Compute. The H100's screen of the
  fused paths against the RTX PRO 6000 reads is in `~/work/cascade_h100b/results/screen/amend1/`.
- Phase C's timings: C7's rows in `<cluster-scratch>/cascade_w4/out/speed/rows.jsonl` (session CJ39, read only), and the
  second H100 pod's rows in `H100_TIMING.md` on `ising-cascade-h100t2`.
- `prof_batched.py report --out DIR` rebuilds a `summary.md` from the JSON files without a GPU.
- To profile on Alpine: `cascade_compare/batchopt/deploy.sh` from a checkout of this branch, then submit `prof.sbatch`
  from the deployed tree (`prof_run.sh`'s steps: shots, env, work, s55, c14, pipe, report, ncu, nsys, report; about 7
  minutes on a full RTX PRO 6000).

[The recipe for profiling on an H100 pod (`pod_deploy.sh`, `pod_run.sh`) is an instruction, not a record, and is not
part of this release.]
