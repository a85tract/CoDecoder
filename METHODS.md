# How we measured accuracy and speed

This document states how the paper's accuracy and speed numbers were measured: the shots, the decoders, the statistics,
what was timed and on which hardware. It summarises; the registration documents it names hold the full rules and govern
where the two differ. Paths are relative to the repository root. The accuracy numbers come from SURF-v2 and RETEST
(section 1), and every speed number from SPEED-RT, the registered speed re-test (section 2), which supersedes the
earlier timing sessions. The exceptions are the times of three sessions of their own, which section 2 names:
D9-SHALLOW's, CASCADE-TRT-REG's and the joint decision's (not registered).

- A shot is one run of the error-correction circuit. A decoder reads its detector bits and predicts the flip of each
  logical observable.
- The composition is CoDecoder with a strong decoder. The network decodes every shot and gives a confidence; a shot whose
  confidence is below the threshold τ goes to the strong decoder, whose answer replaces the network's. τ is set so that
  a fixed fraction q of calibration shots is routed.
- A test is registered when its networks, thresholds, test-seed rule and analysis were committed and pushed to GitHub
  before any test shot existed.

| paper numbers | accuracy measured by | speed measured by | full rules |
|---|---|---|---|
| surface code, d = 7 to 17 | SURF-v2 | SPEED-RT stage 2 | `surface_v2/surf_v2/REGISTRATION.md`; `speed_rt/STAGE2_common.md`, `speed_rt/STAGE2_surface.md` |
| BB72 and the gross code (Table IV) | RETEST, re-running the original BB tests on fresh shots | SPEED-RT stage 2 | `RETEST.md` section 3.1; `bb/BB.md`; `speed_rt/STAGE2_bb.md` |
| Cascade on BB72 and on the surface code at d = 13 | RETEST, re-running the original Cascade tests | SPEED-RT stage 1, on two cards | `RETEST.md` sections 3.2 and 3.3; `cascade/bb72/CASCADE.md`; `cascade/d13/SURFACE_PLAN.md`; `speed_rt/SPEED_RT.md` section 1 |
| output heads (Table II) | RETEST | no speed number | `RETEST.md` section 3.4 |
| gap switching, our Decoder Switching | SURF-v2's secondary comparison | SPEED-RT stage 2, descriptive | `surface_v2/surf_v2/GAPSW_REGISTRATION.md`; `speed_rt/STAGE2_gapsw.md` |

## 1. Accuracy

### 1.1 Common to every test

- Shots. Stim 1.16.0 samples the detector bits and the true logical flips from a fixed circuit with circuit-level
  noise, in chunks of 10,000 shots, one seed per chunk. A manifest lists every chunk's seed and sha256, so a chunk can
  be rebuilt and compared bit for bit. Calibration or development shots, which set the thresholds, use other seeds.
  Every decoder read each test set once, and the frozen analysis ran once on those reads.
- Seeds. T is the time of GitHub's earliest activity record of the push of the freeze commit. The beacon is the first
  drand mainnet round (one every 30 s) released at least 600 s after T, accepted when at least two of four public
  endpoints agree. With R = SHA-256 of the round's signature, chunk j of a set gets

      seed = 2^62 + (first 8 bytes, big-endian, of SHA-256(R ‖ TAG ‖ j as 2 bytes, big-endian)) mod 2^62

  with one fixed TAG per set, such as `convlines retest bb72 p0.003` followed by a newline. RETEST's freeze was pushed
  at 2026-10-13 02:34:48 UTC (round 6547616), SURF-v2's at 11:49:59 UTC (round 6548726). `README.md` ("Checking a
  registration") checks each step.
- Failure. A decoder fails on a shot when its predicted logical flip differs from the true one; on BB72 and the gross
  code, when any of the 12 logical observables is wrong (a block failure). The composition takes the strong decoder's answer
  on a routed shot and the network's otherwise.
- Routing. The confidence is |c| for a surface network's output c, and on the BB codes the margin between the two best
  class scores. τ is the q-quantile of the confidence on calibration shots. q and τ (at full float precision) were
  frozen before any test shot existed.
- Statistics. All decoders read the same shots, so each comparison is paired, and only the discordant shots (where
  exactly one of the two fails) carry evidence. The test is the exact two-sided McNemar test, a binomial test of the
  discordant split against 1/2. The interval is Newcombe's interval for the difference of two paired failure rates,
  which combines each rate's Wilson limits with the paired correlation (method 8 of Newcombe (1998) in SURF-v2, method
  10 in the BB and Cascade tests). "Better" needs a rejection at the comparison's level and the whole interval below
  zero.

### 1.2 Surface code (SURF-v2)

- Circuits. Rotated surface code, X-basis memory, d rounds, d = 7, 9, 11, 13, 15 and 17, built by NVIDIA
  Ising-Decoding's `MemoryCircuit` (commit 3ce4379) at p = 0.6 %: depolarizing error p on idle qubits and after one- and
  two-qubit gates, and measurement and reset flips with probability 2p/3 (`surface_v2/circuits/`, checked by sha256).
- Shots. Calibration set `sv2cal`: 200,000 per distance (seeds 81,000,000 + 10,000 d + j). Test set `sv2test`:
  2,000,000 at d = 7 and 1,000,000 at each other distance, from the beacon.
- Decoders, on the same test shots: the new composition (the network retrained on the observable's flip alone, with correlated
  PM as its strong decoder), the network alone, and eight baselines: plain and correlated PyMatching (PM) 2.4.0;
  NVIDIA's pre-decoders Ising M1 (Fast, receptive field 9) and M4 (Accurate, receptive field 13), run by NVIDIA's own
  PyTorch module (fp32, TF32 as NVIDIA's runner sets it) and followed by plain or correlated PM; and BP-OSD from
  cudaq-qec 0.8.0 (OSD-CS order 10, with min-LLR initialisation and 10 BP iterations, or with 60 iterations), which
  read every test shot at d = 7 to 13 and the first 200,000 at d = 15 and 17. The networks were read by the deployed
  engine at batch one, the code that is timed, and the earlier tests' deployed networks were read beside them.
- Routing. On the calibration shots, the rule considers each q of a grid (0 to 16 %, up to 30 % at d = 17) whose
  projected composite time is within the distance's speed bar (15.16 µs at d = 7 up to 322.23 µs at d = 17), and takes
  the smallest with at most M − 2√(2M) calibration failures, where M is the earlier tests' planning count (3,401 at
  d = 7 down to 554 at d = 17); otherwise a fixed fallback. The speed bars and the projected times are inputs of the
  frozen rule, from the earlier timing sessions (2.8). Frozen: q = 2 % at d = 7, q = 0 at d = 9, 11 and 13 (the
  network decides every shot) and q = 1 % at d = 15 and 17 (`surface_v2/surf_v2/freeze.json`).
- Statistics. Accuracy against each baseline: McNemar and Newcombe's interval at 97.5 %; "better" when p < 0.025 and the
  interval lies below zero. Speed against each baseline: the composition's composite time strictly below the baseline's
  registered time, a measured inequality without a statistical test. A distance's claim needs all 16 comparisons to
  hold, so no multiplicity correction is needed. New against deployed networks, alone and as compositions: McNemar with
  Holm's correction over the 12 tests of all six distances at 0.05.
- SURF-v2 judged its speed comparisons on the earlier registered times (2.10). The paper's speed numbers are SPEED-RT's
  (section 2), under which plain PM is faster than the composition at d = 7 and 9: at those two distances the composition is more
  accurate than every baseline but not faster than all of them.

The earlier surface tests (studies 2, 3 and 4 and R17, on test sets T2, T3, T5 and T6) used the same circuits, code and
statistics with the deployed networks; `results/surface/SURFACE.md` describes them.

### 1.3 BB72 and the gross code (RETEST)

- Circuits. Relay-BP's X-memory circuits (its repository at revision d185194) under uniform circuit-level noise: BB72
  [[72,12,6]], 6 rounds, at p = 0.3 % and 0.2 %; the gross code [[144,12,12]], 12 rounds, at p = 0.4 %.
- Shots. 200,000 test shots at p = 0.3 %, 300,000 at 0.2 % and 150,000 on the gross code, the sizes of the original
  tests (`retest/run/bb/manifest_bb72.csv`, `manifest_gross.csv`).
- Decoders. Our networks (two on BB72, one on the gross code) at batch one through their frozen engines. Relay-BP 0.2.2
  (α = 1.0, γ0 = 0.15, 80 pre-iterations, 300 sets of 60 iterations, γ in [−0.2263, 0.6216], stop after 5 solutions)
  on the X-detector sector ("sector Relay-BP") or on all detectors ("full-DEM Relay-BP"); BP-OSD from ldpc 2.4.1 on the
  CPU and from cudaq-qec 0.8.0 on the GPU, both with min-sum scale 1.0, 80 iterations and OSD-CS order 60.
- Routing, frozen on calibration shots: on BB72 the margin threshold 2.979396343231201 (q = 0.25 %) at p = 0.2 % and
  3.6329734325408936 (q = 2 %) at 0.3 %, with full-DEM or sector Relay-BP as the fallback; on the gross code
  2.2055678367614746 (q = 1 %) to sector and 6.272581100463867 (q = 2 %) to full-DEM Relay-BP. Two further BB72 cells
  test the network alone.
- Statistics. Each comparison: McNemar at p < 0.05/3 and Newcombe's interval at 1 − 0.05/3 entirely below zero. A cell
  (code, p and fallback) holds when every comparison, against sector Relay-BP, both BP-OSDs and, in the full-DEM cells,
  full-DEM Relay-BP, is "better" and its speed condition holds. Each cell spends 0.05/3, so the union bound over the
  nine cells of the BB tests is 0.15. RETEST judged the speed conditions on estimates from the earlier sessions (2.10).
  SPEED-RT re-timed them (rows B1 to B3 of `speed_rt/s2/bb/results/table.json`), and every one holds except the gross
  code's balanced mode (G-4R) against CUDA-Q's BP-OSD: 20.35 ms against 13.70 ms.
- RETEST changed only the shots, through shims that swap in the new seeds (`retest/shim/`). BB72's networks were read
  on an RTX PRO 6000 (the original test used an H100); on the original shots the two cards give identical decisions
  and margins. The one
  recorded deviation is the job layout of the gross-code CPU reads (`RETEST.md`).

### 1.4 Cascade (RETEST)

- BB72, under a fitted noise model (`depspam_fm`: the uniform noise plus final-measurement flips and depolarizing error
  p after resets and before measurements), chosen to reproduce the failure rates that Cascade's paper reports for
  Tesseract, BP-OSD and Relay-BP (`cascade/bb72/noise_fit/CASCADE_CAL.md`). Our Cascade has 14 blocks of width 512
  (19,687,937 parameters). In the composition a six-block CoDecoder network (841,053 parameters) routes q = 2 % of shots at
  p = 0.2 % and 10 % at 0.3 % to it. RETEST drew 11,000,000 and 17,000,000 test shots. Four hypotheses (two p, eager
  and compiled paths) form one Holm family at 0.05; a compiled comparison needs its single-shot and its batched read
  both to be better.
- Surface code at d = 13, the circuit of 1.2 at p = 0.6 % and 0.35 %. Our Cascade has 2,531,841 parameters. In the composition
  the deployed d = 13 network routes q = 10 % at p = 0.6 % and 1 % at 0.35 % to it. RETEST drew 1,000,000 and
  30,000,000 test shots. The 40 hypotheses (two p, two modes, eight superiority and two non-inferiority tests each)
  form one Holm family at 0.05; non-inferiority is one-sided, with a margin of 10 % of Cascade's failure rate on
  development shots. At p = 0.6 % the composition was non-inferior to Cascade but not significantly better.
- Both read every decoder on one NVIDIA RTX PRO 6000 Blackwell Server Edition; the code refuses other cards. SPEED-RT's
  launchers also admit the H100, for timing only (2.2).

## 2. Speed (SPEED-RT)

Every speed number of the paper comes from SPEED-RT, the registered speed re-test, except the few values listed in 2.8
and the times of three sessions of their own. D9-SHALLOW timed the composition at d = 9 in its registered session
(`d9_shallow/`). Cascade on TensorRT (Table X) was timed by CASCADE-TRT-REG, a registered speed re-test of its own
(`cascade_trt_reg/`). The joint decision's times, in Table I's rows "CoDecoder + Cascade, joint decision", come from a
session that was not registered and follows stage 1's rules on fresh timing shots (`joint/timing/`).
SPEED-RT timed again every speed number that this release can run, on fresh timing shots drawn after its rules were
public, and it registered in advance that the paper reports its numbers in place of the earlier ones, whatever they
are (`speed_rt/SPEED_RT.md`). The registrations govern where this summary differs from them: stage 1 is section 1 of
`speed_rt/SPEED_RT.md`, and stage 2 is `speed_rt/STAGE2_common.md` (the rules its three families share, cited as C1 to
C9) with `STAGE2_surface.md`, `STAGE2_bb.md` and `STAGE2_gapsw.md`. Section 2.10 keeps the earlier timing sessions as
history; SPEED-RT supersedes them.

### 2.1 Stages and the order of events

| stage | what it timed | freeze F | beacon round | beacon commit B | analysis | outputs |
|---|---|---|---|---|---|---|
| 1 | the Cascade comparisons, on an RTX PRO 6000 and on an H100 | `73ca9f9`, pushed 2026-10-13T21:48:15Z | 6549923, released 21:58:30Z | `6609d14`, pushed 22:01:23Z | `56e2023`; correction `2ce2e49` | `speed_rt/cascade/results/` |
| 2 | the surface comparisons with Ising, BB72 and the gross code, and Decoder Switching | `7988e04`, pushed 2026-10-14T04:00:56Z | 6550668, released 04:11:00Z | `ea1525c` | `e0a9377`; amendment `54d9eec` | `speed_rt/s2/surface/results/`, `speed_rt/s2/bb/results/`, `speed_rt/s2/gapsw/results/` |

- F_k holds the stage's registration, code and timing sets, and was pushed to GitHub before any timing shot existed. T_k
  is the time of GitHub's earliest activity record whose `after` is F_k, and the beacon is the first drand mainnet round
  released at least 600 s after T_k, accepted when at least two endpoints agree. With R_k the SHA-256 of the round's
  signature, chunk j of timing set SET gets

      seed = 2^62 + (first 8 bytes, big-endian, of SHA-256(R_k ‖ TAG ‖ j as 2 bytes, big-endian)) mod 2^62

  with TAG = `convlines speed-rt SET` followed by a newline (`speed_rt/sets.py`).
- B_k, the beacon record with the stage's seeds or shot manifests, was committed and pushed, and nothing of the stage
  was timed before it was public. Stage 1's first timed process started at 22:02:03Z
  (`speed_rt/cascade/records/H100/R1/attempts.json`), stage 2's at 04:17:08Z
  (`speed_rt/s2/bb/records/gpu/attempts.json`). GitHub's record of the push of `6609d14` is in the activity list that
  `speed_rt/testsets/beacon_surf.json` keeps.
- Stage 2's shots were drawn once, on an x86-64 node with AVX2 (Stim 1.16.0, numpy 2.2.6), and the same bytes went to
  the pods and to the workstation; every machine checked each chunk against B_2's manifests before reading it. Stage 1's
  processes sampled their chunks during the run and recorded each chunk's seed and sha256.
- Stage 2 is one freeze for three families because they share one CPU session and its Grace fit; their claims stay
  separate (`STAGE2_common.md`, first paragraph).
- After the timing, each stage's frozen analysis ran twice with byte-identical outputs. The records were then sanitized
  for publication (`SANITIZE.md`), with every file whose sha256 an analysis checks kept byte for byte, and on the
  published records each analysis reproduces the published outputs byte for byte (`README.md`, "How to verify").

### 2.2 What is timed

For the surface and BB codes, our composite time per shot is

    composite = (a) GPU term + (c) handoff + (b) CPU term on the Grace basis        (per shot)

The GPU term comes from the pod and the CPU term from the workstation, so the stage-2 composites of our compositions, and
Ising's with its PM share, are not measured end to end. CUDA-Q's BP-OSD is measured end to end: host to host on the pod,
its OSD's host work included (2.3). NVIDIA's Ising paper (arXiv 2604.12841 v1, Section VI.3) estimates the transfer of
syndrome data between the pre-decoder on the GB300 GPU and PyMatching on the Grace CPU with NVIDIA's NVQLink
architecture, finds it negligible next to both runtimes and ignores it, and times its pre-decoders with host-device
transfers disabled. We use the same setting and keep our host-side handoff in every composite (C4).

- (a) GPU term: the network on the pod's H100 at batch one, with the input already on the GPU. Surface code: the
  release's `timing.gpu_term`, unchanged, over the 200,000 timing shots of the distance. At d = 9 to 17 each shot is one
  replay of a CUDA graph of the Triton engine, replayed in blocks of 100 back-to-back replays between two CUDA events
  after two warm-up blocks, and the term is the mean per shot. At d = 7 the hand-written CUDA kernel (sm_90a) stays
  resident on the GPU, and the term is its device time from the detection of a shot to the publication of the result,
  read from the kernel's own timer. BB codes: the frozen BB trees' fused Triton engine at batch one, 200 warm-up replays
  of one CUDA graph and then the mean of 5 × 100 timed replays between CUDA events (`bb.latency` of each tree).
- (c) Handoff. Surface code: measured on the pod, at d = 7 as the kernel's host round trip minus its device time (the
  largest of five processes), at d = 9 to 17 as the larger of 1.1 µs and the compiled router's estimate of the host-side
  work (`timing.deployed`). Whatever share of a CPU-GPU transfer this holds stays in the composite. BB codes: the
  registered host-side allowance of 1.1 µs per shot (routing decision, packing, indexing). The measured pinned-memory
  round trip of one shot's copies, 16.9 to 19.5 µs, is the transfer: it is left out, as in NVIDIA's setting, and
  published beside each composite (`speed_rt/s2/bb/results/numbers.json`, `pairs`). Adding 19.5 µs to every BB composite
  would change no verdict: the smallest margin of a comparison that holds against CUDA-Q is 117 µs (BB72, p = 0.2 %,
  the accurate composition).
- (b) CPU term: the fallback on the routed shots, timed on the workstation one shot at a time, single-threaded and
  pinned (50 untimed warm-up decodes, then `perf_counter` around each decode). The term is the sum of the routed shots'
  times over the number of timing shots, that is, the routed fraction times the mean time of a routed shot, put on the
  Grace basis (2.7). Surface code: correlated PM 2.4.0 on the routed shots among the distance's 200,000 timing shots.
  BB codes: sector or full-DEM Relay-BP (relay-bp 0.2.2) on the first routed shots in shot order (2.4 gives the
  counts), times the routed fraction of all the timing shots. A shot is routed when the pod's untimed pass of the
  network on the timing shots puts its confidence below the frozen threshold; the routing masks reach the workstation
  by sha256-checked copy (C2), and no CPU term recomputes a routing decision.
- Cascade (stage 1) has no composite. Each pipeline is timed end to end and host to host on one card, from the shot's
  detector bits in pinned host memory, through the network or Cascade on the GPU and every routed shot's fallback
  (Cascade, CS13, full-DEM Relay-BP or PyMatching in their worker processes), to the decision on the host. Single mode
  times the first 20,000 shots of the stream one at a time after 200 untimed ones and reports the mean and the 99.9th
  percentile of the per-shot latency; batched mode decodes 500,000 shots with the networks at batch 4,096 and routed
  rows queued on the GPU, and reports the wall time per shot; the networks alone (round R3) report the host-to-host
  time per shot at batch 4,096. Four rounds of fresh processes, R1 to R4, follow the order of the original loops
  (`speed_rt/cascade/drive.py`), and both sides of every comparison run on the same card in the same job.
- Decoder Switching (gap switching; descriptive, no verdict) runs entirely on the workstation, with no GPU and no
  handoff. Its front end decodes each shot with plain PM, with the weight, and then the split graph in the other logical
  class (50 untimed warm-up shots), over the distance's 50,000 timing shots; its routed term is correlated PM on the
  shots whose gap is below the frozen threshold, summed over all 50,000 shots. Both terms are put on the Grace basis.

### 2.3 The baselines' times

| baseline | how its time was obtained | machine | basis |
|---|---|---|---|
| plain and correlated PM, surface | the loop of (b) over all 10,000 shots of chunk 0 of the distance's timing set, the fastest of three fresh processes | workstation | Grace |
| Ising-Fast or Ising-Accurate + PM or + correlated PM | Ising's GPU time (2.5) plus the PM share: the loop of (b) over the detectors that the counted plan leaves on chunk 0, the fastest of three | pod; workstation | GPU as measured, PM share Grace |
| BP-OSD (min-LLR and BP 60), surface | CUDA-Q QEC 0.8.0 at batch one, 200 host-to-host calls on chunk 0, the fastest of three repetitions, with the pod's host CPU | pod | as measured |
| sector and full-DEM Relay-BP, ldpc BP-OSD, BB | one decode call per shot on the first shots of timing chunk 0 (2.4), the fastest of three fresh processes | workstation | Grace |
| CUDA-Q BP-OSD, BB | `nv-qldpc-decoder` at batch one, host to host, its OSD on the pod's host pinned to the timed core, the fastest of three processes | pod | as measured |
| Cascade alone | in the same job and loop as the compositions, on the same card, the slowest of three processes | each card | as measured |
| gap switching | 2.2; each term the fastest of three processes | workstation | Grace |

The decoders' settings are those of section 1: Relay-BP's `RelayDecoderF32` with 80 pre-iterations and 300 sets of 60
iterations, stopping after 5 solutions; ldpc 2.4.1's `BpOsdDecoder` and CUDA-Q's decoder with min-sum scale 1.0, 80
iterations and OSD-CS order 60; CUDA-Q's surface BP-OSD with OSD-CS order 10 and min-LLR initialisation with 10 BP
iterations, or 60 iterations (`STAGE2_bb.md` 1.2 and section 1.2).

### 2.4 Shot counts

| item | shots |
|---|---|
| surface GPU terms and our CPU terms | the 200,000 timing shots of the distance (20 chunks of 10,000) |
| surface PM baselines and the PM shares after Ising | chunk 0 of the timing set, 10,000 shots |
| surface BP-OSD | 200 calls on chunk 0 |
| BB72 sector Relay-BP and ldpc BP-OSD | the first 10,000 shots of chunk 0, at each p |
| BB72 full-DEM Relay-BP | the first 2,000 shots of chunk 0, at each p |
| BB72 compositions' CPU terms | timed, for both compositions: the first 1,000 of 3,886 routed shots at p = 0.3 % and all 740 routed shots at p = 0.2 % (`speed_rt/s2/bb/results/numbers.json`, `pairs`, `slowest_cpu_process`); registered caps: the first 1,000 routed shots (balanced at both p, accurate at p = 0.3 %) and the first 5,000 (accurate at p = 0.2 %); routed fractions over 200,000 (p = 0.3 %) and 300,000 (p = 0.2 %) timing shots |
| gross code sector Relay-BP | the first 1,000 shots of chunk 0 |
| gross code ldpc BP-OSD | the first 500 shots of chunk 0, cut from 1,000 |
| gross code full-DEM Relay-BP | the first 75 shots of chunk 0, cut from 300 |
| gross code compositions' CPU terms | the first 300 of 1,443 routed shots (balanced, cut from 1,000) and the first 25 of 2,998 (accurate, cut from 100); routed fractions over 150,000 timing shots |
| Decoder Switching | 50,000 shots per distance (5 chunks), cut from 200,000 |
| Cascade | single mode 20,000 shots after 200 untimed ones; batched mode 500,000 shots |

The authors cut the counts marked "cut" before stage 2's freeze, so that the CPU session would fit one evening (C1,
`STAGE2_bb.md` 1.3, `STAGE2_gapsw.md` 1.3). Every subset is the first N shots, or the first N routed shots, of its set
in shot order.

### 2.5 Ising

For each distance 12 TensorRT 10.14.1 plans were built, six per model (Ising-Fast, M1, and Ising-Accurate, M4): four
registered builds, FP16 and FP8 at optimisation levels 3 and 5, and two built as NVIDIA's code builds them, FP16 and
FP8, with NVIDIA's dynamic profile up to batch 2048 (`speed_rt/STAGE2_surface.md` 1.13.2). FP8 plans were quantized by
ModelOpt 0.47.0 on validation shots. The pod for d = 7, 9 and 11 built its plans after stage 2's freeze and then timed
them. For d = 13, 15 and 17 an earlier H100 pod built the plans before the freeze, from tree files that the frozen
analysis checks against the frozen tree, saved every build output with its sha256 and was released (`STAGE2_surface.md`
1.13.7). The timing pod, with the same GPU model and the same TensorRT, CUDA and torch versions, loaded the 134 saved
files (the 133 outputs that `speed_rt/s2/surface/records/high/session/BUILD_INPUTS.json` lists with their sha256, and
that file) after checking each file's sha256; its build step skipped every plan already built and re-ran every check.
Before the timed block, every plan was checked on 10,000 validation shots against NVIDIA's PyTorch numerics followed by
plain PM: decision agreement at least 99.9 % (FP16) or 99 % (FP8), the paired test not "worse", the residual-detector
density within 5 % of the reference, and for an FP8 plan every convolution that the recipe quantized running an FP8
kernel. A plan that failed was never timed. Each validated plan was timed at batch one on the first shot of the timing
set's chunk 0, inside a CUDA graph (200 warm-up replays, then five repetitions of 100 replays between CUDA events, in
two processes of opposite plan order) and with NVIDIA's command,
`trtexec --loadEngine=PLAN --useCudaGraph --noDataTransfers --useSpinWait` with that shot as input (five repetitions,
the "GPU Compute Time" mean of each). Ising's GPU time per model and distance is the fastest accepted repetition of any
validated plan (`speed_rt/ising_select.py`), and the counted plan's residual detectors on chunk 0 give the PM share.
Ising's accuracy comes from SURF-v2's reads with NVIDIA's PyTorch numerics, so no single run measured both.

### 2.6 Acceptance and checks

- Pods. Before anything was timed on a pod: five checks on the idle GPU, each a separate process that holds 1 GiB and
  runs 3 s of fp16 matrix products, passing when the median SM clock is at least 97 % of the maximum, the clock-event
  reasons are within GPU idle, applications clocks and SW power cap, the enforced power limit is the default, under 64
  MiB were in use before and no other process used the GPU; then CPU steal at most 1 % over 5 s and a container quota
  of at least 20 CPUs (`speed_rt/gpu_prescreen.py`). A pod times only GPU pipelines, one process at a time, pinned to
  one timed core: the networks, Ising's plans, and CUDA-Q's BP-OSD, whose OSD runs on that core and counts in its
  host-to-host time. Host-side load is recorded and decides nothing.
- GPU windows, stage 1 (`SPEED_RT.md` 1.4): a window is rejected for outside interference, that is, another compute
  application on the card, a gap of more than 1 s between `nvidia-smi` samples (every 100 ms), a clock-event reason
  other than GPU idle, applications clocks or SW power cap, an enforced power limit below the default, or an SM clock
  below 90 % of the maximum in a sample without the SW power cap. The workload's own power cap at the default limit is
  recorded, not rejected; samples below 97 % of the maximum are flagged and published.
- GPU windows, stage 2 (`STAGE2_surface.md` 1.4, `STAGE2_bb.md` 1.4): no other compute process on the GPU and
  clock-event reasons none or idle; the timing modules apply a 97 % SM-clock floor inside their own timed spans, and
  Ising's continuous timings require every sample to be at least 97 % of the maximum.
- The workstation (C1): the CPU frequency governor `performance` on every online CPU (amd-pstate-epp, energy-performance
  preference `performance`) with boost on; CCD0's SMT siblings (CPUs 16-23) offline; at most 8 timed processes at once,
  each pinned to one CPU of CCD0 (CPUs 0-7) and so to a physical core of its own; every other process confined to CCD1
  (CPUs 8-15 and 24-31) by systemd `AllowedCPUs`; the Bluetooth service stopped; no other user running anything during
  the timing. Everything a timed process reads sits in a tmpfs. The two sides of each comparison alternate across the
  slots, and a slot whose job ends early runs an untimed filler until the window ends, so that eight cores stay busy.
  Before and after each window, every slot times a reference segment (plain PM per shot at d = 9 on validation chunk
  0), and the window passes when the harness's own block-read counter did not grow, every reference value is at most
  1.10 times the session's reference (the median of 3 segments at full concurrency, 17.32 µs per shot), each segment's
  largest value is at most 1.06 times its smallest, and every process exits 0. `speed_rt/s2/gapsw/records/prescreen/`
  holds the session's topology and settings (`topo.json`) and its reference (`sessionref.json`).
- Replacement and failure, one rule for both stages: a rejected or failed process is replaced by a fresh one with the
  same arguments, at most twice. A slot without an accepted attempt is not established, and every comparison that needs
  it is withheld; no other value stands in. Nothing ran again because of a value it produced, and every attempt is
  published.
- Outcome. Stage 1 established every slot on both cards (`speed_rt/cascade/results/stdout.txt`). On the H100, two
  windows were rejected for another compute application on the card, and their processes were replaced; 78 of the 153
  counted windows ran under the workload's own power cap at the default 700 W, with SM clocks down to 0.682 of the
  maximum, and 77 had samples below 97 % of it. On the RTX PRO 6000, 44 of 153 windows had samples below 97 % (lowest
  0.938) and none was power-capped (`speed_rt/cascade/results/numbers.json`, `clock`). In stage 2, every slot of the
  three timing pods was accepted at its first attempt, and the CPU session passed all 46 windows and accepted all 290
  processes at their first attempt (`speed_rt/s2/bb/records/cpu/session.json`, `speed_rt/s2/README.md`).

### 2.7 Statistics and the Grace basis

- Stage 2 (C6): our side takes the slowest accepted of its fresh processes (three for every CPU term; five for the
  GPU terms of the surface networks and of BB72's, three for the gross code's; for the surface handoff, the largest of
  five at d = 7 and one process at d = 9 to 17), and every baseline the fastest of its three; there are no medians.
  Ising's GPU term is the exception: it is the smallest accepted repetition across all validated plans and both timing
  methods (2.5), which favours Ising. Stage 1 (`SPEED_RT.md` 1.7): the registered analysis takes the slowest of three on
  both sides, with sensitivity rows that repeat every comparison with Cascade's fastest process. The paper reports the
  sensitivity rows, so that every comparison times our side by its slowest process and the baseline by its fastest;
  they lower no speed-up by more than 3.2 % and change no outcome.
- "X takes less time than Y" holds when X's value is strictly below Y's, on the Grace basis for CPU parts. There is no
  statistical test. Speed-up = the baseline's time over ours. Ranges and ratios of times in the prose are rounded
  against us: our times, and our time as a fraction of another decoder's, up; the other decoders' times, and speed-ups
  (also when below 1), down. A time per round is the time per shot over the number of rounds: d on the surface code, 6
  on BB72, 12 on the gross code. Every repetition's value and the range are published beside each value.
- Grace basis (C3). NVIDIA's Ising paper (arXiv 2604.12841 v1, Tables VII and IX) times plain and correlated PyMatching
  on one Grace Neoverse-V2 core at 18 points: plain PM at d = 13, 21 and 31 and correlated PM at d = 5, 9, 13, 17, 21
  and 31, each at p = 0.3 % and 0.6 %, on NVIDIA's legacy memory circuits. The workstation timed the same 18 points in
  stage 2's CPU session, on those circuits rebuilt from NVIDIA's code (`speed_rt/circuits/`), each the slowest of three
  processes, which gives the smaller factor. Per point r = NVIDIA's time per round × d / ours; each decoder's factor is
  exp(mean ln r) over its points, its spread the standard deviation of ln r, and the common factor the geometric mean
  over all 18. The decoders agree when |ln f_plain − ln f_corr| is at most the larger spread. The fit gave f_plain =
  1.4543 (spread 0.400) and f_corr = 1.5383 (spread 0.146), which agree, so the common factor 1.5098 converts every CPU
  time of stage 2: the PM baselines, our CPU terms, the PM shares after Ising, Relay-BP, ldpc BP-OSD and gap switching
  (`speed_rt/s2/surface/results/fit_s2.json`; `speed_rt/grace_fit.py`). The same factor holds at d = 17, where NVIDIA's
  correlated PM at p = 0.6 % (860.71 µs per shot) is one of the 18 points. Per point the ratio ranges from 0.90 to 2.23;
  the residuals are published (`speed_rt/s2/surface/results/table.md` row F1, `speed_rt/s2/gapsw/results/gs_tables.md`).
  Not converted: GPU terms, handoffs, Ising's TensorRT times, CUDA-Q's host-to-host times and every stage 1 time.
- A Grace-basis number is an estimate; we have no GH200 or GB300, and nothing ran on a Grace CPU. Every raw time is
  published beside its converted value, and every verdict is computed on the raw basis as well, as a sensitivity check
  that decides nothing: 45 of 48 surface comparisons hold for each network set on the raw basis (row S4x), and the gross
  code's balanced composition takes 13.57 ms against CUDA-Q's 13.70 ms (`speed_rt/s2/bb/results/numbers.json`,
  `sensitivity_raw_xeon`).

### 2.8 What SPEED-RT left out, and its deviations

- Left out by the authors' decision, as the registrations state: the earlier registered gross-code test's times
  (`STAGE2_bb.md` 1.1); the screening times on MIG slices, a number the paper drops, and the speed limits that the
  frozen routing rules used, kept as the values the rules were frozen with (`STAGE2_surface.md` 1.9, E3 and E9); and
  development measurements, non-time statements and inputs of frozen rules, labelled as such (E4 to E7 and E10).
  SURF-v2's speed bars (1.2) are among the frozen inputs.
- `54d9eec`, after the timing: line 208 of `speed_rt/gapsw/analysis_gs.py` checked for 20 chunks per distance, but the
  authors had cut the draw to 5 before the freeze without updating this check, so the frozen code refused the registered
  draw. The amended line takes the count from the frozen sets file. No other line depends on the old count, and no
  Decoder Switching number existed before the amendment (`speed_rt/s2/README.md`). The surface and BB analyses ran
  F_2's code unchanged.
- `2ce2e49`: stage 1's `speed_rt/cascade/records/H100/R4/derived_lock.json` and `session.json` are published byte for
  byte, unsanitized, because the frozen analysis checks their sha256; their sanitized copies in `56e2023` failed that
  check (`speed_rt/cascade/results/README.md`).

### 2.9 Hardware and versions

| work | hardware | software |
|---|---|---|
| stage 2, surface GPU parts | two rented H100 80GB HBM3 SXM pods: d = 7, 9 and 11 with an Intel Xeon Platinum 8470 host (driver 580.126.09), d = 13, 15 and 17 with an AMD EPYC 9554 host (driver 570.211.01) | Python 3.12.3, torch 2.11.0+cu126, Triton 3.6.0, TensorRT 10.14.1 and its `trtexec`, ModelOpt 0.47.0, nvcc 12.1 for the d = 7 kernel, CUDA-Q QEC 0.8.0 |
| stage 2, BB GPU parts | one rented H100 80GB HBM3 SXM pod, Intel Xeon Platinum 8470 host (driver 580.126.09) | Python 3.12.14, torch 2.11.0 (+cu128 for BB72, +cu126 for the gross code), Triton 3.6.0, cudaq-qec 0.8.0 |
| stage 2, every CPU part | one Linux workstation: AMD Ryzen 9 9950X (16 cores, 32 threads), Ubuntu 24.04, kernel 7.0.0 | Python 3.12.3, numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2, ldpc 2.4.1 |
| stage 1, Cascade | one NVIDIA RTX PRO 6000 Blackwell Server Edition (MIG off) on the Alpine cluster, AMD EPYC 9555 host (driver 590.48.01); one rented H100 80GB HBM3 SXM pod, Intel Xeon Platinum 8480+ host (driver 580.126.09) | Python 3.12.14 (RTX) or 3.12.3 (H100), torch 2.11.0+cu128, Triton 3.6.0, numpy 2.2.6, Stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2 |

### 2.10 The earlier timing sessions (superseded by SPEED-RT)

Before SPEED-RT, the paper's speed numbers came from the timing sessions of the earlier tests. SPEED-RT supersedes
them, and no speed number of the paper or of `README.md` comes from them. Their records stay in the repository
(`results/surface/SURFACE.md` section 4, `bb/BB.md`, `cascade/bb72/CASCADE.md`, `cascade/d13/RESULTS.md`). They differed
from SPEED-RT as follows.

- Surface code: the GPU terms and handoffs were timed on validation shots on one H100 per test. The CPU term ran on one
  pinned core of an exclusive AMD EPYC node (7713 for studies 2 and 4, R17 and SURF-v2; 7443 for study 3) and was
  converted with that node's own correlated-PM speed: (b) on the Grace basis = (b) on the node × G(d) / (the node's
  correlated-PM time per shot on validation chunk 0, median of 3), where G(d) was the Grace-basis correlated-PM time.
- The Grace basis then used NVIDIA's published values where they exist (plain PM at d = 13, 129.61 µs per shot;
  correlated PM at d = 9 and 13, 67.59 and 279.63 µs; Ising's PM share after M1 and M4 at d = 13, 39.65 and 33.15 µs)
  and elsewhere our measured time times the ratio of NVIDIA's value to our time at d = 13 on the same machine: on one
  Apple M4 core 2.594 for plain PM (d = 7 to 11), 2.374 for correlated PM (d = 7 and 11) and 1.880 and 1.908 for the PM
  shares after M1 and M4; on the EPYC 7713 node 1.0339 for plain and 0.9361 for correlated PM at d = 15 and 17
  (`evaluation/timing/grace_basis.csv`). SPEED-RT replaced these anchors with one fit over 18 points timed in its own
  session.
- Ising's plans and BP-OSD were timed in each test's session; SURF-v2 timed only its own networks and compared them with
  the earlier registered baseline times (`surface_v2/surf_v2/config.json`, `registered_times_us`). The two Ising +
  correlated PM controls were post hoc (`results/surface/SURFACE.md` section 6).
- BB codes: the GPU terms and CUDA-Q ran on an H100 pod with an Intel Xeon Platinum 8480+ host; every CPU time ran on
  one Apple M4 core (4 performance and 6 efficiency cores, macOS 26.2), as measured and not converted; the handoff was
  the slowest measured host time of one shot's copies, at least 1.1 µs. RETEST timed nothing and estimated its CPU terms
  as the registered CPU term × the re-test's routed fraction / the registered routed fraction. Full-DEM Relay-BP's own
  time at p = 0.2 % came from a supplementary M4 session after the test (36.41 ms per shot;
  `bb/bb72/ising_compare/data/bb/stage3/SUPPLEMENT_fulldem_time_p0.002.md`).
- Cascade: the original studies timed end to end on one RTX PRO 6000, the slowest of three processes; they recorded the
  clock and gated nothing (rows below 97 % were flagged), and an H100 session was descriptive.

| speed numbers | measured then by | hardware | date |
|---|---|---|---|
| surface: SURF-v2's networks, with the deployed networks beside them | SURF-v2's one qualified session; its CPU term one job right after the session's handoffs | H100; AMD EPYC 7713 | 2026-10-13 |
| surface baselines, d = 7 to 13 | study 2's registered session | H100 | 2026-10-03 |
| surface baselines, d = 15 | study 3's registered session | H100 | 2026-10-04/05 |
| surface baselines, d = 17 | study 4's registered sessions | H100 | 2026-10-06/07 |
| matching times on the Grace basis | NVIDIA's published values and the registered extrapolations | Apple M4; AMD EPYC 7713 | with the registered tests |
| Ising + correlated PM, d = 7 to 15 | the post hoc controls, not registered | H100; Apple M4; AMD EPYC 9534 | after study 3 (M4 part 2026-10-06) |
| BB72, Table IV | the registered sessions, and a later M4 session for amendment S's CPU term | H100; Apple M4 | 2026-10-07 |
| gross code, Table IV | the registered stage-4 sessions | H100; Apple M4 | 2026-10-07 |
| full-DEM Relay-BP on BB72 at p = 0.2 % | a supplementary session after the BB72 test, not registered | Apple M4 | 2026-10-09 |
| Cascade on BB72 | D1's registered timing | RTX PRO 6000 | 2026-10-11 |
| Cascade at d = 13 | the comparison's timing and its repeat | RTX PRO 6000 | 2026-10-12 |

## 3. Re-running the tests

Install the pinned environment of `README.md` (Installation). These checks need only the repository and a CPU:

```bash
python weights/check_weights.py                     # the 38 networks: sha256, build, strict load, parameter count
python retest/seeds.py --verify retest/run/beacon.json retest/run/seeds.json   # every RETEST seed from the beacon
python cascade/bb72/repro/recompute_numbers.py --out /tmp/cascade_numbers    # Cascade on BB72, from the records
python cascade/d13/surface/scripts/results_tables.py cascade/d13/out        # Cascade at d = 13, from the records
cd evaluation && python -m timing.composite && cd ..   # the earlier tests' surface composites and speed cells (2.10)
cd surface_v2
python -m convlines.beacon check --beacon surf_v2/beacon.json                 # SURF-v2's beacon record, offline
python -m convlines.testset seeds --set sv2test --approval surf_v2/beacon.json            # its 700 test seeds
python -m convlines.testset regen --set sv2test --d 9 --chunks 0 --out shots/sv2test      # one chunk against its sha256
```

Rebuilding chunks bit for bit needs an x86-64 CPU with AVX2 and the pinned Stim and numpy. SURF-v2's analysis
(`python -m surf_v2.analysis`, `README.md` step 6) re-runs on the per-shot reads and timing records, which the authors
archive; `surface_v2/surf_v2/analysis/analysis.json` lists each with its sha256, and the analysis refuses any input that
does not match the freeze.

`README.md` ("How to verify") checks SPEED-RT's two beacon records, re-derives stage 1's seeds and reruns its four
analyses on the published records, which needs only Python 3.12 and, for Decoder Switching, numpy.

Re-reading the decoders needs an NVIDIA GPU (sm_90, such as an H100 or H200, for the d = 7 kernel; one RTX PRO 6000
for Cascade), NVIDIA Ising-Decoding at commit 3ce4379 for Ising, and CPUs for Relay-BP and ldpc (`README.md`,
Hardware). Re-timing as SPEED-RT did needs the hardware of 2.9: H100 80GB HBM3 SXM machines for the GPU terms, Ising's
plans and CUDA-Q (`speed_rt/pod/run_s.sh`, `speed_rt/bb/pod_bb.sh`), one RTX PRO 6000 and one H100 for Cascade
(`speed_rt/cascade/drive.py`), and for the CPU terms a Linux machine whose timed cores run nothing else, with their SMT
siblings offline (`speed_rt/cpu_ws/`, whose `README.md` describes its Linux host mode). The documents in the table at
the top give the full rules and every command, with the plans `cascade/bb72/cascade_compare/PHASE_C_PLAN.md`,
`cascade/bb72/cascade_compare/phase_d/PHASE_D1_PLAN.md` and `cascade/d13/SURFACE_PLAN.md` for the original Cascade
tests.
