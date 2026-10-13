# How we measured accuracy and speed

This document states how the paper's accuracy and speed numbers were measured: the shots, the decoders, the statistics,
what was timed and on which hardware. It summarises; the registration documents it names hold the full rules and govern
where the two differ. Paths are relative to the repository root.

- A shot is one run of the error-correction circuit. A decoder reads its detector bits and predicts the flip of each
  logical observable.
- The pair is ConvLines with a strong decoder. The network decodes every shot and gives a confidence; a shot whose
  confidence is below the threshold τ goes to the strong decoder, whose answer replaces the network's. τ is set so that
  a fixed fraction q of calibration shots is routed.
- A test is registered when its networks, thresholds, test-seed rule and analysis were committed and pushed to GitHub
  before any test shot existed.

| paper numbers | accuracy measured by | full rules |
|---|---|---|
| surface code, d = 7 to 17 | SURF-v2 | `surface_v2/surf_v2/REGISTRATION.md` |
| BB72 and the gross code (Table I) | RETEST, re-running the original BB tests on fresh shots | `RETEST.md` section 3.1; `bb/BB.md` |
| Cascade on BB72 and on the surface code at d = 13 | RETEST, re-running the original Cascade tests | `RETEST.md` sections 3.2 and 3.3; `cascade/bb72/CASCADE.md`; `cascade/d13/SURFACE_PLAN.md` |
| output heads (Table III) | RETEST | `RETEST.md` section 3.4 |

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
  code, when any of the 12 logical observables is wrong (a block failure). The pair takes the strong decoder's answer
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
- Decoders, on the same test shots: the new pair (the network retrained on the observable's flip alone, with correlated
  PM as its strong decoder), the network alone, and eight baselines: plain and correlated PyMatching (PM) 2.4.0;
  NVIDIA's pre-decoders Ising M1 (Fast, receptive field 9) and M4 (Accurate, receptive field 13), run by NVIDIA's own
  PyTorch module (fp32, TF32 as NVIDIA's runner sets it) and followed by plain or correlated PM; and BP-OSD from
  cudaq-qec 0.8.0 (OSD-CS order 10, with min-LLR initialisation and 10 BP iterations, or with 60 iterations), which
  read every test shot at d = 7 to 13 and the first 200,000 at d = 15 and 17. The networks were read by the deployed
  engine at batch one, the code that is timed, and the earlier tests' deployed networks were read beside them.
- Routing. On the calibration shots, the rule considers each q of a grid (0 to 16 %, up to 30 % at d = 17) whose
  projected composite time is within the distance's speed bar (15.16 µs at d = 7 up to 322.23 µs at d = 17), and takes
  the smallest with at most M − 2√(2M) calibration failures, where M is the earlier tests' planning count (3,401 at
  d = 7 down to 554 at d = 17); otherwise a fixed fallback. Frozen: q = 2 % at d = 7, q = 0 at d = 9, 11 and 13 (the
  network decides every shot) and q = 1 % at d = 15 and 17 (`surface_v2/surf_v2/freeze.json`).
- Statistics. Accuracy against each baseline: McNemar and Newcombe's interval at 97.5 %; "better" when p < 0.025 and the
  interval lies below zero. Speed against each baseline: the pair's composite time strictly below the baseline's
  registered time, a measured inequality without a statistical test. A distance's claim needs all 16 comparisons to
  hold, so no multiplicity correction is needed. New against deployed networks, alone and as pairs: McNemar with
  Holm's correction over the 12 tests of all six distances at 0.05.

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
  nine cells of the BB tests is 0.15.
- RETEST changed only the shots, through shims that swap in the new seeds (`retest/shim/`). BB72's networks were read
  on an RTX PRO 6000 (the original test used an H100); on the original shots the two cards give identical decisions
  and margins. The one
  recorded deviation is the job layout of the gross-code CPU reads (`RETEST.md`).

### 1.4 Cascade (RETEST)

- BB72, under a fitted noise model (`depspam_fm`: the uniform noise plus final-measurement flips and depolarizing error
  p after resets and before measurements), chosen to reproduce the failure rates that Cascade's paper reports for
  Tesseract, BP-OSD and Relay-BP (`cascade/bb72/noise_fit/CASCADE_CAL.md`). Our Cascade has 14 blocks of width 512
  (19,687,937 parameters). In the pair a six-block ConvLines network (841,053 parameters) routes q = 2 % of shots at
  p = 0.2 % and 10 % at 0.3 % to it. RETEST drew 11,000,000 and 17,000,000 test shots. Four hypotheses (two p, eager
  and compiled paths) form one Holm family at 0.05; a compiled comparison needs its single-shot and its batched read
  both to be better.
- Surface code at d = 13, the circuit of 1.2 at p = 0.6 % and 0.35 %. Our Cascade has 2,531,841 parameters. In the pair
  the deployed d = 13 network routes q = 10 % at p = 0.6 % and 1 % at 0.35 % to it. RETEST drew 1,000,000 and
  30,000,000 test shots. The 40 hypotheses (two p, two modes, eight superiority and two non-inferiority tests each)
  form one Holm family at 0.05; non-inferiority is one-sided, with a margin of 10 % of Cascade's failure rate on
  development shots. At p = 0.6 % the pair was non-inferior to Cascade but not significantly better.
- Both run every decoder on one NVIDIA RTX PRO 6000 Blackwell Server Edition; the code refuses other cards.

## 2. Speed

### 2.1 What is timed

Our decoder's speed score is the composite time per shot at batch one (one shot at a time):

    composite = (a) GPU term + (b) CPU term on the Grace basis + (c) handoff        (µs per shot)

`surface_v2/timing/composite.py` documents and recomputes it for the earlier tests; `surface_v2/surf_v2/analysis.py`
computes the same sum for SURF-v2.

- (a) GPU term: our network on one NVIDIA H100 80GB HBM3, run by our engine at batch one with the input bits already on
  the GPU. At d = 9 to 17 the engine is a Triton program and each shot is one replay of a CUDA graph: a process
  captures 100 graphs on 100 input slots, runs two warm-up blocks, then replays all 200,000 validation shots in blocks
  of 100 back-to-back replays between two CUDA events, and reports the mean per shot. At d = 7 a hand-written CUDA
  kernel for sm_90a stays resident on the GPU; (a) is the device time from its detection of a shot to the publication
  of the result, read from the kernel's own timer.
- (b) CPU term: the validation shots that the deployed decoder routes are decoded one at a time by correlated PM 2.4.0,
  single-threaded and pinned to one core of an exclusive AMD EPYC node (50 warm-up decodes, then `perf_counter` around
  each decode). The term is the sum of those times over all 200,000 validation shots, that is, the routed fraction
  times the mean time of a routed shot, put on the Grace basis (2.4).
- (c) Handoff: moving the shot to the GPU and the result back, and the branch. At d = 7 it is measured as the host round
  trip minus the device time, the largest of five processes. At d = 9 to 17 a compiled router runs the engine as the
  deployed decoder over the 200,000 validation shots, the estimate is mean complete path − mean device-only time −
  routed fraction × mean correlated-PM time of a routed shot, and (c) = max(1.1 µs, estimate). The 1.1 µs is a fixed
  allowance; no measurement shows that it bounds the handoff.

The surface timings use validation shots, never test shots: 200,000 per distance, with seeds fixed before the tests.
The BB composites (`bb/BB.md`) have the same three terms and use test shots: (a) our network through a fused Triton
engine at batch one on an H100 80GB (200 warm-up replays of one CUDA graph, then the mean of 5 × 100 timed replays);
(b) the routed fraction times the fallback's mean time per routed test shot on one Apple M4 core; (c) the slowest
measured host time of one shot's copies to and from the GPU, at least 1.1 µs. The Cascade comparisons time each
pipeline end to end, from shots in host memory to decisions on the host, CPU fallbacks included, on one RTX PRO 6000:
20,000 shots one at a time after 200 untimed ones (single mode), and 500,000 shots with the networks at batch 4,096
(batched mode).

### 2.2 The baselines' times

| baseline | how its time was obtained | basis |
|---|---|---|
| plain and correlated PM, surface | NVIDIA's published per-core times where they exist, our extrapolations elsewhere (2.4) | Grace |
| Ising M1 / M4 + PM or + correlated PM | the fastest validated TensorRT 10.14.1 plan (FP16 or FP8; FP8 won at every d) at batch one on the session's H100: one CUDA graph per plan, 200 warm-up replays, 5 × 100 timed back-to-back replays as NVIDIA's trtexec does, the fastest repetition; plus the matching share on the detectors Ising leaves | H100 plus Grace |
| BP-OSD, surface (two settings) | cudaq-qec 0.8.0 at batch one on the session's host, GPU BP plus host OSD, host to host, the fastest of three repetitions | as measured |
| sector and full-DEM Relay-BP, BP-OSD (ldpc), BB | one Apple M4 core, single-threaded, the mean over the first test shots, the fastest of three fresh processes | as measured |
| BP-OSD (cudaq-qec), BB | 300 batch-one calls on the H100, host to host, the fastest of three processes | as measured |
| Cascade | alone, in the same job and loop as the pairs on the same RTX PRO 6000, the slowest of three processes | as measured |

An Ising plan counts only if it passes an output check, and an FP8 plan only if every convolution the recipe quantized
runs an FP8 kernel (`evaluation/timing/ising_trt.py` enforces both). Ising's accuracy comes from NVIDIA's PyTorch
numerics and its time from these plans, so no single run measured both. Our terms count the slowest process; Ising's
plans, BP-OSD and the BB baselines count the fastest.

### 2.3 The statistic and the checks

- Surface GPU term: five fresh processes per network, interleaved across networks; the slowest counts. A process is
  accepted only if every SM-clock sample (`nvidia-smi` every 100 ms, on other CPU cores) is at least 97 % of the
  maximum, the only throttle reason is idle, no other process uses the GPU and its decisions equal the deployed
  decoder's bit for bit. A network may replace at most two rejected processes.
- Surface session check, at the start and the end: a separate process holds 1 GiB on the GPU and runs 3 s of fp16
  matrix products; it passes with a median SM clock of at least 97 % of the maximum, no throttle reason but idle, under
  64 MiB in use beforehand and no other process. Studies 2 and 3 also gated the host (an idle pinned core with steal at
  most 1 %, plain PM within 1.10 times a reference host's time, the router's launch floor within 1.15 times the
  reference's 4.986 µs). SURF-v2 ran the check five times on each rented machine's idle GPU before using it.
- Surface CPU term: one job holding a whole node, with three interleaved repetitions of every network's term and of the
  node's correlated-PM reference (2.4), each a fresh single-threaded process pinned to core 8 with its SMT sibling idle.
  The slowest term and the median reference count.
- BB: the GPU term is the slowest of five accepted processes (three on the gross code) under the same idle-card and
  97 % clock rule; CPU terms are the slowest of three processes. The M4 ran one process at a time, each after an idle
  check. macOS cannot pin a core type, so each process also timed a fixed integer loop: 0.049 to 0.053 s in every
  registered process, against about twice that on an efficiency core.
- Cascade: three fresh processes per pipeline in a repetition-major loop, the slowest counting. Every row records its
  lowest SM clock, and rows below 97 % are reported and flagged. At d = 13 the timings that fell below the rule were
  repeated; the repeat supplies the displayed batched times of Cascade alone, and the verdicts use the first timing.

### 2.4 The Grace basis

NVIDIA's Ising paper timed PyMatching on one Grace Neoverse-V2 core at p = 0.6 %. We put every surface-code matching
time on that basis, so that our pairs and the matching baselines are compared on NVIDIA's reference CPU.

- NVIDIA's published values, per round times d rounds: plain PM at d = 13, 9.97 µs, so 129.61 µs per shot (Table VII);
  correlated PM at d = 9 and 13, 7.51 and 21.51 µs, so 67.59 and 279.63 µs (Table IX); Ising's PM share after M1 and
  M4 at d = 13, 3.05 and 2.55 µs, so 39.65 and 33.15 µs.
- Every other Grace value is our measured time times the ratio NVIDIA's value / our time at d = 13 on the same machine.
  On one Apple M4 core the ratios are 2.594 for plain PM (d = 7 to 11; for example 6.62 µs × 2.594 = 17.16 µs at
  d = 7), 2.374 for correlated PM (d = 7 and 11), and 1.880 and 1.908 for the PM shares after M1 and M4. On an
  exclusive AMD EPYC 7713 node they are 1.0339 for plain and 0.9361 for correlated PM, used at d = 15 and 17 (for
  example 762.77 µs × 0.9361 = 714.05 µs for correlated PM at d = 17). `evaluation/timing/grace_basis.csv` lists every
  value with its source.
- Our CPU term is converted with the node's own correlated-PM speed:

      (b) on the Grace basis = (b) on the node × G(d) / (the node's correlated-PM time per shot on validation chunk 0, median of 3)

  where G(d) is the Grace-basis correlated-PM time from the same table. The ratio cancels the node's absolute speed to
  first order; in the registered sessions it lay between 0.8164 and 1.1822. At d = 7, 11, 15 and 17, G(d) is itself an
  extrapolation.
- Converted: plain and correlated PM, Ising's matching shares and our surface CPU term. Not converted: every GPU term
  and handoff, Ising's TensorRT times, the surface BP-OSD times, every BB time (H100 and M4 as measured) and the Cascade
  timings. Nothing ran on a Grace CPU.
- At d = 17, NVIDIA's own correlated-PM value (50.63 µs per round × 17 = 860.71 µs) is 20.5 % above our extrapolation.
  No registered result uses it; `evaluation/timing/R17_GRACE_NOTE.md` works out what it would change, and
  `surface_v2/surf_v2/analysis/composite.csv` gives SURF-v2's d = 17 composites on that basis beside the registered ones.

### 2.5 Hardware and versions

| work | hardware | software |
|---|---|---|
| surface: GPU terms, handoffs, Ising's plans, BP-OSD | one NVIDIA H100 80GB HBM3 (SXM) in a rented cloud machine, one session per test | torch 2.11.0+cu126, Triton 3.6.0; the d = 7 kernel built for sm_90a by nvcc 12.1.105; TensorRT 10.14.1, ModelOpt 0.47.0; cudaq-qec 0.8.0 |
| surface: CPU terms | one pinned core of an exclusive node: AMD EPYC 7713 (studies 2 and 4, R17, SURF-v2) or AMD EPYC 7443 (study 3) | Python 3.12.14, PyMatching 2.4.0, numpy 2.2.6, Stim 1.16.0 |
| Grace extrapolations | one Apple M4 core; the AMD EPYC 7713 node at d = 15 and 17 | PyMatching 2.4.0 |
| BB: GPU side | one NVIDIA H100 80GB HBM3 in a rented cloud machine, Intel Xeon Platinum 8480+ host | torch 2.11.0 (+cu128 for BB72, +cu126 for the gross code), Triton 3.6.0, cudaq-qec 0.8.0 |
| BB: CPU timing | one core of an Apple M4 Mac mini (4 performance and 6 efficiency cores, macOS 26.2) | Python 3.12.14, relay-bp 0.2.2, ldpc 2.4.1 |
| Cascade | one NVIDIA RTX PRO 6000 Blackwell Server Edition, AMD EPYC 9555 host | driver 590.48.01, torch 2.11.0+cu128, Triton 3.6.0, relay-bp 0.2.2, PyMatching 2.4.0 |

The records do not state the GPU driver version of the H100 timing sessions.

### 2.6 Which session measured which number

RETEST and SURF-v2 re-tested accuracy on fresh shots. Neither re-measured a baseline's time, and RETEST timed nothing.
Every speed number in the paper comes from one of these sessions:

| speed numbers | measured by | hardware | date |
|---|---|---|---|
| surface: our new networks (the paper's surface rows), with the deployed networks beside them | SURF-v2's one qualified session; its CPU term one job right after the session's handoffs | H100; AMD EPYC 7713 | 2026-10-13 |
| surface baselines, d = 7 to 13 | study 2's registered session | H100 | 2026-10-03 |
| surface baselines, d = 15 | study 3's registered session | H100 | 2026-10-04/05 |
| surface baselines, d = 17 | study 4's registered sessions | H100 | 2026-10-06/07 |
| matching times on the Grace basis | NVIDIA's published values and the registered extrapolations (2.4) | Apple M4; AMD EPYC 7713 | with the registered tests |
| Ising + correlated PM, d = 7 to 15 | the post hoc controls, not registered: the registered plan times plus a correlated-PM share measured afterwards (`results/surface/SURFACE.md` section 6) | H100; Apple M4; AMD EPYC 9534 | after study 3 (M4 part 2026-10-06) |
| BB72, Table I | the registered sessions: GPU term and handoff, cudaq-qec, the CPU baselines and fallback times on the M4, and a later M4 session for amendment S's CPU term | H100; Apple M4 | 2026-10-07 |
| gross code, Table I | the registered stage-4 sessions | H100; Apple M4 | 2026-10-07 |
| full-DEM Relay-BP's own time on BB72 at p = 0.2 % | a supplementary session after the BB72 test, not registered: 36.41 ms per shot, the fastest of three processes; its control re-ran the registered p = 0.3 % value and got 58.05 ms against 58.60 ms (`bb/bb72/ising_compare/data/bb/stage3/SUPPLEMENT_fulldem_time_p0.002.md`) | Apple M4 | 2026-10-09 |
| Cascade on BB72 | D1's registered timing | RTX PRO 6000 | 2026-10-11 |
| Cascade at d = 13 | the comparison's timing and its repeat | RTX PRO 6000 | 2026-10-12 |

RETEST reused every GPU term, handoff and baseline time and recomputed only the BB CPU terms, because the routed
fraction is the one time input that depends on the test shots (`RETEST.md` section 4):

    CPU term (re-test) = registered CPU term × re-test routed fraction / registered routed fraction

for each registered timing process, keeping the slowest process as frozen. On the registered shots this adapter
reproduces every original row bit for bit. RETEST's speed numbers are therefore estimates from the registered sessions,
and the paper labels them as such.

A registered speed re-test of every speed number is in preparation.

## 3. Re-running the tests

Install the pinned environment of `README.md` (Installation). These checks need only the repository and a CPU:

```bash
python weights/check_weights.py                     # the 38 networks: sha256, build, strict load, parameter count
python retest/seeds.py --verify retest/run/beacon.json retest/run/seeds.json   # every RETEST seed from the beacon
python cascade/bb72/repro/recompute_numbers.py --out /tmp/cascade_numbers    # Cascade on BB72, from the records
python cascade/d13/surface/scripts/results_tables.py cascade/d13/out        # Cascade at d = 13, from the records
cd evaluation && python -m timing.composite && cd ..   # every registered surface composite and speed cell
cd surface_v2
python -m convlines.beacon check --beacon surf_v2/beacon.json                 # SURF-v2's beacon record, offline
python -m convlines.testset seeds --set sv2test --approval surf_v2/beacon.json            # its 700 test seeds
python -m convlines.testset regen --set sv2test --d 9 --chunks 0 --out shots/sv2test      # one chunk against its sha256
```

Rebuilding chunks bit for bit needs an x86-64 CPU with AVX2 and the pinned Stim and numpy. SURF-v2's analysis
(`python -m surf_v2.analysis`, `README.md` step 6) re-runs on the per-shot reads and timing records, which the authors
archive; `surface_v2/surf_v2/analysis/analysis.json` lists each with its sha256, and the analysis refuses any input that
does not match the freeze.

Re-reading the decoders needs an NVIDIA GPU (sm_90, such as an H100 or H200, for the d = 7 kernel; one RTX PRO 6000
for Cascade), NVIDIA Ising-Decoding at commit 3ce4379 for Ising, and CPUs for Relay-BP and ldpc (`README.md`,
Hardware). Re-timing needs the hardware of 2.5: an H100 and an exclusive x86-64 CPU node for the surface code (`timing.deployed`,
`timing.gpu_term`, `timing.cpu_term` and `timing.ising_trt`, as `results/surface/SURFACE.md` section 4 describes), an
H100 and an idle Apple M4 for the BB codes, and one RTX PRO 6000 for Cascade. The documents in the table at the top give
the full rules and every command, with the plans `cascade/bb72/cascade_compare/PHASE_C_PLAN.md`,
`cascade/bb72/cascade_compare/phase_d/PHASE_D1_PLAN.md` and `cascade/d13/SURFACE_PLAN.md` for the Cascade tests.
