# ConvLines: code, weights and records

This repository holds the artifacts of the paper "A Small Network Makes Quantum Decoders Up to 100× Faster and More
Accurate" by Yicheng Guang, Yuliang Xiong, Xun Gao and Yueqi Chen (University of Colorado Boulder): the training method,
the trained networks and the test method, with the records that check every table and figure.

[`METHODS.md`](METHODS.md) states how the accuracy and the speed were measured: the shots and their seeds, the decoders
and their settings, the statistics, what was timed, on which hardware and in which session.

## What ConvLines is

We did not invent a new strong decoder. ConvLines is a small network that sits in front of any strong decoder. It
decodes most shots itself and hands the least confident shots to that decoder. The pair is faster and more accurate
than the strong decoder alone. The paper shows this in front of each strong decoder it tests: correlated PyMatching on
the rotated surface code, Relay-BP on all detectors of two bivariate-bicycle codes, and two re-implementations of the
neural decoder Cascade, one for the BB72 code and one for the surface code at d = 13. The strong decoders are fallbacks
behind the network, not rivals.

A quantum error-correction decoder reads the detector bits of one shot (one run of the error-correction circuit) and
decides which logical flip happened. The decoders that fail least, here called strong decoders, are slow. ConvLines
decodes every shot and reports a confidence. A shot whose confidence falls below a frozen threshold goes to the strong
decoder, whose answer replaces the network's. Before any test, the threshold is set on calibration shots so that a
target fraction q of them is routed; on other shots the routed share differs slightly from q. BB72 at p = 0.2 %, set
for q = 0.25 %, routed 740 of SPEED-RT's 300,000 timing shots, 0.2467 % (`speed_rt/s2/bb/results/numbers.json`,
`pairs`; q is `q_cal` in `bb/bb72/ising_compare/data/bb/stage3/cells.json`). The network's output head predicts the
flips of many shifted copies of the logical operators and decides all logical qubits together.

## What the paper shows

The accuracy numbers come from the two newest registered tests, SURF-v2 for the surface code and RETEST for the BB
codes, Cascade and the output heads. Every speed number comes from SPEED-RT, the registered speed re-test ([How the
speed was measured](#how-the-speed-was-measured-speed-rt)). The table compares each pair with its own strong decoder
alone. Speed-ups are rounded down.

| strong decoder behind the network | code and noise | the pair's failures against the strong decoder alone | the strong decoder's time per shot over the pair's |
|---|---|---|---|
| correlated PyMatching (PM) | rotated surface code, d = 7 to 17, p = 0.6 % | 10 % to 31 % fewer (SURF-v2) | 2.2 to 10.3 (SPEED-RT stage 2) |
| Relay-BP on all detectors (full-DEM Relay-BP) | BB72 [[72,12,6]] at p = 0.3 % and 0.2 %, the gross code [[144,12,12]] at p = 0.4 % | 23 %, 29 % and 50 % fewer (RETEST) | 17.5, 124.9 and 8.9 (stage 2) |
| our Cascade for BB72 | BB72 under a fitted noise model, p = 0.2 % and 0.3 % | 1.1 % and 0.3 % fewer, each significant (RETEST) | 5.1 to 16.3 on an RTX PRO 6000, 3.1 to 9.2 on an H100 (stage 1) |
| our Cascade for the surface code | d = 13, p = 0.35 % and 0.6 % | 1.2 % to 1.3 % fewer at 0.35 %, each significant; 0.4 % fewer at 0.6 %, non-inferior but not significantly better (RETEST) | 3.6 to 7.6 on the RTX PRO 6000, 3.3 to 7.2 on the H100 (stage 1) |

The speed column comes from `speed_rt/s2/surface/results/table.md` (row S2, SURF-v2's networks),
`speed_rt/s2/bb/results/table.json` (row B5) and `speed_rt/cascade/results/verdicts.json` (claim H3-cascade, whose
twelve comparisons are rows C1, C2 and C8 of `table.json`). On the surface and BB codes a pair's time is a composite:
the network's time on an H100 GPU, plus the handoff, plus the strong decoder's time on the routed shots, measured on one
CPU core and converted to one core of NVIDIA's Grace CPU; the strong decoder alone is measured and converted the same
way. A Cascade time is measured end to end on one card, with the pair and Cascade alone in the same job, one shot at a
time and in batches.

On the surface code the registered rule chose q = 0 at d = 9, 11 and 13, so the network alone decided every shot there.
Against the other decoders of SURF-v2 (plain PM, NVIDIA's Ising pre-decoders Fast and Accurate followed by plain or
correlated PM, and two BP-OSD settings), the pair was more accurate at every distance. It was also faster than each of
them at every distance except against plain PM at d = 7 and 9, where the pair took 11.942 and 28.927 µs per shot and
plain PM 11.890 and 26.297 µs, so plain PM took 0.995 and 0.909 of the pair's time. With the deployed networks, which
SPEED-RT timed as well, the pair took 12.480 and 29.977 µs, and plain PM 0.952 and 0.877 of that. From d = 11 on, plain
PM took 1.25 to 2.97 times as long as the pair (1.20 to 1.77 with the deployed networks). These numbers are in
`speed_rt/s2/surface/results/table.md` (rows S1 and S4) and in the `ratio` column of `rows.csv` beside it.

On BB72 and the gross code the network runs in three modes (Table I): alone, balanced (with sector Relay-BP, which
decodes one type of check, as the fallback) and accurate (with full-DEM Relay-BP); the table above gives the accurate
mode. On the Grace basis, CUDA-Q's BP-OSD on the GPU, timed host to host, was the fastest decoder other than ours at all
three points, and the network alone took less time than it at each: 0.09072 against 0.8909 ms per shot on BB72 at p =
0.3 %, 0.08941 against 0.5396 ms at 0.2 %, and 0.2555 against 13.70 ms on the gross code, 0.019 of CUDA-Q's time. A pair
with a fallback is not always faster than CUDA-Q. On the gross code the balanced pair took 20.35 ms per shot against
CUDA-Q's 13.70 ms, while sector Relay-BP alone took 6.2 times as long as the pair. The accurate pair took 4.864 against
0.8909 ms on BB72 at p = 0.3 % and 427.5 against 13.70 ms on the gross code. The balanced pair on BB72 at both p and the
accurate pair at p = 0.2 % took less time than CUDA-Q. These numbers are rows B1 to B6 of
`speed_rt/s2/bb/results/table.json`.

In the Cascade comparisons, the pair's 99.9th-percentile latency in single-shot mode is longer than that of Cascade
alone on both cards, since a routed shot runs the network before Cascade. At d = 13 it was 486.2 and 499.5 µs against
404.3 and 422.4 µs on the RTX PRO 6000 (p = 0.6 % and 0.35 %), and 626.4 and 573.9 µs against 564.9 and 554.5 µs on the
H100. On BB72 the pair's was 7.1 % to 9.9 % above Cascade's on the RTX PRO 6000 and 8.4 % to 17.8 % above on the H100
(rows C8 and C10 of `speed_rt/cascade/results/table.json`).

On BB72, the same network with a conventional head that predicts each logical qubit separately failed 3.8 to 5.0 times
as often when it decided every shot itself (RETEST).

SURF-v2 also compared the pair with gap switching, our re-implementation of Decoder Switching, which uses plain PM as
the front end, routes by its complementary gap and falls back to correlated PM; the pair failed less often at every
distance (`surface_v2/surf_v2/GAPSW_REGISTRATION.md`). SPEED-RT timed gap switching on the CPU: 57.41 µs per shot at d =
7 up to 2,402.31 µs at d = 17 on the Grace basis, against the pair's 11.943 to 81.580 µs. This comparison is descriptive
and has no verdict (`speed_rt/s2/gapsw/results/gs_tables.md`).

Most tests were registered: the networks, thresholds, test sets and analyses were fixed before the test shots existed,
and each test set was drawn once and read once. The three newest registrations fixed in advance that the paper reports
their numbers, whatever they turned out to be, in place of the earlier ones (`RETEST.md` section 5,
`surface_v2/surf_v2/REGISTRATION.md` "The paper's rows", `speed_rt/SPEED_RT.md`). The original tests and the earlier
timing sessions stay in this repository with their plans and records.

## How the speed was measured: SPEED-RT

SPEED-RT timed again every speed number of the paper that this release can run, on fresh timing shots drawn from a
public randomness beacon after its rules were public (`speed_rt/SPEED_RT.md`). Nothing was re-chosen, re-run or re-tuned
because of a result, and a claim whose speed condition failed is reported as failing. SPEED-RT makes no
deployment-latency claim. By the authors' decision it left out a few values of the paper, which
[`METHODS.md`](METHODS.md) section 2.8 lists. Section 2 of `METHODS.md` describes the measurement in more detail, and
the registrations hold the full rules.

### Stages, freezes and beacons

| stage | what it timed | registration | freeze commit and GitHub's record of its push | beacon round | beacon commit | analysis commit | outputs |
|---|---|---|---|---|---|---|---|
| 1 | the Cascade comparisons, on an RTX PRO 6000 and on an H100 | `speed_rt/SPEED_RT.md` section 1 | `73ca9f9`: `push` to `main` from `ebbb6b6`, 2026-10-13T21:48:15Z, record 46399062301 | 6549923, released 2026-10-13T21:58:30Z | `6609d14` (`speed_rt/cascade/beacon.json`, `seeds.json`) | `56e2023`; correction `2ce2e49` | `speed_rt/cascade/results/` |
| 2 | every other speed number: the surface comparisons with Ising, BB72 and the gross code, Decoder Switching | `speed_rt/STAGE2_common.md`, `STAGE2_surface.md`, `STAGE2_bb.md`, `STAGE2_gapsw.md` | `7988e04`: `push` to `main` from `56e2023`, 2026-10-14T04:00:56Z, record 46424448042 | 6550668, released 2026-10-14T04:11:00Z | `ea1525c` (`speed_rt/testsets/beacon_surf.json` and the shot manifests) | `e0a9377`; amendment `54d9eec` | `speed_rt/s2/surface/results/`, `speed_rt/s2/bb/results/`, `speed_rt/s2/gapsw/results/` |

Each stage froze its registration, code and timing sets in a freeze commit and pushed it to GitHub before any of its
timing shots existed. Its beacon is the first drand mainnet round at least 600 s after GitHub's earliest activity record
of that push, and every timing chunk's seed derives from the round's randomness (`speed_rt/sets.py`), so nobody could
know a timing shot before the round. The beacon record, which keeps GitHub's activity list as fetched, and the shot
manifests were committed and pushed as the beacon commit, and nothing of the stage was timed before that commit was
public. For stage 1, GitHub recorded the push of `6609d14` at 2026-10-13T22:01:23Z (record 46400110294, in the activity
list that `speed_rt/testsets/beacon_surf.json` keeps), and the first timed process started at 22:02:03Z
(`speed_rt/cascade/records/H100/R1/attempts.json`). Stage 2's record also sits, byte for byte, in
`speed_rt/bb/beacon.json` and `speed_rt/gapsw/beacon.json`. After the timing, each stage's frozen analysis ran twice on
the records with byte-identical outputs, and the records and outputs were committed together.

### What a time is

- Surface code and BB codes: our composite time per shot is the network's GPU term on the H100 at batch one, with the
  input already on the GPU, plus the handoff, plus the fallback's CPU time on the routed shots (the routed fraction
  times the mean time of a routed shot) on the Grace basis. The GPU and CPU terms come from two machines, as in the
  paper's earlier composites, so the stage-2 composites of our pairs, and Ising's with its PM share, are not measured
  end to end. CUDA-Q's BP-OSD is measured end to end: host to host on the GPU pod, its OSD's host work included.
- NVIDIA's Ising paper (arXiv 2604.12841 v1, Section VI.3) estimates the transfer of syndrome data between the
  pre-decoder on the GB300 GPU and PyMatching on the Grace CPU with NVIDIA's NVQLink architecture, finds it negligible
  next to both runtimes and ignores it, and times its pre-decoders with host-device transfers disabled. We use the same
  setting and keep our host-side handoff in every composite. On the surface code the handoff is measured: at d = 7 the
  resident kernel's host round trip minus its device time, at d = 9 to 17 the larger of 1.1 µs and the router's
  estimate, so whatever transfer it holds stays in. On the BB codes it is the registered host-side allowance of 1.1 µs
  per shot, and the measured pinned-memory round trip of one shot, 16.9 to 19.5 µs, is reported beside every composite
  and left out (`speed_rt/s2/bb/results/numbers.json`, `pairs`). Adding the largest, 19.5 µs, to every BB composite
  would change no verdict.
- Baselines: plain and correlated PM, the PM share after Ising, Relay-BP and ldpc's BP-OSD are timed on the CPU and put
  on the Grace basis. Ising's GPU time is the fastest accepted repetition of any validated TensorRT plan on the pod: 12
  plans per distance, six per model (four registered builds, FP16 and FP8 at optimisation levels 3 and 5, and two built
  as NVIDIA's code builds them, FP16 and FP8), each checked against NVIDIA's PyTorch numerics on validation shots before
  it was timed, then timed at batch one in CUDA graphs and with NVIDIA's own `trtexec` command
  (`speed_rt/STAGE2_surface.md` section 1.13). CUDA-Q's BP-OSD is timed at batch one, host to host on the pod, and is
  not converted.
- Cascade: end to end and host to host, from the shot's detector bits in pinned host memory to the decision on the host,
  routed shots' fallbacks included, with both sides of every comparison on the same card in the same job. Single mode
  times 20,000 shots one at a time after 200 untimed ones; batched mode decodes 500,000 shots with the networks at batch
  4,096.

### Hosts

- GPU parts: rented NVIDIA H100 80GB HBM3 SXM pods (RunPod, secure cloud), each pre-screened before anything was timed
  on it: five checks on the idle GPU, CPU steal at most 1 % over 5 s and a quota of at least 20 CPUs
  (`speed_rt/gpu_prescreen.py`, `SPEED_RT.md` section 1.5). Stage 2 timed on one pod per role, with host CPUs Intel Xeon
  Platinum 8470 (surface code at d = 7, 9 and 11), AMD EPYC 9554 (d = 13, 15 and 17) and Intel Xeon Platinum 8470 (BB);
  a pod timed only GPU pipelines, one process at a time: the networks, Ising's plans, and CUDA-Q's BP-OSD with the host
  work of its OSD. Every slot of the three timing pods was accepted at its first attempt
  (`speed_rt/s2/surface/records/low/session/` and `high/session/`, `META.json` and `attempts.json`;
  `speed_rt/s2/bb/records/gpu/`, `prescreen/prescreen.json` and `attempts.json`).
- Cascade: both an NVIDIA RTX PRO 6000 Blackwell Server Edition (MIG off, host AMD EPYC 9555) and an H100 pod (host
  Intel Xeon Platinum 8480+). On the H100, 78 of the 153 counted windows ran under the workload's own power cap at the
  card's default 700 W limit, with SM clocks down to 0.682 of the maximum; the registration counts this as the card's
  speed under the workload. The RTX PRO 6000 had none (`speed_rt/cascade/results/numbers.json`, `clock`). The host
  CPUs are named in the records under `speed_rt/cascade/records/`.
- CPU parts: one Linux workstation, an AMD Ryzen 9 9950X (16 cores, 32 threads; Ubuntu 24.04, Python 3.12.3, numpy
  2.2.6, PyMatching 2.4.0, relay-bp 0.2.2, ldpc 2.4.1), every process single-threaded. The authors set it for the
  session as `speed_rt/STAGE2_common.md` C1 registers, and `speed_rt/s2/gapsw/records/prescreen/topo.json` records the
  settings:
  - governor `performance` on every online CPU (amd-pstate-epp, energy-performance preference `performance`), boost on;
  - 8 timed processes at most, each pinned to its own physical core of one CCD (core complex die; CPUs 0-7) with the
    core's SMT sibling (CPUs 16-23) taken offline, and every other process confined to the other CCD (CPUs 8-15 and
    24-31);
  - everything a timed process reads (interpreter, packages, code, circuits, shots, routing masks) in a tmpfs, and a
    window fails if the harness's own block-read counter grows;
  - before and after every window, each slot times a reference segment (plain PM at d = 9 on a fixed chunk), and the
    window passes only when every reference value is within 1.10 times the session's reference (17.32 µs per shot),
    each segment's largest value is within 6 % of its smallest, nothing was read from disk and every process exited 0.

  All 46 windows passed, and all 290 processes were accepted at their first attempt
  (`speed_rt/s2/bb/records/cpu/session.json`, `speed_rt/s2/README.md`).

### The Grace basis

NVIDIA's Ising paper times PyMatching on one core of an NVIDIA Grace CPU. We have no GH200 or GB300, so every CPU time
of stage 2 is converted to a Grace core by factors fitted over all 18 of NVIDIA's published points (Ising paper, arXiv
2604.12841 v1, Tables VII and IX: plain PM at d = 13, 21 and 31 and correlated PM at d = 5, 9, 13, 17, 21 and 31, each
at p = 0.3 % and 0.6 %), which the workstation timed in the same session on the circuits that NVIDIA's code builds for
those points. The geometric mean of NVIDIA's time over ours is 1.4543 for plain PM and 1.5383 for correlated PM. The two
agree within their spread, so the common factor 1.5098 converts every CPU time of the stage, Relay-BP and BP-OSD
included (`speed_rt/s2/surface/results/fit_s2.json`, `speed_rt/grace_fit.py`). Point by point the ratio ranges from 0.90
to 2.23, so a Grace-basis time is an estimate. Every raw time is published beside its converted value, and every verdict
is also computed on the raw basis, which decides nothing. On the raw basis 45 of the 48 surface comparisons hold for
each network set, plain PM being faster at d = 7, 9 and 11 (`speed_rt/s2/surface/results/table.md`, row S4x). Against
CUDA-Q, which is not converted, the conversion works against the pair: on the raw basis the gross code's balanced pair
takes 13.57 ms, just under CUDA-Q's 13.70 ms (`speed_rt/s2/bb/results/numbers.json`, `sensitivity_raw_xeon`, a key named
after an earlier host that holds the raw basis). GPU terms, handoffs and CUDA-Q's host-to-host times are not converted.

### Statistics

- Stage 2: our side takes the slowest of its fresh processes (three for every CPU term, five for the GPU terms of the
  surface networks and of BB72's, three for the gross code's) and every baseline the fastest of its three, with no
  medians (`STAGE2_common.md` C6). Ising's GPU term is the exception: it is the smallest accepted repetition across all
  validated plans and both timing methods (`STAGE2_surface.md` 1.13; [`METHODS.md`](METHODS.md) section 2.5), which
  favours Ising. The Grace reference points take the slowest of three, which gives the smaller factor.
- Stage 1: the slowest of three on both sides, as the paper's Cascade method did. Sensitivity rows repeat every Cascade
  comparison with Cascade's fastest process and decide nothing; the lowest speed-up is then 3.63 on the RTX PRO 6000
  and 3.17 on the H100 (`speed_rt/cascade/results/table.json`).
- "X takes less time than Y" holds when X's value is strictly below Y's. There is no statistical test. Ranges and
  ratios of times in the prose are rounded against us: our times, and our time as a fraction of another decoder's, up;
  the other decoders' times, and speed-ups (another decoder's time over ours, also when below 1), down.

### The re-measured times

Surface code: composite time per shot in µs on the Grace basis, with BP-OSD as timed on the pod
(`speed_rt/s2/surface/results/table.md`, row S1).

| d | the pair, SURF-v2's networks | the pair, deployed networks | plain PM | correlated PM | Ising-Fast + PM | Ising-Accurate + PM | BP-OSD, min-LLR | BP-OSD, BP 60 |
|---|---|---|---|---|---|---|---|---|
| 7 | 11.942 | 12.480 | 11.890 | 27.598 | 47.144 | 58.995 | 2183.5 | 5950.8 |
| 9 | 28.927 | 29.977 | 26.297 | 63.916 | 57.204 | 67.223 | 8655.4 | 19033.8 |
| 11 | 41.482 | 43.521 | 52.234 | 137.889 | 69.350 | 78.592 | 31365.0 | 52051.1 |
| 13 | 51.245 | 60.313 | 91.704 | 253.071 | 92.311 | 100.809 | 87743.2 | 130648.5 |
| 15 | 59.552 | 90.099 | 160.091 | 562.655 | 123.821 | 128.178 | 193252.4 | 287949.2 |
| 17 | 81.579 | 141.406 | 242.527 | 846.008 | 173.116 | 172.988 | 478767.2 | 647847.1 |

Each network set has 48 registered comparisons (rows S4 and S10): these six baselines and two controls, Ising-Fast and
Ising-Accurate followed by correlated PM (51.067 to 354.429 µs and 61.907 to 321.348 µs), at six distances. For each
set 46 hold; the two that do not are plain PM at d = 7 and 9.

BB codes: time per shot in ms, with CPU times on the Grace basis and CUDA-Q as timed on the pod
(`speed_rt/s2/bb/results/table.json`, rows B1 to B3).

| point | network alone | balanced pair | accurate pair | sector Relay-BP | full-DEM Relay-BP | ldpc BP-OSD (CPU) | CUDA-Q BP-OSD (GPU) |
|---|---|---|---|---|---|---|---|
| BB72, p = 0.3 % | 0.09072 | 0.4148 | 4.864 | 3.302 | 85.37 | 1.818 | 0.8909 |
| BB72, p = 0.2 % | 0.08941 | 0.1169 | 0.4224 | 1.987 | 52.78 | 0.6908 | 0.5396 |
| gross code, p = 0.4 % | 0.2555 | 20.35 | 427.5 | 127.6 | 3811 | 114.0 | 13.70 |

On BB72 the network alone at p = 0.3 % is the first of the two networks; the second, which every BB72 pair uses, took
0.08941 ms alone. The Cascade times are in `speed_rt/cascade/results/table.json`, per card, and with every repetition in
`numbers.json`.

### Shot counts

To fit the CPU session into one evening, the authors cut the shot counts of the slowest CPU items before stage 2's
freeze. Each cut item times the first N shots, or the first N routed shots, of its set in shot order (`STAGE2_bb.md`
1.3, `STAGE2_gapsw.md` 1.3):

| item | count before the cut | count timed |
|---|---|---|
| gross code: full-DEM Relay-BP | 300 shots | 75 shots |
| gross code: ldpc BP-OSD | 1,000 shots | 500 shots |
| gross code: the balanced pair's CPU term (sector Relay-BP) | 1,000 routed shots | 300 routed shots |
| gross code: the accurate pair's CPU term (full-DEM Relay-BP) | 100 routed shots | 25 routed shots |
| Decoder Switching, each distance | 200,000 shots (20 chunks) | 50,000 shots (5 chunks) |

### Deviations

- `54d9eec`, an amendment after the timing: line 208 of `speed_rt/gapsw/analysis_gs.py`, Decoder Switching's analysis,
  checked for 20 chunks per distance. The draw had been cut to 5 before the freeze and this check was not updated, so
  the frozen code refused the registered draw. The amended line takes the count from the frozen sets file. No other line
  depends on the old count, and no Decoder Switching number existed before the amendment (`speed_rt/s2/README.md`).
- `2ce2e49`, a correction of stage 1's publication: `speed_rt/cascade/records/H100/R4/derived_lock.json` and
  `session.json` are published byte for byte as the H100 job wrote them, because the frozen analysis checks their
  sha256. Their sanitized copies in `56e2023` failed that check (`speed_rt/cascade/results/README.md`).

### How to verify

From the repository root, with Python 3.12 (numpy for Decoder Switching; the other steps need only the standard
library), these commands check both beacon records offline, re-derive stage 1's seeds, and rerun each frozen analysis
on the published records. Every `cmp` and `diff` must print nothing.

```bash
export PYTHONPATH=$PWD:$PWD/surface_v2
O=$(mktemp -d)                                    # a new, empty directory for the reruns
# the two beacon records, checked offline, and stage 1's seeds
python3 -m speed_rt.beacon check --beacon speed_rt/cascade/beacon.json --commit 73ca9f9fcef77a3e5c8c170095d6d8c2fe14a5ae
python3 -m speed_rt.beacon check --beacon speed_rt/testsets/beacon_surf.json --commit 7988e04e74153d1234aca4b8889bd176f73e7bac
python3 speed_rt/cascade/srtlib.py seeds --beacon speed_rt/cascade/beacon.json --out $O/seeds.json
cmp $O/seeds.json speed_rt/cascade/seeds.json
# stage 1: the Cascade comparisons
mkdir $O/cas
python3 speed_rt/cascade/analysis_cas.py --records speed_rt/cascade/records --out $O/cas/numbers.json \
    --table $O/cas/table.json --verdicts $O/cas/verdicts.json > $O/cas/stdout.txt 2>&1
diff -r -x README.md $O/cas speed_rt/cascade/results
# stage 2: the surface comparisons with Ising; this also writes the stage's Grace fit, fit_s2.json
python3 -m speed_rt.analysis_surf --records speed_rt/s2/surface/records --out $O/surf
diff -r $O/surf speed_rt/s2/surface/results
# stage 2: BB72 and the gross code
mkdir $O/bb
python3 speed_rt/bb/analysis_bb.py --records speed_rt/s2/bb/records --out $O/bb/numbers.json \
    --table $O/bb/table.json > $O/bb/stdout.txt 2>&1
diff -r $O/bb speed_rt/s2/bb/results
# stage 2: Decoder Switching, with line 208 as amended in 54d9eec
python3 -m speed_rt.gapsw.analysis_gs --session speed_rt/s2/gapsw/records/session \
    --prescreen speed_rt/s2/gapsw/records/prescreen --masks speed_rt/s2/gapsw/records/masks \
    --manifest speed_rt/gapsw/srt-gs.json --fit speed_rt/s2/surface/results/fit_s2.json \
    --beacon speed_rt/gapsw/beacon.json \
    --commits 7988e04e74153d1234aca4b8889bd176f73e7bac ea1525c739613f5bf256a98c15ac54c7cc66051d --out $O/gs
diff -r $O/gs speed_rt/s2/gapsw/results
```

Each analysis checks the records it reads against its freeze (the sha256 of the files the freeze names, the shots'
manifests and seeds, each session's acceptance) before it computes, and stage 2's analyses re-derive the Grace fit from
its 18 points. A value whose slot is missing is withheld, never replaced. `speed_rt/cascade/results/README.md` and
`speed_rt/s2/README.md` say how the records were sanitized for publication; the files whose sha256 the analyses check
are published byte for byte.

## Registered tests and how to verify them

RETEST and SURF-v2, the two accuracy tests, each fixed everything their results depend on (the networks, their
thresholds, the rule that makes the test seeds, and the analysis) in a freeze commit and pushed it to GitHub before any
test shot existed. GitHub records the push in the repository's activity list. The test seeds come from the first round
of the drand public randomness beacon (mainnet, one round every 30 s) at least 600 s after that push, so nobody could
know the test shots when the freeze was made. Each test set was then drawn once, read once by every decoder and analysed
once by the frozen analysis.

RETEST (`RETEST.md`) re-ran the BB tests of Table I, both Cascade comparisons and the output-head comparison on fresh
shots, with the networks, thresholds, decoders and analyses frozen for their original runs; it timed nothing again, and
SPEED-RT re-timed its speed numbers.
SURF-v2 (`surface_v2/surf_v2/REGISTRATION.md`) tested the surface-code networks retrained without line supervision
(trained on the observable's flip alone) at d = 7 to 17 against eight baselines, with the deployed networks read
beside them. It also registered a secondary comparison with gap switching
(`surface_v2/surf_v2/GAPSW_REGISTRATION.md`), in which the pair failed less often than gap switching at every distance.

| test | freeze commit | GitHub push record of the freeze | beacon round | analysis commit | results |
|---|---|---|---|---|---|
| RETEST | `dd332c5` | `push` to `main` from `ab5cf49` to `dd332c5`, 2026-10-13T02:34:48Z, record 46295757992 | 6547616, released 2026-10-13T02:45:00Z; the record, the seeds and the draw manifests are in `390f2da` | one per test: `7c1aae0`, `d92e011`, `a3bd87d`, `4a85d65`, `87a1742`, `301e9e7`; the summary `3cc1258` | `RETEST_RESULTS.md`; each test's records and analysis outputs in `retest/run/` |
| SURF-v2 | `f94f4ed` | `branch_creation` of `surf-v2` at `f94f4ed`, 2026-10-13T11:49:59Z, record 46339453746 | 6548726, released 2026-10-13T12:00:00Z; the record and the test manifest are in `ce8780e` | `88d14de`; gap switching `3b41af7` | `surface_v2/surf_v2/analysis/tables.md`, `surface_v2/surf_v2/gapsw/analysis/gapsw_tables.md` |

- The push records are GitHub's JSON as its API returned it: `retest/run/github_push_activity.json`, and
  `surface_v2/surf_v2/beacon.json`, which keeps the record together with the whole activity list.
- `RETEST.md` as frozen is `git show dd332c5:RETEST.md`. The summary commit `3cc1258` recorded in it the run's one
  deviation, the job shape of the gross-code reads, and corrected or completed several commands and descriptions;
  `git diff dd332c5 3cc1258 -- RETEST.md` shows every change. After `f94f4ed`, SURF-v2's commits only add files.
- `surface_v2/surf_v2/REGISTRATION.md` names the branch `surf-v2`, to which the freeze was pushed (the
  `branch_creation` record above). That branch still exists, and `main` now holds the same commits.
- The plans of the earlier registered tests are listed in the Table IX row of [Tables and figures](#tables-and-figures).
- SPEED-RT follows the same beacon rule for its timing shots. Its commits, push records and rounds are in
  [Stages, freezes and beacons](#stages-freezes-and-beacons), and steps 1 to 3 below apply to them as well;
  [How to verify](#how-to-verify) checks its two beacon records offline.

### Checking a registration

The steps check SURF-v2 from `surface_v2/` (paths are relative to it) in the pinned environment of
[Installation](#installation), with the GitHub CLI `gh` and network access to drand. Each step ends with the same check
for RETEST, run from the repository root.

1. Read the push record.

   ```bash
   cd surface_v2
   gh api 'repos/a85tract/Convlines_Public/activity?per_page=100' > activity.json    # one page: under 100 records
   jq '.[] | select(.after == "f94f4edb843b4a0b4762c556d79782cf2a2167e2") | {id, activity_type, ref, timestamp}' activity.json
   ```

   The earliest record whose `after` is the freeze commit gives the push time T, so a later push of the same commit
   cannot move it: record 46339453746, `branch_creation` of `refs/heads/surf-v2`, 2026-10-13T11:49:59Z, the same
   record as `push_activity` in `surf_v2/beacon.json`. RETEST: select `dd332c5082f44c41d486327400d90c73c77c926a`, which
   gives record 46295757992, `push` to `refs/heads/main`, 2026-10-13T02:34:48Z.

2. Derive the drand round: the first round whose time, 1595431050 + 30 (round − 1) in UNIX seconds, is at least
   T + 600. T = 1791892199 (`date -u -d 2026-10-13T11:49:59Z +%s`) gives round 6548726, released at 1791892800
   (12:00:00 UTC). RETEST: T = 1791858888 gives round 6547616, released at 1791859500 (02:45:00 UTC).
   `convlines.beacon.beacon_round` computes the same.

3. Check the round against drand's public API. `convlines.beacon fetch` derives T and the round from your copy of the
   activity list, fetches the round from four drand endpoints, requires at least two answers that agree, checks them
   and writes a new record, whose round, signature and randomness must equal the committed record's:

   ```bash
   python -m convlines.beacon fetch --commit f94f4edb843b4a0b4762c556d79782cf2a2167e2 --activity activity.json \
       --push-source "verification" --out beacon_check.json
   jq -c '{round, randomness, signature}' surf_v2/beacon.json beacon_check.json
   ```

   The randomness is the SHA-256 of the round's signature, `930cdbaf...` for round 6548726, and drand serves the round
   without our code at
   `https://api.drand.sh/8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce/public/6548726`. RETEST:
   `python retest/beacon.py --push-time 2026-10-13T02:34:48Z --commit dd332c5082f44c41d486327400d90c73c77c926a --out
   rt_beacon_check.json` fetches and checks round 6547616, to compare with `retest/run/beacon.json` (randomness
   `299266d1...`).

4. Re-derive the seeds and check them against the test manifest.

   ```bash
   python -m convlines.beacon check --beacon surf_v2/beacon.json
   python -m convlines.testset seeds --set sv2test --approval surf_v2/beacon.json
   ```

   The first checks the committed record offline: the chain, T as the earliest activity record of the freeze commit,
   the round, the endpoints' agreement, and randomness = SHA-256(signature). The second checks that
   `testsets/sv2test.json` names this record by its sha256 and re-derives all its seeds, 200 chunks of 10,000 shots at
   d = 7 and 100 at each other distance ("700 of 700 seeds re-derived"). RETEST: `python retest/seeds.py --verify
   retest/run/beacon.json retest/run/seeds.json` recomputes every set's seeds from the beacon, `retest/tags.json` and
   the historical seeds and prints `"equal": true`, and it prints the same with step 3's `rt_beacon_check.json` in
   place of the committed record. The draw manifests `retest/run/bb/manifest_bb72.csv`, `manifest_gross.csv` and
   `retest/run/head/rows.csv` list each chunk's seed and sha256.

5. Regenerate a test chunk and compare its sha256.

   ```bash
   python -m convlines.testset regen --set sv2test --d 9 --chunks 0 --out shots/sv2test
   ```

   It prints `d9_p0.006_X_sv2test_000.npz: seed 5867115360100734495 sha256 8c7f5d84... == manifest`. Bit-for-bit
   regeneration needs Stim 1.16.0 and numpy 2.2.6 on an x86-64 CPU with AVX2. RETEST's BB chunks are rebuilt through its
   shim, as `retest/jobs/bb_draw.sbatch` does: `retest/shim/rt_bb.py bb.shots ...` with `RT_BEACON` and `RT_SEEDS` set
   to `retest/run/beacon.json` and `retest/run/seeds.json`; the BB circuits come from Relay-BP's repository
   (Installation).

6. Rerun the analysis and compare it byte for byte.

   ```bash
   python -m surf_v2.analysis --freeze surf_v2/freeze.json --reads READS... --shots shots/sv2test \
       --cal-shots shots/sv2cal --sessions SESSION --cpu CPU --out run1
   diff -r run1 surf_v2/analysis
   ```

   Nothing in the outputs depends on the time, the host or the output directory, so `diff` must report only
   `sessions.log`, a one-line note of the timing session's pod and GPU that the analysis does not write. The test and
   calibration sets are rebuilt with `regen` (`--set sv2cal` for the calibration set). The other inputs (every decoder's
   per-shot reads, the timing session and the CPU-term records) are archived by the authors and are not in this
   repository; `surf_v2/analysis/analysis.json` lists each of them with its sha256 under `inputs`, and
   `surf_v2/args/alpine_analysis.txt` holds the command as it ran. The gap-switching analysis runs the same way
   (`surf_v2/args/alpine_gapsw_analysis.txt`), and `RETEST.md` section 3 gives each RETEST analysis command.

### NVIDIA's Ising decoders

The tables' Ising M1 and Ising M4 are NVIDIA's public pre-decoders Ising-Decoder-SurfaceCode-1-Fast (receptive field 9)
and Ising-Decoder-SurfaceCode-1-Accurate (receptive field 13). Each is followed by plain or by correlated matching on
the detectors it leaves, which gives the rows Ising M1 / M4 + PM and Ising M1 / M4 + correlated PM.

- Accuracy comes from Ising-Decoding's official PyTorch numerics at commit 3ce4379: NVIDIA's own inference module, with
  fp32 weights and TF32 as NVIDIA's runner enables it (`surface_v2/convlines/opponents.py`). SURF-v2 read all four Ising
  rows, with plain and with correlated PM, on every test shot (`surface_v2/surf_v2/args/pod_e1_test.txt`); their
  failures are in `surface_v2/surf_v2/analysis/tables.md` and `family_A.csv`.
- Speed comes from SPEED-RT stage 2 (`speed_rt/STAGE2_surface.md` section 1.13). For each distance 12 TensorRT 10.14.1
  plans were built, six per model: FP16 and FP8 at optimisation levels 3 and 5, and NVIDIA's own build of each
  precision. The pod for d = 7, 9 and 11 built its plans after stage 2's freeze and then timed them. For d = 13, 15 and
  17 an earlier H100 pod built the plans before the freeze, from tree files that the frozen analysis checks against the
  frozen tree, saved every build output with its sha256 and was released. The timing pod, with the same GPU model and
  the same TensorRT, CUDA and torch versions, loaded the 134 saved files (the 133 outputs that
  `speed_rt/s2/surface/records/high/session/BUILD_INPUTS.json` lists with their sha256, and that file) after checking
  each file's sha256, and re-ran every plan's check, as section 1.13.7 registers. Every plan was checked on validation
  shots against the PyTorch numerics above before it was timed, an FP8 plan only if every convolution the recipe
  quantized ran an FP8 kernel. The plans were timed at batch 1 on the first timing shot, with CUDA-graph replays and
  with NVIDIA's `trtexec` command, and the fastest accepted repetition of any validated plan counts as Ising's GPU time.
  The composite adds the PM share on the detectors Ising leaves, timed on the Linux workstation and put on the Grace
  basis (rows S1 and S10 of `speed_rt/s2/surface/results/table.md`). SURF-v2's own speed comparisons used the earlier
  registered times of `results/surface/SURFACE.md` (section 4; `surface_v2/surf_v2/config.json`, `registered_times_us`),
  which SPEED-RT supersedes.

An Ising row's accuracy and its time thus come from two numerics of the same model, and no single run measured both.
The earlier surface-code tests also read every timed TensorRT plan for accuracy (`results/surface/s*_test_eval.csv`,
`r17_test_eval.csv`).

## Layout

| path | contents |
|---|---|
| `evaluation/` | the surface-code decoder (`convlines/`: network, reference decoder, routing, opponents, statistics, evaluation), the six verdict circuits (`circuits/`), the test-set manifests (`testsets/`), the deployed routing masks (`routing/`), the deployed GPU engines and timing harnesses (`timing/`), and `scripts/make_circuits.py` |
| `weights/` | the 38 released networks, `networks.csv` (name, file, parameter count), `MANIFEST.sha256` and `check_weights.py` |
| `training/` | `convlines_train/` (train and tune a front end for any circuit), the frozen programs that trained the released networks (`surface/`, `bb/`), `TRAINING.md`, `LINEAGE.md` (every network's stages) and the smoke test |
| `results/` | the surface-code records (`surface/`, with `SURFACE.md`), R17, the d = 17 re-registration (`r17/`), the analysis of the discordant shots (`discordant/`), the paper's figure scripts and figures (`figures/`), and a reproduction on a rented H100 (`reproduction/`) |
| `bb/` | the BB72 test (`bb72/`), the gross-code tests (`gross/`, `gross_stage4/`), each a frozen code tree with its plans and records, and `BB.md` |
| `cascade/` | ConvLines in front of our Cascade on BB72 (`bb72/`, with `CASCADE.md`) and on the surface code at d = 13 (`d13/`) |
| `head/` | the output-head comparison: its code, read-out rules, training records and results (`A1_RESULT.md`) |
| `RETEST.md`, `RETEST_RESULTS.md`, `retest/` | RETEST, the registered re-test on fresh shots: its plan, every re-tested number beside its original, and its beacon and seed code, shims, jobs, checks before the freeze (`retest/replay/`) and records (`retest/run/`) |
| `surface_v2/` | SURF-v2, the registered test of the retrained surface-code networks: the evaluation package it ran from, with the new networks in `weights/`, and `surf_v2/` with the registrations, their code, the freeze and beacon records and both analyses |
| `speed_rt/` | SPEED-RT, the registered speed re-test: its registrations (`SPEED_RT.md`, `STAGE2_*.md`), its timing code, sets, beacon records and manifests, stage 1's records and outputs (`cascade/records/`, `cascade/results/`) and stage 2's (`s2/<family>/records/`, `s2/<family>/results/`) |
| `SANITIZE.md` | what was replaced before publication |

The frozen trees under `bb/`, `cascade/`, `head/` and `training/` are the programs as they ran, so their own documents
use the names of the research project: study 2 and 3 are the surface-code tests at d = 7-13 and 15 (test sets T2 and
T3), study 4 and R17 the two at d = 17 (T5 and T6), stage 3 and 4 are the BB tests, phase C and D1 the two Cascade tests on BB72, W5 the
comparison at d = 13, A1 the output-head comparison and X2 the analysis of the discordant shots. Table XI of the paper
maps every descriptive name to these short names, and `weights/networks.csv` does the same for the networks.

## Installation

Linux x86-64 with Python 3.12.14:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126    # or .../whl/cu128 (see below)
pip install -r requirements.txt
```

This installs numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, torch 2.11.0, safetensors 0.8.0 and matplotlib
3.11.2, which run the surface-code decoder, the weight check, the figures and the record checks. The surface-code
tests ran torch 2.11.0+cu126 (with Triton 3.6.0); the BB, Cascade and output-head work ran torch 2.11.0+cu128, which
NVIDIA Blackwell GPUs (RTX PRO 6000) need. A CPU wheel of torch runs everything that needs no GPU.

On a GPU, Triton (the deployed surface engine, the BB engines, and `torch.compile` in the training programs) builds a
small C helper at first use, which needs gcc and Python 3.12's development headers (`python3.12-dev` or
`python3.12-devel`). Where the headers are missing, point `C_INCLUDE_PATH` at a copy of them.

| part | also needs |
|---|---|
| BB codes and the Cascade comparison on BB72 | `pip install relay-bp==0.2.2`, and Relay-BP's repository at revision d185194 for its circuits: `QND_RELAY=/path/to/relay-bp bash training/bb/qldpc_neural/scripts/fetch_relay.sh`, then keep `QND_RELAY` set |
| the surface-code opponents Ising-Fast and Ising-Accurate | NVIDIA Ising-Decoding at commit 3ce4379 (tag v0.1.0) and its inference requirements (`results/surface/SURFACE.md`, section 3) |
| the BP-OSD opponent | `cudaq-qec-cu12==0.8.0` on an NVIDIA GPU, in a separate environment |
| Ising's TensorRT plans | the TensorRT stack listed in `results/surface/SURFACE.md`, section 4 |
| the frozen surface-code training program and the output-head comparison's surface arms | NVIDIA Ising-Decoding's `code/` at 3ce4379 (`ISING_DECODING_DIR` or `ICMP_NV`), which builds the circuits and the detector geometry |
| training with BP-OSD as the strong decoder | `pip install ldpc==2.4.1` |

Python packages are imported from the tree they belong to. The surface-code commands run from `evaluation/` (which
makes `convlines` and `timing` importable) and SURF-v2's from `surface_v2/`, the training package needs
`export PYTHONPATH=$PWD/evaluation:$PWD/training` from the repository root, RETEST's tools run from the repository root,
and each frozen tree runs from its `ising_compare/` directory with `PYTHONPATH=.:../qldpc_neural`, as its document says.

## Hardware

| work | hardware | time |
|---|---|---|
| the weight check, the record checks, the figures | any CPU | seconds to minutes |
| rebuilding a surface-code test set bit for bit | an x86-64 CPU with AVX2 and the pinned stim and numpy | minutes per distance; 0.4-0.6 GB per set |
| the surface-code accuracy of ours against matching | a CPU, or one CUDA GPU for speed | minutes to hours per distance |
| the deployed engines and the earlier tests' timings | an NVIDIA H100 (Hopper; the d = 7 kernel is built for sm_90a) and one exclusive CPU node | hours |
| rerunning SPEED-RT's analyses on its published records ([How to verify](#how-to-verify)) | a Linux x86-64 machine with Python 3.12, where the reruns were checked; numpy for Decoder Switching | seconds |
| re-timing as SPEED-RT did | H100 80GB HBM3 SXM machines for the GPU terms, Ising's plans and CUDA-Q; one RTX PRO 6000 and one H100 for Cascade; for the CPU terms, a Linux machine whose timed cores run nothing else, their SMT siblings offline (SPEED-RT used 8 cores of an AMD Ryzen 9 9950X) | hours |
| reading the BB networks on test shots | an NVIDIA GPU; the commit guard of each BB tree checks the frozen engine signature (`bb/BB.md`) | minutes to hours |
| the Cascade comparison's reads on BB72 and at d = 13 | one NVIDIA RTX PRO 6000; the evaluation code refuses other cards (SPEED-RT's launchers also admit the H100, for timing) | hours |
| the output-head comparison's reads | one GPU of any model (H200 and RTX PRO 6000 MIG slices were used) | under an hour per code |
| the training smoke test | 16 CPU cores | about 5 minutes |
| retraining the released networks | CUDA GPUs | about 90 GPU-hours for the surface chain, about 3 per BB72 network, 22 hours on one H200 for CS13 |

## Tables and figures

Paths are relative to the repository root. The checks are described below the table. The records here are those of the
original tests. The paper's surface accuracy comes from SURF-v2, its BB, Cascade and output-head accuracy from RETEST
([Registered tests and how to verify them](#registered-tests-and-how-to-verify-them)), and every speed number from
SPEED-RT, whose outputs the speed entries name ([How the speed was measured](#how-the-speed-was-measured-speed-rt)).

| paper | what it shows | records | check |
|---|---|---|---|
| Fig. 1, Fig. 2 | the pair; the output-head geometry (diagrams) | | |
| Fig. 3 | surface code: failure rate against composite time per shot, d = 7-17 | `results/surface/` (`s2_test_eval.csv`, `s3_test_eval.csv`, `s4_test_eval.csv`, `r17_test_eval.csv`, `s2_composite.json`, `results_accuracy_speed_d15.json`, `results_accuracy_speed_d17.json`, `r17_summary.json`, `controls_*`), `results/r17/`; speed: `speed_rt/s2/surface/results/` (rows S1 to S4), `speed_rt/s2/gapsw/results/` | surface code; figures; [SPEED-RT](#how-to-verify) |
| Table I | BB codes: failures and composite time | `bb/bb72/ising_compare/data/bb/stage3/` (`results.json`, `results_amendment_S.json`), `bb/gross/ising_compare/data/bb/stage3_gross/`, `bb/gross_stage4/ising_compare/data/bb/stage4_gross/`; speed: `speed_rt/s2/bb/results/` (rows B1 to B6) | BB codes; [SPEED-RT](#how-to-verify) |
| Table II | pairs with our Cascade: BB72 under the fitted noise model, and d = 13 | `cascade/bb72/cascade_compare/phase_d/` (`D1_RESULT.md`, `out/`), `cascade/d13/` (`RESULTS.md`, `out/`); speed: `speed_rt/cascade/results/` | Cascade; [SPEED-RT](#how-to-verify) |
| Table III | the output-head comparison | `head/results/{surface,bb72}/` (`standalone.csv`, `pair_curves.csv`), `head/rule/`, `head/A1_RESULT.md` | output heads |
| Fig. 4 | where the pairs gain (exploratory) | `results/discordant/` (`X2_RESULT.md`, `data/`) | figures |
| Fig. 5 | routing frontiers on calibration shots | `results/surface/frontiers/`, `results/r17/policy_d17b5_r787_q08.json`, `bb/bb72/ising_compare/data/bb/fullroute/bb72_cal.json`, `bb/bb72/ising_compare/data/bb/policies/`, `bb/gross/ising_compare/data/bb/stage3_gross/frontierf_g2_D12k113c10.json`, `bb/gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md` | figures |
| Table IV | paired accuracy against the closest baseline | `results/surface/s2_test_eval.csv`, `s3_test_eval.csv`, `s4_test_eval.csv`, `r17_test_eval.csv` | surface code |
| Fig. 6 | BB codes: block failure rate against composite time | the records of Table I; speed: `speed_rt/s2/bb/results/` | figures; [SPEED-RT](#how-to-verify) |
| Table V | where the composite time goes | `speed_rt/s2/surface/results/table.md` (row S6) and `speed_rt/s2/bb/results/table.json` (row B4); the earlier sessions' in `results/surface/s2_composite.json`, `results_accuracy_speed_d15.json`, `results_accuracy_speed_d17.json`, `results/r17/R17_RESULT.md` and the records of Table I | [SPEED-RT](#how-to-verify) |
| Table VI | decoder settings and timing | the settings in `evaluation/convlines/opponents.py`, `bb/bb72/ising_compare/bb/opp.py` and `route.py`, `cascade/bb72/cascade_compare/phase_c/pc_common.py`, `cascade/d13/surface/sv_common.py`; the timing in `speed_rt/SPEED_RT.md` section 1, `STAGE2_surface.md` and `STAGE2_bb.md` | |
| Table VII | the first registered test with our Cascade on BB72 | `cascade/bb72/cascade_compare/phase_c/` (`RESULTS.md`, `out/confirm_*.json`, `out/report/`) | Cascade |
| Fig. 7 | failures of each output-head arm's pair against q | `head/results/{surface,bb72}/pair_curves.csv` | figures |
| Table VIII | every comparison of the output-head study | `head/results/{surface,bb72}/comparisons.csv` | output heads |
| Table IX | provenance of the claims | the plans: `results/r17/PLAN_R17.md`; `bb/bb72/ising_compare/bb/PLAN.md`, `STAGE3_PLAN.md`, `AMENDMENT_S.md`; `bb/gross/ising_compare/bb/STAGE3_GROSS_PLAN.md`; `bb/gross_stage4/ising_compare/bb/STAGE4_GROSS_PLAN.md`; `cascade/bb72/cascade_compare/PHASE_C_PLAN.md`, `phase_d/PHASE_D1_PLAN.md`; `cascade/d13/SURFACE_PLAN.md`; the test-set manifests in `evaluation/testsets/` | |
| Table X | medians of the shot descriptors (exploratory) | `results/discordant/data/tables.md`, `data/surface/summary.json` | |
| Fig. 8 | matching's class gap at d = 13 (exploratory) | `results/discordant/data/surface/fig_d13.npz` | figures |
| Fig. 9 | where the network fails, on shots read before any test | `bb/bb72/ising_compare/data/bb/fullroute/bb72_cal.json`, `policies/bb72_p0.003_rep2.json`, `bb/gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md`, `results/surface/development/a2_summary.json` | figures |
| Table XI | names in the paper and in this repository | `weights/networks.csv`, this README | |

### Weights

```bash
python weights/check_weights.py
```

checks every file of `weights/MANIFEST.sha256`, then builds each of the 38 networks with the code that uses it, loads
its weights strictly and compares its parameter count with `weights/networks.csv` (CPU, a few minutes).

### Surface code

From `evaluation/`:

```bash
cd evaluation
python -m timing.composite                                                      # the earlier tests' speed scores from their recorded components
python -m convlines.testset regen --set T2 --d 9 --chunks 0-99 --out shots/T2    # rebuild T2 at d = 9 bit for bit
python -m convlines.evaluate --set T2 --d 9 --chunks 0-99 --shots shots/T2 \
    --opponents "plain PM" "correlated PM" --out d9.json                         # ours against matching, with the statistics
```

`timing.composite` recomputes the earlier tests' composite times and checks their speed cells against the records;
SPEED-RT supersedes these times. `convlines.evaluate` gives the failures, the paired test and the interval of each
comparison, which `results/surface/s2_test_eval.csv` records. `results/surface/SURFACE.md` covers the rest: the other
distances and test sets, the Ising and BP-OSD opponents, the deployed engines and their earlier timing, the CPU term on
the earlier Grace basis, the two post hoc controls, study 4 and R17 (whose statistics it recomputes from the recorded
counts), and what is not established.

SURF-v2 ran from its own copy of this package, `surface_v2/`, which adds the retrained networks, the set manifests
`sv2cal`, `sv2dry` and `sv2test`, and `surf_v2/` (the registration, its code and its records); run SURF-v2's commands
there. The copy in `evaluation/` is an earlier state of the package, and the commands above for the released networks
use it. [Registered tests and how to verify them](#registered-tests-and-how-to-verify-them) shows how to check SURF-v2.

### BB codes

`bb/BB.md` gives, for BB72 and both gross-code tests, how each test ran, its disclosed changes, and the commands that
rebuild its test shots, read the networks and the opponents, and analyse them. The records of these three tests hold
every decoder's per-shot predictions (`test_outputs_BB72-2.npz` and `test_outputs_BB72-3.npz` in
`bb/bb72/ising_compare/data/bb/stage3/`, `test_outputs_G-4.npz` in `bb/gross/ising_compare/data/bb/stage3_gross/` and
`test4_outputs_G-4RF.npz` in `bb/gross_stage4/ising_compare/data/bb/stage4_gross/`) and the timing rows, so the
statistics can be checked without decoding again. RETEST's reruns of the BB72 test and of the gross code's stage-4
test keep theirs in `retest/run/bb/stage3/outputs/` and `retest/run/bb/stage4/outputs/`. The other tests' full
per-shot predictions are not in this repository ([What is not included](#what-is-not-included)).

### Cascade

```bash
python cascade/bb72/repro/recompute_numbers.py --out /tmp/cascade_numbers        # BB72: every number of the comparison from its records
python cascade/d13/surface/scripts/results_tables.py cascade/d13/out             # d = 13: the tables of cascade/d13/RESULTS.md
```

`recompute_numbers.py` compares each recomputed number with its record and with the value printed in the paper at the
time it was written (CPU, under a minute). `cascade/bb72/CASCADE.md` describes the BB72 comparison: how it ran, where
each number comes from, the per-shot files and the GPU check of the networks (`cascade/bb72/repro/check_networks.py`).
`cascade/d13/RESULTS.md` and `cascade/d13/SURFACE_PLAN.md` do the same for d = 13. The speed numbers of both comparisons
in the paper come from SPEED-RT's stage 1 (`speed_rt/cascade/results/`), not from these records.

### Output heads

`head/A1_RESULT.md` gives every verdict with its counts, and `head/results/{surface,bb72}/` the analysis outputs
(`analysis.json`, `standalone.csv`, `pair_curves.csv`, `comparisons.csv`, `references.csv`, `sizes.csv`,
`train_checks.csv`). The runs that drew the test sets, read the networks and the fallbacks, and analysed them are the
argument files `head/ising_compare/a1/jobs/args/draw_sets.txt`, `reads_main.txt`, `fallback_pm.txt`,
`fallback_relay.txt` and `analysis_main.txt`; they run from `head/ising_compare` under `a1/jobs/job.sbatch` with your
own directories for the networks, the sets and the reads. The sets' seeds and digests are in
`head/ising_compare/data/a1/sets_manifest.csv`.

### Figures

```bash
python results/figures/make_figs.py        # Figs. 3, 5, 6 and 9
python results/figures/make_head_fig.py    # Fig. 7
python results/figures/make_why_fig.py     # Figs. 4 and 8, from results/discordant/data
```

Each script writes its PDFs next to itself and names, beside every block of numbers, the record it was copied from.

## Training

`training/TRAINING.md` is the guide. In short, from the repository root:

```bash
bash training/smoke.sh                     # train a tiny d = 5 surface-code front end and tune its threshold (CPU, 5 minutes)
export PYTHONPATH=$PWD/evaluation:$PWD/training
python -m convlines_train.train --circuit my_circuit.stim --out runs/mine      # a front end for your own Stim circuit
python -m convlines_train.tune --frontend runs/mine/final_ema.pt --shots 200000 --seed 7 \
    --strong convlines_train.strong:pymatching_correlated --max-fraction 0.1 \
    --rule budget --front-us 2 --budget-us 3 --out runs/mine/tune.json       # its threshold, for any strong decoder
```

The released networks were trained by the frozen programs in `training/surface` (13 stages for the surface chain,
`bash training/surface/run_stage.sh STAGE`) and `training/bb` (BB72, gross and the Cascade comparison on BB72), by the
output-head comparison's code in `head/` and by `cascade/d13/surface/surface_train.py` (CS13). `training/LINEAGE.md`
gives each network's initialisation, stages, samples, schedule, learning rates and data seeds; `TRAINING.md` gives the
commands, and how to check that the frozen programs rebuild the released networks (`convlines_train.check_frozen`) and
how a retrained network compares with the released one. The threshold-tuning rule of the paper is `--rule paper` of
`convlines_train.tune`.

## Checks already run

- The Cascade comparison on BB72 was checked from fresh clones on the cluster (`cascade/bb72/REPRODUCTION_CHECK.md`):
  every manifest and archived per-shot file matched; the networks' GPU check passed; of 192 recomputed numbers, the 70
  that have a record all equal it, and of the 146 compared with the paper, one was not as printed (5.1 against 5.2, a
  speed-up's low end), which the paper has since corrected.
- R17's record check (`results/r17/REPRODUCTION_CHECK.md`) and a full re-run of the surface-code reproduction steps on a
  rented H100 (`results/reproduction/REPRODUCTION.md`).
- RETEST's replay, seed and platform checks before its freeze (`RETEST.md` section 6, records in `retest/replay/`).
- SPEED-RT's reruns: on 14 October 2026 (UTC) the commands of [How to verify](#how-to-verify) ran from a fresh clone of
  `main` at `e0a9377` on a Linux x86-64 machine with Python 3.12.3 and numpy 1.26.4. Both beacon records passed the
  offline check, stage 1's seeds file was re-derived byte for byte, and every output of the four analyses was
  byte-identical to the published one.
- The verification of this repository's first commit from a fresh clone (next section).

## Verification of this repository

On 12 October 2026 this repository's first commit, `ab5cf49`, was cloned fresh on the Alpine cluster (CU Boulder
Research Computing) and checked there with the commands above.

- Installation: a new virtual environment from `requirements.txt` with torch 2.11.0+cu128, which brought Triton 3.6.0
  and cuDNN 9.19, plus relay-bp 0.2.2 and ldpc 2.4.1. Every pin installed as listed. The nodes' Python was 3.12.12
  (pinned: 3.12.14), and the GPU node needed `C_INCLUDE_PATH` for the Python headers (Installation).
- Weights: all 45 files match `weights/MANIFEST.sha256`, and all 38 networks build, load strictly and have the
  parameter counts of `weights/networks.csv`. `convlines_train.check_frozen` confirmed that the frozen training
  programs rebuild the 16 networks of the surface chain, BB72, the gross code and the Cascade comparison on BB72.
- Code and documents: every Python file compiles, and the 42 bash blocks of `README.md`, `results/surface/SURFACE.md`,
  `bb/BB.md`, `cascade/bb72/CASCADE.md` and `training/TRAINING.md` parse.
- Records: `timing.composite` reproduced every registered composite time and speed cell. `recompute_numbers.py`
  recomputed the 192 numbers of the Cascade comparison on BB72: none of the 70 that have a record differs from it, and
  of the 146 compared with the paper's values when the script was written, the one not as printed is the 5.1 that the
  paper now prints. `results_tables.py` printed the d = 13 tables, and the three figure scripts redrew every figure.
- Surface code, on a dry set (T2 at d = 9, chunks 0 and 99): both chunks were rebuilt with the manifest's sha256, and the
  reference decoder failed 94 and 124 times, as the earlier release's verification recorded. On chunk 0,
  `convlines.evaluate` gave ours 94 failures against 233 for plain PM and 124 for correlated PM (better than both), and
  the deployed Triton engine on an H200 MIG slice gave the same 94 failures, with the routing of every shot equal to the
  published mask.
- SURF-v2's options in `evaluation/`: `convlines.evaluate --save` wrote every decoder's per-shot arrays and record for
  the same chunk, on the CPU and through the deployed engine, with the same counts and the network's sha256 equal to the
  manifest's; `convlines.beacon` checked BB72's beacon record offline and re-derived its 50 test seeds.
- BB72, on a dry set (test chunk 0 at p = 0.3 %, 10,000 shots): the shots were rebuilt from their seed, and the second
  network's read on an H200 MIG slice, through the commit guard, matched the published per-shot outputs exactly: all
  10,000 decisions equal and every margin bitwise equal.
- Training: `training/smoke.sh` (4·10^4 samples) and `convlines_train.train --bb` on BB72 (2·10^4 samples) ran on a
  CPU; the frozen BB program (16 steps of `bb.train`) and the frozen surface program (20 steps of the d = 7 stage, with
  `torch.compile`) ran on an H200 MIG slice.

## What is not included

- The test shots, and the full per-shot outputs of every test read except the BB tests' ([BB codes](#bb-codes)). The
  shots are rebuilt from the manifests' seeds. The per-shot outputs of the surface-code tests (SURF-v2 included), the
  Cascade tests and the output-head study are archived by the authors.
- SPEED-RT's timing shots. Stage 2's are rebuilt from the beacon's seeds and checked against the manifests in
  `speed_rt/testsets/`, `speed_rt/bb/` and `speed_rt/gapsw/`; stage 1's processes sampled their chunks during the run
  and recorded each chunk's seed and sha256.
- The fp32 module of the d = 17 surface network: it is released as its deploy tables, which the deployed engine reads.
- NVIDIA's Ising-Decoding code and checkpoints, and Relay-BP's repository: fetch them from their public releases.
- The research project's internal plans that the frozen trees cite but that no result depends on, the review texts,
  and the logs of runs that no table reads.

## License

The code and the trained network weights are licensed under the Apache License, Version 2.0 (`LICENSE`, `NOTICE`).
NVIDIA's Ising-Decoding code and checkpoints are only referenced and never redistributed; they keep their own terms
(the code under Apache 2.0, the checkpoints under their model cards on Hugging Face). PyMatching, Stim and Relay-BP keep
their own licenses.

## Citation

```bibtex
@misc{guang2026convlines,
  title  = {A Small Network Makes Quantum Decoders Up to 100x Faster and More Accurate},
  author = {Guang, Yicheng and Xiong, Yuliang and Gao, Xun and Chen, Yueqi},
  year   = {2026},
  note   = {Code, weights and records: \url{https://github.com/a85tract/Convlines_Public}}
}
```

`CITATION.cff` holds the same.
