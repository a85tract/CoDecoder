# Speed measurements

Part of the [CoDecoder release](../README.md).

## How the speed was measured: SPEED-RT

SPEED-RT timed again every speed number of the paper that this release can run, on fresh timing shots drawn from a
public randomness beacon after its rules were public (`speed_rt/SPEED_RT.md`). Nothing was re-chosen, re-run or re-tuned
because of a result, and a claim whose speed condition failed is reported as failing. SPEED-RT makes no
deployment-latency claim. By the authors' decision it left out a few values of the paper, which
[`METHODS.md`](../METHODS.md) section 2.8 lists. Section 2 of `METHODS.md` describes the measurement in more detail, and
the registrations hold the full rules. Its third stage added a new measurement: the decoders of the paper's Figs. 3 and
4 in throughput mode, in batches of 4,096 shots (`speed_rt/STAGE3_batched.md`).

### Stages, freezes and beacons

| stage | what it timed | registration | freeze commit and GitHub's record of its push | beacon round | beacon commit | analysis commit | outputs |
|---|---|---|---|---|---|---|---|
| 1 | the Cascade comparisons, on an RTX PRO 6000 and on an H100 | `speed_rt/SPEED_RT.md` section 1 | `73ca9f9`: `push` to `main` from `ebbb6b6`, 2026-10-13T21:48:15Z, record 46399062301 | 6549923, released 2026-10-13T21:58:30Z | `6609d14` (`speed_rt/cascade/beacon.json`, `seeds.json`) | `56e2023`; correction `2ce2e49` | `speed_rt/cascade/results/` |
| 2 | every other speed number: the surface comparisons with Ising, BB72 and the gross code, Decoder Switching | `speed_rt/STAGE2_common.md`, `STAGE2_surface.md`, `STAGE2_bb.md`, `STAGE2_gapsw.md` | `7988e04`: `push` to `main` from `56e2023`, 2026-10-14T04:00:56Z, record 46424448042 | 6550668, released 2026-10-14T04:11:00Z | `ea1525c` (`speed_rt/testsets/beacon_surf.json` and the shot manifests) | `e0a9377`; amendment `54d9eec` | `speed_rt/s2/surface/results/`, `speed_rt/s2/bb/results/`, `speed_rt/s2/gapsw/results/` |
| 3 | the decoders of Figs. 3 and 4 in throughput mode (batches of 4,096): the matching and Relay-BP compositions, CoDecoder alone and their baselines, at 1 and 8 CPU processes | `speed_rt/STAGE3_batched.md` | `3842ad8`: `push` to `main` from `928d00b`, 2026-10-19T22:58:25Z, record 47065873371 | 6567343, released 2026-10-19T23:08:30Z | `217dced` (`speed_rt/s3/beacon.json` and the shot manifests in `speed_rt/s3/records/manifests/`) | `561a0d7`; correction `25463ac` | `speed_rt/s3/results/` |

Each stage froze its registration, code and timing sets in a freeze commit and pushed it to GitHub before any of its
timing shots existed. Its beacon is the first drand mainnet round at least 600 s after GitHub's earliest activity record
of that push, and every timing chunk's seed derives from the round's randomness (`speed_rt/sets.py`), so nobody could
know a timing shot before the round. The beacon record, which keeps GitHub's activity list as fetched, and the shot
manifests were committed and pushed as the beacon commit, and nothing of the stage was timed before that commit was
public. For stage 1, GitHub recorded the push of `6609d14` at 2026-10-13T22:01:23Z (record 46400110294, in the activity
list that `speed_rt/testsets/beacon_surf.json` keeps), and the first timed process started at 22:02:03Z
(`speed_rt/cascade/records/H100/R1/attempts.json`). Stage 2's record also sits, byte for byte, in
`speed_rt/bb/beacon.json` and `speed_rt/gapsw/beacon.json`. For stage 3, GitHub recorded the push of `217dced` at
2026-10-19T23:11:24Z (record 47066848302), before any of the stage's timing. After the timing, each stage's frozen
analysis ran twice on the records with byte-identical outputs, and the records and outputs were committed together.
Stage 3's correction `25463ac` replaced other processes' command names in its CPU window records by a generic name;
nothing the analysis reads changed, and its outputs stayed byte-identical.

### What a time is

- Surface code and BB codes: our composite time per shot is the network's GPU term on the H100 at batch one, with the
  input already on the GPU, plus the handoff, plus the fallback's CPU time on the routed shots (the routed fraction
  times the mean time of a routed shot) on the Grace basis. The GPU and CPU terms come from two machines, as in the
  paper's earlier composites, so the stage-2 composites of our compositions, and Ising's with its PM share, are not measured
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
- Stage 3, throughput mode: our cost per shot is G + C(N). G is the network's host-to-host wall time per shot at batch
  4,096 on one H100, from detector bits in pinned host memory to the decisions on the host, with the routing and the
  packing of the routed shots inside the timed loop (the multi-shot engine on BB72, the Triton engine on the surface
  code, the BB engine's kernels on the gross code). C(N) is the wall time of N = 1 or 8 single-threaded fallback
  processes on the workstation, each on its own core of the timed CCD, decoding the routed shots with the decoder's
  batch call, divided by all timing shots, on the Grace basis. Each baseline runs every timing shot, or its registered
  first shots, in the same harness. Cost per round divides by d on the surface code, 6 on BB72 and 12 on the gross code.
  The pipelined bound max(G, C(N)) is reported beside the sum, descriptively.

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
- Stage 3: one H100 80GB HBM3 SXM pod (host Intel Xeon Platinum 8462Y+, container quota 13.6 CPUs, screened with a
  quota limit of 12, a registered departure) timed every GPU term. No RTX PRO 6000 was in hand before the GPU timing,
  so the stage timed the H100 only. The workstation, under the same C1 settings, timed every CPU term with N = 1 and 8
  processes on CPUs 0-7. Its session reference was 17.046 µs per block against stage 2's 17.306, within the 1.10 rule,
  so stage 2's Grace factor applies. All 89 GPU processes and all 24 CPU windows were accepted at their first attempt
  (`speed_rt/s3/records/`).

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
CUDA-Q, which is not converted, the conversion works against the composition: on the raw basis the gross code's balanced composition
takes 13.57 ms, just under CUDA-Q's 13.70 ms (`speed_rt/s2/bb/results/numbers.json`, `sensitivity_raw_xeon`, a key named
after an earlier host that holds the raw basis). GPU terms, handoffs and CUDA-Q's host-to-host times are not converted.

### Statistics

- Stage 2: our side takes the slowest of its fresh processes (three for every CPU term, five for the GPU terms of the
  surface networks and of BB72's, three for the gross code's) and every baseline the fastest of its three, with no
  medians (`STAGE2_common.md` C6). Ising's GPU term is the exception: it is the smallest accepted repetition across all
  validated plans and both timing methods (`STAGE2_surface.md` 1.13; [`METHODS.md`](../METHODS.md) section 2.5), which
  favours Ising. The Grace reference points take the slowest of three, which gives the smaller factor.
- Stage 3: as stage 2, our side takes the slowest of five GPU processes (three for the gross code) and of three CPU runs
  per N, and every baseline the fastest of three; speed-ups compare the same N on the same basis
  (`speed_rt/STAGE3_batched.md` section 6).
- Stage 1: the registered analysis takes the slowest of three on both sides. Its sensitivity rows repeat every Cascade
  comparison with Cascade's fastest process, and the paper reports these rows, so that every comparison of the paper
  times our side by its slowest process and the baseline by its fastest. They lower no speed-up by more than 3.2 % and
  change no outcome; the lowest speed-up is 3.63 on the RTX PRO 6000 and 3.17 on the H100
  (`speed_rt/cascade/results/table.json`). Mean times on both sides would only raise every speed-up (Table VIII of
  the paper).
- "X takes less time than Y" holds when X's value is strictly below Y's. There is no statistical test. Ranges and
  ratios of times in the prose are rounded against us: our times, and our time as a fraction of another decoder's, up;
  the other decoders' times, and speed-ups (another decoder's time over ours, also when below 1), down.

### The re-measured times

Surface code: composite time per shot in µs on the Grace basis, with BP-OSD as timed on the pod
(`speed_rt/s2/surface/results/table.md`, row S1).

| d | the composition, SURF-v2's networks | the composition, deployed networks | plain PM | correlated PM | Ising-Fast + PM | Ising-Accurate + PM | BP-OSD, min-LLR | BP-OSD, BP 60 |
|---|---|---|---|---|---|---|---|---|
| 7 | 11.942 | 12.480 | 11.890 | 27.598 | 47.144 | 58.995 | 2183.5 | 5950.8 |
| 9 | 28.927 | 29.977 | 26.297 | 63.916 | 57.204 | 67.223 | 8655.4 | 19033.8 |
| 11 | 41.482 | 43.521 | 52.234 | 137.889 | 69.350 | 78.592 | 31365.0 | 52051.1 |
| 13 | 51.245 | 60.313 | 91.704 | 253.071 | 92.311 | 100.809 | 87743.2 | 130648.5 |
| 15 | 59.552 | 90.099 | 160.091 | 562.655 | 123.821 | 128.178 | 193252.4 | 287949.2 |
| 17 | 81.579 | 141.406 | 242.527 | 846.008 | 173.116 | 172.988 | 478767.2 | 647847.1 |

Each network set has 48 registered comparisons (rows S4 and S10): these six baselines and two controls, Ising-Fast and
Ising-Accurate followed by correlated PM (51.067 to 354.429 µs and 61.907 to 321.348 µs), at six distances. For each
set 46 hold; the two that do not are plain PM at d = 7 and 9. The paper's composition at d = 9 is D9-SHALLOW's, which took
21.234 µs, below plain PM's 26.297 µs ([D9-SHALLOW](REGISTERED_TESTS.md#d9-shallow-the-network-at-d--9)), so there only plain PM at d = 7 is faster
than the composition.

BB codes: time per shot in ms, with CPU times on the Grace basis and CUDA-Q as timed on the pod
(`speed_rt/s2/bb/results/table.json`, rows B1 to B3).

| point | network alone | balanced composition | accurate composition | sector Relay-BP | full-DEM Relay-BP | ldpc BP-OSD (CPU) | CUDA-Q BP-OSD (GPU) |
|---|---|---|---|---|---|---|---|
| BB72, p = 0.3 % | 0.09072 | 0.4148 | 4.864 | 3.302 | 85.37 | 1.818 | 0.8909 |
| BB72, p = 0.2 % | 0.08941 | 0.1169 | 0.4224 | 1.987 | 52.78 | 0.6908 | 0.5396 |
| gross code, p = 0.4 % | 0.2555 | 20.35 | 427.5 | 127.6 | 3811 | 114.0 | 13.70 |

On BB72 the network alone at p = 0.3 % is the first of the two networks; the second, which every BB72 composition uses, took
0.08941 ms alone. The Cascade times are in `speed_rt/cascade/results/table.json`, per card, and with every repetition in
`numbers.json`.

Stage 3, throughput mode, cost per syndrome round on the Grace basis (`speed_rt/s3/results/table.md`, which also gives
the raw basis and every speed-up): CoDecoder + correlated PM takes 0.187, 0.698 and 0.822 µs at d = 7, 9 and 11 with one
CPU process and 0.0856, 0.187 and 0.822 µs with eight, against 3.22, 6.45 and 10.9 µs (one process) and 0.399, 0.833 and
1.51 µs (eight) for correlated PM alone. From d = 13 the network alone takes 1.154 to 2.156 µs per round on the H100. On
BB72 at p = 0.2 %, CoDecoder + sector Relay-BP takes 4.50 µs per round with one process and 1.12 µs with eight. Every
composition of the stage costs less than its fallback alone, at both N and on both bases.

### Shot counts

To fit the CPU session into one evening, the authors cut the shot counts of the slowest CPU items before stage 2's
freeze. Each cut item times the first N shots, or the first N routed shots, of its set in shot order (`STAGE2_bb.md`
1.3, `STAGE2_gapsw.md` 1.3):

| item | count before the cut | count timed |
|---|---|---|
| gross code: full-DEM Relay-BP | 300 shots | 75 shots |
| gross code: ldpc BP-OSD | 1,000 shots | 500 shots |
| gross code: the balanced composition's CPU term (sector Relay-BP) | 1,000 routed shots | 300 routed shots |
| gross code: the accurate composition's CPU term (full-DEM Relay-BP) | 100 routed shots | 25 routed shots |
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
library), these commands check the three beacon records offline, re-derive stage 1's seeds, and rerun each frozen analysis
on the published records. Every `cmp` and `diff` must print nothing.

```bash
export PYTHONPATH=$PWD:$PWD/surface_v2
O=$(mktemp -d)                                    # a new, empty directory for the reruns
# the beacon records, checked offline, and stage 1's seeds
python3 -m speed_rt.beacon check --beacon speed_rt/cascade/beacon.json --commit 73ca9f9fcef77a3e5c8c170095d6d8c2fe14a5ae
python3 -m speed_rt.beacon check --beacon speed_rt/testsets/beacon_surf.json --commit 7988e04e74153d1234aca4b8889bd176f73e7bac
python3 -m speed_rt.beacon check --beacon speed_rt/s3/beacon.json --commit 3842ad8920128574d28456363d7e89a392e590d2
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
# stage 3: the decoders of Figs. 3 and 4 in throughput mode (the figures are drawn separately)
python3 -m speed_rt.s3.analysis3 --records speed_rt/s3/records --out $O/s3
diff -r -x figures $O/s3 speed_rt/s3/results
```

Each analysis checks the records it reads against its freeze (the sha256 of the files the freeze names, the shots'
manifests and seeds, each session's acceptance) before it computes, and stage 2's analyses re-derive the Grace fit from
its 18 points. A value whose slot is missing is withheld, never replaced. `speed_rt/cascade/results/README.md` and
`speed_rt/s2/README.md` say how the records were sanitized for publication; the files whose sha256 the analyses check
are published byte for byte.
