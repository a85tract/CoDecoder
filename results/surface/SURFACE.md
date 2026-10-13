# The surface-code tests: results, method and reproduction

ConvLines decodes the rotated surface code under circuit-level noise. A small convolutional network reads the detector
bits of one shot and returns one number, the logit c of a logical flip. When the network is confident (|c| ≥ τ) its
sign is the answer. The few shots where it is not (1-6 % of shots, depending on the distance) are decoded by correlated
PyMatching instead.

This repository contains what an outside reader needs to check the accuracy-against-speed results of three registered
studies, at code distances d = 7, 9, 11, 13 (study 2, test set T2), d = 15 (study 3, test set T3) and d = 17 (study 4,
test set T5, re-registered as R17 on test set T6): the six trained networks (d = 17 in its deployed form), the
circuits, the seeds of the four test sets and the rule that rebuilds T2, T3 and T5 bit for bit, the evaluation against
six opponent decoders (eight at d = 17) with the registered statistics, the deployed GPU executables (a Triton engine
at d = 9-17, a hand-written CUDA kernel at d = 7) with their timing harnesses and the routing they recorded, our
TensorRT build, evaluation and timing script for NVIDIA's Ising opponents, the rules that qualified a timing session,
and the conversion of CPU times to the Grace basis. It also holds two post hoc controls, Ising followed by
correlated PM, with their data and the commands that rebuild them. Study 4 ran under its third re-registration; [its
subsection](#study-4-d--17) gives the history. R17 then re-registered the d = 17 row at a lower routing share and read
it on a new test set, T6 ([R17](#r17-d--17-re-registered)).

The registered tests on IBM's bivariate bicycle codes are in [`../../bb/BB.md`](../../bb/BB.md), and ConvLines in front of
our re-implementation of Cascade on BB72 is in [`../../cascade/bb72/CASCADE.md`](../../cascade/bb72/CASCADE.md).

Ising M1 / M4 followed by correlated PM are two post hoc controls at d = 7-15, measured after studies 2 and 3; they are
not registered claims ([the two post hoc controls](#the-two-post-hoc-controls-ising--correlated-pm); how to
rebuild them: [section 6](#6-the-two-post-hoc-controls)). At d = 17 the same two decoders are registered opponents
(of study 4 and of R17). At d = 17 (R17 on T6), ours failed 2,708
times in 1,000,000 shots and takes 135.02 µs per shot on the registered composite (133.48 µs on the matched boundary).

**Two speed boundaries.** The registered composite is the primary score; the matched boundary puts both sides on the
same per-shot boundary.

| boundary | ours | opponents (Ising + PM, PM, BP-OSD) |
|---|---|---|
| registered composite (primary) | (a) GPU time + (b) the routed CPU part on the Grace basis + (c) the handoff, max(1.1 us, measured): getting the shot to the GPU, the result back to the host, and the branch | Ising: its TensorRT engine timed with back-to-back CUDA-graph replays (the fastest of 5 x 100, as NVIDIA's trtexec does) + its PM share on the Grace basis; PM: its CPU time on the Grace basis. No per-shot host round trip is charged |
| matched boundary | (a) + (b), without (c) | unchanged |

The registered boundary charges the per-shot host round trip to our pipeline only, which favours the opponents; the
matched boundary removes it from ours too. Each of our points moves left by 1.1-2.6 us, and no conclusion changes.
Neither boundary is an end-to-end latency: on both sides the input is already on the GPU (section 4, "The
opponents' times" and "What is not established"). Same-host end-to-end measurements come from an independent
reproduction that is not part of this repository. Its d = 17 supplement reproduced study 4 before R17,
so its d = 17 figures show study 4's point (T5, q = 10 %: ours 2,687 failures).

Ising's rows carry no handoff, so the two controls do not move between the two boundaries.

Terms used below:

- **shot**: one simulated memory experiment of d rounds. **detector**: the parity of two consecutive measurements of
  one stabiliser (1 = something changed). **failure**: the decoder's predicted logical flip differs from the true one.
  **LER**: logical error rate, failures / shots.
- **PM**: PyMatching 2.4.0 (minimum-weight perfect matching). **correlated PM**: its two-pass mode that also uses the
  correlations between X and Z errors.
- **Ising M1 / M4 + PM**: NVIDIA's public neural pre-decoders "Ising-Decoder-SurfaceCode-1-Fast" (M1) and
  "-Accurate" (M4), each followed by PM on the detectors they leave.
- **BP-OSD**: belief propagation with ordered-statistics post-processing (NVIDIA CUDA-Q QEC's GPU decoder), in two
  settings: min-LLR (10 BP iterations, OSD initialised from the minimum LLRs) and BP 60 (60 iterations, the
  conventional final-LLR initialisation).
- **Grace basis**: CPU times expressed as if they ran on one core of an NVIDIA Grace CPU, the CPU of NVIDIA's own Ising
  timings. Most of these numbers were converted from other CPUs (section 4).

## Results

Setting: rotated surface code, X-basis memory, d rounds, NVIDIA's legacy single-parameter circuit noise at p = 0.006
(gate, idle and two-qubit error p, measurement and reset error 2p/3). Each study was registered first: the networks,
their thresholds, every opponent setting, the statistics and the test-set rule were fixed and hashed before the test
set existed, and the test set was read once. Study 4 changed its timing session and dropped its guard rules after
failed attempts, and drew a new test set after that change was public; [its subsection](#study-4-d--17) says what
changed. R17 re-registered study 4's d = 17 row at a lower routing share (q = 8 %) and read it on a new test set, T6;
the d = 17 rows below are R17's ([R17](#r17-d--17-re-registered)).

### Accuracy

Failures on the test set; every decoder decoded the same shots:

| d | test shots | ours | Ising M1 + PM | Ising M4 + PM | plain PM | correlated PM | min-LLR BP-OSD | BP 60 BP-OSD | what it means |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 2,000,000 | **33,582** (1.68 %) | 48,875 | 38,000 | 60,138 | 35,291 | 34,227 | 41,124 | closest opponent: min-LLR BP-OSD; ours fails 1.9 % less often |
| 9 | 1,000,000 | **10,789** (1.08 %) | 22,841 | 17,001 | 27,317 | 12,917 | 16,332 | 19,873 | closest: correlated PM; ours fails 16 % less often |
| 11 | 1,000,000 | **7,215** (0.72 %) | 20,261 | 14,817 | 24,358 | 9,325 | 15,078 | 18,881 | closest: correlated PM; ours fails 23 % less often |
| 13 | 1,000,000 | **5,627** (0.56 %) | 17,478 | 12,399 | 21,501 | 6,549 | 13,317 | 17,623 | closest: correlated PM; ours fails 14 % less often |
| 15 | 1,000,000 | **3,911** (0.39 %) | 15,232 | 10,077 | 18,442 | 4,311 | 2,318 * | 3,247 * | closest: correlated PM; ours fails 9 % less often |
| 17 | 1,000,000 | **2,708** (0.27 %) | 12,840 | 8,070 | 15,658 | 2,943 | 2,114 * | 3,141 * | closest: correlated PM; ours fails 8.0 % less often |

\* The BP-OSD arms read the first 200,000 shots of T3 and of T6 only (a registered subset); ours failed 768 (d = 15)
and 502 (d = 17) times on those shots. The Ising columns show the variant with the fewest failures: the studies read
the official PyTorch numerics and every TensorRT plan they timed, and ours had to beat each of them
(`results/surface/s2_test_eval.csv`, `results/surface/s3_test_eval.csv`, `results/surface/r17_test_eval.csv` hold every row).

Study 4's d = 17 row, the earlier registration (T5, q = 10 %): ours 2,687, Ising M1 + PM 12,843, Ising M4 + PM 8,081,
plain PM 15,556, correlated PM 2,907, and on 200,000 shots min-LLR BP-OSD 2,070 and BP 60 BP-OSD 3,016 (ours 542). Its
records are `results/surface/s4_test_eval.csv` and [its subsection](#study-4-d--17).

All 30 accuracy cells of studies 2 and 3 (5 distances × 6 opponents) are "better" by the registered test: on the same
shots, the exact two-sided McNemar test gives p < 0.025 and the 97.5 % paired Newcombe interval (method 8) of
LER(ours) − LER(opponent) lies below 0. The weakest cell is d = 7 against min-LLR BP-OSD: p = 2.3·10⁻⁴, interval
[−5.2·10⁻⁴, −1.3·10⁻⁴]. At d = 17 all 8 accuracy cells of R17 (these six opponents and Ising M1 / M4 + correlated PM)
are "better" by the same test; the weakest is correlated PM, p = 9.9·10⁻¹⁷, interval [−3.0·10⁻⁴, −1.7·10⁻⁴]. Study
4's 8 cells on T5 were "better" too.

### Speed

µs per shot at batch 1, the registered composite score on the Grace basis (lower is faster):

| d | ours | Ising M1 + PM | Ising M4 + PM | plain PM | correlated PM | min-LLR BP-OSD | BP 60 BP-OSD | what it means |
|---|---|---|---|---|---|---|---|---|
| 7 | **12.92** | 48.76 | 61.41 | 17.16 † | 35.71 † | 2,864 | 5,668 | 4.2 µs below plain PM, the fastest opponent |
| 9 | **30.05** | 56.40 | 68.08 | 36.99 † | 67.59 | 9,354 | 18,211 | 6.9 µs below plain PM |
| 11 | **43.93** | 68.63 | 78.67 | 72.02 † | 155.61 † | 25,203 | 46,430 | 24.7 µs below Ising M1 + PM |
| 13 | **57.96** | 90.26 | 98.34 | 129.61 | 279.63 | 68,177 | 115,302 | 32.3 µs below Ising M1 + PM |
| 15 | **86.08** | 122.85 | 129.63 | 208.53 † | 464.95 † | 197,370 | 294,551 | 36.8 µs below Ising M1 + PM |
| 17 | **135.02** | 161.33 | 165.97 | 325.23 † | 714.05 † | 444,502 | 619,209 | 26.3 µs below Ising M1 + PM |

† Extrapolated to the Grace basis (`evaluation/timing/grace_basis.csv`); Ising's PM share is extrapolated at d = 7-11, 15 and 17.
BP-OSD times are measured on the timing host and not converted. All 30 speed cells of studies 2 and 3 hold (ours
strictly below the opponent), and so do R17's 8 at d = 17 (the two not in this table are in its subsection). The
score is a registered benchmark and does not measure end-to-end latency (section 4 and "What is not established").

The registered d = 17 row converts both sides from NVIDIA's d = 13 values. NVIDIA's Ising paper also timed correlated
PM on Grace at d = 17 and plain PM and the Ising PM shares at d = 21, values the registrations missed. Converting every
CPU time at d = 17, ours and every opponent's, by one rule at that distance (the NVIDIA-distance basis: NVIDIA's value
where it has one, otherwise a log-log interpolation between its d = 13 and d = 21 values;
[evaluation/timing/R17_GRACE_NOTE.md](../../evaluation/timing/R17_GRACE_NOTE.md)), ours takes 148.74 µs against 170.47 µs (Ising M1 + PM), 173.87 µs (Ising M4 + PM), 313.59 µs (plain PM) and 860.71 µs
(correlated PM). The narrowest margin is then 21.73 µs, over Ising M1 + PM, and all 8 speed cells hold.

[R17's subsection](#r17-d--17-re-registered) says which terms of its d = 17 composite were measured. Study 4's earlier
composite was 151.35 µs, 9.98 µs below Ising M1 + PM, against the same opponents' times; its 8 speed cells held too
([its subsection](#study-4-d--17), `results/surface/results_accuracy_speed_d17.json`). On the NVIDIA-distance basis its
composite is 168.42 µs and its 8 cells still hold, the two Ising + PM cells by 2.05 and 5.45 µs.

The next two subsections cover d = 17: study 4 and how it came to run, then R17, which re-registered its row. The one
after them adds two post hoc controls, Ising M1 / M4 followed by correlated PM, measured after studies 2 and 3 and not
registered ([the two post hoc controls](#the-two-post-hoc-controls-ising--correlated-pm)).

The registered claim reads: on T2 (2,000,000 shots at d = 7 and 1,000,000 at d = 9, 11, 13) and T3 (1,000,000
shots at d = 15, the BP-OSD opponents on the first 200,000), rotated surface-code X memory with legacy noise
p = 0.006, the frozen ConvLines decoder with confidence routing made fewer failures than each of the six opponents in
the registered paired tests and had lower registered Grace-basis composite scores, measured on validation shots at
batch 1 in one qualified H100 session per study. At d = 17 the claim is R17's: on T6 (1,000,000 shots at d = 17,
17-round X memory with legacy noise p = 0.006, the BP-OSD opponents on the first 200,000), the frozen d = 17 decoder
at R17's threshold made fewer failures than each of the eight registered opponents in the registered paired tests and
had a lower registered Grace-basis composite score than each ([R17](#r17-d--17-re-registered) says how that score was
put together). Study 4's claim (S4-R3) read: on T5 (1,000,000 shots at d = 17, the BP-OSD
opponents on the first 200,000), the frozen d = 17 decoder made fewer failures than each of eight opponents in the
registered paired tests and had a lower registered Grace-basis composite than each, measured on validation shots at
batch 1 in one H100 session that was gated on the GPU side only.

### Study 4: d = 17

Study 4 is the earlier registration at d = 17; R17 ([next subsection](#r17-d--17-re-registered)) re-registered its row.
Study 4's own results stay below and in its records (`results/surface/s4_test_eval.csv`,
`results/surface/results_accuracy_speed_d17.json`, `evaluation/testsets/T5.json`, `evaluation/routing/test_T5_d17.npz`).

Study 4 tests one distance, d = 17, with the network `d17_b5_r787` (5 blocks × 128 channels; τ = 5.1220, which routes 10
% of the calibration shots) against eight opponents: the six above and two that we added for this study, Ising M1 / M4
followed by correlated PM (the post hoc controls of d = 7-15). Each opponent has an accuracy cell and a speed cell,
tested as in studies 2 and 3, and the claim needs all 16. The test set T5 holds 1,000,000 fresh shots; the BP-OSD arms
read its first 200,000.

Study 4 was frozen on 2026-10-06 under the timing rules of studies 2 and 3 (section 4, "Qualifying a timing
session"), and it took three re-registrations before a session timed it:

| registration | what changed | what happened | what it means |
|---|---|---|---|
| first freeze (2026-10-06) | the original registration | The search for a quiet rented H100 host ended at its 6-hour bound: two shared single-GPU pods, neither quiet enough to begin. | no timing; its test set T4 was never read |
| S4-R1 (2026-10-07) | timing on a whole 8-GPU H100 SXM machine | It passed the quiet-host run and the screen. Its one gate attempt failed one of ten checks: three repeats of plain PM at d = 9 took 36.73, 35.63 and 35.88 µs per block, a spread of 3.10 % against the 3 % limit. The rule allows no second attempt on a host, so the session ended. | no timing; no test shot read |
| S4-R2 (2026-10-07) | S4-R1 plus a CPU warm-up: the gate's own plain-PM benchmark run once on the gate's core, at most 30 s, before the gate measured | It never ran: no whole 8-GPU machine was available. | nothing ran |
| S4-R3 (2026-10-07) | a minimal re-freeze, below | It ran, and its analysis ran once. | all 16 cells hold |

S4-R3 kept what our rule of 2026-10-07 asked of the study: an unseen test set, the frozen method, equal
conditions, a single analysis and public reproducibility. The network, its threshold and every opponent setting are
byte for byte those of the first freeze, and so are the statistics, the test-set size, the counting rules and the CPU
term. What it changed:

- It dropped the guard layers (the arbiter, receipts, reading tickets, addenda and test guards) that tied every test
  read to a session that had passed the host gate.
- Timing ran on one single-GPU H100 SXM pod. The GPU-side checks stayed hard gates, at the start and at the end: the
  median SM clock at least 97 % of its maximum under load, no throttle reason but idle, the whole 80 GB card, MIG off
  and no other process on the GPU. The host's load was recorded every second and did not gate.
- T4 was retired unread. T5 was drawn from the first drand round at least 600 s after the S4-R3 commit was public
  (round 6,532,408), and its manifest was committed before any read.
- It added Ising M1 / M4 + correlated PM. Their correlated-PM share was measured with the registered PM-share protocol
  on one Apple M4 core and converted with the registered d = 13 ratios.
- It dropped Ising's batch sweep (plans built for batch sizes up to 64, which studies 2 and 3 also read for accuracy)
  before the sweep started, to stay within the pods' budget. The Ising rows are the 8 timed batch-1 plans (M1 / M4 ×
  FP16 / FP8 × builder level 3 / 5) and the PyTorch numerics, each followed by plain and by correlated PM.
- BP-OSD ran on four rented single-H100 pods, plus a fifth for one share, instead of a cluster's A100s. Each pod first
  regenerated T5 and checked it against the manifest.

These changes were made after failures, and the hardware that was available shaped them. The analysis ran once, from
a commit that fixed its inputs and its command, and every cell is reported.

Results on T5, ours against each opponent's weakest row (the variant whose interval lies closest to 0):

| opponent | opponent's failures (rows) | LER(ours) − LER(opponent), per 1,000 shots [97.5 %] | McNemar p | opponent's time, µs per shot | what it means |
|---|---|---|---|---|---|
| plain PM | 15,556 | −12.87 [−13.15, −12.60] | < 10⁻³⁰⁰ | 325.23 | ours fails 83 % less often; plain PM takes 2.1 times as long |
| correlated PM | 2,907 | −0.22 [−0.28, −0.16] | 2.6·10⁻¹⁸ | 714.05 | 7.6 % less often; 4.7 times as long |
| Ising M1 + PM | 12,843-12,873 (5) | −10.16 [−10.41, −9.90] | < 10⁻³⁰⁰ | 161.33 | 79 % less often; 6.6 % longer |
| Ising M4 + PM | 8,081-8,094 (5) | −5.39 [−5.60, −5.19] | < 10⁻³⁰⁰ | 165.97 | 67 % less often; 9.7 % longer |
| Ising M1 + correlated PM | 12,419-12,449 (5) | −9.73 [−9.99, −9.48] | < 10⁻³⁰⁰ | 297.51 | 78 % less often; 2.0 times as long |
| Ising M4 + correlated PM | 7,903-7,918 (5) | −5.22 [−5.42, −5.01] | < 10⁻³⁰⁰ | 275.79 | 66 % less often; 1.8 times as long |
| min-LLR BP-OSD | 2,070 of 200,000 | −7.64 [−8.17, −7.12] | 5.2·10⁻²⁵⁵ | 444,502 | 74 % less often; 2,937 times as long |
| BP 60 BP-OSD | 3,016 of 200,000 | −12.37 [−13.01, −11.75] | < 10⁻³⁰⁰ | 619,209 | 82 % less often; 4,091 times as long |

Ours failed 2,687 times in 1,000,000 shots (LER 0.27 %), 542 times in the BP-OSD subset, and routed 10.08 % of the
shots to correlated PM. A p shown as < 10⁻³⁰⁰ underflows double precision. The times are on the Grace basis; every one
except BP-OSD's contains an extrapolated Grace value (`evaluation/timing/grace_basis.csv`).

Ours takes 151.35 µs per shot: (a) 66.66 µs, the slowest of 5 accepted processes on the pod's H100 (66.62-66.66 µs;
the Triton engine with 4 pipeline stages); (b) 83.14 µs, the routed shots' correlated PM on one core of an exclusive
AMD EPYC 7713 node, measured on 2026-10-06 before any test read (the slowest of three repetitions, 91.26 µs) × 714.05 /
783.78 (the Grace value of correlated PM over the node's own, the median of three); and (c) 1.54 µs, the router
process's handoff estimate on the pod, above the 1.1 µs allowance.

The narrowest margins are correlated PM in accuracy (220 fewer failures per million shots) and Ising M1 + PM in speed
(9.98 µs). That speed margin rests on two conversions to the Grace basis, since neither CPU time was measured on
Grace: Ising M1's PM share (98.73 µs) is the Apple M4 core's time × the d = 13 ratio, and our CPU term is the EPYC
node's time × correlated PM's ratio. Measured again on the same M4 core during the run, that PM share came out at
98.91 µs.

NVIDIA's Ising paper timed correlated PM on Grace at d = 17 (Table IX: 860.71 µs per shot) and plain PM and the Ising
PM shares at d = 21, values the registration missed. With every CPU time converted by one rule at d = 17 on both sides
([evaluation/timing/R17_GRACE_NOTE.md](../../evaluation/timing/R17_GRACE_NOTE.md), sections 2 and 4), (b) = 91.26 × 860.71 / 783.78 = 100.22 µs
and ours takes 168.42 µs, against 170.47 µs for Ising M1 + PM and 173.87 µs for Ising M4 + PM, whose PM shares are
interpolated between Table VII's d = 13 and d = 21 values (107.87 and 89.18 µs). All 8 speed cells hold, these two by
2.05 and 5.45 µs. The result above stays study 4's registered result.

The pod's host was busy: over the session's 2,476 s its 1-minute load average had a median of 20.3 (16.6-58.2) on the
208 CPUs the container saw; the host was 9.9 % busy (median) with no steal, and the timed core 5.8 % busy (median)
outside our own processes. Host noise can only lengthen our GPU term (the slowest of 5 processes counts) and our
handoff. Every opponent part timed on the pod counts its fastest repetition, and the PM times that decide were not
measured on the pod.

On the M4 basis (decides nothing), ours took 104.15 µs (its CPU term measured on the M4 core: 35.95 µs), Ising M1 + PM
115.11 µs and plain PM 124.58 µs. R17 has no M4-basis time. `results/surface/s4_test_eval.csv` holds all 24 rows in the
format of the other studies' files, and `results/surface/results_accuracy_speed_d17.json` holds every decoder's failures
and times, the composite's inputs and the session's host-load summary. Section 7 gives the commands.

### R17: d = 17 re-registered

R17 re-registered study 4's d = 17 row and read it once on a new test set. It kept the network (`d17_b5_r787`), the
deployed engine (FROZEN, 4 pipeline stages), every opponent with its variants, the statistics and the 16 cells, and
changed two things. The threshold became τ = 4.539173126220703, which routes 8 % of the calibration shots (q = 8 %;
study 4's τ = 5.121974468231201 routes 10 %). The test set became T6, 1,000,000 new shots. The plan added one test:
whether accuracy at q = 8 % is preserved against q = 10 % on the same T6 shots. The plan, the result and every record
are in `results/r17/` (`results/r17/PLAN_R17.md`, `results/r17/R17_RESULT.md`).

The pre-stated rule C4.2 takes the smallest q on its grid (0 and 1-30 %) whose calibration failures stay within a
limit derived from correlated PM's, and if no q qualifies, the q with the fewest calibration failures. On a new
calibration set, K17d (200,000 shots), no q met the limit (470.5 failures), so the fallback took q = 8 % (540
failures); study 4's q = 10 % came from the same fallback on its own calibration set. The decision to re-register at
q = 8 % came after an exploratory comparison on E17d (500,000 shots: 1,364 failures at q = 8 % against 1,356 at study
4's τ, and a projected 137.3 µs against 155.8 µs per shot), so R17 tested that choice on new shots under criteria
fixed in advance.

T6's seeds, 91,170,000 + j for chunk j = 0-99, were written into R17's plan in advance; studies 2-4 had derived
theirs from a beacon round after the freeze. No branch of the hub, this release or the paper's repository used that
range (`results/r17/seedcheck.sh`). The plan was frozen and pushed before T6 was drawn, and T6's manifest was committed before
any read.

Results on T6, ours at q = 8 % against each opponent's weakest row (the variant whose interval lies closest to 0):

| opponent | opponent's failures (rows) | LER(ours) − LER(opponent), per million shots [97.5 %] | McNemar p | opponent's time, µs per shot | what it means |
|---|---|---|---|---|---|
| plain PM | 15,658 | −12,950 [−13,227, −12,676] | < 10⁻³⁰⁰ | 325.23 | ours fails 83 % less often; plain PM takes 2.4 times as long |
| correlated PM | 2,943 | −235 [−299, −172] | 9.9·10⁻¹⁷ | 714.05 | 8.0 % less often; 5.3 times as long |
| Ising M1 + PM | 12,840-12,961 (5) | −10,132 [−10,389, −9,878] | < 10⁻³⁰⁰ | 161.33 | 79 % less often; 19 % longer |
| Ising M4 + PM | 8,070-8,110 (5) | −5,362 [−5,570, −5,156] | < 10⁻³⁰⁰ | 165.97 | 66 % less often; 23 % longer |
| Ising M1 + correlated PM | 12,405-12,524 (5) | −9,697 [−9,950, −9,447] | < 10⁻³⁰⁰ | 297.51 | 78 % less often; 2.2 times as long |
| Ising M4 + correlated PM | 7,877-7,955 (5) | −5,169 [−5,375, −4,966] | < 10⁻³⁰⁰ | 275.79 | 66 % less often; 2.0 times as long |
| min-LLR BP-OSD | 2,114 of 200,000 | −8,060 [−8,593, −7,541] | 7.7·10⁻²⁸⁷ | 444,502 | 76 % less often; 3,292 times as long |
| BP 60 BP-OSD | 3,141 of 200,000 | −13,195 [−13,839, −12,567] | < 10⁻³⁰⁰ | 619,209 | 84 % less often; 4,586 times as long |

Ours failed 2,708 times in 1,000,000 shots (LER 0.27 %) and 502 times in the BP-OSD subset, and routed 7.97 % of the
shots (79,652) to correlated PM. All 24 rows are "better", so the 8 accuracy cells hold; the narrowest is correlated
PM, 235 fewer failures per million shots.

On the same T6 shots, study 4's τ routed 10.08 % and failed 2,698 times, so q = 8 % failed 10 more times per million
(113 shots failed only at q = 8 %, 103 only at q = 10 %; McNemar p = 0.54). The 97.5 % interval of LER(q = 8 %) −
LER(q = 10 %) is [−23.3, +43.3] per million. Its upper end lies below the registered tolerance δ = 81.7 per million,
half of study 4's established advantage over correlated PM and about 3 % (3.04 %) of study 4's LER. Accuracy at
q = 8 % is therefore preserved within that tolerance: T6 rules out a loss of more than 81.7 failures per million shots
against q = 10 %, while a smaller loss remains possible, so the result does not show that accuracy is unchanged.

Ours takes 135.02 µs per shot on the registered composite: (a) 66.66 µs + (b) 66.82 µs + (c) 1.54 µs. R17 measured
only (b), on validation shots (V17b): the routed shots' correlated PM on core 8 of one exclusive AMD EPYC 7713 node, in
three repetitions, the slowest (73.04 µs) × 714.05 / 780.54 (the Grace value of correlated PM over the node's own, the
median of three). The GPU term (a) and the handoff estimate (c) are study 4's. Stage E3 had timed a persistent engine,
eng17, as FROZEN's replacement and ended NO-GO (74.57 µs against FROZEN's 67.20 µs in the same session), so R17 kept
FROZEN and its registered terms (`results/r17/e3/E3_SUMMARY.md`). The opponents' times are study 4's registered constants from
its sessions of 2026-10-06 and 2026-10-07, with the host noise and Grace conversions described above. R17's speed
numbers are therefore validation-based Grace-basis composite scores whose terms come from different sessions.

R17's composite is 16.33 µs below study 4's 151.35 µs. Of that, 14.89 µs is the CPU saving measured on one node: the
q = 10 % term, timed beside the q = 8 % term in the same job, converts to 81.70 µs. The other 1.44 µs is the gap
between study 4's CPU session (83.14 µs) and R17's node at q = 10 %. All 8 speed margins are positive, the narrowest
being Ising M1 + PM at 26.31 µs (study 4: 9.98 µs). E3 timed FROZEN at 67.20 µs in its own session, 0.54 µs above
(a); with that value ours would be 135.56 µs and the narrowest margin 25.77 µs. The +0.54 µs is a sensitivity, neither
an uncertainty bound nor an end-to-end measurement, and no margin is smaller.

These margins are on the registered conversion, which takes both sides from NVIDIA's d = 13 values. With every CPU time
converted by one rule at d = 17 instead (NVIDIA's measured correlated PM, 860.71 µs, and Table VII's plain PM and Ising
PM shares interpolated between d = 13 and 21; [evaluation/timing/R17_GRACE_NOTE.md](../../evaluation/timing/R17_GRACE_NOTE.md)), ours takes
148.74 µs, the narrowest margin is 21.73 µs over Ising M1 + PM (21.19 µs with the +0.54 µs), and all 8 cells hold. On
that basis R17's composite is 19.68 µs below study 4's 168.42 µs.

The GPU read (ours, plain and correlated PM, and every Ising part) ran on one RunPod H100 80GB HBM3 pod, the GPU type
of study 4's reads, after the same job had reproduced S4-R3's H100 outputs bit for bit. The Ising parts were checked on
T5 chunks 0-19 (study 4's test set, already analysed), because no per-shot H100 outputs of the plans exist on
validation shots; ours was checked on all 200,000 V17b shots against study 4's development record. BP-OSD read the
first 200,000 shots of T6 on Alpine A100 80GB cards (study 4 registered A100s; S4-R3 used H100 pods).

One departure from the plan: the plan put the CPU term before any read of T6, but the three BP-OSD shares started
while the CPU job was still timing, and their decoding began 71 s (chunks 0-6) and about 100 s (the other two shares)
before it ended. The CPU term timed validation shots on an exclusive node that BP-OSD did not use, so the overlap does
not touch it (`results/r17/R17_RESULT.md` section 6).

All 16 cells hold and accuracy is preserved within the tolerance. That is outcome 1 of the plan's pre-stated outcomes,
so the d = 17 rows of the tables above are R17's. R17's analysis output is published whole: `results/surface/r17_test_eval.csv`
(all 24 rows and the preservation row), `results/surface/r17_summary.json` (the cells, counts and composite inputs),
`evaluation/testsets/T6.json` (T6's manifest), and in `results/r17/` the inventory that the analysis checked its inputs against, the
analysis log and every job and pod record. Section 8 gives the commands.

### The two post hoc controls: Ising + correlated PM

The registered Ising rows decode the pre-decoder's residual detectors with plain PM, as NVIDIA's paper does, while our
routing sends its few uncertain shots to correlated PM. After both studies had been read and published, an external
review asked for controls with the same post-processor on both sides. A post hoc supplement to the two studies answers
it with two controls: NVIDIA's M1 and M4, each followed by correlated PM on the residual detectors, the post-processor
our routing uses. They were measured after studies 2 and 3 on the same sealed test sets, with every read of those sets
logged, and an independent audit checked their results. Their plan was fixed only after the studies, so they are not
registered claims, and the comparison is descriptive. At d = 17 study 4 and
R17 registered the same two decoders as opponents (the two previous subsections).

Failures on the same shots (2,000,000 at d = 7, 1,000,000 otherwise), and the paired comparison with ours by the
registered method (the exact McNemar test and the 97.5 % Newcombe interval of LER(ours) − LER(control)):

| d | ours | Ising M1 + correlated PM | Ising M4 + correlated PM | LER(ours) − LER(M1 + corr. PM), per 1,000 shots [97.5 %] | LER(ours) − LER(M4 + corr. PM), per 1,000 shots [97.5 %] | what it means |
|---|---|---|---|---|---|---|
| 7 | **33,582** | 49,061 | 38,154 | −7.74 [−7.96, −7.52] | −2.29 [−2.48, −2.09] | ours fails 12 % less often than the closer control (M4) |
| 9 | **10,789** | 22,851 | 16,972 | −12.06 [−12.37, −11.75] | −6.18 [−6.45, −5.92] | 36 % less often than M4 + corr. PM |
| 11 | **7,215** | 19,989 | 14,829 | −12.77 [−13.08, −12.48] | −7.61 [−7.87, −7.36] | 51 % less often |
| 13 | **5,627** | 17,105 | 12,255 | −11.48 [−11.76, −11.19] | −6.63 [−6.87, −6.39] | 54 % less often |
| 15 | **3,911** | 14,845 | 9,916 | −10.93 [−11.21, −10.66] | −6.00 [−6.23, −5.78] | 61 % less often |

All 10 cells are "better" for ours: every interval lies below 0, and every McNemar p is below 10⁻¹⁴⁶ (1.5·10⁻¹⁴⁷ at
d = 7 against M4; elsewhere it underflows double precision). The supplement also adjusted its p-values with Holm's
method over its 20 primary cells, and no verdict changed. `results/surface/controls_test_eval.csv` holds every row in the
format of `results/surface/s2_test_eval.csv`.

Speed in µs per shot on the Grace basis:

| d | ours, registered composite | ours without the handoff (c) | Ising M1 + correlated PM | Ising M4 + correlated PM | Ising M1 / M4 + PM, registered | what it means |
|---|---|---|---|---|---|---|
| 7 | **12.92** | 10.68 | 53.34 (55.48) | 65.07 (66.93) | 48.76 / 61.41 | ours is 40.4 µs below the faster control |
| 9 | **30.05** | 28.95 | 68.45 (73.53) | 78.23 (82.55) | 56.40 / 68.08 | 38.4 µs below |
| 11 | **43.93** | 42.83 | 97.05 (108.01) | 100.97 (109.98) | 68.63 / 78.67 | 53.1 µs below |
| 13 | **57.96** | 56.86 | 144.02 (164.06) | 139.84 (156.01) | 90.26 / 98.34 | 81.9 µs below |
| 15 | **86.08** | 84.75 | 215.90 (249.91) | 203.29 (231.13) | 122.85 / 129.63 | 117.2 µs below |

A control's speed is Ising's registered TensorRT plan time at batch 1 (the same plan as its registered row) plus its
correlated-PM share on the Grace basis. The share is the time correlated PM takes per shot on Ising's residual
detectors of validation chunk 0 (10,000 shots), measured on one pinned core of an exclusive AMD EPYC 9534 node, the
median of three repetitions. It is converted with the same node's plain PM on the same model's residuals at d = 13:
× 0.9864 after M1 and × 0.9847 after M4, the paper's Grace PM share after the model at d = 13 (39.65 / 33.15 µs, Table
VII) divided by the node's (40.20 / 33.66 µs). In brackets: the times under the other factor, × 1.198 (Grace
correlated PM at d = 13, 279.63 µs from Table IX, divided by the node's 233.41 µs). The paper gives no correlated-PM
time after its pre-decoders, so both factors are normalisation assumptions; ours stays faster under both, on both
boundaries, in all 40 speed cells. Ising's rows carry no handoff, so the controls' values are the same on both
boundaries. These speeds are composite estimates: the accuracy comes from the official PyTorch numerics and the time
from the TensorRT plan, and no run showed both in one decoder.

On the M4 basis, which decides nothing, a control's speed is the same plan time plus its
correlated-PM share measured on one Apple M4 core, unconverted: from 48.38 / 60.81 µs (M1 / M4) at d = 7 to
140.82 / 142.19 µs at d = 15, against ours' 11.69 and 66.63 µs there (section 6, step 6).

What else the controls show:

- Correlated PM barely changes Ising's accuracy. Against the same model's PyTorch numerics with plain PM (the
  registered parts `M1_torch` / `M4_torch`), it moves the failures by at most 0.44 per 1,000 shots, and not always
  down: at d = 7, M1 + correlated PM fails on 186 more shots (49,061 against 48,875), and at d = 15 on 443 fewer
  (14,845 against 15,288). Correlated PM costs 1.5 to 2.5 times as much as plain PM on the same residuals.
- Both controls fail more often than correlated PM alone at every d (at d = 15, 9,916 against 4,311 for M4), which
  agrees with what NVIDIA's paper says about its models 1-5.
- The same supplement ran the reverse control: our network routing its uncertain shots to plain PM, with its threshold
  refitted on calibration data, against Ising + plain PM. There ours fails less often in 9 of 10 cells; at d = 7
  against M4 it fails more often (41,196 against 38,095 of 2,000,000 shots, 1.55 more per 1,000). Its rows are in
  `results/surface/controls_reverse_eval.csv`, and section 6 gives the command that reruns it.

## The decoder

The network (`evaluation/convlines/net.py`) places every detector of the d-round circuit on a T × H × W grid (time, two space
coordinates; 336 detectors on 8 × 8 × 8 at d = 7, 3,360 on 16 × 16 × 16 at d = 15, 4,896 on 18 × 18 × 18 at d = 17). A
type-aware 3 × 5 × 5 convolution (separate weights for X- and Z-type sites) embeds the detector bits; L pre-norm
residual blocks with a type-aware 3 × 3 × 3 convolution follow. The read-out has one query per logical line (row j of
data qubits): each query pools the site features with sigmoid gates biased by the site's distance to its line, and an
MLP gives one logit z_j per line. Since every line's flip equals the observable's flip XOR a known parity g_j of the
detectors, the output is c = Σ_j w_j (−1)^{g_j} z_j + b₀.

The routing rule (`evaluation/convlines/decoder.py`) sends a shot with |c| < τ to correlated PM and decides any other shot by
c > 0. τ is the q-quantile of the deployed executable's |c| on 200,000 separate calibration shots (seeds
60,000,000 + 10,000 d + j at d = 7-13, 64,000,000 + 150,000 + j at d = 15 and 68,000,000 + 170,000 + j at d = 17,
j = 0-19), fitted before any test shot existed.

The six networks (`weights/surface/`; a name reads d{distance}_b{blocks}, with w64 for width 64, and the replica's number):

| d | network | blocks × width | parameters | q | τ | calibration failures: network alone / correlated PM / routed | deployed executable |
|---|---|---|---|---|---|---|---|
| 7 | d7_b1w64_r740 | 1 × 64 | 90,257 | 4 % | 1.3934324979782104 | 4,260 / 3,433 / 3,204 | hand-written CUDA kernel `evaluation/timing/d7/` |
| 9 | d9_b3_r764 | 3 × 128 | 389,395 | 1 % | 0.6820775270462036 | 2,484 / 2,562 / 2,188 | Triton engine `evaluation/timing/engine.py` |
| 11 | d11_b5_r777 | 5 × 128 | 517,669 | 1 % | 1.0014245510101318 | 1,682 / 1,855 / 1,408 | Triton engine |
| 13 | d13_b5_r775 | 5 × 128 | 518,007 | 2 % | 1.8486117124557495 | 1,626 / 1,259 / 1,082 | Triton engine |
| 15 | d15_b5_r782 | 5 × 128 | 518,345 | 6 % | 3.9547388553619385 | 1,737 / 891 / 806 | Triton engine |
| 17 | d17_b5_r787 | 5 × 128 | 518,683 | 10 % | 5.121974468231201 | 1,719 / 615 / 560 | Triton engine, 4 pipeline stages |

On the calibration shots the network alone fails more often than correlated PM at d = 7, 13, 15 and 17, and the
combination fails less often than either. On the test sets the routed fraction was 4.06 %, 0.99 %, 0.99 %, 1.94 % and
6.01 % at d = 7, 9, 11, 13 and 15; at d = 17 it was 10.08 % with study 4's τ on T5 and 7.97 % with R17's on T6. For
study 4 at d = 17 the rule's grid reached q = 30 %, no q met its calibration-failure limit (487.4), and its fallback
took the q with the fewest calibration failures. R17 re-registered d = 17 at q = 8 %, τ = 4.539173126220703
(`results/r17/policy_d17b5_r787_q08.json`; [R17](#r17-d--17-re-registered) says how it was chosen). `weights/surface/configs.json`
keeps study 4's τ, which `--routing deployed` uses.

The weights are the exponential moving averages at the last training step, stored as fp32 state dicts
(safetensors) with the grid of each distance (`weights/surface/grid_d*.npz`) and their configuration (`weights/surface/configs.json`:
architecture, grid, d, τ at full precision, q, the policy and the sha256 of the original checkpoint and deployed
artifact). `weights/MANIFEST.sha256` lists every file's sha256 (one manifest for all networks of this repository, paths
relative to `weights/`). The networks were trained on fresh Stim samples of the
same circuits (2·10⁸ samples in the last stage, 10⁸ for the d = 15 extension), the larger ones initialised from
smaller-distance or shallower networks (d = 17 from the d = 15 network that d15_b5_r782 continued); the training code
is not part of this release.

The d = 17 network is here in its deployed form only. `weights/surface/d17_b5_r787_deployed.safetensors` holds the deploy
tables of study 4's deployed artifact (sha256 bc00eb30…), every tensor bit for bit, and `evaluation/timing/engine.py` runs them.
Its fp32 state dict (the final EMA checkpoint, sha256 3627e743…) is not in this release, so the PyTorch reference
(`convlines.decoder`, `--routing fp32` and `published`) does not run at d = 17; our d = 17 decisions need an NVIDIA GPU
(`--routing deployed`).

## Reproduction

### 1. Environment

Tested with Python 3.12.14 on Linux x86-64 (`requirements.txt`):

```bash
cd evaluation
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126    # the studies' build; a CPU wheel runs convlines/
pip install -r ../requirements.txt
```

numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, torch 2.11.0 (+cu126, with Triton 3.6.0), safetensors 0.8.0
and matplotlib 3.11.2. The opponents need more (section 3). Run every command of this document from `evaluation/`;
outputs (`shots/`, `masks/`, `timing_out/`, `cpu_out/`, `build/` and so on) are written there.

### 2. Circuits and test sets

`evaluation/circuits/d{d}_legacy.stim` are the verdict circuits as Stim text: NVIDIA Ising-Decoding's `MemoryCircuit` at commit
3ce4379 with `noise_model=None` (the legacy single-p path), `code_rotation='XV'`, all detectors plus boundary
detectors, n_rounds = d, p = 0.006. `evaluation/convlines/circuits.py` checks each file against its registered sha256 (the
sha256 of the text). `evaluation/scripts/make_circuits.py` rebuilds the six files and the network grids from NVIDIA's code, for
anyone who wants to check them.

A test set is a list of chunks of 10,000 shots. Chunk j at distance d is

```python
det, obs = circuit.compile_detector_sampler(seed=seed(d, j)).sample(10_000, separate_observables=True)
```

stored byte-deterministically as `d{d}_p0.006_X_test2_{j:03d}.npz` (T3: `test3`; T5 keeps study 4's frozen name
`test4`; T6: `test6`): `det` = np.packbits of the detector bits per shot, `obs` = observable 0, `seed`; zip members
uncompressed in name order with a fixed 1980-01-01 timestamp, so a file's sha256 is a function of the shots.

| set | study | distances | chunks | shots | drand round | size on disk |
|---|---|---|---|---|---|---|
| T2 | 2 | 7 / 9, 11, 13 | 200 / 100 each | 2,000,000 / 1,000,000 each | 6,520,833 | about 0.6 GB |
| T3 | 3 | 15 | 100 | 1,000,000 (BP-OSD: chunks 0-19) | 6,524,587 | about 0.4 GB |
| T5 | 4 | 17 | 100 | 1,000,000 (BP-OSD: chunks 0-19) | 6,532,408 | about 0.6 GB |
| T6 | R17 | 17 | 100 | 1,000,000 (BP-OSD: chunks 0-19) | none: seed 91,170,000 + j | about 0.6 GB |

The manifests `evaluation/testsets/T2.json`, `evaluation/testsets/T3.json`, `evaluation/testsets/T5.json` and `evaluation/testsets/T6.json` list every chunk with
its seed and sha256. T6's seeds are not derived (section 8); for T2, T3 and T5 the seeds were derived after the
freeze, so nobody could know the test shots before the decoders were fixed:

    seed(d, j) = 2^62 + (first 8 bytes of sha256(TAG ‖ A ‖ R ‖ d (1 byte) ‖ j (2 bytes, big-endian)) mod 2^62)

TAG is the manifest's `tag`, A the bytes of the study's approval record (made in the freeze commit; for T5, the record
of the re-registration S4-R3; its sha256 is in the manifest) and R the 32 randomness bytes of the drand mainnet beacon
(League of Entropy) round fixed by the freeze commit's time (for T5, the time the S4-R3 commit was public): the first
round at least 600 s later. The round, its signature and randomness = sha256(signature) are in the manifest, and any
drand endpoint (for example `https://api.drand.sh/<chain_hash>/public/<round>`) serves the same round. The approval
records themselves are not published here, so `python -m convlines.testset seeds` (which re-derives every seed) needs
them; regenerating the shots needs only the manifest's seeds.

```bash
python -m convlines.testset regen --set T2 --d 9 --chunks 0-99 --out shots/T2     # sample, write, compare each sha256
python -m convlines.testset regen --set T3 --d 15 --chunks 0-99 --out shots/T3
python -m convlines.testset regen --set T5 --d 17 --chunks 0-99 --out shots/T5
python -m convlines.testset check --set T2 --dir shots/T2                         # re-hash existing files
```

Bit-for-bit regeneration needs stim 1.16.0 and numpy 2.2.6 on an x86-64 CPU with AVX2: Stim's sampler output depends
on its version and SIMD build, and the sets were sampled on x86-64 CPUs (T2: AMD EPYC 9534, T3: AMD EPYC 7543, T5:
Intel Xeon Platinum 8470, T6: AMD EPYC 9534).

### 3. Accuracy

```bash
python -m convlines.decoder --d 9 --chunk shots/T2/d9_p0.006_X_test2_000.npz       # one chunk: failures, routed shots
python -m convlines.evaluate --set T2 --d 9 --chunks 0-99 --shots shots/T2 \
    --opponents "plain PM" "correlated PM" --out d9.json                            # ours against opponents, with statistics
```

`convlines.evaluate` decodes chunk by chunk (each file checked against the manifest first), then compares ours with
every opponent on the same shots with `evaluation/convlines/stats.py`: the exact two-sided McNemar test on the shots exactly one
decoder fails, and Newcombe's (1998) method-8 interval of the LER difference (Wilson limits of each failure rate
combined with the signed phi coefficient of the paired outcomes) at 97.5 %. A cell is "better" when p < 0.025 and the
interval lies below 0. The registered claims are intersection-union tests over all their cells, so no multiplicity
correction applies.

The opponents (`evaluation/convlines/opponents.py`):

- Plain and correlated PM need nothing more: PyMatching on the circuit's DEM (`decompose_errors=True`,
  `approximate_disjoint_errors=True`).
- Ising M1 / M4 + PM run NVIDIA's own inference module (`PreDecoderMemoryEvalModule`: the pre-decoder's logical
  frame and its residual detectors), then plain PM on the residual; flip = frame XOR PM. Clone NVIDIA's repository at
  the studies' commit, fetch its two checkpoints and install its inference requirements, then point
  `ISING_DECODING_DIR` at the clone:

  ```bash
  git clone https://github.com/NVIDIA/Ising-Decoding && cd Ising-Decoding
  git checkout 3ce4379fa6cf475789007bde4b6a8206e8622992 && git lfs pull
  pip install hydra-core==1.3.7 omegaconf==2.3.1 ldpc==2.4.1 beliefmatching==0.2.0   # code/requirements_public_inference.txt beyond ours
  export ISING_DECODING_DIR=$PWD
  ```

  The checkpoints must have sha256 08ce4b1e… (Fast, M1) and f49df7fc… (Accurate, M4); the code checks both.
- Ising's TensorRT plans: the studies also read on the test sets every TensorRT plan of Ising they timed (part names
  `M*_fp*_L{level}_B1`) and the plans of a batch sweep (`M*_fp*_B{B}`): each model in FP16 and FP8, built for one
  fixed batch size B between 1 and 64, kept where its outputs on the validation shots differed from the smaller
  batches'. TensorRT picks different kernels for different batch sizes, so the parts round differently and their
  failure counts differ by a few dozen per million. `results/surface/s*_test_eval.csv` and `results/surface/r17_test_eval.csv` have
  one row per part, and the Ising columns of the accuracy table show each model's fewest failures over its parts (the
  official PyTorch numerics included):

  | d | Ising M1 + PM: part (failures) | Ising M4 + PM: part (failures) | what it means |
  |---|---|---|---|
  | 7 | `M1_torch` (48,875) | `M4_fp8_B1` (38,000) | the PyTorch numerics and a batch-1 FP8 sweep plan |
  | 9 | `M1_fp8_B1` (22,841) | `M4_fp16_B16` (17,001) | sweep plans |
  | 11 | `M1_fp16_L5_B1` (20,261) | `M4_fp8_B32` (14,817) | a timed plan and a sweep plan |
  | 13 | `M1_fp16_L5_B1` (17,478) | `M4_fp8_B16` (12,399) | a timed plan and a sweep plan |
  | 15 | `M1_fp8_B1` (15,232) | `M4_fp16_B64` (10,077) | sweep plans; at d = 15 M1's two timed FP8 plans tie at 15,232 |
  | 17 | `M1_fp8_L3_B1` (12,840) | `M4_fp8_L5_B1` (8,070) | R17 on T6: timed plans (R17 and study 4 read no sweep plans); with correlated PM the fewest are `M1_fp8_L3_B1_corr` (12,405) and `M4_fp8_L5_B1_corr` (7,877). Study 4 on T5: `M1_fp8_L3_B1` (12,843), `M4_fp8_L3_B1` (8,081), `M1_fp8_L3_B1_corr` (12,419), `M4_fp16_L3_B1_corr` (7,903) |

  `evaluation/timing/ising_trt.py` builds every part a study read at a distance and counts its failures on the test chunks
  (exactly B shots per call, the last call padded with all-zero shots, then plain PM on the residual detectors, as the
  studies' read did); it needs the TensorRT environment of section 4:

  ```bash
  python -m timing.ising_trt prep  --d 15 --val shots/val --onnx onnx           # ONNX export, FP8 quantisation
  python -m timing.ising_trt build --d 15 --onnx onnx --work trt --parts registered
  python -m timing.ising_trt eval  --d 15 --onnx onnx --work trt --parts registered --set T3 --shots shots/T3 \
      --chunks 0-99 --out ising_eval                                            # failures per part
  ```

  A plan's kernels depend on the GPU and on the build's own timing measurements, so a plan rebuilt on another machine
  can decide a few shots per million differently from the studies' plan (the section "Verification of the earlier release"
  compares two). The registered counts are those of the studies' plans. The studies' sweep traced its own ONNX export
  on an all-zero shot; `build --parts` uses `prep`'s file instead, which has the same nodes, weights, inputs and
  outputs.
- BP-OSD needs an NVIDIA GPU and `cudaq-qec-cu12==0.8.0` (the studies used a separate environment with numpy
  2.5.3); the settings are in `evaluation/convlines/opponents.py`. At d = 15 and 17 run it on `--chunks 0-19`, the registered
  subset.

#### Our decisions: the deployed routing

The registered decisions of our decoder came from the deployed executables of section 4 (the Triton engine at
d = 9-17, the hand-written kernel at d = 7), which compute c with fp16 matrix inputs and fp32 accumulation, one shot
at a time. `convlines.evaluate --routing` chooses where our c and our routing come from:

| `--routing` | c from | routed shots | needs | what it gives |
|---|---|---|---|---|
| `fp32` (default) | the trained fp32 network (`convlines.decoder`) | \|c\| < τ | a CPU | the registered decisions up to rounding: a shot whose \|c\| lies within rounding of τ can be routed differently, and in principle decided differently |
| `deployed` | the deployed executable (`evaluation/timing/deployed.py`) | \|c\| < τ | an NVIDIA GPU; at d = 7 an sm_90 GPU (H100, H200) and `evaluation/timing/d7/build.sh` | the registered decisions; c bit for bit, or within a few float32 ulps where the rebuilt deploy tables differ (section 4) |
| `published` | the fp32 network, for the sign | the studies' mask in `evaluation/routing/` | a CPU | the registered routing; a decision can differ only where an unrouted shot's fp32 sign differs from the deployed one |

At d = 17 only `deployed` runs: this release holds the d = 17 network's deploy tables, not its fp32 network.

`evaluation/routing/` holds the deployed routing the studies recorded, one value per shot (1 = sent to correlated PM), with
`evaluation/routing/manifest.json`. The test-set masks cover every shot of T2, T3 and T5 in chunk order; they come from the
per-chunk records of the studies' test reads, and the manifest gives the sha256 the studies registered for those records
(the sha256 of the chunk files' sha256 strings in chunk order) and the routed and failure counts. The validation masks
are the files the CPU term read (section 4, (b)), byte for byte, with the sha256 the studies' freezes pinned; at d = 17
the CPU term read its mask from the development record that study 4's freeze pinned, and `evaluation/routing/val_d17.npz` is that
mask written as a file of its own. Every chunk that `convlines.evaluate` decodes also reports how many shots its routing
sends differently from the published mask.

```bash
python -m convlines.evaluate --set T2 --d 9 --chunks 0-99 --shots shots/T2 --routing deployed --out d9.json
python -m convlines.evaluate --set T2 --d 7 --chunks 0-199 --shots shots/T2 --routing deployed --build build/d7
python -m convlines.evaluate --set T3 --d 15 --chunks 0-99 --shots shots/T3 --routing published --out d15.json
```

At d = 7, `python -m timing.d7.run` runs the whole deployed executable (the kernel, the compiled router and its
compiled correlated PM) on chunk files (section 4).

### 4. Speed

The registered speed score of our decoder is a composite per shot at batch 1:

    composite = (a) GPU term + (b) CPU term on the Grace basis + (c) handoff

| d | (a) GPU | routed fraction (val) | (b) on the CPU node | ratio | (b) Grace basis | (c) handoff | composite | what it means |
|---|---|---|---|---|---|---|---|---|
| 7 | 8.443 | 4.05 % | 2.342 | 0.9566 | 2.240 | 2.240, measured | 12.924 | the d = 7 kernel's own handoff |
| 9 | 27.907 | 1.03 % | 1.277 | 0.8164 | 1.042 | 1.100, allowance | 30.049 | the allowance exceeded the estimate (0.586) |
| 11 | 40.613 | 1.02 % | 2.340 | 0.9453 | 2.212 | 1.100, allowance | 43.925 | estimate 0.589 |
| 13 | 49.153 | 2.00 % | 7.961 | 0.9678 | 7.705 | 1.100, allowance | 57.957 | estimate 0.299 |
| 15 | 50.885 | 5.91 % | 28.642 | 1.1822 | 33.860 | 1.335, estimate | 86.080 | the estimate exceeded the allowance |
| 17 | 66.660 | 9.98 % | 91.262 | 0.9110 | 83.142 | 1.543, estimate | 151.345 | study 4; the estimate exceeded the allowance |
| 17 | 66.660 | 7.88 % | 73.038 | 0.9148 | 66.816 | 1.543, estimate | 135.019 | R17: (a) and (c) are study 4's, (b) was measured again |

All times in µs per shot; `python -m timing.composite` recomputes every composite and all 38 speed cells (30 of
studies 2 and 3, 8 of study 4) from `results/surface/` and `evaluation/timing/grace_basis.csv`. It does not read R17's composite, which
is in `results/surface/r17_summary.json` ("ours_speed") with its inputs.

#### (a) GPU term

Code: `evaluation/timing/gpu_term.py`, `evaluation/timing/engine.py`. The studies used one NVIDIA H100 80GB HBM3 (SXM) in a rented cloud node,
one session per study (study 2 on 2026-10-03, study 3 on 2026-10-04/05, study 4 on 2026-10-07 on a single-GPU pod). The
input bits are already on the GPU; batch 1; each shot is one replay of a CUDA graph of the engine's call. Per process:
100 graphs captured on 100 input slots, then every one of the 200,000 validation shots in order, 100 back-to-back
replays between two CUDA events per block of 100 shots, two warm-up blocks; the process's value is the mean per shot.
(At d = 7 the kernel stays resident and has no graph to replay; its (a) is the device-side time from the kernel's
detection of a shot's request to the publication of its result, read from the kernel's own timer stamps, over the same
200,000 shots.) Five fresh processes per network, interleaved across networks; the slowest counts. Each process sampled
nvidia-smi every 100 ms on other CPU cores: it was accepted only with every SM-clock sample at least 97 % of the
maximum, no throttle reason other than idle, and no other process on the GPU (at most two replacement processes per
network). In studies 2 and 3, before any timing the host had to pass a frozen quality gate (subsection "Qualifying a
timing session"); study 4's session had the GPU-side checks only.

```bash
python -m convlines.testset val --d 9 --chunks 0-19 --out shots/val                # the timing shots (validation, not test)
python -m timing.deployed --d 9 --val shots/val --out masks                         # the deployed c and routing, once
for p in 0 1 2 3 4; do python -m timing.gpu_term --d 9 --val shots/val --proc $p --deployed masks/val_d9.npz --out timing_out; done
```

`timing.deployed` runs the deployed executable once over the 200,000 validation shots and saves its c and routing
(`masks/val_d9.npz`). Each `timing.gpu_term` process checks its own outputs bit for bit against that file (the first
2,000 shots; at d = 7 every shot), as the studies checked theirs against a development record, and writes its
acceptance (the subsection "Qualifying a timing session").

The validation chunks use the studies' seeds (40,000,000 + 10,000 d + j at d = 7-13, 65,000,000 + 150,000 + j at d = 15,
69,000,000 + 170,000 + j at d = 17). The d = 7-13 chunks were originally sampled on an ARM64 machine, so on x86-64 the
same seeds can give different shots; the timing depends on their statistics only, and the routed set follows from the
network.

`evaluation/timing/engine.py` is the production Triton engine, its code unchanged (only comments and docstrings were edited). The
studies ran it at d = 9-15 with its one frozen configuration (3 pipeline stages) on deploy tables folded from the
weights on the CPU in fp32 (`convlines.net.deploy_tables`). Rebuilt on another CPU, those tables equal the deployed
artifacts' bit for bit except the folded stem table, which differs by at most 1.2·10⁻⁶ (one or two float32 ulps, from
the CPU convolution's summation order). Study 4 ran the engine at d = 17 with the same configuration but 4 pipeline
stages, on its deployed artifact's tables, which `weights/surface/` holds unchanged; `timing.deployed` and `timing.gpu_term`
take the stage count from `weights/surface/configs.json`.

#### The d = 7 executable

At d = 7 the studies used a different executable, in `evaluation/timing/d7/`. A hand-written CUDA kernel for sm_90a
(`kernel.cu`, `dtc.cuh`) stays resident on the GPU as one cluster of seven thread blocks, keeps the weights in shared
memory and serves shots through mapped host memory. A compiled C++ router (`router.cc`) feeds it the shots and sends
the routed ones to correlated PyMatching, which is compiled from PyMatching's v2.4.0 sources into the same library.
`engine.py` and `cudrv.py` drive both from Python through ctypes. The kernel and the router are the studies' code with
only comments edited; the Python driver keeps only the mode the studies ran, and the identifiers keep their
development names (`s7_*`, `s9_*`). With the Triton engine instead, d = 7's GPU term on the H100 is about 17.2-17.3 µs
(a development measurement), which would put the d = 7 composite above plain PM's 17.16 µs, so the d = 7 speed result
depends on this kernel.

`build.sh` fetches PyMatching 2.4.0 and Stim (at the commit PyMatching 2.4.0 pins) from GitHub and builds the cubin
and the router library. It needs Linux x86-64, git, g++ 10 or later and nvcc from CUDA 12.0 or later; the kernel runs
on sm_90 GPUs (H100, H200). With nvcc 12.1.105 (CUDA 12.1 Update 1) the cubin is byte for byte the one the studies ran
(sha256 8f4180d7ffe607c0f8ac023afdb6ec2fbce9f9a59a5ff873e71908e067ea2c2d); other nvcc versions can give a different
cubin. The router library's bytes depend on the compiler; one built with g++ 12.2 made the same decisions as the
studies' library on every verification shot (section "Verification of the earlier release").

```bash
bash timing/d7/build.sh build/d7                                          # kernel_sm90a.cubin, libs7router.so
python -m timing.d7.run --chunks shots/T2/d7_p0.006_X_test2_000.npz       # decisions, failures and checks per chunk
python -m convlines.testset val --d 7 --chunks 0-19 --out shots/val
python -m timing.deployed --d 7 --val shots/val --out masks --build build/d7
for p in 0 1 2 3 4; do python -m timing.gpu_term --d 7 --val shots/val --proc $p --deployed masks/val_d7.npz --out timing_out; done   # (a)
for p in 0 1 2 3 4; do python -m timing.gpu_term --d 7 --val shots/val --proc $p --handoff --out timing_out; done   # (c)
```

The router library `libs7router.so` also drives the Triton engine in the handoff measurement at d = 9-15 ((c) below).

#### (b) CPU term

Code: `evaluation/timing/cpu_term.py`. The validation shots the deployed decoder routes are decoded one at a time by correlated PM
in a single-threaded process pinned to one core of one exclusive CPU node (study 2: AMD EPYC 7713, study 3: AMD EPYC
7443, study 4: AMD EPYC 7713): 50 warm-up decodes, then `perf_counter` around each `decode`. The term is the sum of
those times over all 200,000 validation shots, i.e. the routed fraction times the mean cost of a routed shot. Three
fresh-process repetitions, the slowest counts.

The routed shots come from a mask computed once, before the repetitions, as in the studies, whose CPU term read the
masks now in `evaluation/routing/val_d*.npz` (the deployed executables' routing of the studies' validation shots: 8,095, 2,051,
2,040, 4,000, 11,822 and 19,957 of 200,000 at d = 7, 9, 11, 13, 15, 17; R17's mask at q = 8 % routes 15,759, and
`results/r17/run/records/cpu_J23/mask_q08.json` records its sha256, not the mask). `--mask` takes either that mask or the one
`timing.deployed` computes, and `timing.cpu_term` refuses a mask that was not made on the shots it loaded (it compares
the shots' digest). `convlines.testset val` regenerates the studies' validation shots at d = 15 and 17 on x86-64, so
the published masks apply there. At d = 7-13 the studies' validation shots were sampled on an ARM64 machine and x86-64
gives other shots from the same seeds, so use the mask `timing.deployed` computes on your shots; the term depends on
the shots' statistics only. Without a suitable GPU, `timing.deployed --logits fp32` makes the mask from the fp32
network instead; it routes a few near-τ shots differently from the deployed engine (the section "Verification of the
earlier release" counts them).

The term is converted to the Grace basis by the same node's correlated-PM speed:

    (b) on the Grace basis = (b) on the node × G(d) / (the node's correlated PM per block on validation chunk 0, median of 3)

where G(d) is the Grace-basis time of correlated PM per block. The ratio cancels the node's absolute speed to first
order, but it is only as good as G(d).

```bash
python -m timing.deployed --d 9 --val shots/val --out masks                                 # once, on a GPU
for r in 1 2 3; do
  python -m timing.cpu_term term --d 9 --val shots/val --mask masks/val_d9.npz --rep $r --core 8 --out cpu_out  # timed
  python -m timing.cpu_term ref --d 9 --val shots/val --rep $r --core 8 --out cpu_out     # the node's correlated PM per block
done
python -m timing.cpu_term convert --d 9 --out cpu_out                                       # b_node, the ratio, b on the Grace basis
python -m timing.cpu_term term --d 15 --val shots/val --mask routing/val_d15.npz --rep 1 --core 8 --out cpu_out   # the studies' mask
```

#### Where every Grace number comes from

The table is `evaluation/timing/grace_basis.csv`. NVIDIA's Ising paper timed PyMatching on one Grace Neoverse-V2 core at
p = 0.006 at more distances than the registered results use: plain PM and Ising's PM share after M1 and M4 at d = 13,
21 and 31 (Table VII), and correlated PM at d = 5, 9, 13, 17, 21 and 31 (Table IX). The registered results take three
of these places: plain PM at d = 13 (9.97 µs per round, Table VII: 129.61 µs per shot), correlated PM at d = 9 and 13
(7.51 and 21.51 µs per round, Table IX: 67.59 and 279.63 µs), and Ising's PM share after M1 and M4 at d = 13 (3.05
and 2.55 µs per round: 39.65 and 33.15 µs). Every other registered Grace value is an extrapolation:

| quantity | d | Grace value (µs per shot) | how it was obtained |
|---|---|---|---|
| plain PM | 7, 9, 11 | 17.16, 36.99, 72.02 | one Apple M4 core's time × 2.594 (the paper's d = 13 value / the M4's d = 13 time) |
| correlated PM | 7, 11 | 35.71, 155.61 | the M4 core's time × 2.374 (the same ratio for correlated PM) |
| plain / correlated PM | 15 | 208.53 / 464.95 | an exclusive x86 CPU node's time × that node's d = 13 ratio (paper / node); band 206-227 / 450-506 |
| Ising's PM share, M1 / M4 | 7, 9, 11, 15 | M1 5.37, 11.63, 22.69, 65.50; M4 4.93, 9.79, 19.30, 54.92 | PM on Ising's residual detectors timed on the M4 core × 1.880 (M1) / 1.908 (M4), the d = 13 ratios |
| plain / correlated PM | 17 | 325.23 / 714.05 | the same x86 node's time (AMD EPYC 7713) × the same d = 13 ratios; band 319.34-382.25 / 712.03-840.31 |
| Ising's PM share, M1 / M4 | 17 | 98.73 / 81.28 | the M4 core's time × the same d = 13 ratios (52.51 × 1.880, 42.58 × 1.909) |
| correlated-PM share after Ising, M1 / M4 | 17 | 234.91 / 191.11 | study 4: correlated PM on Ising's residual detectors timed on the M4 core (124.94 / 100.11 µs) × the same ratios |

These numbers are also the CPU term's numerator G(d), so at d = 7, 11, 15 and 17 our (b) on the Grace basis is itself
built on an extrapolation.

At d = 17, `evaluation/timing/grace_basis.csv` also lists, beside the registered extrapolations, the values of the
NVIDIA-distance basis, which converts every CPU time by one rule at that distance. Their decoder names end in
"(NVIDIA-distance basis)", and no registered result or script uses them. Correlated PM is NVIDIA's own value, 50.63 µs
per round × 17 = 860.71 µs, 20.5 % above the extrapolation and above its band. Plain PM and the PM shares after M1 and
M4, 313.59, 107.87 and 89.18 µs, are log-log interpolations between Table VII's d = 13 and d = 21 values (628.95,
237.30 and 194.46 µs at d = 21). The correlated-PM shares after M1 and M4, 256.66 and 209.68 µs, are the M4 core's
times × the interpolated share over the M4 core's d = 17 share; NVIDIA gives no correlated-PM share after M1 or M4 at
any distance. [evaluation/timing/R17_GRACE_NOTE.md](../../evaluation/timing/R17_GRACE_NOTE.md) works out what this basis changes at d = 15 and 17.

#### (c) Handoff

The handoff is what a shot costs beyond the GPU work and the CPU term: getting the shot to the GPU, the result back to
the host, and the branch. `timing.gpu_term --handoff` measures it with the studies' code at every distance.

- d = 7: the kernel's own handoff, the host round trip (timing stamps off) minus the device time (stamps on) in the
  same process, on validation chunks 0-1; five processes, the largest counts (2.240 µs).
- d = 9-17: the estimate of the studies' router process, for the Triton engine. The compiled router (`router.cc`, in
  the library that `evaluation/timing/d7/build.sh` builds) runs the engine as the deployed decoder over all 200,000 validation
  shots at the network's τ: per shot the shot's bits are copied into the input of a CUDA graph (untimed), then one
  launch of that graph (the engine's call followed by a small tail kernel that writes the logit, with a launch counter,
  into mapped host memory), the host's wait for that word, the branch |c| < τ and, for a routed shot, correlated PM
  compiled into the same library. Then the engine's device-only time at batch 1 is measured on every 10th of the first
  10,000 shots (per shot the median of 5 replays between CUDA events). The estimate is

      handoff = mean complete path - mean device-only time - routed fraction × mean correlated-PM time of a routed shot

  and the registered (c) is max(1.1 µs, the estimate), from one such process on the session host. The 1.1 µs is a
  fixed allowance; no measurement shows that it bounds the handoff.

| d | complete path | device only | routed fraction × routed PM | estimate | registered (c) | what it means |
|---|---|---|---|---|---|---|
| 9 | 32.848 | 31.018 | 1.03 % × 121.2 | 0.586 | 1.100 | the allowance counted |
| 11 | 46.699 | 43.727 | 1.02 % × 233.6 | 0.589 | 1.100 | the allowance counted |
| 13 | 60.147 | 52.167 | 2.00 % × 384.0 | 0.299 | 1.100 | the allowance counted |
| 15 | 98.654 | 53.922 | 5.91 % × 734.2 | 1.335 | 1.335 | the estimate counted |
| 17 | 174.554 | 69.765 | 9.98 % × 1,034.7 | 1.543 | 1.543 | the estimate counted |

All times in µs per shot, on the sessions' H100 hosts (`results/surface/s2_composite.json`, `c_beside`;
`results/surface/results_accuracy_speed_d15.json` and `results/surface/results_accuracy_speed_d17.json`, `c_measured_detail`). Each
process also checks its routed set and decisions against the deployed logits and the PyMatching wheel, and records the
router's floor (the same loop around a graph without the engine), one of the host gate's measurements.

```bash
for p in 0 1 2 3 4; do python -m timing.gpu_term --d 7 --val shots/val --proc $p --handoff --build build/d7 --out timing_out; done
python -m timing.deployed --d 15 --val shots/val --out masks
python -m timing.gpu_term --d 15 --val shots/val --proc 0 --handoff --build build/d7 --deployed masks/val_d15.npz --out timing_out
```

#### The opponents' times

Ising: its fastest validated TensorRT plan at batch 1 on the session H100 (NVIDIA's
`PreDecoderMemoryEvalModule` exported to ONNX opset 18; TensorRT 10.14.1 FP16 and FP8 plans, the FP8 ones quantised
with ModelOpt on the convolutions and calibrated on 256 validation shots; builder optimisation levels 3 and 5; one
CUDA graph per plan, 200 warm-up replays, then 5 × 100 timed back-to-back replays, as in NVIDIA's trtexec protocol;
the fastest repetition counts), plus its PM share on the Grace basis. The fastest plans were FP8 at every d. Plain
and correlated PM: the Grace values above. BP-OSD: the fastest of three repetitions at batch 1 on the session host, GPU
BP plus host OSD, host to host, not converted; they are 200 to 4,100 times slower than ours, so even a tenth of them
would not change a cell. Study 4 timed Ising's 8 plans and BP-OSD on its pod, whose host was busy; their fastest
repetition counts.

The two post hoc controls add correlated PM after Ising; section 6 measures and converts their share.

`evaluation/timing/ising_trt.py` is our script that exported, built, checked and timed Ising's plans in the studies. A plan
counts only if its check passes, and an FP8 plan only if it also runs every convolution the recipe quantized with an
FP8 kernel. The script enforces this: `check` passes a plan only when both conditions hold and the plan file is the
one its build recorded (same sha256), and `time` times a plan only after a passed check of exactly that file, so a
failed, unchecked or replaced plan is reported as not timed. The script imports NVIDIA's module from
`ISING_DECODING_DIR` as in section 3 and needs the TensorRT environment the studies used. The FP8 calibration reads
validation chunk 0; at d = 7-13 the regenerated chunk can differ from the studies' (sampled on ARM64), and so can the
FP8 plans' calibration scales.

```bash
pip install tensorrt_cu12==10.14.1.48.post1 nvidia-modelopt==0.47.0 onnx==1.21.0 onnxruntime-gpu==1.24.4 \
    onnxscript==0.7.2 onnx_graphsurgeon==0.6.1 polygraphy==0.53.6 onnxslim==0.1.97 onnxconverter-common==1.16.0 \
    onnx-ir==1.0.0 cupy-cuda12x==14.2.0 ml_dtypes==0.6.0 lief==1.0.0 cppimport==26.4.17
python -m convlines.testset val --d 13 --chunks 0 --out shots/val
python -m timing.ising_trt prep  --d 13 --val shots/val --onnx onnx                    # ONNX export, FP8 quantisation
python -m timing.ising_trt build --d 13 --onnx onnx --work trt                         # M1 / M4 x FP16 / FP8 x level 3 / 5
python -m timing.ising_trt check --d 13 --val shots/val --onnx onnx --work trt         # the acceptance test of each plan
python -m timing.ising_trt time  --d 13 --val shots/val --onnx onnx --work trt --order fwd --out ising_b1.csv
python -m timing.ising_trt time  --d 13 --val shots/val --onnx onnx --work trt --order rev --out ising_b1.csv
```

#### Qualifying a timing session

The registered GPU times come from one qualified session per study on a rented H100 node, and the CPU terms from one
job per study on one exclusive CPU node. Every rule below was fixed before the study's freeze, and no number from an
unqualified session or a rejected process counts. Study 4's session under S4-R3 kept rule 4 and the GPU check of rule
3's table (its last row, run at the start and at the end of the session) and recorded the host's load instead of
gating on rules 1-3 (the results subsection on study 4).

1. Quiet host (study 3; added after a study-3 host failed the gate's core check, study 2 had no such step). After the
   node's setup a light sampler read every 30 s the 1-minute load average and the whole host's busy share over those
   30 s (from /proc/stat); the session started only after 20 consecutive samples (10 minutes) with the load at most
   3.0 and the busy share at most 3 %.
2. Screen, which spends no attempt. Windows of 120 s, sampled every 10 s, each on the core the session's rule picks:
   among the cores of the GPU's NUMA node that the job may use, outside core 0's physical core, the one whose physical
   core (with its SMT siblings) was least busy over 1 s. A window holds when that core and each of its siblings were
   at most 5 % busy and the 1-minute load average stayed at most 8.0. The first window that holds ends the screen; if
   none holds within 30 minutes, the host is rejected.
3. Gate, one attempt per host; at most three attempts per study, never two on one host, and no change of host once
   timing had begun. Everything in this table had to hold:

| measurement | limit | what it means |
|---|---|---|
| the pinned core's busy and steal fractions over 5 s (/proc/stat) | busy ≤ 5 %, steal ≤ 1 % | the timed core is idle, and the hypervisor takes no time from it |
| plain PM per block on validation chunk 0 with the CPU term's loop (`timing.cpu_term ref --plain`): 3 repeats at d = 9, 2 at d = 13 | every repeat ≤ 1.10 × the reference host's 35.895 / 133.638 µs; largest / smallest − 1 ≤ 3 % | the core is about as fast as the reference host's, and steady |
| the router's floor: the compiled loop around a CUDA graph without the engine, 200 warm-up and 20,000 timed launches (`floor_mapped` of `timing.gpu_term --handoff` at d = 9-15) | mean ≤ 1.15 × 4.986 µs | launch and completion cost about as much as on the reference host |
| a separate process holding 1 GiB on the GPU and running 3 s of fp16 matrix products, the GPU's clocks sampled every 100 ms | median SM clock ≥ 97 % of its maximum; throttle reasons none or idle; no other process on the GPU before, during or after, and under 64 MiB in use before | the GPU runs at full clock and alone |

   The reference host was the development H100 node of the speed work (Intel Xeon Platinum 8480+, core 1).
4. Every timing process afterwards is accepted or replaced, at most twice per network: no other process on the GPU
   before or after it, every SM-clock sample of its timed parts at least 97 % of the maximum with no throttle reason
   but idle, its outputs equal to the deployed ones, and at d = 7 every device interval shorter than its host round
   trip. `timing.gpu_term` writes this verdict for every process, the handoff processes included (`acceptance`,
   `accepted`).

The CPU node: the CPU term of a study ran as one job holding a whole CPU node (no other jobs on it), after the freeze.
Every repetition was a fresh single-threaded process pinned to one core (core 8) whose SMT sibling stayed idle; the job
ran three interleaved repetitions of the node's correlated PM per block and of every network's term, and the slowest
term and the median per-block time count. The gate's own script is not part of this release; its measurements are the
commands named in the table plus /proc/stat and nvidia-smi readings.

### 5. The figure

The paper's figures are drawn by `results/figures/make_figs.py`, `make_head_fig.py` and `make_why_fig.py` from numbers
copied from the records (see [`../../README.md`](../../README.md)). R17 timed its CPU term on an x86 node only, so ours
has no M4-basis time at d = 17.

### 6. The two post hoc controls

These steps rebuild the two post hoc controls from the test sets: the failures of Ising M1 / M4 followed by correlated PM,
and their speed on the Grace basis and on the M4 basis. The last part reruns the reverse control.

Hardware: an NVIDIA GPU with CUDA for Ising's pre-decoder (the controls used one H100 80GB HBM3); the GPU and TensorRT
environment of section 4 if you also re-time Ising's plans; one core of an otherwise idle CPU node for the
correlated-PM share, and one core of an otherwise idle Apple M4 for the M4 basis. Environment: section 1, plus
NVIDIA's code and checkpoints of section 3 (`ISING_DECODING_DIR`).

Inputs: the test sets (section 2) and validation chunk 0 at every distance.

```bash
python -m convlines.testset regen --set T2 --d 9 --chunks 0-99 --out shots/T2      # d = 7: 0-199; d = 11, 13 likewise
python -m convlines.testset regen --set T3 --d 15 --chunks 0-99 --out shots/T3
for d in 7 9 11 13 15; do python -m convlines.testset val --d $d --chunks 0 --out shots/val; done
```

1. Ising's plan time. The controls reuse the registered time of Ising's fastest validated TensorRT plan at batch 1
   (`results/surface/s2_composite.json`: `opponents`, `plan`, `plan_us`; `results/surface/results_accuracy_speed_d15.json`:
   `composite_inputs.ising_plans`). No plan was rebuilt. To measure it again, run the `timing.ising_trt` commands
   of section 4 ("The opponents' times") on an H100.
2. The residual detectors for the timing (GPU): Ising M1 and M4 with the official numerics on validation chunk 0.
   The command prints the content sha256 of the residual detectors and of the pre-decoder's logical frame.

   ```bash
   for d in 7 9 11 13 15; do python -m timing.cpu_term residuals --d $d --val shots/val --out res; done
   ```

3. Correlated PM on the test sets, the counts and the statistics (GPU). Each control decodes every shot (Ising's
   pre-decoder, then correlated PM on the residual detectors) next to ours; `convlines.evaluate` prints every
   chunk's failures, then the paired test against ours, and `--out` records the sha256 of each control's decisions
   and of the observables. `--routing published` decodes ours without the deployed executable (section 3).

   ```bash
   python -m convlines.evaluate --set T2 --d 9 --chunks 0-99 --shots shots/T2 --routing deployed \
       --opponents "Ising M1 + correlated PM" "Ising M4 + correlated PM" --out ctrl_d9.json   # likewise every d
   ```

4. The CPU timing on one pinned core, three fresh-process repetitions, interleaved: correlated PM on the residual
   detectors at every d, plus the conversion's denominators, plain PM on them at d = 13 and the node's correlated
   PM on raw validation chunk 0 at d = 13.

   ```bash
   for r in 1 2 3; do
     for d in 7 9 11 13 15; do for m in M1 M4; do
       python -m timing.cpu_term share --d $d --model $m --res res --rep $r --core 8 --out ctrl_cpu
     done; done
     for m in M1 M4; do
       python -m timing.cpu_term share --d 13 --model $m --res res --plain --rep $r --core 8 --out ctrl_cpu
     done
     python -m timing.cpu_term ref --d 13 --val shots/val --rep $r --core 8 --out ctrl_cpu
   done
   ```

5. The conversion to the Grace basis. `convert --share` writes `ctrl_cpu/share_grace_d<d>.json` with the medians,
   both factors and the share on the Grace basis (`share_grace_us`, and `share_grace_beside_us` under the other
   factor). A control's speed is step 1's plan time plus `share_grace_us`.

   ```bash
   for d in 7 9 11 13 15; do python -m timing.cpu_term convert --share --d $d --out ctrl_cpu; done
   ```

6. The M4 basis (decides nothing). Copy the residual detectors (`res/`) and validation chunk 0 at d = 13
   (`shots/val`) to an Apple M4, and run steps 4 and 5 there on one core of the otherwise idle machine, with
   `--out ctrl_m4`. macOS has no CPU-affinity call, so the processes run unpinned and `--core` has no effect (the
   records say `"pinned": false`). On this basis the slowest of the three repetitions counts, as for its other CPU
   parts: a control's speed is step 1's plan time plus `node_corr_share_slowest_us` from
   `ctrl_m4/share_grace_d<d>.json`, unconverted (`m4_corr_share` and `m4_us` in `results/surface/controls_speed.json`).

7. The check. `python -m timing.composite` recomputes the controls' speeds from the medians in
   `results/surface/controls_speed.json`, checks them against the recorded values and the registered plan times, and counts
   the 40 speed cells; it recomputes the M4-basis speeds from the recorded repetitions the same way.

Expected outputs. Step 3, as in `results/surface/controls_test_eval.csv`:

| d | ours | Ising M1 + corr. PM | only ours fails / only M1 + corr. PM fails | Ising M4 + corr. PM | only ours fails / only M4 + corr. PM fails | what it means |
|---|---|---|---|---|---|---|
| 7 | 33,582 | 49,061 | 12,198 / 27,677 | 38,154 | 13,382 / 17,954 | LER 1.68 % (ours), 2.45 % and 1.91 % (the controls) |
| 9 | 10,789 | 22,851 | 3,689 / 15,751 | 16,972 | 3,987 / 10,170 | LER 1.08 %, 2.29 %, 1.70 % |
| 11 | 7,215 | 19,989 | 2,651 / 15,425 | 14,829 | 2,802 / 10,416 | LER 0.72 %, 2.00 %, 1.48 % |
| 13 | 5,627 | 17,105 | 2,423 / 13,901 | 12,255 | 2,571 / 9,199 | LER 0.56 %, 1.71 %, 1.23 % |
| 15 | 3,911 | 14,845 | 1,954 / 12,888 | 9,916 | 2,122 / 8,127 | LER 0.39 %, 1.48 %, 0.99 % |

"Only ours fails" and "only the control fails" count the discordant shots, the ones McNemar's test uses. The controls'
rerun of the pre-decoder gave the registered plain-PM decisions of M1 and M4 on all 12,000,000 test shots, so on an
H100 these counts should come out again; TF32 arithmetic on another GPU can change a few decisions per million.
`results/surface/controls_chunks.csv` gives each control's failures in every chunk (1,200 rows), as step 3 prints them.
`results/surface/controls_decisions.csv` gives, per distance and model, the sha256 of the control's decisions (uint8, in chunk
order) and of the observables; `--out` records the same two hashes as `decisions_sha256` and `obs_sha256` when
`--chunks` covers the whole test set.

Steps 2, 4 and 5, as in `results/surface/controls_speed.json` (µs per shot; the node: AMD EPYC 9534):

| d | correlated PM after M1 / M4, on the node | the same on the Grace basis | plain PM after M1 / M4, on the node | Ising's plan M1 / M4: precision, builder level, µs | control's speed M1 / M4 | what it means |
|---|---|---|---|---|---|---|
| 7 | 10.094 / 8.726 | 9.957 / 8.593 | 6.289 / 5.689 | FP8 L3 43.386 / FP8 L5 56.480 | 53.34 / 65.07 | correlated PM costs 1.61 / 1.53 times plain PM on the residuals |
| 9 | 24.008 / 20.245 | 23.681 / 19.936 | 12.342 / 10.752 | FP8 L3 44.771 / FP8 L5 58.293 | 68.45 / 78.23 | 1.95 / 1.88 times |
| 11 | 51.808 / 42.245 | 51.102 / 41.601 | 23.385 / 19.907 | FP8 L3 45.943 / FP8 L3 59.366 | 97.05 / 100.97 | 2.22 / 2.12 times |
| 13 | 94.701 / 75.807 | 93.410 / 74.650 | 40.198 / 33.664 | FP8 L3 50.612 / FP8 L5 65.192 | 144.02 / 139.84 | 2.36 / 2.25 times; these plain-PM times set the factors |
| 15 | 160.739 / 130.568 | 158.549 / 128.576 | 64.664 / 54.369 | FP8 L5 57.348 / FP8 L5 74.711 | 215.90 / 203.29 | 2.49 / 2.40 times |

The node's correlated PM on raw validation chunk 0 at d = 13 took 233.41 µs per shot. Another CPU gives other raw
times; the conversion is meant to cancel that, and the section "Verification of the earlier release" repeats it on an
Apple M4 core. The residuals' content sha256 are in `results/surface/controls_speed.json`. At d = 15 x86-64 regenerates the
studies' validation shots, so the hashes can be compared there (M1 6d69da7b…, M4 1cf53dd0…). At d = 7-13 the studies'
validation shots were sampled on ARM64, and x86-64 gives other shots from the same seeds: the residuals then differ,
and the shares only through the shots' statistics.

Step 6, as in `results/surface/controls_speed.json` (µs per shot; one Apple M4 core, one session on 2026-10-06):

| d | correlated PM after M1 / M4 on the M4 core, slowest of 3 | control's speed on the M4 basis, M1 / M4 | Ising M1 / M4 + PM on the M4 basis, registered | what it means |
|---|---|---|---|---|
| 7 | 4.997 / 4.327 | 48.38 / 60.81 | 46.26 / 59.08 | ours 11.69 µs on this basis, 36.7 µs below the faster control |
| 9 | 12.962 / 10.601 | 57.73 / 68.89 | 50.81 / 63.38 | ours 29.55 µs, 28.2 µs below |
| 11 | 27.660 / 22.626 | 73.60 / 81.99 | 57.95 / 69.31 | ours 42.70 µs, 30.9 µs below |
| 13 | 49.902 / 40.064 | 100.51 / 105.26 | 72.08 / 82.70 | ours 53.50 µs, 47.0 µs below |
| 15 | 83.476 / 67.479 | 140.82 / 142.19 | 92.19 / 103.48 | ours 66.63 µs, 74.2 µs below |

The reverse control (GPU). The supplement also ran ours with its routed shots sent to plain PM instead of correlated
PM, against the registered Ising + plain PM rows (`Ising M1 + PM` and `Ising M4 + PM`, the official numerics). Its τ
was refitted for plain PM on the calibration sets by the registered rule and frozen before the control read any test
shot. `convlines.decoder.TAU_PLAIN` holds it, the q-quantile of |c| with q = 2, 1, 0, 1 and 1 % at d = 7-15. At d = 11
the rule chose q = 0, so ours is the network alone there, and at d = 7 its time limit used a projected path time
instead of a measured one, a deviation the supplement disclosed. `--post plain` applies this τ, and `--routing
deployed` gives the deployed executables' c, which the supplement used.

```bash
python -m convlines.evaluate --set T2 --d 7 --chunks 0-199 --shots shots/T2 --routing deployed --post plain \
    --opponents "Ising M1 + PM" "Ising M4 + PM" --out rev_d7.json      # likewise every d; T3 at d = 15
```

Expected, as in `results/surface/controls_reverse_eval.csv`:

| d | τ (q) | ours + plain PM | Ising M1 + PM | only ours fails / only M1 + PM fails | Ising M4 + PM | only ours fails / only M4 + PM fails | what it means |
|---|---|---|---|---|---|---|---|
| 7 | 0.7138 (2 %) | 41,196 | 48,875 | 17,194 / 24,873 | 38,095 | 19,937 / 16,836 | worse than M4 + PM: 1.55 more failures per 1,000 shots [97.5 %: 1.34, 1.77], p = 7.5·10⁻⁵⁹ |
| 9 | 0.6821 (1 %) | 12,139 | 22,900 | 4,369 / 15,130 | 17,016 | 4,932 / 9,809 | better than both |
| 11 | 0 (0 %) | 8,431 | 20,280 | 3,373 / 15,222 | 14,871 | 3,668 / 10,108 | better than both; the network alone |
| 13 | 1.0116 (1 %) | 7,856 | 17,491 | 3,687 / 13,322 | 12,422 | 4,098 / 8,664 | better than both |
| 15 | 1.0091 (1 %) | 7,475 | 15,288 | 4,206 / 12,019 | 10,081 | 4,750 / 7,356 | better than both |

The other nine cells are "better", with McNemar p at most 7.0·10⁻¹²⁵; Holm's adjustment over the supplement's 20
primary cells changes no verdict.

Runtimes, from the controls' run: Ising's pre-decoder took 11-71 s of H100 time per distance and model, about 5
minutes for all ten, and correlated PM on the residual detectors about 13 minutes of one CPU core in all (the controls
spread it over 20 processes); `convlines.evaluate` adds ours (section 3). Step 4 takes a few minutes on one core; on
the Apple M4, its 39 processes took 24 s in all.

### 7. Study 4 (d = 17)

These steps rebuild study 4's numbers. They are sections 2-4 at d = 17, with three differences: our decisions need an
NVIDIA GPU, because the network is here in its deployed form (`--routing deployed`); the Triton engine runs with 4
pipeline stages, which `timing.deployed` and `timing.gpu_term` read from `weights/surface/configs.json`; and Ising + correlated
PM are registered opponents.

```bash
python -m convlines.testset regen --set T5 --d 17 --chunks 0-99 --out shots/T5   # files named test4
python -m convlines.testset val --d 17 --chunks 0-19 --out shots/val             # V17b, the timing shots
python -m convlines.evaluate --set T5 --d 17 --chunks 0-99 --shots shots/T5 --routing deployed \
    --opponents "plain PM" "correlated PM" "Ising M1 + PM" "Ising M4 + PM" \
    "Ising M1 + correlated PM" "Ising M4 + correlated PM" --out d17.json          # Ising: section 3's set-up
python -m convlines.evaluate --set T5 --d 17 --chunks 0-19 --shots shots/T5 --routing deployed \
    --opponents "min-LLR BP-OSD" "BP 60 BP-OSD" --out d17_bposd.json             # the registered subset
python -m timing.ising_trt prep  --d 17 --val shots/val --onnx onnx              # Ising's plans (section 4)
python -m timing.ising_trt build --d 17 --onnx onnx --work trt --parts registered
python -m timing.ising_trt eval  --d 17 --onnx onnx --work trt --parts registered --set T5 --shots shots/T5 \
    --chunks 0-99 --corr --out ising_eval                                        # + plain and + correlated PM
```

The timing follows section 4: the GPU term and the handoff on an H100, the CPU term on one core of an exclusive x86
node, then the composite check.

```bash
python -m timing.deployed --d 17 --val shots/val --out masks
for p in 0 1 2 3 4; do python -m timing.gpu_term --d 17 --val shots/val --proc $p --deployed masks/val_d17.npz --out timing_out; done
python -m timing.gpu_term --d 17 --val shots/val --proc 0 --handoff --build build/d7 --deployed masks/val_d17.npz --out timing_out
for r in 1 2 3; do
  python -m timing.cpu_term term --d 17 --val shots/val --mask routing/val_d17.npz --rep $r --core 8 --out cpu_out
  python -m timing.cpu_term ref --d 17 --val shots/val --rep $r --core 8 --out cpu_out
done
python -m timing.cpu_term convert --d 17 --out cpu_out                           # b on the Grace basis
python -m timing.composite                                                       # d = 17's composite, 8 cells
```

Study 4 measured correlated PM's share after Ising on one Apple M4 core with the registered PM-share protocol (the
median of three interleaved repetitions, on Ising's residual detectors of V17b chunk 0) and converted it with the
registered d = 13 ratios, 1.8801 (M1) and 1.9090 (M4). `timing.cpu_term residuals --d 17` and `share --d 17 --model M1`
(and `M4`) measure the same quantity; `convert --share` would use the core's own d = 13 factor instead (section 6).

Expected: the counts and statistics of `results/surface/s4_test_eval.csv` (ours 2,687 failures and 100,797 routed shots, as
`evaluation/routing/manifest.json` records them) and the times of `results/surface/results_accuracy_speed_d17.json`. On a GPU other than
the H100 the engine's c can differ by rounding, so a few shots near τ can be routed differently (the section
"Verification of the earlier release" counts them on an RTX 3060), and a TensorRT plan rebuilt elsewhere can decide a few
shots per million differently from the study's plan. BP-OSD is slow at d = 17: the study split the subset over H100
pods running five decoder processes each, and one pod took 33-42 minutes per arm for 50,000 shots.

### 8. R17 (d = 17 on T6)

R17's records are in `results/r17/`, under the hub's paths; `results/r17/SANITIZE.txt` lists what changed for publication.
`evaluation/testsets/T6.json`, `results/surface/r17_test_eval.csv` and `results/surface/r17_summary.json` are byte-for-byte copies of its T6
manifest and its analysis outputs. R17's scripts ran from study 4's frozen tree (freeze c500ef8e, tree f1c6a0e9), which
this release does not include, and `convlines` has no option for R17's τ or for T6, so this release's commands do not
rebuild R17's numbers. Three parts can be checked with what is here, each with one block below. On 2026-10-10 the
three blocks ran as written from a fresh clone, on one Alpine CPU node with Python 3.12 and the pinned packages
(torch is not needed); `results/r17/REPRODUCTION_CHECK.md` records the commands and what they printed.

T6 itself. `r17.py sample` drew chunk j with the call of section 2 and the seed 91,170,000 + j, and wrote it in
section 2's format. This release's sampler makes the same call, so the lines below write T6 (0.61 GB) and compare
every file's sha256 with `evaluation/testsets/T6.json`. They printed True for all 100 files, in 89 s on an AMD EPYC 9534 node.

```bash
python - <<'EOF'
import json, os
from convlines import circuits, testset
man = json.load(open("testsets/T6.json"))
assert man["circuit_sha256"] == circuits.CIRCUIT_SHA256[17]
c = circuits.circuit(17)
os.makedirs("shots/T6", exist_ok=True)
for f, e in sorted(man["files"].items()):
    p = os.path.join("shots/T6", f)
    testset.write_npz(p, testset.sample(c, e["seed"]))
    print(f, e["seed"], testset.sha256_bytes(open(p, "rb").read()) == e["sha256"])
EOF
```

The statistics. `results/surface/r17_test_eval.csv` holds every row's failures and discordant counts with its p, interval and
verdict, in the format of the other studies' files, plus the preservation row (`delta`, `preserved`). The lines below
rebuild each row's per-shot failures from its counts, recompute the row with `convlines.stats`, and derive δ from study
4's correlated-PM row in `results/surface/s4_test_eval.csv` (half the end of its interval nearest 0). They printed all 24 rows
"better", 8 of 8 accuracy cells held, the preservation interval [−23.3, +43.3] per million below δ = 81.7 per million,
and no difference from the file in any p, interval or phi.

```bash
python - <<'EOF'
import csv
import numpy as np
from convlines import stats
s4 = {r["vs"]: r for r in csv.DictReader(open("../results/surface/s4_test_eval.csv"))}
delta = -float(s4["pm_corr"]["diff_hi"]) / 2       # half of the advantage study 4 established over correlated PM
worst, held = 0.0, {}
for r in csv.DictReader(open("../results/surface/r17_test_eval.csv")):
    n, fa, fb, oa, ob = (int(r[k]) for k in ("shots", "fail_a", "fail_b", "only_a", "only_b"))
    a, b = np.zeros(n, bool), np.zeros(n, bool)     # per-shot failures rebuilt from the counts:
    a[:fa] = True                                   # both fail, then only ours,
    b[:fa - oa] = True; b[fa:fa + ob] = True        # then only the opponent
    s = stats.paired(a, b, float(r["alpha"]))
    assert s["fail_b"] == fb
    worst = max([worst] + [abs(s[k] - float(r[k])) / max(abs(float(r[k])), 1e-300) for k in
                           ("mcnemar_p", "diff_ler", "diff_lo", "diff_hi", "diff_lo95", "diff_hi95", "phi")])
    if r["opponent"] == "preservation":
        print(f"q = 8 % vs 10 %: {fa} vs {fb} failures, p {s['mcnemar_p']:.2f}, interval [{s['diff_lo']:.3g}, "
              f"{s['diff_hi']:.3g}], delta {delta:.4g} (file {float(r['delta']):.4g}), preserved "
              f"{s['diff_hi'] < delta} (file {r['preserved']})")
    else:
        held.setdefault(r["opponent"], []).append(s["verdict"] == "better" == r["verdict"])
        print(f"{r['vs']:20s} {fa:5d} {fb:6d}  p {s['mcnemar_p']:.2g}  [{s['diff_lo']:+.6f}, {s['diff_hi']:+.6f}]  "
              f"{s['verdict']}")
print("accuracy cells held:", sum(all(v) for v in held.values()), "of", len(held))
print(f"largest relative difference from the file: {worst:.1e}")
EOF
```

The composite. `results/r17/run/records/cpu_J23/` holds the CPU job's three repetitions of each quantity
(`term_r17_q08_rep*.json`, `ref_d17_correlated_rep*.json`, and the q = 10 % term beside them): (b) = 73.038 × 714.05 /
780.54 = 66.816 µs, and ours = 66.660 + 66.816 + 1.543 = 135.019 µs, as `results/surface/r17_summary.json` records; (a) and (c)
are study 4's, from `results/surface/results_accuracy_speed_d17.json`. The lines below printed these numbers, the 16.33 µs
below study 4 (14.89 µs of same-node CPU saving and 1.44 µs between the two CPU sessions) and all 8 speed margins
positive, the narrowest 26.31 µs against Ising M1 + PM.

```bash
python - <<'EOF'
import json, statistics
R = "../results/r17/run/records/cpu_J23/"
q08, q10, ref = ([json.load(open(f"{R}{f}_rep{i}.json"))[k] for i in (1, 2, 3)] for f, k in
                 (("term_r17_q08", "term_us"), ("term_r17_q10", "term_us"), ("ref_d17_correlated", "per_block_us")))
s4 = json.load(open("../results/surface/results_accuracy_speed_d17.json"))["arms"]
x = s4["ours (d17_b5_r787, FROZEN, 4 stages)"]["components_us"]     # study 4's (a), (b) and (c)
grace, node = s4["correlated PM"]["speed_us"]["grace"], statistics.median(ref)
b, b10 = max(q08) * grace / node, max(q10) * grace / node        # the slowest repetition, converted
ours = x["a_gpu"] + b + x["c"]
r17 = json.load(open("../results/surface/r17_summary.json"))
print(f"(b) = {max(q08):.3f} x {grace} / {node:.2f} = {b:.3f} us; the q = 10 % term beside it: {b10:.3f} us")
print(f"ours = {x['a_gpu']:.3f} + {b:.3f} + {x['c']:.3f} = {ours:.3f} us (r17_summary.json: "
      f"{r17['ours_speed']['total']:.3f})")
s4c = x["a_gpu"] + x["b_grace"] + x["c"]
print(f"{s4c - ours:.2f} us below study 4's {s4c:.2f}: {b10 - b:.2f} same-node CPU saving + "
      f"{x['b_grace'] - b10:.2f} (study 4's CPU session {x['b_grace']:.2f} vs this node at q = 10 %)")
for k, c in sorted(r17["cells"].items(), key=lambda kc: kc[1]["speed"]["opponent_us"]):
    t = c["speed"]["opponent_us"]
    print(f"  {k:8s} {t:12.2f} us, margin {t - ours:12.2f} us, holds {t - ours > 0} (file {c['speed']['holds']})")
EOF
```

What cannot run here. Two steps of R17's run need H100 numerics or the RunPod pod, so this release cannot repeat them
and the check above did not:
- The GPU job (`pod.sh push`, `start 0` and `pull`; the job runs `r17.py pins`, ours on V17b, `hopper` and `read`) ran
  on a RunPod H100 80GB HBM3 pod, and its outputs depend on H100 numerics: the deployed Triton engine's c, which on
  another GPU can differ by rounding and route shots near τ differently (section 7), and Ising's TensorRT plans from
  S4-R3's session. Those plans are bound to the H100's compute capability, their kernels were chosen on the H100, and
  they are not in this release.
- The CPU term (`alpine/r17_cpu.sbatch`) is CPU work on an exclusive EPYC 7713 node, but it times the V17b shots that
  R17's τ routes, and that mask comes from the H100 engine's c in study 4's development record, which is not
  published: `results/r17/run/records/cpu_J23/mask_q08.json` gives the mask's sha256 and counts, and `evaluation/routing/val_d17.npz`
  holds only study 4's q = 10 % mask.

The other steps need files that are not here either. T6's BP-OSD shares (`alpine/r17_bposd.sbatch`, on A100s) and the
analysis (`alpine/r17_analyze.sbatch`) run `r17.py` from study 4's frozen tree, and the analysis reads every read's
per-shot outputs, which are not published (`results/r17/run/final_inputs.sha256` lists their sha256).

R17's own commands ran from a clone of the hub (where `results/r17/` is `ising_compare/r17/`), on Alpine and on one H100 pod.
The run's note with every command in order (`R17_READY.md`) is not included; `results/r17/R17_RESULT.md`
section 6 gives the departures from it, and `results/r17/PLAN_R17.md` section 7.4 summarises the commands. The scripts
`deploy.sh` and `pod.sh`, which copied files to the cluster and the pod, are not included either. The steps: `deploy.sh code` (R17's files and S4-R3's records to the cluster); the CPU term (`alpine/r17_cpu.sbatch` on an
exclusive EPYC 7713 node); T6 (`alpine/r17_sample.sbatch`, which also regenerates T5 chunks 0-19 for the Ising check);
the GPU job on the pod (`pod.sh push`, `start 0` and `pull`; the job runs `r17.py pins`, ours on V17b, `hopper` and
`read`); the three BP-OSD shares (`alpine/r17_bposd.sbatch`); and `alpine/r17_analyze.sbatch` with `WHAT=merge`,
`inventory` and, once, `analyze`.

Expected: T6's 100 sha256 as in `evaluation/testsets/T6.json`, and R17's counts as in `results/surface/r17_test_eval.csv` (ours 2,708
failures and 79,652 routed shots at q = 8 %, 2,698 and 100,754 at q = 10 %).

## What is not established

- The composite does not measure end-to-end latency. It is a registered benchmark score: the input is already on the
  GPU, the CPU part is an average (the routed fraction times the mean cost of a routed shot), and the handoff at
  d = 9-13 is an allowance. We claim nothing about matched end-to-end latency, tail latency (p99, worst case) or
  throughput.
- Nothing ran on a Grace CPU. The registered results take only the paper's measurements at d = 9 and 13 as Grace
  numbers; every other registered Grace value, including the CPU term's numerator at d = 7, 11, 15 and 17, is a
  conversion. The paper also timed correlated PM at d = 17 and plain PM and the PM shares at d = 21 and 31
  ([section 4](#where-every-grace-number-comes-from)); the NVIDIA-distance basis at d = 17 takes its values from them.
- On a measured CPU, plain PM is faster than ours at d = 7, 9, 11 and 13: with the CPU parts timed on one Apple M4
  core, plain PM takes 6.68 / 14.16 / 27.77 / 49.47 µs against our 11.69 / 29.55 / 42.70 / 53.50 µs. At d = 15 and 17
  ours is faster (66.63 against 81.60 µs; at d = 17, study 4's 104.15 against 124.58 µs, since R17 did not time the M4
  core). The registered basis is the Grace one.
- Each study's timing used one host, one session and one GPU type; R17's composite joins terms from study 4's sessions
  and a new CPU job ([R17](#r17-d--17-re-registered)). Development rows on an RTX PRO 6000 exist but decide
  nothing, and the BP-OSD times are host measurements, unconverted.
- Only X-basis memory with d rounds under NVIDIA's legacy single-parameter noise at p = 0.006 was registered. Nothing is
  claimed for other error rates, other noise models, the Z basis, other codes or hardware data.
- Network sizes, replicas and thresholds were chosen on validation and calibration sets that were read many times
  during development, and the two replicas of each size share their ancestry. The test sets were fresh and read once;
  they test these particular networks and say nothing about how reliably the training method produces such networks.
- The registered decisions are the deployed executables' (fp16 inputs, fp32 accumulation). `--routing deployed`
  reproduces them on a GPU; the fp32 reference, the default, can route near-τ shots differently (section 3).
- This release does not include the training code, the host gate's script or the registration machinery (freeze,
  claim and audit records); section 4 states the gate's rules.
- The two controls with correlated PM after Ising were measured after both studies had been read and published. They
  are descriptive and make no registered claim. Their speeds are composite estimates whose correlated-PM share was
  converted to the Grace basis with a ratio measured at d = 13 for plain PM; on the M4 basis the share comes from one
  session on one Apple M4 core.
- Study 4 ran under a re-registration made after its first registration and S4-R1 had failed to time it, and shaped
  by the hardware available. Its session was gated on the GPU side only: the host ran other work (median 1-minute load
  20.3 on 208 CPUs), and the opponents' parts timed there (Ising's TensorRT plans, BP-OSD's GPU BP and host OSD)
  count their fastest repetition but were not timed on a quiet host.
- At d = 17 every opponent time except BP-OSD's, and our CPU term, is a conversion to the Grace basis: plain and
  correlated PM from an x86 node's times, the PM shares from an Apple M4 core's, both scaled by d = 13 ratios. The
  narrowest d = 17 speed margin, against Ising M1 + PM (26.31 µs in R17, 9.98 µs in study 4), depends on these
  conversions. Converted by one rule at d = 17 on both sides, with NVIDIA's own d = 17 and d = 21 values, it is
  21.73 µs in R17 and 2.05 µs in study 4, and every speed cell of both holds. A one-sided bound is in
  [evaluation/timing/R17_GRACE_NOTE.md](../../evaluation/timing/R17_GRACE_NOTE.md).
- This release holds the d = 17 network's deploy tables, not its fp32 network, so the CPU reference does not run at
  d = 17. S4-R3's registration record (the A of T5's seeds) is not published, like the approval records. R17's scripts
  need study 4's frozen tree, which is not published either (section 8).

## Verification of the earlier release

This section records a check run on the earlier release (2026-10-05) whose files are carried here unchanged apart from paths.

Run on 2026-10-05 from a fresh clone of the earlier release's repository at commit 37fe993, on an HPC cluster, with Python 3.12.14 and
the pinned versions above: once on an AMD EPYC 7713P CPU node and once on an NVIDIA H200 NVL MIG slice (3g.71gb, the
same Hopper architecture as the studies' H100). The last part of this section covers what changed after that commit.

- Test sets: two chunks per distance were regenerated with `convlines.testset regen` on the EPYC 7713P (AVX2, no
  AVX-512) and on the GPU node's EPYC 9555: T2 chunks 0 and 199 at d = 7 and chunks 0 and 99 at d = 9, 11 and 13, T3
  chunks 0 and 99 at d = 15. All ten files have the manifest's sha256 on both machines. `evaluation/scripts/make_circuits.py`
  rebuilt the five circuits and grids from NVIDIA's code identical to the files here, and `weights/MANIFEST.sha256`
  checks. With the two approval records (held by the authors), `python -m convlines.testset seeds` re-derives all 600
  seeds from the drand randomness in the manifests.
- The reference decoder (`convlines.decoder`, fp32, on the CPU and on the GPU) decoded those ten chunks, 100,000
  shots, and was compared with the studies' per-shot records of the deployed decoder (each shot's c, routing and
  decision, kept with the test reads). All 100,000 decisions are equal, so the failure counts match chunk by chunk:
  186 and 186 at d = 7, 94 and 124 at d = 9, 71 and 70 at d = 11, 71 and 49 at d = 13, 41 and 37 at d = 15. The
  routing differs on 11 shots, all with |c| near τ, and on each of them the network's sign and correlated PM give the
  same answer. The largest difference between the reference's c and the deployed c is 0.02 at d = 7, 0.6-1.0 at
  d = 9, 0.1-0.2 at d = 11, 0.3 at d = 13 and 1.2-1.5 at d = 15 (fp32 against the engine's fp16 inputs). Plain and
  correlated PM reproduce the studies' per-shot decisions on all 100,000 shots, and the regenerated observables equal
  the recorded ones.
- The Triton engine (`evaluation/timing/engine.py`, built from the released weights) on the H200 reproduces the deployed c bit
  for bit on every shot of the d = 9, 13 and 15 chunks (60,000 shots) and on 19,793 of the 20,000 d = 11 shots (the
  others within 7·10⁻⁶); its decisions equal the deployed ones on all 100,000 shots. At d = 7 it stands in for the
  hand-written kernel: c within 0.005 of the kernel's, the same decisions.
- Timing code, as a smoke test on d = 9: `timing.cpu_term` gave (b) = 1.222 µs per shot on the EPYC 7713P node
  (registered: 1.277 on an EPYC 7713) and the node's correlated PM 82.54 µs per block (registered: 82.79), so (b) on
  the Grace basis is 1.001 µs; on the GPU node's EPYC 9555 the raw times are 37-40 % lower and the converted value is
  1.060 µs (registered: 1.042). `timing.gpu_term` ran on the MIG slice (32.6 µs per shot on 20,000 shots, a shared
  slice, so not comparable with the registered 27.9 µs of a whole H100).
- The speed cells: all 70 numbers that the earlier release's figure script plotted (failures, shots and Grace times of
  5 distances × 7 decoders, failures and shots on the M4 basis) equal the registered analysis summaries and composite
  records. `python -m timing.composite` reproduces the five composites and all 30 speed cells.
- Ising M1 / M4 + PM through `evaluation/convlines/opponents.py` and NVIDIA's code at 3ce4379 (with the extras of section 3),
  on the H200: chunk 0 at each of the five distances, both models, every one of the 10 × 10,000 decisions equal to the
  studies' records of the official PyTorch numerics + PM (failures 260 / 200 at d = 7, 233 / 152 at d = 9, 195 / 162 at
  d = 11, 181 / 140 at d = 13, 148 / 97 at d = 15 for M1 / M4).
- The command-line tools: `python -m convlines.evaluate --set T2 --d 9 --chunks 0 ... --opponents "plain PM"
  "correlated PM" "Ising M1 + PM" "Ising M4 + PM"` gave ours 94, plain PM 233, correlated PM 124, Ising M1 233 and
  Ising M4 152 failures with the paired statistics (all four "better" on that one chunk), and
  `python -m convlines.decoder --d 15` on T3 chunk 0 gave 41 failures, both as recorded.
- Not re-run for this release: the BP-OSD opponents, and the Ising TensorRT plans at d = 7, 9, 11 and 15.

Commit 7513cda added the d = 7 executable, the Ising TensorRT script and the license, and split the Ising loader of
`evaluation/convlines/opponents.py` into the two functions the script shares with it (same numerics). It was verified on
2026-10-06 from a fresh clone on the same cluster, on an AMD EPYC 9534 CPU node and on an H200 NVL MIG slice (3g.71gb)
whose GPU other users' jobs were sharing; the next commit, f7ea664, changed only the earlier release's README and a docstring.

- `evaluation/timing/d7/build.sh` (nvcc 12.1.105, g++ 12.2) built a cubin byte for byte equal to the one the studies ran (sha256
  8f4180d7…). The studies' kernel source and the released one compile to identical cubins with nvcc 12.1.105 and with
  nvcc 13.1.115 (whose cubin differs from the studies'), and the two versions of `router.cc` to identical object files
  with g++ 12.2.
- `python -m timing.d7.run` on T2 chunks 0 and 199 at d = 7 (20,000 regenerated shots, sha256 as in the manifest)
  matched the studies' per-shot records exactly: every c bit for bit, the same 403 and 412 routed shots, all 20,000
  decisions, 186 and 186 failures. On every routed shot the router's compiled correlated PM agreed with the PyMatching
  wheel.
- Timing on the shared slice, as a smoke test only (not comparable with the registered whole-H100 values, and rejected
  by the acceptance rule because other processes ran on the GPU): `timing.gpu_term --d 7` gave (a) = 9.109-9.112 µs per
  shot in five processes (registered: 8.443 µs), and `--handoff` gave (c) = 1.29-1.44 µs (registered: 2.240 µs).
- `evaluation/timing/ising_trt.py prep` exported both Ising models at d = 13 and 15 to FP32 ONNX files byte for byte equal to the
  studies'. At d = 15 the regenerated calibration rows equal the studies' and so do both FP8 ONNX files; at d = 13 the
  calibration rows differ (the studies' validation chunk was sampled on ARM64), and so do the FP8 files. At d = 13 all
  eight plans (M1 / M4 × FP16 / FP8 × level 3 / 5) built and passed the check, the four FP8 plans with FP8 execution
  confirmed. The fastest repetitions on the slice were FP8 level 3 plans: 63.2 µs (M1) and 84.4 µs (M4) per call,
  against the registered 50.6 µs (M1, FP8 level 3) and 65.2 µs (M4, FP8 level 5) on the H100.
- With the split loader, `python -m convlines.evaluate --set T2 --d 9 --chunks 0` gave the recorded failure counts
  again (ours 94, plain PM 233, correlated PM 124, Ising M1 233, Ising M4 152), and `timing.gpu_term --d 9` still ran
  the Triton path (32.7 µs per shot on 20,000 shots of the slice, against 32.6 µs before).

Commit 8b1274c put the deployed routing into the reproduction (`evaluation/routing/`, `evaluation/timing/deployed.py`, `convlines.evaluate
--routing`, `timing.cpu_term --mask`); gave `timing.gpu_term` the d = 9-15 handoff driver and a per-process
acceptance record, and `evaluation/timing/ising_trt.py` the test-set evaluation and the enforced acceptance; and quoted the paths
in `evaluation/timing/d7/build.sh`. It was verified on 2026-10-06 from a fresh clone on the same cluster, on an AMD EPYC 7713P
CPU node and on an H200 NVL MIG slice (3g.71gb) whose GPU other users' jobs were sharing (its node's CPU is an AMD
EPYC 9555).

- `evaluation/routing/`: all ten files have the manifest's sha256, shot count and routed count. The test masks equal the routing
  in the studies' per-shot records on T2 chunk 0 at d = 7-13 and T3 chunk 0 at d = 15 (403, 86, 97, 196 and 630
  routed shots); the regenerated chunks have the manifest's sha256.
- Our decisions on those five chunks (50,000 shots) against the records: with `--routing published` and with
  `--routing fp32` on the EPYC node all 50,000 are equal (186, 94, 71, 71 and 41 failures, as recorded); `published`
  routes every shot as recorded, `fp32` routes one shot differently at d = 9 and seven at d = 15. With
  `--routing deployed` on the H200 every routing and every decision is equal, and c is bit for bit equal on all
  10,000 shots at d = 7, 9, 13 and 15 and on 9,887 at d = 11 (the others within 7·10⁻⁶). Through the command line
  (`convlines.evaluate`), `--routing deployed` gave 41 failures at d = 15 and 186 at d = 7, no shot routed
  differently from the published mask, and `--routing published` at d = 9 gave ours 94 against correlated PM's 124
  (paired test "better").
- `python -m timing.deployed` over the 200,000 validation shots: at d = 15, whose regenerated shots are the studies'
  (same digest), its mask equals the studies' (11,822 routed shots, none different); at d = 7-13 the regenerated
  shots differ from the studies' ARM64 ones (7,955, 1,980, 1,945 and 3,879 routed). The fp32 reference routes 3 of
  the 200,000 shots differently at d = 9 and 59 at d = 15.
- `timing.cpu_term term --d 15` on the GPU node, one repetition with the deployed mask and one with
  `evaluation/routing/val_d15.npz`: the same 11,822 routed shots, 21.27 and 21.30 µs per shot. Given `evaluation/routing/val_d9.npz` with
  the regenerated d = 9 shots, it stopped (other shots). `ref --plain --d 9`: 21.10 µs per block.
- `timing.gpu_term` on the slice, numbers only: the clock checks passed, but other processes were on the GPU, so the
  acceptance rule rejected every process. (a): 32.65 µs per shot at d = 9 (20,000 shots) and 9.112 µs at d = 7, the
  outputs bit for bit equal to `timing.deployed`'s. (c) at d = 7: 1.344 µs (round trip 10.454 µs, device 9.110 µs).
  The router process at d = 9: complete path 36.188 µs, device only 34.281 µs, 1,980 routed shots at 83.2 µs,
  estimate 1.084 µs, counted 1.100 µs; at d = 15: 128.999 µs, 105.036 µs, 11,822 at 401.0 µs, estimate 0.261 µs,
  counted 1.100 µs; the router's floor 4.21 µs. In both, the routed set is |c| < τ, the unrouted decisions are the
  sign of c and the routed ones equal the PyMatching wheel's.
- `evaluation/timing/d7/build.sh` built the studies' cubin (sha256 8f4180d7…) and a router library byte for byte equal to the
  previous script's, also into a directory whose path contains a space; both libraries' compiled correlated PM decided
  all 10,000 shots of a validation chunk as the wheel did.
- Ising at d = 15: `prep`, `build --parts M4_fp16_B64 M1_fp8_B1` (the plotted parts) and `eval` on T3 chunk 0 gave 95
  and 150 failures, all 2 × 10,000 decisions equal to the studies' records of those parts (whose 100 chunk files have
  the registered digests). On an FP8 level-3 plan, `time` refused the plan without a check report, with its FP8
  verdict set to false, and after the plan file was replaced by another valid plan, and timed it only after its check
  passed (71.76 µs per call).
- `weights/MANIFEST.sha256` checks with the corrected `configs.json`, and `python -m timing.composite` reproduces the
  composites.

Commit 58316b3 changed the message `timing.ising_trt time` prints for an FP8 plan whose execution is not confirmed
and added `c_measured_detail` (the d = 15 row of the handoff table) to `results/surface/results_accuracy_speed_d15.json`.
From a fresh clone of it on the slice: `python -m timing.composite` reproduced the composites; `check` passed the FP8
level-3 plan again,
`time` timed it (71.82 µs per call), and `time` refused it with no check report, with the FP8 verdict false, with
the check failed and after the plan file was replaced, each time naming the reason. An export traced on an all-zero
shot, as the sweep's code does, has the same nodes (405 for M1, 433 for M4), weights, inputs and outputs as `prep`'s
d = 15 files; only `prep`'s files carry intermediate shape annotations. The commit after it changed only the earlier
release's README and a docstring.

Not re-run for these two commits: the BP-OSD opponents, the Ising parts other than the two above, the full test sets
(one chunk per distance), and any timing on a whole H100 or an exclusive CPU node.

The commit that added the two post hoc controls was checked on 2026-10-06 without a GPU, on Linux x86-64 with the
pinned Python packages. `python -m timing.composite` reproduced the registered composites,
the controls' 10 speeds and their 40 speed cells. The statistics of the 10 accuracy cells, recomputed from the counts
with `convlines.stats`, equal the supplement's records to 10⁻¹² (p-values, intervals and phi).

On one Apple M4 core (macOS, so unpinned; three repetitions), `timing.cpu_term share` timed the supplement's own
residual detectors of validation chunk 0 at d = 9 and 13 (their content sha256 equal to the records'). Correlated PM
after M1 / M4 took 12.83 / 10.59 µs per shot at d = 9 and 49.63 / 40.03 µs at d = 13, plain PM 21.35 / 17.45 µs at
d = 13, and `convert --share` gave Grace-basis shares of 23.84 / 20.12 µs at d = 9 and 92.19 / 76.03 µs at d = 13,
within 2 % of the published 23.68 / 19.94 and 93.41 / 74.65 µs from the x86-64 node. The other factor depends more on
the CPU: the M4 core's correlated PM on raw validation chunk 0 at d = 13 took 118.21 µs, and that factor's shares came
out 3-6 % above the node's. On that ARM64 machine `convlines.testset val` gave d = 13's chunk 0 with the studies'
observables and mean detector count.

Not re-run for this commit: every GPU step (Ising's pre-decoder on the test sets and on validation chunk 0, its
TensorRT plans, `convlines.evaluate` with the controls), the shares at d = 7, 11 and 15, and any timing on an
exclusive x86-64 node.

The commit that added the controls' M4-basis points, their per-chunk counts and the reverse control's rows was checked
on 2026-10-06 on the same Linux x86-64 machine, plus one session on an Apple M4. `python -m timing.composite`
reproduced the registered composites, the controls' 10 speeds and 40 speed cells, and their 10 M4-basis speeds.

The per-chunk counts come from the controls' per-shot outputs: every chunk file's sha256 was checked against the
supplement's list of its outputs, and the counts add up to the totals in `results/surface/controls_test_eval.csv`. M1 / M4
followed by correlated PM failed 261 / 203 times in T2 chunk 0 and 247 / 202 times in chunk 199 at d = 7, and
227 / 155 times in chunk 0 and 192 / 162 in chunk 99 at d = 9. The same two chunks gave 199 / 159 and 182 / 140 at
d = 11, 188 / 137 and 156 / 109 at d = 13, and 145 / 100 and 141 / 83 at d = 15 (T3). The reverse control's
statistics, recomputed from its counts with `convlines.stats`, equal the supplement's records exactly, and its Ising
failures equal the registered rows of `Ising M1 + PM` and `Ising M4 + PM` with the official numerics.

On one core of an Apple M4 (macOS 26.2, unpinned; Python 3.12.14, numpy 2.2.6, Stim 1.16.0, PyMatching 2.4.0), step 6
of section 6 ran in one session with no other Python process running and a load average of 1.4-1.7 from the desktop's
background. The three repetitions of each share lay within 1.5 % of each other. Plain PM on the same residuals at
d = 13 took 21.59 / 17.51 µs (the slowest repetition), against the registered M4-basis shares of 21.46 / 17.51 µs.
`convert --share` on these times gave Grace-basis shares within 4 % of the node's at d = 9-15 and 8 % / 5 % below
them at d = 7 (9.17 / 8.18 against 9.96 / 8.59 µs); under the other factor they came out within 6 % of the node's.
The new `convlines.evaluate --post plain` and the decision hashes were exercised on synthetic inputs only.

Not re-run for this commit: every GPU step (Ising's pre-decoder, the controls' and the reverse control's evaluation)
and any timing on an exclusive x86-64 node.

The commit that added study 4 (d = 17) was checked on 2026-10-07 on one Linux x86-64 workstation (AMD Ryzen 9 9950X;
Python 3.12 with the pinned packages; an NVIDIA GeForce RTX 3060 for the GPU steps), against the study's records.

- `evaluation/scripts/make_circuits.py` rebuilt all six circuits and grids from NVIDIA's code equal to the files here; the
  d = 17 circuit has the registered sha256 (5a17d282…). `convlines.testset regen` rebuilt T5 chunks 0, 19 and 99 with
  the manifest's sha256, and with S4-R3's registration record `convlines.testset seeds` re-derived all 100 T5 seeds.
  `convlines.testset val --d 17` rebuilt the 20 V17b chunks equal to the study's.
- Every tensor of `weights/surface/d17_b5_r787_deployed.safetensors`, read by `convlines.decoder.read_tables`, is bit for bit
  the deployed artifact's (sha256 bc00eb30…), and its site order and grid positions match `weights/surface/grid_d17.npz`.
- On the RTX 3060 (Ampere, no PDL; the study ran on Hopper) the Triton engine on these tables never gave the study's c
  bit for bit: the median difference was 2.2·10⁻³ and the largest 2.85. It routed 24 of the 200,000 V17b shots
  differently from the published mask, and 1, 1 and 3 of the 10,000 shots of T5 chunks 0, 19 and 99 differently from
  the test read's records; on every other shot the sign agreed. On those three chunks `convlines.evaluate --set T5
  --d 17 --routing deployed` gave ours 27, 32 and 26 failures, plain PM 155, 146 and 131 and correlated PM 28, 35 and
  27, all as recorded, and both PM opponents' decisions equal the records shot for shot. On 500 shots the engine's c
  was bit for bit the same with 3 and 4 pipeline stages.
- The statistics of all 24 rows of `results/surface/s4_test_eval.csv`, recomputed with `convlines.stats` from the study's
  per-shot records (each record's sha256 as in the inventory the analysis committed before it ran), equal the study's
  counts, discordant pairs and verdicts, and its p-values, intervals and phi to 4·10⁻¹⁶. `evaluation/routing/test_T5_d17.npz`
  comes from the same records (record digest b813d31a…, the one the analysis read).
- `python -m timing.composite` reproduced the d = 17 composite (151.345 µs) from the recorded components and found its
  8 speed cells held; its output for d = 7-15 is unchanged.

Not re-run for this commit: any timing (the GPU term and handoff on an H100, the CPU term, the PM shares), Ising's
pre-decoder and its TensorRT plans (`timing.ising_trt eval --corr` is new and has not run), BP-OSD, and our decoder on
an H100.

The commit that added R17 (d = 17 on T6) ran nothing: it adds records and text. On 2026-10-10 each of the 163 files
that `results/r17/` copies from the hub was compared with the hub's copy at the commit in `results/r17/SOURCE`, and `results/r17/SANITIZE.txt`
lists the 80 that differ and why. `evaluation/testsets/T6.json`, `results/surface/r17_test_eval.csv`, `results/surface/r17_summary.json`,
`results/r17/r17.py`, `results/r17/r17.json` and `results/r17/run/final_inputs.sha256` are byte for byte the hub's. Not run for this commit:
any step of R17 and the T6 lines of section 8.

The commit that drew R17 in the figures was checked on 2026-10-10. From a
fresh clone of the previous commit (f31209f), on Alpine with Python 3.12 and the pinned packages, section 8's T6 lines
ran as written and printed True for all 100 files, and the statistics and composite lines this commit adds reproduced
every row of `results/surface/r17_test_eval.csv` with no difference and the composite of `results/surface/r17_summary.json`
(135.019 µs); `results/r17/REPRODUCTION_CHECK.md` has their output. Not run for this commit: R17's GPU job and CPU term, which
need H100 numerics, and its BP-OSD shares and analysis, which need files this release does not hold (section 8).

The commits that added the Cascade comparison (`cascade/bb72/`) were checked on 2026-10-12 on Alpine, each job in a fresh
clone of branch `cascade-results` at c263f7c. Every file under `cascade/bb72/` matches its manifest sha256, and so do the
archived per-shot files: phase C's 68, D1's 4, and the curve sets' 1,285 shot chunks and 7,310 prediction files.
`recompute_numbers.py` recomputed 192 numbers. None of the 70 that have a record differs from it; of the 146 that the
paper prints, one is not as printed: the low end of D1's batched speed-up range comes out at 5.1 (5.15 to two
places), which the paper gives as 5.2. The curve sets' counts and plot, rebuilt from the archived predictions, are
byte for byte the committed files. `cascade/bb72/REPRODUCTION_CHECK.md` has the commands and their output. Not run for
this commit: `check_networks.py`, which loads the seven networks and the curve sets' r11 and r12 and decodes recorded
shots again on an RTX PRO 6000; its job was still waiting for a card when the check was recorded. It ran later the
same day and passed (the last section of `cascade/bb72/REPRODUCTION_CHECK.md` records the result).
