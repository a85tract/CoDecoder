# Phase D1 results: S55'-deep routed to Cascade against Cascade alone, on fresh shots

11 October 2026, on the Linux workstation, branch `ising-cascade-d1`. The plan (`phase_d/PHASE_D1_PLAN.md`) and the lock
(`phase_d/d1_locked_depspam_fm.json`) were frozen in commit F = 310ce8e0f8bfb9d3b97b27186547a836d29fc9ca after the gate review
and the study lead's six dispositions were folded in (plan, section 0). Four jobs ran after the freeze, all
COMPLETED with exit code 0: the reads R1 (CJ71, 0.2 %) and R2 (CJ72, 0.3 %), the timing job D1-S (CJ73) and the one
analysis D1-A (CJ74). This file reports what D1-A computed (`out/confirm/run_1/`) and what D1-S measured
(`out/speed/rows.jsonl`). It changes no result, lock or code.

D is the pair of the lock: S55'-deep (ConvLines-BB, 6 blocks, 841k parameters, on its fused engine) decides every shot
whose margin is at least tau1, and C14' decides the rest. C14' is our re-implementation of Cascade (L 14, H 512, 19.7M
parameters, BF16), not the paper's network. The setting is `depspam_fm`, the fitted noise variant of the noise-model
fit. The two configurations are the first lock's (1e8d1c6d) claim-point configurations, copied with their thresholds:

| claim point | first-lock configuration | q1 | tau1 | shots (chunks) |
|---|---|---|---|---|
| p = 0.2 % | 2 | 2 % | 7.3331888389587405 | 11,000,000 (1,100) |
| p = 0.3 % | 7 | 10 % | 7.60439167022705 | 17,000,000 (1,700) |

Paths below are under `cascade_compare/phase_d/` unless stated otherwise. Every GPU number comes from one card model, a
full NVIDIA RTX PRO 6000 Blackwell Server Edition (driver 590.48.01, torch 2.11.0+cu128, triton 3.6.0).

## 1. The verdicts

All four comparisons of the family are "better": D failed less often than C14' at both claim points, on the eager path
and on both compiled paths.

| comparison | p | C14' path | D / C14' failures | only D / only C14' | McNemar p | Holm level | difference [95 %] | interval at the Holm level | verdict |
|---|---|---|---|---|---|---|---|---|---|
| H1e | 0.2 % | eager | 7,855 / 7,922 | 19 / 86 | 2.30e-11 | 0.025 | -6.09e-06 [-7.96e-06, -4.24e-06] | [-8.24e-06, -3.97e-06] | better |
| H2e | 0.3 % | eager | 76,469 / 76,730 | 81 / 342 | 3.10e-39 | 0.0167 (0.05 / 3) | -1.54e-05 [-1.77e-05, -1.30e-05] | [-1.83e-05, -1.24e-05] | better |
| H1c | 0.2 % | compiled, single (compiled@1) | 7,853 / 7,912 | 19 / 78 | 1.15e-09 | 0.05 | -5.36e-06 [-7.16e-06, -3.59e-06] | the same | better |
| | | compiled, batched (compiled@4096) | 7,858 / 7,918 | 19 / 79 | 7.13e-10 | | -5.45e-06 [-7.26e-06, -3.67e-06] | the same | |
| H2c | 0.3 % | compiled, single (compiled@1) | 76,409 / 76,676 | 81 / 348 | 1.71e-40 | 0.0125 | -1.57e-05 [-1.81e-05, -1.33e-05] | [-1.88e-05, -1.26e-05] | better |
| | | compiled, batched (compiled@4096) | 76,445 / 76,711 | 81 / 347 | 2.77e-40 | | -1.56e-05 [-1.80e-05, -1.33e-05] | [-1.87e-05, -1.26e-05] | |

The shots are 11,000,000 at 0.2 % and 17,000,000 at 0.3 %. A difference is D's block failure rate minus C14''s on the
same shots; "only D" counts the shots that D fails and C14' decodes, and "only C14'" the reverse. McNemar p is the exact
two-sided test and the intervals are Newcombe's (phase C's `pc_report.paired_counts`).

The single and batched compiled paths did not give the same failures on every shot (`compiled_coincide` is false at
both points), so each compiled comparison is the intersection-union test of plan 1.2. Its p-value is the larger of its
two McNemar p-values: 1.15e-09 for H1c and 2.77e-40 for H2c. Holm at 0.05 over the four p-values gave H2c (2.77e-40)
the level 0.0125, H2e (3.10e-39) 0.05 / 3, H1e (2.30e-11) 0.025 and H1c (1.15e-09) 0.05, each p-value below its level.
Each eager interval at its level lies below zero, and so do both intervals of each compiled comparison.

D1-P's design-scenario powers for these four decisions under Holm were 0.9293, 0.9179, 0.9293 and 0.9179 (H1e, H2e,
H1c, H2c) when the four tests are treated as identical, and 0.9535, 0.9491, 0.9584 and 0.9496 when they are treated as
independent (`dry/power_CJ68.json`). The design scenario halved the gain seen in EXPLORE_DEEP (plan 2.4).

### What may be claimed (plan 4.4)

At both claim points the compiled comparison is "better", D's mean single-shot latency and its batched time per shot are
below Cascade alone's in the same job on both engines (section 3), and every check of plan 4.2 held (section 4).
D1-A's `claims` record "more_accurate_and_faster" for both engines at both points. The claims read:

> In the fitted `depspam_fm` setting at p = 0.2 %, S55'-deep routing its 2 % least sure shots to C14' failed less often
> than our re-implementation of Cascade (C14') on fresh shots, on the computation that was timed, by 5.36e-06 [3.59e-06,
> 7.16e-06] in single mode and 5.45e-06 [3.67e-06, 7.26e-06] in batched mode, and was faster on average on the RTX PRO
> 6000: 10.78 times per single shot and 8.86 times batched with today's engine (12.68 and 16.36 with milestone 2's),
> with a slower tail.

> In the fitted `depspam_fm` setting at p = 0.3 %, S55'-deep routing its 10 % least sure shots to C14' failed less
> often than our re-implementation of Cascade (C14') on fresh shots, on the computation that was timed, by 1.57e-05
> [1.33e-05, 1.81e-05] in single mode and 1.56e-05 [1.33e-05, 1.80e-05] in batched mode, and was faster on average on
> the RTX PRO 6000: 5.87 times per single shot and 5.15 times batched with today's engine (6.38 and 7.02 with
> milestone 2's), with a slower tail.

The intervals in the claims are 95 % intervals. At 0.2 % the Holm level is 0.05, so they are also the intervals at the
level; at 0.3 % the intervals at the level 0.0125 are [1.26e-05, 1.88e-05] and [1.26e-05, 1.87e-05].

The eager comparisons, reported beside them, are "better" at both points: 6.09e-06 [4.24e-06, 7.96e-06] fewer
failures per shot at 0.2 % and 1.54e-05 [1.30e-05, 1.77e-05] at 0.3 %. C14' runs in BF16 on every path. S55'-deep
runs with FP16 matmul inputs and FP32 accumulation. The per-round times are in section 3.

The gains are small: 0.75 % (single) and 0.76 % (batched) of C14''s block failure rate at 0.2 %, and 0.35 % of it in
both modes at 0.3 %. Routing cannot gain more than C14''s failures on the shots that S55'-deep keeps: 316 and
317 of C14''s 7,912 and 7,918 failures at 0.2 % (4.0 %), and 2,325 and 2,324 of 76,676 and 76,711 at 0.3 % (3.0 %).
D recovered 59 and 60 of them net at 0.2 % and 267 and 266 at 0.3 % (section 2).

## 2. The claim points in detail

### Failure rates

The block failure rates with Wilson's 95 % intervals, and P_L, the logical error rate per logical qubit per round by
the paper's Eq. (1):

| p | C14' path | D: failures, block rate [95 %] | C14': failures, block rate [95 %] | P_L: D / C14' | difference as a share of C14''s rate |
|---|---|---|---|---|---|
| 0.2 % | eager | 7,855, 7.141e-04 [6.985e-04, 7.301e-04] | 7,922, 7.202e-04 [7.045e-04, 7.362e-04] | 9.922e-06 / 1.001e-05 | -0.85 % |
| 0.2 % | single | 7,853, 7.139e-04 [6.983e-04, 7.299e-04] | 7,912, 7.193e-04 [7.036e-04, 7.353e-04] | 9.919e-06 / 9.994e-06 | -0.75 % |
| 0.2 % | batched | 7,858, 7.144e-04 [6.987e-04, 7.303e-04] | 7,918, 7.198e-04 [7.041e-04, 7.358e-04] | 9.925e-06 / 1.000e-05 | -0.76 % |
| 0.3 % | eager | 76,469, 4.498e-03 [4.466e-03, 4.530e-03] | 76,730, 4.514e-03 [4.482e-03, 4.546e-03] | 6.262e-05 / 6.284e-05 | -0.34 % |
| 0.3 % | single | 76,409, 4.495e-03 [4.463e-03, 4.527e-03] | 76,676, 4.510e-03 [4.479e-03, 4.542e-03] | 6.257e-05 / 6.279e-05 | -0.35 % |
| 0.3 % | batched | 76,445, 4.497e-03 [4.465e-03, 4.529e-03] | 76,711, 4.512e-03 [4.481e-03, 4.544e-03] | 6.260e-05 / 6.282e-05 | -0.35 % |

The P_L intervals are in `out/confirm/run_1/d1_result.json` (`points.<p>.paths.<path>.D` and `.C`).

### Routing, the kept shots and the cap

A kept shot is one that S55'-deep decides (margin at least tau1). Among the kept
shots, a break is a shot that S55'-deep fails and C14' decodes, and a repair the reverse.

| p | C14' path | routed | kept: breaks / repairs / both fail | C14''s failures: routed / kept (kept share) | S55'-deep alone / C14' failures |
|---|---|---|---|---|---|
| 0.2 % | eager | 220,058 (2.00 %) | 19 / 86 / 238 | 7,598 / 324 (4.1 %) | 10,348 / 7,922 |
| 0.2 % | single | 220,058 (2.00 %) | 19 / 78 / 238 | 7,596 / 316 (4.0 %) | 10,348 / 7,912 |
| 0.2 % | batched | 220,058 (2.00 %) | 19 / 79 / 238 | 7,601 / 317 (4.0 %) | 10,348 / 7,918 |
| 0.3 % | eager | 1,701,544 (10.01 %) | 81 / 342 / 1,977 | 74,411 / 2,319 (3.0 %) | 112,636 / 76,730 |
| 0.3 % | single | 1,701,544 (10.01 %) | 81 / 348 / 1,977 | 74,351 / 2,325 (3.0 %) | 112,636 / 76,676 |
| 0.3 % | batched | 1,701,544 (10.01 %) | 81 / 347 / 1,977 | 74,387 / 2,324 (3.0 %) | 112,636 / 76,711 |

On every path D's failures on the routed shots are C14''s on that path (`D_equals_C_on_routed` holds for all six).
In batched mode D's routed shots were decided in shot order in batches of 4,096, as the pipeline queues them, and C14'
alone decided the block in consecutive batches of 4,096; on the routed shots the two gave the same failures.

S55'-deep alone, as a description: it fails 10,348 times at 0.2 % and 112,636 times at 0.3 %, against C14''s 7,922 and
76,730 on the eager path (differences 2.21e-04 [2.06e-04, 2.36e-04] and 2.11e-03 [2.08e-03, 2.14e-03], 4,735 / 2,309
and 54,491 / 18,585 discordant shots). The compiled paths give nearly the same numbers (`d1_tables.md`).

### How the compiled paths differ from the eager one

| p | compiled path | C14' decisions that differ from eager | C14' failures that change (eager only / compiled only) | of them on kept shots | D failures that change |
|---|---|---|---|---|---|
| 0.2 % | single | 759 | 214 (112 / 102) | 68 | 146 |
| 0.2 % | batched | 768 | 208 (106 / 102) | 69 | 139 |
| 0.3 % | single | 6,183 | 1,326 (690 / 636) | 232 | 1,094 |
| 0.3 % | batched | 6,137 | 1,307 (663 / 644) | 237 | 1,070 |

The compiled paths decide about 7 in 100,000 shots differently from the eager one at 0.2 % and about 36 in 100,000 at
0.3 %. The failures they change fall mostly on routed shots, where D takes C14''s decision, so most of them change D and
C14' together and leave the paired difference alone. The gain on the compiled paths is 10-12 % smaller than on the
eager path at 0.2 % (59 and 60 shots net against 67) and about 2 % larger at 0.3 % (267 and 266 against 261). In phase
C's first-lock check on 100,000 shots the one-shot eager gain at 0.2 % was gone on both compiled paths (plan 1.2); on
D1's 11,000,000 fresh shots both compiled intervals lie below zero.

### Descriptive rows (outside the family, no p-value, no verdict)

S55'-deep -> C14' at the first lock's other grid points, each with the first lock's tau1, on the same shots:

| p | q1 | routed | eager: D / C14' (only D / only C14'), difference [95 %] | compiled@1: the same |
|---|---|---|---|---|
| 0.2 % | 1 % | 1.00 % | 7,892 / 7,922 (79 / 109), -2.73e-06 [-5.20e-06, -2.64e-07] | 7,889 / 7,912 (79 / 102), -2.09e-06 [-4.52e-06, 3.28e-07] |
| 0.2 % | 5 % | 4.99 % | 7,852 / 7,922 (4 / 74), -6.36e-06 [-7.99e-06, -4.77e-06] | 7,850 / 7,912 (4 / 66), -5.64e-06 [-7.18e-06, -4.12e-06] |
| 0.2 % | 10 % | 9.98 % | 7,867 / 7,922 (3 / 58), -5.00e-06 [-6.45e-06, -3.58e-06] | 7,859 / 7,912 (3 / 56), -4.82e-06 [-6.24e-06, -3.42e-06] |
| 0.2 % | 20 % | 19.96 % | 7,886 / 7,922 (1 / 37), -3.27e-06 [-4.44e-06, -2.13e-06] | 7,878 / 7,912 (1 / 35), -3.09e-06 [-4.23e-06, -1.98e-06] |
| 0.2 % | 40 % | 39.97 % | 7,902 / 7,922 (1 / 21), -1.82e-06 [-2.73e-06, -9.20e-07] | 7,894 / 7,912 (1 / 19), -1.64e-06 [-2.51e-06, -7.74e-07] |
| 0.3 % | 1 % | 1.00 % | 85,258 / 76,730 (14,508 / 5,980), 5.02e-04 [4.85e-04, 5.18e-04] | 85,199 / 76,676 (14,502 / 5,979), 5.01e-04 [4.85e-04, 5.18e-04] |
| 0.3 % | 2 % | 2.00 % | 79,210 / 76,730 (4,954 / 2,474), 1.46e-04 [1.36e-04, 1.56e-04] | 79,154 / 76,676 (4,952 / 2,474), 1.46e-04 [1.36e-04, 1.56e-04] |
| 0.3 % | 5 % | 5.00 % | 76,710 / 76,730 (637 / 657), -1.18e-06 [-5.33e-06, 2.98e-06] | 76,663 / 76,676 (641 / 654), -7.65e-07 [-4.92e-06, 3.39e-06] |
| 0.3 % | 20 % | 20.02 % | 76,526 / 76,730 (12 / 216), -1.20e-05 [-1.38e-05, -1.03e-05] | 76,468 / 76,676 (12 / 220), -1.22e-05 [-1.40e-05, -1.05e-05] |
| 0.3 % | 40 % | 40.00 % | 76,617 / 76,730 (1 / 114), -6.65e-06 [-7.91e-06, -5.39e-06] | 76,572 / 76,676 (1 / 105), -6.12e-06 [-7.33e-06, -4.91e-06] |

At 0.2 % routing 2 % to 10 % of the shots gives similar gains on the eager path (5.0e-06 to 6.4e-06); routing 1 %
gives less, and the gain shrinks beyond 10 %. At 0.3 % D fails more often than C14' when it routes 1 % or 2 %, the
interval at 5 % contains zero, and the gain at 20 % (1.20e-05) is below the locked configuration's 1.54e-05 at 10 %.

## 3. Speed

D1-S (job CJ73, node gpu-node-07) ran phase C's pipeline protocol with milestone 2's harness (`m2.py pipeline`,
48150303, through `d1_m2.py` with D1's lock): 36 fresh processes in a rep-major loop, each run three times, Cascade
alone in the same job and loop. Single mode decodes 20,000 shots one at a time after 200 untimed ones; batched mode
decodes 500,000 shots at batch 4,096. The shots are phase C's speed stream (seed kind 12), the same as in C7 and
milestone 2. Each value is the slowest of the three processes (`pc_report.pipes`, once per engine, with Cascade alone's
rows in both). Per round is per shot divided by 6. A speed-up is Cascade alone's value divided by D's.

| p, q1 | S55'-deep engine | single mean, us (per round): D / Cascade alone | speed-up | p99.9, us: D / Cascade alone | batched, us per shot (per round): D / Cascade alone | speed-up | clock flags: D / Cascade alone | S checks |
|---|---|---|---|---|---|---|---|---|
| 0.2 %, 2 % | today's (fused) | 163.2 (27.2) / 1,759.8 (293.3) | 10.78 | 1,842 / 1,777 | 8.13 (1.355) / 72.03 (12.00) | 8.86 | 0 of 6 / 4 of 6 | all zero |
| 0.2 %, 2 % | milestone 2's (`mfused`) | 138.8 (23.1) / 1,759.8 (293.3) | 12.68 | 1,914 / 1,777 | 4.40 (0.734) / 72.03 (12.00) | 16.36 | 1 of 6 / 4 of 6 | all zero |
| 0.3 %, 10 % | today's (fused) | 300.5 (50.1) / 1,763.8 (294.0) | 5.87 | 1,844 / 1,781 | 13.99 (2.332) / 72.04 (12.01) | 5.15 | 0 of 6 / 3 of 6 | all zero |
| 0.3 %, 10 % | milestone 2's (`mfused`) | 276.6 (46.1) / 1,763.8 (294.0) | 6.38 | 1,847 / 1,781 | 10.26 (1.710) / 72.04 (12.01) | 7.02 | 0 of 6 / 3 of 6 | all zero |

D is faster than Cascade alone on average in both modes, on both engines, at both points. Its 99.9th percentile is
higher than Cascade alone's in all four rows (by 3.5-7.7 %), because the routed shots pass through both S55'-deep and
C14'. "Clock flags" counts the processes below the 97 % rule among the six of each run (three single, three batched);
Cascade alone's six processes per point serve both engines. "S checks" is each D process's check of S55'-deep against
C4b's frozen read (2,000 shots in single mode, 32,768 in batched mode): no decision and no margin bit differed in any of
the 24 D processes.

### Against the predictions of plan 4.3

| p | engine | D single mean, us: predicted / measured | D batched, us per shot: predicted / measured | speed-ups single, batched: predicted / measured |
|---|---|---|---|---|
| 0.2 % | today's | 166.5 / 163.2 | 8.2 / 8.13 | 10.2, 8.8 / 10.78, 8.86 |
| 0.2 % | milestone 2's | 136 / 138.8 | 4.6 / 4.40 | 12.5, 15.6 / 12.68, 16.36 |
| 0.3 % | today's | 309 / 300.5 | 13.9 / 13.99 | 5.5, 5.2 / 5.87, 5.15 |
| 0.3 % | milestone 2's | 279 / 276.6 | 10.4 / 10.26 | 6.1, 6.9 / 6.38, 7.02 |

The predictions used C7's Cascade alone (1,696.5 us single and 71.87 us per shot batched at 0.2 %, 1,700.8 and 71.88
at 0.3 %). In D1-S Cascade alone's single-shot mean was 3.7 % higher at both points (1,759.8 and 1,763.8 us) and its
batched time 0.2 % higher. D's own times differed from the predictions by 4.3 % or less.

### The SM clock flags

Every timing row records the lowest SM clock that `gpu_watch` sampled, as a fraction of the card's maximum
(`clock.clock_min_ratio`), and whether it met the 97 % rule (`clock.accepted`). As in phase C, no process was rerun or
replaced, and the reported value is the slowest process whether or not it met the rule. `other_pids` is empty in every
row, so no other process was on the card.

Eight of the 36 processes were flagged, all at 94.12-94.73 %. Accepted processes were at 97.82-99.67 %. The flagged rows
(line numbers of `out/speed/rows.jsonl`):

- All six single-mode processes of Cascade alone: rows 1, 13 and 25 at 0.2 % and rows 7, 19 and 31 at 0.3 %. Cascade
  alone's single-shot means and 99.9th percentiles in the table come from flagged processes only.
- One batched process of Cascade alone at 0.2 % (row 2, 71.99 us per shot). It was the fastest of the three, so the
  reported 72.03 comes from an accepted process.
- One single-mode process of D on milestone 2's engine at 0.2 % (row 4: 138.8 us, p99.9 1,914 us). It is the slowest of
  the three and supplies the reported values; the other two, accepted, gave 136.4 and 136.5 us with p99.9 1,813 us.

In C7 (job CJ39) none of Cascade alone's twelve single-mode processes was flagged, and its 15 flagged rows were
all in batched mode (`phase_c/out/speed/rows.jsonl`). Phase C's timings ran on gpu-node-06 and u5 and D1-S on
gpu-node-07, all with driver 590.48.01. Divided into C7's Cascade-alone single-shot means instead, D's
single-shot means would give speed-ups of 10.4 and 12.2 at 0.2 % and 5.66 and 6.15 at 0.3 %. The registered speed-ups
are those in the table above; these four show how much they depend on Cascade alone's slower single-shot runs in D1-S.

## 4. Checks and provenance

- The freeze. F = 310ce8e0f8bfb9d3b97b27186547a836d29fc9ca was committed at 16:29:32 MDT on 11 October and pushed to
  the hub (`ising-cascade-d1`) before the four jobs were submitted (16:29:58-16:30:01 MDT). It holds the plan, the lock
  (sha256 3ab39941df356883...), the code and the dry-run records. The deployed tree is F, clean (`DEPLOYED`: "net
  64b2e25d d1 310ce8e0...").
- The chunks. c0 = int("310ce8e", 16) mod 9,000,000 = 6,433,102. R1 read chunks 6,433,102 to 6,434,201 at 0.2 % and
  R2 chunks 6,433,102 to 6,434,801 at 0.3 %, with seed kind 14 (`eval_seed(14, p, j, "depspam_fm")`, so the two
  points draw different shots). Each read recorded every chunk's seed, and D1-A recomputed c0 from F and checked every
  seed. The compiled reads checked every chunk's digest against the eager read's. D1-A refused nothing; it ran once and
  wrote `run_1`.
- The guards. Each read job's guard passed before it drew a shot: the tree was F and clean, no record outside the read's
  own directories named kind 14, and the read was not done (`out/logs/d1_r1_CJ71.out`, `d1_r2_CJ72.out`).
- The circuits. The reads used the same circuits as phase C's development and confirmation reads at each p (circuit
  sha256 dcaae19e... at 0.2 % and 647fda02... at 0.3 %).
- The compiled reads (`out/compiled/p0.00200.json`, `p0.00300.json`). The concurrent single read (32 streams x 16
  batch-1 calls) was bitwise equal to phase C's own compiled@1 path object, `pc_speed.build(C14', "compiled",
  1).run`, on the first 20,000 rows before the read and on every 1,000th row after it (11,000 and 17,000 rows): no
  decision and no confidence bit differed. The S anchor passed at both points: S55'-deep's read engine gave every
  decision and margin bit of C4b's frozen read on its first 32,768 rows. Both compiled graphs loaded from the copies of
  phase C's caches (two FX-graph cache hits per read).
- The first lock. D1-A checked the lock's copied configurations against phase C's first-lock file
  (`cascade_w4/out/locked_depspam_fm.json`, sha256 02b19c45...).
- The outputs. The npz files stay on Alpine under `<cluster-scratch>/cascade_d1/out/`; their sha256 sums are in
  `out/npz_sha256.txt`.
- Before the freeze. The dry jobs (CJ67, CJ69, CJ70) ran the whole pipeline on seed kind 13 and made every
  refusal and guard fire; D1-P (CJ68) computed the design-scenario powers, which kept the block sizes. Their records
  are in `dry/` and were committed in F. Nothing from them is reported as a result.

## 5. Disclosures

- Origin. D1's hypotheses came from a post hoc look at phase C's confirmation block. EXPLORE_DEEP evaluated 52
  configurations on those reads after the registered results were known, without adjusting for multiplicity. D1 does
  not claim that its hypotheses were specified before phase C's confirmation data were seen.
- Superseded lock. D1 tested the two claim-point configurations that phase C's first lock (1e8d1c6d) picked on the
  development block before any confirmation read, with the first lock's thresholds, never refitted. Amendment 1
  (629ca1c4) replaced that lock before any confirmation read, and Amendment 1's lock remains phase C's lock of record.
- Scope. `depspam_fm` is a fitted noise model (the noise-model fit's variant), not the paper's noise, and C14' is our
  re-implementation of Cascade, not the published network. Everything ran on one card model with one software stack
  (section 1).
- Effect sizes. On phase C's confirmation block the two configurations were 7.2e-06 and 7.0e-06 below Cascade's block
  failure rate (0.93 % and 0.16 % of it), capped by Cascade's failures on kept shots (4.7 % and 3.1 % of them). On D1's
  fresh shots the gains on the timed computation are 5.36e-06 (single) and 5.45e-06 (batched) at 0.2 % (0.75 % and
  0.76 % of C14''s rate) and 1.57e-05 and 1.56e-05 at 0.3 % (0.35 %), with caps of 4.0 % and 3.0 %. At 0.3 % the gain
  is about twice the one on phase C's block, which rested on 15 discordant shots.
- Phase C's verdicts do not change.
- Fresh shots. Seed kind 14 was reserved for D1. Its chunks were set by the freeze commit, drawn after the plan was
  frozen and pushed, and analysed once. Their freshness is audited provenance (plan 2.3): the audit and both read jobs'
  searches found no other use of kind 14, and the analysis checked every seed against c0 of F. A commit hash can be
  computed before it is published, so the offset alone does not prove freshness.
- The compiled paths. C14' was read on the eager path and on both compiled paths on the same shots; section 2 gives how
  often they differ.
- The clocks. Eight of the 36 timing processes, including every single-mode process of Cascade alone, ran below the
  97 % rule at some point (section 3).
- The speed shots. The timings use phase C's speed stream (kind 12), not D1's fresh shots; the pipelines select nothing
  from them.

D1 claims none of the following (plan 4.4): superiority over the published Cascade; anything at another p or in
`uniform`, which D1 does not read; anything on the H100, which D1 does not time; superiority in tail latency; a gain
beyond the intervals; or phase C's "beats Cascade" (its section 6), whose condition that the decoder is not worse at any
other p where it was read is not tested for these configurations.

Nothing was read again, enlarged or retested, and no other configuration is claimed from these shots. Now that they have
been analysed, D1's shots are no longer fresh.

## 6. Files committed with this report

| what | file |
|---|---|
| verdicts, tables, claims | `out/confirm/run_1/d1_result.json`, `out/confirm/run_1/d1_tables.md` (D1-A, job CJ74) |
| eager reads (per-chunk seeds and digests, rows) | `out/read/p0.00200.json`, `p0.00300.json`, `rows_0.002.json`, `rows_0.003.json`, `block_0.002.json`, `block_0.003.json` (R1, R2) |
| compiled reads (checks, S anchor, cache counters, times) | `out/compiled/p0.00200.json`, `p0.00300.json` (R1, R2) |
| timing rows | `out/speed/rows.jsonl` (D1-S, job CJ73) |
| job logs | `out/logs/d1_r1_CJ71.out`, `d1_r2_CJ72.out`, `d1_s_CJ73.out`, `d1_a_CJ74.out` |
| sha256 of the npz files on Alpine | `out/npz_sha256.txt` |
