# Phase C results: our BB72 decoders against Cascade

Written on 11 October 2026. Phase C ran on Alpine on 10 and 11 October (`phase_c/RUN.md`). `$R/PHASE_C_JOBS_DONE` (`$R`
= `<cluster-scratch>/cascade_w4`) lists 15 jobs. All ended COMPLETED except the first lock's C7 (CJ35), which the
amendment's run cancelled before it started. Every job on the new lock completed: the check CJ38, C7 CJ39, C5 CJ40 and
CJ41, C6 CJ42 and C8 CJ43. This file reports what the report job (C8) computed from the confirmation analyses, the
checks and the timings. It changes no result, lock or code. PHASE_C_PLAN.md (section 6) calls this report
`cascade_compare/PHASE_C_RESULT.md`; it is kept here, with the other records of phase C.

This file was revised on 11 October after the review of the results (SOUND WITH FIXES). The revision states the
accuracy and the speed of S -> B -> R separately, discloses the timings' clock flags (section 4.3), narrows the
conclusions for the paper (section 1, "What may be claimed") and points every H100 number at committed files. It changes
no number, verdict, lock or code, and nothing was measured again.

"Cascade" is our re-implementation of the paper's network: C14 in `uniform` and C14' in `depspam_fm` (plan 2.1). It
is neither the paper's network nor its numbers. `depspam_fm` is the noise-model fit to the paper's setting (17-20
Tesseract failures), not an identification of it. The work is exploratory: every shot is fresh (seed kinds 10, 11 and
12), and no registered test set was read. Every primary GPU number comes from one card model, a full NVIDIA RTX PRO 6000
Blackwell Server Edition (driver 590.48.01, torch 2.11.0+cu128). The Cascade networks run in BF16. Our ConvLines
networks run on their fused engine (FP16 matmul inputs, FP32 accumulation). Relay-BP runs on the node's AMD EPYC 9555.

Paths below are under `cascade_compare/phase_c/out/` unless stated otherwise. `$H` is
`cascade_compare/phase_c/out_h100/` on branch `ising-cascade-h100t2` (commit d1575924), which holds the H100 outputs.

| what | file | written by |
|---|---|---|
| verdicts; curves, confirmation, check and speed tables | `report/phase_c_tables.md`, `report/phase_c_summary.json` | C8, job CJ43 |
| the confirmation analyses | `confirm_uniform.json`, `confirm_depspam_fm.json` | C6, job CJ42 (reads: C5, jobs CJ40 and CJ41) |
| the lock under Amendment 1 | `locked_uniform.json`, `locked_depspam_fm.json` | C4 again, job CJ37 (commit 629ca1c4) |
| the claim-point check of that lock | `check_uniform.json`, `check_depspam_fm.json` | C4b again, job CJ38 (commit 1637db9c) |
| speed rows and frozen paths | `speed/rows.jsonl`, `speed_components.json` | C3 (job CJ30), C7 (job CJ39) |
| H100 column | `H100_TIMING.md` and `$H` on `ising-cascade-h100t2` (d1575924): `$H/first_pod/`, `$H/second_pod/` and `$H/screen/`, copied from `~/work/cascade_h100t/results/out/`, `~/work/cascade_h100b/results/out/` and `~/work/cascade_h100b/results/screen/` on the Linux workstation | the first and second pods' runs; screen job CJ47 |

## 1. The verdicts

The report applies section 6 of the plan mechanically. A decoder of ours beats Cascade in a setting when four things
hold at p = 0.2 % and at p = 0.3 %. First, it is "better" than Cascade alone on the same shots in the headline
family: 132 comparisons over both settings, judged by one Holm procedure at alpha = 0.05. Second, its mean
single-shot latency and its amortized latency are both lower than Cascade's, measured in the same job. Third, it is
not "worse" at any other p where it was read. Fourth, every accuracy check behind it passed: the screen of its frozen
paths and the claim-point check, which test that the timed paths decide as the reads do. "Level" means no significant
difference under Holm; it does not mean equivalence.

In `depspam_fm`, S -> B -> R meets these four conditions, so by the plan's rule it beats Cascade (C14'). In this
decoder S55' decides most shots, C14' decides the shots S55' is least sure of, and full-DEM Relay-BP decides the routed
shots on which C14' is least sure. Its accuracy and its speed were measured with Relay-BP used in two different ways,
so they are stated separately: S -> B -> R reduced confirmed block failures at both claim points in fitted
`depspam_fm`; separately, its retained-decoder deployment was faster on average in both modes on the RTX PRO 6000 with
the node's CPU (an AMD EPYC 9555), with slower tails. Its 99.9th percentile single-shot latency is above Cascade's at
both claim points. The accuracy reads build a new seeded Relay-BP decoder for every routed shot, while the timings keep
one decoder per process. At 0.3 %, 7.4 ms per build is about 17.8 us of added CPU work per incoming shot, spread over
16 asynchronous workers in batched mode. That is not a measured wall-time overhead, so it is not to be subtracted from
the 7.9 us batched lead over Cascade. No claim is made about the speed of the seeded implementation (section 4.1).

In `uniform`, no tested decoder met the joint win criterion, so none of ours beats Cascade (C14). S -> B -> R there is
better at 0.3 % and worse at 0.2 %.

The accuracy checks removed nothing. The claim-point check on the new lock passed all 48 rows in each setting, no entry
of `decoders` is false in either file, and Cascade's own paths passed (`cascade_not_eligible` is empty in both
settings). The timings also record whether the GPU's SM clock met `gpu_watch`'s 97 % rule. That rule is not one of
the checks, and the report ignores it; section 4.3 lists the processes that missed it and what they affect.

Source: `report/phase_c_summary.json` (`verdicts`), `report/phase_c_tables.md` (section "Verdicts").

| setting | decoder | 0.2 % | 0.3 % | other p | faster (single shot, and amortized or batched) | beats Cascade |
|---|---|---|---|---|---|---|
| `uniform` | S55 alone | worse | worse | worse at all 8 | yes | no |
| `uniform` | S55' alone | worse | worse | worse at all 8 | yes | no |
| `uniform` | S55'-deep alone | worse | worse | worse at all 8 | yes | no |
| `uniform` | r11 alone | worse | worse | worse at all 8 | yes | no |
| `uniform` | r12 alone | worse | worse | worse at all 8 | yes | no |
| `uniform` | S -> B | level | worse | worse at 0.45 %, level at the other 7 | yes (slower tail) | no |
| `uniform` | S -> B -> R | worse | better | not locked there | yes (slower tail) | no |
| `uniform` | S -> R | worse | worse | not locked there | no (slower at 0.3 %) | no |
| `uniform` | r12 -> R | worse | worse | not locked there | no (slower at 0.3 %) | no |
| `depspam_fm` | S55 alone | worse | worse | worse at all 8 | yes | no |
| `depspam_fm` | S55' alone | worse | worse | worse at all 8 | yes | no |
| `depspam_fm` | S55'-deep alone | worse | worse | level at 0.1 %, worse at the other 7 | yes | no |
| `depspam_fm` | r11 alone | worse | worse | worse at all 8 | yes | no |
| `depspam_fm` | r12 alone | worse | worse | worse at all 8 | yes | no |
| `depspam_fm` | S -> B | level | worse | worse at 0.25 % and 0.4 %, level at the other 6 | yes (slower tail) | no |
| `depspam_fm` | S -> B -> R | better | better | not locked there | yes (slower tail) | yes |
| `depspam_fm` | S -> R | worse | worse | not locked there | no (slower at 0.3 %) | no |
| `depspam_fm` | r12 -> R | worse | worse | not locked there | no (slower at 0.3 %) | no |

In S -> B, S -> B -> R and S -> R, the small network S is S55 in `uniform` and S55' in `depspam_fm` at both claim
points (S55'-deep at some other p, section 3); in r12 -> R it is r12. B is C14 or C14'. R is full-DEM Relay-BP
(relay-bp 0.2.2). A network alone is judged on the curves block (seed kind 10), and its speed column compares its
single-shot and amortized latencies with Cascade's in C3. A routed decoder is judged on the confirmation block (seed
kind 11), and its speed column compares it with Cascade alone in C7, in single and batched mode at both claim points.

### What may be claimed

the review (finding 6) asks that the paper's conclusions stay narrow. These are the statements it supports, in the
review's wording where the review gives one:

- S -> B -> R reduced confirmed block failures at both claim points in fitted `depspam_fm`; separately, its
  retained-decoder deployment was faster on average in both modes on the RTX PRO 6000 with the node's CPU, with slower
  tails.
- S -> B was faster on average at both claim points, with excess block failure rate bounded by the frozen margin (10 %
  of Cascade's development-block rate). At 0.3 % it is significantly worse than Cascade in both settings.
- No tested decoder met the joint win criterion in `uniform`.
- The H100 results are secondary and speed only, and they depend on the GPU and CPU combination; `depspam_fm`
  S -> B -> R at 0.3 % was slower than Cascade there in batched mode (section 4.2).
- Every routed decoder has a higher 99.9th percentile single-shot latency than Cascade on the RTX PRO 6000. This holds
  on that card only (section 4.2).

Claims the paper must not make:

- superiority over the published Cascade implementation ("Cascade" here is our re-implementation, C14 and C14');
- superiority at every p (S -> B -> R is locked only at the two claim points, and S -> B is worse than Cascade at
  0.3 % in both settings and at some other p);
- superiority on the H100;
- superiority in tail latency;
- that S -> B is as accurate as Cascade;
- anything about the speed of the seeded Relay-BP decoder that the accuracy reads use (section 4.1);
- that the selection procedure as a whole was preregistered (Amendment 1 changed its rule after the development
  curves had been seen, section 6).

## 2. The claim points

The confirmation block was read once, after the new lock had been committed and after its claim-point check had ended
without error. The C5 jobs waited for the check with `afterok`, and before reading a shot each verified that its
setting's check recorded the new lock's sha256. The check's files were committed only now (commit 1637db9c; `RUN.md`,
deviations 4 and 6), so the plan's order "check committed before C5" did not hold for the commit
itself. Per configuration the tables
give its block failures, the failures of Cascade alone and of its S alone on the same shots, and the difference of
block failure rates (the decoder minus Cascade) with Newcombe's 95 % interval. The McNemar test is exact. Then come
the planned power from the lock, the level at which Holm judged the comparison ("-" where it was not rejected) and
the verdict. The last column is the secondary statement of plan 4.5, which enters no verdict. It asks whether the
upper end of Newcombe's interval at the one-sided level 0.05 / 32 (`vs_B_ni` in `confirm_<setting>.json`) lies below
the margin frozen in the lock: 10 % of Cascade's block failure rate on the development block at that p.

Sources: `report/phase_c_tables.md` (sections "Routing confirmation"), `confirm_<setting>.json` (`rows`: `fail`,
`B_alone`, `vs_S_95`, `vs_B_95`, `vs_B_ni`, `ni_margin`, `noninferior`), `locked_<setting>.json` (`configs`).

### `uniform` (Cascade = C14; 1,000,000 shots at 0.2 %, 1,500,000 at 0.3 %)

| p | configuration (lock index) | thresholds | failures | Cascade alone | S alone | difference [95 %] | McNemar p | planned power | Holm level | verdict | excess bounded by the margin? (upper end; margin) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 % | S -> B (2), S55 -> C14 | q1 1 % | 356 | 360 | 783 | -4.0e-06 [-1.2e-05, 3.7e-06] | 0.39 | - | - | level | yes (9.25e-06; 3.72e-05) |
| 0.2 % | S -> B -> R (3) | q1 1 %, r 0.1 %, min \|logit\| | 435 | 360 | 783 | 7.50e-05 [4.0e-05, 1.1e-04] | 3.6e-05 | 0.054 | 2.5e-03 | worse | no (1.30e-04; 3.72e-05) |
| 0.2 % | S -> R (4), S55 | q 0.25 % | 609 | 360 | 783 | 2.49e-04 [2.1e-04, 2.9e-04] | 9.2e-31 | 0.030 | 1.2e-03 | worse | no (3.16e-04; 3.72e-05) |
| 0.2 % | r12 -> R (5) | q 0.25 % | 591 | 360 | 815 | 2.31e-04 [1.9e-04, 2.7e-04] | 9.8e-27 | 0.029 | 1.4e-03 | worse | no (2.98e-04; 3.72e-05) |
| 0.3 % | S -> B (7), S55 -> C14 | q1 5 % | 3,770 | 3,716 | 8,958 | 3.60e-05 [2.2e-05, 5.0e-05] | 2.5e-07 | - | 2.0e-03 | worse | yes (5.77e-05; 2.53e-04) |
| 0.3 % | S -> B -> R (8) | q1 5 %, r 0.1 %, min \|logit\| | 3,307 | 3,716 | 8,958 | -2.73e-04 [-3.1e-04, -2.4e-04] | 1.1e-52 | 1.000 | 1.0e-03 | better | yes (-2.19e-04; 2.53e-04) |
| 0.3 % | S -> R (9), S55 | q 2 % | 6,419 | 3,716 | 8,958 | 1.80e-03 [1.7e-03, 1.9e-03] | 0 (underflow) | 0.965 | 6.4e-04 | worse | no (1.94e-03; 2.53e-04) |
| 0.3 % | r12 -> R (10) | q 2 % | 6,508 | 3,716 | 10,417 | 1.86e-03 [1.8e-03, 2.0e-03] | 0 (underflow) | 0.956 | 6.5e-04 | worse | no (2.00e-03; 2.53e-04) |

Per logical qubit per round (Eq. (1)): Cascade has P_L 5.00e-06 at 0.2 % and 3.45e-05 at 0.3 %; S -> B 4.95e-06 and
3.50e-05; S -> B -> R 6.04e-06 and 3.07e-05.

### `depspam_fm` (Cascade = C14'; 5,000,000 shots at 0.2 %, 1,000,000 at 0.3 %)

| p | configuration (lock index) | thresholds | failures | Cascade alone | S alone | difference [95 %] | McNemar p | planned power | Holm level | verdict | excess bounded by the margin? (upper end; margin) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.2 % | S -> B (2), S55' -> C14' | q1 1 % | 3,896 | 3,858 | 6,858 | 7.6e-06 [2.1e-06, 1.3e-05] | 7.4e-03 | - | - | level | yes (1.60e-05; 7.01e-05) |
| 0.2 % | S -> B -> R (3) | q1 2 %, r 0.1 %, min \|logit\| | 3,596 | 3,858 | 6,858 | -5.24e-05 [-7.1e-05, -3.4e-05] | 1.9e-08 | 1.000 | 1.9e-03 | better | yes (-2.49e-05; 7.01e-05) |
| 0.2 % | S -> R (4), S55' | q 0.25 % | 5,330 | 3,858 | 6,858 | 2.94e-04 [2.7e-04, 3.2e-04] | 6.2e-124 | 0.977 | 8.2e-04 | worse | no (3.32e-04; 7.01e-05) |
| 0.2 % | r12 -> R (5) | q 0.25 % | 5,359 | 3,858 | 7,151 | 3.00e-04 [2.8e-04, 3.3e-04] | 6.9e-127 | 0.973 | 7.9e-04 | worse | no (3.38e-04; 7.01e-05) |
| 0.3 % | S -> B (7), S55' -> C14' | q1 5 % | 4,550 | 4,459 | 9,708 | 9.10e-05 [6.2e-05, 1.2e-04] | 6.8e-10 | - | 1.7e-03 | worse | yes (1.36e-04; 4.61e-04) |
| 0.3 % | S -> B -> R (8) | q1 5 %, r 0.25 %, min \|logit\| | 3,957 | 4,459 | 9,708 | -5.02e-04 [-5.7e-04, -4.3e-04] | 9.4e-47 | 1.000 | 1.1e-03 | better | yes (-3.98e-04; 4.61e-04) |
| 0.3 % | S -> R (9), S55' | q 2 % | 6,827 | 4,459 | 9,708 | 2.37e-03 [2.2e-03, 2.5e-03] | 6.3e-259 | 0.999 | 6.8e-04 | worse | no (2.58e-03; 4.61e-04) |
| 0.3 % | r12 -> R (10) | q 2 % | 6,988 | 4,459 | 11,382 | 2.53e-03 [2.4e-03, 2.7e-03] | 6.3e-289 | 0.998 | 6.8e-04 | worse | no (2.74e-03; 4.61e-04) |

Per logical qubit per round (Eq. (1)): Cascade has P_L 1.07e-05 at 0.2 % and 6.21e-05 at 0.3 %; S -> B 1.08e-05 and
6.33e-05; S -> B -> R 9.99e-06 and 5.51e-05.

At 0.2 % the 95 % interval of S -> B lies above zero (38 more failures than Cascade in 5,000,000 shots), but the
comparison did not reach its Holm level, so its verdict is "level".

### S -> B's accuracy

For S -> B, the excess block failure rate is bounded by the frozen margin at both claim points in both settings. The
measured differences from Cascade are -4.0e-06 (`uniform`, 0.2 %), +3.6e-05 (`uniform`, 0.3 %), +7.6e-06
(`depspam_fm`, 0.2 %) and +9.1e-05 (`depspam_fm`, 0.3 %). The upper ends at the one-sided level 0.05 / 32 are
9.25e-06, 5.77e-05, 1.60e-05 and 1.36e-04, against margins of 3.72e-05, 2.53e-04, 7.01e-05 and 4.61e-04. This is
the secondary statement of plan 4.5. The headline verdicts of S -> B are "level" at 0.2 % and "worse" at 0.3 % in
both settings. At 0.3 % S -> B is therefore significantly less accurate than Cascade in both settings, and the paper
must not call it equally accurate.

Every discordant shot between S -> B and Cascade is a shot that S keeps; on the routed shots S -> B takes B's own
decision. In `depspam_fm` at 0.3 %, S55' repairs 64 of C14''s failures on the shots it keeps and adds 155 of its own;
in `uniform` at 0.3 %, S55 repairs 28 and adds 82 (column "kept: repairs / regressions vs B" in the tables).

## 3. The other p

Only S -> B is locked at the eight other p. The lock gives S -> B no planned power at any p, the claim points
included: for every one of its ten configurations in each setting, `sizing.why` in `locked_<setting>.json` reads "the
development block's discordant rate is below 10 % of B's failure rate" (plan 4.4). Sources as in section 2.

| setting | p | S, q1 | failures / shots | Cascade alone | difference [95 %] | McNemar p | verdict | excess bounded by the margin? (upper end; margin) |
|---|---|---|---|---|---|---|---|---|
| `uniform` | 0.10 % | S55, 1 % | 260 / 10,000,000 | 260 | 0.0 [-6.1e-07, 6.1e-07] | 1.0 | level | yes (1.13e-06; 2.38e-06) |
| `uniform` | 0.15 % | S55, 1 % | 349 / 3,000,000 | 348 | 3.33e-07 [-1.1e-06, 1.8e-06] | 1.0 | level | yes (3.43e-06; 1.21e-05) |
| `uniform` | 0.25 % | S55, 2 % | 506 / 500,000 | 492 | 2.80e-05 [7.8e-06, 4.9e-05] | 6.6e-03 | level | yes (6.27e-05; 9.92e-05) |
| `uniform` | 0.35 % | S55, 10 % | 1,579 / 300,000 | 1,577 | 6.67e-06 [-3.0e-05, 4.4e-05] | 0.85 | level | yes (6.63e-05; 5.38e-04) |
| `uniform` | 0.40 % | S55, 20 % | 3,336 / 300,000 | 3,328 | 2.67e-05 [-1.7e-05, 7.1e-05] | 0.28 | level | yes (9.68e-05; 1.15e-03) |
| `uniform` | 0.45 % | S55, 20 % | 6,328 / 300,000 | 6,275 | 1.77e-04 [1.0e-04, 2.5e-04] | 5.0e-06 | worse | yes (2.94e-04; 2.11e-03) |
| `uniform` | 0.50 % | S55, 40 % | 11,358 / 300,000 | 11,352 | 2.00e-05 [-3.1e-05, 7.1e-05] | 0.51 | level | yes (9.98e-05; 3.86e-03) |
| `uniform` | 0.55 % | S55, 40 % | 19,151 / 300,000 | 19,124 | 9.00e-05 [1.3e-05, 1.7e-04] | 2.6e-02 | level | yes (2.08e-04; 6.31e-03) |
| `depspam_fm` | 0.10 % | S55', 1 % | 567 / 10,000,000 | 574 | -7.00e-07 [-1.5e-06, 4.0e-08] | 6.5e-02 | level | yes (5.84e-07; 5.61e-06) |
| `depspam_fm` | 0.15 % | S55', 1 % | 717 / 3,000,000 | 719 | -6.67e-07 [-4.2e-06, 2.9e-06] | 0.85 | level | yes (5.12e-06; 2.50e-05) |
| `depspam_fm` | 0.25 % | S55'-deep, 1 % | 1,062 / 500,000 | 1,022 | 8.00e-05 [4.1e-05, 1.2e-04] | 5.5e-05 | worse | yes (1.41e-04; 1.89e-04) |
| `depspam_fm` | 0.35 % | S55'-deep, 5 % | 3,005 / 300,000 | 2,973 | 1.07e-04 [2.1e-05, 1.9e-04] | 1.7e-02 | level | yes (2.38e-04; 9.70e-04) |
| `depspam_fm` | 0.40 % | S55'-deep, 10 % | 5,836 / 300,000 | 5,789 | 1.57e-04 [5.9e-05, 2.6e-04] | 2.0e-03 | worse | yes (3.07e-04; 1.92e-03) |
| `depspam_fm` | 0.45 % | S55'-deep, 20 % | 10,892 / 300,000 | 10,869 | 7.67e-05 [-7.0e-06, 1.6e-04] | 8.3e-02 | level | yes (2.05e-04; 3.56e-03) |
| `depspam_fm` | 0.50 % | S55', 40 % | 18,773 / 300,000 | 18,741 | 1.07e-04 [2.5e-05, 1.9e-04] | 1.2e-02 | level | yes (2.32e-04; 6.24e-03) |
| `depspam_fm` | 0.55 % | S55'-deep, 40 % | 30,881 / 300,000 | 30,861 | 6.67e-05 [-3.4e-05, 1.7e-04] | 0.21 | level | yes (2.19e-04; 1.01e-02) |

Our networks alone are worse than Cascade at every p of both settings, with one exception: S55'-deep is level at
0.1 % in `depspam_fm` (`report/phase_c_tables.md`, sections "Curves").

## 4. Speed

### 4.1 RTX PRO 6000 (primary)

The networks alone were timed in C3 (job CJ30), all in one job. Each value is the slowest of 3 processes. Single
shot is the mean of 10,000 host round trips at batch 1 on the frozen single path. Amortized is the host-to-host time per shot on
the frozen amortized path, the fastest screened batch size up to 4,096. Per round is per shot / 6. The two H100
columns (secondary, section 4.2) come from the first pod. Sources: `report/phase_c_tables.md` (section "Speed"),
`speed_components.json` (`nets`: `t1_us`, `p999_us`, `a_best`, `a_best_us`); H100: `$H/screen/first/summary.txt`.
The single-shot values of C14, C14', C6, C6', S55' and S55'-deep come from processes whose SM clock missed the 97 %
rule (section 4.3).

| network | precision | single shot (us) | per round (us) | p99.9 (us) | amortized path | amortized per shot (us) | per round (us) | H100 single (us) | H100 amortized (us per shot) |
|---|---|---|---|---|---|---|---|---|---|
| C14 | BF16 | 1,783.4 | 297.24 | 1,790.2 | compiled@4096 | 71.63 | 11.94 | 1,666.5 | 42.951 |
| C14' | BF16 | 1,781.9 | 296.98 | 1,790.0 | compiled@4096 | 71.63 | 11.94 | 1,664.9 | 42.985 |
| C6 | BF16 | 798.4 | 133.06 | 805.8 | compiled@4096 | 31.35 | 5.23 | 761.0 | 18.674 |
| C6' | BF16 | 794.4 | 132.40 | 805.9 | compiled@4096 | 31.36 | 5.23 | 759.8 | 18.685 |
| S55 | fused | 88.2 | 14.70 | 93.6 | fused@1024 | 4.14 | 0.69 | 98.5 | 5.631 |
| S55' | fused | 88.0 | 14.67 | 93.7 | fused@1024 | 4.14 | 0.69 | 96.4 | 5.629 |
| S55'-deep | fused | 130.9 | 21.81 | 141.2 | fused@4096 | 6.26 | 1.04 | 146.2 | 9.029 |
| r11 | fused | 88.1 | 14.68 | 93.2 | fused@1024 | 4.14 | 0.69 | 97.0 | 5.628 |
| r12 | fused | 88.0 | 14.67 | 93.2 | fused@1024 | 4.14 | 0.69 | 97.2 | 5.628 |

On the RTX PRO 6000, our 3-block networks take about 1/20 of Cascade's single-shot time and about 1/17 of its
amortized time per shot.

The routed pipelines were timed in C7 (job CJ39), with Cascade alone in the same job and the same rep-major
loop. Each value is the slowest of 3 processes. Single mode decodes 20,000 shots one at a time after 200 untimed
ones. Batched mode decodes 500,000 shots, with S and B at batch 4,096 and Relay-BP in 16 worker processes. "Speed-up" is Cascade's
time divided by the decoder's, from the two values in the same group; below 1 the decoder is slower. "to B, to R" are
the fractions of shots routed in batched mode. Sources: `report/phase_c_tables.md` (section "Speed", second table),
`report/phase_c_summary.json` (`verdicts.<setting>.decoders[].by_p`), `speed/rows.jsonl`. Fifteen batched processes
missed the 97 % clock rule, none of them of S -> B or S -> B -> R (section 4.3).

| setting, p | decoder | single mean (us) | per round (us) | p99.9 (us) | speed-up, single | batched per shot (us) | per round (us) | speed-up, batched | to B, to R |
|---|---|---|---|---|---|---|---|---|---|
| `uniform`, 0.2 % | Cascade alone (C14) | 1,701.0 | 283.49 | 1,718.6 | | 71.891 | 11.98 | | |
| | S -> B | 106.8 | 17.80 | 1,837.9 | 15.9 | 5.325 | 0.89 | 13.5 | 1.00 %, 0 |
| | S -> B -> R | 213.2 | 35.54 | 1,903.3 | 7.98 | 14.058 | 2.34 | 5.11 | 1.00 %, 0.10 % |
| | S -> R | 264.8 | 44.14 | 60,257.8 | 6.42 | 19.420 | 3.24 | 3.70 | 0, 0.25 % |
| | r12 -> R | 290.3 | 48.39 | 77,633.7 | 5.86 | 17.883 | 2.98 | 4.02 | 0, 0.26 % |
| `uniform`, 0.3 % | Cascade alone (C14) | 1,701.0 | 283.49 | 1,718.2 | | 71.905 | 11.98 | | |
| | S -> B | 176.8 | 29.47 | 1,836.7 | 9.62 | 8.254 | 1.38 | 8.71 | 4.97 %, 0 |
| | S -> B -> R | 332.3 | 55.38 | 1,846.0 | 5.12 | 37.821 | 6.30 | 1.90 | 4.97 %, 0.10 % |
| | S -> R | 4,600.7 | 766.79 | 705,510.6 | 0.37 | 263.273 | 43.88 | 0.27 | 0, 2.00 % |
| | r12 -> R | 4,591.2 | 765.21 | 869,281.5 | 0.37 | 264.629 | 44.10 | 0.27 | 0, 2.02 % |
| `depspam_fm`, 0.2 % | Cascade alone (C14') | 1,696.5 | 282.75 | 1,713.5 | | 71.873 | 11.98 | | |
| | S -> B | 105.5 | 17.58 | 1,835.7 | 16.1 | 5.325 | 0.89 | 13.5 | 1.02 %, 0 |
| | S -> B -> R | 254.3 | 42.38 | 1,855.6 | 6.67 | 10.174 | 1.70 | 7.06 | 2.03 %, 0.11 % |
| | S -> R | 512.7 | 85.45 | 68,332.3 | 3.31 | 19.308 | 3.22 | 3.72 | 0, 0.25 % |
| | r12 -> R | 720.9 | 120.15 | 80,001.2 | 2.35 | 20.224 | 3.37 | 3.55 | 0, 0.26 % |
| `depspam_fm`, 0.3 % | Cascade alone (C14') | 1,700.8 | 283.47 | 1,718.4 | | 71.883 | 11.98 | | |
| | S -> B | 178.2 | 29.70 | 1,838.7 | 9.54 | 8.251 | 1.38 | 8.71 | 4.98 %, 0 |
| | S -> B -> R | 882.4 | 147.07 | 129,623.6 | 1.93 | 63.935 | 10.66 | 1.12 | 4.98 %, 0.24 % |
| | S -> R | 5,138.5 | 856.42 | 824,594.7 | 0.33 | 318.522 | 53.09 | 0.23 | 0, 2.00 % |
| | r12 -> R | 5,569.5 | 928.25 | 772,625.6 | 0.31 | 319.853 | 53.31 | 0.22 | 0, 1.96 % |

On the RTX PRO 6000, every routed decoder has a higher 99.9th percentile than Cascade, so each is "faster on average,
slower in the tail" where it is faster at all. S -> B's 99.9th percentile is about 1.84 ms against Cascade's
1.71-1.72 ms: with 1-5 % of the shots routed, more than 0.1 % of them pass through both S and B. With Relay-BP the
99.9th percentile reaches 60-869 ms in S -> R and r12 -> R, and 129.6 ms in `depspam_fm` S -> B -> R at 0.3 %.

In these timings Relay-BP is deployed with one decoder per process, kept across shots. The accuracy reads instead build
a new decoder for every routed shot, seeded by the shot's index; that build took 7.37-7.43 ms on the timing node
(`relay_build_ms` in `check_<setting>.json`) and is in no timing (plan 2.3). The accuracy and the speed of a decoder
with Relay-BP therefore come from two different uses of it, and this file states them separately. In dry run 11 a
decoder kept across shots failed about as often as new seeded ones (499.5 against 508.5 failures summed over 8 unions,
plan 2.3); that comparison does not establish that the two are equally accurate. The report's last column of the
pipeline table multiplies the build time by the fraction of shots routed to Relay-BP. For S -> B -> R in
`depspam_fm` that gives 7.0 us (single) and 7.9 us (batched) per incoming shot at 0.2 %, and 15.2 us and 17.8 us at
0.3 %. These are added CPU work per incoming shot, which the batched pipeline would spread over its 16 asynchronous
Relay-BP workers; they are not measured wall-time overheads. At 0.3 % the 17.8 us is therefore not to be subtracted
from the batched lead of S -> B -> R over Cascade, 7.9 us per shot (63.935 against 71.883 us). No claim is made about
the speed of the seeded implementation; that would need a timing that includes the builds.

The paper's FP8 H200 timings were never digitised (`$R/out/paper_timing.json` does not exist), so the report's paper
column reads "not digitised yet", and no comparison with the paper's speed is made here.

### 4.2 H100 (secondary, speed only)

The H100 results are secondary and speed only, and they depend on the GPU and CPU combination; `depspam_fm`
S -> B -> R at 0.3 % was slower than Cascade there in batched mode (below). No H100 value enters a lock, a check, a
confirmation read or a verdict (`H100_TIMING.md`). Two RunPod pods, each with one NVIDIA H100 80GB HBM3 (700 W, MIG
disabled), ran the same timing code with the same pins as the RTX PRO 6000 jobs:

- The first pod (10 October, driver 570.211.01, CPU AMD EPYC 9554) timed every network alone and the first lock's
  configurations at the claim points (`$H/first_pod/`, from `~/work/cascade_h100t/results/out/`). Its rows stand for the
  networks and for the seven claim-point configurations that Amendment 1 did not change.
- The second pod (11 October, driver 580.126.09, CPU Intel Xeon Platinum 8470) timed the nine claim-point
  configurations that Amendment 1 changed, with Cascade alone beside them in every (setting, p) group of the same job
  (`$H/second_pod/`, from `~/work/cascade_h100b/results/out/`).

Relay-BP runs on the pod's CPU. The rows with Relay-BP therefore come from two CPU models: the EPYC 9554 for S -> R and
r12 -> R in `uniform`, r12 -> R in `depspam_fm` and `depspam_fm` S -> B -> R at 0.3 %, and the Xeon 8470 for the
changed S -> B -> R and S -> R rows. The RTX PRO 6000 nodes run it on an EPYC 9555. Each speed-up below divides by
Cascade alone on the same pod in the same job.

The screen ran as job CJ47 on Alpine (acpu, 16 cores). It compared both pods' outputs on the check's shots (seed
kind 12, chunks 10-19, at 0.2 % and 0.3 %) with the RTX PRO 6000 reads, each against the lock it was timed with. Every row
passed, 58 of 58 in each of the four files (two pods, two settings: `$H/screen/{first,amend1}/screen_<setting>.json`,
`rows`), and no decoder failed. The ConvLines fused paths
on the H100 were bitwise their RTX PRO 6000 reads: no decision and no confidence differed, although the kernels compile
for sm_90 there and sm_120 on the RTX PRO 6000. The Cascade paths passed by the plan's tolerance. The first screen job
(CJ36) failed after 4 s and a resubmission (CJ46) after 9 s, both because of a path in the sbatch file
(`H100_TIMING.md`, "The screen"); CJ47 replaced them. `$H/screen/STATUS` reads "DONE: the screen's results
copied and verified".

Sources: `$H/screen/amend1/h100_speed.json` (second pod) and `$H/screen/first/h100_speed.json` (first pod),
`pipelines`, the slowest of 3 processes; `screen_<setting>.json` beside them for the screen. The rows behind them are
in `$H/second_pod/rows.jsonl` and `$H/first_pod/rows.jsonl`.

| setting, p | decoder (pod) | single mean (us) | p99.9 (us) | speed-up, single | batched per round (us) | speed-up, batched |
|---|---|---|---|---|---|---|
| `uniform`, 0.2 % | Cascade alone (2) | 1,484.2 | 1,779.6 | | 7.282 | |
| | S -> B (2) | 119.6 | 1,712.1 | 12.4 | 1.067 | 6.82 |
| | S -> B -> R (2) | 284.3 | 3,799.8 | 5.22 | 3.458 | 2.11 |
| | Cascade alone (1) | 1,549.8 | 1,634.1 | | 7.211 | |
| | S -> R (1) | 379.9 | 94,495.6 | 4.08 | 4.926 | 1.46 |
| | r12 -> R (1) | 418.0 | 120,958.6 | 3.71 | 4.829 | 1.49 |
| `uniform`, 0.3 % | Cascade alone (2) | 1,484.1 | 1,725.4 | | 7.296 | |
| | S -> B (2) | 183.8 | 1,716.5 | 8.07 | 1.355 | 5.38 |
| | S -> B -> R (2) | 377.0 | 1,765.3 | 3.94 | 9.748 | 0.75 |
| | Cascade alone (1) | 1,548.4 | 1,597.9 | | 7.216 | |
| | S -> R (1) | 7,096.7 | 1,089,826.1 | 0.22 | 66.817 | 0.108 |
| | r12 -> R (1) | 7,276.7 | 1,366,208.1 | 0.21 | 67.364 | 0.107 |
| `depspam_fm`, 0.2 % | Cascade alone (2) | 1,483.3 | 1,609.4 | | 7.293 | |
| | S -> B (2) | 116.7 | 1,703.9 | 12.7 | 1.067 | 6.83 |
| | S -> B -> R (2) | 338.9 | 2,237.0 | 4.38 | 2.827 | 2.58 |
| | S -> R (2) | 765.3 | 107,894.0 | 1.94 | 4.987 | 1.46 |
| | Cascade alone (1) | 1,548.1 | 1,636.5 | | 7.214 | |
| | r12 -> R (1) | 1,084.0 | 124,670.2 | 1.43 | 5.112 | 1.41 |
| `depspam_fm`, 0.3 % | Cascade alone (2) | 1,482.6 | 1,717.9 | | 7.296 | |
| | S -> B (2) | 184.2 | 1,715.9 | 8.05 | 1.354 | 5.39 |
| | S -> R (2) | 7,999.1 | 1,305,661.2 | 0.19 | 81.165 | 0.090 |
| | Cascade alone (1) | 1,549.7 | 1,683.3 | | 7.215 | |
| | S -> B -> R (1) | 1,278.4 | 195,454.1 | 1.21 | 16.384 | 0.44 |
| | r12 -> R (1) | 8,556.7 | 1,183,753.7 | 0.18 | 81.68 | 0.088 |

`H100_TIMING.md` ("Amendment 1", "Rows") gives the second pod's ratios the other way round (the decoder's time over
Cascade's, for example 0.081 and 0.147 for `uniform` S -> B at 0.2 %). The batched Cascade, S -> B and S -> B -> R
rows had at least one process with the SM clock below `gpu_watch`'s 97 % rule; the H100 lowers its clock under these
loads, and the values are what it sustains.

On the H100, Cascade's batched time falls to 7.2-7.3 us per round from 11.98 on the RTX PRO 6000, while our fused
engine is slower there (S55 5.631 against 4.139 us per shot amortized). Every batched speed-up is therefore smaller on
the H100, and both S -> B -> R configurations at 0.3 % fall below 1. `depspam_fm` S -> B -> R at 0.3 %, the winning
decoder's configuration, is 1.21 times faster than Cascade in single mode on the H100 and slower in batched mode
(16.384 against 7.215 us per round). That row is from the first pod, with Relay-BP on the EPYC 9554.

On the H100 the 99.9th percentile of S -> B is below Cascade's in three of the four groups (1,712.1 against 1,779.6 us
in `uniform` at 0.2 %, 1,716.5 against 1,725.4 at 0.3 %, and 1,715.9 against 1,717.9 in `depspam_fm` at 0.3 %). The
statement that every routed decoder has the slower tail therefore holds on the RTX PRO 6000 only.

### 4.3 The SM clock flags

Every timing row records the lowest SM clock that `gpu_watch` sampled, as a fraction of the card's maximum, and whether
it met the 97 % rule (`clock` in `speed/rows.jsonl`: `clock_min_ratio`, `accepted`). Each reported value is the
slowest process whether or not it met the rule (C3's `speed_components.json` for the networks, `pc_report.py` for the
pipelines), and no check tests the rule. A row is flagged below when `accepted` is false, and row numbers are line
numbers of `speed/rows.jsonl`. Every flagged row cited here is at 96.30-96.58 %. `other_pids` is empty in every row,
so no other process was on the card.

C3 (job CJ30), on the frozen paths of the networks alone:

- All three processes on the single-shot path (batch 1) of each Cascade network were flagged: C14 in rows 14, 218 and
  422 and C14' in rows 74, 278 and 482, all at 96.30 %, and also C6 (rows 30, 234, 438) and C6' (rows 202, 406, 610).
  Their single-shot values in section 4.1 come from flagged processes only.
- S55, S55' and S55'-deep had one flagged process of three on that path (rows 50, 94 and 122, at 96.58 %). It supplies
  the reported value of S55' and of S55'-deep.
- No process on an amortized frozen path was flagged, and no process of r11 or r12 on a frozen path.

No verdict of a network alone depends on these values: every network alone is worse than Cascade in accuracy, and
each takes at most 1/13 of Cascade's single-shot time. The same values are also the frozen components from which the
lock predicted latencies for its speed-feasibility rule (plan 4.3, where Cascade's L1 is t1(C14) or t1(C14')).

C7 (job CJ39): 15 of the 120 rows were flagged, all in batched mode.

- Three are Cascade alone's: rows 654 and 694 (`uniform`, 0.2 %, repetitions 2 and 3, at 96.58 % and 96.30 %) and row
  714 (`depspam_fm`, 0.2 %, repetition 3, at 96.58 %). None of them supplies a reported baseline, since the slowest
  Cascade process in both groups is the accepted first repetition (rows 614 and 634: 71.891 and 71.873 us per shot).
  Without the flagged rows every Cascade baseline stays as reported and the speed ordering is unchanged, but Cascade's
  batched timing at 0.2 % then rests on one accepted process in `uniform` and two in `depspam_fm`, fewer than three.
- No S -> B or S -> B -> R process was flagged. Every value behind the speed verdict of `depspam_fm` S -> B -> R, its
  own and the reported Cascade baselines, comes from an accepted process.
- The other 12 are batched processes of S -> R and r12 -> R: rows 632, 650, 652, 670, 672, 690, 692, 710, 712, 730 and
  732 at 0.3 % in both settings, and row 682 (`depspam_fm` r12 -> R at 0.2 %). They supply the reported batched values
  of these decoders in those groups. At 0.3 %, `uniform` S -> R keeps one accepted process and the other three
  decoders keep none. S -> R and r12 -> R are worse than Cascade at both claim points, and where accepted processes
  remain they stay on the same side of Cascade (`uniform` S -> R at 0.3 %: 260.414 us per shot; `depspam_fm` r12 -> R
  at 0.2 %: 18.083 and 19.271 us).

These values are disclosed as the performance the card sustained and were not rerun. Strict compliance with the clock
rule would need only the affected matched timings rerun. The H100's flagged rows are described in section 4.2.

## 5. The four questions

1. Alone, our already trained networks r11 and r12 are worse than Cascade at every p in both settings: in
   `depspam_fm` at 0.2 % r12 fails 1,420 times in 1,000,000 shots against C14''s 701 (P_L 1.97e-05 against
   9.74e-06), and at 0.3 % 5,697 times against 2,305 in 500,000. They take 88.0-88.1 us per single shot (14.67-14.68
   us per round) and 4.14 us per shot amortized on the RTX PRO 6000, against C14''s 1,781.9 us and 71.63 us. Routing
   to Relay-BP (r12 -> R) cuts r12's failures (5,359 against r12 alone's 7,151 in 5,000,000 shots at 0.2 % in
   `depspam_fm`), but r12 -> R stays worse than Cascade at both claim points in both settings, and at 0.3 % it is
   slower than Cascade (5,569.5 against 1,700.8 us single, 319.853 against 71.883 us per shot batched, `depspam_fm`).
2. Our network retrained in Cascade's setting, S55' (3 blocks) or S55'-deep (6 blocks), fails 1,322 or 943 times in
   1,000,000 shots at 0.2 % and 4,931 or 3,374 times in 500,000 at 0.3 %, against C14''s 701 and 2,305, so both are
   worse than Cascade (S55'-deep is level only at 0.1 %). S55' takes 88.0 us per single shot and 4.14 us per shot
   amortized, S55'-deep 130.9 us and 6.26 us, against C14''s 1,781.9 us and 71.63 us on the same card. In `uniform`,
   S55 fails 799 times against C14's 372 at 0.2 % and 2,954 against 1,264 at 0.3 %.
3. The reproduced large, deep network C14' (L 14, H 512, trained in `depspam_fm`) has P_L 7.79e-07 [7.17e-07,
   8.46e-07], 9.74e-06 [9.04e-06, 1.05e-05] and 6.42e-05 [6.16e-05, 6.69e-05] at 0.1, 0.2 and 0.3 %, and the
   paper's digitised Cascade values (8.17e-07, 1.04e-05, 6.58e-05) lie inside these intervals; at 0.4 % C14' fails
   more often than the paper's network (2.69e-04 [2.62e-04, 2.76e-04] against 2.19e-04). All of this rests on a
   fitted setting and on digitised values (`report/phase_c_summary.json`, `curves.depspam_fm.points`). In BF16 on the
   RTX PRO 6000, C14' takes 1,781.9 us per single shot (296.98 us per round) and 71.63 us per shot amortized (11.94 us
   per round); the H100 gives 1,664.9 us and 42.985 us, and the paper's FP8 timings were not digitised.
4. Routing our small network's least confident shots to the reproduced Cascade network, S -> B (S55' -> C14' in
   `depspam_fm`) was faster on average at both claim points, with excess block failure rate bounded by the frozen
   margin (10 % of Cascade's development-block rate; measured +7.6e-06 and +9.1e-05). On the RTX PRO 6000 it is 16.1
   times faster than Cascade per single shot and 13.5 times faster batched at 0.2 %, and 9.54 and 8.71 times at
   0.3 %. Its verdicts are level at 0.2 % and worse at 0.3 %, where it is significantly less accurate than Cascade.
   Sending the shots C14' is least sure of on to Relay-BP (S -> B -> R) reduced confirmed block failures at both
   claim points (3,596 against 3,858 failures in 5,000,000 shots at 0.2 %, 3,957 against 4,459 in 1,000,000 at
   0.3 %). Separately, its retained-decoder deployment was 6.67 and 1.93 times faster than Cascade in single mode and
   7.06 and 1.12 times batched on the RTX PRO 6000 with the node's EPYC 9555, with a slower tail. The accuracy reads
   used a new seeded decoder per routed shot, whose speed is not claimed (section 4.1). In `uniform` the same
   combination is better at 0.3 % and worse at 0.2 %.

## 6. Amendment 1

The lock that the confirmation read and the verdicts rest on is not the first lock. Amendment 1 changed the selection
rule of plan 4.3 (step 3) on 11 October 2026 (commit 08542cd8), and the new lock is commit 629ca1c4. The first lock,
commit 1e8d1c6d of 10 October, is superseded and stays in the branch's history. Its claim-point check (job CJ34,
all 48 rows passed in each setting) is kept as its record in `out/lock_1e8d1c6d/` and enters no verdict.

The registered rule took the speed-feasible candidate with the fewest development failures and sent only exact ties
to the lower predicted latency. The development curves are flat above a small routed fraction, so it picked large
routed fractions for one or two fewer failures. In `uniform` at 0.3 % the first lock took S55 -> C14 at
q1 = 40 % (1,263 failures on 500,000 development shots, predicted single-shot latency L1 802 us), where q1 = 5 % had
1,276 failures at 177 us and Cascade alone 1,264 at 1,783 us. At 01:20 PDT on 11 October the authors chose the rule
called version A: take the candidate with the largest overall gain, without putting accuracy or speed first. Among
the speed-feasible candidates (all of them if none is feasible), F is the fewest development failures; the candidates
within F + sqrt(F) are tied, and the rule takes the tied one with the lowest predicted L1 (then fewer failures). It
applies to S -> B at every p and to S -> B -> R at 0.2 % and 0.3 % in both settings. The gate review on the amendment
(PROCEED WITH CHANGES) preferred a paired band for S -> B and the registered accuracy-first rule for S -> B -> R. The
authors chose version A for both.

The note at the end of plan 4.3 states the disclosure:

> This amendment is outcome-informed. It was made before any confirmation shot (seed kind 11) was read, but after the
> development curves had been seen and, on another branch (`ising-cascade-h100t`), the H100 failure counts of the
> first lock's configurations on kind 12, chunks 10-19, the shots of the claim-point check (2.4). The gate review on it
> (PROCEED WITH CHANGES) preferred a paired band for S -> B and the accuracy-first rule for S -> B -> R. The authors
> chose this version for both, for the largest overall gain, without putting accuracy or speed first. It is frozen
> before the new claim-point check, and nothing is tuned afterwards to obtain a pass.

Everything else stayed as registered: the candidates, feasibility, the headline family of 132 comparisons and its one
Holm procedure, the sizing of the confirmation block and its caps, the non-inferiority margins, the check's
tolerance, the timing protocol and the verdict rules of section 6. The picks changed in 21 of the 32 configurations
(`RUN.md`, "The new lock"). Two consequences bear on the headlines:

- The winning decoder's 0.2 % configuration is an amended pick. Its first-lock pick was S55'-deep, q1 5 %, r 0.1 %,
  with 657 development failures against Cascade's 701; the new pick is S55', q1 2 %, r 0.1 %, with 670. Its 0.3 %
  configuration (S55', q1 5 %, r 0.25 %) is the same in both locks.
- Under the amended rule S -> B routes 1 % (0.2 %) and 5 % (0.3 %) of the shots at the claim points, against 2-40 %
  under the first lock. The confirmation finds this S -> B worse than Cascade at 0.3 % in both settings. The first
  lock's picks were never read on the confirmation block, so how they would have fared is unknown.

Beyond its registered verdicts, S -> B's accuracy is stated in this file only as the plan's secondary statement: the
excess block failure rate is bounded by the frozen margin, with the measured differences of section 2. Making
non-inferiority a headline would be another amendment.

## 7. What did not work

- S -> B -> R in `uniform` fails 435 times in 1,000,000 shots at 0.2 % against Cascade's 360 (+7.50e-05 [4.0e-05,
  1.1e-04]). It is worse there, and that veto holds although it is better at 0.3 %. Its planned power was 0.054.
  Every S -> B -> R candidate on the development block had at least 454 failures against Cascade's 372
  (the gate review on the amendment, finding 4), so no candidate under either rule was expected to be better there.
- S -> R and r12 -> R, the known routing to Relay-BP at the registered operating points (q 0.25 % at 0.2 %, 2 % at
  0.3 %), are worse than Cascade at both claim points in both settings. At 0.3 % they are also slower: 2.7-3.3 times
  Cascade's single-shot time and 3.7-4.4 times its batched time. Their 99.9th percentiles run from 60 ms to 869 ms.
- S -> B is 8.7-9.6 times faster than Cascade at 0.3 % and worse there in both settings. It is also worse at 0.45 %
  in `uniform` and at 0.25 % and 0.4 % in `depspam_fm`. Its extra failures are S's own on the shots it keeps.
- None of our small networks alone is better than Cascade at any p. All are worse at every p, except S55'-deep,
  which is level at 0.1 % in `depspam_fm`.
- On the RTX PRO 6000, every routed decoder has a higher 99.9th percentile single-shot latency than Cascade's.
  `depspam_fm` S -> B -> R reaches 129.6 ms at 0.3 % against Cascade's 1.72 ms. On the H100 the 99.9th percentile of
  S -> B is below Cascade's in three of the four groups (section 4.2).
- On the H100, `depspam_fm` S -> B -> R at 0.3 % and `uniform` S -> B -> R at 0.3 % are slower than Cascade in
  batched mode (speed-ups 0.44 and 0.75). On the RTX PRO 6000 both are faster.
- The paper's FP8 H200 timings were not digitised, so phase C makes no comparison with the paper's speed.

## 8. Files committed with this report

On `ising-cascade-phasec`, under `cascade_compare/phase_c/out/`. Every file except `npz_sha256.txt` is a byte-for-byte
copy from `$R/out`, and the sha256 of each copy was compared with Alpine's before the commit. `npz_sha256.txt` is the
output of `sha256sum`, run on Alpine over the 68 `p*.npz` reads (paths relative to `$R/out`).

| file | sha256 |
|---|---|
| `check_uniform.json` (commit 1637db9c) | a4f38b65215c6b78fe1d00268321000c3d7c59a404ceabafb331a69ac869932f |
| `check_depspam_fm.json` (commit 1637db9c) | 95a23bdcaeb7546fefa7fa00073d517670952c8022f2c2d768966db26e9737ba |
| `confirm_uniform.json` | fbbbf05457f82309065e99e32f9b15603d87ca8cce4dc5be5cc637e5e9cc3956 |
| `confirm_depspam_fm.json` | e010600b3fcd7c2c45462a4fdaa3bcd7e070d3fcc8aa31d6537b9fbbb5947e3e |
| `report/phase_c_tables.md` | b0357b9ab81ff217fda5e6d6ae2cc1283388c0eeaa6393d098ecc715f5737e9c |
| `report/phase_c_summary.json` | bc3aef65147c28a98e0808d75346617c178acde71b99e5d92187dcfa529be085 |
| `speed/rows.jsonl` | a2499bddb4d6e17180770cd5e6389c73ea48d6a2e720df5901a62966d26b182e |
| `npz_sha256.txt` | 6fc7f2eec6f88b1ef7789bc5f8ae637959b37f92b16120b6d1735dcc3a009db5 |

Also committed: the `rows_*.json` of every read (`dev/`, `dev_late/`, `check/<setting>/{read,eager}/`, `conf/`), the
`relay_*.json` of `relay_dev/` and `relay_conf/`, and `speed/clocks_CJ30.txt`. The confirmation analyses and the
checks give `locked_sha256` 1748c2d4d3b94fbc2df427df4f9bd6a1026c5ba2163266f5e57691f9704da6b1 (`uniform`) and
398d02c60e08535a16a1d37a325e3aec9537f8a3608c794c985f471c0be1c41d (`depspam_fm`), the new lock. The checks give
`speed_sha256` 1ec6b2563060f0042f866f19841631e8e1499872e6c15ed77dae7a01c8c811f5, the sha256 of the committed
`speed_components.json`.
The `p*.npz` reads (about 14 GB) stay on Alpine.

The H100 files used in sections 4.1 and 4.2 are committed on `ising-cascade-h100t2` (commit d1575924) under `$H`
(`cascade_compare/phase_c/out_h100/`): the text outputs of both pods (`first_pod/`, `second_pod/`) and the screen of
job CJ47 (`screen/`). `H100_TIMING.md` ("The committed outputs") lists them and names the file behind each H100
number. Their binary files (the pods' `h100_p*.npz` and the profile's binaries) stay on the Linux workstation and on
Alpine (`<cluster-scratch>/cascade_h100t/out`, `<cluster-scratch>/cascade_h100b/out`), and each pod's committed
`SHA256SUMS` lists their sha256. In `$H/screen/`, `SHA256SUMS` verifies all 8 files. The first 16 hex digits of their
sha256: `amend1/h100_speed.json` fd3d88a36929578d, `amend1/screen_uniform.json` 9ba9d1e2689ffaa0,
`amend1/screen_depspam_fm.json` d5e75a4407d83050, `amend1/summary.txt` 7c71a25859d16c31, `first/h100_speed.json`
42b9d26b86bc3ad4, `first/screen_uniform.json` d9c645c0d474fc7a, `first/screen_depspam_fm.json` b5674eebf6a81ba6,
`first/summary.txt` 2a03627361884884. The pods' rows: `first_pod/rows.jsonl` 50c7310c4e61d743,
`second_pod/rows.jsonl` d0886cfd3c4a75fa (the `rows_sha256` recorded in the screened `h100_speed.json` files).
