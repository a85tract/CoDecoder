# Phase D1: S55'-deep routed to Cascade against Cascade alone, on fresh shots (plan)

11 October 2026, branch `ising-cascade-d1` from 90307cad (`ising-cascade-explore`). Revised on 11 October 2026, after
the gate review of the draft (5f9249ff), which returned PROCEED WITH CHANGES. The study lead's six dispositions are folded in
(section 0).
The freeze commit (section 5.3) comes after the dry runs and D1-P (5.6), and no shot of D1's seed kind is drawn before
it. This file is frozen with the commit that adds this sentence; the reads cite that commit as F.
"Registered" here means that the hypotheses, the configurations, the block sizes and the analysis are fixed in this
file and frozen before any shot is drawn. As in every stage of `cascade_compare`, no registered test set of
`ising_compare` is read.

## 0. The gate and the dispositions

The draft (5f9249ff) had two comparisons, both on phase C's accuracy path for Cascade. The gate's findings and the study
lead's dispositions, and where this revision carries them out:

1. Finding 1, accuracy on the timed computation. Cascade is read on both of its paths on the same fresh shots: phase C's
   accuracy path (eager BF16, where the lead was found), and the compiled paths that the timings use, single
   (compiled@1) and batched (compiled@4096), which differ. The family is four comparisons (2 p x 2 Cascade paths) under
   one Holm procedure at 0.05 (1.2). A claim of "more accurate and faster" needs "better" on the compiled path, and the
   eager verdicts are reported beside it (4.4). The powers are recomputed for four tests, and the block sizes stay unless
   the compiled comparison's power under the design scenario falls below 0.8 (2.4).
2. Finding 2. The seed namespace's freshness is described as audited provenance, and the categorical claim is deleted
   (2.3).
3. Finding 4. The powers are labelled design-scenario powers, and the pre-freeze CPU step D1-P computes the exact
   simulated power with the interval condition (2.4, 5.3).
4. Finding 5. After any result, no configuration is adapted, no shot is added and no other analysis is substituted.
   Disclosed corrections and reproducibility reruns with unchanged data and rules are allowed, and every earlier output
   is kept (3.1).
5. Finding 6. Both engines are timed as planned (4.1). Before the freeze, the dry runs check that the copied first-lock
   configurations, the thresholds and the new wrappers work with the files they need: the original lock JSON, `m2.py`,
   `pc_report.py` and `bb.route` (5.3).
6. Finding 3 stands. The results disclose the post hoc origin (52 looks), the superseded-lock history, the fitted-noise
   and re-implementation scope, the effect sizes, and that phase C's verdicts do not change (5.4).

## 1. The question

### 1.1 Where it comes from

Phase C (`ising-cascade-phasec`, 1f2280b3; `PHASE_C_PLAN.md` with Amendment 1, `phase_c/RESULTS.md`) routed the shots
that a small network S is least sure of to Cascade (S -> B). Under Amendment 1's lock (629ca1c4) the rule picked the
3-block S55' at both claim points in `depspam_fm`. There S -> B was level with Cascade at 0.2 % (3,896 against 3,858
failures in 5,000,000 shots) and significantly worse at 0.3 % (4,550 against 4,459 in 1,000,000). The verdicts in
`uniform` were the same.

An exploratory analysis of the same confirmation reads (`phase_c/EXPLORE_DEEP.md` on `ising-cascade-explore`, 90307cad)
then evaluated 52 configurations, after the registered results were known and without adjusting for multiplicity. Two
of them are the claim-point picks of phase C's first lock (1e8d1c6d), which Amendment 1 replaced before any
confirmation read. Both route the 6-block S55'-deep to C14':

| p | first lock's pick | confirmation: pair / Cascade (shots) | breaks / repairs | difference [95 %] | McNemar p |
|---|---|---|---|---|---|
| 0.2 % | S55'-deep -> C14', q1 = 2 % | 3,822 / 3,858 (5,000,000) | 7 / 43 | -7.2e-06 [-1.0e-05, -4.4e-06] | 2.1e-07 |
| 0.3 % | S55'-deep -> C14', q1 = 10 % | 4,452 / 4,459 (1,000,000) | 4 / 11 | -7.0e-06 [-1.6e-05, 1.4e-06] | 0.12 |

A break is a shot that S keeps and gets wrong while Cascade gets it right; a repair is the reverse. The difference is
the pair's block failure rate minus Cascade's, with Newcombe's interval. At 0.2 % the p-value is below 0.05 / 52 =
9.6e-04, and EXPLORE_DEEP calls the result a lead for a registered test on new shots. At 0.3 % the pair is level. The
authors expect (11 October) that a better configuration can beat Cascade on accuracy at 0.3 %. D1 checks the lead on new
shots with today's networks.

The pair takes Cascade's decision on every routed shot, so it can differ from Cascade only on the shots S keeps, and its
gain is capped by Cascade's failures there. With 2 % routed at 0.2 %, 181 of Cascade's 3,858 confirmation failures
(4.7 %) fell on kept shots. With 10 % routed at 0.3 %, 138 of 4,459 did (3.1 %), and the other 97 % fell on routed
shots, so even a pair that never erred on a kept shot could beat Cascade by no more than these fractions.

### 1.2 The hypotheses and the family

The setting is `depspam_fm`, the fitted noise variant of the noise-model fit. B is C14' (Cascade, L 14, H 512, 19.7M
parameters, read in BF16), and S is S55'-deep (ConvLines-BB, 6 blocks, 841k parameters, on its fused engine). The pair D
decides as S on
every shot whose S margin is at least tau1 and as C14' on every other shot.

C14' is read on two paths, and on each path D's routed shots take C14''s decision on that path:

- eager: phase C's accuracy path, `bb.cascade_eval`'s evaluation backend under BF16 autocast, run eagerly in batches of
  8,192 within each 10,000-shot chunk (8,192 and 1,808 shots). The lead of EXPLORE_DEEP was found on this path.
- compiled: the frozen timed paths of phase C's speed components, which are the computations the timings run. In single
  mode this is compiled@1 (torch.compile of the same BF16 computation, default mode, captured in a CUDA graph at batch 1),
  for C14' alone and for D's routed shots. In batched mode it is compiled@4096. C14' alone decides consecutive batches of
  4,096 shots of the block, the last padded with zeros, as Cascade alone runs in the batched pipeline. D's routed shots
  are decided in shot order in batches of 4,096, the last padded, as the pipeline queues them.

The paths differ by rounding. In phase C's claim-point check under the first lock (100,000 shots of kind 12 per point,
`phase_c/out/lock_1e8d1c6d/check_depspam_fm.json`) the compiled paths decided 8 (single) and 7 (batched) of C14''s
shots differently from its eager read at 0.2 %, and 39 and 37 at 0.3 %. At 0.2 % C14' failed 74 times on the eager
path, 70 on compiled@1 and 72 on compiled@4096, and D (configuration 2) failed 73, 70 and 72 times, so the one-shot gain
on the eager path was gone on both compiled paths. S55'-deep decides bitwise alike on every path (4.2).

The hypotheses are, each against the null of equal block failure rates:

- H1e, p = 0.2 %, eager: D with q1 = 2 % has a lower block failure rate than C14' alone, both on the eager path.
- H2e, p = 0.3 %, eager: D with q1 = 10 % has a lower block failure rate than C14' alone, both on the eager path.
- H1c, p = 0.2 %, compiled: D with q1 = 2 % has a lower block failure rate than C14' alone on the compiled path, in
  single mode (compiled@1) and in batched mode (compiled@4096).
- H2c, p = 0.3 %, compiled: the same with q1 = 10 %.

Each is a paired comparison on the same shots. A compiled comparison holds two paired comparisons, single and batched.
If they give the same failures for D and for C14' on every shot of the block, they are one comparison. Otherwise the
compiled comparison is an intersection-union test. Its p-value is the larger of the two exact McNemar p-values. It is
"better" only if both Newcombe intervals at its Holm level lie below zero, "worse" only if both lie above zero, and
"level" otherwise. Rejecting it means rejecting both nulls at the same level, so it keeps that level without a further
correction.

The family is these four comparisons, judged by one Holm procedure at alpha = 0.05 with phase C's exact two-sided
McNemar test and Newcombe's interval (section 3). The tests are two-sided as in phase C, so "worse" is also a possible
verdict. A claim that D is more accurate than Cascade at a point rests on the compiled comparison alone. The eager
comparison's verdict is reported beside it.

### 1.3 Why `uniform` does not join

Phase C has no deeper small network in `uniform`: its one 6-block ConvLines network, S55'-deep, was trained in
`depspam_fm`, and the hypothesis is about that network. The 3-block pairs in `uniform` show nothing to confirm at 0.3 %.
The first lock's pick there (S55 -> C14, 40 % routed) failed 3,715 times against Cascade's 3,716, with no break and one
repair, and only 23 of Cascade's failures (0.6 %) fell on kept shots. At 0.2 % the closest S55 pair was 5 failures below
Cascade (p = 0.12). Adding `uniform` would double the family and lower the power of the comparisons that the lead
concerns. A deeper network for `uniform` would have to be trained first.

### 1.4 The configurations, fixed before any read

Both are copied unchanged from configurations 2 and 7 of the first lock's `cascade_compare/phase_c/out/locked_depspam_fm.json`
(commit 1e8d1c6d, sha256 02b19c451c6a38f6854a6be665b9bdcc87f8feb53d86841309833a70f9be0e81; on Alpine the same file is
`cascade_w4/out/locked_depspam_fm.json`):

| | p | pair | q1 | tau1 | development block: D / Cascade (only D / only Cascade), shots | predicted L1 / A (us) |
|---|---|---|---|---|---|---|
| H1 | 0.2 % | S55'-deep -> C14' | 2 % | 7.3331888389587405 | 696 / 701 (1 / 6), 1,000,000 | 166.5 / 7.69 |
| H2 | 0.3 % | S55'-deep -> C14' | 10 % | 7.60439167022705 | 2,297 / 2,305 (1 / 9), 500,000 | 309.1 / 13.42 |

tau1 is the q1-quantile of S55'-deep's fused margins on phase C's development block (seed kind 10), computed by
`pc_route.py select` on 10 October. A shot goes to C14' when its margin is strictly below tau1. Nothing is refitted. L1
and A are the lock's predicted single-shot and amortized latencies; Cascade's are 1,781.9 and 71.63 us. The development
counts were read on the eager path; the configurations and thresholds are the same for both paths.

These are the only S55'-deep configurations at the claim points that a rule chose before any confirmation shot was read.
The rule was phase C's registered step 3 before Amendment 1: among the speed-feasible candidates (predicted L1 and A both
at most 0.9 times Cascade's), take the one with the fewest development failures, with exact ties going to the lower L1.
At 0.2 % it took q1 = 2 % (696 failures, tied with 5 %, which has the higher L1), and at 0.3 % it took 10 % (2,297). The
post hoc look only decided that these two configurations deserve a test; their q1 and tau1 are the first lock's. The
other rows of the exploratory grid were never locked, and picking one of them now would select on the confirmation block
(EXPLORE_DEEP, section 7).

The authors prefer the largest overall gain to putting accuracy or speed first; both picks are already predicted 5.3 to
10.7 times faster than Cascade in both modes, so the overall gain now turns on accuracy, and the first lock's rule (the
speed-feasible candidate with the fewest development failures) is both the overall-gain rule and the stated accuracy
rule that a superiority test needs. Amendment 1's band, applied to the S55'-deep candidates alone, would take q1 = 1 % at
0.2 % and 5 % at 0.3 %. By the lock's predictions these are about 1.1 and 1.4 times faster than the picks, with 5 and 1
more development failures; D1 does not test them.

### 1.5 The lock

`cascade_compare/phase_d/d1_locked_depspam_fm.json` holds the two configurations as copied, the first lock's path and
sha256, the networks' sha256, read precisions and frozen paths, phase C's card, driver and torch, the sha256 of phase C's
speed components, the two Cascade paths and the four comparisons, the compiled read's parameters (2.6), the descriptive
grid (3.4), the seed kind, the block sizes, the alternatives of the sizing, alpha and the power rule. It has no
`conf_chunks` field, so phase C's default confirmation groups, which start at chunk 0, cannot be applied to it by
mistake. D1-P writes the simulated powers into it, and the dry runs fix the compiled read's stream count; after the
freeze nothing in it changes.

## 2. Fresh shots

### 2.1 The seed kind

D1 reads the shots of `bb.cascade_data.eval_seed(14, p, j, "depspam_fm")`, that is 2^55 + 5·10^15 + 14·10^13 +
10^7·round(10^6 p) + j for chunk j of 10,000 shots (j < 10^7). Kind 14 is a range of 10^13 seeds in the training stage's
evaluation namespace that no other kind shares, disjoint from the training seeds ([2^55 + 4·10^15, 2^55 + 5·10^15) in
`depspam_fm`) and from the `uniform` family. It is reserved for D1's two reads; D1's dry runs use phase C's dry kind 13,
which is never reported.

### 2.2 The audit

The audit for this plan, made on 11 October (commands in the appendix):

1. Code. `bb.cascade_data` (64b2e25d) documents kinds 0-7 for the training stage, and `phase_c/pc_common.py` reserves
   10-13. At the heads of all 35 hub branches, no script or job file passes `eval_seed`, `cascade_eval`, `pc_eval.py` or
   `eval_shots` a kind outside 0-7 and 10-13.
2. History. No commit on any branch has written a seed kind from 14 to 19 into a `.py`, `.sh`, `.sbatch` or `.md` file
   (`git log --all -G`). The search's three hits are study 4's "d = 17".
3. Committed outputs. Every `kind` and `kind_seed` field in the JSON committed on any branch is 0, 2, 3, 5, 6, 7, 10,
   11, 12 or 13.
4. Alpine. Every read of `bb.cascade_eval` writes its kind into its `p*.json`. A read-only search of all the JSON under
   `cascade_w1` to `cascade_w5`, `cascade_batchopt`, `cascade_explore`, `cascade_h100b` and `cascade_h100t` finds the
   same kinds and no 14.

Before it draws a shot, the read job repeats the Alpine search, adding `cascade_d1` outside D1's read directories. It
stops if any record there names kind 14, or if its p group is already marked done.

### 2.3 The chunks, the freeze and provenance

The chunks depend on the freeze commit. With F the full sha of the freeze commit, each read uses chunks c0 to
c0 + n - 1, where c0 = int(F[:7], 16) mod 9,000,000, so that c0 + n stays below eval_seed's limit of 10^7 chunks. The
read job takes F as a parameter and runs only if the deployed tree is F (its `DEPLOYED_SHA` file names F, and its
`DEPLOYED` file records no uncommitted change). It computes c0 itself and records F, c0 and every chunk's seed (in
`pc_eval.py`'s `chunk_meta`). The analysis recomputes c0 from F and checks every seed.

The offset does not prove that the shots are fresh. A commit's hash can be computed before the commit is published, so
the hash alone cannot show that no shot was drawn before the freeze. D1's freshness is audited provenance: the
namespace was reserved for D1 (2.1); the audit found no use of it (2.2), and the read job repeats the Alpine search
before it draws; the freeze commit is pushed to the hub before any read is submitted; and a failed read is retried
unchanged (5.5).

### 2.4 Block sizes and design-scenario powers

With four comparisons, Holm judges the smallest p-value at 0.05 / 4 = 0.0125, the next at 0.05 / 3, then 0.025 and
0.05, each step only if every earlier one rejected. The draft sized each block for power 0.9 at 0.025 (two-sided), the
lowest level of its two-test procedure. Disposition 1 keeps those sizes, 1,100 chunks at 0.2 % and 1,700 at 0.3 %,
unless the compiled comparison's power under the design scenario and the four-test procedure falls below 0.8 at either
point.

The design scenario starts from the breaks and repairs per shot of the same two configurations on phase C's
confirmation block (EXPLORE_DEEP, section 4.2):

| p | observed breaks, repairs per shot | discordant per shot | observed gain (share of Cascade's rate) | design alternative: breaks, repairs per shot | design gain |
|---|---|---|---|---|---|
| 0.2 % | 1.4e-06, 8.6e-06 (7 and 43 in 5,000,000) | 1.0e-05 | 7.2e-06 (0.93 %) | 3.2e-06, 6.8e-06 | 3.6e-06 (0.47 %) |
| 0.3 % | 4.0e-06, 1.1e-05 (4 and 11 in 1,000,000) | 1.5e-05 | 7.0e-06 (0.16 %) | 5.75e-06, 9.25e-06 | 3.5e-06 (0.08 %) |

The design alternative keeps the observed discordant rate and halves the net gain. The lead came from a post hoc look,
which tends to overstate an effect, and at 0.3 % it rests on 15 discordant shots.

D1-P (job CJ68, `phase_d/dry/power_CJ68.json`) computed the design-scenario powers by `bb.route.power`'s
multinomial simulation (fixed seed 20261006, 20,000 repetitions) from the lock's cells. The decision is the whole one:
the exact McNemar p below the level and Newcombe's interval at that level below zero.

| p | chunks (shots) | expected discordant shots | one test at 0.0125 | at 0.05 / 3 | at 0.025 | at 0.05 | four-test procedure: eager / compiled comparison |
|---|---|---|---|---|---|---|---|
| 0.2 % | 1,100 (11,000,000) | 110 | 0.895 | 0.911 | 0.934 | 0.960 | 0.929 / 0.929 |
| 0.3 % | 1,700 (17,000,000) | 255 | 0.882 | 0.901 | 0.924 | 0.957 | 0.918 / 0.918 |

The four-test column uses the joint model: on both paths C14' decides alike on all but a few shots per 100,000 (1.2),
so the model gives each compiled comparison the cells of the eager comparison at its point, and the two points are
independent (different shots). The pair at the point with the smaller p-value must then pass at 0.0125 (its twin passes
at 0.05 / 3) and the other pair at 0.025, and the intersection-union step changes nothing. All four comparisons are
"better" together with probability 0.860. As a bracket, drawing the compiled comparisons' cells independently of the
eager ones gives 0.954 and 0.949 for the eager comparisons, 0.958 and 0.950 for the compiled ones, and 0.841 for all
four. Under the observed rates every power is 1.000 to three decimals. In no repetition, at any level, did McNemar
reject while the interval at that level failed to lie below zero. The draft's hand method (power about
Phi((N (r - b) - z(alpha / 2) sqrt(N (r + b)) - 1.5) / sqrt(N (r + b)))) gave 0.87, 0.89, 0.92 and 0.95 at the four
levels and about 0.91 for the four-test procedure, on the low side as intended. The gate review computed about 0.933 and
0.926 at 0.025, where D1-P has 0.934 and 0.924.

These are design-scenario powers. They hold if half of the observed lead is real and the discordant rates are as
observed. They do not cover the uncertainty of those rates, which at 0.3 % rest on 15 discordant shots, nor a loss of
repairs on the compiled path like the one in phase C's check (1.2), which the test itself measures.

The rule: a block grows only if the compiled comparison's power at its point falls below 0.8 under the design scenario
and the four-test procedure. It is 0.929 at 0.2 % and 0.918 at 0.3 %, so the sizes stay at 1,100 and 1,700 chunks.
D1-P's numbers are in the lock (`planned_power`).

Phase C's confirmation block at 0.3 % (1,000,000 shots) had a power of only about 0.2-0.3 against the observed rates at
0.025.

### 2.5 Cost

In GPU minutes on a full RTX PRO 6000:

| part | basis | GPU minutes |
|---|---|---|
| eager read at 0.2 % (11,000,000 shots) | C14' at 155 us per shot and S55'-deep at 6.3 us, as phase C's C5 read them (775.8 s and 31.3 s for 5,000,000 shots at 0.2 %): about 161.5 us per shot together | 30, plus about 2 for start-up |
| eager read at 0.3 % (17,000,000 shots) | the same | 46, plus about 2 |
| compiled reads, batched | C14' compiled@4096 at 71.6 us per shot on all 28,000,000 shots, and on D's routed shots (about 1,920,000) | about 36 |
| compiled reads, single | C14' compiled@1: 1.77 ms of GPU time per shot one at a time (13.8 GPU-hours for 28,000,000 shots); as 32 concurrent graphs (2.6), 179 us per shot in the dry profile | about 33 at 0.2 % and 51 at 0.3 % |
| speed job (36 processes, 4.1) | phase C's C7 and milestone 2's pipeline jobs, and the dry run's processes | about 40 |
| dry runs (5.3) | | about 30 |

Each read job runs the eager read at its p and then the compiled reads of the same shots, on one card: about 80 minutes
at 0.2 % and 125 at 0.3 %. The two read jobs and the speed job run on three cards at most at once. The analysis and D1-P
take a few CPU minutes. The per-shot files take about 4 GB under `<cluster-scratch>/cascade_d1/`.

### 2.6 The compiled reads

`d1_compiled.py` reads C14' on its compiled paths over the detector rows that the eager read saved (`--save-det`). It
first checks every chunk's digest against the eager read's `chunk_meta`.

- Batched: `pc_speed.build(C14', "compiled", 4096)`, phase C's own path object, decides every row in order, in
  consecutive batches of 4,096 from the block's first row, the last padded with zeros. It then decides the rows that D
  routes (S55'-deep's margin below tau1, compared in float64), in shot order and in batches of 4,096, the last padded.
- Single: compiled@1 on every row. One row at a time, as phase C's `Graph.run` does it, this takes 1.77 ms of GPU time
  per shot, most of it idle hardware. The read therefore runs K = 32 copies of the batch-1 graph at once on K CUDA
  streams, each capturing R = 16 consecutive batch-1 calls of the same compiled function on its own static input. Each
  call is phase C's compiled@1 computation on one row: the same kernels on the same input shape. The read checks this
  bitwise, every decision and confidence, against `pc_speed.build(C14', "compiled", 1).run` one row at a time in the
  same process, before it starts (the block's first 20,000 rows) and after it ends (every 1,000th row). It writes
  nothing if one bit differs.

Before these, the read job checks S55'-deep's read engine (the S anchor): `pc_eval.py`'s fused path on the first 32,768
rows of C4b's frozen read at p must give that read's every decision and margin bit, as each D1-S process checks its
engines (4.2). The compiled graphs come from copies of phase C's Triton and Inductor caches (5.1), so they are the
binaries and autotuning choices that phase C timed and checked. Each read records Inductor's cache counters. The dry
run fixed K and R by profiling (5.6), checked the concurrent path bitwise on 100,000 dry rows, and reproduced phase C's
claim-point check counts for C14' on that check's own rows as an anchor. If the concurrent path had failed its dry
check, the read would have used phase C's `Graph.run` one row at a time, split over more jobs.

## 3. The one analysis

### 3.1 Refusals and reruns

`d1_confirm.py` runs on acpu after both read jobs and the speed job have ended. It writes no result, and says why, if:

- a read's family is not `depspam_fm`;
- a network's sha256 or read precision differs from the lock;
- the card or torch differs from the lock's (phase C's full RTX PRO 6000, torch 2.11.0+cu128);
- the seed kind is not the lock's (14);
- the chunks are not c0 to c0 + n - 1 for the c0 of F, or a chunk's seed is not eval_seed(14, p, j, "depspam_fm");
- the number of shots is not 10,000 n;
- the lock's configurations are not exactly configurations 2 and 7 of the first lock, whose file must have the sha256
  above;
- the compiled read's chunk digests differ from the eager read's, its paths are not the lock's frozen paths, a bitwise
  check of its concurrent path or the S anchor did not pass, its routed rows are not the rows that the analysis routes,
  or its card, torch or tree differ from the lock's and F;
- a timing row's lock or tree is not D1's frozen lock and F.

The driver is recorded and reported but not checked, as in phase C.

If the analysis stops on a refusal or an error before it writes its result, the cause is fixed in a commit that leaves
the rules of this section unchanged, the fix is disclosed, and the analysis runs again. After any result exists, no
configuration is adapted, no shot is added and no other analysis is substituted. A discovered bug may be corrected,
and the analysis or a read may be rerun to reproduce a result, with the same data and the same rules. The correction is
disclosed, and every earlier output is kept: each run of the analysis writes a new numbered directory.

### 3.2 Per claim point and path

From the eager read at p the analysis takes S55'-deep's margins and failures, and C14''s eager failures. From the
compiled read it takes C14''s decisions on compiled@1 and compiled@4096 for every shot, and B's compiled@4096
decisions on the routed shots. It forms the routed mask (margin below tau1, `pc_route.masks`) and, per path, D's
failures: S's failure on a kept shot, and C14''s failure on that path on a routed shot (`pc_route.config_fail` on the
eager path; the same rule on the compiled paths, where in batched mode a routed shot takes B's decision from the routed
batches). `bb.route.paired(D, C14', 0.05)` then gives, per path, both failure counts, the shots that only D fails and
that only C14' fails, the exact two-sided McNemar p and Newcombe's 95 % interval (method 10) for P(D) - P(C14').
Relay-BP plays no part.

### 3.3 Holm and the verdicts

`pc_common.holm` at alpha = 0.05 over the four p-values (H1e, H2e, H1c, H2c; a compiled comparison's p-value as in 1.2)
gives each comparison its level: 0.0125 for the smallest, then 0.05 / 3, 0.025 and 0.05 while the earlier ones are
rejected, and none otherwise. As in phase C's `pc_report.headline`, a comparison with no level is "level". An eager
comparison with a level is "better" when Newcombe's interval at that level (`pc_report.paired_counts`) lies below zero,
"worse" when it lies above zero, and "level" otherwise. A compiled comparison with a level is "better" when both its
single and its batched intervals at that level lie below zero, "worse" when both lie above zero, and "level"
otherwise. "D is more accurate than Cascade at both claim points" needs "better" in both compiled comparisons.

### 3.4 What is reported

At each claim point, on each path (eager, compiled single, compiled batched), whatever the verdict:

- D's and C14''s failures and block failure rates with Wilson's intervals, and P_L by Eq. (1);
- the difference with Newcombe's 95 % interval and its interval at the Holm level, and the difference as a share of
  C14''s rate;
- McNemar p and the Holm level;
- the routed fraction, and among the kept shots the breaks, the repairs and the shots that both fail;
- C14''s failures on routed and on kept shots, which give the cap;
- S55'-deep alone against C14', as a description;
- how the compiled paths differ from the eager one: the shots C14' decides differently, the failures that change, and
  how many of them fall on kept shots.

The verdict decides how the point is stated:

- Better: the gain with its interval, with the cap beside it.
- Level: "no significant difference from Cascade at p", which means neither equivalence nor "as accurate". The report
  gives what the 95 % interval leaves open: a gain up to the size of its lower end, an excess up to its upper end. The
  lead of EXPLORE_DEEP is then not confirmed at p on that path.
- Worse: "worse than Cascade at p", and the lead is reversed there on that path.

Whatever the verdicts, nothing is read again, enlarged or retested, no other configuration is claimed from these shots,
and phase C's verdicts stand. Once analysed, D1's shots are no longer fresh.

Descriptive rows, outside the family and never a claim: S55'-deep -> C14' at the first lock's other grid points (q1 =
1, 5, 10, 20 and 40 % at 0.2 %, and 1, 2, 5, 20 and 40 % at 0.3 %), each with the first lock's tau1, counts and
Newcombe's 95 % interval, with no p-value and no verdict, on the eager path and on compiled@1 (whose decisions do not
depend on the routed set). They show how the gain and the cap move with the routed fraction on fresh shots.

## 4. Speed

### 4.1 What is timed, and how

One GPU job, D1-S, times D at both claim points with phase C's pipeline protocol (`PHASE_C_PLAN.md` 5.3), using
milestone 2's harness, `m2.py pipeline` from `ising-cascade-batchopt` 48150303, unchanged, with D1's lock in place of
Amendment 1's (`d1_m2.py`, 5.2). At each claim point there are six runs:

| run | S | B | single mode | batched mode |
|---|---|---|---|---|
| Cascade alone | - | C14' | compiled@1 | compiled@4096 |
| D, today's engine | S55'-deep on the registered fused engine | C14' | fused@1, B compiled@1 | fused@4096, B compiled@4096 |
| D, milestone 2's engine | S55'-deep on `mfused.Engine` | C14' | batch 1, B compiled@1 | batch 4,096, B compiled@4096 |

Single mode decodes 20,000 shots one at a time after 200 untimed ones, from pinned host memory to the decision on the
host. Batched mode decodes 500,000 shots with S and B at batch 4,096. The routed rows queue on the GPU, and B runs
whenever 4,096 are waiting and once more at the end on a padded batch. Milestone 2's lever 2 is not used, because its
small Cascade graph is not bitwise the full one. The thresholds are applied to the timed margins in float64, as in the
reads. The shots are the speed stream of phase C's C7 and of milestone 2 (seed kind 12 from chunk 0). Since the
pipelines select nothing from these shots, reusing them is harmless, and it keeps the times comparable with C7's and
milestone 2's.

The protocol is phase C's. Each run goes in three fresh processes in a rep-major loop (the twelve runs of both claim
points, then all twelve again, then a third time), with Cascade alone in the same job and loop. Each value is the slowest
of the three processes: the mean single-shot latency, the 99.9th percentile and the batched time per shot
(`pc_report.pipes`, applied once to today's engine's rows and once to the new engine's, with Cascade alone's rows in
both). Per round is per shot divided by 6. The SM clock record (`bb.timing.gpu_watch`, the 97 % rule) is kept and
disclosed, and as in phase C no process is rerun or replaced. A speed-up is Cascade alone's value divided by D's, at the
same claim point in the same job.

### 4.2 Eligibility: why the times describe the tested decoder

- The read runs S55'-deep on the fused engine (`pc_fused.Batched` at 4,096). Phase C showed it bitwise its fused@1 and
  fused@4096 paths (the screen in C3 and the claim-point checks).
- Milestone 2's engine is bitwise today's for S55'-deep. Its screen found no decision and no margin bit pattern that
  differs in 17,750,000 evaluations at 14 batch sizes, and every intermediate buffer matched launch by launch. Each D1-S
  process checks this again (`S_check`: 2,000 shots in single mode and 32,768 in batched mode, against C4b's frozen
  read). One differing decision or margin bit removes that engine's values at that claim point from the claim. With S
  bitwise, D routes and decides every shot the same way on both engines, so the accuracy verdicts hold for both.
- C14''s accuracy is read on the computations the timings run (2.6): compiled@1 for single mode and compiled@4096, with
  the pipeline's batching, for batched mode. The compiled graphs come from the same cache copies in the reads and in
  D1-S, and the concurrent single read is checked bitwise against phase C's own compiled@1 path object.
- The first lock's claim-point check (job CJ34, `phase_c/out/lock_1e8d1c6d/check_depspam_fm.json`) passed
  configurations 2 and 7, the two configurations of D1, in single and batched mode, with S bitwise and the routing equal
  (at 0.2 % in single mode, 2,066 of 100,000 shots routed in both).

### 4.3 Expected values

Predicted from the lock's components and milestone 2's times; these are not claims:

| p, q1 | engine | single mean, us (per round) | batched, us per shot (per round) | Cascade alone in C7: single / batched | predicted speed-up: single / batched |
|---|---|---|---|---|---|
| 0.2 %, 2 % | today's | 166.5 (27.8) | 8.2 (1.37) | 1,696.5 / 71.87 | 10.2 / 8.8 |
| 0.2 %, 2 % | milestone 2's | 136 (22.7) | 4.6 (0.77) | | 12.5 / 15.6 |
| 0.3 %, 10 % | today's | 309 (51.5) | 13.9 (2.32) | 1,700.8 / 71.88 | 5.5 / 5.2 |
| 0.3 %, 10 % | milestone 2's | 279 (46.5) | 10.4 (1.73) | | 6.1 / 6.9 |

Single mode adds S's single-shot time (130.9 us today, 100.5 us on milestone 2's engine) to q1 times C14''s 1,782 us.
Batched mode adds S's time per shot at 4,096 (6.26 and 2.69 us) to q1 times 71.6 us, plus about 0.5 us for the padded
last B batch, the excess over the components that C7 and milestone 2 measured. The 99.9th percentile will be higher
than Cascade's: with 2 % or 10 % routed, more than 0.1 % of the shots pass through both S and C14', about 1.9 ms against
Cascade's 1.72 ms in C7.

### 4.4 What may be claimed

A claim at a claim point needs three things: the compiled comparison there is "better"; D's mean single-shot latency
and its batched time per shot are both lower than Cascade alone's in the same job; and every check of 4.2 holds. The
claim then reads:

> In the fitted `depspam_fm` setting at p, S55'-deep routing its q1 least sure shots to C14' failed less often than our
> re-implementation of Cascade (C14') on fresh shots, on the computation that was timed, by x [interval] in single mode
> and x' [interval] in batched mode, and was faster on average on the RTX PRO 6000: y times per single shot and z times
> batched with today's engine (y' and z' with milestone 2's), with a slower tail.

The eager comparison's verdict and interval stand beside it, with the per-round values, the precisions (BF16 for C14',
FP16 matmul inputs with FP32 accumulation for S55'-deep) and the card. If the eager comparison is "better" and the
compiled one is not, the report says that D failed less often than C14' on phase C's accuracy path but not on the timed
computation, and makes no claim that joins accuracy and speed. At a point where the compiled comparison is level or
worse, the speed results are reported as speed only.

D1 claims nothing of the following: superiority over the published Cascade; anything at another p or in `uniform`, which
D1 does not read; anything on the H100, which D1 does not time; superiority in tail latency; a gain beyond the interval;
phase C's "beats Cascade" (its section 6), whose condition that the decoder is not worse at any other p where it was read
is not tested for these configurations; or that the hypothesis was specified before phase C's confirmation data were
seen.

## 5. Code, jobs, disclosure and risks

### 5.1 Reused unchanged

- Phase C: `cascade_compare/phase_c/` as on this branch, with the Cascade tree of `ising-cascade-net` 64b2e25d. That
  covers the read (`pc_eval.py`, `jobs/confirm.sh`, `jobs/curves.sh`, `pc_fused.py`), the helpers in `pc_common.py`
  (the networks, `load_read`, `point`, `holm`, Eq. (1), the card guard), the routing (`pc_route.masks`, `config_fail`),
  the compiled paths (`pc_speed.setup`, `make_fwd`, `build`, `Graph`), the verdicts and aggregates
  (`pc_report.paired_counts`, `pipes`), `bb.route.paired` and `power`, the frozen speed components (sha256
  1ec6b256...), and the first lock with its claim-point check.
- Milestone 2: `cascade_compare/batchopt/` at 48150303, that is `mfused.py`, `m2.py pipeline`, and the `m1.py` and
  `prof_batched.py` that `m2.py` imports. D1's branch takes them by merging 48150303, whose commits since phase C's
  629ca1c4 touch only that directory. The later 4a5b9e6a changes only `m2.py`'s screen and drive steps, which D1 does
  not run.
- Phase C's environment: bb's `venv_cu128` (torch 2.11.0+cu128) in the `pyfull` container, on a full RTX PRO 6000.
  Triton's and Inductor's caches are copied from phase C's (`cascade_w4/triton_cache` and `inductor_cache`, read only)
  into D1's directory once, as milestone 2 did, and each GPU job works on its own copy of that seed, so that Cascade's
  compiled paths and S55'-deep's fused engine run the binaries and autotuning choices phase C timed and read.

### 5.2 New

Everything new goes under `cascade_compare/phase_d/`:

| file | what |
|---|---|
| `d1_locked_depspam_fm.json` | the lock (1.5) |
| `d1_common.py` | D1's constants and guards: the directories, the lock, c0 from F, the first lock's check, the tree record |
| `d1_compiled.py` | the compiled reads (2.6), the profile and the dry checks of the concurrent path |
| `d1_confirm.py` | the analysis of section 3 and the speed summary of 4.1, built from phase C's functions |
| `d1_power.py` | D1-P (2.4) |
| `d1_drytest.py`, `d1_drylock.py` | the dry runs' refusal cases and compatibility checks, and their dry lock (kind 13, 2 chunks per point) |
| `d1_m2.py` | `m2.py` with D1's lock: it replaces `m2.frozen_speed`, which names Amendment 1's lock, and then calls `m2.main()`, as `m2.py` itself replaces `pc.gpu_env` for the H100 |
| `jobs/d1_guard.sh` | the guards of 2.2 and 2.3 |
| `jobs/d1_read.sh` | the guards, c0 from F, phase C's `curves.sh` with kind 14, C14' and S55'-deep, the lock's chunk group, then `d1_compiled.py` on the same shots |
| `jobs/d1_speed.sh` | the rep-major loop of 36 fresh `d1_m2.py pipeline` processes, stopping at the first one that fails |
| `jobs/d1_confirm.sh`, `jobs/d1_power.sh` | run `d1_confirm.py` and `d1_power.py` |
| `jobs/d1_dry_gpu.sh`, `jobs/d1_dry_cpu.sh` | the dry runs of 5.3 |
| `scripts/` | `deploy.sh`, `submit.sh`, `gpu.sbatch` and `cpu.sbatch`: phase C's, with D1's directory `<cluster-scratch>/cascade_d1` and a tree of `phase_c`, `batchopt` and `phase_d` |

No file of phase C or of milestone 2 is edited. On Alpine nothing under `cascade_w4` or `cascade_batchopt` is written;
both are only read.

### 5.3 Jobs and their order

| step | what | where | GPU minutes | after |
|---|---|---|---|---|
| D1-0 | the seed audit (2.2) | this plan | 0 | done |
| gate | An external review of the draft; the dispositions are committed with this revision | | | |
| D1-1 | the new code (5.2); merge 48150303 | | 0 | gate |
| D1-dry | GPU: the profile of the concurrent compiled@1 read; its bitwise check on 100,000 rows of kind 13; the anchor (C14''s compiled@1 and compiled@4096 on the first lock's claim-point check rows must reproduce that check's counts for C14' and for configurations 2 and 7); `d1_read.sh` on kind 13 (2 chunks per p at the c0 of the dry tree's commit); `d1_speed.sh` with one repetition and 2,000 / 50,000 shots. CPU: `d1_confirm.py` on the dry read with a dry lock; each refusal of 2.2, 2.3 and 3.1 made to fire once on a perturbed copy; the compatibility checks of disposition 5 (the first lock JSON by path and sha256, configurations 2 and 7 and their thresholds, `d1_m2.py` with `m2.py`, `pc_report.pipes` and `paired_counts` on the dry rows, `bb.route.paired` and `power`) | Alpine, a full RTX PRO 6000; acpu | about 30 | D1-1 |
| D1-P | the simulated powers (2.4), written into the lock and section 2.4 | Alpine acpu | 0 | D1-1 |
| freeze | commit F (plan, lock, code, dry-run records), pushed to the hub and deployed | | | D1-dry, D1-P |
| D1-R1 | read at 0.2 %: C14' and S55'-deep on 1,100 chunks of kind 14, then the compiled reads | a full RTX PRO 6000, 16 cores | eager about 32, compiled measured | freeze |
| D1-R2 | read at 0.3 %: 1,700 chunks, the same | the same | eager about 48, compiled measured | freeze |
| D1-S | the pipelines' times (4.1) | a full RTX PRO 6000, 16 cores | about 40 | freeze |
| D1-A | `d1_confirm.py` | Alpine acpu | 0 | D1-R1, D1-R2, D1-S |
| report | `D1_RESULT.md`; the JSON, timing rows, logs and the reads' sha256 committed, the reads kept on Alpine | | | D1-A |

D1 touches nothing in phase C's run directory.

### 5.4 Disclosure

The results will say:

- Origin. D1's hypothesis came from a post hoc look at phase C's confirmation block. EXPLORE_DEEP evaluated 52
  configurations on those reads after the registered results were known, without adjusting for multiplicity.
- Superseded lock. D1 tests the two claim-point configurations that phase C's first lock (1e8d1c6d) picked on the
  development block before any confirmation read, with the first lock's thresholds, never refitted. Amendment 1
  (629ca1c4) replaced that lock before any confirmation read, and Amendment 1's lock remains phase C's lock of record.
- Scope. `depspam_fm` is a fitted noise model (the noise-model fit's variant), not the paper's noise, and C14' is our
  re-implementation of Cascade, not the published network.
- Effect sizes. On phase C's confirmation block the two configurations were 7.2e-06 and 7.0e-06 below Cascade's block
  failure rate (0.93 % and 0.16 % of it), capped by Cascade's failures on kept shots (4.7 % and 3.1 % of them).
- Phase C's verdicts do not change, whatever D1 finds.
- Fresh shots. Seed kind 14, with chunks set by the freeze commit, drawn after the plan was frozen and published, and
  analysed once; their freshness is audited provenance (2.3).
- The compiled paths. C14' was read on the eager path and on the compiled paths; how often they differ is reported.

### 5.5 Risks

| risk | how it is handled |
|---|---|
| The lead does not replicate (selection, regression to the mean). | The blocks are sized for half the gain. A level result is reported as in 3.4 and closes the lead on that path. |
| The gain is a rounding effect of the eager path. | The compiled comparisons test the computation the timings run, and the claim rests on them (1.2, 4.4). |
| Even a confirmed gain is small. | The cap (4.7 % and 3.1 % of Cascade's failures) is reported beside every result. D1 makes no claim of a large gain. |
| Alpine's GPU software has changed since phase C. | The read refuses another card or torch. A new driver is recorded and disclosed; it may change Cascade's BF16 rounding but not the pairing, since D and C14' come from the same read on each path. |
| A read job fails midway. | It is resubmitted unchanged. `curves.sh` marks the eager read done only when it is complete, the compiled read has its own marker, the same seeds give the same shots, and nothing partial is looked at. |
| The concurrent compiled@1 read is not bitwise phase C's path. | It is checked before and after the read against phase C's own path object, and it writes nothing if one bit differs; the dry run checks it on 100,000 rows first. |
| Processes miss the clock rule. | They are recorded and disclosed, as in phase C. The predicted speed-ups (5 to 16 times) are far larger than the 3-4 % by which flagged processes moved times in C7 and milestone 2. |
| The new engine is not checked on D1's own shots. | Its equality rests on milestone 2's screen and on D1-S's in-process checks (4.2). |
| Other work draws kind 14 before the read. | The read job's search (2.2) stops the read. |
| The new analysis has a bug. | The dry run on kind 13 makes each refusal fire, and every statistic comes from phase C's functions. A bug found after a result is corrected as in 3.1. |

### 5.6 The dry runs and D1-P, before the freeze

They ran on Alpine on full RTX PRO 6000 cards and on acpu, on the tree of commits d285960f and 80976c46, and they are
never reported as results. Their records are in `phase_d/dry/`; the outputs stay under `cascade_d1/dry/` on Alpine.

- Profile (GPU job CJ67, kind 13 at 0.3 %). Phase C's compiled@1 Graph, one row at a time, took 1,772 us per row.
  The concurrent path took 1,768 us with one stream, 593 at 4 x 16, 351 at 8 x 16, 232 at 16 x 16, 222 at 16 x 64, and
  179 at 32 x 16, 64 x 16 and 32 x 32. Every configuration was bitwise phase C's path on the 2,000 rows compared. The
  lock takes 32 streams of 16 calls.
- Verify. On 100,000 dry rows the concurrent path was bitwise phase C's Graph, and bitwise on a repeat, at 16 x 16
  (job CJ67) and at 32 x 16 (job CJ69, 182 us per row against 1,689).
- Anchor (both jobs). On the rows of the first lock's claim-point check (C4b's frozen read, 100,000 shots per point),
  C14''s compiled@1 (concurrent) and compiled@4096 reproduced all eight of that check's rows exactly. For C14' alone in
  both modes at both points this covers the shots decided differently from the eager read (8 and 7 at 0.2 %, 39 and 37
  at 0.3 %), the confidences that differ (60,238, 59,721, 56,697 and 56,356) and the failures (74 eager against 70 and
  72 compiled at 0.2 %, 444 against 445 at 0.3 %). For configurations 2 and 7 it covers D's failures in both modes (73
  eager against 70 and 72 at 0.2 %, 443 against 445 at 0.3 %) and the shots routed (2,066 and 10,017).
- Guard. The read job's search over the audit's directories and D1's own took 10 s and found no other record of kind
  14.
- Dry read (both GPU jobs). `d1_read.sh` on kind 13, 2 chunks per point at the c0 of the dry tree's commit: the eager
  read, the S anchor (32,768 rows bitwise), the compiled reads and their bitwise checks before and after all passed.
- Dry speed (job CJ67). The twelve runs once each, at 2,000 and 50,000 shots, through `d1_m2.py` and `m2.py`'s
  pipeline. Every in-process S check was zero. All six single-mode processes missed the 97 % clock rule (lowest SM
  clock ratio 0.966) and all six batched ones met it.
- CPU (job CJ70). Every new file parses. `d1_confirm.py` accepted the dry read and wrote its result. Each of the 22
  refusal cases of 3.1 fired with its reason, and the unperturbed inputs were accepted. The read guard fired on a wrong
  F, a dirty tree, a kind-14 record elsewhere and a read already done, and passed with a kind-14 record inside the read
  directory and a kind-140 record elsewhere. The compatibility checks of disposition 5 passed: the first lock by its
  Alpine path and sha256; the lock's configurations equal to its configurations 2 and 7; every grid row's tau1,
  development counts and predicted latencies equal to the first lock's candidates; `pc_report.pipes` and
  `paired_counts` on the dry timing rows; `bb.route.paired` and `power`.
- D1-P (job CJ68): section 2.4.

## Appendix: the audit commands

On a clone of the hub with all 35 branches fetched (11 October 2026):

    refs=$(git for-each-ref --format='%(refname:short)' refs/remotes/origin | grep -v HEAD)
    git grep -h -o -I -E -- '--kind[ =]+[0-9]+|KIND[A-Z_]*=\$?\{?[A-Z_]*:?-?[0-9]+|KIND_[A-Z0-9_]+ *= *[0-9]+|eval_seed\( *[0-9]+|kind *= *[0-9]+' $refs
        # kinds 0-7 and the defaults and constants of phase C's 10-13; nothing else
    git log --all -G'kind[ =:"_seed]*1[4-9][^0-9.]' -- '*.py' '*.sh' '*.sbatch' '*.md'
        # 3 commits, all study 4's "d = 17" (85b2c893, 222fe3c7, 1e07b9db)
    git log --all -G'KIND[A-Z_]*[ =]*1[4-9][^0-9.]' -- '*.py' '*.sh' '*.sbatch' '*.md'    # none
    git log --all -G'eval_seed( *1[4-9]' -- '*.py' '*.sh' '*.sbatch' '*.md'               # none
    git grep -h -o -I -E '"kind(_seed)?": *[0-9]+' $refs -- '*.json' '*.jsonl'           # 0, 2, 3, 5, 6, 7, 10-13

On Alpine, read only:

    cd <cluster-scratch>
    for d in cascade_w1 cascade_w2 cascade_w3 cascade_w4 cascade_w5 cascade_batchopt cascade_explore cascade_h100b \
             cascade_h100t; do grep -rhoE --include=*.json '"kind(_seed)?": *[0-9]+' $d | tr -d ' ' | sort | uniq -c; done
        # kinds 0, 2, 3, 5, 6, 7 and 10-13 only; no JSON in cascade_w1, cascade_w2 or cascade_w5 records a kind

From D1's revision on, D1's own code and plan name kind 14 (the lock's `kind_seed` and the read job), so a repeat of the
hub searches finds those; the read job's Alpine search excludes only D1's read directories.
