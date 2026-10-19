# ARBCP: the frozen joint decision on Cascade's compiled paths, on Table I's shots

19 October 2026. Exploratory follow-up of ARB (branch `arb-exp` of the research repository). The parallel run ARB4 counted
the frozen joint decision on Table I's test shots on Cascade's eager accuracy path, because the re-test's compiled
files hold only Cascade's decisions. This note counts the same rule on the two compiled paths whose failures Table I
reports: single (Cascade compiled at batch 1) and batched (compiled at batch 4,096 on BB72, 16 at d = 13). The shots,
the network's decisions and margins, the routing and the rule are the re-test's and arb4's; only Cascade's path
changes. Nothing here is registered.

## Result

Failures on all of Table I's test shots: 11,000,000 at BB72 p = 0.2 %, 17,000,000 at 0.3 %, 1,000,000 at d = 13
p = 0.6 % and 30,000,000 at 0.35 %. The BB72 eager rows are the cross-check against arb4 and equal its counts exactly
(8,063 / 7,981 / 6,580 and 75,957 / 75,700 / 65,166).

| setting | path | Cascade alone | composition | joint decision | repairs / breaks | joint vs composition | McNemar p | joint vs Cascade alone | McNemar p (only Cascade fails / only joint fails) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| BB72 0.2 % | single | 8,074 | 7,986 | 6,597 | 1,424 / 35 | -17.4 % | 1e-368 | -18.3 % | 1e-386 (1,520 / 43) |
| BB72 0.2 % | batched | 8,073 | 7,984 | 6,588 | 1,425 / 29 | -17.5 % | 1e-377 | -18.4 % | 1e-394 (1,522 / 37) |
| BB72 0.2 % | eager | 8,063 | 7,981 | 6,580 | 1,433 / 32 | -17.6 % | 1e-375 | -18.4 % | 1e-391 (1,523 / 40) |
| BB72 0.3 % | single | 75,979 | 75,740 | 65,189 | 11,369 / 818 | -13.9 % | 1e-2367 | -14.2 % | 1e-2379 (11,698 / 908) |
| BB72 0.3 % | batched | 75,973 | 75,735 | 65,185 | 11,368 / 818 | -13.9 % | 1e-2367 | -14.2 % | 1e-2379 (11,696 / 908) |
| BB72 0.3 % | eager | 75,957 | 75,700 | 65,166 | 11,348 / 814 | -13.9 % | 1e-2365 | -14.2 % | 1e-2383 (11,693 / 902) |
| d = 13, 0.6 % | single | 4,936 | 4,917 | 4,886 | 249 / 218 | -0.6 % | 0.165 | -1.0 % | 0.033 (289 / 239) |
| d = 13, 0.6 % | batched | 4,947 | 4,927 | 4,884 | 260 / 217 | -0.9 % | 0.054 | -1.3 % | 0.008 (301 / 238) |
| d = 13, 0.35 % | single | 2,354 | 2,325 | 2,306 | 90 / 71 | -0.8 % | 0.156 | -2.0 % | 6.1e-04 (119 / 71) |
| d = 13, 0.35 % | batched | 2,352 | 2,322 | 2,308 | 86 / 72 | -0.6 % | 0.301 | -1.9 % | 0.002 (116 / 72) |

How to read the table:

| term | plain meaning |
|---|---|
| Cascade alone | Cascade decodes every shot on that path: the re-test's own numbers (Table I) |
| composition | the network answers, except on its least confident shots ("routed": margin below the frozen threshold tau1), which Cascade answers on that path: Table I's numbers |
| joint decision | the composition, except on routed shots where the network and Cascade disagree: there the frozen rule picks the answer from both decoders' scores |
| repairs / breaks | shots the composition fails and the joint decision gets right / the reverse |
| joint vs ... | relative change in failures, (joint - other) / other |
| McNemar p | exact two-sided binomial test on the shots where exactly one of the two fails; 1e-368 means 10^-368 (exact value, printed by its exponent) |

On BB72 the joint decision repairs 41 to 49 shots for each one it breaks at 0.2 % and 14 at 0.3 %. Against the eager
path, the compiled paths change every failure count by at most 0.3 %. At 0.2 % (0.3 %) the rule picks a third answer,
neither decoder's, on 1,373 (12,335) routed shots on the single path; 528 (4,144) of them are repairs, 10 (206) breaks,
and the rest change one wrong answer into another.

At d = 13 the joint decision is rule J of the d = 13 exploration (ARB2, `~/work/arb2` on the Linux workstation) with its fitted
w: 7.86574441929562 at 0.6 % and 13.842311471679352 at 0.35 %, fitted there on the even chunks of its own fresh shots
(`out/analysis/summary.json`, sha256 bd4e9f3d...). It removes 0.6 to 0.9 % of the composition's failures on Table I's
shots, with McNemar p from 0.054 to 0.30 against the composition on every path. arb2's evaluation half shows the
same size (-1.1 % at 0.6 %, p = 0.0002 on 2,000,000 shots; -0.5 % at 0.35 %, p = 0.53 on 15,000,000). By its own
criterion arb2 carried rule P forward, not J; this note applies J, the joint decision's rule at d = 13.

## Do the recomputed Cascade outputs equal the re-test's?

On every routed shot the recomputed decisions and confidences equal the re-test's bit for bit:

| setting | routed shots | single: decisions differing from bits1 | single: confidences differing | batched: decisions differing from bitsR | batched: confidences differing |
|---|---:|---:|---:|---:|---:|
| BB72 0.2 % | 219,891 | 0 | 0 | 0 | 0 |
| BB72 0.3 % | 1,699,201 | 0 | 0 | 0 | 0 |

The confidence is min |z| over the 12 logits in float32, compared by bit pattern. The batched decisions also equal the
re-test's bits4 (Cascade alone, batched over the whole read) on every routed shot, as the re-test's own check found.
On the first 8,192 routed shots of each setting, the unmodified compiled computation (no logit output) gave the
re-test's decisions and confidences bit for bit too, and the computation with the logit output gave the unmodified
one's.

At d = 13 nothing was recomputed. The re-test's confirmation read saved Cascade's float32 logit and decision on both
compiled paths for every shot it decided, and the joint decision uses exactly those. Each saved decision is its logit's
sign on every shot (B1 and Bb on all shots, BQ5 on all routed shots), and BQ5 holds exactly the routed shots. On the
routed shots BQ5's logits equal Bb's bit for bit at both p (the routed queue and the whole read give the same batch-16
outputs), while B1 and BQ5 differ in 62 (0.6 %) and 29 (0.35 %) decisions: the batch-1 and batch-16 kernels' own
difference, which Table I's two modes already carry.

## What was run

### BB72 (one GPU job, Alpine AJ50, 3.6 min at 0.2 % and 8.0 min at 0.3 %)

- Shots: the re-test's D1 reads (`retest/out/d1/<p>/read/p<p>.npz`, packed detector rows included) and compiled reads
  (`compiled/p<p>.npz`: bits1, conf1, bits4, conf4, routed, bitsR, confR).
- Routed shots: S55'-deep's read margin (float32, compared in float64) below D1's tau1, 7.3331888389587405 at 0.2 %
  and 7.60439167022705 at 0.3 % (219,891 and 1,699,201 shots, equal to the compiled read's `routed`). The network's
  decisions, margins and class scores are those of the read.
- Cascade C14' (the release's `weights/cascade_bb72/C14p.pt`, EMA weights, bf16 autocast) through
  `pc_speed.make_fwd`'s computation (decision z > 0, confidence min |z|) with the 12 logits as one more output, under
  `torch.compile` (default mode) and CUDA graphs:
  - single: compiled@1 through the re-test's `d1_compiled.Multi` (32 streams, 16 batch-1 calls per graph), one shot
    per call, on every routed shot. A batch-1 call sees one shot, so the order does not matter.
  - batched: compiled@4096 over the routed shots in shot order, consecutive batches of 4,096 from the first routed
    shot, the last one zero-padded. This is how the re-test computed bitsR (`d1_compiled.py read`: "B in D's batched
    mode, as the pipeline queues them"), the decisions that `d1_confirm.py` uses for the composition's batched
    failures. Cascade alone in batched mode (bits4) batched every shot of the read in order instead; its failures are
    taken from the re-test's file.
- Environment: one full RTX PRO 6000 Blackwell Server Edition (node gpu-node-07, driver 590.48.01), D1's venv
  `bb/venv_cu128` (torch 2.11.0+cu128, triton 3.6.0) on Python 3.12.14 (uv's build, because D1's container image is
  gone from Alpine), the re-test's frozen release tree, and D1's seed kernel caches (the copy in
  `speedrt/cascade/caches/r1_*.tar`, made from `cascade_d1/cache` on 13 October) unpacked into the job. Compiling the
  unmodified computation hit the caches exactly as the re-test's compiled read did (FX graph cache: 2 hits; async
  compile: 120 misses, 80 hits). Triton rebuilt its two small launcher modules for this Python; the kernels came from
  the caches.
- Network class scores: S55'-deep's 4,096 class scores from its fused engine (ARB's `arb_fused.Scored`), on the
  routed shots where its decision differs from Cascade's on any path (9,040 at 0.2 %, 94,980 at 0.3 %), with
  `arb_scores.py`'s checks: decisions and margins bitwise the read's, every program's top two and the argmax and margin
  recomputed from the scores; all passed.
- Joint decision, ARB's A3 as arb4 froze it: on a routed shot where S55'-deep's and Cascade's decisions differ,
  answer argmax over the 4,096 classes y of s(y) + w (y . z), with s the network's class scores, z Cascade's 12 logits
  on that path and the class bit j = observable j, in float64 on the GPU as `arb_analysis.a3_answers`;
  w = 10^(32/40) = 6.309573444801933 at 0.2 % and 10^(23/40) = 3.758374042884441 at 0.3 %. Everywhere else the
  composition's answer stands. The eager rows use the read's float16 logits z0, whose signs are the read's decisions
  on every routed shot. No objective was within 1e-6 of a tie on any path.

### d = 13 (no GPU; Alpine CPU job AJ51)

- Shots: the re-test's confirmation read of C5, "the pair" (ConvLines `d13_b5_r775` with Cascade CS13 behind it),
  `retest/out/d13/conf/p<p>`, loaded with the release's `sv_common.load_read`: 1,000,000 shots at 0.6 % and 30,000,000
  at 0.35 %.
- Routed: |c| (float32, compared in float64) strictly below C5's tau1 from the release's lock, 5.242445659637451 at
  0.6 % and 8.111059150695802 at 0.35 %: 99,845 and 300,924 shots. The network's batch-1 (S1) and batched (SB) logits
  are bitwise equal, so both modes route the same shots.
- Paths, as `sv_common.assemble` builds C5 and C2: single = S1, with B1 (Cascade compiled@1) on the routed shots, and
  B1 for Cascade alone; batched = SB, with BQ5 (compiled@16 over the queue of routed shots in shot order, the last
  batch zero-padded) on the routed shots, and Bb (compiled@16 over every shot) for Cascade alone.
- Rule J (arb2's PLAN.md step 4): on a routed shot where the network's and Cascade's decisions differ, answer 1 iff
  c + w z > 0 in float64, Cascade's answer on a tie (no ties occurred); everywhere else the composition's answer.

### Counting

`code/cp_count.py` (BB72) and `code/cp_count_d13.py` (d = 13), light numpy on the per-routed-shot classes the jobs
wrote. Kept shots answer with the network in the composition and in the joint decision alike, so the two differ only
on routed shots; against Cascade alone the kept shots where exactly one of the network and Cascade fails count too
(BB72 kept repairs / breaks of the composition: 96 / 8 single and 97 / 8 batched at 0.2 %, 329 / 90 and 328 / 90 at
0.3 %, the re-test's numbers). McNemar p: exact two-sided binomial test at 1/2 on the discordant shots
(`bb.route.paired`, `sv_common.mcnemar_p`); `cp_count.mcnemar` reproduces the re-test's recorded p-values (for
example 5.452e-172 at BB72 0.2 % eager, composition against Cascade alone).

## Files

- This branch (`arb-table1-compiled` of `<research-repo>`): this note; `code/` (cp_run.py and
  cp_bb72.sbatch, the GPU job; cp_d13x.py, the d = 13 extraction; cp_count.py, cp_count_d13.py, cp_tables.py;
  arb_fused.py copied unchanged from `arb-exp`); `out/` (the jobs' records, `bb72_counts.json`, `d13_counts.json`)
  and the Alpine jobs' logs (AJ49 is a first submission that stopped at once: the Relay-BP circuit path was
  unset).
- Per-shot arrays, not in git: `~/work/arbcp/out/bb72/p<p>_logits.npz` (Cascade's 12 float32 logits, decision and
  confidence on both compiled paths for every routed shot), `p<p>_routed.npz` (per routed shot: the classes of the
  observables, the network, Cascade on each path and the joint decision on each path), `~/work/arbcp/out/d13/d13_<p>.npz`.
  The network's class scores (1.5 GB at 0.3 %) were not copied back; `cp_run.py` recomputes them in seconds.
