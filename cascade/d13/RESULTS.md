# Cascade behind ConvLines on the d = 13 surface code: results

12 October 2026. The evaluation of SURFACE_PLAN section 9 has
run to its end: the confirmation reads (jobs e5 and e6), the pipelines with Cascade alone timed beside them (e7), and
the bitwise check, the one analysis and the report (e8). This file gives the verdicts of SURFACE_PLAN section 7, the
counts behind them, the speed ratios with the clock flags of their timed windows, and what the conditions P1 and P2
of SURFACE_PLAN section 1 say about d = 17. `R` is `<cluster-scratch>/cascade_w5`; times are MDT.

Revised the same day after a review of this file (sound with caveats). The
headline now gives the size of C5's gain and the scope of the claim; the speed ratios carry their clock
qualifications; the tail latency, the per-p thresholds, the power labels and the batch size of BQ1 and BQ5 are
corrected (the review's findings 4-12). Section 3.1 is new: a clean repeat (job e7r) of the eight e7 timings whose
windows fell below the clock rule, with exact windows.

| name | what it is (SURFACE_PLAN 4.1-4.2) |
|---|---|
| S | ConvLines `d13_b5_r775`, the release's network, on the release's Triton engine (fp16 matmul inputs, fp32 accumulation) |
| B | Cascade-surface: our reproduction of Cascade at d = 13 with H = 256 (CS13 `final_ema.pt`, sha256 5351b27b...), bf16 |
| M | correlated PyMatching 2.4.0 on the node's CPU (AMD EPYC 9555), one thread per decode |
| C1 / C2 / C3 / C4 / C5 | S -> B -> M / B alone ("Cascade alone", the opponent of every verdict) / M alone / S -> M (the registered configuration) / S -> B |
| single, batched | the two modes: one shot at a time (mean latency per shot), or a stream of shots in batches (time per shot) |
| p | the circuit's physical error rate: 0.6 % (the registered regime) and p_low = 0.35 % |

The labels (SURFACE_PLAN 4.5) come from one Holm procedure (a step-down correction for many tests) over 40 hypotheses at
alpha = 0.05, ten per p and mode. X is "better" (or "worse") than Y when the paired exact McNemar test (on the shots
where exactly one of the two fails) is rejected and X fails alone on fewer (or more) of those shots than Y (EVAL_READY
5.2). In all 24 rejected tests of this kind, Newcombe's paired interval for P(X) - P(Y) at the Holm level, the plan's
first rule, lies on the same side of zero. X is "non-inferior" when the test against the margin delta_p is rejected (X
fails at most 10 % more often than Cascade alone) and X is not "worse". "Level" means neither direction was shown; it
does not mean equal.

Sources, copied unchanged from `R/eval` into `surface/out/` (sha256 checked against Alpine): `confirm_d13.json` (the
one analysis, `sv_route confirm`), `bitwise_c7.json` (the bitwise check of the pipelines, with each process's timing),
`SURFACE_RESULT_draft.md` and its `.json` (`sv_route report`), `pipes.jsonl` (the pipelines' timing rows); `speed.json`
and `locked_d13.json` were committed with the lock (bddd3f9). The clock flags `clockflags_e1.jsonl` and
`clockflags_e7.jsonl` come from `scripts/clockflags.py` over the samples in `R/clocks/`. `scripts/results_tables.py
out` prints the tables of sections 2 and 3.

## 1. Verdicts

| p | chain | accuracy against C2, single / batched | mean time per shot, chain vs C2, single | batched | bitwise check | the plan's verdict label (SURFACE_PLAN 7) | plain meaning |
|---|---|---|---|---|---|---|---|
| 0.6 % | C1 | better / better | 229.0 vs 397.0 µs | 72.8 vs 136.0 µs | pass | faster and more accurate | fails 12 % less often than Cascade alone; on average 0.58 (single) and 0.54 (batched) of its time per shot; slower in the tail |
| 0.6 % | C5 | better / better | 107.4 vs 397.0 µs | 30.4 vs 136.0 µs | pass | faster and more accurate | fails 0.54 % less often (27 fewer failures in 1,000,000 shots); on average 0.27 and 0.22 of the time; slower in the tail |
| 0.35 % | C1 | level / non-inferior | 312.7 vs 397.1 µs | 100.8 vs 136.0 µs | pass | no claim | faster on average in both modes, but its accuracy within the margin is not shown in single mode |
| 0.35 % | C5 | better / better | 69.4 vs 397.1 µs | 17.8 vs 136.0 µs | pass | faster and more accurate | fails 1.4 % (single) and 1.5 % (batched) less often (33 and 35 fewer in 30,000,000); on average 0.17 and 0.13 of the time; slower in the tail |

Both chains are faster on average and slower in the tail. In single mode at both p their p99.9 latency (the slowest
0.1 % of shots) is 870 µs (C1) and 490 µs (C5) at 0.6 %, and 695 µs and 490 µs at 0.35 %, against 411 µs and 417 µs
for Cascade alone: a shot that S sends on pays for S and B in sequence. So these results do not show that a chain
meets a per-shot deadline better than Cascade alone. Batched mode gives only the time per shot of a stream.

The headline of SURFACE_PLAN 7 holds through C5, in this form: ConvLines with Cascade behind it (C5) is faster on
average and slightly more accurate than Cascade alone on the d = 13 surface code. C5 meets the plan's three conditions
(accuracy, speed, bitwise check) at both p, with "better" in both modes at both p. The two modes read the same
confirmation shots, so they are two analyses of one sample, not independent replications. Its planned power for
non-inferiority, joint over the two p, was 1.0 (C1's 0.92). The planned power of the "better" test was lower: 0.342
for C5 at 0.6 % and 0.999 at 0.35 % (`planned_power`, `S:C5-C2`, in `locked_d13.json`). The test was rejected at
0.6 % all the same, so the label stands; the plan had given it about a one-in-three chance there.

The claim holds only in this setting. The opponent is Cascade-surface, our reproduction of Cascade at d = 13 with a
hidden width H = 256 that we inferred (SURFACE_PLAN 3.2). It is not the authors' implementation, and nothing here
compares C5 with the authors' Cascade. The circuits are the release's legacy X-basis memory circuits
(`d13_legacy.stim`, SURFACE_PLAN 2.1) with one noise parameter p: depolarizing noise on idle qubits and after each CX
layer, bit or phase flips at 2p/3 after resets and before measurements, and no noise in the last syndrome round or the
data readout. No other noise model was tested. The times come from one NVIDIA RTX PRO 6000 Blackwell Server Edition
(driver 590.48.01, torch 2.11.0+cu128 in bb's `venv_cu128`), with bf16 for Cascade, fp16 inputs and fp32 accumulation
for ConvLines' engine, and PyMatching on the node's CPU.

The routing thresholds were fitted at each p separately, on that p's development block, and locked before the first
confirmation shot. C5 sends on to B the 10 % of shots where S is least sure at 0.6 % and 1 % at 0.35 %; C1 sends 40 %
to B and 0.5 % on to M at 0.6 %, and 60 % and 0.1 % at 0.35 %. The separate, locked confirmation keeps the tests valid,
but no single threshold serves both p. C1's grid always sends a share of at least 0.1 % on to M, so its result at
0.35 % does not rule out a better routing policy.

C5's accuracy gain is small. It has 27 fewer failures than Cascade alone in 1,000,000 shots at 0.6 % (4,936 against
4,963 in single mode, 4,926 against 4,953 batched: 0.54 % and 0.55 % fewer), and 33 (single) or 35 (batched) fewer in
30,000,000 shots at 0.35 % (1.37 % and 1.45 % fewer). Each difference is a shot that S decided itself: 40 shots that B
would have failed and S got right, against 13 the other way, at 0.6 %; 33 or 35 against none at 0.35 %. Its speed gain
is large because S alone decides 90 % and 99 % of the shots. As measured on this card, Cascade alone takes 3.70 (single)
and 4.48 (batched) times C5's mean time per shot at 0.6 %, and 5.72 and 7.66 times at 0.35 %; for C1 at 0.6 % the
factors are 1.73 and 1.87. All the single-mode timings behind these factors met the clock rule over their reconstructed
windows (section 3). The batched factors divide C2's batched time, whose six e7 windows fell below the rule in their
first second (section 3). The clean repeat of those windows with exact bounds (section 3.1) met the rule throughout, and
with its C2 the batched factors grow by 0.15 % or less (C5: 4.49 and 7.67).

## 2. Accuracy: labels and counts

The confirmation blocks (seed kind 11, drawn only after the lock commit bddd3f9) hold 1,000,000 shots at 0.6 %
(chunks 0-99, the floor) and 30,000,000 at 0.35 % (chunks 0-2999); both modes read the same shots. The margins
delta_p, 10 % of Cascade alone's development failure rate, are 5.1e-4 at 0.6 % (5,102 failures in 1,000,000
development shots) and 7.5e-6 at 0.35 % (750 in 10,000,000).

### Labels against Cascade alone (C2)

| p | mode | C1 vs C2 | C3 vs C2 | C4 vs C2 | C5 vs C2 | plain meaning |
|---|---|---|---|---|---|---|
| 0.6 % | single | better | worse | worse | better | C1 and C5 fail less often; C3 and C4 fail more often |
| 0.6 % | batched | better | worse | worse | better | C1 and C5 fail less often; C3 and C4 fail more often |
| 0.35 % | single | level | worse | level | better | C5 fails less often; C1 and C4 show no difference; C3 fails more often |
| 0.35 % | batched | non-inferior | worse | level | better | C5 fails less often; C1 stays within the 10 % margin; C4 shows no difference; C3 fails more often |

### Failures on the confirmation block

| p | mode | shots | C1 | C2 | C3 | C4 | C5 | plain meaning |
|---|---|---|---|---|---|---|---|---|
| 0.6 % | single | 1,000,000 | 4,372 | 4,963 | 6,320 | 5,388 | 4,936 | C1 -11.9 %, C3 +27.3 %, C4 +8.6 %, C5 -0.5 % failures against C2 |
| 0.6 % | batched | 1,000,000 | 4,368 | 4,953 | 6,320 | 5,388 | 4,926 | C1 -11.8 %, C3 +27.6 %, C4 +8.8 %, C5 -0.5 % failures against C2 |
| 0.35 % | single | 30,000,000 | 2,506 | 2,406 | 2,909 | 2,502 | 2,373 | C1 +4.2 %, C3 +20.9 %, C4 +4.0 %, C5 -1.4 % failures against C2 |
| 0.35 % | batched | 30,000,000 | 2,499 | 2,411 | 2,909 | 2,502 | 2,376 | C1 +3.6 %, C3 +20.7 %, C4 +3.8 %, C5 -1.5 % failures against C2 |

Plain PyMatching, context only: 21,061 failures at 0.6 % and 18,936 at 0.35 %, on the same shots.

What reaches each stage (the same in both modes to within 0.001 percentage points): C1 sends 40.0 % of the shots to B
and 0.49 % to M at 0.6 %, 60.0 % and 0.10 % at 0.35 %; C5 sends 9.96 % and 1.00 % to B; C4 sends 1.94 % and 0.029 %
to M (the registered threshold 1.849 at both p).

### Each chain and baseline against Cascade alone (paired on the same shots)

| p | mode | X | X fails, C2 not | C2 fails, X not | McNemar p | Holm level | P(X) - P(C2), Newcombe 95 % | non-inferiority p (margin delta_p) | label | plain meaning |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.6 % | single | C1 | 661 | 1,252 | 4.03e-42 | 0.00179 | (-0.000677, -0.000506) | 7.27e-119 (rejected; delta_p 0.00051) | better | fails less often than Cascade alone (shown) |
| 0.6 % | single | C5 | 13 | 40 | 0.000269 | 0.00357 | (-4.19e-05, -1.23e-05) | 1.56e-115 (rejected; delta_p 0.00051) | better | fails less often than Cascade alone (shown) |
| 0.6 % | single | C4 | 2,415 | 1,990 | 1.62e-10 | 0.00313 | (0.000295, 0.000555) | - | worse | fails more often than Cascade alone (shown) |
| 0.6 % | single | C3 | 3,886 | 2,529 | 9.09e-65 | 0.00152 | (0.0012, 0.00151) | - | worse | fails more often than Cascade alone (shown) |
| 0.6 % | batched | C1 | 660 | 1,245 | 1.84e-41 | 0.00185 | (-0.000671, -0.0005) | 4.46e-118 (rejected; delta_p 0.00051) | better | fails less often than Cascade alone (shown) |
| 0.6 % | batched | C5 | 13 | 40 | 0.000269 | 0.00385 | (-4.19e-05, -1.23e-05) | 1.57e-115 (rejected; delta_p 0.00051) | better | fails less often than Cascade alone (shown) |
| 0.6 % | batched | C4 | 2,413 | 1,978 | 5.57e-11 | 0.00278 | (0.000305, 0.000565) | - | worse | fails more often than Cascade alone (shown) |
| 0.6 % | batched | C3 | 3,884 | 2,517 | 7.55e-66 | 0.00147 | (0.00121, 0.00152) | - | worse | fails more often than Cascade alone (shown) |
| 0.35 % | single | C1 | 1,405 | 1,305 | 0.0572 | not reached | (-6.83e-08, 6.74e-06) | 0.00826 (not rejected; delta_p 7.5e-06) | level | no difference shown either way |
| 0.35 % | single | C5 | 0 | 33 | 2.33e-10 | 0.00333 | (-1.5e-06, -7.11e-07) | 5.84e-56 (rejected; delta_p 7.5e-06) | better | fails less often than Cascade alone (shown) |
| 0.35 % | single | C4 | 1,046 | 950 | 0.0334 | not reached | (2.81e-07, 6.12e-06) | - | level | no difference shown either way |
| 0.35 % | single | C3 | 1,824 | 1,321 | 2.95e-19 | 0.00227 | (1.31e-05, 2.04e-05) | - | worse | fails more often than Cascade alone (shown) |
| 0.35 % | batched | C1 | 1,397 | 1,309 | 0.0944 | not reached | (-4.66e-07, 6.34e-06) | 0.00429 (rejected; delta_p 7.5e-06) | non-inferior | at most 10 % more failures than Cascade alone (shown) |
| 0.35 % | batched | C5 | 0 | 35 | 5.82e-11 | 0.00294 | (-1.58e-06, -7.67e-07) | 2.96e-56 (rejected; delta_p 7.5e-06) | better | fails less often than Cascade alone (shown) |
| 0.35 % | batched | C4 | 1,045 | 954 | 0.0441 | not reached | (1.12e-07, 5.96e-06) | - | level | no difference shown either way |
| 0.35 % | batched | C3 | 1,822 | 1,324 | 6.73e-19 | 0.00238 | (1.29e-05, 2.03e-05) | - | worse | fails more often than Cascade alone (shown) |

The "Holm level" is the level at which the step-down procedure tested that hypothesis. 31 of the 40 hypotheses were
rejected. The nine that were not are all at 0.35 %: the difference tests of C1-C2, C4-C2, C1-C4 and C1-C5 in both
modes, and the non-inferiority of C1 in single mode (p 0.0083 against its step's level 0.05 / 9 = 0.0056), where the
procedure stopped.

### The other comparisons of the family

| p | mode | C1 vs C4 | C5 vs C4 | C1 vs C5 | C4 vs C3 | plain meaning |
|---|---|---|---|---|---|---|
| 0.6 % | single | better | better | better | better | C1 fails less often than C4; C5 fails less often than C4; C1 fails less often than C5; C4 fails less often than C3 |
| 0.6 % | batched | better | better | better | better | C1 fails less often than C4; C5 fails less often than C4; C1 fails less often than C5; C4 fails less often than C3 |
| 0.35 % | single | level | better | level | better | C1 shows no difference from C4; C5 fails less often than C4; C1 shows no difference from C5; C4 fails less often than C3 |
| 0.35 % | batched | level | better | level | better | C1 shows no difference from C4; C5 fails less often than C4; C1 shows no difference from C5; C4 fails less often than C3 |

Repairs and regressions (SURFACE_PLAN 4.5):
- C1's third stage, M on the shots where B is least confident: at 0.6 % it repairs 1,244 of B's failures and makes 661
  new ones (batched: 1,237 and 660); at 0.35 % it repairs 1,298 and makes 1,405 (batched: 1,302 and 1,397). PM behind
  Cascade helps at 0.6 % and costs about 100 failures at 0.35 %, which is why C1 is only "level" or "non-inferior"
  there.
- The shots S keeps: for C5, B's failures that S repairs and S's failures that B would not have made are 40 and 13 at
  0.6 %, and 33 (batched 35) and 0 at 0.35 %. For C1 they are 8 and 0 at 0.6 %, 7 and 0 at 0.35 %.

Computations that agree bitwise on every confirmation shot (SURFACE_PLAN 4.4): S at batch 1 and S at 4,096 (S1, SB:
no logit differs); M one shot per call and M batched (MS, MB: no decision differs); B on the routed queues and B on
all shots, both at its batched size b* = 16 (BQ1 and BQ5 against Bb: no logit differs; SURFACE_PLAN 4.4's table had
4,096 for BQ1 and BQ5, amended to b* in EVAL_READY 5.1 and fixed so in the lock). B at batch 1 (B1) and Bb differ in 48
decisions at 0.6 % and 29 at 0.35 %, as bf16 tiles depend on the batch size; each mode uses its own outputs, which is
why C2's count differs between the modes (4,963 and 4,953; 2,406 and 2,411).

## 3. Speed, with the clock flags of the timed windows

e7 (AJ34, on gpu-node-08) timed C1, C4, C5 and Cascade alone (C2) at both p in both modes, three fresh processes
each, in one rep-major loop in one job: in single mode 20,000 shots after 200 untimed ones, in batched mode 500,000
shots (SURFACE_PLAN 5). Each value is the slowest process's mean. C2 runs B on its frozen paths (`compiled@1` in single
mode; `compiled@16` batched, b* = 16).

A timing's clock flag follows phase C's `gpu_watch` rule, applied afterwards by `scripts/clockflags.py` to the samples
in `R/clocks/sv_e7_AJ34_*`: the timing is accepted when the lowest SM-clock sample in its own window is at least 97
% of the card's maximum of 2,430 MHz (2,357 MHz) and no other process is on the card. e7's windows are reconstructed
from the timing rows. A batched window is the timed part, from the row's time and wall time, plus 1 s because the row
time has whole seconds. A single-mode window is n times the mean latency before the row time, plus the same 1 s; it
leaves out the elapsed time between shots (the Python work outside the timer), so the start of a single-mode timing can
fall before its window. The row is written after the timer stops, and after the kernel-cache digest and the output file,
so any reconstructed window can take in samples from just before the timed part or miss part of its first second.

### Speed: each chain against Cascade alone timed in the same job (e7)

| p | mode | chain | chain, us per shot | C2, us per shot | C2 / chain | p99.9 latency, chain / C2 (us) | clock flags, chain | clock flags, C2 | plain meaning |
|---|---|---|---|---|---|---|---|---|---|
| 0.6 % | single | C1 | 229.0 | 397.0 | 1.73 | 870 / 411 | all 3 ok | all 3 ok | C1 takes 0.58 of C2's time per shot; slower in the tail |
| 0.6 % | single | C5 | 107.4 | 397.0 | 3.70 | 490 / 411 | all 3 ok | all 3 ok | C5 takes 0.27 of C2's time per shot; slower in the tail |
| 0.6 % | single | C4 | 67.7 | 397.0 | 5.87 | 455 / 411 | all 3 ok | all 3 ok | C4 takes 0.17 of C2's time per shot; slower in the tail |
| 0.6 % | batched | C1 | 72.8 | 136.0 | 1.87 | - | all 3 ok | below 97 %: rep 1, rep 2, rep 3 (value from rep 1) | C1 takes 0.54 of C2's time per shot |
| 0.6 % | batched | C5 | 30.4 | 136.0 | 4.48 | - | all 3 ok | below 97 %: rep 1, rep 2, rep 3 (value from rep 1) | C5 takes 0.22 of C2's time per shot |
| 0.6 % | batched | C4 | 16.4 | 136.0 | 8.32 | - | all 3 ok | below 97 %: rep 1, rep 2, rep 3 (value from rep 1) | C4 takes 0.12 of C2's time per shot |
| 0.35 % | single | C1 | 312.7 | 397.1 | 1.27 | 695 / 417 | all 3 ok | all 3 ok | C1 takes 0.79 of C2's time per shot; slower in the tail |
| 0.35 % | single | C5 | 69.4 | 397.1 | 5.72 | 490 / 417 | all 3 ok | all 3 ok | C5 takes 0.17 of C2's time per shot; slower in the tail |
| 0.35 % | single | C4 | 65.3 | 397.1 | 6.08 | 76 / 417 | below 97 %: rep 2, rep 3 (value from rep 3) | all 3 ok | C4 takes 0.16 of C2's time per shot |
| 0.35 % | batched | C1 | 100.8 | 136.0 | 1.35 | - | all 3 ok | below 97 %: rep 1, rep 2, rep 3 (value from rep 3) | C1 takes 0.74 of C2's time per shot |
| 0.35 % | batched | C5 | 17.8 | 136.0 | 7.66 | - | all 3 ok | below 97 %: rep 1, rep 2, rep 3 (value from rep 3) | C5 takes 0.13 of C2's time per shot |
| 0.35 % | batched | C4 | 16.3 | 136.0 | 8.32 | - | all 3 ok | below 97 %: rep 1, rep 2, rep 3 (value from rep 3) | C4 takes 0.12 of C2's time per shot |

C4 is in the table for context; the verdicts compare C1 and C5 with C2. 40 of e7's 48 timings were accepted. No
window had another process on the card: 11,659 clock samples at 100 ms over 11:06:06-11:25:56, and the process list,
read every 2 s, never showed more than one process. Eight windows fall below 97 %:
- all six C2 batched windows (three per p): 2 or 3 of their 674-675 samples at 2,340 or 2,355 MHz (96.3 % or 96.9 %),
  all within the first 1.1 s of the window, while the card's power rose from about 130 W to 240 W (it runs at about
  490 W once loaded). No clock-event reason was active (`0x0`): the clock was still ramping up from idle as the work
  started.
- C4 single at 0.35 %, reps 2 and 3: one sample each, 0.1 s into a 2.2-2.3 s window (1,987 and 2,235 MHz, 82 % and
  92 %, at about 70 W). The window's first second can precede the timed shots.

A slow clock can only slow the timing it falls in. A dip in C2's batched timing would make C2 look slower and so
overstate each chain's batched factor; a dip in C4's single timing would understate C4's. The samples, 100 ms apart over
reconstructed windows, cannot say how much of a timing ran below 97 %, so these eight flags stay exceptions and the
eight timings are not accepted under the protocol on e7's evidence. Two facts limit what they could change. The three C2
batched processes at each p agree to 0.01 % (136.03-136.04 µs per shot at 0.6 %, 135.99-136.00 µs at 0.35 %), and the
smallest batched factor of a chain is 1.35 (C1 at 0.35 %, no claim); C5's are 4.48 and 7.66. Every reconstructed
single-mode window of C1, C5 and C2 met the rule. If a dip hidden before one of them had slowed C2's slowest single-mode
process (397.0 µs at 0.6 %, 397.1 µs at 0.35 %), C2's fastest (384.7 µs at both p) would still be slower than every
chain, the closest being C1 at 0.35 % (312.7 µs); a dip in a chain's own timing would only understate its lead. C4's
single value at 0.35 % (65.3 µs) comes from flagged rep 3; its accepted rep 1 gave 60.8 µs. Section 3.1 re-timed the
eight timings with exact windows.

The decoders alone (e1, `speed.json`): S takes 63.6 µs single (from rep 1, whose window ran at 95.3 % of the clock
and is flagged; 63.18 µs without it, see below) and 16.2 µs per shot at batch 4,096; B takes 408 µs
single (`compiled@1`) and 139 µs per shot at b* = 16; M takes 180 µs (0.6 %) and 84 µs (0.35 %) per single-threaded
decode, and 12.3 µs and 5.9 µs per shot over 16 workers (CPU work, to which the GPU clock does not apply).

### e1's flag on S's single time

53 of e1's 54 GPU timings held at least 97 % of 2,430 MHz. The one that did not is S `engine@1` in rep 1 (57 of its 60
samples at 2,317 MHz, 95.3 %), and as the slowest of S's three single-shot processes it supplies `speed.json`'s t1 of
S, 63.6 µs. Its GPU replay took 47.1 µs per shot against 45.1 µs in reps 2 and 3 (4.5 % slower, in line with its
clock), and its host round trip 63.65 µs against 63.18 µs and 60.33 µs; without it t1 would be 63.18 µs, 0.47 µs
(0.7 %) less. t1(S) enters only the lock's predicted costs, which rank the threshold candidates, and the line on the
decoders alone above. It could not have changed the lock: at each p, the chosen thresholds of C1 and of C5 have the
fewest development failures of all their candidates, feasible or not, and their predicted single-shot costs lie
58.7 µs (C1 at 0.35 %) to 299.6 µs below the cap of 0.9 x t1(B) = 367.3 µs. No verdict uses it: the verdicts rest
on e7's end-to-end timings.

### 3.1 Clean repeat of the eight flagged timings (e7r)

e7r (job AJ36, on gpu-node-06, 12:20:00-12:27:54) re-timed only the eight timings that e7 flagged: C2 batched at
both p, reps 1-3, and C4 single at 0.35 %, reps 2 and 3, in e7's rep-major order. It ran the pinned tree
`R/code.20261012_060258` with the lock and its kernel caches in `R/eval`, on one full RTX PRO 6000 with e7's clock
sampler, and wrote to `R/eval_retime`, apart from the evaluation's records. Each pipeline ran unchanged under
`scripts/winmark.py` (copy `R/retime/winmark.py`, sha256 24ab0381...), which records the exact bounds of the timed part.
The wrapper sits outside the tree, so the code digest each pipeline compares with the lock's stays the lock's. Python's
`sys.monitoring` reports each line of the pipeline function once, and at three lines the wrapper reads the timer values
the pipeline itself took. A single-mode window runs from the start of the first timed shot to the end of the last, so it
includes the time between shots; a batched window starts with the timer and lasts `wall_s`. Each line event is removed
after its first report, so the timed loop runs as in e7. `scripts/clockflags.py` takes the windows from the wrapper's
records when given them: `out/marks_e7r.jsonl`, with the timing rows `out/pipes_e7r.jsonl` and the flags
`out/clockflags_e7r.jsonl` (run in the cluster's time zone, `TZ=America/Denver`, as the samples carry local time).

All eight timings meet the 97 % rule over their exact windows. The sampler took 4,562 samples at 100 ms over
12:20:04-12:27:50, and the process list never showed more than one process.

| p | mode | config | rep | e7: µs per shot, lowest clock in its reconstructed window | e7r: µs per shot | e7r: exact window, start and length | e7r: samples, lowest clock | e7r flag | plain meaning |
|---|---|---|---|---|---|---|---|---|---|
| 0.6 % | batched | C2 | 1 | 136.04, 2,355 MHz (96.9 %) | 136.17 | 12:20:18.758, 68.086 s | 666, 2,385 MHz (98.2 %) | ok | meets the rule; 0.10 % slower than in e7 |
| 0.35 % | batched | C2 | 1 | 135.99, 2,355 MHz (96.9 %) | 136.20 | 12:21:33.535, 68.102 s | 665, 2,385 MHz (98.2 %) | ok | meets the rule; 0.16 % slower than in e7 |
| 0.6 % | batched | C2 | 2 | 136.04, 2,355 MHz (96.9 %) | 136.20 | 12:22:48.500, 68.100 s | 665, 2,385 MHz (98.2 %) | ok | meets the rule; 0.12 % slower than in e7 |
| 0.35 % | single | C4 | 2 | 59.33, 1,987 MHz (81.8 %) | 59.10 | 12:24:00.484, 1.189 s | 12, 2,422 MHz (99.7 %) | ok | meets the rule; 0.40 % faster than in e7 |
| 0.35 % | batched | C2 | 2 | 135.99, 2,340 MHz (96.3 %) | 136.15 | 12:24:07.629, 68.075 s | 665, 2,385 MHz (98.2 %) | ok | meets the rule; 0.11 % slower than in e7 |
| 0.6 % | batched | C2 | 3 | 136.03, 2,340 MHz (96.3 %) | 136.21 | 12:25:22.644, 68.106 s | 666, 2,385 MHz (98.2 %) | ok | meets the rule; 0.13 % slower than in e7 |
| 0.35 % | single | C4 | 3 | 65.33, 2,235 MHz (92.0 %) | 62.60 | 12:26:34.546, 1.260 s | 13, 2,415 MHz (99.4 %) | ok | meets the rule; 4.18 % faster than in e7 |
| 0.35 % | batched | C2 | 3 | 136.00, 2,340 MHz (96.3 %) | 136.08 | 12:26:41.665, 68.042 s | 665, 2,385 MHz (98.2 %) | ok | meets the rule; 0.06 % slower than in e7 |

Each e7r pipeline failed on the same number of shots as its e7 twin (2,568 of 500,000 for C2 at 0.6 %, 34 at 0.35 %, 1
of 20,000 for C4), as the same code on the same shots should.

With e7r's slowest C2 batched process (136.21 µs per shot at 0.6 %, 136.20 µs at 0.35 %) in place of e7's (136.04 and
136.00 µs), every batched factor of section 3 grows by 0.12 % at 0.6 % and 0.15 % at 0.35 %. At two decimals C1's stay
1.87 and 1.35, C5's become 4.49 (from 4.48) and 7.67 (from 7.66), and C4's 8.33 (from 8.32) at both p. These factors
take C2 from e7r's card and the chains from e7's (gpu-node-08), so a change of this size can come from the card as much
as from the clock. For C4 single at 0.35 %, e7r's reps 2 and 3 took 59.10 and 62.60 µs (e7: 59.33 and 65.33 µs). With
e7's accepted rep 1 (60.83 µs), the slowest of the three is 62.60 µs, and Cascade alone takes 6.34 times C4's time
instead of 6.08. No verdict changes.

The exact windows also show what the reconstruction does. Applied to e7r's rows, the rule of section 3 would have opened
each window 0.60-0.85 s before the timed part and closed it 0.15-0.40 s after. The only samples below 97 % in those
reconstructed windows, two at 1,485 MHz in C4's rep 3, lie 0.66-0.76 s before its timed part, while the clock came back
from a short drop to 480 MHz about a second earlier. e7's dips also lay in the first 1.1 s of their reconstructed
windows, so they may have come before the timed part, but e7's samples cannot show it. The time between shots that the
single-mode reconstruction leaves out was 7-8 ms in 1.19-1.26 s for C4 here (0.6 %).

## 4. Checks

- e8's bitwise check of the pipelines (`bitwise --expect c7`, `out/bitwise_c7.json`) passed in both modes. e8 re-ran
  S1, B1 and MS on the single-mode pipelines' shots and SB, Bb, BQ1, BQ5 and MB on the batched ones (seed kind 12,
  chunks 0-1 and 0-49, both p). All 48 pipeline processes match these reads bitwise: no decision, stage assignment, S
  or B logit or M decision differs, and no kernel-cache digest (the pipeline's at its start and end, the reads') differs
  from the lock's.
- Each confirmation read completed in one pass (no resume): e5 (SB, Bb, BQ1, BQ5, MB and MP, plain PM) in 234 s at
  0.6 % and 7,356 s at 0.35 %; e6 (S1, B1, MS) in 163 s and 4,742 s.
- `confirm` (`out/confirm_d13.json`) ran on lock 5767020407ed..., gate 7ba1fa22... and lock commit bddd3f9. `report`
  wrote `out/SURFACE_RESULT_draft.md` and its `.json`, whose verdicts this file restates.
- Every step of e5-e8 returned 0, including all 48 pipelines of e7.

## 5. P1, P2 and d = 17

P1 holds: Cascade-surface alone is "better" than correlated PM alone at both p in both modes. C3's label against C2
is "worse" in all four cases: PM alone fails 27 % more often than Cascade alone at 0.6 % (6,320 against 4,963 and
4,953) and 21 % more often at 0.35 % (2,909 against 2,406 and 2,411), with McNemar p of 7e-19 or less.

P2 holds through C5 alone. It asks that one and the same chain be "faster within the predefined accuracy tolerance"
or "faster and more accurate" than Cascade alone at 0.6 % (a), and "better" or "non-inferior" in both modes at 0.35 %
(b). C5 is "faster and more accurate" at 0.6 % and "better" in both modes at 0.35 %. C1 meets (a) but not (b), being
"level" in single mode at 0.35 %.

So the plan's rule proposes d = 17 as the second step (`report`: `"d17": "propose"`). By the plan's figures its
training alone takes about 137 RTX PRO 6000-hours (2.8 times d = 13 per sample; the pilot measured 540 samples/s at
d = 17), and ConvLines at d = 17 runs only through the release's deployed engine, since d = 17 ships as deploy tables.
It was deferred (12 October). Two facts from d = 13 bear on it: the chain that passed, C5, wins on
accuracy by a small margin (27 failures in 1,000,000 shots at 0.6 %, 33-35 in 30,000,000 at 0.35 %) and on mean time
per shot by factors of 3.70 to 7.66 as measured; and C1's third stage, PM behind Cascade, helped at 0.6 % but not at
0.35 %.

## 6. Jobs and GPU time

| job id | job | node | ran (MDT) | wall time | what it did |
|---|---|---|---|---|---|
| AJ32 | sv_e5 | gpu-node-08 | 08:59:14-11:06:02 | 2:06:48 | the batched confirmation reads, both p |
| AJ33 | sv_e6 | gpu-node-06 | 09:03:57-10:26:00 | 1:22:03 | the single confirmation reads, both p |
| AJ34 | sv_e7 | gpu-node-08 | 11:06:02-11:25:59 | 0:19:57 | the 48 pipelines (C7), with the clock sampler |
| AJ35 | sv_e8 | gpu-node-08 | 11:29:03-11:34:02 | 0:04:59 | the reads on the pipelines' shots, bitwise, confirm, report |
| AJ36 | sv_e7r | gpu-node-06 | 12:20:00-12:27:54 | 0:07:54 | the clean repeat of the eight flagged timings, exact windows (section 3.1) |

e1 to e8 used 5.2 full-card GPU-hours together (e1 0:38:44, e2 0:29:34, e3 0:06:54, e4 0:04:44 and e5-e8 above), against
the plan's 3.3 at the confirmation floor and 8.4 at the caps, plus the 52 s of the void first e1; e7r added 0.13 (7 min
54 s). Every job ran tree `R/code.20261012_060258` (a431032 with release 8d65844, clean) with `--nice=1000`, on at most
two cards at a time.

Not written here: SURFACE_PLAN 7's full report, `cascade_compare/SURFACE_RESULT.md` (the setting and its fidelity, the
training run, the full speed tables); `out/SURFACE_RESULT_draft.md` holds its generated tables.
