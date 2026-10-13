# gross2, stage 1: results (exploratory)

## 1. Scope and sources

2026-10-07 (stage 1 ended 01:27 PDT, exit 0, 3.80 pod-hours; keeper verdict complete). Exploratory.
Gross's registered probe (PLAN.md 2.3) stays FAILED: r21 alone failed 7,403 of the 100,000 cal shots, sector Relay-BP
490. Nothing here is registrable; any later claim needs a fresh registration and an untouched test set.

The pod ran `ising_compare` at commit 5515f709 (5515f70914aef53ea7ec631687b57384f20d1a7a, branch `ising-bb-gross2`) on
one RunPod H100 80GB HBM3 (session start 2026-10-06 21:39 PDT, driver start 21:43:49 PDT, no tuning flags, 20 sampler
processes). Plan: `PLAN_gross2.md` (rules in its section 4). Sources, all
under `~/work/stage_gross2/pod_out/` (the Linux workstation's copy of the pod's `/workspace/g2`): `out/selection_a.json`,
`out/pair_*.json`, `out/s0.json`, `out/runs_bb.csv`, `logs/run_all.log`, `logs/g2_*.log` ("eval" lines), `preds/`
(per-shot reads), `ckpt/*/log.json`.

Terms:
- Shot: one simulated memory experiment of the bb144 (gross) code; it fails if the decoded logical flips differ from
  the true ones. LER = failures / shots; pp = percentage points of LER. EMA = exponential moving average of the weights.
- D-bank: 50,000 shots (monitor chunks 10-14, p = 0.004), sampled once on the pod; every decision reads it. CAL: the
  100,000 cal shots (chunks 0-9, p = 0.004), used for the frontier only. Relay-BP = sector Relay-BP, the CPU fallback.
- Reads: fp32 = PyTorch fp32 batched; graph32 = the deployed engine (CUDA graph, fp32, batch 1); monitor = the EMA's
  bf16 evaluation on the D-bank's first 20,000 shots during training.
- Interval = Newcombe 95 % interval of LER(a) - LER(b) on the same shots; McNemar p = exact paired test on the shots
  where the two reads differ ("a only" = only a fails).

## 2. Tier A (warm start from r21, 19,996,672 samples per arm, run id 31, p = 0.004, no ramp)

| arm | recipe | MACs/shot, params | failures / 50,000 (fp32) | LER |
|---|---|---|---|---|
| r21 | the starting network (1e8 samples) | 486.6 M, 821,085 | 3,671 | 7.342 % |
| A0F (F) | lr 2e-5 flat | 486.6 M, 821,085 | 3,650 | 7.300 % |
| A1S (S) | lr 3e-4 held to 0.8 of the run, then cosine to 2 % | 486.6 M, 821,085 | 3,516 | 7.032 % |
| A2T (T) | S + time blocks 4, 8 | 507.7 M, 844,509 | 3,116 | 6.232 % |
| A3D (D6) | S + 6 blocks (blocks 4-6 new, zero-initialised up-projections) | 759.7 M, 1,455,261 | 1,344 | 2.688 % |

Every graph32 read equals its fp32 batched read (0 of 50,000 decisions differ) for r21 and the four arms. A4 (R) did not
run: its rule needs neither T nor D6 to qualify.

| comparison | a only / b only | fewer failures | LER(a) - LER(b), paired 95 % | McNemar p | rule | plain meaning |
|---|---|---|---|---|---|---|
| S vs F | 283 / 417 | 3.7 % | -0.268 pp [-0.372, -0.164] | 4.6e-7 | >= 25 %, diff_hi < 0: no | the schedule gain is real but small |
| T vs S | 232 / 632 | 11.4 % | -0.800 pp [-0.916, -0.685] | 1.6e-43 | >= 25 %, diff_hi < 0: no | the time blocks help, below the bar |
| D6 vs S | 80 / 2,252 | 61.8 % | -4.344 pp [-4.532, -4.161] | < 1e-300 | >= 25 %, diff_hi < 0, GPU term <= 5 ms: yes | D6 fixes 2,252 shots S fails and breaks 80 |
| D6 vs T | 87 / 1,859 | 56.9 % | -3.544 pp [-3.716, -3.376] | < 1e-300 | >= 20 %: yes | D6 preferred over T |

D6's GPU term is 1.79 ms (cap 5 ms). Against r21 itself (no paired file), A3D has 63.4 % fewer failures (1,344 vs
3,671). Decision (`out/selection_a.json`): schedule lever false; T false, D6 true; winner g2_A3D, control g2_A1S,
stage-2 shape D6; decision pass; undetermined "tier B pending" (tier A's selection never says kill).

Monitor curves (EMA, bf16, failures of the first 20,000 D-bank shots; divide by 200 for %). r21's own fp32 read on the
same 20,000 shots: 1,503.

| arm | 5.0e6 samples | 1.0e7 | 1.5e7 | 2.0e7 (final) | plain meaning |
|---|---|---|---|---|---|
| A0F | 1,498 | 1,495 | 1,470 | 1,478 | -1.3 % from 5e6 to 2e7 samples |
| A1S | 1,481 | 1,469 | 1,426 | 1,418 | -4.3 % from 5e6 to 2e7 samples |
| A2T | 1,453 | 1,365 | 1,315 | 1,264 | -13 % from 5e6 to 2e7 samples |
| A3D | 1,040 | 756 | 611 | 540 | -48 % from 5e6 to 2e7 samples; 2.70 % at the end |

Stop rule (a variant at least 25 % worse than its control at both 5e6 and 1e7): no stop. A2T [1,453, 1,365] against
the control S [1,481, 1,469]; A3D [1,040, 756] against [1,481, 1,469].

## 3. Tier B (from scratch, 3e7 samples, p ramp 0.001 -> 0.004 over 0.4 of the run, decay start 0.75, lr 1e-3, run id
41)

| arm | recipe | 5e6 | 1e7 | 1.2e7 | 1.5e7 | 2e7 | 2.5e7 | 3e7 (final) | failures / 50,000 (fp32) |
|---|---|---|---|---|---|---|---|---|---|
| B1 C | L3 (r21's shape) | 16,537 | 7,502 | 5,501 | 3,900 | 2,740 | 2,282 | 2,016 | 4,963 (9.926 %) |
| B2 | D6 from scratch | 14,473 | 4,405 | 2,884 | 1,741 | 1,095 | 836 | 684 | 1,698 (3.396 %) |

Curve cells are EMA monitor failures of 20,000 shots (bf16). Stop rule for B2 against C at 5e6 and 1e7: [14,473,
4,405] against [16,537, 7,502], no stop. C's graph32 and fp32 reads give 4,963 failures each (1 of 50,000 decisions
differs); B2's give 1,698 each (0 differ).

Ramp check: C's EMA fails 27.5 % (5,501 of 20,000) at 12,001,280 samples, the ramp's end; the fallback threshold is
50 %, so no fallback. C from scratch at 3e7 samples (9.926 %) has 1.35x the failures of r21 (7.342 %, 1e8 samples).

Rule: B2 must beat C by >= 1.5x, 2 fail(C) >= 3 fail(B2) in integers: fail(B2) <= 3,308 of 50,000 (LER <= 6.616 %).
B2 fails 1,698: 2.92x fewer than C (65.8 % fewer; B2 only 142, C only 3,407; LER(B2) - LER(C) = -6.53 pp, paired 95 %
[-6.76, -6.31]; McNemar p < 1e-300). B2 qualifies, so D6 is confirmed from scratch and is stage 2's shape.

| comparison | failures | ratio | plain meaning |
|---|---|---|---|
| B2 (D6, scratch, 3e7) vs C (L3, scratch, 3e7) | 1,698 vs 4,963 | 2.92x | depth wins without r21's head start |
| B2 (D6, scratch, 3e7) vs A3D (D6, r21 + 2e7) | 1,698 vs 1,344 | 1.26x | samples, schedule and start differ |
| C (L3, scratch, 3e7) vs r21 (L3, 1e8) | 4,963 vs 3,671 | 1.35x | the same mix for L3 |

Final selection (`out/selection.json`): winner g2_A3D, control g2_A1S, stage-2 shape D6, B2 qualifies, decision pass,
nothing undetermined, no caveats. The session ended with exit 0 after 3.80 pod-hours; the keeper's verdict: complete,
176 recorded artifacts and the D-bank's 7 shot files matching their sha256.

## 4. p = 0.003 diagnostic (diagnostic only, never a decision input)

Source: the driver's `out/p003_summary.json` (fp32 reads, tag RUN_f: failures at 0.004 over D-bank chunks 10-14, 50,000
shots, and at 0.003 over chunks 10-11, 20,000 shots of a separate bank; Clopper-Pearson 95 % intervals). An independent
recount of r21, A3D and B1C from the per-shot reads gives the same counts. slope = ln(LER4 / LER3) / ln(4/3); its range
combines the two intervals; d_eff ~ 2 x slope.

| network | fails at 0.004 / 50k (LER, 95 %) | fails at 0.003 / 20k (LER, 95 %) | LER ratio (times fewer failures at 0.003) | slope | slope range | d_eff |
|---|---|---|---|---|---|---|
| r21 | 3,671 (7.342 %, 7.115-7.574) | 134 (0.670 %, 0.562-0.793) | 10.96 | 8.32 | 7.63-9.04 | 16.6 |
| A0F | 3,650 (7.300 %, 7.073-7.532) | 133 (0.665 %, 0.557-0.788) | 10.98 | 8.33 | 7.63-9.05 | 16.7 |
| A1S | 3,516 (7.032 %, 6.809-7.260) | 130 (0.650 %, 0.543-0.771) | 10.82 | 8.28 | 7.57-9.01 | 16.6 |
| A2T | 3,116 (6.232 %, 6.022-6.447) | 106 (0.530 %, 0.434-0.641) | 11.76 | 8.57 | 7.79-9.38 | 17.1 |
| A3D | 1,344 (2.688 %, 2.548-2.834) | 32 (0.160 %, 0.109-0.226) | 16.80 | 9.81 | 8.42-11.31 | 19.6 |
| B1C | 4,963 (9.926 %, 9.665-10.191) | 210 (1.050 %, 0.913-1.201) | 9.45 | 7.81 | 7.25-8.38 | 15.6 |
| B2 | 1,698 (3.396 %, 3.239-3.558) | 51 (0.255 %, 0.190-0.335) | 13.32 | 9.00 | 7.89-10.19 | 18.0 |

Reading: the slope is the local steepness of LER between two points 1.33x apart in p, for networks trained at 0.004
(tier B's ramp covers 0.001-0.004). The plan's note expected it well below ceil(d_c / 2) <= 6 at p = 0.3-0.4 %; every
value here is above 6. At this p the local slope reflects the typical number of faults in a failing shot, not the
minimal one, so d_eff does not read as a distance. A3D's range (8.42-11.31) overlaps r21's (7.63-9.04): the slopes do
not separate the arms. At 0.003 A3D has 4.2x fewer failures than r21 (32 vs 134 in 20,000 shots), at 0.004 2.7x; B2
from scratch, 2.6x and 2.2x.

## 5. GPU terms and throughputs

| shape | MACs/shot | params | S0 steady samples/s (batch 1,024, bf16, compile) | S0 GPU term, batch 1, graph32 fp32 | plain meaning |
|---|---|---|---|---|---|
| L3 (r21) | 486.6 M | 821,085 | 15,308 | 1.026 ms | the baseline shape |
| T | 507.7 M | 844,509 | 13,572 | 1.100 ms | 1.04x MACs, 1.07x the GPU term |
| D6 | 759.7 M | 1,455,261 | 9,030 | 1.794 ms | 1.56x MACs, 1.77x parameters, 1.75x the GPU term |
| R | 632.7 M | 871,262 | 13,973 | 1.075 ms | measured in S0, not trained |
| W | 810.8 M | 1,593,789 | 7,468 | 1.355 ms | measured in S0, not trained |

The GPU term is one shot's time on the H100 in the deployed engine (random weights, accepted process); the budget is
15.8 ms per shot on average, the driver's cap 5 ms; handoff 0.016-0.019 ms.

| run | samples | wall time | whole-run samples/s | vs S0 steady |
|---|---|---|---|---|
| g2_A0F | 19,996,672 | 1,318 s (0.366 h) | 15,174 | -0.9 % |
| g2_A1S | 19,996,672 | 1,319 s (0.366 h) | 15,156 | -1.0 % |
| g2_A2T | 19,996,672 | 1,488 s (0.413 h) | 13,437 | -1.0 % |
| g2_A3D | 19,996,672 | 2,231 s (0.620 h) | 8,965 | -0.7 % |
| g2_B1C | 29,999,104 | 2,014 s (0.559 h) | 14,897 | -2.7 % |
| g2_B2D6 | 29,999,104 | 3,409 s (0.947 h) | 8,799 | -2.6 % |

Wall time = the run's recorded seconds (the reads after it excluded). D6 takes 1.69x the wall time of S for the same
samples.

## 6. CAL frontier of the selected model (g2_A3D, control g2_A1S, reference r21)

`bb.route frontier` on CAL (100,000 shots at p = 0.004, chunks 0-9). The network decodes every shot; the fraction q of
shots with the lowest margin goes to sector Relay-BP, whose stored predictions and recorded per-shot CPU times count
for those shots. Composite time = GPU term + handoff + Relay-BP's time on the routed shots / 100,000; a point is
admissible if that is <= 15.8 ms. GPU terms (the slowest of 3 accepted processes on the pod's H100, bound to each
network's graph32 read): A3D 1.836 ms, A1S 1.051 ms, r21 1.051 ms; handoff 0.017 ms.

| q (routed) | A3D failures / 100k | A3D time (ms) | Relay-BP per routed shot, A3D (ms) | A1S failures / 100k | A1S time (ms) | plain meaning |
|---|---|---|---|---|---|---|
| 0 | 2,682 | 1.9 | - | 6,946 | 1.1 | the network alone |
| 0.5 % | 2,299 | 9.6 | 1,554 | 6,523 | 6.4 | A3D's best point within 15.8 ms |
| 1 % | 1,923 | 17.5 | 1,570 | 6,114 | 11.2 | A1S's best; A3D over the limit |
| 2 % | 1,244 | 31.3 | 1,471 | 5,299 | 20.6 | |
| 3 % | 727 | 42.6 | 1,358 | 4,498 | 29.6 | |
| 4 % | 482 | 51.9 | 1,251 | 3,723 | 38.4 | A3D's first point below 490: 3.3x the limit |
| 6 % | 429 | 64.9 | 1,050 | 2,339 | 53.0 | |
| 10 % | 435 | 80.3 | 785 | 787 | 74.5 | |
| 20 % | 474 | 100.8 | 495 | 479 | 98.0 | |

- Alone on CAL: A3D 2,682 (2.682 %; D-bank 2.688 %), A1S 6,946 (6.946 %; D-bank 7.032 %), r21 7,403.
- Within 15.8 ms: A3D 2,299 at q = 0.5 % (9.6 ms), A1S 6,114 at q = 1 % (11.2 ms), r21 6,552 at q = 1 % (10.3 ms). For
  all three the screen says `policy_beats_fallback` false and `net_within_3x_fallback` false; status "stop".
- Between the grid's q = 0.5 % and 1 %, a linear interpolation puts A3D's limit near q ~ 0.9 % and ~2,000 failures (an
  estimate; the frozen grid has no point there).
- Relay-BP spends 1.55-1.57 s on average on A3D's lowest-margin shots (q = 0.5-1 %), against 0.92 s on r21's: the
  shots the deeper network still finds hard are also slow for Relay-BP. Routing gets more expensive as the network
  improves, so the budget admits under 1 % of the shots for A3D.

## 7. Standing against sector Relay-BP (490 / 100,000 = 0.49 % on CAL)

Sector Relay-BP alone stays far ahead. On CAL, A3D alone fails 2,682 (5.5x Relay-BP's 490) and A3D's best hybrid
within 15.8 ms fails 2,299 (4.7x). Gross's registered probe stays FAILED; stage 1 does not change that.

What the continuation gate needs (the merged plan: the network alone <= 1.5 %, the hybrid < 0.49 % within 15.8 ms on
actual routed-shot costs). The hybrid condition binds. At ~1.55 s per routed shot, a network with A3D's 1.8 ms GPU
term can route ~0.9 % of the shots, and one with 3.3 ms (a projected 12-block network) ~0.8 %. On A3D's frontier that
removes ~600-700 failures. So the hybrid beats 490 only if the network alone is below roughly 1,100-1,200 / 100k
(~1.1-1.2 %), and probably lower, since a better network leaves fewer failures among its lowest-margin shots (a rough
extrapolation from one frontier, with Relay-BP's cost per routed shot near 1.5 s). From A3D's 2,682 that means at least
~55-60 % fewer failures, about the size of stage 1's D6 step (62 % fewer than S).

What stage 1 does not establish:
- Capacity against the extra samples: every arm warm-starts from r21 and adds 2e7 samples. S and F (same shape) give
  4.2 % and 0.6 % fewer failures than r21; D6's 62 % mixes added capacity with how well new blocks use those samples.
- Cost: D6 has 1.56x the MACs, 1.77x the parameters and 1.69x the wall time of S; no control is matched in either.
- Seeds: one training seed and one data stream per tier. The paired intervals cover the sampling noise of 50,000
  shots, not the noise between training runs.
- Selection: all decisions read the same D-bank, re-read for every arm; the winner's 1,344 is a selected value.
- CAL: the frontier is exploratory; CAL already served r21's frontier and Relay-BP's reads, and no test set is
  untouched.
- Numerics: curves are bf16 on 20,000 shots, decisions fp32 / graph32 on 50,000; where both exist they agree (A3D
  final 540 / 20,000 = 2.70 % against 2.688 %).
- Tier B confirms the depth gain from scratch at fixed samples (B2 2.92x better than C at 3e7), so it is not only a
  warm-start effect. Neither tier separates capacity from compute (D6 has 1.56x the MACs) or shows robustness across
  training seeds, and the warm/scratch gaps mix samples, schedule and starting point.
- The hybrid's requirement above extrapolates one network's frontier; a deeper network's margins and routed-shot costs
  will differ.

## 8. What follows

Stage 2 is depth scaling on one GPU (D9 against D12 from A3D, then the winner's extension), at most 9 GPU-hours on this
pod; its plan is `PLAN_gross2_stage2.md`. The gate it is held to is the merged plan's (section 7): the network alone
<= 1.5 % and the hybrid < 0.49 % within 15.8 ms on actual routed-shot costs, the second being the binding one.
