# ARB3: the joint decision timed the way Table I was timed (unregistered)

2026-10-19. The timing ran from 02:33 to 03:25 PDT, and an order test on the RTX followed at 03:26-03:28.
Nothing here is registered. The timing follows SPEED-RT stage 1's protocol (the rules behind Table I) on both of Table
I's cards with fresh shots, and adds a third decoder: the composition with the joint decision. For each setting the
three decoders (Cascade alone, the composition, the composition with the joint decision) ran in one job per card, in
three fresh processes each.

The joint decision used ARB's w on BB72 (6.309573444801933 at 0.2 %, 3.758374042884441 at 0.3 %). At d = 13 it used the
w fitted for rule J (~/work/arb2, `out/analysis/summary.json`: 7.86574441929562 at 0.6 %, 13.842311471679352 at
0.35 %). That file appeared at 02:46 PDT, before the d = 13 timing started, so the w = 1.0 fallback was not used. Every
timed record carries the w it ran with.

Cards:
- RTX PRO 6000 Blackwell Server Edition on Alpine (partition artxpro6000, node gpu-node-07, AMD EPYC 9555, driver
  590.48.01), the node type of Table I's RTX column. Slurm jobs AJ52 (BB72, 02:34-02:53 PDT) and AJ54 (d = 13,
  02:53-03:10), and the untimed jobs AJ53 (d = 13 kernel caches), AJ55 (benchmark of the joint step) and
  AJ56 (order test).
- H100 80GB HBM3 on a RunPod pod (secure cloud, Intel Xeon Platinum 8480+, 23.8-CPU quota, driver
  580.126.09): BB72 02:33-02:55 PDT, d = 13 03:03-03:24. The two cards ran at the same time.

## 1. Results

Times are in µs. Single mode gives the mean and 99.9th-percentile latency per shot; batched mode gives wall time per
syndrome round. Cascade alone's value is the fastest of its three processes and each composition's the slowest of its
three, statistic by statistic, as in Table I. Speed-up = Cascade alone / composition. The italic rows are stage 1's
records put through the same script (`timing/summarize.py`); they reproduce Table I.

### BB72, p = 0.2 %

| | RTX: single mean | RTX: p99.9 | RTX: batched per round | H100: single mean | H100: p99.9 | H100: batched per round |
|---|---|---|---|---|---|---|
| Cascade alone (fastest of 3) | 1,762.6 | 1,776.2 | 11.999 | 1,894.7 | 1,955.1 | 6.872 |
| composition (slowest of 3) | 174.4 | 1,959.8 | 1.355 | 195.0 | 2,184.3 | 1.706 |
| composition + joint decision (slowest of 3) | 174.9 | 1,961.7 | 1.356 | 195.8 | 2,184.6 | 1.707 |
| speed-up of the composition | 10.11 |  | 8.85 | 9.71 |  | 4.03 |
| speed-up of the composition + joint decision | 10.08 |  | 8.85 | 9.68 |  | 4.03 |
| *Table I (stage 1): Cascade alone* | 1,719.5 | 1,732.8 | 11.995 | 1,477.1 | 1,620.9 | 7.339 |
| *Table I (stage 1): composition* | 168.1 | 1,910.6 | 1.355 | 189.4 | 1,777.0 | 1.702 |
| *Table I (stage 1): speed-up* | 10.23 |  | 8.85 | 7.80 |  | 4.31 |

Routed share (mean of the three composition processes): RTX single 1.95 %; RTX batched 1.97 %; H100 single 1.95 %; H100 batched 1.97 %.

### BB72, p = 0.3 %

| | RTX: single mean | RTX: p99.9 | RTX: batched per round | H100: single mean | H100: p99.9 | H100: batched per round |
|---|---|---|---|---|---|---|
| Cascade alone (fastest of 3) | 1,770.6 | 1,788.6 | 12.002 | 1,889.6 | 1,932.2 | 6.877 |
| composition (slowest of 3) | 295.9 | 1,877.9 | 2.331 | 352.6 | 2,218.6 | 2.247 |
| composition + joint decision (slowest of 3) | 315.7 | 1,972.7 | 2.334 | 351.8 | 2,203.1 | 2.252 |
| speed-up of the composition | 5.98 |  | 5.15 | 5.36 |  | 3.06 |
| speed-up of the composition + joint decision | 5.61 |  | 5.14 | 5.37 |  | 3.05 |
| *Table I (stage 1): Cascade alone* | 1,724.1 | 1,740.2 | 12.001 | 1,482.8 | 1,611.7 | 7.342 |
| *Table I (stage 1): composition* | 296.6 | 1,864.4 | 2.331 | 316.3 | 1,847.8 | 2.315 |
| *Table I (stage 1): speed-up* | 5.81 |  | 5.15 | 4.69 |  | 3.17 |

Routed share (mean of the three composition processes): RTX single 9.67 %; RTX batched 10.01 %; H100 single 9.67 %; H100 batched 10.01 %.

### d = 13, p = 0.6 %

| | RTX: single mean | RTX: p99.9 | RTX: batched per round | H100: single mean | H100: p99.9 | H100: batched per round |
|---|---|---|---|---|---|---|
| Cascade alone (fastest of 3) | 384.8 | 390.7 | 10.437 | 420.1 | 433.7 | 9.077 |
| composition (slowest of 3) | 108.1 | 501.8 | 2.335 | 129.6 | 643.8 | 2.112 |
| composition + joint decision (slowest of 3) | 108.1 | 506.0 | 2.343 | 127.2 | 596.2 | 2.148 |
| speed-up of the composition | 3.56 |  | 4.47 | 3.24 |  | 4.30 |
| speed-up of the composition + joint decision | 3.56 |  | 4.45 | 3.30 |  | 4.23 |
| *Table I (stage 1): Cascade alone* | 384.7 | 396.8 | 10.467 | 430.1 | 545.6 | 9.156 |
| *Table I (stage 1): composition* | 106.0 | 486.2 | 2.341 | 129.1 | 626.4 | 2.103 |
| *Table I (stage 1): speed-up* | 3.63 |  | 4.47 | 3.33 |  | 4.35 |

Routed share (mean of the three composition processes): RTX single 9.90 %; RTX batched 9.98 %; H100 single 9.90 %; H100 batched 9.98 %.

### d = 13, p = 0.35 %

| | RTX: single mean | RTX: p99.9 | RTX: batched per round | H100: single mean | H100: p99.9 | H100: batched per round |
|---|---|---|---|---|---|---|
| Cascade alone (fastest of 3) | 384.9 | 390.6 | 10.434 | 423.3 | 461.1 | 9.086 |
| composition (slowest of 3) | 67.9 | 475.6 | 1.364 | 89.2 | 600.0 | 1.282 |
| composition + joint decision (slowest of 3) | 69.4 | 499.7 | 1.365 | 91.3 | 615.8 | 1.287 |
| speed-up of the composition | 5.67 |  | 7.65 | 4.74 |  | 7.09 |
| speed-up of the composition + joint decision | 5.54 |  | 7.65 | 4.64 |  | 7.06 |
| *Table I (stage 1): Cascade alone* | 384.8 | 401.9 | 10.460 | 429.3 | 542.1 | 9.120 |
| *Table I (stage 1): composition* | 69.5 | 499.5 | 1.365 | 83.0 | 573.9 | 1.270 |
| *Table I (stage 1): speed-up* | 5.53 |  | 7.66 | 5.17 |  | 7.18 |

Routed share (mean of the three composition processes): RTX single 0.98 %; RTX batched 0.99 %; H100 single 0.98 %; H100 batched 0.99 %.

### The joint decision's added time

- "Routed shots" looks only at the shots that went to Cascade. In single mode it is their mean latency (via B) in the
  joint process minus that in the composition process of the same repetition, which decoded the same shots. In batched
  mode it is the B stage's wall time divided by the number of routed shots, paired the same way. The next column
  multiplies it by the routed share.
- "End to end, paired" is the difference in mean latency (single) or in wall time per shot (batched), repetition by
  repetition. For batched mode the bracket divides it by the routed share.
- "Slowest - slowest" is the difference between the values in the tables above, the protocol's own comparison.
- "Benchmark" is `checks/bench_joint.py`: Cascade's compiled graphs with and without the joint step, on the same inputs,
  in one process. Single mode: the host round trip when the two decisions agree / differ, and the GPU time per replay of
  the batch-1 graph. Batched: per routed row of a 4,096-row replay on BB72 (at the share of differing rows seen in the
  dry shots; in brackets with no row and with every row differing), or per row of a batch-16 replay at d = 13. The RTX
  values are from this run (job AJ55). The H100 values are from the build stage's pod (TEST.md section 4), which
  used w = 1.0 at d = 13; w is a runtime scalar there, so the graphs are the same. This run's H100 benchmark was lost
  with the pod (departure 6).

| setting | card | mode | routed | routed shots: added per routed shot, paired (range) | the same per incoming shot | end to end, paired: per incoming shot (per routed shot) | end to end, slowest - slowest | benchmark, per routed shot |
|---|---|---|---|---|---|---|---|---|
| BB72, p = 0.2 % | RTX | single | 1.95 % | 82.00 µs (9.16 to 119.77) | 1.599 µs | 8.47 µs | 0.50 µs per shot | host 2.3 / 12.1 µs (agree / differ); GPU 12.7 µs per replay |
| BB72, p = 0.2 % | RTX | batched | 1.97 % | 0.088 µs (0.08 to 0.10) | 0.0017 µs | 0.0051 µs (0.26) | 0.0035 µs per shot (0.0006 per round) | 0.195 µs (4.1 % of rows differ; 0.153 at none, 0.667 at all) |
| BB72, p = 0.2 % | H100 | single | 1.95 % | 7.43 µs (2.06 to 13.76) | 0.145 µs | 0.91 µs | 0.78 µs per shot | host -1.8 / -6.7 µs (agree / differ); GPU 3.3 µs per replay |
| BB72, p = 0.2 % | H100 | batched | 1.97 % | 0.048 µs (0.00 to 0.09) | 0.0009 µs | 0.0149 µs (0.75) | 0.0063 µs per shot (0.0010 per round) | 0.145 µs (4.1 % of rows differ; 0.111 at none, 0.608 at all) |
| BB72, p = 0.3 % | RTX | single | 9.67 % | 116.28 µs (108.08 to 120.99) | 11.245 µs | 20.61 µs | 19.77 µs per shot | host 6.4 / 19.3 µs (agree / differ); GPU 11.8 µs per replay |
| BB72, p = 0.3 % | RTX | batched | 10.01 % | 0.118 µs (0.11 to 0.13) | 0.0118 µs | 0.0144 µs (0.14) | 0.0149 µs per shot (0.0025 per round) | 0.198 µs (5.5 % of rows differ; 0.163 at none, 0.659 at all) |
| BB72, p = 0.3 % | H100 | single | 9.67 % | 3.64 µs (1.01 to 7.27) | 0.352 µs | -0.62 µs | -0.76 µs per shot | host -9.4 / -4.3 µs (agree / differ); GPU 4.0 µs per replay |
| BB72, p = 0.3 % | H100 | batched | 10.01 % | 0.037 µs (-0.02 to 0.13) | 0.0037 µs | 0.0154 µs (0.15) | 0.0341 µs per shot (0.0057 per round) | 0.142 µs (5.5 % of rows differ; 0.068 at none, 0.594 at all) |
| d = 13, p = 0.6 % | RTX | single | 9.90 % | 12.89 µs (-0.34 to 21.55) | 1.277 µs | 1.89 µs | 0.01 µs per shot | host 0.0 / 0.1 µs (agree / differ); GPU 6.7 µs per replay |
| d = 13, p = 0.6 % | RTX | batched | 9.98 % | 0.434 µs (0.31 to 0.50) | 0.0433 µs | 0.1176 µs (1.18) | 0.0998 µs per shot (0.0077 per round) | -0.869 µs (batch 16, every row) |
| d = 13, p = 0.6 % | H100 | single | 9.90 % | -2.51 µs (-4.66 to 0.89) | -0.249 µs | -0.60 µs | -2.33 µs per shot | host -2.3 / -2.5 µs (agree / differ); GPU -2.6 µs per replay |
| d = 13, p = 0.6 % | H100 | batched | 9.98 % | 1.263 µs (0.68 to 1.97) | 0.1260 µs | 0.3266 µs (3.27) | 0.4675 µs per shot (0.0360 per round) | -0.079 µs (batch 16, every row) |
| d = 13, p = 0.35 % | RTX | single | 0.98 % | 18.38 µs (16.16 to 21.92) | 0.180 µs | 1.87 µs | 1.52 µs per shot | host -0.1 / -0.2 µs (agree / differ); GPU 0.5 µs per replay |
| d = 13, p = 0.35 % | RTX | batched | 0.99 % | 0.494 µs (0.47 to 0.53) | 0.0049 µs | 0.0133 µs (1.34) | 0.0122 µs per shot (0.0009 per round) | -2.175 µs (batch 16, every row) |
| d = 13, p = 0.35 % | H100 | single | 0.98 % | 0.01 µs (-1.60 to 2.12) | 0.000 µs | 0.65 µs | 2.10 µs per shot | host -2.6 / -2.5 µs (agree / differ); GPU -2.7 µs per replay |
| d = 13, p = 0.35 % | H100 | batched | 0.99 % | 1.578 µs (1.47 to 1.69) | 0.0156 µs | 0.0477 µs (4.81) | 0.0661 µs per shot (0.0051 per round) | -0.452 µs (batch 16, every row) |

Slots not established at their first attempt: H100 r1.rep2.dfm.p0.003.c-1.single.base (established after 2 attempts).

On the RTX the single-mode rows mix the joint step with a clock difference (see below). These pairs ran at the same SM
clock:

| setting | processes | SM clock | composition: mean / via B | with the joint decision: mean / via B | added per routed shot |
|---|---|---|---|---|---|
| BB72, p = 0.2 % | job AJ52, repetition 1 | 2,295 MHz | 174.4 / 1,950.5 | 174.9 / 1,959.7 | +9.2 |
| BB72, p = 0.2 % | order test: joint, composition, joint | 2,310 MHz | 174.2 / 1,942.9 | 174.8 / 1,948.2; 174.4 / 1,949.5 | +5.3; +6.6 |
| BB72, p = 0.3 % | order test: joint, composition, joint | 2,310 MHz | 313.9 / 1,946.8 | 314.4 / 1,948.8; 314.4 / 1,949.7 | +2.0; +2.9 |
| d = 13, p = 0.6 % | job AJ54, repetition 1 | 2,297 MHz (mean) | 108.1 / 499.1 | 108.1 / 498.8 | -0.3 |

### What the numbers say

- Batched, the joint decision adds at most 0.006 µs per round on BB72 (0.25 % of the composition's cost; H100, 0.3 %)
  and at most 0.036 µs per round at d = 13 (1.7 %; H100, 0.6 %). No batched speed-up moves by more than 0.07 (4.30 to
  4.23; H100, d = 13, 0.6 %), and on BB72 none moves by more than 0.01.
- In single mode on the H100 the step costs nothing measurable at the slowest-of-three level. The means move by -2.3 to
  +2.1 µs, less than the spread between the three processes of one decoder (2.0 to 6.0 µs). The speed-ups go from 9.71
  to 9.68 and from 5.36 to 5.37 on BB72, and from 3.24 to 3.30 and from 4.74 to 4.64 at d = 13. On the routed shots
  alone the step adds 7.4 and 3.6 µs per routed BB72 shot (ranges 2.1 to 13.8 and 1.0 to 7.3), that is 0.15 and 0.35 µs
  per incoming shot, and nothing at d = 13 (-2.5 and 0.0 µs).
- In single mode on the RTX the joint processes ran at a lower SM clock than their composition twins in 10 of the 12
  pairs, and the lower clock slowed every shot, including the shots that never reach the joint step. The table's joint
  values (174.9, 315.7, 108.1 and 69.4 µs) and their speed-ups (10.08, 5.61, 3.56 and 5.54, against 10.11, 5.98, 3.56
  and 5.67 without the step) include that clock difference. At equal clock the step adds +2.0 to +9.2 µs per routed
  BB72 shot (0.10 to 0.18 µs per incoming shot at 0.2 %, 0.19 to 0.28 µs at 0.3 %) and nothing measurable at d = 13
  (-0.3 µs).
- Cascade alone and the composition reproduce Table I on the RTX, within 4 % in single mode and 0.3 % batched. On the
  H100 pod Cascade alone's BB72 batch-1 path took 28 % longer than in stage 1, so this pod's BB72 single-mode speed-ups
  are higher than Table I's (9.71 and 5.36 against 7.80 and 4.69) for a reason that has nothing to do with the joint
  step.

#### BB72

All 72 slots were accepted: the RTX's 36 at their first attempt, the H100's 35 at their first and one at its second.
That slot was Cascade alone, single, p = 0.3 %, repetition 2, whose first window saw a second compute process on the
card.

On the RTX PRO 6000, Cascade alone and the composition come out close to Table I. Batched, the composition costs 1.355
and 2.331 µs per round at 0.2 and 0.3 % (Table I: 1.355 and 2.331) and Cascade alone 12.00 (12.00). In single mode the
composition's mean is 174.4 and 295.9 µs (Table I: 168.1 and 296.6) and Cascade alone's 1,762.6 and 1,770.6 µs (1,719.5
and 1,724.1). Cascade alone ran at 2,295 MHz here, against 2,355-2,362 MHz in stage 1.

With the joint decision the RTX's batched cost is 1.356 and 2.334 µs per round, +0.0006 and +0.0025 µs per round, and
the speed-ups are 8.85 and 5.14, against 8.85 and 5.15 without the joint step. In the B stage the step adds 0.088 and
0.118 µs per routed shot.

In single mode the RTX takes 174.9 and 315.7 µs with the joint decision (speed-ups 10.08 and 5.61, against 10.11 and
5.98). The gap at 0.3 % (295.9 against 315.7 µs) comes from the SM clock. All six single-mode processes with the joint
step ran at an SM clock of 2,295 MHz: 94.4 % of the card's 2,430 MHz maximum, above the gate's 90 % floor, with no
clock-event reason. Five of the six composition processes ran at 2,422 MHz, as stage 1's composition processes did. At
2,295 MHz everything is slower, including the shots that never reach the joint step: the S-only shots took 138.9-139.6
µs in the joint processes, against 128.6-129.3 µs in the composition processes at 2,422 MHz. The one composition process
that also ran at 2,295 MHz (repetition 1 at 0.2 %) gives an equal-clock pair. Its routed shots took 1,950.5 µs, against
1,959.7 µs with the joint step (+9.2 µs per routed shot), and its S-only shots 139.1 µs, against 139.4.

The order test (job AJ56; single mode, one repetition per p, outside the protocol) ran a joint process first, then
the composition, then a joint process again, so the joint step no longer always came second. All six of its processes
ran at 2,310 MHz. The composition took 174.2 and 313.9 µs and the joint processes 174.8 / 174.4 and 314.4 / 314.4 µs.
The routed shots took +5.3 and +6.6 µs longer with the joint step at 0.2 %, and +2.0 and +2.9 µs at 0.3 %. At 2,310 MHz
the composition alone takes 313.9 µs, close to the 315.7 µs of the slowest joint process in the table. Why the card ran
every joint process of job AJ52 at the lower clock was not established. The benchmark on the same card (job
AJ55) gives +2.3 and +6.4 µs per round trip when the two decisions agree and +12.1 and +19.3 µs when they differ
(about 4-5 % of the routed shots), in line with the equal-clock pairs.

On the H100 pod the composition and its joint twin take the same time within noise. Single: 195.0 / 195.8 µs at 0.2 %
and 352.6 / 351.8 µs at 0.3 %. Batched: 1.706 / 1.707 and 2.247 / 2.252 µs per round. The speed-ups are 9.71 / 9.68 and
5.36 / 5.37 in single mode, and 4.03 / 4.03 and 3.06 / 3.05 batched. On the routed shots the step adds 7.4 and 3.6 µs
per routed shot in single mode, and 0.048 and 0.037 µs batched (B stage). The build stage's H100 benchmark gave +3.3 and
+4.0 µs of GPU time per replay.

Cascade alone's batch-1 path took 1,894.7 and 1,889.6 µs on this pod, 28 % above Table I's H100 (1,477.1 and 1,482.8).
The S-only shots are unchanged (155 µs here, 157 µs in stage 1), and Cascade alone's batched cost is 6 % below Table I's
(6.87 against 7.34 µs per round). The SM clock stayed at its 1,980 MHz maximum, and the code on the timed path is the D1
tree's (departure 1). The build stage's H100 pod, a different machine, measured 1.83 ms of GPU time per replay of the
same batch-1 graph, more than stage 1's whole 1.48 ms latency, so the slower path is on the GPU and showed on both pods.
The cause was not established. This H100's single-mode speed-ups over Cascade alone are therefore higher than Table I's,
and its Cascade-alone values should not be read against Table I's. Its composition's single mean at 0.3 % is higher than
Table I's for the same reason (352.6 against 316.3 µs), since its routed shots take that path.

Failures on these timing shots (descriptive; the same on both cards and in all three repetitions): batched, 355 -> 292
at 0.2 % and 2,225 -> 1,913 at 0.3 % (Cascade alone 361 and 2,232); single, 13 -> 11 and 92 -> 79 (Cascade alone 14 and
93).

#### d = 13

All 72 slots were accepted at their first attempt.

On the RTX PRO 6000, Cascade alone and the composition match Table I. Cascade alone takes 384.8 and 384.9 µs in single
mode and 10.44 µs per round batched (Table I: 384.7, 384.8 and 10.47, 10.46). The composition takes 108.1 and 67.9 µs
single and 2.335 and 1.364 µs per round batched (Table I: 106.0, 69.5; 2.34, 1.36).

With the joint decision the RTX takes 2.343 and 1.365 µs per round batched (+0.008 and +0.001; speed-ups 4.45 and 7.65,
against 4.47 and 7.65) and 108.1 and 69.4 µs single (speed-ups 3.56 and 5.54, against 3.56 and 5.67). Here too the joint
processes ran at lower mean SM clocks than their composition twins (2,295-2,381 MHz against 2,389-2,417 MHz) in five of
the six pairs, and their S-only shots took 65.1-65.2 µs, against 63.1-64.8. In the sixth pair (repetition 1 at 0.6 %)
both ran at 2,297 MHz: routed shots took 498.8 µs with the joint step and 499.1 µs without, and the S-only shots 65.2 µs
in both, so no cost shows at equal clock. The RTX benchmark agrees (round trip +0.0 / +0.1 µs at 0.6 % and -0.1 / -0.2
µs at 0.35 %).

On the H100 pod the values are close to Table I's H100 column. Cascade alone takes 420.1 and 423.3 µs single (Table I:
430.1 and 429.3) and 9.08 and 9.09 µs per round batched (9.16 and 9.12). The composition takes 129.6 and 89.2 µs single
(129.1 and 83.0; at 0.35 % its other two processes took 84.6 and 83.2) and 2.112 and 1.282 µs per round batched (2.103
and 1.270). With the joint decision it takes 127.2 and 91.3 µs single (speed-ups 3.30 and 4.64, against 3.24 and 4.74)
and 2.148 and 1.287 µs per round batched (speed-ups 4.23 and 7.06, against 4.30 and 7.09). In single mode the step adds
nothing measurable on the routed shots (-2.5 and 0.0 µs per routed shot).

Batched, the step at d = 13 costs more than its kernel. In the B stage it adds 1.26 and 1.58 µs per routed shot on the
H100 (ranges 0.68-1.97 and 1.47-1.69; the build stage measured 1.0 and 1.1) and 0.43 and 0.49 µs on the RTX, while the
graph benchmark at batch 16 finds no difference on either card. End to end, the paired difference is larger still: 0.33
and 0.048 µs per shot on the H100 (3.3 and 4.8 µs per routed shot) and 0.118 and 0.013 µs per shot on the RTX (1.2 and
1.3 µs per routed shot). The B stage's extra time therefore lies outside the graph replays, and the B stage accounts for
only about a third of the end-to-end difference. This run did not locate the rest. In the tables' terms (slowest minus
slowest) the cost stays small next to the composition's: +0.036 µs per round at 0.6 % on the H100 (1.7 %), and at most
+0.008 µs per round (0.3 %) in the other three cells.

Failures on these timing shots (descriptive; the same in all three repetitions): at 0.35 %, the same on both cards,
batched 40 -> 38 (Cascade alone 40) and single 3 -> 3 (3). At 0.6 % the cards differ by a few shots: batched 2,452 ->
2,442 on the RTX and 2,455 -> 2,445 on the H100 (Cascade alone 2,469 and 2,472); single 90 -> 90 and 91 -> 90 (Cascade
alone 90 and 91).

## 2. How it was timed

Each setting ran in one job per card, in fresh processes and rep-major order, as SPEED-RT's rounds R1 and R4 did. The
decoders were Cascade alone, the composition, and the composition with the joint decision, three processes each. Single
mode decodes 20,000 shots one at a time after 200 untimed shots and reports each process's mean and 99.9th-percentile
latency. Batched mode decodes 500,000 shots and reports wall time per shot and per syndrome round (per shot / 6 on BB72
and / 13 at d = 13). On BB72 the networks run at batch 4,096 and the routed rows go to Cascade at 4,096; at d = 13 the
network runs at 4,096 and Cascade at b* = 16. A composition's value is the slowest of its three processes and Cascade
alone's the fastest, statistic by statistic, as in Table I. Speed-up = Cascade alone / composition.

- Harness: `speed_rt/cascade/drive.py` with two new options. `--joint-both` follows each composition slot with its joint
  twin (slot id + `.joint`), which reads the same shots. `--engines base` keeps BB72's compositions on the BB engine,
  Table I's. Round r1 ran 36 slots per card, and round r4, run with `--only .C5. .C2.`, also 36. The gate was on:
  srtlib's rule judged every window (the SM clock at or above 90 % of its maximum throughout, no other compute
  process on the card), and a rejected attempt was replaced, at most twice.
- Shots: all fresh. The seed rule and the beacon record (B_1) are stage 1's, but the sets were renamed from `srt-cas-*`
  to `arb3-cas-*` (`speed_rt/cascade/sets_arb3.json`, through `SRT_SETS_FILE`). New names give new TAGs and so new
  seeds: none of the 320 is one of stage 1's. Every process wrote the sha256 of every chunk it sampled.
- Code: convlines_public 94aaaa7 + arb-prod (the joint decision, TEST.md) + two timing commits (116f979: `drive.py
  --joint-both --engines`, `SRT_SETS_FILE`, `relmap.py`; 3ffc276: `srt_d13.py joint-lock` with `SRT_JOINT_VENV=1`). They
  are local to the release clone; `timing/code/release_timing.patch` holds them and `timing/code/release/` the changed
  files (`SHA256SUMS`).

## 3. Where this timing departs from Table I's

1. Trees. The pinned D1 and d = 13 trees, their checkpoints (`cascade_w3`, `cascade_w5`) and the container
   (`pyfull.sif`) are no longer on Alpine's scratch; they went in the cleanup. Both cards ran the release's tree
   instead. Against the D1 commits (<research-repo> 310ce8e0 and 64b2e25d), 16 of its Python files differ, in docstrings,
   cluster paths, job labels and the error handling of `m2.py`'s screen and drive steps. None of these differences is on
   a timed path. The weights are the release's re-saved inference copies. `speed_rt/cascade/relmap.py` (with
   `SRT_RELMAP=1`) maps their identity to the D1 lock's checkpoint sha256, the same way as the build stage's
   `pod/podrun.py`.
2. Venv on Alpine. A new uv venv with the D1 tree's pins (Python 3.12.14, torch 2.11.0+cu128, Triton 3.6.0, numpy 2.2.6,
   Stim 1.16.0, PyMatching 2.4.0; `PINS_EQUAL`; `records/RTX/venv_freeze.txt`) replaced the container's venv. The kernel
   caches were compiled inside the jobs before any window: on BB72 by the job's first processes, as `common.sh` seeds
   would have held them, and at d = 13 by an untimed build pass and a verify pass that found the caches unchanged, as
   stage 1 built the H100's d = 13 caches.
3. d = 13 lock. The timing used the study's lock with the session's code digest, kernel-cache digest and venv fields
   (`srt_d13.py joint-lock`; on the RTX with `SRT_JOINT_VENV=1`, because the study's venv is gone).
4. The H100 pod was not put through `h100_pod.sh prescreen`. The gate's clock windows applied to every process.
5. Alpine. From 02:36:20 to 02:41:31 PDT the untimed d = 13 cache build (job AJ53) ran on another card of the same
   node (gpu-node-07) while BB72's first-repetition slots were timed. The node is shared with other users' jobs anyway,
   and each slot's twin ran right after it.
6. The H100 pod became unreachable at about 03:39 PDT (connection refused, heartbeat 18.6 minutes old), after both of
   its timings had been copied back. Its benchmark run and its job logs were lost, so the H100 benchmark values are the
   build stage's, and the chain scripts that ran its jobs (`after_bb.sh`, `after_prep.sh`, `after_timed.sh`) are given
   below by content.
7. The order test (job AJ56) is outside the protocol: single mode, one repetition per p, in the order joint,
   composition, joint. It is reported only to separate the clock from the joint step and is not in the tables.

## 4. Commands (as run)

The patched release (`release/` at the commits above) was packed with ARB3's `pod/` helpers and the `timing/` scripts
and unpacked as `$R` on each machine.

RTX PRO 6000 (Alpine; `timing/arb3_rtx_*.sbatch` hold the paths). The venv was built with `uv` on the login node (`uv
venv --python 3.12.14`, then `uv pip install -r req_cu128.txt` from the cu128 index), then:
```
sbatch arb3_rtx_bb.sbatch                                                     # BB72: timing/bb_job.sh, CARD=rtx
PHASE=prep  W13="0.006=1.0,0.0035=1.0" sbatch --export=ALL arb3_rtx_d13.sbatch          # d = 13 caches, untimed
PHASE=timed W13="0.006=7.86574441929562,0.0035=13.842311471679352" \
    sbatch --export=ALL --dependency=afterany:<BB job> arb3_rtx_d13.sbatch                # d = 13 timed
sbatch --dependency=afterany:<d = 13 job> arb3_rtx_bench.sbatch                           # benchmark of the joint step
sbatch arb3_rtx_order.sbatch                                                              # order test (not in the tables)  
```
H100 pod (`pod/pod_setup.sh` for the venv; then one GPU process at a time, chained):
```
R=/workspace/arb3t/release PY=/workspace/arb3t/venv/bin/python OUT=out/bb_h100 CARD=h100 JC=jc_bb W4=w4 \
  RELAY=relay_min WB="0.002=6.309573444801933,0.003=3.758374042884441" bash timing/bb_job.sh
# after_bb.sh: wait for bb_job.sh, then the d = 13 caches (untimed; the caches do not depend on w)
R=... PY=... OUTD=out/d13_h100 CARD=h100 K=kc_d13 D13ROOT=d13root DRY=timing/dry PHASE=prep \
  W13="0.006=1.0,0.0035=1.0" bash timing/d13_job.sh
# after_prep.sh: wait for the prep, require PREPDONE, then the timed phase
... PHASE=timed W13="0.006=7.86574441929562,0.0035=13.842311471679352" bash timing/d13_job.sh
# after_timed.sh: the benchmark (timing/bench_job.sh; its output was lost with the pod)
```
Inside the jobs: `bb_job.sh` builds C4b's read for m2.py's S check (untimed: `pod/podrun.py pc_eval.py ... --kind 12
--chunks 10:20`), then runs `drive.py r1 --tree $R/cascade/bb72 --card CARD --joint-w "$WB" --joint-both --engines base`
with `SRT_RELMAP=1 SRT_SETS_FILE=sets_arb3.json SRT_SEEDS=seeds_arb3.json SRT_BEACON=beacon.json`. `d13_job.sh` runs
`srtlib.py dry-lock`, `srt_d13.py session` and `joint-lock`, a build pass and verify passes (`drive.py r4 ... --gate off
--reps 1` with the dry beacon and `SRT_DRY=1`, the build with `SRT_KCACHE_BUILD=1`). Then, after checking that the
caches are still the settled session's, it runs `drive.py r4 --lock <joint lock of the study's lock> --card CARD
--joint-w13 "$W13" --joint-both --only .C5. .C2.`.

The tables:
```
cd timing
python3 summarize.py --rec RTX=records/RTX/out/bb_rtx_AJ52 RTX=records/RTX/out/d13_rtx/drive_20261019_035343 \
  H100=records/H100/out/bb_h100 H100=records/H100/out/d13_h100/drive_20261019_100326 --out timing.json --md timing_cells.md
python3 make_tables.py timing.json stage1.json bench > tables.md
```
`stage1.json` is the same summary of `speed_rt/cascade/records` (stage 1), which reproduces Table I.

## 5. Files

- `TIMING.md`: this file.
- `timing/records/{RTX,H100}/out/`: every timed slot (attempts.json; per attempt `rows.jsonl`, `trace.jsonl` with the
  chunk sha256, `windows.jsonl` with the clock samples, `log.txt`); `order_rtx_AJ56/` (order test); `bench_rtx/`.
  `records/RTX/logs/`: the Slurm logs.
- `timing/timing.json`, `timing/timing_cells.md`: per-process values and the summary cells; `timing/tables.md`: the
  tables above; `timing/stage1.json`, `timing/stage1_cells.md`: stage 1 by the same script.
- `timing/bench/{RTX,H100}/`: benchmark outputs (RTX from job AJ55, H100 from the build stage's pod).
- `timing/*.sh`, `timing/*.sbatch`, `timing/summarize.py`, `timing/make_tables.py`, `timing/dry/`: the scripts and the
  dry beacon and seeds of the untimed passes.
- `timing/code/`: the timing commits as a patch and as files.

## 6. Not done

- No bitwise check of this run's joint decisions against the reads (`joint_count.py compare`). The joint twins saved
  their shots and decisions (`pipe.npz`, not copied back), but these are new shots, so their reads would have to be
  computed first. The build stage's check (TEST.md 3b) covers the same pipelines on its own shots. The RTX's files go
  with the Alpine scratch; the H100's went with the pod.
- `analysis_cas.py` and the FREEZE pins were not run or renewed: nothing here is registered, and `summarize.py`
  reproduces Table I from stage 1's records.
- No H100 prescreen (departure 4); no H100 benchmark in this run (departure 6).
- The cause of the RTX's clock bins (2,295-2,310 against 2,422 MHz, chosen per process) and of the H100 pod's slower
  Cascade batch-1 path was not established.
- No registered test was run.
