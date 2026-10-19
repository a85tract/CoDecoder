### p = 0.6 % (evaluation half: odd chunks, 2,000,000 shots; tau1 = 5.242445659637451)

| decoder | failures | repairs | breaks | net (repairs - breaks) per 10^4 shots, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|
| Cascade alone | 10,110 | | | | | | | |
| network alone | 16,447 | | | | | | | |
| composition (the comparison's C5) | 10,064 | | | | | | | |
| J (joint decision) | 9,949 | 529 | 414 | +0.575 [+0.274, +0.877] | 1.14 % [0.54, 1.74] | 3.8 % | 0.000202 | w = 7.866 (step [7.865678, 7.865811)) (+77) |
| P (preference) | 10,006 | 338 | 280 | +0.29 [+0.0459, +0.535] | 0.58 % [0.09, 1.06] | 1.9 % | 0.0218 | a = 0.1069, b = 0.001 (+89) |
| T (two thresholds) | 10,005 | 327 | 268 | +0.295 [+0.0554, +0.535] | 0.59 % [0.11, 1.06] | 1.9 % | 0.0173 | t_C = 0.1091, t_S = 0.07241 (+90) |
| reference: always the network on the disagreement rows | 16,447 | 3,040 | 9,423 | -31.9 [-33, -30.8] | -63.42 % [-65.60, -61.26] | -210.0 % | < 1e-300 |  |
| reference: routed oracle | 7,024 | 3,040 | 0 | +15.2 [+14.7, +15.7] | 30.21 % [29.14, 31.29] | 100.0 % | < 1e-300 |  |

Fit half (even chunks, 2,000,000 shots; composition 10,103 failures, routed oracle repairs 3,094): J 518 repairs, 441 breaks, net +77 (0.76 %); P 340 repairs, 251 breaks, net +89 (0.88 %); T 335 repairs, 245 breaks, net +90 (0.89 %).

Evaluation half: routed 199,753 (9.99 %), disagreement rows 12,463 (only Cascade fails 3,040, only the network fails 9,423); routed oracle failures 7,024. Fit half: disagreement rows 12,380 (only Cascade fails 3,094, only the network fails 9,286).

J diagnostics: the evaluation half's own best w = 8.346 (step [8.344573, 8.347169), evaluation net +127); the fitted w's evaluation net +115.

Per-category counts of the whole read (4,000,000 shots) beside the re-test (paper Table I, single mode) and the comparison's original single-mode counts (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | ARB rate | re-test | re-test rate | z | original | original rate | z |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C2) | 20,260 | 5065.0 | 4,936 | 4936.0 | 1.63 | 4,963 | 4963.0 | 1.29 |
| composition (C5) | 20,167 | 5041.8 | 4,917 | 4917.0 | 1.58 | 4,936 | 4936.0 | 1.34 |
| only the composition fails | 75 | 18.8 | 21 | 21.0 | -0.459 | 13 | 13.0 | 1.23 |
| only Cascade alone fails | 168 | 42.0 | 40 | 40.0 | 0.277 | 40 | 40.0 | 0.277 |
| routed share | 9.977 % | | 9.98 % | | | 9.96 % | | |

Whole read: network alone 32,742, only the network fails 18,784, only Cascade fails 6,302, both 13,958, routed 399,064, routed oracle failures 14,033.

### p = 0.35 % (evaluation half: odd chunks, 15,000,000 shots; tau1 = 8.111059150695802)

| decoder | failures | repairs | breaks | net (repairs - breaks) per 10^4 shots, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|
| Cascade alone | 1,195 | | | | | | | |
| network alone | 1,997 | | | | | | | |
| composition (the comparison's C5) | 1,178 | | | | | | | |
| J (joint decision) | 1,172 | 35 | 29 | +0.004 [-0.00674, +0.0148] | 0.51 % [-0.86, 1.88] | 1.7 % | 0.532 | w = 13.84 (step [13.53909, 14.15232)) (+19) |
| P (preference) | 1,170 | 36 | 28 | +0.00533 [-0.00539, +0.0161] | 0.68 % [-0.69, 2.05] | 2.3 % | 0.382 | a = -0.08824, b = 0.1122 (+27) |
| T (two thresholds) | 1,171 | 66 | 59 | +0.00467 [-0.0101, +0.0195] | 0.59 % [-1.29, 2.49] | 2.0 % | 0.592 | t_C = 0.5645, t_S = 1.87 (+22) |
| reference: always the network on the disagreement rows | 1,997 | 344 | 1,163 | -0.546 [-0.597, -0.496] | -69.52 % [-76.06, -63.13] | -238.1 % | 5.77e-104 |  |
| reference: routed oracle | 834 | 344 | 0 | +0.229 [+0.206, +0.254] | 29.20 % [26.19, 32.39] | 100.0 % | 5.58e-104 |  |

Fit half (even chunks, 15,000,000 shots; composition 1,124 failures, routed oracle repairs 346): J 50 repairs, 31 breaks, net +19 (1.69 %); P 49 repairs, 22 breaks, net +27 (2.40 %); T 68 repairs, 46 breaks, net +22 (1.96 %).

Evaluation half: routed 150,042 (1.00 %), disagreement rows 1,507 (only Cascade fails 344, only the network fails 1,163); routed oracle failures 834. Fit half: disagreement rows 1,487 (only Cascade fails 346, only the network fails 1,141).

J diagnostics: the evaluation half's own best w = 4.271 (step [4.241655, 4.301133), evaluation net +15); the fitted w's evaluation net +6.

Per-category counts of the whole read (30,000,000 shots) beside the re-test (paper Table I, single mode) and the comparison's original single-mode counts (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | ARB rate | re-test | re-test rate | z | original | original rate | z |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C2) | 2,344 | 78.1 | 2,354 | 78.5 | -0.146 | 2,406 | 80.2 | -0.9 |
| composition (C5) | 2,302 | 76.7 | 2,325 | 77.5 | -0.338 | 2,373 | 79.1 | -1.04 |
| only the composition fails | 1 | 0.0 | 0 | 0.0 | 1 | 0 | 0.0 | 1 |
| only Cascade alone fails | 43 | 1.4 | 29 | 1.0 | 1.65 | 33 | 1.1 | 1.15 |
| routed share | 1.002 % | | 1.00 % | | | 1.00 % | | |

Whole read: network alone 3,916, only the network fails 2,305, only Cascade fails 733, both 1,611, routed 300,484, routed oracle failures 1,612.

Carried forward (fit half only): P; mean fit-half relative reduction over the two p: J 1.23 %, P 1.64 %, T 1.42 %.

One w for both p (diagnostic): w = 10.59, mean fit-half relative reduction 1.04 %; p = 0.6 %: fit net +48, evaluation net +111 (1.10 %); p = 0.35 %: fit net +18, evaluation net +11 (0.93 %).

