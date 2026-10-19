### p = 0.6 % (evaluation half: odd chunks, 10,000 shots; tau1 = 5.242445659637451)

| decoder | failures | repairs | breaks | net (repairs - breaks) per 10^4 shots, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|
| Cascade alone | 53 | | | | | | | |
| network alone | 83 | | | | | | | |
| composition (the comparison's C5) | 53 | | | | | | | |
| J (joint decision) | 53 | 0 | 0 | +0 [-3.8, +3.8] | 0.00 % [-7.17, 7.17] | 0.0 % | 1 | w = 177 (step [31.48491, 995.4891)) (+1) |
| P (preference) | 54 | 3 | 4 | -1 [-7.5, +5.35] | -1.89 % [-14.15, 10.10] | -5.6 % | 1 | a = -9.744, b = 4.467 (+3) |
| T (two thresholds) | 53 | 3 | 3 | +0 [-6.12, +6.12] | 0.00 % [-11.55, 11.55] | 0.0 % | 1 | t_C = 3.992, t_S = 2.717 (+2) |
| reference: always the network on the disagreement rows | 83 | 18 | 48 | -30 [-47, -14.3] | -56.60 % [-88.60, -26.94] | -166.7 % | 0.000287 |  |
| reference: routed oracle | 35 | 18 | 0 | +18 [+9.92, +28.1] | 33.96 % [18.72, 52.96] | 100.0 % | 7.63e-06 |  |

Fit half (even chunks, 10,000 shots; composition 50 failures, routed oracle repairs 18): J 1 repairs, 0 breaks, net +1 (2.00 %); P 5 repairs, 2 breaks, net +3 (6.00 %); T 3 repairs, 1 breaks, net +2 (4.00 %).

Evaluation half: routed 958 (9.58 %), disagreement rows 66 (only Cascade fails 18, only the network fails 48); routed oracle failures 35. Fit half: disagreement rows 68 (only Cascade fails 18, only the network fails 50).

J diagnostics: the evaluation half's own best w = 5.738 (step [5.618987, 5.859159), evaluation net +5); the fitted w's evaluation net +0.

Per-category counts of the whole read (20,000 shots) beside the re-test (paper Table I, single mode) and the comparison's original single-mode counts (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | ARB rate | re-test | re-test rate | z | original | original rate | z |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C2) | 103 | 5150.0 | 4,936 | 4936.0 | 0.427 | 4,963 | 4963.0 | 0.372 |
| composition (C5) | 103 | 5150.0 | 4,917 | 4917.0 | 0.466 | 4,936 | 4936.0 | 0.427 |
| only the composition fails | 0 | 0.0 | 21 | 21.0 | -0.648 | 13 | 13.0 | -0.51 |
| only Cascade alone fails | 0 | 0.0 | 40 | 40.0 | -0.894 | 40 | 40.0 | -0.894 |
| routed share | 9.865 % | | 9.98 % | | | 9.96 % | | |

Whole read: network alone 165, only the network fails 98, only Cascade fails 36, both 67, routed 1,973, routed oracle failures 67.

### p = 0.35 % (evaluation half: odd chunks, 10,000 shots; tau1 = 8.111059150695802)

| decoder | failures | repairs | breaks | net (repairs - breaks) per 10^4 shots, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|
| Cascade alone | 2 | | | | | | | |
| network alone | 4 | | | | | | | |
| composition (the comparison's C5) | 2 | | | | | | | |
| J (joint decision) | 2 | 0 | 0 | +0 [-3.84, +3.84] | 0.00 % [-191.92, 191.92] | - % | 1 | w = inf (step [inf, inf)) (+0) |
| P (preference) | 2 | 0 | 0 | +0 [-3.84, +3.84] | 0.00 % [-191.92, 191.92] | - % | 1 | a = -inf, b = 0 (+0) |
| T (two thresholds) | 2 | 0 | 0 | +0 [-3.84, +3.84] | 0.00 % [-191.92, 191.92] | - % | 1 | t_C = -inf, t_S = inf (+0) |
| reference: always the network on the disagreement rows | 4 | 0 | 2 | -2 [-7.35, +1.96] | -100.00 % [-367.72, 97.95] | - % | 0.5 |  |
| reference: routed oracle | 2 | 0 | 0 | +0 [-3.84, +3.84] | 0.00 % [-191.92, 191.92] | - % | 1 |  |

Fit half (even chunks, 10,000 shots; composition 0 failures, routed oracle repairs 0): J 0 repairs, 0 breaks, net +0 (0.00 %); P 0 repairs, 0 breaks, net +0 (0.00 %); T 0 repairs, 0 breaks, net +0 (0.00 %).

Evaluation half: routed 103 (1.03 %), disagreement rows 2 (only Cascade fails 0, only the network fails 2); routed oracle failures 2. Fit half: disagreement rows 0 (only Cascade fails 0, only the network fails 0).

J diagnostics: the evaluation half's own best w = inf (step [inf, inf), evaluation net +0); the fitted w's evaluation net +0.

Per-category counts of the whole read (20,000 shots) beside the re-test (paper Table I, single mode) and the comparison's original single-mode counts (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | ARB rate | re-test | re-test rate | z | original | original rate | z |
|---|---|---|---|---|---|---|---|---|
| Cascade alone (C2) | 2 | 100.0 | 2,354 | 78.5 | 0.344 | 2,406 | 80.2 | 0.313 |
| composition (C5) | 2 | 100.0 | 2,325 | 77.5 | 0.361 | 2,373 | 79.1 | 0.332 |
| only the composition fails | 0 | 0.0 | 0 | 0.0 | - | 0 | 0.0 | - |
| only Cascade alone fails | 0 | 0.0 | 29 | 1.0 | -0.139 | 33 | 1.1 | -0.148 |
| routed share | 0.950 % | | 1.00 % | | | 1.00 % | | |

Whole read: network alone 4, only the network fails 2, only Cascade fails 0, both 2, routed 190, routed oracle failures 2.

Carried forward (fit half only): P; mean fit-half relative reduction over the two p: J 1.00 %, P 3.00 %, T 2.00 %.

One w for both p (diagnostic): w = 944.1, mean fit-half relative reduction 1.00 %; p = 0.6 %: fit net +1, evaluation net +0 (0.00 %); p = 0.35 %: fit net +0, evaluation net +0 (0.00 %).

