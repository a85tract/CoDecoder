### p = 0.2 % (evaluation half: odd chunks, 5,500,000 shots; tau1 = 7.3331888389587405)

| decoder | failures | repairs | breaks | third-answer changes (repairs / breaks / neutral) | net (repairs - breaks) / N_eval, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|---|
| Cascade alone (C14') | 4,097 | | | | | | | | |
| composition (D1's rule) | 4,053 | | | | | | | | |
| A1m | 3,400 | 838 | 185 | - | 1.19e-04 [1.07e-04, 1.30e-04] | 16.11 % [14.58, 17.68] | 55.2 % | 9.19e-100 | t_C = 0.2173, t_S = 0.0005913 (+616) |
| A1s | 3,607 | 702 | 256 | - | 8.11e-05 [7.01e-05, 9.22e-05] | 11.00 % [9.51, 12.51] | 37.7 % | 1.21e-48 | t_C = -2.263, t_S = 0.0005913 (+406) |
| A2 | 3,341 | 721 | 9 | - | 1.29e-04 [1.20e-04, 1.39e-04] | 17.57 % [16.28, 18.90] | 60.1 % | 5.54e-200 | a = -4.281, b = 1.059 (+699) |
| A3 | 3,329 | 742 | 18 | 714 (283 / 4 / 427) | 1.32e-04 [1.22e-04, 1.42e-04] | 17.86 % [16.55, 19.22] | 61.1 % | 3.08e-193 | w = 6.31 (+719) |
| reference: always CoDecoder on the disagreement rows | 5,204 | 1,184 | 2,335 | - | -2.09e-04 [-2.31e-04, -1.88e-04] | -28.40 % | -97.2 % | 3.07e-85 | |
| reference: routed oracle (two answers) | 2,869 | 1,184 | 0 | - | 2.15e-04 [2.03e-04, 2.28e-04] | 29.21 % | 100.0 % | 0.00e+00 | |

Fit half (even chunks, 5,500,000 shots; composition 4,018 failures, routed oracle repairs 1,146): A1m 808 repairs, 192 breaks, net +616 (15.33 %); A1s 660 repairs, 254 breaks, net +406 (10.10 %); A2 707 repairs, 8 breaks, net +699 (17.40 %); A3 733 repairs, 14 breaks, net +719 (17.89 %).

Evaluation half: routed 110,268, disagreement rows 4,530 (only C14' fails 1,184, only CoDecoder fails 2,335, both fail with different answers 1,011); routed oracle failures 2,869.

Per-category counts of the whole read (11,000,000 shots) beside D1's original reads (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | D1 | ARB rate | D1 rate | z |
|---|---|---|---|---|---|
| C | 8,163 | 7,922 | 742.1 | 720.2 | +1.90 |
| S | 10,501 | 10,348 | 954.6 | 940.7 | +1.06 |
| only_S | 4,776 | 4,735 | 434.2 | 430.5 | +0.42 |
| only_C | 2,438 | 2,309 | 221.6 | 209.9 | +1.87 |
| both | 5,725 | 5,613 | 520.5 | 510.3 | +1.05 |
| routed | 220,654 | 220,058 | 20059.5 | 20005.3 | +0.91 |
| kept_breaks | 16 | 19 | 1.5 | 1.7 | -0.51 |
| kept_repairs | 108 | 86 | 9.8 | 7.8 | +1.58 |
| kept_both | 266 | 238 | 24.2 | 21.6 | +1.25 |
| C_fail_routed | 7,789 | 7,598 | 708.1 | 690.7 | +1.54 |
| composition | 8,071 | 7,855 | 733.7 | 714.1 | +1.71 |
| oracle_fail | 5,741 | 5,632 | 521.9 | 512.0 | +1.02 |
| oracle_repairs | 2,330 | 2,223 | 211.8 | 202.1 | +1.59 |

### p = 0.3 % (evaluation half: odd chunks, 8,500,000 shots; tau1 = 7.60439167022705)

| decoder | failures | repairs | breaks | third-answer changes (repairs / breaks / neutral) | net (repairs - breaks) / N_eval, 95 % CI | relative failure reduction [approx. 95 % CI] | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|---|
| Cascade alone (C14') | 37,998 | | | | | | | | |
| composition (D1's rule) | 37,888 | | | | | | | | |
| A1m | 33,705 | 5,875 | 1,692 | - | 4.92e-04 [4.72e-04, 5.12e-04] | 11.04 % [10.59, 11.49] | 46.0 % | 0.00e+00 | t_C = 0.2231, t_S = 0.0004895 (+4,134) |
| A1s | 35,312 | 3,919 | 1,343 | - | 3.03e-04 [2.86e-04, 3.20e-04] | 6.80 % [6.42, 7.18] | 28.3 % | 3.98e-288 | t_C = -3.132, t_S = 7.629e-05 (+2,436) |
| A2 | 32,950 | 5,109 | 171 | - | 5.81e-04 [5.64e-04, 5.98e-04] | 13.03 % [12.66, 13.41] | 54.3 % | 0.00e+00 | a = -1.086, b = 0.5012 (+4,867) |
| A3 | 32,504 | 5,763 | 379 | 6,082 (2,055 / 105 / 3,922) | 6.33e-04 [6.15e-04, 6.52e-04] | 14.21 % [13.81, 14.62] | 59.2 % | 0.00e+00 | w = 3.758 (+5,306) |
| reference: always CoDecoder on the disagreement rows | 56,092 | 9,102 | 27,306 | - | -2.14e-03 [-2.19e-03, -2.10e-03] | -48.05 % | -200.0 % | 0.00e+00 | |
| reference: routed oracle (two answers) | 28,786 | 9,102 | 0 | - | 1.07e-03 [1.05e-03, 1.09e-03] | 24.02 % | 100.0 % | 0.00e+00 | |

Fit half (even chunks, 8,500,000 shots; composition 37,985 failures, routed oracle repairs 9,009): A1m 5,835 repairs, 1,701 breaks, net +4,134 (10.88 %); A1s 3,828 repairs, 1,392 breaks, net +2,436 (6.41 %); A2 5,049 repairs, 182 breaks, net +4,867 (12.81 %); A3 5,688 repairs, 382 breaks, net +5,306 (13.97 %).

Evaluation half: routed 849,558, disagreement rows 47,202 (only C14' fails 9,102, only CoDecoder fails 27,306, both fail with different answers 10,794); routed oracle failures 28,786.

Per-category counts of the whole read (17,000,000 shots) beside D1's original reads (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | D1 | ARB rate | D1 rate | z |
|---|---|---|---|---|---|
| C | 76,110 | 76,730 | 4477.1 | 4513.5 | -1.59 |
| S | 112,144 | 112,636 | 6596.7 | 6625.6 | -1.04 |
| only_S | 54,480 | 54,491 | 3204.7 | 3205.4 | -0.03 |
| only_C | 18,446 | 18,585 | 1085.1 | 1093.2 | -0.72 |
| both | 57,664 | 58,145 | 3392.0 | 3420.3 | -1.42 |
| routed | 1,701,204 | 1,701,544 | 100070.8 | 100090.8 | -0.19 |
| kept_breaks | 98 | 81 | 5.8 | 4.8 | +1.27 |
| kept_repairs | 335 | 342 | 19.7 | 20.1 | -0.27 |
| kept_both | 2,032 | 1,977 | 119.5 | 116.3 | +0.87 |
| C_fail_routed | 73,743 | 74,411 | 4337.8 | 4377.1 | -1.74 |
| composition | 75,873 | 76,469 | 4463.1 | 4498.2 | -1.53 |
| oracle_fail | 57,762 | 58,226 | 3397.8 | 3425.1 | -1.36 |
| oracle_repairs | 18,111 | 18,243 | 1065.4 | 1073.1 | -0.69 |

Carried forward (fit half only; largest mean over p of the fit-half relative failure reduction; ties: fewer parameters, then A1m, A1s, A2, A3; none if not positive): **A3**; mean fit-half relative reduction A1m 13.11 %, A1s 8.26 %, A2 15.10 %, A3 15.93 %.
