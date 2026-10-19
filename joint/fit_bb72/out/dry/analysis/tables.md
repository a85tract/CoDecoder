### p = 0.2 % (evaluation half: odd chunks, 10,000 shots; tau1 = 7.3331888389587405)

| decoder | failures | repairs | breaks | third-answer changes (repairs / breaks / neutral) | net (repairs - breaks) / N_eval, 95 % CI | relative failure reduction | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|---|
| Cascade alone (C14') | 6 | | | | | | | | |
| composition (D1's rule) | 5 | | | | | | | | |
| A1m | 5 | 0 | 0 (<= 3.0e-04 per shot) | - | 0.00e+00 [-3.84e-04, 3.84e-04] | 0.00 % | nan % | 1.000 | t_C = -inf, t_S = inf (+0) |
| A1s | 5 | 0 | 0 (<= 3.0e-04 per shot) | - | 0.00e+00 [-3.84e-04, 3.84e-04] | 0.00 % | nan % | 1.000 | t_C = -inf, t_S = inf (+0) |
| A2 | 5 | 0 | 0 (<= 3.0e-04 per shot) | - | 0.00e+00 [-3.84e-04, 3.84e-04] | 0.00 % | nan % | 1.000 | a = 1.388, b = 0 (+1) |
| A3 | 5 | 0 | 0 (<= 3.0e-04 per shot) | 0 (0 / 0 / 0) | 0.00e+00 [-3.84e-04, 3.84e-04] | 0.00 % | nan % | 1.000 | w = 12.59 (+2) |
| reference: always CoDecoder on the disagreement rows | 8 | 0 | 3 | - | -3.00e-04 [-8.79e-04, 1.32e-04] | -60.00 % | nan % | 0.250 | |
| reference: routed oracle (two answers) | 5 | 0 | 0 | - | 0.00e+00 [-3.84e-04, 3.84e-04] | 0.00 % | nan % | 1.000 | |

Evaluation half: routed 201, disagreement rows 4 (only C14' fails 0, only CoDecoder fails 3, both fail with different answers 1); routed oracle failures 5.

Per-category counts of the whole read (20,000 shots) beside D1's original reads (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | D1 | ARB rate | D1 rate | z |
|---|---|---|---|---|---|
| C | 11 | 7,922 | 550.0 | 720.2 | -0.90 |
| S | 17 | 10,348 | 850.0 | 940.7 | -0.42 |
| only_S | 9 | 4,735 | 450.0 | 430.5 | +0.13 |
| only_C | 3 | 2,309 | 150.0 | 209.9 | -0.58 |
| both | 8 | 5,613 | 400.0 | 510.3 | -0.69 |
| routed | 388 | 220,058 | 19400.0 | 20005.3 | -0.61 |
| kept_breaks | 0 | 19 | 0.0 | 1.7 | -0.19 |
| kept_repairs | 2 | 86 | 100.0 | 7.8 | +4.61 |
| kept_both | 0 | 238 | 0.0 | 21.6 | -0.66 |
| C_fail_routed | 9 | 7,598 | 450.0 | 690.7 | -1.30 |
| composition | 9 | 7,855 | 450.0 | 714.1 | -1.40 |
| oracle_fail | 8 | 5,632 | 400.0 | 512.0 | -0.70 |
| oracle_repairs | 1 | 2,223 | 50.0 | 202.1 | -1.51 |

### p = 0.3 % (evaluation half: odd chunks, 10,000 shots; tau1 = 7.60439167022705)

| decoder | failures | repairs | breaks | third-answer changes (repairs / breaks / neutral) | net (repairs - breaks) / N_eval, 95 % CI | relative failure reduction | share of routed oracle | McNemar p vs composition | fitted on the fit half (fit-half net) |
|---|---|---|---|---|---|---|---|---|---|
| Cascade alone (C14') | 36 | | | | | | | | |
| composition (D1's rule) | 36 | | | | | | | | |
| A1m | 30 | 7 | 1 | - | 6.00e-04 [-1.66e-05, 1.32e-03] | 16.67 % | 50.0 % | 0.070 | t_C = 0.3174, t_S = 0.6388 (+5) |
| A1s | 33 | 4 | 1 | - | 3.00e-04 [-2.49e-04, 9.11e-04] | 8.33 % | 25.0 % | 0.375 | t_C = -2.507, t_S = 0.9167 (+3) |
| A2 | 26 | 10 | 0 (<= 3.0e-04 per shot) | - | 1.00e-03 [3.62e-04, 1.81e-03] | 27.78 % | 83.3 % | 0.002 | a = 0.1427, b = 0.1259 (+7) |
| A3 | 26 | 10 | 0 (<= 3.0e-04 per shot) | 7 (3 / 0 / 4) | 1.00e-03 [3.62e-04, 1.81e-03] | 27.78 % | 83.3 % | 0.002 | w = 10 (+6) |
| reference: always CoDecoder on the disagreement rows | 55 | 12 | 31 | - | -1.90e-03 [-3.30e-03, -6.21e-04] | -52.78 % | -158.3 % | 0.005 | |
| reference: routed oracle (two answers) | 24 | 12 | 0 | - | 1.20e-03 [5.23e-04, 2.07e-03] | 33.33 % | 100.0 % | 4.88e-04 | |

Evaluation half: routed 957, disagreement rows 50 (only C14' fails 12, only CoDecoder fails 31, both fail with different answers 7); routed oracle failures 24.

Per-category counts of the whole read (20,000 shots) beside D1's original reads (rates per 10^6 shots; z of the difference, independent samples):

| count | ARB | D1 | ARB rate | D1 rate | z |
|---|---|---|---|---|---|
| C | 82 | 76,730 | 4100.0 | 4513.5 | -0.87 |
| S | 126 | 112,636 | 6300.0 | 6625.6 | -0.57 |
| only_S | 69 | 54,491 | 3450.0 | 3205.4 | +0.61 |
| only_C | 25 | 18,585 | 1250.0 | 1093.2 | +0.67 |
| both | 57 | 58,145 | 2850.0 | 3420.3 | -1.38 |
| routed | 1,957 | 1,701,544 | 97850.0 | 100090.8 | -1.06 |
| kept_breaks | 0 | 81 | 0.0 | 4.8 | -0.31 |
| kept_repairs | 1 | 342 | 50.0 | 20.1 | +0.94 |
| kept_both | 2 | 1,977 | 100.0 | 116.3 | -0.21 |
| C_fail_routed | 79 | 74,411 | 3950.0 | 4377.1 | -0.91 |
| composition | 81 | 76,469 | 4050.0 | 4498.2 | -0.95 |
| oracle_fail | 57 | 58,226 | 2850.0 | 3425.1 | -1.39 |
| oracle_repairs | 24 | 18,243 | 1200.0 | 1073.1 | +0.55 |

Carried forward (fit half only; largest mean over p of the fit-half relative failure reduction; ties: fewer parameters, then A1m, A1s, A2, A3; none if not positive): **A3**; mean fit-half relative reduction A1m 5.56 %, A1s 3.33 %, A2 20.28 %, A3 31.67 %.
