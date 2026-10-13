# Cascade-surface at d = 13: the confirmation

Lock sha256 5767020407ed11f3; GPU NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 590.48.01, torch 2.11.0+cu128, venv <cluster-scratch>/bb/venv_cu128 (packages 6fd0f5933f01c59c). B is Cascade-surface at H = 256, an inferred reproduction (SURFACE_PLAN 3.2); bf16 for B, fp16 matmul inputs with fp32 accumulation for S's engine, PyMatching on the node's CPU.

## p = 0.6 %

Confirmation block: 1,000,000 shots (chunks 0-99); delta_p = 0.00051 (10 % of C2's development failure rate). Thresholds: C1 tau1 9.55498, tau2 0.867188 (q1 0.4, r 0.005); C5 tau1 5.24245 (q1 0.1); C4 tau 1.8486117124557495.

### single

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 4,351 | 0.00435 (0.00422, 0.00448) | 0.000336 | 0.4005 | 0.0049 |
| C2 | 4,936 | 0.00494 (0.0048, 0.00508) | 0.000381 | 1.0000 | 0.0000 |
| C3 | 6,226 | 0.00623 (0.00607, 0.00638) | 0.000482 | 0.0000 | 1.0000 |
| C4 | 5,341 | 0.00534 (0.0052, 0.00549) | 0.000413 | 0.0000 | 0.0194 |
| C5 | 4,917 | 0.00492 (0.00478, 0.00506) | 0.00038 | 0.0998 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 708 | 1293 | 1.69e-39 | 0.00185 | (-0.000673, -0.000498) | better |
| C5 vs C2 | S | 21 | 40 | 0.0204 | - | (-3.48e-05, -3.29e-06) | non-inferior |
| C4 vs C2 | S | 2418 | 2013 | 1.25e-09 | 0.00278 | (0.000275, 0.000536) | worse |
| C3 vs C2 | S | 3896 | 2606 | 6.88e-58 | 0.00147 | (0.00113, 0.00145) | worse |
| C1 vs C4 | S | 1090 | 2080 | 3.33e-70 | 0.00139 | (-0.0011, -0.00088) | better |
| C5 vs C4 | S | 1973 | 2397 | 1.52e-10 | 0.0025 | (-0.000554, -0.000295) | better |
| C1 vs C5 | S | 718 | 1284 | 4.99e-37 | 0.002 | (-0.000654, -0.000479) | better |
| C4 vs C3 | S | 1392 | 2277 | 1.07e-48 | 0.00167 | (-0.001, -0.000767) | better |
| C1 vs C2 | N | 708 | 1293 | 4.93e-114 | 0.00128 | (-0.000673, -0.000498) | rejected |
| C5 vs C2 | N | 21 | 40 | 1.19e-112 | 0.00135 | (-3.48e-05, -3.29e-06) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 65, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

### batched

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 4,353 | 0.00435 (0.00423, 0.00448) | 0.000336 | 0.4005 | 0.0049 |
| C2 | 4,947 | 0.00495 (0.00481, 0.00509) | 0.000382 | 1.0000 | 0.0000 |
| C3 | 6,226 | 0.00623 (0.00607, 0.00638) | 0.000482 | 0.0000 | 1.0000 |
| C4 | 5,341 | 0.00534 (0.0052, 0.00549) | 0.000413 | 0.0000 | 0.0194 |
| C5 | 4,927 | 0.00493 (0.00479, 0.00507) | 0.000381 | 0.0998 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 703 | 1297 | 1.04e-40 | 0.00179 | (-0.000682, -0.000507) | better |
| C5 vs C2 | S | 21 | 41 | 0.0151 | - | (-3.6e-05, -4.17e-06) | non-inferior |
| C4 vs C2 | S | 2418 | 2024 | 3.63e-09 | 0.00357 | (0.000263, 0.000525) | worse |
| C3 vs C2 | S | 3896 | 2617 | 7.78e-57 | 0.00152 | (0.00112, 0.00144) | worse |
| C1 vs C4 | S | 1093 | 2081 | 7.8e-70 | 0.00143 | (-0.0011, -0.000878) | better |
| C5 vs C4 | S | 1983 | 2397 | 4.24e-10 | 0.00263 | (-0.000544, -0.000284) | better |
| C1 vs C5 | S | 713 | 1287 | 4.36e-38 | 0.00192 | (-0.000662, -0.000487) | better |
| C4 vs C3 | S | 1392 | 2277 | 1.07e-48 | 0.00172 | (-0.001, -0.000767) | better |
| C1 vs C2 | N | 703 | 1297 | 9.72e-116 | 0.00125 | (-0.000682, -0.000507) | rejected |
| C5 vs C2 | N | 21 | 41 | 8.45e-113 | 0.00132 | (-3.6e-05, -4.17e-06) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 65, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

## p = 0.35 %

Confirmation block: 30,000,000 shots (chunks 0-2999); delta_p = 7.5e-06 (10 % of C2's development failure rate). Thresholds: C1 tau1 13.6375, tau2 4.59375 (q1 0.6, r 0.001); C5 tau1 8.11106 (q1 0.01); C4 tau 1.8486117124557495.

### single

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 2,632 | 8.77e-05 (8.44e-05, 9.11e-05) | 6.75e-06 | 0.6003 | 0.0010 |
| C2 | 2,354 | 7.85e-05 (7.54e-05, 8.17e-05) | 6.04e-06 | 1.0000 | 0.0000 |
| C3 | 3,057 | 0.000102 (9.84e-05, 0.000106) | 7.84e-06 | 0.0000 | 1.0000 |
| C4 | 2,537 | 8.46e-05 (8.13e-05, 8.79e-05) | 6.51e-06 | 0.0000 | 0.0003 |
| C5 | 2,325 | 7.75e-05 (7.44e-05, 8.07e-05) | 5.96e-06 | 0.0100 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 1483 | 1205 | 8.93e-08 | 0.00417 | (5.88e-06, 1.27e-05) | worse |
| C5 vs C2 | S | 0 | 29 | 3.73e-09 | 0.00385 | (-1.35e-06, -5.99e-07) | better |
| C4 vs C2 | S | 1091 | 908 | 4.63e-05 | 0.00714 | (3.18e-06, 9.03e-06) | worse |
| C3 vs C2 | S | 1936 | 1233 | 5.67e-36 | 0.00217 | (1.98e-05, 2.71e-05) | worse |
| C1 vs C4 | S | 905 | 810 | 0.0232 | - | (4.6e-07, 5.88e-06) | level |
| C5 vs C4 | S | 879 | 1091 | 1.95e-06 | 0.00556 | (-9.97e-06, -4.17e-06) | better |
| C1 vs C5 | S | 1486 | 1179 | 2.96e-09 | 0.00313 | (6.86e-06, 1.36e-05) | worse |
| C4 vs C3 | S | 818 | 1338 | 2.83e-29 | 0.00227 | (-2.04e-05, -1.43e-05) | better |
| C1 vs C2 | N | 1483 | 1205 | 1 | - | (5.88e-06, 1.27e-05) | - |
| C5 vs C2 | N | 0 | 29 | 2.22e-55 | 0.00161 | (-1.35e-06, -5.99e-07) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 30, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

### batched

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 2,628 | 8.76e-05 (8.43e-05, 9.1e-05) | 6.74e-06 | 0.6003 | 0.0010 |
| C2 | 2,352 | 7.84e-05 (7.53e-05, 8.16e-05) | 6.03e-06 | 1.0000 | 0.0000 |
| C3 | 3,057 | 0.000102 (9.84e-05, 0.000106) | 7.84e-06 | 0.0000 | 1.0000 |
| C4 | 2,537 | 8.46e-05 (8.13e-05, 8.79e-05) | 6.51e-06 | 0.0000 | 0.0003 |
| C5 | 2,322 | 7.74e-05 (7.43e-05, 8.06e-05) | 5.95e-06 | 0.0100 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 1484 | 1208 | 1.13e-07 | 0.00455 | (5.81e-06, 1.26e-05) | worse |
| C5 vs C2 | S | 0 | 30 | 1.86e-09 | 0.00294 | (-1.39e-06, -6.27e-07) | better |
| C4 vs C2 | S | 1093 | 908 | 3.85e-05 | 0.00625 | (3.25e-06, 9.1e-06) | worse |
| C3 vs C2 | S | 1941 | 1236 | 4.41e-36 | 0.00208 | (1.98e-05, 2.72e-05) | worse |
| C1 vs C4 | S | 901 | 810 | 0.0295 | - | (3.3e-07, 5.74e-06) | level |
| C5 vs C4 | S | 878 | 1093 | 1.4e-06 | 0.005 | (-1.01e-05, -4.27e-06) | better |
| C1 vs C5 | S | 1487 | 1181 | 3.4e-09 | 0.00333 | (6.83e-06, 1.36e-05) | worse |
| C4 vs C3 | S | 818 | 1338 | 2.83e-29 | 0.00238 | (-2.04e-05, -1.43e-05) | better |
| C1 vs C2 | N | 1484 | 1208 | 1 | - | (5.81e-06, 1.26e-05) | - |
| C5 vs C2 | N | 0 | 30 | 1.58e-55 | 0.00156 | (-1.39e-06, -6.27e-07) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 30, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

## Speed (one GPU model; slowest of three processes)

Decoders alone: S t1 63.6 us, a(4096) 16.2 us; B t1 408 us (compiled@1), a 139 us (compiled@16); M {"0.006": {"single_us": 180.22469806310255, "processes_single": 3, "p999_us": 435.4103802470491, "batched_us": 12.275974120013416, "processes_batched": 3}, "0.0035": {"single_us": 83.84101139672566, "processes_single": 3, "p999_us": 196.08013454126197, "batched_us": 5.899101796094328, "processes_batched": 3}}.

| p | chain | mode | accuracy | pipeline us/shot | C2 us/shot | faster | bitwise | tail |
|---|---|---|---|---|---|---|---|---|
| 0.006 | C1 | single | better | 229 | 397 | True | True | faster on average, slower in the tail |
| 0.006 | C1 | batched | better | 72.8 | 136 | True | True |  |
| 0.006 | C5 | single | non-inferior | 107 | 397 | True | True | faster on average, slower in the tail |
| 0.006 | C5 | batched | non-inferior | 30.4 | 136 | True | True |  |
| 0.0035 | C1 | single | worse | 313 | 397 | True | True | faster on average, slower in the tail |
| 0.0035 | C1 | batched | worse | 101 | 136 | True | True |  |
| 0.0035 | C5 | single | better | 69.4 | 397 | True | True | faster on average, slower in the tail |
| 0.0035 | C5 | batched | better | 17.8 | 136 | True | True |  |

## Verdicts

- p = 0.6 %, C1: faster and more accurate
- p = 0.6 %, C5: faster within the predefined accuracy tolerance (up to 10 % more failures than Cascade alone)
- p = 0.35 %, C1: no claim
- p = 0.35 %, C5: faster and more accurate

Headline chains: ['C5'] (joint planned power {"C1": 0.919, "C5": 1.0}); more accurate: none. P1 (C2 better than C3 at both p, both modes): True; P2 chains: ['C5']; d = 17: propose.
