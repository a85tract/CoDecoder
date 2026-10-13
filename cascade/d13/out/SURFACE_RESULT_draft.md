# Cascade-surface at d = 13: the confirmation

Lock sha256 5767020407ed11f3; GPU NVIDIA RTX PRO 6000 Blackwell Server Edition, driver 590.48.01, torch 2.11.0+cu128, venv <cluster-scratch>/bb/venv_cu128 (packages 6fd0f5933f01c59c). B is Cascade-surface at H = 256, an inferred reproduction (SURFACE_PLAN 3.2); bf16 for B, fp16 matmul inputs with fp32 accumulation for S's engine, PyMatching on the node's CPU.

## p = 0.6 %

Confirmation block: 1,000,000 shots (chunks 0-99); delta_p = 0.00051 (10 % of C2's development failure rate). Thresholds: C1 tau1 9.55498, tau2 0.867188 (q1 0.4, r 0.005); C5 tau1 5.24245 (q1 0.1); C4 tau 1.8486117124557495.

### single

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 4,372 | 0.00437 (0.00424, 0.0045) | 0.000338 | 0.4000 | 0.0049 |
| C2 | 4,963 | 0.00496 (0.00483, 0.0051) | 0.000384 | 1.0000 | 0.0000 |
| C3 | 6,320 | 0.00632 (0.00617, 0.00648) | 0.000489 | 0.0000 | 1.0000 |
| C4 | 5,388 | 0.00539 (0.00525, 0.00553) | 0.000417 | 0.0000 | 0.0194 |
| C5 | 4,936 | 0.00494 (0.0048, 0.00508) | 0.000381 | 0.0996 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 661 | 1252 | 4.03e-42 | 0.00179 | (-0.000677, -0.000506) | better |
| C5 vs C2 | S | 13 | 40 | 0.000269 | 0.00357 | (-4.19e-05, -1.23e-05) | better |
| C4 vs C2 | S | 2415 | 1990 | 1.62e-10 | 0.00313 | (0.000295, 0.000555) | worse |
| C3 vs C2 | S | 3886 | 2529 | 9.09e-65 | 0.00152 | (0.0012, 0.00151) | worse |
| C1 vs C4 | S | 1062 | 2078 | 1.26e-74 | 0.00143 | (-0.00113, -0.000907) | better |
| C5 vs C4 | S | 1950 | 2402 | 7.78e-12 | 0.00263 | (-0.000581, -0.000323) | better |
| C1 vs C5 | S | 668 | 1232 | 1.03e-38 | 0.00192 | (-0.00065, -0.000479) | better |
| C4 vs C3 | S | 1345 | 2277 | 1.44e-54 | 0.00167 | (-0.00105, -0.000814) | better |
| C1 vs C2 | N | 661 | 1252 | 7.27e-119 | 0.00125 | (-0.000677, -0.000506) | rejected |
| C5 vs C2 | N | 13 | 40 | 1.56e-115 | 0.00132 | (-4.19e-05, -1.23e-05) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 48, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

### batched

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 4,368 | 0.00437 (0.00424, 0.0045) | 0.000337 | 0.4000 | 0.0049 |
| C2 | 4,953 | 0.00495 (0.00482, 0.00509) | 0.000383 | 1.0000 | 0.0000 |
| C3 | 6,320 | 0.00632 (0.00617, 0.00648) | 0.000489 | 0.0000 | 1.0000 |
| C4 | 5,388 | 0.00539 (0.00525, 0.00553) | 0.000417 | 0.0000 | 0.0194 |
| C5 | 4,926 | 0.00493 (0.00479, 0.00507) | 0.000381 | 0.0996 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 660 | 1245 | 1.84e-41 | 0.00185 | (-0.000671, -0.0005) | better |
| C5 vs C2 | S | 13 | 40 | 0.000269 | 0.00385 | (-4.19e-05, -1.23e-05) | better |
| C4 vs C2 | S | 2413 | 1978 | 5.57e-11 | 0.00278 | (0.000305, 0.000565) | worse |
| C3 vs C2 | S | 3884 | 2517 | 7.55e-66 | 0.00147 | (0.00121, 0.00152) | worse |
| C1 vs C4 | S | 1058 | 2078 | 2.63e-75 | 0.00139 | (-0.00113, -0.000911) | better |
| C5 vs C4 | S | 1938 | 2400 | 2.46e-12 | 0.0025 | (-0.000591, -0.000333) | better |
| C1 vs C5 | S | 667 | 1225 | 4.48e-38 | 0.002 | (-0.000644, -0.000473) | better |
| C4 vs C3 | S | 1345 | 2277 | 1.44e-54 | 0.00172 | (-0.00105, -0.000814) | better |
| C1 vs C2 | N | 660 | 1245 | 4.46e-118 | 0.00128 | (-0.000671, -0.0005) | rejected |
| C5 vs C2 | N | 13 | 40 | 1.57e-115 | 0.00135 | (-4.19e-05, -1.23e-05) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 48, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

## p = 0.35 %

Confirmation block: 30,000,000 shots (chunks 0-2999); delta_p = 7.5e-06 (10 % of C2's development failure rate). Thresholds: C1 tau1 13.6375, tau2 4.59375 (q1 0.6, r 0.001); C5 tau1 8.11106 (q1 0.01); C4 tau 1.8486117124557495.

### single

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 2,506 | 8.35e-05 (8.03e-05, 8.69e-05) | 6.43e-06 | 0.6003 | 0.0010 |
| C2 | 2,406 | 8.02e-05 (7.71e-05, 8.35e-05) | 6.17e-06 | 1.0000 | 0.0000 |
| C3 | 2,909 | 9.7e-05 (9.35e-05, 0.000101) | 7.46e-06 | 0.0000 | 1.0000 |
| C4 | 2,502 | 8.34e-05 (8.02e-05, 8.67e-05) | 6.42e-06 | 0.0000 | 0.0003 |
| C5 | 2,373 | 7.91e-05 (7.6e-05, 8.23e-05) | 6.09e-06 | 0.0100 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 1405 | 1305 | 0.0572 | - | (-6.83e-08, 6.74e-06) | level |
| C5 vs C2 | S | 0 | 33 | 2.33e-10 | 0.00333 | (-1.5e-06, -7.11e-07) | better |
| C4 vs C2 | S | 1046 | 950 | 0.0334 | - | (2.81e-07, 6.12e-06) | level |
| C3 vs C2 | S | 1824 | 1321 | 2.95e-19 | 0.00227 | (1.31e-05, 2.04e-05) | worse |
| C1 vs C4 | S | 828 | 824 | 0.941 | - | (-2.53e-06, 2.79e-06) | level |
| C5 vs C4 | S | 917 | 1046 | 0.00385 | 0.00417 | (-7.2e-06, -1.41e-06) | better |
| C1 vs C5 | S | 1406 | 1273 | 0.0108 | - | (1.05e-06, 7.82e-06) | level |
| C4 vs C3 | S | 817 | 1224 | 1.94e-19 | 0.00208 | (-1.65e-05, -1.06e-05) | better |
| C1 vs C2 | N | 1405 | 1305 | 0.00826 | - | (-6.83e-08, 6.74e-06) | - |
| C5 vs C2 | N | 0 | 33 | 5.84e-56 | 0.00161 | (-1.5e-06, -7.11e-07) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 29, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

### batched

| configuration | failures | P_block (Wilson 95 %) | P_L per round | to B | to M |
|---|---|---|---|---|---|
| C1 | 2,499 | 8.33e-05 (8.01e-05, 8.66e-05) | 6.41e-06 | 0.6003 | 0.0010 |
| C2 | 2,411 | 8.04e-05 (7.72e-05, 8.36e-05) | 6.18e-06 | 1.0000 | 0.0000 |
| C3 | 2,909 | 9.7e-05 (9.35e-05, 0.000101) | 7.46e-06 | 0.0000 | 1.0000 |
| C4 | 2,502 | 8.34e-05 (8.02e-05, 8.67e-05) | 6.42e-06 | 0.0000 | 0.0003 |
| C5 | 2,376 | 7.92e-05 (7.61e-05, 8.24e-05) | 6.09e-06 | 0.0100 | 0.0000 |

| comparison | test | X only | Y only | p | Holm level | Newcombe 95 % | label |
|---|---|---|---|---|---|---|---|
| C1 vs C2 | S | 1397 | 1309 | 0.0944 | - | (-4.66e-07, 6.34e-06) | non-inferior |
| C5 vs C2 | S | 0 | 35 | 5.82e-11 | 0.00294 | (-1.58e-06, -7.67e-07) | better |
| C4 vs C2 | S | 1045 | 954 | 0.0441 | - | (1.12e-07, 5.96e-06) | level |
| C3 vs C2 | S | 1822 | 1324 | 6.73e-19 | 0.00238 | (1.29e-05, 2.03e-05) | worse |
| C1 vs C4 | S | 822 | 825 | 0.961 | - | (-2.75e-06, 2.55e-06) | level |
| C5 vs C4 | S | 919 | 1045 | 0.00478 | 0.005 | (-7.1e-06, -1.3e-06) | better |
| C1 vs C5 | S | 1398 | 1275 | 0.0183 | - | (7.22e-07, 7.48e-06) | level |
| C4 vs C3 | S | 817 | 1224 | 1.94e-19 | 0.00217 | (-1.65e-05, -1.06e-05) | better |
| C1 vs C2 | N | 1397 | 1309 | 0.00429 | 0.00455 | (-4.66e-07, 6.34e-06) | rejected |
| C5 vs C2 | N | 0 | 35 | 2.96e-56 | 0.00156 | (-1.58e-06, -7.67e-07) | rejected |

Bitwise notes: {"S1_SB_logits_differ": 0, "MS_MB_differ": 0, "B1_Bb_decisions_differ": 29, "BQ1_Bb_logits_differ": 0, "BQ5_Bb_logits_differ": 0}

## Speed (one GPU model; slowest of three processes)

Decoders alone: S t1 63.6 us, a(4096) 16.2 us; B t1 408 us (compiled@1), a 139 us (compiled@16); M {"0.006": {"single_us": 180.22469806310255, "processes_single": 3, "p999_us": 435.4103802470491, "batched_us": 12.275974120013416, "processes_batched": 3}, "0.0035": {"single_us": 83.84101139672566, "processes_single": 3, "p999_us": 196.08013454126197, "batched_us": 5.899101796094328, "processes_batched": 3}}.

| p | chain | mode | accuracy | pipeline us/shot | C2 us/shot | faster | bitwise | tail |
|---|---|---|---|---|---|---|---|---|
| 0.006 | C1 | single | better | 229 | 397 | True | True | faster on average, slower in the tail |
| 0.006 | C1 | batched | better | 72.8 | 136 | True | True |  |
| 0.006 | C5 | single | better | 107 | 397 | True | True | faster on average, slower in the tail |
| 0.006 | C5 | batched | better | 30.4 | 136 | True | True |  |
| 0.0035 | C1 | single | level | 313 | 397 | True | True | faster on average, slower in the tail |
| 0.0035 | C1 | batched | non-inferior | 101 | 136 | True | True |  |
| 0.0035 | C5 | single | better | 69.4 | 397 | True | True | faster on average, slower in the tail |
| 0.0035 | C5 | batched | better | 17.8 | 136 | True | True |  |

## Verdicts

- p = 0.6 %, C1: faster and more accurate
- p = 0.6 %, C5: faster and more accurate
- p = 0.35 %, C1: no claim
- p = 0.35 %, C5: faster and more accurate

Headline chains: ['C5'] (joint planned power {"C1": 0.919, "C5": 1.0}); more accurate: ['C5']. P1 (C2 better than C3 at both p, both modes): True; P2 chains: ['C5']; d = 17: propose.
