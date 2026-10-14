# CASCADE-TRT-REG: Cascade on two engines

Times are in microseconds (us). Single: mean latency of one shot. Batched: time per round (the per-shot time divided by 6 for BB72 and by 13 for d = 13), as stage 1 reports it.

R1: BB72, D1 harness, deep pairs (S55p_deep -> C14p); S = the S network's engine (base = BB engine, new = multi-shot engine). R2: BB72, phase C pairs (S55p -> C14p). R4: d = 13 pair (ConvLines -> CS13).

Cascade alone is the fastest of 3 fresh processes, a pair the slowest of 3. The chosen engine is the Cascade engine with the lower Cascade-alone time (a tie goes to compiled). Speed-up = Cascade alone / pair, both on the chosen engine. The other engine's pair and speed-up decide nothing.

The stage 1 columns are the published stage-1 values (compiled path, slowest of 3 on both sides, speed-up = alone / pair); n/a = no matching published value.

## RTX: Cascade alone, pairs and speed-ups

| Comparison | Cascade alone, compiled (fastest of 3) | Cascade alone, TensorRT (fastest of 3) | Chosen engine | Pair, chosen engine (slowest of 3) | Speed-up | Stage 1 alone | Stage 1 pair | Stage 1 speed-up | Other engine's pair (decides nothing) | Other engine's speed-up (decides nothing) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 p=0.2 % single S=base (us) | 1729.3 | 371.5 | trt | 151.6 | 2.45 | 1723.0 | 168.1 | 10.25 | compiled: 179.4 | compiled: 9.64 |
| R1 p=0.2 % single S=new (us) | 1729.3 | 371.5 | trt | 120.9 | 3.07 | 1723.0 | 137.5 | 12.53 | compiled: 150.8 | compiled: 11.47 |
| R1 p=0.2 % batched S=base (us per round) | 12.0 | 18.9 | compiled | 1.4 | 8.63 | 12.0 | 1.4 | 8.86 | trt: 1.6 | trt: 12.15 |
| R1 p=0.2 % batched S=new (us per round) | 12.0 | 18.9 | compiled | 0.8 | 15.57 | 12.0 | 0.7 | 16.38 | trt: 0.9 | trt: 19.88 |
| R1 p=0.3 % single S=base (us) | 1725.8 | 372.5 | trt | 181.4 | 2.05 | 1724.8 | 296.6 | 5.81 | compiled: 322.6 | compiled: 5.35 |
| R1 p=0.3 % single S=new (us) | 1725.8 | 372.5 | trt | 151.1 | 2.47 | 1724.8 | 276.1 | 6.25 | compiled: 290.4 | compiled: 5.94 |
| R1 p=0.3 % batched S=base (us per round) | 12.0 | 18.9 | compiled | 2.5 | 4.89 | 12.0 | 2.3 | 5.15 | trt: 3.1 | trt: 6.05 |
| R1 p=0.3 % batched S=new (us per round) | 12.0 | 18.9 | compiled | 1.8 | 6.76 | 12.0 | 1.7 | 7.02 | trt: 2.6 | trt: 7.33 |
| R2 p=0.2 % single (us) | 1732.4 | 374.9 | trt | 104.8 | 3.58 | 1754.6 | 107.1 | 16.39 | compiled: 119.5 | compiled: 14.50 |
| R2 p=0.2 % batched (us per round) | 12.0 | 18.9 | compiled | 0.9 | 12.99 | 12.0 | 0.9 | 13.52 | trt: 1.1 | trt: 17.52 |
| R2 p=0.3 % single (us) | 1727.8 | 373.2 | trt | 118.7 | 3.15 | 1750.9 | 175.0 | 10.01 | compiled: 189.0 | compiled: 9.14 |
| R2 p=0.3 % batched (us per round) | 12.0 | 18.9 | compiled | 1.5 | 8.29 | 12.0 | 1.4 | 8.72 | trt: 1.9 | trt: 9.89 |
| R4 p=0.6 % single (us) | 408.4 | 278.6 | trt | 97.9 | 2.85 | 384.8 | 106.0 | 3.63 | compiled: 112.5 | compiled: 3.63 |
| R4 p=0.6 % batched (us per round) | 10.5 | 5.8 | trt | 1.9 | 3.08 | 10.5 | 2.3 | 4.47 | compiled: 2.4 | compiled: 4.39 |
| R4 p=0.35 % single (us) | 408.1 | 282.1 | trt | 69.9 | 4.03 | 397.3 | 69.5 | 5.71 | compiled: 72.7 | compiled: 5.61 |
| R4 p=0.35 % batched (us per round) | 10.5 | 5.8 | trt | 1.3 | 4.38 | 10.5 | 1.4 | 7.67 | compiled: 1.4 | compiled: 7.61 |

## H100: Cascade alone, pairs and speed-ups

| Comparison | Cascade alone, compiled (fastest of 3) | Cascade alone, TensorRT (fastest of 3) | Chosen engine | Pair, chosen engine (slowest of 3) | Speed-up | Stage 1 alone | Stage 1 pair | Stage 1 speed-up | Other engine's pair (decides nothing) | Other engine's speed-up (decides nothing) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 p=0.2 % single S=base (us) | 1476.0 | 479.3 | trt | 170.2 | 2.82 | 1484.6 | 189.4 | 7.84 | compiled: 192.0 | compiled: 7.69 |
| R1 p=0.2 % single S=new (us) | 1476.0 | 479.3 | trt | 134.3 | 3.57 | 1484.6 | 160.8 | 9.23 | compiled: 161.8 | compiled: 9.12 |
| R1 p=0.2 % batched S=base (us per round) | 7.2 | 14.6 | compiled | 1.7 | 4.23 | 7.4 | 1.7 | 4.32 | trt: 2.3 | trt: 6.41 |
| R1 p=0.2 % batched S=new (us per round) | 7.2 | 14.6 | compiled | 0.9 | 8.16 | 7.4 | 0.9 | 8.25 | trt: 1.2 | trt: 12.67 |
| R1 p=0.3 % single S=base (us) | 1474.4 | 478.7 | trt | 201.5 | 2.38 | 1484.2 | 316.3 | 4.69 | compiled: 314.7 | compiled: 4.69 |
| R1 p=0.3 % single S=new (us) | 1474.4 | 478.7 | trt | 173.2 | 2.76 | 1484.2 | 285.7 | 5.20 | compiled: 284.2 | compiled: 5.19 |
| R1 p=0.3 % batched S=base (us per round) | 7.2 | 14.6 | compiled | 2.3 | 3.15 | 7.4 | 2.3 | 3.18 | trt: 3.1 | trt: 4.75 |
| R1 p=0.3 % batched S=new (us per round) | 7.2 | 14.6 | compiled | 1.4 | 4.98 | 7.4 | 1.5 | 4.96 | trt: 3.8 | trt: 3.87 |
| R2 p=0.2 % single (us) | 1476.5 | 479.0 | trt | 110.6 | 4.33 | 1481.9 | 123.5 | 12.00 | compiled: 124.3 | compiled: 11.88 |
| R2 p=0.2 % batched (us per round) | 7.2 | 14.6 | compiled | 1.1 | 6.28 | 7.5 | 1.1 | 6.55 | trt: 1.3 | trt: 11.62 |
| R2 p=0.3 % single (us) | 1479.4 | 482.1 | trt | 133.2 | 3.62 | 1483.2 | 189.6 | 7.82 | compiled: 189.9 | compiled: 7.79 |
| R2 p=0.3 % batched (us per round) | 7.2 | 14.6 | compiled | 3.0 | 2.42 | 7.4 | 1.4 | 5.44 | trt: 1.8 | trt: 8.17 |
| R4 p=0.6 % single (us) | 425.2 | 324.8 | trt | 115.4 | 2.81 | 431.4 | 129.1 | 3.34 | compiled: 132.8 | compiled: 3.20 |
| R4 p=0.6 % batched (us per round) | 9.0 | 8.0 | trt | 1.9 | 4.10 | 9.2 | 2.1 | 4.38 | compiled: 2.1 | compiled: 4.34 |
| R4 p=0.35 % single (us) | 428.1 | 338.3 | trt | 83.9 | 4.03 | 433.0 | 83.0 | 5.22 | compiled: 89.1 | compiled: 4.81 |
| R4 p=0.35 % batched (us per round) | 9.0 | 8.0 | trt | 1.2 | 6.44 | 9.2 | 1.3 | 7.23 | compiled: 1.2 | compiled: 7.22 |

## Decision checks (TensorRT against the compiled path, fresh shots)

| Card | Cascade | p | Path | Shots | Decisions differ | Agreement | Failures, compiled | Failures, TensorRT | Only compiled fails | Only TensorRT fails | Exact McNemar p |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX | BB72 (C14p) | 0.2 % | batch 4096 | 100000 | 11 | 0.999890 | 63 | 64 | 1 | 2 | 1 |
| RTX | BB72 (C14p) | 0.3 % | batch 4096 | 100000 | 34 | 0.999660 | 448 | 448 | 4 | 4 | 1 |
| RTX | BB72 (C14p) | total | batch 4096 | 200000 | 45 | 0.999775 | 511 | 512 | 5 | 6 | 1 |
| RTX | BB72 (C14p) | 0.2 % | batch 1 | 20000 | 1 | 0.999950 | 9 | 10 | 0 | 1 | 1 |
| RTX | BB72 (C14p) | 0.3 % | batch 1 | 20000 | 9 | 0.999550 | 87 | 87 | 0 | 0 | 1 |
| RTX | BB72 (C14p) | total | batch 1 | 40000 | 10 | 0.999750 | 96 | 97 | 0 | 1 | 1 |
| RTX | d = 13 (CS13) | 0.35 % | batch 16 | 100000 | 0 | 1.000000 | 8 | 8 | 0 | 0 | 1 |
| RTX | d = 13 (CS13) | 0.6 % | batch 16 | 100000 | 11 | 0.999890 | 481 | 480 | 6 | 5 | 1 |
| RTX | d = 13 (CS13) | total | batch 16 | 200000 | 11 | 0.999945 | 489 | 488 | 6 | 5 | 1 |
| RTX | d = 13 (CS13) | 0.35 % | batch 1 | 20000 | 0 | 1.000000 | 2 | 2 | 0 | 0 | 1 |
| RTX | d = 13 (CS13) | 0.6 % | batch 1 | 20000 | 0 | 1.000000 | 90 | 90 | 0 | 0 | 1 |
| RTX | d = 13 (CS13) | total | batch 1 | 40000 | 0 | 1.000000 | 92 | 92 | 0 | 0 | 1 |
| H100 | BB72 (C14p) | 0.2 % | batch 4096 | 100000 | 11 | 0.999890 | 63 | 62 | 1 | 0 | 1 |
| H100 | BB72 (C14p) | 0.3 % | batch 4096 | 100000 | 36 | 0.999640 | 448 | 446 | 3 | 1 | 0.625 |
| H100 | BB72 (C14p) | total | batch 4096 | 200000 | 47 | 0.999765 | 511 | 508 | 4 | 1 | 0.375 |
| H100 | BB72 (C14p) | 0.2 % | batch 1 | 20000 | 0 | 1.000000 | 9 | 9 | 0 | 0 | 1 |
| H100 | BB72 (C14p) | 0.3 % | batch 1 | 20000 | 8 | 0.999600 | 87 | 87 | 0 | 0 | 1 |
| H100 | BB72 (C14p) | total | batch 1 | 40000 | 8 | 0.999800 | 96 | 96 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | 0.35 % | batch 16 | 100000 | 0 | 1.000000 | 8 | 8 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | 0.6 % | batch 16 | 100000 | 8 | 0.999920 | 481 | 481 | 4 | 4 | 1 |
| H100 | d = 13 (CS13) | total | batch 16 | 200000 | 8 | 0.999960 | 489 | 489 | 4 | 4 | 1 |
| H100 | d = 13 (CS13) | 0.35 % | batch 1 | 20000 | 0 | 1.000000 | 2 | 2 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | 0.6 % | batch 1 | 20000 | 0 | 1.000000 | 90 | 90 | 0 | 0 | 1 |
| H100 | d = 13 (CS13) | total | batch 1 | 40000 | 0 | 1.000000 | 92 | 92 | 0 | 0 | 1 |

## Engines (TensorRT builds)

| Card | Net | Batch | Precision | Convolutions held | Optimisation level | Workspace GB (0 = default) | ONNX sha256 | Plan sha256 | TensorRT | GPU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| RTX | bb | 1 | bf16 | False | 3 | 0.0 | f96da0f18f404376bdceb706428012dc4c19b7f7972f190388d2b87fc9920c8e | 784a89955607c7c473fc2ca9ed8bcc33ca86b1860de673724c61ebf2692f6d13 | 10.16.1.11 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| RTX | bb | 4096 | bf16 | False | 3 | 0.0 | a057e9e672762f8ba90e93ac2632a34b27dcde5c6548deadea555e1f2d8ec53a | 310146d9f968ba84390d0d35fabb860d00de188c7d515a2b94eaa2e3ca7cb0aa | 10.16.1.11 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| RTX | d13 | 1 | bf16 | True | 3 | 0.0 | 2ddbe1f3ad3910532b47acaf975b02e4729db0fa0d0a22597f54302bac6d080b | 3219e19264f9c34c790e6973c8f623832a7a927e6aceea2e45b75f36a53a2087 | 10.16.1.11 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| RTX | d13 | 16 | bf16 | True | 3 | 0.0 | 2a592c122a83a1b494a59a0f1b3249b28a9fe83775e64cbcb8c5aa141631c266 | e10b811c3aad9922ab77d4625bbd100008a589922404c2aed3bf2759e41bf73c | 10.16.1.11 | NVIDIA RTX PRO 6000 Blackwell Server Edition |
| H100 | bb | 1 | bf16 | False | 3 | 0.0 | f96da0f18f404376bdceb706428012dc4c19b7f7972f190388d2b87fc9920c8e | 7d38d5ad75f1d36fdd19b13d599c00f4e4524ce932fca84dee24cdc13c4e95b4 | 10.16.1.11 | NVIDIA H100 80GB HBM3 |
| H100 | bb | 4096 | bf16 | False | 3 | 0.0 | a057e9e672762f8ba90e93ac2632a34b27dcde5c6548deadea555e1f2d8ec53a | 1a774165ba4a9067ffd91603e9887ec5b68e4959b4f718398278a0e02bda1698 | 10.16.1.11 | NVIDIA H100 80GB HBM3 |
| H100 | d13 | 1 | bf16 | True | 3 | 0.0 | 2ddbe1f3ad3910532b47acaf975b02e4729db0fa0d0a22597f54302bac6d080b | fa12621a6985480d5e42ed45717736e03898ce4e93415465789c7d9613758683 | 10.16.1.11 | NVIDIA H100 80GB HBM3 |
| H100 | d13 | 16 | bf16 | True | 3 | 0.0 | 2a592c122a83a1b494a59a0f1b3249b28a9fe83775e64cbcb8c5aa141631c266 | 4996aad8f5e058ed89f0c962494487f2d030f2276126ed6936f7fc3d4976b793 | 10.16.1.11 | NVIDIA H100 80GB HBM3 |

## Checks

none
