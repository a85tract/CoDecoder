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

| setting | card | mode | routed | routed shots: added per routed shot, paired (range) | the same per incoming shot | end to end, paired: per incoming shot (per routed shot) | end to end, slowest - slowest | benchmark, per routed shot |
|---|---|---|---|---|---|---|---|---|
| BB72, p = 0.2 % | RTX | single | 1.95 % | 82.00 us (9.16 to 119.77) | 1.599 us | 8.47 us | 0.50 us per shot | host 2.3 / 12.1 us (agree / differ); GPU 12.7 us per replay |
| BB72, p = 0.2 % | RTX | batched | 1.97 % | 0.088 us (0.08 to 0.10) | 0.0017 us | 0.0051 us (0.26) | 0.0035 us per shot (0.0006 per round) | 0.195 us (4.1 % of rows differ; 0.153 at none, 0.667 at all) |
| BB72, p = 0.2 % | H100 | single | 1.95 % | 7.43 us (2.06 to 13.76) | 0.145 us | 0.91 us | 0.78 us per shot | host -1.8 / -6.7 us (agree / differ); GPU 3.3 us per replay |
| BB72, p = 0.2 % | H100 | batched | 1.97 % | 0.048 us (0.00 to 0.09) | 0.0009 us | 0.0149 us (0.75) | 0.0063 us per shot (0.0010 per round) | 0.145 us (4.1 % of rows differ; 0.111 at none, 0.608 at all) |
| BB72, p = 0.3 % | RTX | single | 9.67 % | 116.28 us (108.08 to 120.99) | 11.245 us | 20.61 us | 19.77 us per shot | host 6.4 / 19.3 us (agree / differ); GPU 11.8 us per replay |
| BB72, p = 0.3 % | RTX | batched | 10.01 % | 0.118 us (0.11 to 0.13) | 0.0118 us | 0.0144 us (0.14) | 0.0149 us per shot (0.0025 per round) | 0.198 us (5.5 % of rows differ; 0.163 at none, 0.659 at all) |
| BB72, p = 0.3 % | H100 | single | 9.67 % | 3.64 us (1.01 to 7.27) | 0.352 us | -0.62 us | -0.76 us per shot | host -9.4 / -4.3 us (agree / differ); GPU 4.0 us per replay |
| BB72, p = 0.3 % | H100 | batched | 10.01 % | 0.037 us (-0.02 to 0.13) | 0.0037 us | 0.0154 us (0.15) | 0.0341 us per shot (0.0057 per round) | 0.142 us (5.5 % of rows differ; 0.068 at none, 0.594 at all) |
| d = 13, p = 0.6 % | RTX | single | 9.90 % | 12.89 us (-0.34 to 21.55) | 1.277 us | 1.89 us | 0.01 us per shot | host 0.0 / 0.1 us (agree / differ); GPU 6.7 us per replay |
| d = 13, p = 0.6 % | RTX | batched | 9.98 % | 0.434 us (0.31 to 0.50) | 0.0433 us | 0.1176 us (1.18) | 0.0998 us per shot (0.0077 per round) | -0.869 us (batch 16, every row) |
| d = 13, p = 0.6 % | H100 | single | 9.90 % | -2.51 us (-4.66 to 0.89) | -0.249 us | -0.60 us | -2.33 us per shot | host -2.3 / -2.5 us (agree / differ); GPU -2.6 us per replay |
| d = 13, p = 0.6 % | H100 | batched | 9.98 % | 1.263 us (0.68 to 1.97) | 0.1260 us | 0.3266 us (3.27) | 0.4675 us per shot (0.0360 per round) | -0.079 us (batch 16, every row) |
| d = 13, p = 0.35 % | RTX | single | 0.98 % | 18.38 us (16.16 to 21.92) | 0.180 us | 1.87 us | 1.52 us per shot | host -0.1 / -0.2 us (agree / differ); GPU 0.5 us per replay |
| d = 13, p = 0.35 % | RTX | batched | 0.99 % | 0.494 us (0.47 to 0.53) | 0.0049 us | 0.0133 us (1.34) | 0.0122 us per shot (0.0009 per round) | -2.175 us (batch 16, every row) |
| d = 13, p = 0.35 % | H100 | single | 0.98 % | 0.01 us (-1.60 to 2.12) | 0.000 us | 0.65 us | 2.10 us per shot | host -2.6 / -2.5 us (agree / differ); GPU -2.7 us per replay |
| d = 13, p = 0.35 % | H100 | batched | 0.99 % | 1.578 us (1.47 to 1.69) | 0.0156 us | 0.0477 us (4.81) | 0.0661 us per shot (0.0051 per round) | -0.452 us (batch 16, every row) |

Slots not established at their first attempt: H100 r1.rep2.dfm.p0.003.c-1.single.base (established after 2 attempts).
