### RTX, BB72, p = 0.2 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,762.6 | 1,776.2 |  |  | 1,762.6 / 1,776.2; 1,770.6 / 1,788.2; 1,770.4 / 1,787.6 |
| composition | 174.4 | 1,959.8 | 10.11 | 1.95 % | 174.4 / 1,959.8; 162.0 / 1,842.2; 162.1 / 1,843.0 |
| with the joint step | 174.9 | 1,961.7 | 10.08 | 1.95 % | 174.9 / 1,961.7; 174.5 / 1,961.7; 174.6 / 1,960.8 |

Joint step, added per incoming shot (mean): slowest - slowest 0.50 us; paired mean 8.47 us (range 0.50 to 12.46); per routed shot 434.23 us. On the routed shots alone (via-B latency, paired mean over repetitions): 82.00 us per routed shot (9.16, 119.77, 117.07), 1.599 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 14, "2": 14, "3": 14}, "comp": {"1": 13, "2": 13, "3": 13}, "joint": {"1": 11, "2": 11, "3": 11}}

### RTX, BB72, p = 0.2 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 11.9989 | 71.9935 |  |  | 12.0019; 11.9989; 12.0006 |
| composition | 1.3553 | 8.1320 | 8.85 | 1.97 % | 1.3547; 1.3547; 1.3553 |
| with the joint step | 1.3559 | 8.1355 | 8.85 | 1.97 % | 1.3559; 1.3556; 1.3558 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.0035 us; paired mean 0.0051 us (range 0.0027 to 0.0074); per routed shot 0.26 us. On the routed shots alone (B stage's wall time, paired mean): 0.088 us per routed shot (0.086, 0.082, 0.096), 0.0017 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 361, "2": 361, "3": 361}, "comp": {"1": 355, "2": 355, "3": 355}, "joint": {"1": 292, "2": 292, "3": 292}}

### RTX, BB72, p = 0.3 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,770.6 | 1,788.6 |  |  | 1,770.6 / 1,791.6; 1,771.3 / 1,788.6; 1,771.0 / 1,789.2 |
| composition | 295.9 | 1,877.9 | 5.98 | 9.67 % | 294.1 / 1,843.3; 294.1 / 1,844.8; 295.9 / 1,877.9 |
| with the joint step | 315.7 | 1,972.7 | 5.61 | 9.67 % | 315.3 / 1,966.4; 315.0 / 1,966.0; 315.7 / 1,972.7 |

Joint step, added per incoming shot (mean): slowest - slowest 19.77 us; paired mean 20.61 us (range 19.77 to 21.17); per routed shot 213.13 us. On the routed shots alone (via-B latency, paired mean over repetitions): 116.28 us per routed shot (120.99, 119.79, 108.08), 11.245 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 93, "2": 93, "3": 93}, "comp": {"1": 92, "2": 92, "3": 92}, "joint": {"1": 79, "2": 79, "3": 79}}

### RTX, BB72, p = 0.3 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 12.0017 | 72.0100 |  |  | 12.0028; 12.0033; 12.0017 |
| composition | 2.3311 | 13.9869 | 5.15 | 10.01 % | 2.3308; 2.3305; 2.3311 |
| with the joint step | 2.3336 | 14.0018 | 5.14 | 10.01 % | 2.3336; 2.3330; 2.3331 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.0149 us; paired mean 0.0144 us (range 0.0118 to 0.0168); per routed shot 0.14 us. On the routed shots alone (B stage's wall time, paired mean): 0.118 us per routed shot (0.113, 0.116, 0.125), 0.0118 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2232, "2": 2232, "3": 2232}, "comp": {"1": 2225, "2": 2225, "3": 2225}, "joint": {"1": 1913, "2": 1913, "3": 1913}}

### RTX, d = 13, p = 0.6 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 384.8 | 390.7 |  |  | 384.8 / 390.7; 384.8 / 390.8; 385.2 / 403.1 |
| composition | 108.1 | 501.8 | 3.56 | 9.90 % | 108.1 / 501.8; 105.3 / 478.2; 104.1 / 477.8 |
| with the joint step | 108.1 | 506.0 | 3.56 | 9.90 % | 108.1 / 502.4; 107.4 / 501.6; 107.6 / 506.0 |

Joint step, added per incoming shot (mean): slowest - slowest 0.01 us; paired mean 1.89 us (range 0.01 to 3.55); per routed shot 19.11 us. On the routed shots alone (via-B latency, paired mean over repetitions): 12.89 us per routed shot (-0.34, 17.47, 21.55), 1.277 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 90, "2": 90, "3": 90}, "comp": {"1": 90, "2": 90, "3": 90}, "joint": {"1": 90, "2": 90, "3": 90}}

### RTX, d = 13, p = 0.6 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 10.4374 | 135.6864 |  |  | 10.4374; 10.4383; 10.4400 |
| composition | 2.3353 | 30.3585 | 4.47 | 9.98 % | 2.3353; 2.3335; 2.3315 |
| with the joint step | 2.3429 | 30.4582 | 4.45 | 9.98 % | 2.3417; 2.3429; 2.3427 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.0998 us; paired mean 0.1176 us (range 0.0838 to 0.1456); per routed shot 1.18 us. On the routed shots alone (B stage's wall time, paired mean): 0.434 us per routed shot (0.310, 0.495, 0.495), 0.0433 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2469, "2": 2469, "3": 2469}, "comp": {"1": 2452, "2": 2452, "3": 2452}, "joint": {"1": 2442, "2": 2442, "3": 2442}}

### RTX, d = 13, p = 0.35 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 384.9 | 390.6 |  |  | 384.9 / 390.6; 399.5 / 422.1; 401.1 / 424.3 |
| composition | 67.9 | 475.6 | 5.67 | 0.98 % | 67.1 / 474.8; 67.9 / 474.8; 67.5 / 475.6 |
| with the joint step | 69.4 | 499.7 | 5.54 | 0.98 % | 69.3 / 489.7; 69.4 / 491.4; 69.4 / 499.7 |

Joint step, added per incoming shot (mean): slowest - slowest 1.52 us; paired mean 1.87 us (range 1.44 to 2.23); per routed shot 190.70 us. On the routed shots alone (via-B latency, paired mean over repetitions): 18.38 us per routed shot (17.05, 16.16, 21.92), 0.180 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 3, "2": 3, "3": 3}, "comp": {"1": 3, "2": 3, "3": 3}, "joint": {"1": 3, "2": 3, "3": 3}}

### RTX, d = 13, p = 0.35 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 10.4340 | 135.6422 |  |  | 10.4340; 10.4362; 10.4377 |
| composition | 1.3637 | 17.7276 | 7.65 | 0.99 % | 1.3635; 1.3637; 1.3632 |
| with the joint step | 1.3646 | 17.7398 | 7.65 | 0.99 % | 1.3646; 1.3644; 1.3644 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.0122 us; paired mean 0.0133 us (range 0.0099 to 0.0162); per routed shot 1.34 us. On the routed shots alone (B stage's wall time, paired mean): 0.494 us per routed shot (0.531, 0.486, 0.466), 0.0049 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 40, "2": 40, "3": 40}, "comp": {"1": 40, "2": 40, "3": 40}, "joint": {"1": 38, "2": 38, "3": 38}}

### H100, BB72, p = 0.2 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,894.7 | 1,955.1 |  |  | 1,894.7 / 1,955.1; 1,896.5 / 2,057.2; 1,895.8 / 1,968.3 |
| composition | 195.0 | 2,184.3 | 9.71 | 1.95 % | 194.4 / 2,184.3; 192.8 / 2,168.0; 195.0 / 2,182.8 |
| with the joint step | 195.8 | 2,184.6 | 9.68 | 1.95 % | 195.8 / 2,184.0; 194.5 / 2,184.6; 194.6 / 2,180.7 |

Joint step, added per incoming shot (mean): slowest - slowest 0.78 us; paired mean 0.91 us (range -0.44 to 1.76); per routed shot 46.49 us. On the routed shots alone (via-B latency, paired mean over repetitions): 7.43 us per routed shot (6.47, 13.76, 2.06), 0.145 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 14, "2": 14, "3": 14}, "comp": {"1": 13, "2": 13, "3": 13}, "joint": {"1": 11, "2": 11, "3": 11}}

### H100, BB72, p = 0.2 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 6.8723 | 41.2335 |  |  | 6.8851; 6.8723; 6.8745 |
| composition | 1.7059 | 10.2355 | 4.03 | 1.97 % | 1.7025; 1.7059; 1.7023 |
| with the joint step | 1.7070 | 10.2418 | 4.03 | 1.97 % | 1.7070; 1.7049; 1.7063 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.0063 us; paired mean 0.0149 us (range -0.0062 to 0.0268); per routed shot 0.75 us. On the routed shots alone (B stage's wall time, paired mean): 0.048 us per routed shot (0.093, 0.003, 0.047), 0.0009 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 361, "2": 361, "3": 361}, "comp": {"1": 355, "2": 355, "3": 355}, "joint": {"1": 292, "2": 292, "3": 292}}

### H100, BB72, p = 0.3 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,889.6 | 1,932.2 |  |  | 1,889.6 / 1,949.7; 1,890.2 / 1,932.2; 1,891.8 / 1,970.5 |
| composition | 352.6 | 2,218.6 | 5.36 | 9.67 % | 352.6 / 2,198.2; 352.2 / 2,218.6; 350.6 / 2,196.1 |
| with the joint step | 351.8 | 2,203.1 | 5.37 | 9.67 % | 351.1 / 2,200.8; 350.6 / 2,202.8; 351.8 / 2,203.1 |

Joint step, added per incoming shot (mean): slowest - slowest -0.76 us; paired mean -0.62 us (range -1.62 to 1.24); per routed shot -6.39 us. On the routed shots alone (via-B latency, paired mean over repetitions): 3.64 us per routed shot (1.01, 2.62, 7.27), 0.352 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 93, "2": 93, "3": 93}, "comp": {"1": 92, "2": 92, "3": 92}, "joint": {"1": 79, "2": 79, "3": 79}}

### H100, BB72, p = 0.3 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 6.8768 | 41.2609 |  |  | 6.8822; 6.8768; 6.8797 |
| composition | 2.2466 | 13.4798 | 3.06 | 10.01 % | 2.2461; 2.2466; 2.2455 |
| with the joint step | 2.2523 | 13.5139 | 3.05 | 10.01 % | 2.2467; 2.2523; 2.2469 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.0341 us; paired mean 0.0154 us (range 0.0038 to 0.0341); per routed shot 0.15 us. On the routed shots alone (B stage's wall time, paired mean): 0.037 us per routed shot (-0.018, 0.126, 0.005), 0.0037 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2232, "2": 2232, "3": 2232}, "comp": {"1": 2225, "2": 2225, "3": 2225}, "joint": {"1": 1913, "2": 1913, "3": 1913}}

### H100, d = 13, p = 0.6 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 420.1 | 433.7 |  |  | 430.0 / 588.3; 424.6 / 521.9; 420.1 / 433.7 |
| composition | 129.6 | 643.8 | 3.24 | 9.90 % | 125.5 / 567.7; 129.6 / 643.8; 125.7 / 573.9 |
| with the joint step | 127.2 | 596.2 | 3.30 | 9.90 % | 126.3 / 570.0; 127.2 / 596.2; 125.4 / 571.4 |

Joint step, added per incoming shot (mean): slowest - slowest -2.33 us; paired mean -0.60 us (range -2.33 to 0.83); per routed shot -6.07 us. On the routed shots alone (via-B latency, paired mean over repetitions): -2.51 us per routed shot (0.89, -4.66, -3.77), -0.249 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 91, "2": 91, "3": 91}, "comp": {"1": 91, "2": 91, "3": 91}, "joint": {"1": 90, "2": 90, "3": 90}}

### H100, d = 13, p = 0.6 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 9.0773 | 118.0053 |  |  | 9.2147; 9.0814; 9.0773 |
| composition | 2.1125 | 27.4621 | 4.30 | 9.98 % | 2.1125; 2.1012; 2.0844 |
| with the joint step | 2.1484 | 27.9295 | 4.23 | 9.98 % | 2.1484; 2.1194; 2.1056 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.4675 us; paired mean 0.3266 us (range 0.2365 to 0.4675); per routed shot 3.27 us. On the routed shots alone (B stage's wall time, paired mean): 1.263 us per routed shot (1.968, 0.685, 1.137), 0.1260 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2472, "2": 2472, "3": 2472}, "comp": {"1": 2455, "2": 2455, "3": 2455}, "joint": {"1": 2445, "2": 2445, "3": 2445}}

### H100, d = 13, p = 0.35 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 423.3 | 461.1 |  |  | 429.7 / 562.6; 423.8 / 468.9; 423.3 / 461.1 |
| composition | 89.2 | 600.0 | 4.74 | 0.98 % | 89.2 / 600.0; 84.6 / 565.3; 83.2 / 558.8 |
| with the joint step | 91.3 | 615.8 | 4.64 | 0.98 % | 91.3 / 615.8; 84.7 / 564.6; 82.9 / 560.9 |

Joint step, added per incoming shot (mean): slowest - slowest 2.10 us; paired mean 0.65 us (range -0.29 to 2.10); per routed shot 65.97 us. On the routed shots alone (via-B latency, paired mean over repetitions): 0.01 us per routed shot (-1.60, -0.48, 2.12), 0.000 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 3, "2": 3, "3": 3}, "comp": {"1": 3, "2": 3, "3": 3}, "joint": {"1": 3, "2": 3, "3": 3}}

### H100, d = 13, p = 0.35 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 9.0859 | 118.1173 |  |  | 9.1341; 9.1042; 9.0859 |
| composition | 1.2818 | 16.6632 | 7.09 | 0.99 % | 1.2818; 1.2802; 1.2798 |
| with the joint step | 1.2869 | 16.7293 | 7.06 | 0.99 % | 1.2869; 1.2844; 1.2816 |

Joint step, added per incoming shot (per_shot): slowest - slowest 0.0661 us; paired mean 0.0477 us (range 0.0230 to 0.0661); per routed shot 4.81 us. On the routed shots alone (B stage's wall time, paired mean): 1.578 us per routed shot (1.691, 1.573, 1.469), 0.0156 us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 40, "2": 40, "3": 40}, "comp": {"1": 40, "2": 40, "3": 40}, "joint": {"1": 38, "2": 38, "3": 38}}

