### RTX, BB72, p = 0.2 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,719.5 | 1,732.8 |  |  | 1,722.6 / 1,732.8; 1,723.0 / 1,739.6; 1,719.5 / 1,740.0 |
| composition | 168.1 | 1,910.6 | 10.23 | 2.00 % | 168.1 / 1,910.6; 162.9 / 1,839.7; 162.9 / 1,840.8 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 17, "2": 17, "3": 17}, "comp": {"1": 17, "2": 17, "3": 17}, "joint": {}}

### RTX, BB72, p = 0.2 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 11.9953 | 71.9717 |  |  | 11.9953; 12.0020; 12.0047 |
| composition | 1.3551 | 8.1304 | 8.85 | 2.01 % | 1.3549; 1.3551; 1.3549 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 371, "2": 371, "3": 371}, "comp": {"1": 373, "2": 373, "3": 373}, "joint": {}}

### RTX, BB72, p = 0.3 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,724.1 | 1,740.2 |  |  | 1,724.8 / 1,740.2; 1,724.6 / 1,741.9; 1,724.1 / 1,741.7 |
| composition | 296.6 | 1,864.4 | 5.81 | 9.82 % | 296.6 / 1,864.4; 296.5 / 1,841.9; 296.6 / 1,842.7 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 86, "2": 86, "3": 86}, "comp": {"1": 86, "2": 86, "3": 86}, "joint": {}}

### RTX, BB72, p = 0.3 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 12.0007 | 72.0042 |  |  | 12.0007; 12.0048; 12.0058 |
| composition | 2.3315 | 13.9889 | 5.15 | 10.03 % | 2.3310; 2.3314; 2.3315 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2275, "2": 2275, "3": 2275}, "comp": {"1": 2264, "2": 2264, "3": 2264}, "joint": {}}

### RTX, d = 13, p = 0.6 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 384.7 | 396.8 |  |  | 384.7 / 404.3; 384.8 / 396.8; 384.8 / 403.7 |
| composition | 106.0 | 486.2 | 3.63 | 10.13 % | 106.0 / 485.0; 105.3 / 477.2; 105.1 / 486.2 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 90, "2": 90, "3": 90}, "comp": {"1": 91, "2": 91, "3": 91}, "joint": {}}

### RTX, d = 13, p = 0.6 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 10.4665 | 136.0649 |  |  | 10.4665; 10.4694; 10.4668 |
| composition | 2.3410 | 30.4325 | 4.47 | 10.05 % | 2.3402; 2.3410; 2.3409 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2442, "2": 2442, "3": 2442}, "comp": {"1": 2425, "2": 2425, "3": 2425}, "joint": {}}

### RTX, d = 13, p = 0.35 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 384.8 | 401.9 |  |  | 384.8 / 402.8; 384.8 / 401.9; 397.3 / 422.4 |
| composition | 69.5 | 499.5 | 5.53 | 0.94 % | 67.9 / 474.9; 67.5 / 474.9; 69.5 / 499.5 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 0, "2": 0, "3": 0}, "comp": {"1": 0, "2": 0, "3": 0}, "joint": {}}

### RTX, d = 13, p = 0.35 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 10.4598 | 135.9774 |  |  | 10.4675; 10.4615; 10.4598 |
| composition | 1.3646 | 17.7401 | 7.66 | 1.00 % | 1.3645; 1.3642; 1.3646 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 40, "2": 40, "3": 40}, "comp": {"1": 38, "2": 38, "3": 38}, "joint": {}}

### H100, BB72, p = 0.2 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,477.1 | 1,620.9 |  |  | 1,477.1 / 1,639.5; 1,484.6 / 1,620.9; 1,482.3 / 1,631.9 |
| composition | 189.4 | 1,777.0 | 7.80 | 2.00 % | 188.6 / 1,777.0; 188.8 / 1,772.2; 189.4 / 1,768.3 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 17, "2": 17, "3": 17}, "comp": {"1": 17, "2": 17, "3": 17}, "joint": {}}

### H100, BB72, p = 0.2 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 7.3394 | 44.0361 |  |  | 7.3394; 7.3480; 7.3506 |
| composition | 1.7021 | 10.2124 | 4.31 | 2.01 % | 1.6976; 1.7019; 1.7021 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 371, "2": 371, "3": 371}, "comp": {"1": 373, "2": 373, "3": 373}, "joint": {}}

### H100, BB72, p = 0.3 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 1,482.8 | 1,611.7 |  |  | 1,484.2 / 1,614.4; 1,482.8 / 1,611.7; 1,483.0 / 1,616.0 |
| composition | 316.3 | 1,847.8 | 4.69 | 9.82 % | 315.2 / 1,815.8; 316.3 / 1,827.5; 313.3 / 1,847.8 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 86, "2": 86, "3": 86}, "comp": {"1": 86, "2": 86, "3": 86}, "joint": {}}

### H100, BB72, p = 0.3 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 7.3416 | 44.0495 |  |  | 7.3416; 7.3508; 7.3536 |
| composition | 2.3154 | 13.8926 | 3.17 | 10.03 % | 2.2916; 2.2894; 2.3154 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2275, "2": 2275, "3": 2275}, "comp": {"1": 2264, "2": 2264, "3": 2264}, "joint": {}}

### H100, d = 13, p = 0.6 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 430.1 | 545.6 |  |  | 430.1 / 562.3; 431.3 / 545.6; 431.4 / 564.9 |
| composition | 129.1 | 626.4 | 3.33 | 10.13 % | 126.9 / 603.4; 129.1 / 626.4; 127.5 / 616.7 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 90, "2": 90, "3": 90}, "comp": {"1": 91, "2": 91, "3": 91}, "joint": {}}

### H100, d = 13, p = 0.6 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 9.1562 | 119.0311 |  |  | 9.2151; 9.1562; 9.1906 |
| composition | 2.1025 | 27.3328 | 4.35 | 10.05 % | 2.0930; 2.1005; 2.1025 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 2446, "2": 2446, "3": 2446}, "comp": {"1": 2427, "2": 2427, "3": 2427}, "joint": {}}

### H100, d = 13, p = 0.35 %, single
| decoder | mean (us) | p99.9 (us) | speed-up (mean) | routed | reps: mean / p99.9 (us) |
|---|---|---|---|---|---|
| Cascade alone | 429.3 | 542.1 |  |  | 429.3 / 553.0; 432.0 / 542.1; 433.0 / 554.5 |
| composition | 83.0 | 573.9 | 5.17 | 0.94 % | 82.9 / 573.1; 83.0 / 573.9; 82.5 / 566.2 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (mean): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (via-B latency, paired mean over repetitions): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 0, "2": 0, "3": 0}, "comp": {"1": 0, "2": 0, "3": 0}, "joint": {}}

### H100, d = 13, p = 0.35 %, batched
| decoder | per round (us) | per shot (us) | speed-up | routed | reps: per round (us) |
|---|---|---|---|---|---|
| Cascade alone | 9.1202 | 118.5625 |  |  | 9.1202; 9.1756; 9.1484 |
| composition | 1.2700 | 16.5098 | 7.18 | 1.00 % | 1.2657; 1.2679; 1.2700 |
| with the joint step | - | - | - | 0.00 % |  |

Joint step, added per incoming shot (per_shot): slowest - slowest - us; paired mean - us (range - to -); per routed shot - us. On the routed shots alone (B stage's wall time, paired mean): - us per routed shot (), - us per incoming shot.
Failures per process (descriptive): {"alone": {"1": 40, "2": 40, "3": 40}, "comp": {"1": 38, "2": 38, "3": 38}, "joint": {}}

