# E3 pod go/no-go: NO-GO

Timing verdict: **REVISE**: grid: block phase 5.87 us (GO <= 5.25, REVISE <= 7.0); whole shot 74.57 us vs FROZEN 67.20 us same session, bound 56.66 us (not met).

Not GO because:

- timing REVISE: grid: block phase 5.87 us (GO <= 5.25, REVISE <= 7.0); whole shot 74.57 us vs FROZEN 67.20 us same session, bound 56.66 us (not met)

## Timing (median over the 5 accepted processes t_1.json, t_2.json, t_3.json, t_4.json, t_5.json, us)

| config | whole shot | block phase (slope) | full coverage |
|---|---|---|---|
| FROZEN | 67.20 | | |
| grid | 74.57 | 5.87 | True |
| custom | 74.93 | 5.74 | False |
| staged | 81.30 | - | False |

grid per-step increments (prefix graphs, time1 only): {"1": 10.093, "2": 6.316, "3": 5.922, "4": 5.861, "5": 5.946, "6": 10.084, "7": 29.822}

staged per-step increments (prefix graphs, time1 only): {"1": 10.855, "2": 7.629, "3": 7.448, "4": 7.253, "5": 6.633, "6": 11.273, "7": 29.627}

FROZEN per-kernel increments (time1 only): {"stem": 8.816, "block0": 8.559, "block1": 8.163, "block2": 8.28, "block3": 8.261, "block4_ro": 14.158, "readout": 10.432}

## Exactness

| run | pass | mismatches | deep | graph | graph-deep | alternate | replays | full coverage |
|---|---|---|---|---|---|---|---|---|
| custom | True | 0 | [0, 20000] | [0, 200000] | None | 1000 | None | False |
| main | True | 0 | [0, 200000] | [0, 200000] | [0, 512] | 2000 | 100000 | True |
| staged | True | 0 | [0, 20000] | [0, 200000] | None | 1000 | None | False |

## Sanitizers

- san_memcheck_main: 0 errors
- san_memcheck_staged: 0 errors
- san_memcheck_custom: 0 errors
- san_racecheck_main: 0 hazards (0 errors, 0 warnings)
- san_racecheck_staged: 0 hazards (0 errors, 0 warnings)
- san_racecheck_custom: 0 hazards (0 errors, 0 warnings)
- san_synccheck_main: 0 errors
- san_synccheck_staged: 0 errors
- san_initcheck_main: 0 errors
- san_initcheck_staged: 0 errors

## Resources

GPU NVIDIA H100 80GB HBM3, 132 SMs, opt-in shared memory 232448 B per block; main configuration co-resident: True; head variant: 1.

## Steps

- bundle: exit 0, 0 s RESULT bundle manifest OK
- venv: exit 0, 22 s 
- versions: exit 0, 4 s RESULT {'python': '3.12.15', 'torch': '2.11.0+cu126', 'cuda': '12.6', 'triton': '3.6.0', 'numpy': '2.2.6', 'stim': '1.16.0', 'gpu': 'NVIDIA H100 80GB HBM3', 'cc': (9, 0)}
- build: exit 0, 21 s 
- report: exit 0, 8 s RESULT {"sms": 132, "smem_optin": 232448, "main_t3_nbuf2_grid_coresident": true, "configs": 30}
- shots: exit 0, 4 s RESULT d17 val digest 04c501a00e1412e1c5260cd066172355a7e86a1180db3d0f7787a196fe2720b2 manifest 04c501a00e1412e1c5260cd066172355a7e86a1180db3d0f7787a196fe2720b2 OK (200000, 4896)
- head1: exit 0, 7 s RESULT {"pass_": true, "mismatch_total": 0, "nonzero": {}, "routing": null, "seconds": 5.4}
- time1: exit 0, 21 s RESULT {"frozen": 67.20911521911621, "new": {"grid": 74.48106088638306, "custom": 74.82574713230133, "staged": 81.21921763420104}, "slope": {"grid": 5.869709650675456, "custom": 5.748402277628581, "staged": null}, "accepted": true, "acceptance": {"gpu_alone": true, "clocks": true}, "clocks_timed": {"samples": 85, "sm_min": 1980.0, "why": ""}, "nvml_pid": "2192820"}
- time2: exit 0, 13 s RESULT {"frozen": 67.28943059444427, "new": {"grid": 74.56568169593811, "custom": 74.89371838569642, "staged": 81.2927595615387}, "slope": {"grid": 5.86346435546875, "custom": 5.734284400939941, "staged": null}, "accepted": true, "acceptance": {"gpu_alone": true, "clocks": true}, "clocks_timed": {"samples": 84, "sm_min": 1980.0, "why": ""}, "nvml_pid": "2193175"}
- time3: exit 0, 13 s RESULT {"frozen": 67.20146408081055, "new": {"grid": 74.57076001167297, "custom": 74.93238246440887, "staged": 81.32310099601746}, "slope": {"grid": 5.864734967549642, "custom": 5.74206002553304, "staged": null}, "accepted": true, "acceptance": {"gpu_alone": true, "clocks": true}, "clocks_timed": {"samples": 86, "sm_min": 1980.0, "why": ""}, "nvml_pid": "2193436"}
- main: exit 0, 90 s RESULT {"pass_": true, "mismatch_total": 0, "nonzero": {}, "routing": 0, "seconds": 88.0}
- custom: exit 0, 45 s RESULT {"pass_": true, "mismatch_total": 0, "nonzero": {}, "routing": 0, "seconds": 42.7}
- staged: exit 0, 47 s RESULT {"pass_": true, "mismatch_total": 0, "nonzero": {}, "routing": 0, "seconds": 45.5}
- san_memcheck_main: exit 0, 3 s RESULT {"kernel": "e2_eng_t3_b1_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 1}
- san_memcheck_staged: exit 0, 3 s RESULT {"kernel": "e2_eng_t3_b0_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 0}
- san_memcheck_custom: exit 0, 3 s RESULT {"kernel": "e2_eng_t3_b2_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 1}
- san_racecheck_main: exit 0, 15 s RESULT {"kernel": "e2_eng_t3_b1_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 1}
- san_racecheck_staged: exit 0, 22 s RESULT {"kernel": "e2_eng_t3_b0_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 0}
- san_racecheck_custom: exit 0, 14 s RESULT {"kernel": "e2_eng_t3_b2_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 1}
- san_synccheck_main: exit 0, 3 s RESULT {"kernel": "e2_eng_t3_b1_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 1}
- san_synccheck_staged: exit 0, 3 s RESULT {"kernel": "e2_eng_t3_b0_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 0}
- san_initcheck_main: exit 0, 3 s RESULT {"kernel": "e2_eng_t3_b1_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 1}
- san_initcheck_staged: exit 0, 3 s RESULT {"kernel": "e2_eng_t3_b0_h1", "shots": 4, "graph_replays": 3, "ln32": false, "c": [-7.093709468841553, 7.908435344696045, -8.104992866516113, -9.010062217712402], "cooperative_node": 0}
- time4: exit 0, 14 s RESULT {"frozen": 67.10285613536834, "new": {"grid": 74.58334717750549, "custom": 75.04720973968506, "staged": 81.30652465820313}, "slope": {"grid": 5.866635958353679, "custom": 5.740269025166829, "staged": null}, "accepted": true, "acceptance": {"gpu_alone": true, "clocks": true}, "clocks_timed": {"samples": 84, "sm_min": 1980.0, "why": ""}, "nvml_pid": "2199661"}
- time5: exit 0, 14 s RESULT {"frozen": 67.14629907608033, "new": {"grid": 74.57452170848846, "custom": 74.92800152301788, "staged": 81.29930539131165}, "slope": {"grid": 5.869241396586101, "custom": 5.738713264465332, "staged": null}, "accepted": true, "acceptance": {"gpu_alone": true, "clocks": true}, "clocks_timed": {"samples": 87, "sm_min": 1980.0, "why": ""}, "nvml_pid": "2199952"}
- summary: exit 0, 0 s VERDICT NO-GO (timing REVISE): grid: block phase 5.87 us (GO <= 5.25, REVISE <= 7.0); whole shot 74.57 us vs FROZEN 67.20 us same session, bound 56.66 us (not met) | not GO: timing REVISE: grid: block phase 5.87 us (GO <= 5.25, REVISE <= 7.0); whole shot 74.57 us vs FROZEN 67.20 us same session, bound 56.66 us (not met)
