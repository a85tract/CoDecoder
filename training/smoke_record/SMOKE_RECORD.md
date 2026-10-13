# The training release's check on the cluster (2026-10-12)

We ran the steps of `../TRAINING.md` that fit on a CPU, in two short jobs on single CPU nodes of the
Alpine cluster, each with 16 cores and 32 GB. The environment was the release's (Python 3.12.14, numpy 2.2.6, Stim
1.16.0, PyMatching 2.4.0, torch 2.11.0+cu126) plus ldpc 2.4.1, inside the studies' container. Each job copied the
repository and ran from its root with relative paths. The second job's copy has this branch's code, file for file. The
first lacked only the loader of `bb.train` checkpoints (`convlines_train.frontend.TrainedBB`), which it did not use.
`job_check.sbatch` and `job_bb_references.sbatch` are the two job scripts, with the cluster's scratch directory
written as `<cluster-scratch>`. `job_check.log` and `job_bb_references.log` are their outputs.

| step | command (job_check.sbatch unless noted) | result | files |
|---|---|---|---|
| 0, smoke test | `bash training/smoke.sh` | Trained 10^6 samples in 4.4 minutes (3,800 samples/s on 16 cores, evaluations included); the EMA network fails 113 of 20,000 monitor shots. On 100,000 calibration shots, the front end alone fails 510 and PyMatching alone 381. The paper's rule with M = 343 finds no qualifying q, and its fallback takes q = 1 % (τ = 1.466540219783783) with 369 failures. Wall time 293 s. An earlier job trained the same network, with the same 113 and 510. | `smoke/` |
| 4, the budget rule | `convlines_train.tune` on the smoke front end, `--front-us 1 --budget-us 1.04` | The budget admits q = 0 only (routing 1 % costs 1.08 µs), so the rule takes q = 0 with 510 failures, although q = 1 % has fewer. | `smoke_budget/` |
| 2, released d = 7 | `--frontend d7_b1w64_r740 --chunks 60070000:20`, paper rule with `--target-strong 0.9` | Network alone 4,262 (record 4,260); correlated PM alone 3,433 (record 3,433); routed at q = 4 %: 3,201 (record 3,204), τ = 1.394216504096985 (record 1.3934324979782104). | `released/tune_d7*` |
| 2, released d = 9 | `--frontend d9_b3_r764 --chunks 60090000:20`, paper rule, `--target 2500` | Network alone 2,484 (record 2,484); routed at q = 1 %: 2,188 (record 2,188); the rule picks q = 1 % as the study did, τ = 0.6813445055484771 (record 0.6820775270462036). | `released/tune_d9*` |
| 3, growth | d = 3 trained from scratch (3·10^4 samples), then d = 5 grown from it with `--init-from` | Both ran. The d = 5 run mapped the line read-out from 3 to 5 lines (`g5_run.json`, `init.mapped`). Networks this small and this short are not meant to decode well. | `g3.log`, `g5.log`, `g5_run.json` |
| 3, BB code | BB72 from Relay-BP's circuit, `--bb 6,6,3,1,2,3,1,2`, width 32, 1 block, 2·10^4 samples; then 2 blocks grown from it | `bb.design` and the translate-label check passed (`bb_design.json`). Both runs ended normally. The second began with the warm start's step-0 check: on 2,048 shots no decision differed from the 1-block network's (`bb2_run.json`, `init.step0`). | `bb.log`, `bb2.log`, `bb_run.json`, `bb2_run.json` |
| 4, BB with BP-OSD | `convlines_train.tune` on the 1-block BB front end, 3,000 shots, `convlines_train.strong:bposd` | Ran: BP-OSD on the 150 least confident shots, at about 0.9 s per shot on this CPU (OSD order 10). | `bb_tune*` |
| 1, frozen programs | `python -m convlines_train.check_frozen --surface 7,9,11,13,15,17 --bb` | Every check passed in 23 s (17 s in an earlier job). | `check_frozen.log` |
| 6, released BB networks | `job_bb_references.sbatch`: `--grid 0` on fresh shots, seed 5 | r11: 640 of 100,000. r12: 687 of 100,000. g2_D12k113c10: 36 of 5,000 at p = 0.4 %. | `released/tune_r11*`, `tune_r12*`, `tune_g2*` |
| 5, surface stage launcher | `PY=echo bash training/surface/run_stage.sh <stage>` for all 13 stages, on the cluster's login node | Every stage reads its parent from `$ICMP_CKPT`, carries no `--init-sha` and has `--no-val`. | `run_stage_dryrun.txt` |

Section F of `job_bb_references.log` is empty because that job's dry run set `ICMP_CKPT=/CKPT`, which the script
could not create. The dry run was repeated on the login node with a writable directory; `run_stage_dryrun.txt` holds
its output.

Not run here:
- The frozen programs' GPU training, that is retraining a released network.
- The five-block surface networks' calibration reads at d = 11-15, which are slow on a CPU.
- `convlines_train.train` on a GPU.
