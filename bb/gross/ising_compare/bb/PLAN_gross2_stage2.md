# gross2, stage 2: one D12 run on one GPU (exploratory)

2026-10-07, branch `ising-bb-gross2`. Revised after the review of the first version (VERDICT
REVISE; [review text not included]); the
study lead adopted the review's recommended design. The re-review of that revision (VERDICT REVISE;
[review text not included]) is handled in section 9. Stage 1's results: `RESULTS_gross2_stage1.md`. Gross's registered
probe stays FAILED; nothing here is registrable.

Budget: the authors allow $100 of RunPod for the whole night, and study 4 may take up to $68. Stage 2 gets at most 6.5
pod-hours in total, from the pod's resume (or a new pod's creation) to its stop, setup and copy-back included. Stage
1's pod (one H100 80GB HBM3) was stopped at 02:14 PDT. No data-parallel training.

## 1. Design

One continuous D12 run, the trunk `g2_D12`:
- 12 trunk blocks, bottleneck 32, no widening. Warm start from stage 1's D6 (`ckpt/g2_A3D/final_ema.pt`): blocks 1-6
  copied, blocks 7-12 with zero up-projections, so the network equals D6 at step 0 (bb.train checks it).
- Run id 51, lr 3e-4, warm-up 1,000 steps, then flat (`--decay-start 1`): the stable phase of a warm-up/stable/decay
  schedule. p = 0.004, batch 1024, bf16, compile, 20 sampler processes, no tuning flags (stage 1's S0).
- It pauses (`bb.train --stop-at S`, exit 3) at 2e7 and 5e7 samples and at an end sized from the measured
  throughput. At each pause bb.train writes the EMA network as `ema_<samples>.pt`. The optimizer, the EMA and the
  data stream continue across pauses and sessions (`--resume`).

Cooldowns, the only networks that are evaluated or selected:
- From each pause's EMA, a branch run (run id 53, so every branch sees the same samples): a 200-step warm-up to 3e-4,
  then the cosine to 2 %. 3e6 samples at the two inspections, 1e7 (~14 % of the trunk) at the end.
- A cooldown starts a fresh AdamW from the trunk's EMA (bb.train's existing `--init` path), not from the trunk's
  optimizer state. The procedure is the same at every pause, so cooldowns compare like with like, and the trunk's
  own optimizer and EMA are never reset. Annealed networks are compared only with annealed networks; D6's final was
  annealed by its own schedule.
- Each cooldown network gets stage 1's reads: graph32 and fp32 on the 50k D-bank (p = 0.004), fp32 on the 20k bank
  at p = 0.003. It is paired (exact McNemar, Newcombe 95 %) with the best network so far.

Selection and stops:
- The best verified network: D6 is the floor. A cooldown replaces the best only with strictly fewer paired D-bank
  failures, so the final network is never a regression. The best (run, D-bank failures, checkpoint path, sha256)
  and every comparison are written atomically to `out/s2_best.json` after each comparison and read from there; no
  failure (an error, a non-finite loss, the cutoff) resets it.
- Futility: the trunk stops if two cooldowns in a row, inspections and session ends alike, each miss a 5 % paired
  gain on the best so far (fewer failures by at least 5 % with the interval below zero). It is checked after every
  comparison and at a session's start before any extension. The earliest case is 5e7: C20 against D6 and C50
  against the best are both flat. A stopped trunk is not resumed.
- Restarts: each pause is registered with its cooldown's length (`out/s2_points.json`) before the trunk trains
  towards it. A session first recovers: registered pauses the trunk did not reach are dropped; the cooldowns of
  pauses it reached are finished (a cooldown cut off resumes; a snapshot whose export was cut is written again from
  `last.pt`'s EMA) and compared in trunk order. Only then is futility checked and the trunk's new end sized from the
  time left. If a cooldown is still unfinished or uncompared (no time, an error), the trunk does not train: the
  session reports the persisted best and stops.
- The GPU term (batch 1, graph32) must be at most 5 ms for D12 before any training (the smoke) and for the selected
  network at the frontier; above it, D6 is selected.

The smoke, before any training (~10 min): 2e6 samples of D12 from D6 on the trunk's schedule. It records the steady
samples/s and the data wait, the peak memory (`torch.cuda.max_memory_allocated`, the backward pass included),
bb.train's step-0 check against D6 (decisions and scores equal on the 20k monitor shots), and the norms of the six
new up-projections, which must have left zero. Then D12's GPU term (bb.latency, graph32, one accepted process). The
trunk starts only if all of this passes. Its measured rate sizes the trunk.

Not adopted in this session: the review's optional benchmark of `channels_last_3d` and cuDNN autotuning. A second smoke
variant costs ~5-8 min (~1.5-2.4e6 samples) for an unverified gain; it is a candidate for a follow-on session.

## 2. Budget and projected samples (resume path)

S2 = the pod's resume. DEADLINE = S2 + 6.5 h; the driver runs with `G2_SINCE=S2 G2_CAP_H=6.5`, so its limit is the
same S2 + 6.5 h. Its cutoff is S2 + 6.0 h; the keeper's final copy-back ends by S2 + 6.17 h at the latest. The trunk's
end E is the furthest chunk boundary such that the trunk, the pending inspections (a resume, a 3e6 cooldown and its
reads each), the final 1e7 cooldown and its reads, and the frontier end before the cutoff with the 10-min copy-back
allowance (`bb.gross2.plan_end`). An inspection that does not fit is dropped and caps E: the trunk then ends at or
before it (at it, it is the final 1e7 pause), so no inspection is passed without a cooldown.

Projected at 4,961 samples/s (D12, linear per block from stage 1's L3 and D6; the smoke's rate replaces it):

| step | samples | hours | ends at S2 + (h) | plain meaning |
|---|---|---|---|---|
| resume, setup (venv from the lock, code, probe) | | 0.20-0.25 | 0.25 | container disk is new after a stop |
| smoke + D12 GPU term | 2e6 | 0.17 | 0.42 | measured rate, memory, learning, step 0 |
| trunk to 2e7 | 2.0e7 | 1.14 | 1.56 | |
| cooldown C20 + reads | 3e6 | 0.25 | 1.81 | inspection 1 |
| trunk to 5e7 | 3.0e7 | 1.70 | 3.51 | |
| cooldown C50 + reads | 3e6 | 0.25 | 3.76 | inspection 2; futility possible |
| trunk to E ~ 7.3e7 | ~2.3e7 | 1.29 | 5.05 | sized at the smoke |
| final cooldown + reads | 1e7 | 0.65 | 5.70 | the final candidate |
| fine frontier of the best | | 0.14 | 5.84 | D6's is CPU only |
| driver exit, final copy-back, pod stop | | ~0.15 | ~6.0 | within 6.5 |

Projected new samples within 6.5 h: the trunk reaches ~7.3e7 (resume path; ~7.0e7 on a new pod, whose setup and seeding
take ~0.1-0.15 h longer), and the final network has ~8.3e7 D12 samples on its path (trunk + final cooldown), plus
6e6 in the two inspection cooldowns. The review's power-law extrapolation of D6's curve puts 1.2 % at ~86-112M additional
samples, so this session may stop short of it; the trunk can be extended later (section 4). Every 1 h more of a
follow-on session adds ~1.8e7 trunk samples.

## 3. The gate and what follows

The frontier is computed on CAL (100k shots, p = 0.004, the split of r21's and stage 1's frontiers) with sector
Relay-BP's stored predictions and per-shot times, the selected network's GPU term measured on the pod (the slowest of
3 accepted processes) and the 15.8 ms limit. The q grid is predefined: stage 1's grid plus q = 0.2 % to 1.0 % in
0.1 % steps (`bb.route frontier --fine`).

D6's fine frontier (computed on the Mac mini from stage 1's CAL read and its measured 1.84 ms GPU term): q = 0.8 %
gives 2,076 failures at 14.40 ms; q = 0.9 % needs 15.89 ms. On the coarse grid the best point was 2,299 at q = 0.5 %.
Relay-BP spends ~1.57 s per routed shot there. With D12's projected 3.3 ms GPU term the boundary moves to q ~ 0.7 %,
where D6's margin removes ~530 failures; the network alone then needs roughly 1,000-1,100 / 100k for the hybrid to
reach 490, and the review's estimate (at most 872-989 at q = 0.5 %) shows how quickly that tightens if fewer shots fit.

Gate (merged plan): the selected network alone <= 1,500 / 100k and the hybrid within 15.8 ms below 490 (the fine
frontier's policy). The driver writes `out/selection_d.json`: the plan, statuses, the trunk's position, every
cooldown with its failures and improvement, the best, futility, the gate and the decision (pass, fail or undetermined).
- Pass: no more training; a fresh registration (frozen network, routing threshold fixed on CAL, an untouched test
  set, the latency protocol), reviewed before it runs.
- Fail with progress (the last cooldowns still improve on the best): the study lead may extend the trunk in a
  follow-on session if the night's money allows (section 4).
- Fail without progress (futility, or no cooldown beats D6): gross stops; report honestly.

## 4. Follow-on sessions (extension)

A follow-on session runs the same commit with a new S2', DEADLINE' = S2' + X h and `G2_SINCE=S2' G2_CAP_H=X`. The
driver skips the done smoke, recovers (section 1, Restarts), resumes the trunk from its last checkpoint (a pause, or
the last evaluation if the trunk was cut), sizes a new end from the new budget, cools the trunk down there (1e7),
keeps the best of all cooldowns and evaluates the fine frontier of the best. Inspections are not repeated. On a new
pod the trunk's `last.pt` comes from the Linux workstation: the driver records it at every pause and at its end, and the keeper
copies it; `out/s2_points.json` and `out/s2_best.json` come with `out/`.

## 5. Code (stage 1's behaviour and every default unchanged)

- `bb.train --stop-at S`: a pause like `--max-minutes` at the first chunk boundary at or after S samples (exit 3,
  `last.pt`), with the EMA written as `ema_<samples>.pt`; a resumed segment may set another S. A `memory` line
  (peak allocated and reserved GB) on CUDA.
- `bb.route frontier --fine`: q = 0.2 %, 0.3 %, ..., 1.0 % added to the frozen grid.
- `bb.gross2` stage `d`: `smoke2` (with D12's GPU term), `deep` (D12's terms, the smoke's measurements over the
  projections), `plan_end`, `trunk_seg`, the pause registry (`points`), `snapshot`, `branch` (one cooldown),
  `compare` and `best_state` (the persisted best), the futility rule,
  `fine_frontier`, the gate, `out/selection_d.json`, `out/p003_stage2.json`. The budget: `G2_SINCE` and `G2_CAP_H`.
- `gross2_setup.sh`: the input push extracts with `--no-same-owner` (stage 1's first setup failed on MooseFS's chown).
- Tests (Mac mini CPU, 02:26-02:42 PDT): `bb.test_gross2_driver` ALL PASS, 19 cases, among them trunk (improving:
  the final cooldown is best; a final regression keeps C50; two flat cooldowns stop the trunk at 5e7), extend (a
  second session resumes from E1 = 104.1M to E2 = 130.1M, no new inspections; a stopped trunk stays stopped), s2fail
  (new blocks stuck at zero or a 6 ms GPU term: no training; a trunk that diverges keeps C20), since. `bb.test_gross2`
  ALL PASS (D6 to 9 and 12 blocks: 24 and 48 new keys, step 0 bit-equal). `bb.train --stop-at 8192` on L3: exit 3
  with `ema_8192.pt`, `--resume` to 16,384 samples (exit 0, 2 segments), and a 12-block warm start from the snapshot
  (72 new keys, step 0 equal, max score difference 0.0). `bb.test_gross2_train` ALL PASS. `bb.route frontier --fine`
  on D6's real CAL read gives the numbers in section 3.
- Tests after the re-review (Mac mini CPU, 03:11-03:16 PDT; 03:27, 28 cases, after the confirmation's two fixes):
  `bb.test_gross2_driver` ALL PASS, 26 cases; the new
  ones inject faults at the cutoff and at each restart point (section 9). A real `bb.train --stop-at 8192` at batch
  128 with its snapshot removed afterwards: `snapshot` writes it again from `last.pt`, bit-equal to bb.train's own
  export (46 tensors, cfg, hp, samples).

## 6. Operation: the exact commands

`POD_SSH` is the pod's ssh command; for stage 1's pod
`POD_SSH="ssh -i <key> -p PORT root@HOST"` with the port and host RunPod gives after the resume.
Never study 4's pod.

1. Keep stage 1's verdict, then fix S2 when the pod is up (resumed or created) and write the deadline; the
   study lead's watchdog stops the pod at it:
   `cp ~/work/stage_gross2/pod_out/COPYBACK.txt ~/work/stage_gross2/COPYBACK_stage1.txt`
   `S2=$(date +%s); echo $S2 > ~/work/stage_gross2/S2_START; echo $(( S2 + 23400 )) > ~/work/stage_gross2/DEADLINE`
2. A clean clone at the gated commit (stage 1's clone stays at 5515f709):
   `git clone -q ~/work/repo ~/work/clone && git -C ~/work/clone checkout -q --detach COMMIT`
3. New pod only: seed stage 1's state from the Linux workstation (the records, D6 and every other final network, the reads, the
   D-bank and its rows, the ledger), then setup pushes the 47 inputs:
   `tar cf - -C ~/work/stage_gross2/pod_out out ckpt preds shots ledger.csv | $POD_SSH 'mkdir -p /workspace/g2 && tar xf - --no-same-owner -C /workspace/g2 && rm -f /workspace/g2/out/DRIVER_EXIT.json'`
4. Setup, both paths (code re-sync, ledger union, DEADLINE, the 47 inputs checked and sent if missing, the venv from
   the lock, the probe): `cd ~/work/clone && POD_SSH="$POD_SSH" bash ising_compare/bb/scripts/pod/gross2_setup.sh`.
   It must end with `environment: every step done`, and the probe must say `accepts`.
5. The driver:
   `$POD_SSH "cd /workspace/g2 && rm -f out/DRIVER_EXIT.json && G2_SINCE=$(cat ~/work/stage_gross2/S2_START) G2_CAP_H=6.5 setsid -f bash code/ising_compare/bb/scripts/pod/gross2_run.sh d >> logs/run_d.log 2>&1 < /dev/null"`
   (double quotes: S2 is filled in on the Linux workstation). Its first line must give `limit` = S2 + 6.5 h.
6. At once, the keeper: `cd ~/work/clone && POD_SSH="$POD_SSH" setsid -f bash ising_compare/bb/scripts/pod/gross2_keep.sh >> ~/work/stage_gross2/keep.log 2>&1 < /dev/null`
7. Watch: `grep -E "^(==|smoke|plan|g2_|pair|BUDGET|DEADLINE|futility|WARNING|ERROR|frontier|selection|p003)" ~/work/stage_gross2/pod_out/logs/run_d.log`.
   Expected: `smoke D12` (rate, memory, up-projections, step 0) within ~25 min of S2, then `plan` with the trunk's end,
   the trunk's pauses at 20.0M and 50.0M samples, each followed by a cooldown and a pair.
8. When `pod_out/COPYBACK.txt` says complete: stop the pod (the watchdog stops it at DEADLINE in any case).

What the driver runs (`COMMON` as in stage 1):
```
smoke     $PY -m bb.train --run g2_smoke_D12 --run-id 81 $COMMON --samples 2e6 --warmup 1000 --eval-every 1e9 --profile 300:100 --init $D6 --lr 3e-4 --decay-start 1 --blocks 12
trunk     $PY -m bb.train --run g2_D12 $COMMON --samples 1e9 --run-id 51 --warmup 1000 --eval-every 5e6 --init $D6 --lr 3e-4 --decay-start 1 --blocks 12 --stop-at S [--resume]
cooldown  $PY -m bb.train --run g2_D12k<M>c<K> $COMMON --samples K --run-id 53 --warmup 200 --eval-every 1e6 --lr 3e-4 --decay-start 0 --init /workspace/g2/ckpt/g2_D12/ema_<at>.pt --blocks 12
```
`--samples 1e9` only sets the flat schedule's horizon; the trunk always pauses before it. Each cooldown then gets
stage 1's four reads and a pair with the best so far; the best gets `do_frontier` (its CAL read and 3 accepted
GPU-term processes) and the fine frontier; D6 gets the fine frontier on stage 1's read.

## 7. Risks

- A cooldown with a fresh optimizer approximates the trunk's own decay; it is the same at every pause, so the
  comparisons hold, but the absolute numbers may differ slightly from an in-run decay.
- The EMA (decay 0.999, ~1,000 steps) spans about a third of a 3e6 cooldown (2,930 steps); the cooldown network is
  the EMA, as in stage 1.
- Selection over several cooldowns on one D-bank favours the best by chance; the CAL frontier is a second split, and
  the gap between candidates is reported with paired intervals.
- Memory and rate are known only after the smoke; an out-of-memory smoke ends stage 2 with D6 as the result.
- The resume path reinstalls the venv on the new container disk; if RunPod cannot resume the pod (no free GPU on the
  host), the new-pod path needs ~0.1-0.15 h more and a new ssh host and port.

## 8. The review's findings (2026-10-07 02:13) and their handling

| # | finding | handling |
|---|---|---|
| 1 HIGH | a regression could become the final model | best verified network kept, D6 the floor; replacement only with strictly fewer paired failures (test: regress) |
| 2 HIGH | the coarse q grid undermines the target | predefined fine grid q 0.2-1.0 % (`--fine`); D6's fine frontier: 2,076 at q 0.8 % (14.40 ms) |
| 3 HIGH | futility compared peak-LR with annealed networks | cooldowns before every comparison; futility only after two flat cooldowns in a row (test: futile) |
| 4 MED | failure handling discarded usable work | one run, no race; every usable cooldown kept; GPU-term check on the selected network, D6 as fallback (test: s2fail) |
| 5 MED | 2e7 answers only the short horizon | no D9/D12 race; nothing is decided at 2e7 alone |
| 6 MED | measure D12 before committing hours | the smoke (memory, rate, learning, step 0) and D12's GPU term gate the trunk |
| 7 MED | stage 1's scope | the results now say "at fixed samples", no capacity-vs-compute or seed claim, warm/scratch gaps mixed |
| 8 MED | 1.2 % plausible, unproven | stated in section 2; the gate is unchanged; the extension path exists |
| 9 MED | the fixed 1e8 cap was not budget-optimal | the trunk's end is sized from the measured rate and the remaining time |
| 10 LOW | conservative arithmetic | kept; `G2_SINCE` makes the driver's limit equal the DEADLINE |
| 11 MED | paid idle review time | the pod was stopped at 02:14; the 6.5 h count from the resume |
| 12 LOW | warm start and keeper consistent | no change |
| 13 MED | benchmark channels_last_3d and cuDNN autotuning | not in this session (optional; ~5-8 min for an unverified gain); follow-on candidate |

## 9. The re-review (2026-10-07 02:54) and its handling

| # | finding | handling (test in `bb.test_gross2_driver`) |
|---|---|---|
| 1 HIGH | the cutoff reset the best to D6 | the best lives in `out/s2_best.json`, written after each comparison and never reset; the selection reads it (cut: the trunk cut on its way to 5e7 keeps C20, its failures, path and sha256) |
| 2 HIGH | a restart sized E before the overdue cooldowns | recovery first (finish and compare every reached pause's cooldown, drop unreached pauses), then futility, then `plan_end` from the time left; an unfinished cooldown stops the trunk (recover: the cooldown runs before `plan_end` and the trunk; with 6 min left nothing trains) |
| 3 MED | E could pass a dropped inspection | a dropped inspection caps E; at it, it is the final pause (plan: E 4.90e7 with C50, 5.20e7 without, now ends at 5.0004e7 with 1e7; a sweep of 3,000 budgets passes no inspection) |
| 4 MED | futility skipped the final cooldowns | any two flat cooldowns in a row, checked after each comparison and before any extension (futfinal: flat C50 and final stop the trunk, also when the final is compared only by the next session) |
| 5 MED | an interrupted final cooldown could be abandoned | registered pauses are recovered before any extension (final: the cut final cooldown runs before the trunk resumes) |
| 6 MED | a pause without its exported snapshot | `snapshot` writes it again from `last.pt`'s EMA when `last.pt` is at the pause (snap; real bb.train check bit-equal) |
| 7 MED | at/k were added after the reads | the registry holds at and k before the trunk trains towards the pause (meta: a failed cooldown, then a cut comparison, are finished with the registered at and k) |
| 8 LOW | first review's #2 and #3 | Resolved per the review; no change |

The review's confirmation of 8ea11932 (REVISE): five resolved, two fixed:

| finding | handling (test) |
|---|---|
| new HIGH: a failed comparison let the trunk train on | a cooldown left without its comparison is an error and ends the session's trunk training; a restart compares it again (nocompare: the pair exits 1, the trunk stays at 2e7; the restart compares and goes on) |
| #4 partial: batched comparisons checked futility on the last two only | `compare` checks futility after each comparison in trunk order and stops there; no later cooldown is compared (batch: [flat, flat, improved] stops at the second; the third is not compared, no trunk training) |
