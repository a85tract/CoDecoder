# gross2, stage 1: plan of the pod session (exploratory)

2026-10-06, branch `ising-bb-gross2` from `ising-bb` 01755dde. It implements the study lead's merged plan
(2026-10-06 ~17:45 PDT, after a two-round design discussion). Gross's registered
probe (PLAN.md 2.3) stays FAILED: r21 alone failed 7,403 of the 100,000 cal shots, sector Relay-BP 490. Nothing here is
registrable, and any later claim needs a fresh registration and an untouched test set. The session spends at most
8 H100-hours on one RunPod H100. This document and the diff are reviewed before the pod starts.
The stage-gate review on this plan (2026-10-06, 14 findings, verdict REVISE) and the study lead's dispositions are
in section 9; this revision implements them, and the pod session runs only after it.

## 1. Code (defaults unchanged)

| change | where | default |
|---|---|---|
| `--decay-start F`: linear warm-up, the peak LR until F of the steps, then the cosine to 2 % (F = 1: flat) | `bb.train.lr_factor` | 0 = r21's schedule |
| `--init PATH`: warm start; non-strict load where only the new modules' keys may be missing and every old key must load; the init network's base (code, widths, heads, taps) must match; step-0 decisions and scores must equal the init network's on the monitor shots, else exit; fresh optimizer and EMA (the EMA starts at the loaded weights) | `bb.train.warm_start`, `step0_check` | none |
| `--tblocks 4,8`: time-only residual bottleneck blocks after the trunk: LayerNorm, 128 -> 32, conv kernel (3,1,1) with dilation (d,1,1) and zero time padding, 32 -> 128 zero-initialised | `bb.net.Block(dil=d)` | none |
| `--blocks 6` from r21: blocks 1-3 copied, blocks 4-6 new with zero up-projections (the existing Block initialisation) | existing flag + `--init` | 3 |
| `--rbins 0,4,9,13`: read-out branch R (conditional arm only): per query, the head's pooled summary kept apart per time bin [0:4], [4:9], [9:13] (the same gate and position weights) -> LayerNorm(384) -> 128 -> SiLU -> 1, final linear zero-initialised, added to the logit | `bb.net.TranslateQueries(bins)` | none |
| `macs()` counts the time blocks and R; `bb.checks` and `bb.latency` build every shape | `bb.net`, `bb.checks`, `bb.latency` | |
| `bb.engine` refuses the fused rung for T and R; new shapes run graph32 (fp32 PyTorch) | `bb.engine.Deployed` | |
| `bb.route pair`: paired comparison of two exploratory network reads (exact McNemar, Newcombe 95 %; never dev or test) | `bb.route.pair` | |
| `--resume` treats a setting added after the checkpoint as its default | `bb.train` | |
| `--profile S:K`, `eval_s` in every eval row, each step line's rate and data wait over its last 100 steps (measurement only) | `bb.train` | off |
| `--lean`: the labels take the forward's detector parity (the same bits) and the EMA is one foreach update (gate #13) | `bb.train`, `bb.net.labels(g)` | off |
| `--chunk-h2d`: one pinned host-to-device copy per chunk; the batches are slices of it (gate #14) | `bb.train` | off |
| final files written to a temporary name and renamed, `log.json` last; `--resume` of a run that ended but was cut off before its export exports again (gate #5) | `bb.train` | |
| the pod driver: stages, budget, ledger, completion records, statuses, selection (gate #1-#12) | `bb.gross2` (new), started by `scripts/pod/gross2_run.sh` | |
| the keeper on the Linux workstation: copy-back and ledger (gate #1, #6); the secured input list (gate #10) | `scripts/pod/gross2_keep.sh`, `gross2_inputs.sha256` (new) | |

The new modules are built after every old one, so a shape without them builds r21's network: the same cfg, parameter
names and shapes, the same initial weights for a given seed. The time blocks and R also leave the old weights'
initialisation unchanged. As the merged plan says, bf16 training, `torch.compile`, the joint vote and tbias's weight
decay stay as they are.

| shape | flags | active MACs per shot | x r21 | parameters |
|---|---|---|---|---|
| L3 (r21) | none | 486.6 M | 1.00 | 821,085 |
| T | `--tblocks 4,8` | 507.7 M | 1.04 | 844,509 |
| D6 | `--blocks 6` | 759.7 M | 1.56 | 1,455,261 |
| R | `--rbins 0,4,9,13` | 632.7 M | 1.30 | 871,262 |
| W | `--bott 48` | 810.8 M | 1.67 | 1,593,789 |

Active MACs count only the taps that carry weights (`bb.net.macs`). The masked dense conv3d does more: about 826 M
for L3.

Tests without a GPU (Mac mini, CPU fp32; all pass, section 9 gives the run):
- `bb.checks` on its default test network (2 blocks, width 32) and with T, D6, R and all three combined: equivariance
  error at most 1.2e-15, true vote and strict bits;
- `bb/test_gross2.py`: r21's command line parsed by the new `bb.train` gives r21's stored hp plus only the new
  settings at their defaults; r21's cfg rebuilds its cfg, parameter names, shapes, MACs and parameter count; with
  01755dde's `net.py` the same seed gives bit-equal weights (L3, and the old weights of T and R), and r21's weights
  give bit-equal scores; `lr_factor` at decay-start 0 equals the old formula at all 97,656 steps of r21's run (A0 is
  flat at 2e-5 after warm-up; A1 holds 3e-4 to 0.8, then 1.53e-4 at 0.9 and 6e-6 at the end; tier B holds 1e-3 to
  0.75, then 3.6e-4 at 0.9); warm starts of L3, T, D6, R from r21 miss only the new keys (0, 16, 24, 6) and are
  bit-equal to r21 at step 0 on 300 monitor shots; `--init` refuses fewer blocks, another bottleneck and a checkpoint
  that lacks an old key; the engine refuses fused for T and R; `pair` gives the expected counts on synthetic reads;
- 8-step `bb.train` runs: T warm-started with decay-start 0.8 and L3 from scratch with the ramp and decay-start 0.75
  complete, and the T network decodes through `bb.engine`'s fp32 path. The plain T run is bit-equal to the same run
  of 5d9d2527's code;
- `bb/test_gross2_train.py`: T for 8 steps plainly, with `--lean`, with `--chunk-h2d` and with `--profile 2:4` ends
  bit-equal (the profile and `eval_s` lines are there); a run whose final files are removed after it ended is
  exported again by `--resume`, bit-equal, with one row in `runs_bb.csv`;
- `bb/test_gross2_driver.py`: the driver's own code on simulated commands (section 9 lists the cases).

S0 repeats the checks on the pod's GPU (section 3). The compiled smokes, graph32 and the GPU terms need the H100.

## 2. Banks and seeds

| bank | split, chunks | shots | p | seeds (`bb.shots.seed_of`) | use |
|---|---|---|---|---|---|
| D-bank | monitor 10-14 | 50,000 | 0.004 | 2^41 + 3e8 + 1e7 + 3e6 + j = 2,199,336,255,562-566 | decisions: graph32 reads of all 50k; curves: its 20k prefix (chunks 10-11) every 5e6 samples in `bb.train` (bf16) |
| D-bank | monitor 10-11 | 20,000 | 0.003 | 2,199,333,255,562-563 | effective-distance diagnostic only, with binomial uncertainty; never a decision input |
| F-bank | cal 0-9 | 100,000 | 0.004 | 2,199,136,255,552-561 (stored on Alpine, in the manifest) | the final frontier of the selected arm and its control only (sector Relay-BP's stored reads and per-shot times) |

- Disjoint: every training chunk seed is at least 2^42 (`chunk_seed`: 2^42 + 1e9 (10 run id + p index) + c). r21's
  monitor chunk is j = 0, and dev, cal and smoke are other splits.
- Stim's bits depend on the host CPU, so the D-bank is sampled once on the pod in S0. Its sha256 rows go to
  `out/rows_dbank.csv` and to the pod's manifest copy, and every later read is sha256-checked.
- Training streams, each shared by all arms of a tier: tier A run id 31 (p 0.004, no ramp), tier B run id 41 (ramp),
  smokes run id 81. r21 was 21.

- The D-bank cannot be sampled again bit for bit on another CPU, so its 7 files and `rows_dbank.csv` are copied
  back to the Linux workstation by the keeper within 5 minutes of S0 recording them, and checked there again at the end.

## 3. Runs: the exact command lines

The driver `bb.gross2` (started by `bb/scripts/pod/gross2_run.sh STAGE`, STAGE = `s0`, `a`, `b`, `f` or `all`) runs
these lines unchanged, stage by stage. A step is done only if its completion record (section 6) still matches its
files; a run cut off resumes from its `last.pt`. Logs go to `/workspace/g2/logs`, results to `/workspace/g2/out`.

On the pod, in `/workspace/g2/code/ising_compare` (the wrapper sets this environment):
```
export PYTHONPATH=/workspace/g2/code/ising_compare:/workspace/g2/code/qldpc_neural QND_RELAY=/workspace/g2/relay \
  BB_SHOTS=/workspace/g2/shots BB_PREDS=/workspace/g2/preds BB_CKPT=/workspace/g2/ckpt BB_OUT=/workspace/g2/out \
  TRITON_CACHE_DIR=/workspace/g2/triton_cache OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 G2_ROOT=/workspace/g2 \
  G2_PIDNS=1
PY=<pod-home>/g2venv/bin/python; R21=/workspace/g2/ckpt/r21/final_ema.pt
WORKERS=<min(the CPUs of the process's affinity, the cgroup quota) - 2>
COMMON="--code bb144 --p 0.004 --design data/bb/design_bb144.json --compile --workers $WORKERS --monitor 10-11 --monitor-shots 10000"
TUNE=<S0's flags, the same for every arm of both tiers: none, --lean and/or --chunk-h2d (section 6)>
```

### S0

Target 0.5 h; any failure stops the session.
```
$PY -m bb.checks --code bb144 --p 0.004 --design data/bb/design_bb144.json            # and with each of: --tblocks 4,8 |
                                                     # --blocks 6 | --rbins 0,4,9,13 | --blocks 6 --tblocks 4,8 --rbins 0,4,9,13
$PY -m bb.test_gross2 --init $R21 --log /workspace/g2/ckpt/r21/log.json --design data/bb/design_bb144.json --dev cuda
$PY -m bb.shots --code bb144 --p 0.004 --split monitor --chunks 10-14 --rows /workspace/g2/out/rows_dbank.csv
$PY -m bb.shots --code bb144 --p 0.003 --split monitor --chunks 10-11 --rows /workspace/g2/out/rows_dbank.csv
#   (the rows are appended to data/bb/shot_manifest.csv; a cut-off sampling starts over)
$PY -m bb.route infer --ckpt $R21 --code bb144 --p 0.004 --split monitor --chunks 10-14 --tag r21 --engine graph32
$PY -m bb.route infer --ckpt $R21 --code bb144 --p 0.004 --split monitor --chunks 10-14 --tag r21_f --engine fp32
$PY -m bb.route infer --ckpt $R21 --code bb144 --p 0.003 --split monitor --chunks 10-11 --tag r21_f --engine fp32
$PY -m bb.route pair --code bb144 --p 0.004 --split monitor --chunks 10-14 --tags r21,r21_f --min-reduction 0.25 \
  --out /workspace/g2/out/pair_r21_r21_f.json      # these four are timed: the projections' read time
$PY -m bb.route infer --ckpt $R21 --code bb144 --p 0.004 --split cal --chunks 0-9 --tag r21_h100 --engine fp32
$PY -m bb.route pair --code bb144 --p 0.004 --split cal --chunks 0-9 --tags r21_h100,r21 --min-reduction 0.25 \
  --out /workspace/g2/out/pair_r21_h100_r21.json     # against the stored RTX PRO 6000 read (7,403): stop if > 100 differ
# for X in L3 T D6 R W (flags: none | --tblocks 4,8 | --blocks 6 | --rbins 0,4,9,13 | --bott 48):
$PY -m bb.train --run g2_smoke_X --run-id 81 $COMMON --samples 1e6 --warmup 1000 --eval-every 1e9 --profile 300:100 \
  <L3 T D6 R: --init $R21 --lr 3e-4 --decay-start 0.8; W: from scratch at lr 1e-3> <flags X>
$PY -m bb.latency --code bb144 --p 0.004 --design data/bb/design_bb144.json --engine graph32 <flags X> --proc I \
  --out /workspace/g2/out/latency_X.csv              # until one process is accepted, at most 3 (I = 1, 2, 3)
```
Then the summary writes `out/s0.json`: per shape the rate, the evaluation time, the fixed time, the GPU term, the
reads' and the frontier's time; the tuning; N (section 6). It stops if N < 1e7. W is measured like the others (gate #8).
A GPU-term process counts only if `gpu_watch` accepted it (every SM-clock sample at least 97 % of the maximum, no
other process on the GPU) and its rows carry this pod's graph32 engine signature with random weights (r21's D-bank
read's signature with the network replaced by `random000000`, gate #7).

### Tier A

Warm start from r21's final_ema.pt, p 0.004, no ramp, run id 31 for every arm, N samples, warm-up 1,000:
```
A0 F  $PY -m bb.train --run g2_A0F $COMMON --samples $N --run-id 31 --warmup 1000 --eval-every 5e6 --init $R21 --lr 2e-5 --decay-start 1 $TUNE
A1 S  $PY -m bb.train --run g2_A1S $COMMON --samples $N --run-id 31 --warmup 1000 --eval-every 5e6 --init $R21 --lr 3e-4 --decay-start 0.8 $TUNE
A2 T  $PY -m bb.train --run g2_A2T $COMMON --samples $N --run-id 31 --warmup 1000 --eval-every 5e6 --init $R21 --lr 3e-4 --decay-start 0.8 --tblocks 4,8 $TUNE
A3 D  $PY -m bb.train --run g2_A3D $COMMON --samples $N --run-id 31 --warmup 1000 --eval-every 5e6 --init $R21 --lr 3e-4 --decay-start 0.8 --blocks 6 $TUNE
A4 R  $PY -m bb.train --run g2_A4R $COMMON --samples $N --run-id 31 --warmup 1000 --eval-every 5e6 --init $R21 --lr 3e-4 --decay-start 0.8 --rbins 0,4,9,13 $TUNE
```
(`--resume` is added to an arm that was cut off.) A4 runs only if A2 and A3 were both evaluated and neither
qualifies. After each arm RUN, the reads are:
```
$PY -m bb.route infer --ckpt /workspace/g2/ckpt/RUN/final_ema.pt --code bb144 --p 0.004 --split monitor --chunks 10-14 --tag RUN --engine graph32
$PY -m bb.route infer --ckpt /workspace/g2/ckpt/RUN/final_ema.pt --code bb144 --p 0.004 --split monitor --chunks 10-14 --tag RUN_f --engine fp32
$PY -m bb.route infer --ckpt /workspace/g2/ckpt/RUN/final_ema.pt --code bb144 --p 0.003 --split monitor --chunks 10-11 --tag RUN_f --engine fp32
$PY -m bb.route pair --code bb144 --p 0.004 --split monitor --chunks 10-14 --tags RUN,RUN_f   # the batch check (warn if > 50 differ)
```
Comparisons: `bb.route pair ... --tags A,B --out out/pair_A_B.json` for S,F (g2_A1S,g2_A0F), T,S, D,S and R,S at
`--min-reduction 0.25`, and for D,T at `--min-reduction 0.20`, each only when both arms are valid.

### Tier B

From scratch, only if B2's shape has a feasible GPU term in S0 (at most `G_MAX_MS`; else B2 could not qualify, so tier
B is skipped with that reason, round 3 #6) and both arms and the frontier after them fit the remaining budget; run id
41 for both; lr 1e-3 and warm-up 500 (r21's):
```
B1 C  $PY -m bb.train --run g2_B1C $COMMON --samples 3e7 --run-id 41 --p-ramp 0.001:0.4 --decay-start 0.75 --eval-every 1e6 $TUNE
B2 X  $PY -m bb.train --run g2_B2X $COMMON --samples 3e7 --run-id 41 --p-ramp 0.001:0.4 --decay-start 0.75 --eval-every 1e6 <flags X> $TUNE
```
X is tier A's winning shape (D6, T, or R), else W (`--bott 48`). Each arm gets the same four reads, then
`pair g2_B2X,g2_B1C --min-reduction 0.3333333333333333`; the driver's test is the exact one, fail(C) >= 1.5 fail(B2)
(2 fail(C) >= 3 fail(B2) in integers, gate #3). The ramp check reads C's EMA at the first evaluation at or after
1.2e7 samples (the ramp's end).

### Frontier

The final selection's arm and its control, then r21 (its stored RTX PRO 6000 read) as a reference; for RUN in WINNER,
CONTROL, r21:
```
$PY -m bb.route infer --ckpt /workspace/g2/ckpt/RUN/final_ema.pt --code bb144 --p 0.004 --split cal --chunks 0-9 --tag RUN --engine graph32   # not for r21
$PY -m bb.latency --ckpt /workspace/g2/ckpt/RUN/final_ema.pt --engine graph32 --proc I --out /workspace/g2/out/latency_f_RUN.csv
#   until 3 accepted processes bound to RUN's graph32 signature on this pod, at most 6 (I = 1..6)
$PY -m bb.route frontier --code bb144 --p 0.004 --chunks 0-9 --tag RUN --replica 1 --gpu-us G --handoff-us H \
  --gpu-src "bb.latency graph32 on the pod's H100 (slowest of 3 accepted processes)" --out /workspace/g2/out/frontier_RUN.json
```
G and H are the largest GPU term and handoff of the accepted processes, in us. The composite uses sector Relay-BP's
recorded per-shot times on the routed shots, the same CPU terms r21's frontier used. The limit is 15.8 ms.

### Operation

1. The study lead creates the pod, writes `~/work/stage_gross2/DEADLINE` (epoch: the pod's start + 8 h) and
   runs its watchdog, which stops the pod through the RunPod API at DEADLINE and never logs in to it.
2. From the Linux workstation, in a clean clone at the gated commit, setup (section 5):
   `POD_SSH="ssh -i <key> -p PORT root@HOST" bash ising_compare/bb/scripts/pod/gross2_setup.sh`.
   Setup stops while a driver holds `/workspace/g2/driver.lock`. Its environment part must end with `environment:
   every step done`, and its GPU-term probe must say `accepts`; a `REJECTS` line means S0's GPU-term steps would fail
   on this pod, so the driver is not started. The probe line also gives `accepted true|false`, the whole rule (one
   compute app and the SM clock at >= 97 % of its maximum through the probe); report it.
3. The driver: `$POD_SSH 'cd /workspace/g2 && rm -f out/DRIVER_EXIT.json && setsid -f bash code/ising_compare/bb/scripts/pod/gross2_run.sh all >> logs/run_all.log 2>&1 < /dev/null'`
   (appended: a restart keeps the earlier session's log; a second driver does not start while one runs; an earlier
   run's exit record is removed first, so the keeper cannot take it for this run's).
4. At once, the keeper on the Linux workstation (it ends after its final copy-back, section 6; a new keeper ends an earlier one's
   session and takes over; a keeper started after the driver's exit still gives the verdict):
   `POD_SSH="..." setsid -f bash ising_compare/bb/scripts/pod/gross2_keep.sh >> ~/work/stage_gross2/keep.log 2>&1 < /dev/null`.
5. At about 25 min (after S0's `dbank` step in the log), the keeper's latest poll line in `keep.log` must say
   `D-bank 7/7`: `pod_out` holds the D-bank's 7 shot files and they match `rows_dbank.csv` (the D-bank cannot be
   sampled again on another CPU). If it does not by the next poll (5 min), read `keep.log` and start a new keeper
   (step 4; it takes over).
6. To watch (the keeper refreshes the copy every 5 minutes): `grep -E "^(==|s0|tuning|N |selection|pair|BUDGET|DEADLINE|stop-rule|WARNING|ERROR|g2_|ramp|p003|frontier)" ~/work/stage_gross2/pod_out/logs/run_all.log`.
7. When `~/work/stage_gross2/pod_out/COPYBACK.txt` says complete, stop the pod (the watchdog stops it at DEADLINE
   in any case). The copy holds `out`, `logs`, the ledger and every recorded artifact (final networks, logs,
   reads, the D-bank shots), about 90 MB; `last.pt` stays on the pod. INCOMPLETE names what is missing.
After a pod restart: setup again (it merges the ledgers and reinstalls the venv from the lock), then the driver (it
resumes and counts the earlier sessions from the ledger) and the keeper (it takes over from a keeper still running).

## 4. Decision rules and stops

Verbatim from the merged plan:

> ## Decision rules (paired fp32 reads on the 50k D-bank)
> - Schedule lever: S vs F. At least 25 % fewer failures, with the paired 95 % interval of the difference excluding zero.
> - Architecture: T or D vs S. At least 25 % fewer failures, the paired interval excluding zero, and a feasible GPU term
>   (well below 15.8 ms). Prefer D over T only if D gives a further >= 20 % reduction with paired support; otherwise
>   choose the cheaper one.
> - Tier B: B2 must beat C by >= 1.5x at 3e7 to set stage 2's shape.
> - Stops:
>   - non-finite loss;
>   - a variant >= 25 % worse than its control at both the 5M and 10M checkpoints;
>   - a projected overrun of the 8 h.
> - Kill: if no architectural arm qualifies AND the schedule lever gives < 25 %, gross stops. Report honestly; do not
>   propose unchanged long training.
> - Pass: stage 2 (~12-15 H100-h, planned after this stage's results are reviewed) scales the winning axis to ~1.5e8
>   samples with hold-then-cosine. Then a gate on the selected network: alone <= 1.5 %, hybrid < 0.49 % within 15.8 ms on
>   actual routed-shot costs. Only then a fresh registered test.

And from its runs section: "From the measured rates, fix N (chunk-aligned, between 1e7 and 2e7) so that tier A fits
~4.5 h. If N < 1e7, report that the budget is insufficient and stop." "A4 R: A1 + the read-out branch, ONLY if neither
A2 nor A3 qualifies." "Tier B, only if the budget remains after tier A." "Ramp fallback: if C's EMA fails more than
50 % at the ramp's end, report it and do not re-run inside this budget." "The winner's exploratory frontier on the
F-bank, against its control."

How the driver applies them (our reading; points 1, 2, 6 and 8 are open questions in
section 8):
1. Feasible GPU term: at most `G_MAX_MS` = 5 ms, S0's accepted graph32 batch-1 term of the shape on the pod's H100.
   That is about a third of 15.8 ms, and leaves at least ~1.1 % of shots routable at Relay-BP's 0.92 s per routed shot.
2. The stop rule compares the architecture variants with their control (T, D, R against S; B2 against C). It uses
   the EMA's bf16 monitor failures on the 20k prefix at the first evaluations at or after 5e6 and 1e7 samples. A
   watcher stops the run's process group and marks the run STOPPED. S against F is not stopped early: S holds a 15x
   higher LR until 0.8 of the run, so a mid-run deficit is expected.
3. "The paired interval excluding zero" means the Newcombe 95 % interval of LER(variant) - LER(control) lies
   entirely below zero (`bb.route pair`, field `diff_hi < 0`). The reduction is 1 - fail(variant) / fail(control),
   and both together give `qualifies`.
4. D over T: `pair D,T` qualifies at 20 %. Otherwise the cheaper shape is the one with the smaller GPU term.
5. Tier B: fail(C) / fail(B2) >= 1.5 exactly on the 50k graph32 reads, reported with the paired interval; B2 also
   needs a feasible GPU term (point 1; this matters only for W, tier A's shapes are checked there).
6. B2's shape is the qualifying architecture of tier A (D6 or T as above; R if only R qualified), else W.
7. Tier B runs whenever it fits, also after a tier A with no qualifying arm (a guard against a warm-start false null),
   and the final selection (`out/selection.json`) combines both tiers (gate #3; re-review #4):
   - frontier pair: tier A's architecture winner against S; else B2 against C if B2 qualifies; else S against F if only
     the schedule lever qualifies; else no frontier;
   - stage 2's shape: B2's if B2 qualifies; none if tier A's winner X missed 1.5x as B2 = X (X stays the winner and
     the frontier pair, with the caveat "tier A only, unconfirmed from scratch"; the post-stage review decides stage
     2's shape); tier A's winner if tier B did not run or is incomplete; else L3 if only the schedule lever qualifies;
   - decision: pass (some winner); kill (every comparison evaluated, tier B included, and none qualifies);
     inconclusive (some comparison undetermined; a tier B skipped for budget, or because B2's shape has a GPU term
     above `G_MAX_MS` in S0, is undetermined, "from-scratch capacity untested"). Tier A's own selection (`out/selection_a.json`) never says kill: tier B is pending there.
8. After a ramp fallback, B2 still runs (the comparison stays matched), and the fallback is a caveat of the selection.
9. Valid controls (gate #2): a comparison counts only when both arms are valid (trained, recorded, read); a variant
   stopped by the stop rule or by a non-finite loss counts as not qualifying; an arm that failed, did not fit the
   budget or was stopped at the cutoff leaves its comparisons undetermined. Any error makes the session's exit 1, also
   when the frontier then succeeds; a skip or the cutoff gives exit 2.
10. Projected overrun: section 6.
11. The p = 0.003 diagnostic (gate #11): `out/p003_summary.json`, per network the fp32 failures at 0.004 (50k) and
    0.003 (20k) with Clopper-Pearson 95 % intervals, the slope ln(LER(0.004) / LER(0.003)) / ln(4/3) with the range
    from the intervals, and d_eff ~ 2 x slope, with a note that it is never a selection input. The selection code
    does not read it.

## 5. Pod setup

`bb/scripts/pod/gross2_setup.sh` runs from the Linux workstation in a clean clone at the commit to run, while the driver does not
run: `POD_SSH="ssh -i <key> -p PORT root@HOST" bash ising_compare/bb/scripts/pod/gross2_setup.sh`.
It is idempotent: a rerun repeats only what is missing or different. Steps:
0. Budget: it refuses to run without `~/work/stage_gross2/DEADLINE`, copies it to `/workspace/g2/DEADLINE`,
   writes the union of the Linux workstation's ledger and the pod's to both, and adds its own row.
1. Code: `git archive` of HEAD (bb, data/bb, qldpc_neural's qnd, data, bench) to `/workspace/g2/code` (COMMIT file).
   The D-bank's manifest rows are appended again after a re-sync.
2. Inputs (gate #10): the 47 files below are checked on the Linux workstation against the committed list
   `bb/scripts/pod/gross2_inputs.sha256` (Alpine's sha256 for the 33 Alpine files, `qldpc_neural/data/
   circuit_manifest.csv`'s for the 14 circuits) before anything is sent; a staged file that differs stops the setup.
   The pod's copies are then checked against the same list, and sent only if they differ.
3. Environment (gate #4): a uv venv `<pod-home>/g2venv` and its lock `/workspace/g2/env.lock` (copy:
   `~/work/stage_gross2/pod_env.lock`, pushed to a pod that lacks it). The first install takes Python 3.12.14
   (r21's patch; `G2_PYTHON` overrides it if uv cannot provide it) with torch 2.11.0+cu126 (wheel index cu126; its
   triton 3.6.0), numpy 2.2.6, stim 1.16.0, scipy 1.18.1 (r21's `venv_gpu` versions, checked), then writes the lock:
   the Python patch and every installed package. A reinstall (after a pod restart: the container disk is new)
   installs exactly the lock without resolving and must freeze to it, so the packages' digest in the sampler
   environment is unchanged. gcc (build-essential) is installed if missing (Triton compiles a launcher), then a CUDA
   check; the freeze goes to `out/freeze.txt`. The remote part runs as one shell function with its stdin from
   /dev/null (bash reads all of it before it runs, so no command can read the rest of the script from ssh's stdin)
   and ends with `environment: every step done`; setup stops if the venv or the lock is missing afterwards
   (re-review #1).
4. GPU-term probe (re-review #2): in the container's own PID namespace `nvidia-smi --query-compute-apps` lists host
   PIDs, so bb.timing's registered rule (no compute app other than this PID) would reject every GPU-term process.
   `gross2_run.sh` sets `G2_PIDNS=1`, under which a process counts as alone on the GPU iff exactly one compute app is
   listed (itself; Alpine's protocol, without the variable, is unchanged). Setup runs a 10-s CUDA process under this
   rule and prints the result with `accepted` (the whole rule, the clock check included); `REJECTS` means S0's
   GPU-term steps would fail on this pod.
- The pod: one H100 (80 GB) on a RunPod PyTorch image with CUDA >= 12.6 drivers, at least 16 vCPU, a container disk of
  at least 30 GB (the venv) and /workspace of at least 10 GB.
- Sampler processes: the CPUs of the driver's affinity and the cgroup quota, the smaller, minus 2 (gate #12). r21's
  six sampler processes kept up with 7.2k samples/s (0 % data wait) on Alpine's EPYC cores; S0 measures the samplers' capacity.

What the pod needs from this Linux workstation (`~/work/stage_gross2`, secured from Alpine on 2026-10-06, sha256 equal to
Alpine's and to `gross2_inputs.sha256`):

| path | files | bytes | note |
|---|---|---|---|
| ckpt/r21/final_ema.pt | 1 | 3,304,449 | sha256 8543f99b...72fb (r21, completed, 99,999,744 samples) |
| ckpt/r21/log.json | 1 | 9,859 | r21's cfg, MACs, monitor log |
| shots/bb144_p0.004_cal0000-0009.npz | 10 | 21,812,240 | the F-bank |
| shots/bb144_p0.004_monitor0000.npz | 1 | 2,181,224 | r21's monitor chunk (bb.test_gross2) |
| preds/bb144_p0.004_cal000j_relay.npz | 10 | 1,809,820 | sector Relay-BP: pred, per-shot secs |
| preds/bb144_p0.004_cal000j_net-r21.npz | 10 | 1,730,280 | r21's RTX PRO 6000 graph32 read (7,403 failures) |
| relay/tests/testdata/bicycle_bivariate/*144_12_12_memory_X* | 14 | 7,202,374 | relay d185194 circuits, sha256 = qldpc_neural/data/circuit_manifest.csv |

The code archive is about 1 MB more. Kept on the Linux workstation only: r21's last.pt (13 MB), the policy file, the engine job's
choice and timing, logs.

## 6. Budget

The budget is 8.0 H100-hours in all, setup included, enforced twice (gate #1):
- The study lead's watchdog stops the pod through the RunPod API at DEADLINE = the pod's start + 8 h (the epoch in
  `~/work/stage_gross2/DEADLINE`; setup copies it to `/workspace/g2/DEADLINE`).
- The driver works to LIMIT = min(DEADLINE, the session's start + 8 h - PRIOR). The session starts with the container
  (PID 1, read from /proc: the same integer in every call). PRIOR is the earlier sessions' use from the ledger: per
  session, from its start to its last row's end, plus 300 s, so a restart starts from the hours already used. Every
  row carries its session's start, computed from btime, which can step by a second: start keys at most 2 s apart
  are one session (re-review #6). Without the ledger file the driver does not start (setup writes it, empty at
  first), and a second driver does not start while one holds `/workspace/g2/driver.lock` (re-review #7); setup stops
  while one holds it, and a lock error other than a held lock (e.g. ENOLCK) stops the driver with its errno (round 3
  #4).
- The ledger (`/workspace/g2/ledger.csv`, rows `session,start,end,what,state`): the driver adds one row per step
  (each S0 part, each arm's training, its reads, each frontier part) with its status, a heartbeat row every 5
  minutes and its exit; setup adds its row, also when it fails; the keeper adds an `alive` row at every copy-back
  and keeps the union on the Linux workstation (`~/work/stage_gross2/ledger.csv`), which setup pushes back to a restarted
  pod. The driver writes LIMIT and its start (`started`) to `out/budget.json` for the keeper.
- An arm (training and its reads), a read or a frontier part starts only if now + its projection + 10 min (the
  copy-back) is at most LIMIT - 30 min. At LIMIT - 30 min the driver stops whatever still runs (the process group:
  TERM, then KILL to the whole group once its leader has exited or 60 s later) and starts nothing more; a step cut
  there counts as not run (exit 2). With E = min(DEADLINE, LIMIT), the keeper's final copy-back starts as soon as the
  driver of `all` or `f` exits, or at E - 27 min (3 minutes for what the driver writes after its cutoff; a regular
  poll that would run into this window is skipped), and is retried until it succeeds with the exit record of the run
  that `out/budget.json` describes (`out/DRIVER_EXIT.json` with t >= `started`, also when that exit came before the
  keeper started; round 3 #1), or until E - 20 min. `pod_out/COPYBACK.txt` says complete only if that final copy-back
  succeeded, every recorded artifact matches its sha256 on the Linux workstation and `rows_dbank.csv` lists the D-bank's 7 shot
  files, all matching; else INCOMPLETE and why (re-review #3). One keeper at a time: a flock on
  `~/work/stage_gross2/keep.lock`; a new keeper ends the earlier one's session (GNU `timeout`'s process group in
  it too; round 3 #3) and takes over, and removes an earlier COPYBACK.txt and pod_out's earlier `out/budget.json` and
  `out/DRIVER_EXIT.json` (round 3 #2). Every poll line gives `D-bank k/7`, the shot files here that match
  `rows_dbank.csv`.
- Completion (gate #5): a step is done only if its record `out/done/STEP.json` (written last, temp + rename) lists
  files whose sizes and sha256 still match (a comparison's record covers both reads it used). A read counts only if
  all its chunks load with 10,000 decisions and the
  signature of this network and engine; a final network only if it is completed, has the run's samples and was
  trained in its tier's sampler environment. Anything else runs again; a training run cut off resumes from its
  `last.pt`, and one that ended but was cut off before its export is exported again by `--resume`.
- Sampler environment (gate #4): each tier's first arm fixes `out/env_A.json` or `out/env_B.json`
  (`bb.train.sampler_env`: Stim, NumPy, Python, the venv's packages, Stim's output fingerprint; and the code's
  commit, `code/COMMIT`, re-review #7); every later arm of the tier, started or resumed, must match it, else it is an
  error. All arms of a tier therefore share one stream and one code.

Projections, from S0's measurements (gates #8, #9):
- An arm: its remaining samples / v + (its remaining evaluations + 1) x eval_s + fixed_s + its reads. v = the
  smoke's steady rate (the 100-step windows of its second half, compile and queue backlog left out), at most the
  samplers' capacity from the profile; eval_s = the smoke's evaluation (current and EMA networks on the 20k prefix,
  bf16); fixed_s = the smoke's wall time - samples / v - eval_s (start-up, compile, step-0 check). A resumed arm
  counts only the samples after its `last.pt`; a trained arm only its unfinished reads.
- Reads: r21's measured S0 reads (graph32 50k + fp32 50k and 20k + the batch check) x max(1, the shape's GPU term /
  L3's). A frontier part: 2 x r21's graph32 read time x the same factor + 3 latency calls + 120 s.
- N for tier A, chunk-aligned: N = 8192 floor( min(2e7, (A - sum_i (fixed_i + reads_i + 5 eval_i)) / sum_i (1 / v_i)) / 8192 ),
  over A0-A4 (v: L3, L3, T, D6, R), with A = min(4.5 h, the time to the cutoff - the copy-back - the frontier's
  reserve). If N < 1e7: BUDGET INSUFFICIENT, stop.
- Tier B starts only if B2's shape has a feasible GPU term in S0 (else skipped with that reason, not a budget
  shortfall) and C's and B2's remaining time plus the frontier's reserve fit; else it is skipped and noted.

Tuning (gates #12-#14), from S0's L3 profile (`--profile 300:100`: 100 steps after compile, a device sync between the
phases data, h2d, labels, fwd, bwd, ema, other; plus the samplers' steady capacity from their per-chunk times):
`--lean` iff labels + EMA take at least 3 % of a profiled step; `--chunk-h2d` iff h2d takes at least 2 %; the
sampler processes as in section 5; the same flags for every arm of both tiers; batch 1024, bf16 and compile kept. The decision
and its basis (the phase shares, the capacity, the GPU-bound rate, whether the samplers bind) go to `out/s0.json`
("tuning"). Removing a data-wait share f can speed a run up by at most 1 / (1 - f); the review's estimates for #13 and
#14 are 1.00-1.05x and 1.00-1.02x.

Expected pod hours (estimates; no H100 measurement exists; other shapes scaled by active MACs, ~400 s per arm of
fixed time, evaluations and reads; the work ends by LIMIT - 40 min = 7.3 h at the latest):

| L3 samples/s | N | setup + S0 | tier A: A0-A3 (+A4) | tier B: C + B2 | frontier | total |
|---|---|---|---|---|---|---|
| 15k | 2.0e7 | 0.75 h | 2.2 h (+0.6) | 1.7 h (B2 = D6) | 0.3 h | 4.9-5.5 h |
| 10k | 2.0e7 | 0.75 h | 3.0 h (+0.8) | 2.4 h (D6); does not fit after A4 | 0.3 h | 6.5 h; 4.9 h with A4 |
| 7k | 1.68e7 | 0.75 h | 3.5 h (+1.0) | 2.7 h only if B2 = T and no A4 | 0.3 h | 4.6-5.6 h; 7.3 h with tier B |

Best guess: 5-7 h; the watchdog stops the pod at 8 h in any case.

## 7. Risks

- CRITICAL: adaptive exploration cannot overturn gross's failed registered probe. Every result here is exploratory,
  and any claim needs a fresh registration and an untouched test set.
- HIGH: the H100 rate and the pod's CPUs are unmeasured. N, the number of arms and tier B depend on S0. With fewer
  than ~16 vCPU the samplers may cap the rate (S0 measures their capacity).
- HIGH: the pod's Stim bits differ from Alpine's (SIMD width), so all arms of a tier must run in one sampler
  environment. The driver binds every arm to its tier's (section 6); after a move to another host the tier's next
  arm stops with an error, so two streams never mix within a tier.
- HIGH: zero-initialised additions may under-train in 1-2e7 samples, a false null on capacity. Tier B covers depth
  or width from scratch, but only at 3e7.
- HIGH: routed-shot costs change with the architecture and the confidence order. The frontier uses the actual routed
  shots' Relay-BP times and the H100's GPU term, not the registered RTX PRO 6000.
- MED: the keeper must run on the Linux workstation for the whole session; without it nothing is copied back before the pod stops,
  and the D-bank cannot be sampled again elsewhere. Operation starts it right after the driver.
- MED: the acceptance rule of a GPU-term process (SM clock >= 97 % of its maximum, no other process) may not hold on
  a shared RunPod host; S0 then stops after 3 unaccepted processes per shape. In the container's PID namespace "no
  other process" means exactly one compute app listed (`G2_PIDNS=1`, re-review #2); setup's probe shows whether that
  holds on the pod before the driver starts.
- MED: bf16 monitor against fp32 deployment (98.85 % agreement on gross's validation), so decisions use fp32 graph32
  reads with a batched check.
- MED: new shapes run graph32 only (fused refused). D6's GPU term is about 1.5x r21's, and each ms costs about 0.1 %
  of routable shots.
- MED: the D-bank's 20k prefix is seen every 5e6 samples and all 50k at every decision, which invites selection bias.
  The F-bank is read once at the end, for the frontier only.
- MED: cu126 wheels need a driver that supports CUDA 12.6, and Triton needs gcc. The setup checks both, and S0's
  compiled smokes exercise them. A reinstall needs the indexes to still serve the locked versions.
- LOW: the stop rule kills a process group; a STOPPED run is never resumed.

## 8. Open questions for the study lead (with the driver's current choice)

1. Feasible GPU term = at most 5 ms (`G_MAX_MS`)? Expected terms: ~2 ms (L3, T), ~3 ms (D6), ~2.5 ms (R).
2. The stop rule only for the architecture variants (T, D, R against S; B2 against C), never S against F?
3. If only R qualifies in tier A, B2 = R (the merged plan names D6 or T, else W)?
4. Settled by gate #3: tier B runs after an empty tier A, and B2's 1.5x counts in the kill rule.
5. After a ramp fallback, B2 still runs (reported, not re-run)?
6. "Reproduce r21's count on the D-bank": the D-bank is new, so S0 measures r21's D-bank baseline and reproduces
   r21's stored F-bank read (7,403 of 100,000) on the H100 (stop if more than 100 decisions differ).
7. Settled by the re-review (#4): when tier A's architecture winner X does not beat C by 1.5x as B2 = X, X stays the
   winner and the frontier pair, labelled "tier A only, unconfirmed from scratch", and stage 2's shape is left open
   (None) for the post-stage review.
8. Settled by the re-review (#4): tier B skipped for budget after a tier A with no qualifying arm gives inconclusive
   ("from-scratch capacity untested"), not kill; so does a tier B skipped because B2's shape (W) has a GPU term above
   `G_MAX_MS` in S0 (round 3 #6).

## 9. Gate dispositions

the stage-gate review on PLAN_gross2.md and the diff 01755dde..5d9d2527 (2026-10-06):
14 findings, verdict REVISE. The study lead accepted all 14 (#13 and #14 only as uniform options that S0 switches on
when the profile shows the bottleneck). Commits: C1 = 14180241, the code (`bb.train`, `bb.net`, `bb.gross2`,
`bb/test_gross2_driver.py`); C2 = 522c1de6, the pod scripts; C3 = the commit of this plan.

| # | severity | finding | disposition and implementation | test |
|---|---|---|---|---|
| 1 | CRITICAL | the 8-h cap was an estimate; setup, S0 and running processes had no deadline; `PRIOR_H` defaulted to 0 | accepted. Two limits: the study lead's watchdog stops the pod at DEADLINE; the driver works to LIMIT = min(DEADLINE, session start + 8 h - PRIOR), starts a step only if its end + the copy-back is >= 30 min before LIMIT, stops what runs at LIMIT - 30 min; the keeper's final copy-back ends by DEADLINE - 20 min; PRIOR from the ledger (section 6). C1 (`bb.gross2`), C2 (`gross2_keep.sh`, `gross2_setup.sh`) | ledger, overrun, deadline |
| 2 | HIGH | failures swallowed: every tier-A arm could fail with exit 0; a tier-B crash hidden by the frontier | accepted. Statuses ok / skip / stop / deadline / error; a comparison needs both arms valid; exit 1 on any error, also when the frontier then succeeds (section 4, point 9). C1 | failure |
| 3 | HIGH | tier B never updated the selection or the kill; rounded 0.3333 | accepted. The final selection combines both tiers (section 4, point 7); B2 qualifies iff 2 fail(C) >= 3 fail(B2). C1 | tierb (1,500/1,000 passes; 30,001/20,001 = 1.49997 and 1,490/1,000 do not) |
| 4 | HIGH | same-stream matching not enforced across new arms; no lock | accepted. Each tier's first arm fixes its sampler environment and every later arm must match it; Python 3.12.14 pinned; the lock written at the first install and installed exactly on a reinstall (section 5). C1, C2 | env |
| 5 | HIGH | restarts trusted file existence; a crash between the completed last.pt and the export blocked the resume | accepted. Completion records with sizes and sha256 (temp + rename); reads, final networks and timing rows validated; `bb.train` writes its final files atomically, `log.json` last, and `--resume` of an ended run exports again (section 6). C1 | artifacts; the Mac mini's crash-between-saves run of the real `bb.train` |
| 6 | HIGH | the copy-back omitted the D-bank shots | accepted. The D-bank's record lists its 7 files and `rows_dbank.csv`; the keeper fetches every recorded artifact within 5 minutes and checks them and the D-bank rows at the end. C1, C2 | the keeper is shell (syntax checked); its first real run is on the pod |
| 7 | MED | latency ignored `gpu_watch.accepted`; any non-empty CSV counted as three processes | accepted. Only accepted rows of distinct processes with both roles and this pod's engine signature count: 1 per shape in S0 (at most 3 calls), 3 per frontier network (at most 6). C1 | artifacts (a cut-off and an unaccepted process are measured again) |
| 8 | MED | projections from a one-evaluation smoke; W from active MACs | accepted. Every smoke records `eval_s`; each arm's evaluations are counted; W is smoked and timed like the others. C1 | s0 |
| 9 | MED | a resume charged the whole horizon again; tier B charged a finished B2 | accepted. Remaining samples from `last.pt`; tier B counts unfinished arms and reads only. C1 | artifacts (resume projection, remaining time) |
| 10 | MED | setup checked the transfer against fresh hashes | accepted. The committed `gross2_inputs.sha256` (47 files); the staged files are checked before anything is sent, the pod's against the same list. C2 | the Linux workstation's 47 staged files match the list |
| 11 | MED | the p = 0.003 summary was missing | accepted. `out/p003_summary.json` with Clopper-Pearson intervals, the slope and its range, the note; the selection does not read it. C1 | p003 |
| 12 | MED | no steady-state profile; sampler processes bounded by the quota only | accepted. `--profile 300:100` in every smoke (phases and the samplers' capacity); v = min(steady rate, capacity); sampler processes = min(affinity, quota) - 2; one tuning for all arms. C1 | s0; the Mac mini's profile run |
| 13 | LOW | duplicated detector parity; per-parameter EMA updates | accepted as the option `--lean` (bit-equal labels, foreach EMA); on iff labels + EMA >= 3 % of a profiled step. C1 | the Mac mini: `--lean` equals the plain run after 8 steps |
| 14 | LOW | pageable per-batch transfers | accepted as the option `--chunk-h2d` (one pinned copy per chunk); on iff h2d >= 2 %. C1 | the Mac mini: `--chunk-h2d` equals the plain run after 8 steps |

`bb/test_gross2_driver.py` (CPU, simulated `bb.train`, `bb.route`, `bb.latency`; the driver's own code):
- ledger: an earlier session's rows give PRIOR = 3.08 h, and an arm that fits 8 h but not 8 h - PRIOR is skipped;
  without a ledger file the driver does not start;
- overrun: an arm whose projected end + copy-back passes the cutoff does not start (skip, BUDGET, exit 2);
- deadline: at the cutoff the running process group (a grandchild too) is stopped within seconds and later steps
  are not started;
- failure: a failing arm is an error (exit 1, its comparisons undetermined), a failing control leaves T and D
  undetermined, four failing arms give four errors, a tier-B crash fails the session although the frontier succeeds;
- tierb: after an empty tier A, B2 = W with 1,000 failures against C's 1,500 gives pass, the pair B2 / C and W;
- more: tier B skipped for budget after an empty tier A gives kill with its caveat (exit 2; inconclusive since round 2); a final network
  trained in another sampler environment is an error; a non-finite loss stops a variant, which does not qualify;
- env: an arm in another sampler environment than its tier's is an error and does not start;
- artifacts: a changed recorded read is read again; an invalid final network is exported again with `--resume`; a
  crash between the terminal `last.pt` and the export resumes; a run cut off halfway is projected on its remaining
  half; a cut-off and an unaccepted timing process are measured again;
- stoprule: through the driver's command runner and its watcher thread, 130/105 against 100/80 stops, 130/99 does
  not;
- s0: from smoke logs, the rates, the fixed time and the tuning, N = 19,996,672 at 15k samples/s and BUDGET
  INSUFFICIENT at a quarter of the rates;
- p003: counts, intervals and the slope on simulated reads.

The keeper (`gross2_keep.sh`) was run on the Linux workstation against a fake local pod tree (I/O only, seconds): a complete copy
with the D-bank checked against `rows_dbank.csv`; an ssh failure is a failure; a recorded file missing on the pod
gives retries until E - 20 min, then INCOMPLETE; a LIMIT in `out/budget.json` earlier than DEADLINE moves the final
copy-back.

An independent review of this revision before the push (read-only, 2026-10-06) found 15 points. C4,
the commit that adds this paragraph, fixes every correctness point and adds the tests it found missing
(`bb/test_gross2_train.py`; the cases more, grandchild and stop rule through the runner): the keeper's pull could
report success after an ssh failure, and its final copy-back started 20 s after the driver's cutoff without a retry
(both HIGH); an arm's start test left out its reads; the cutoff's kill could be lost to a ledger error or to a step
starting at the same moment; a rerun of S0's GPU-term step could divide by zero; a missing ledger gave PRIOR = 0, and
a failed setup left no row; a comparison's record did not cover its reads; the `bb.checks` line was cut before its
test; the copy-back list was written after the record; a p = 0.003 failure could fail a stage; apt could read setup's
remote script from stdin; the D-bank manifest rows were compared with their CR line ends; the keeper ignored LIMIT.
Point 15 (kill after a budget-skipped tier B) is open question 8. Not taken (over-design or against a disposition):
dropping the keeper's alive rows (they cover time without the driver), trimming the tuning's recorded basis (S0's
decision must be recorded), dropping `--chunk-h2d` (disposition #14).

### Round 2: the re-review of b0988cfc

An independent re-review of b0988cfc (2026-10-06) found the 14 round-1 fixes
in the code and 7 new points, verdict REVISE. The study lead accepted all 7 as the reviewer proposed them. Commits:
C5 = 07b3fce6 (code, pod scripts, tests), C6 = the commit of this section. The driver tests run on the Mac mini's CPU;
`bb/scripts/pod/test_gross2_pod.sh` (new) runs on the Linux workstation (Linux, I/O only, about 20 s).

| # | severity | finding | disposition and implementation | test |
|---|---|---|---|---|
| 1 | CRITICAL | `exec < /dev/null` in setup's remote script ended the script that `bash -s` reads from ssh's stdin: no uv, venv, lock, CUDA check or freeze ran, the script exited 0, and the failing lock copy was the first command of an `&&` list, so setup said ok | accepted. The remote part runs as one function called with `< /dev/null` (bash reads all of it first) and ends with `environment: every step done`; a missing venv or lock afterwards stops setup (section 5, step 3). C5 | `test_gross2_pod.sh`: the remote part piped into `bash -s` with stub uv, python, nvidia-smi and apt-get; the first install, a reinstall from the lock and a rerun reach the end; a failing step and a venv off the lock exit 1 before it. b0988cfc's remote part in the same harness: exit 0, no output, no venv |
| 2 | HIGH | in the container's PID namespace nvidia-smi lists host PIDs, so `gpu_watch` counted the measuring process as another one: every S0 GPU-term call would be rejected | accepted. `G2_PIDNS=1`, set only by `gross2_run.sh`: alone on the GPU iff exactly one compute app is listed; without it the registered rule is unchanged. Setup's 10-s CUDA probe prints the rule's result (section 5, step 4). C5 | driver test pidns (simulated nvidia-smi): under the rule one host PID is accepted, two or none are not; without it a foreign PID is rejected and this PID accepted |
| 3 | HIGH | the keeper could write complete for a stale copy (every final pull failed), before the driver's last records (no fetched exit record required) or with a header-only `rows_dbank.csv`; nothing stopped an earlier keeper | accepted. Complete only if a final pull succeeded in this run with a driver exit record from after the keeper's start, every recorded artifact matches and `rows_dbank.csv` lists 7 files, all matching; a flock on `keep.lock` (section 6). C5 | `test_gross2_pod.sh` (fake pod, `POD_SSH="bash -c"`): complete; INCOMPLETE for failing final pulls over a complete earlier copy, a driver exit from before the keeper's start and a header-only D-bank; a takeover. b0988cfc's keeper: complete for the failing pulls |
| 4 | HIGH | Q7 and Q8: a budget-skipped tier B after an empty tier A gave kill; a tier-A winner whose B2 missed 1.5x set stage 2's shape without a label | accepted. Kill needs tier B evaluated; a skipped tier B is undetermined ("from-scratch capacity untested": inconclusive); tier A's own selection says "tier B pending"; the unconfirmed winner is kept with the caveat "tier A only, unconfirmed from scratch" and `stage2_shape` None (section 4, point 7; section 8, questions 7 and 8). C5 | driver tests more (inconclusive, exit 2), tierb (tier A alone: inconclusive, tier B pending), q7 (C 1,400 / B2 1,000: pass, T kept, shape None, caveat; 1,500 / 1,000: shape T) |
| 5 | LOW | `kill()` returned before SIGKILL once the direct child had exited, so a grandchild that ignores SIGTERM survived | accepted. SIGKILL to the group always follows the SIGTERM wait. C5 | driver test deadline: the leader dies on SIGTERM, its grandchild ignores it and is gone within seconds |
| 6 | LOW | an arm cut at the cutoff was not counted (exit 0 instead of 2); B2's 1.5x skipped the 5-ms GPU term; PRIOR split a session whose start key stepped by a second; the last regular poll could eat the final copy-back's window | accepted. The cut arm goes through `err()` (counted); B2 needs a GPU term <= `G_MAX_MS`; start keys within 2 s are one session; a poll that would run into E - 27 min is skipped (sections 4, 6). C5 | driver tests overrun (a cut arm: deadline, exit 2), tierb (W at 6 ms: no win, kill), ledger (keys a second apart: PRIOR 3.08 h, not 5.67 h); `test_gross2_pod.sh` (no poll before the final copy-back) |
| 7 | LOW | a restart truncated the driver log; a tier's arms could run under different code; no single-instance lock for the driver | accepted. The driver's (and the keeper's) log is appended; each tier records `code/COMMIT` and refuses another; `driver.lock` (flock) (sections 3, 6). C5 | driver tests env (another commit: error), lock (a second driver does not start and writes nothing) |

Implementation choices beyond the reviewer's text:
- #3: the lock makes the newer keeper win. It stops the process group named in `keep.lock` and takes the lock (a plain
  refusal would have left the keeper from before the restart running and writing the verdict); the long-running
  children close the lock's descriptor, so the lock is free once that group is gone. At its start a keeper removes an
  earlier `COPYBACK.txt`.
- #3: the final copy-back is retried until it has fetched the driver's exit record, not only until a pull succeeds, so
  a pull at E - 27 min that runs before the driver has written its last records does not end as INCOMPLETE.
- The keeper's per-poll ssh (the session's start) now has a 60-s timeout: a hung ssh there would have blocked the
  final copy-back.
- #2: the probe prints its result and does not stop setup (the study lead's disposition); operation step 2 says
  not to start the driver on `REJECTS`.

### Round 3: the third review of a59d0b71

A third review of a59d0b71 (2026-10-06) found all 7 round-2 fixes in the
code and 6 new points (1 MED, 5 LOW), verdict PROCEED. The study lead accepted all 6 as the reviewer proposed them,
and its review point (c) as two checks. Commits: C7 = b2ebfb89 (code, pod scripts, tests), C8 = the commit of this
section.

| # | severity | finding | disposition and implementation | test |
|---|---|---|---|---|
| 1 | MED | the keeper accepted only a driver exit later than its own start: a keeper started after the exit (a driver that failed at once, a replaced keeper, a late takeover) polled until E - 27 min and ended INCOMPLETE | accepted. The driver writes `started` into `out/budget.json`; the keeper accepts `out/DRIVER_EXIT.json` with t >= started (section 6). C7 | Linux workstation: a keeper started 30 s after an exit with t = started is complete; an exit record from before the driver's start is INCOMPLETE |
| 2 | LOW | a stale `pod_out/out/budget.json` with a past LIMIT ended a late keeper at once, INCOMPLETE, without a pull | accepted. After taking the lock the keeper removes pod_out's `out/budget.json` and `out/DRIVER_EXIT.json`; the next pull fetches the pod's. C7 | Linux workstation: a stale budget.json with LIMIT 100 s ago: the keeper polls and is complete |
| 3 | LOW | the takeover's `kill -- -PID` missed GNU `timeout`'s own process group: the old keeper's ssh and tar kept writing into pod_out for up to 240 or 900 s | accepted. If the old keeper leads a session (setsid, as section 3 starts it), `pkill -TERM -s PID`; else its process group as before. C7 | Linux workstation: the first keeper hangs in a pull under `timeout` (its ssh outside the keeper's group); after the takeover nothing is left in its session |
| 4 | LOW | the driver lock's `except OSError` also reported ENOLCK or EOPNOTSUPP as "another driver holds" | accepted. Only `BlockingIOError` means another driver; any other errno stops the driver with that errno. Setup stops while a driver holds `/workspace/g2/driver.lock` (`flock -n`; section 5's "while the driver does not run"). C7 | Mac mini: lock (ENOLCK: "cannot lock ... errno 77" on macOS, nothing written); Linux workstation: setup's three lines, free rc 0, held rc 1 |
| 5 | LOW | section 3's environment block lacked `G2_PIDNS=1`; the probe's accepts/REJECTS word reads only the app count | accepted. `G2_PIDNS=1` in the block; the probe line adds `accepted true or false` (the whole rule, the clock check included); operation step 2 says to report it. C7, C8 | the printed line, rendered with fake results (accepted true; app rule accepts with accepted false) |
| 6 | LOW | with B2's S0 GPU term above `G_MAX_MS` tier B (about 2.5 h) could only end with B2 not qualifying | accepted. Before tier B the driver checks B2's shape's S0 GPU term; above `G_MAX_MS` tier B is skipped, the reason recorded in the selection's `tier_b.reason` and its log line; with an empty tier A the decision is inconclusive (question 8). C7 | Mac mini: gskip (W at 6 ms: no arm trained, exit 0, inconclusive, "tier B skipped (B2's shape W: GPU term 6.00 ms > 5 ms): from-scratch capacity untested") |
| (c) | - | nothing confirms that the keeper runs before tier A; its first poll comes before the D-bank exists | accepted as checks, not a pod-side wait. The keeper's poll line gives `D-bank k/7`; operation step 5 checks `D-bank 7/7` at about 25 min. C7, C8 | Linux workstation: the poll line says 7/7 for a complete pod and 0/7 for a header-only `rows_dbank.csv` |

Implementation choices beyond the reviewer's text:
- #1: the driver's start command (operation step 3, setup's printed line) removes an earlier `out/DRIVER_EXIT.json`
  before it starts the driver. With t >= started alone, a keeper whose first pull comes before a rerun's driver has
  written its `budget.json` would pair the previous run's budget.json and exit record and end at once.
- #6: a GPU-term skip is not a budget shortfall (the stage exits 0, not 2); a budget skip now records the reason
  "budget", and the undetermined text names the reason. The driver test's tier-B case with W at 6 ms (kill when tier
  B ran) became gskip; decide() keeps its own GPU-term check on B2.
- #3: when the old keeper leads no session (started without setsid), the takeover stops its process group as before.

Tests: Linux workstation `bb/scripts/pod/test_gross2_pod.sh` ALL PASS (21 s). Under the same harness a59d0b71's keeper keeps
polling after an exit that came before its start (the case was cut off after 300 s), ends at once over a stale
budget.json, and leaves 4 processes (timeout, its shell, the fake ssh, tar) in the first keeper's session after the takeover.
Mac mini `bb.test_gross2_driver` ALL PASS in 15 cases (C7's two Python files, sha256 checked). Nothing ran against a
real pod.
