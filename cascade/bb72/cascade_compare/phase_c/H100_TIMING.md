# Phase C's timings on an H100 (secondary column, speed only)

The authors asked for phase C's decoders and routing pipelines to be timed on an H100 too, and this file describes that
run. The H100 values form a second column and describe speed only. Phase C's one GPU model stays the RTX PRO 6000
Blackwell Server Edition (PHASE_C_PLAN.md 5.1), and no H100 value enters a lock, a check, a confirmation read, the
report or a verdict. Every H100 row carries `card_role: "H100 (secondary, speed only)"`. The code is on branch
`ising-cascade-h100t`. `pc_common.py`, `pc_speed.py`, the locks and the primary outputs are unchanged. The final
outputs of both pods and of the screen are committed in `out_h100/` beside this file ("The committed outputs", at the
end), and every H100 number below traces to a file there.

## The pod

- RunPod secure cloud, rented at 22:05 PDT on 10 October 2026. The container disk is lost when the pod stops.
- NVIDIA H100 80GB HBM3 (SXM, 700 W power limit, maximum SM clock 1,980 MHz, MIG disabled), driver 570.211.01. CPU: AMD
  EPYC 9554, 224 threads visible. Python 3.12.3.
- [The pod's spend limit and the watchdog that enforced it are not part of this release.]

## Venv, tree and inputs

- Venv `/opt/h100t/venv` (`scripts/h100_setup_venv.sh`): the 44 distributions of Alpine's `bb/venv_cu128`, read from
  its dist-info directories (`scripts/h100_req_cu128.txt` plus pip 25.0.1). These include torch 2.11.0+cu128, triton
  3.6.0, numpy 2.2.6, scipy 1.18.1, stim 1.16.0, pymatching 2.4.0 and the CUDA 12.8 nvidia libraries. They came from
  the PyTorch cu128 index and PyPI. relay-bp 0.2.2 was added with `--no-deps`, the version in Alpine's `bb/venv`,
  which the GPU jobs put on `sys.path`. The venv's `pip freeze` matches that list line for line (its sha256 is
  367bc6f7…). torch sees the H100 (capability 9.0).
- Tree: `scripts/h100_deploy.sh` ships the same files as `deploy.sh`: ising-cascade-net at 64b2e25d (deploy.sh's
  list) plus `cascade_compare/phase_c` without `dry/`. The deployed tree is `net 64b2e25d phasec+h100t 0bc88bf2` in
  `/opt/h100t/code.20261010_222305`. I compared the sha256 of every file with the primary run's deployed tree on
  Alpine (`code.20261010_174740`, net 64b2e25d phasec d9b0b899). All 310 files of that tree are identical on the pod.
  The pod has six more files: `pc_h100.py`, `jobs/h100.sh`, `scripts/h100_deploy.sh`, and the three `out/` files of
  1e8d1c6d. The hash lists are in `~/work/cascade_h100t/stage/`.
- The lock and the frozen paths were copied read-only from the primary run's directory on Alpine to the same paths on
  the pod. `locked_uniform.json` is 4644fbcc…, `locked_depspam_fm.json` 02b19c45… and `speed/speed_components.json`
  1ec6b256…, the sha256 values recorded in 1e8d1c6d. Every process refuses speed components that the lock did not
  freeze.
- Checkpoints: all nine were copied to their Alpine paths on the pod. The eight networks in the locks match the
  locks' sha256. C6′ is 5fd3bca6…, the file C3 timed. Relay-BP's 14 BB72 X-memory circuits match the circuit manifest.

## What runs (`jobs/h100.sh`, `pc_h100.py`)

`pc_h100.py` lets the H100 through where `pc_common.gpu_env` lets through only the RTX PRO 6000. It refuses every
other card, phase C's included, and otherwise calls `pc_speed`'s own builds and measurements. The steps run in order,
each process fresh:

1. Networks (`nets`). C14, C6, S55, C14′, S55′, S55′-deep, r11, r12 and C6′ each run in 3 processes, rep-major
   as in C3. Each network runs on its frozen paths from the lock's `speed_components.json`, at the batch sizes C3 times
   on those paths: Cascade compiled at 1 and 4,096, ConvLines fused at 1, 16, 64, 256, 1,024 and 4,096. At batch 1
   the process measures the GPU term (200 warm-up replays, then 5 × 100) and 10,000 host round trips (mean, p50, p99,
   p99.9, max). At every batch size it measures the amortized time per shot, with the input resident and host to host.
   The functions are C3's (`gpu_us`, `host_rt`, `amortized`), and the rows are C3's (kind 12, p = 0.55 %, the
   network's setting; the measurements read the first 32,768). Unlike C3, no other path is timed and nothing is
   screened in the process. Each path first decodes 20,000 rows (below batch 64) or 40,000 rows once before its
   timing.
2. The H100 half of the screen (`outputs`). The pod regenerates C4b's check shots (kind 12, chunks 10-19 at 0.2 %
   and 0.3 %, in each setting) and records each chunk's digest the way `bb.cascade_eval` does. It then runs them
   through every frozen path that `pc_speed.check` runs: every network alone (Cascade of the setting, its depth-rule
   variant and our five) on its single, amortized and batched paths, and every locked configuration at the claim
   point in both modes. It saves the decisions, confidences and routing masks, and times the seeded Relay-BP
   decoder's build on the pod's CPU.
3. Pipelines (`pipeline`, which is `pc_speed.pipeline` unchanged). In each setting, at 0.2 % and 0.3 %, it runs
   Cascade alone and the four locked configurations (S → B, S → B → R, S → R, r12 → R) in both modes, 3 processes
   each, rep-major as in C7. Single mode runs 20,000 shots after 200 untimed ones. Batched mode runs 500,000 shots,
   with S and B at 4,096. Relay-BP runs full-DEM on the pod's EPYC 9554: single-threaded in the process in single
   mode, in 16 worker processes in batched mode.
4. Summary (`summarize`, which writes `h100_speed.json`). For each network and path it keeps the slowest process at
   every batch size, and for each configuration and mode the slowest process (plan 5.2-5.3). The RTX PRO 6000's frozen
   values sit beside them, and each value carries the screen's verdict.

Two settings differ from C3 and C7, and neither touches a timed computation. Shot generation uses 25 processes; each
chunk comes from its own seed, so the shots are the same. `BB_PIDNS=1` lets `gpu_watch` count the one compute app in
the pod's PID namespace, as on earlier pods.

A smoke test on seed kind 13 (`jobs/h100_smoke.sh`, written to `/opt/h100t/smoke`, never reported) ran nets, both
pipeline modes, outputs (one chunk per p) and summarize once each, all with exit 0, before the run started at
22:29 PDT.

## The screen

C4b had not run when the pod started: it is queued on Alpine, and Slurm estimates its start at 11:11 PDT on
11 October. The pod therefore saves the H100's outputs on the check shots now, and the comparison runs once C4b's
reads exist. Alpine job CJ36 (`scripts/h100_screen.sbatch`, acpu, 8 cores, CPU only) was submitted with
`--dependency=afterok:CJ34`, which is C4b. It reads the primary run's lock, frozen paths and reads in place,
read-only, and writes only under `<cluster-scratch>/cascade_h100t/`. Per setting, `pc_h100.py screen`
judges the H100 outputs as `pc_speed.check` judges a timed path against the read:

- a ConvLines path passes only if it is bitwise its RTX PRO 6000 read, every decision and every confidence;
- a Cascade path passes by the plan's tolerance (`pc_common.agrees`) on its failures;
- a configuration passes on its failures and its shots routed to B and to R, each by the tolerance, and on its S being
  bitwise. Relay-BP decides every shot that either version routes to it, with the decoder seeded by the shot's index.

The screen refuses outputs whose chunk digests differ from the read's. `summarize --screen` then marks every H100
value as passed, failed or not screened. A path that fails keeps its time in the table, marked as failing. Until the
job runs, every H100 value is "not screened". The fused kernels compile for sm_90 on the H100 and for sm_120 on the
RTX PRO 6000, so bitwise equality between the two cards is not expected; the ConvLines paths may well fail. (They
passed: every decision and confidence was bitwise its RTX PRO 6000 read; see "The screen" under "Amendment 1".)

Job CJ36 failed in 4 s; job CJ47 screened both runs instead, and every row passed ("Amendment 1" below gives the cause,
the fix and the result).

## First rows (rep 1 of 3, 22:29-22:36 PDT)

The table sets one H100 process per network against the RTX PRO 6000's frozen values from C3, each of which is the
slowest of 3 processes. Single shot is the mean host round trip at batch 1. Amortized is host to host per shot on the
frozen amortized path; per round is per shot / 6.

| network | single H100 (µs) | single RTX (µs) | amortized path | amortized H100 (µs/shot) | amortized RTX (µs/shot) |
|---|---:|---:|---|---:|---:|
| C14 | 1,662.5 | 1,783.4 | compiled@4096 | 42.92 | 71.63 |
| C6 | 761.0 | 798.4 | compiled@4096 | 18.64 | 31.35 |
| C14′ | 1,664.9 | 1,781.9 | compiled@4096 | 42.95 | 71.63 |
| C6′ | 759.8 | 794.4 | compiled@4096 | 18.63 | 31.36 |
| S55 | 96.6 | 88.2 | fused@1024 | 5.62 | 4.14 |
| S55′ | 96.1 | 88.0 | fused@1024 | 5.62 | 4.14 |
| S55′-deep | 146.2 | 130.9 | fused@4096 | 9.03 | 6.26 |
| r11 | 96.1 | 88.1 | fused@1024 | 5.62 | 4.14 |
| r12 | 96.1 | 88.0 | fused@1024 | 5.63 | 4.14 |

On the H100 the Cascade networks take 4-7 % less time per single shot and about 40 % less per shot at 4,096. The
fused ConvLines engine takes 9-12 % more per single shot and 36-44 % more amortized. At batch 4,096 the Cascade
networks pulled the H100's SM clock down to 73-86 % of 1,980 MHz, so `gpu_watch` marks those rows as not accepted
(its rule is 97 %). In C3 the RTX PRO 6000 held at least 98 % for C14 and C6 at 4,096. These are the values the H100
sustains under that load, and the rows keep the clock samples.

The table above holds the first repetition only (the `rep: 1` rows of `out_h100/first_pod/rows.jsonl`). The final
values, each the slowest of 3 processes, are in `out_h100/screen/first/summary.txt` (and in `nets` of
`out_h100/screen/first/h100_speed.json`), and the screen passed every network on its single, amortized and batched
paths. RESULTS.md (section 4.1) quotes these final values:

| network | single H100 (µs) | single RTX (µs) | amortized path | amortized H100 (µs/shot) | amortized RTX (µs/shot) |
|---|---:|---:|---|---:|---:|
| C14 | 1,666.5 | 1,783.4 | compiled@4096 | 42.951 | 71.629 |
| C6 | 761.0 | 798.4 | compiled@4096 | 18.674 | 31.353 |
| C14′ | 1,664.9 | 1,781.9 | compiled@4096 | 42.985 | 71.626 |
| C6′ | 759.8 | 794.4 | compiled@4096 | 18.685 | 31.359 |
| S55 | 98.5 | 88.2 | fused@1024 | 5.631 | 4.139 |
| S55′ | 96.4 | 88.0 | fused@1024 | 5.629 | 4.139 |
| S55′-deep | 146.2 | 130.9 | fused@4096 | 9.029 | 6.257 |
| r11 | 97.0 | 88.1 | fused@1024 | 5.628 | 4.139 |
| r12 | 97.2 | 88.0 | fused@1024 | 5.628 | 4.139 |

The final values give the same ranges as the first repetition: on the H100 the Cascade networks take 4-7 % less time
per single shot and about 40 % less per shot amortized, and the fused engine 9-12 % more per single shot and 36-44 %
more amortized.

[The forecast of the first run's end is not part of this release.]

[The description of the loop that copied the first pod's outputs is not part of this release.]

## Files

- `~/work/cascade_h100t/results/out`: `rows.jsonl` (every network and pipeline row), `outputs/<setting>/h100_p*.npz|json`
  (the screen's H100 half), `h100_speed.json` and `summary.txt` (no screen verdicts yet), `env.json`, `progress`,
  `SHA256SUMS`, `END`.
- `~/work/cascade_h100t/results/screen`: planned for the screen's output, it holds only the `STATUS` of the failed job
  CJ36. Job CJ47 screened this run too, and its `screen_uniform.json`, `screen_depspam_fm.json`, and
  `h100_speed.json` and `summary.txt` with the verdicts are in `~/work/cascade_h100b/results/screen/first/`.
- Committed copies: `out_h100/first_pod/` and `out_h100/screen/first/` ("The committed outputs").
- On the branch: `pc_h100.py`, `jobs/h100.sh`, `jobs/h100_smoke.sh`, and `scripts/h100_deploy.sh`,
  `h100_pull.sh`, `h100_screen.sbatch`, `h100_setup_venv.sh` and `h100_req_cu128.txt`.

## Amendment 1 (the second pod's run, 11 October)

Amendment 1 (`RUN.md`, commit 08542cd8; new lock 629ca1c4) changed the picks of 21 of the 32 configurations, 9 of
them at the claim points. Those 9 were timed on a second rented H100, with the protocol above, and Cascade alone was
timed beside them in each (setting, p) group in the same job. The H100 is still a secondary column for speed only: every
row carries `card_role: "H100 (secondary, speed only)"`, and nothing here enters a lock, a check, a confirmation read,
the report or a verdict. The code is on branch `ising-cascade-h100t2`, which is `ising-cascade-h100t` merged onto
d4402750 (the new lock and `RUN.md`).

The first run's 234 rows remain valid for every network alone and for the 7 configurations at the claim points that
Amendment 1 did not change: S → R and r12 → R in `uniform` at both claim points, r12 → R in `depspam_fm` at both, and
`depspam_fm` S → B → R at 0.3 %. Under the new lock the H100 column is therefore the first run's rows for those, and
the rows below for the changed configurations.

| setting | p | configuration | first lock | new lock (index) |
|---|---|---|---|---|
| `uniform` | 0.2 % | S → B | S55, q1 2 % | S55, q1 1 % (2) |
| `uniform` | 0.2 % | S → B → R | S55, q1 40 %, r 0.1 %, softplus | S55, q1 1 %, r 0.1 %, minabs (3) |
| `uniform` | 0.3 % | S → B | S55, q1 40 % | S55, q1 5 % (7) |
| `uniform` | 0.3 % | S → B → R | S55, q1 40 %, r 0.1 %, minabs | S55, q1 5 %, r 0.1 %, minabs (8) |
| `depspam_fm` | 0.2 % | S → B | S55′-deep, q1 2 % | S55′, q1 1 % (2) |
| `depspam_fm` | 0.2 % | S → B → R | S55′-deep, q1 5 %, r 0.1 %, minabs | S55′, q1 2 %, r 0.1 %, minabs (3) |
| `depspam_fm` | 0.2 % | S → R | S55′-deep, q 0.25 % | S55′, q 0.25 % (4) |
| `depspam_fm` | 0.3 % | S → B | S55′-deep, q1 10 % | S55′, q1 5 % (7) |
| `depspam_fm` | 0.3 % | S → R | S55′-deep, q 2 % | S55′, q 2 % (9) |

B is C14 in `uniform` and C14′ in `depspam_fm`, as before. Before its first process the job compares the two locks
and stops unless exactly these configurations differ in their picks (S, B, q1, r, q, score, thresholds).

### The pod

- RunPod secure cloud, rented at 02:45:50 PDT on 11 October. This is another pod than the first run's. Image
  `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`; its system torch 2.8.0 is not used.
  The container disk is lost when the pod stops.
- NVIDIA H100 80GB HBM3 (700 W power limit, maximum SM clock 1,980 MHz, MIG disabled), driver 580.126.09 (the first
  pod had 570.211.01). The study lead checked it at 02:48: no ECC errors, no pending row remaps. CPU: Intel Xeon
  Platinum 8470, 208 threads (the first pod had an AMD EPYC 9554). Python 3.12.3.
- Relay-BP runs on the pod's CPU, so S → B → R and S → R are timed on another CPU than in the first run. Cascade alone
  and S → B do not use it. Every group's Cascade baseline was timed in this job, on this pod.
- [The pod's spend limit and the watchdog that enforced it are not part of this release.]

### Pins, trees and inputs

- Venv `/opt/h100b/venv` (`scripts/h100_setup_venv.sh` with `Q=/opt/h100b`): `batchopt/req_cu128.txt`, the same 43
  pins as `scripts/h100_req_cu128.txt` (Alpine's `bb/venv_cu128`: torch 2.11.0+cu128, triton 3.6.0, numpy 2.2.6, scipy
  1.18.1, stim 1.16.0, pymatching 2.4.0, the CUDA 12.8 nvidia libraries), plus pip 25.0.1 and relay-bp 0.2.2
  (`--no-deps`). Its `pip freeze` has sha256 367bc6f7…, the first pod's. torch sees the H100 (capability 9.0).
- Timing tree: `scripts/h100_deploy.sh` with `Q=/opt/h100b INPUTS=1`, now in `/opt/h100b/code.20261011_025754`
  (`net 64b2e25d phasec+h100t e4db6698`). I compared it file by file with the primary run's current deployed tree on
  Alpine (`code.20261011_013018`, `net 64b2e25d phasec 08542cd8`, which ran the second C4 and runs the jobs after it).
  All 313 files of that tree match on the pod except `out/locked_*.json`, which are the repository's copies: the first
  lock at 08542cd8 and the new lock at d4402750. No process reads them. The pod also has the H100 files, `RUN.md`, the
  amendment's gate review and the first lock's two check files. The hash lists are in `~/work/cascade_h100b/stage/`.
- Lock: `$R/out/amend1/locked_uniform.json` (1748c2d4…) and `locked_depspam_fm.json` (398d02c6…), the sha256 values
  in `RUN.md`, copied read-only from Alpine to the same paths. `speed_components.json` is the unchanged 1ec6b256….
  The first lock (4644fbcc…, 02b19c45…) is there too, for the comparison of picks. The nine checkpoints and Relay-BP's
  BB72 X-memory circuits were copied as in the first run (`INPUTS=1` now does this).
- Profile tree: `batchopt/pod_deploy.sh` from `ising-cascade-batchopt` 425640d1 with `Q=/opt/h100b/batchopt`, in
  `/opt/h100b/batchopt/code.20261011_025651` (`net 64b2e25d batchopt 425640d1`). S55 87886c65…, C14 0f44e614… (as in
  PHASE_C_PLAN.md 2.1), the circuit at 0.3 % ac17ceae….

### What ran (`jobs/h100_a1.sh`)

Each step ran in a fresh process, in `jobs/h100.sh`'s order. The networks alone were not rerun.

1. `outputs` on the new lock, per setting: the H100 half of the screen, as in the first run (02:58-03:19 PDT).
2. Pipelines: in each of the 4 (setting, p) groups, Cascade alone and the group's changed configurations, both modes,
   3 processes each, rep-major (13 configurations, 78 processes, 03:19-04:06 PDT). Single mode: 20,000 shots after
   200 untimed ones. Batched mode: 500,000 shots, S and B at 4,096. These are `pc_speed.pipeline`'s settings, and
   Relay-BP runs full-DEM on the pod's CPU as before.
3. `summarize` on the new lock: `h100_speed.json`, the slowest process per configuration and mode.
4. The profile (below).

[The time limits that the second run's timing and profile steps ran under are not part of this release.]

### Rows

`h100_speed.json` from the pod's `summarize` gives each value as the slowest of 3 processes. Single mode gives the mean
and p99.9 of the per-shot latency over 20,000 shots; batched mode gives the time per round (per shot / 6) over 500,000
shots. Each ratio divides by the Cascade baseline of the same group and mode, timed in the same job. "to B, to R" are
the mean fractions of shots the batched processes routed. A * marks a batched value where at least one process had its
SM clock below `gpu_watch`'s 97 % rule; as the first run found, the H100 lowers its clock under these loads. The last
column gives the first lock's pick at the same index, timed on the first pod, as ratios to that pod's Cascade (for
Cascade, its values there). The screen (job CJ47, below) passed every value. The values are in `pipelines` of
`out_h100/screen/amend1/h100_speed.json`, and the last column's in `out_h100/screen/first/h100_speed.json`.

| setting | p | decoder (index) | single mean (µs) | single p99.9 (µs) | single / Cascade | batched per round (µs) | batched / Cascade | to B, to R (%) | first lock's pick, first pod: single, batched / its Cascade |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| `uniform` | 0.2 % | Cascade alone (C14) | 1,484.2 | 1,779.6 |  | 7.282 * |  |  | 1,549.8 µs, 7.211 µs |
| `uniform` | 0.2 % | S → B: S55, q1 1 % (2) | 119.6 | 1,712.1 | 0.081 | 1.067 * | 0.147 | 1.00, 0.00 | 0.087, 0.155 |
| `uniform` | 0.2 % | S → B → R: S55, q1 1 %, r 0.1 % (3) | 284.3 | 3,799.8 | 0.192 | 3.458 * | 0.475 | 1.00, 0.10 | 0.588, 0.551 |
| `uniform` | 0.3 % | Cascade alone (C14) | 1,484.1 | 1,725.4 |  | 7.296 * |  |  | 1,548.4 µs, 7.216 µs |
| `uniform` | 0.3 % | S → B: S55, q1 5 % (7) | 183.8 | 1,716.5 | 0.124 | 1.355 * | 0.186 | 4.97, 0.00 | 0.496, 0.527 |
| `uniform` | 0.3 % | S → B → R: S55, q1 5 %, r 0.1 % (8) | 377.0 | 1,765.3 | 0.254 | 9.748 * | 1.336 | 4.97, 0.10 | 0.624, 1.297 |
| `depspam_fm` | 0.2 % | Cascade alone (C14′) | 1,483.3 | 1,609.4 |  | 7.293 * |  |  | 1,548.1 µs, 7.214 µs |
| `depspam_fm` | 0.2 % | S → B: S55′, q1 1 % (2) | 116.7 | 1,703.9 | 0.079 | 1.067 * | 0.146 | 1.02, 0.00 | 0.119, 0.236 |
| `depspam_fm` | 0.2 % | S → B → R: S55′, q1 2 %, r 0.1 % (3) | 338.9 | 2,237.0 | 0.228 | 2.827 * | 0.388 | 2.03, 0.11 | 0.285, 0.423 |
| `depspam_fm` | 0.2 % | S → R: S55′, q 0.25 % (4) | 765.3 | 107,894.0 | 0.516 | 4.987 | 0.684 | 0.00, 0.25 | 0.525, 0.703 |
| `depspam_fm` | 0.3 % | Cascade alone (C14′) | 1,482.6 | 1,717.9 |  | 7.296 * |  |  | 1,549.7 µs, 7.215 µs |
| `depspam_fm` | 0.3 % | S → B: S55′, q1 5 % (7) | 184.2 | 1,715.9 | 0.124 | 1.354 * | 0.186 | 4.98, 0.00 | 0.207, 0.315 |
| `depspam_fm` | 0.3 % | S → R: S55′, q 2 % (9) | 7,999.1 | 1,305,661.2 | 5.395 | 81.165 | 11.125 | 0.00, 2.00 | 4.232, 10.616 |

All 78 processes exited 0. Cascade alone came out within 0.1 % across the four groups: 1,482.6-1,484.2 µs single
and 7.28-7.30 µs per round batched. The first pod gave 1,548-1,550 µs and 7.21-7.22 µs.

- S → B takes 8-12 % of Cascade's single-shot time and 15-19 % of its batched time in all four groups.
- S → B → R takes 19-25 % of Cascade single. Batched, it takes 0.39-0.48 of Cascade at 0.2 % but 1.34 at `uniform`
  0.3 %, where Relay-BP's share of the batched time dominates.
- S → R in `depspam_fm` takes 0.52 (single) and 0.68 (batched) of Cascade at 0.2 %, and 5.4 and 11.1 times Cascade at
  0.3 %, where 2 % of the shots go to Relay-BP.
- Against the first lock's picks, the amended S → B and S → B → R picks route fewer shots to B and are faster in every
  group, by the largest margin where q1 fell from 40 % (`uniform` S → B at 0.3 %: 0.50 to 0.12 of Cascade single).
  The exception is `uniform` S → B → R at 0.3 % batched (1.30 to 1.34), which is set by Relay-BP. S → R at 0.3 % went
  from 4.23 to 5.40 of Cascade single, but its Relay-BP runs on another CPU now, so the pick and the CPU cannot be
  separated there. The RTX PRO 6000's values for these configurations come from C7 under Amendment 1
  (`pc_c7_pipe_a1`, job CJ39, still running at 04:10 PDT; its rows go to `$R/out/speed/rows.jsonl`).

### The screen

The screen job of the first run (CJ36) failed in 4 s at its first python call. Its tree has `pc_h100.py`. The
sbatch took the tree as `readlink -f $A/code`, which on the compute node resolves to the `/gpfs/alpine1` path, and the
container is bound only to `<cluster-scratch>`, so python could not open the file. Phase C's own sbatch files use plain
`readlink`, and both H100 screen files now do too (commit 1ee790c9). `scripts/h100_screen_a1.sbatch` replaces the
job. It runs from the tree `<cluster-scratch>/cascade_h100b/code.20261011_042239` (`h100_deploy.sh` with
`TO=alpine`; its files match the pod's, plus the two screen sbatch files). Both runs go through `pc_h100.py screen`
on C4b's reads (`$R/out/check/<setting>/read`, which the second C4b reuses), each against the lock it was made with:

- `first`: the first pod's outputs and rows (`<cluster-scratch>/cascade_h100t/out`, read in place) against the first
  lock;
- `amend1`: this run's outputs and rows (`<cluster-scratch>/cascade_h100b/out`, uploaded from the Linux workstation's
  copy) against Amendment 1's lock.

Each run then gets its `summarize` with the verdicts: `screen/{first,amend1}/h100_speed.json` and `summary.txt` under
`<cluster-scratch>/cascade_h100b/`. The first submission, job CJ46 at 04:21 PDT, still had `readlink -f`
and failed the same way in 9 s. Job CJ47 (acpu, 16 cores) started at 04:23 PDT with the fix and was running when this
was written. The ConvLines paths compile for sm_90 here and for sm_120 on the RTX PRO 6000, so their bitwise screen may
fail; `summarize` then keeps their times and marks them as failing.

Job CJ47 ended at 04:28 PDT (`END`: `2026-10-11T05:28:14-06:00 job CJ47`), and its output was copied and verified at
04:35 (`STATUS`: "DONE: the screen's results copied and verified"). Every row passed, 58 of 58 in each of the four files
(`screen_uniform.json` and `screen_depspam_fm.json` of `first` and of `amend1`). In each file the 30 ConvLines rows were
bitwise their RTX PRO 6000 reads, with no decision and no confidence different, and the 12 Cascade rows and 16
configuration rows passed as described under "The screen" above. Both screened `h100_speed.json` files mark every value
"passed". The files are in `out_h100/screen/`.

### The profile

`jobs/h100_a1.sh` ran `batchopt/prof_run.sh`'s steps after the timing, each in a fresh process from the batchopt
tree, from 04:06 to 04:10 PDT: shots, env, work, s55, c14, pipe, report, ncu, nsys, report. The outputs are in
`~/work/cascade_h100b/results/out/prof/` on the Linux workstation and `<cluster-scratch>/cascade_h100b/out/prof/` on
Alpine, and their text files are committed in `out_h100/second_pod/prof/`. `summary.md` there holds PROFILE.md's tables
for the H100. The RTX PRO 6000 ran the same steps as job CJ45 (03:22-03:29 PDT), with output in
`<cluster-scratch>/cascade_batchopt/out/CJ45/`. PROFILE.md covers both cards.

- ncu: the container has no access to the GPU's performance counters (ERR_NVGPUCTRPERM). `prof_run.sh` checks for
  that and skipped both ncu profiles, as designed. Only the RTX PRO 6000 has them.
- nsys: the image has no Nsight Systems. `scripts/h100b_nsys_fetch.sh` fetched 2025.5.2.266, the version of the RTX
  PRO 6000's run (CUDA 13.1's), from NVIDIA's ubuntu2404 repository. It checked the sha256 against the repository's
  index and unpacked the package with `dpkg -x` (`/usr/local/bin/nsys` runs it). It waited for the last timing
  process, so nothing ran beside the timing. The fetch and unpacking (04:06:25-04:06:32, nice 19, idle I/O class, two
  cores) overlapped the profile's shots, env and work steps. nsys_pipe exited 0 and wrote the same files as on the
  RTX PRO 6000.
- `progress` has `prof ncu exit=127` and `prof nsys exit=127`. The 127 comes from the last command of `prof_run.sh`,
  the listing of its output directory through `column`, which the image does not have. The steps ended as described
  above.

What the H100's `summary.md` shows (the RTX PRO 6000's job CJ45 in brackets):

- Ceilings: BF16 GEMM 777 TFLOP/s of a nominal 989 (390 of 504), DRAM copy 3,032 GB/s of 3,350 (1,461 of 1,792),
  pinned PCIe 55.5 GB/s each way, a one-kernel CUDA graph replay 3.43 µs (2.19).
- S55 alone, fused path: 5.60-5.83 µs per shot from batch 256 to 16,384, 5.61 at 4,096 (4.09), bitwise against the
  reads' engine. The roofline bound is the L2 traffic its programs request, 114 % of the measured L2-resident copy
  rate (125 %), against 6 % of the FP16 GEMM ceiling and 6 % of DRAM. `_k_block` takes 65 % of the time, `_k_pool`
  20 %, `_k_vote` 10 %, `_k_stem` 5 %.
- C14 alone, compiled path: 43.4 µs per shot at 4,096 (71.6) and 43.2 at 8,192, 23 % of the BF16 GEMM ceiling and 69 %
  of the DRAM copy rate by the bytes model (86 %). The Triton bipartite convolution `_gmm` (wgmma) takes 44 %,
  inductor's fused batch-norm and SiLU kernels 40 %, cuBLAS GEMMs 16 %. The SM clock fell to 70-76 % of its maximum.
- S55 → C14 batched: S alone 5.61 µs per shot, B alone 43.13 µs. Wall time per shot stays within 0.07-0.35 µs of
  aS + q1 · aB, for example 8.11 µs at q1 = 5 % against 7.77 (8.27 against 7.67). Most of that gap is the padded last
  B batch (0.34 µs per shot over 500,000 shots). Copies, routing, the queue and the GPU's idle gaps add at most
  0.05 µs per shot. S and B on two streams give no speedup (0.99-1.01).
- With twice the RTX PRO 6000's GEMM and DRAM rates, the H100 runs C14 39 % faster but S55 37 % slower. The routed
  pipeline at q1 = 5 % costs about the same per shot on both cards.

### The files

- The pod's outputs were copied from `/opt/h100b/out` to `~/work/cascade_h100b/results/out` on the Linux workstation
  (the pod's logs to `results/logs_pod/`) and, once the copy verified against `SHA256SUMS`, uploaded to
  `<cluster-scratch>/cascade_h100b/out` (verified there). The screen job's output was copied, verified, into
  `results/screen/`; `results/screen/STATUS` says how that ended. The copy's `DONE` marker was written at 04:20:16 PDT:
  `complete; 2 failed processes; copied and verified`. The two are the `column` exits described under the profile; no
  timing process or profile step failed. The upload to Alpine verified at 04:20:37.
- `~/work/cascade_h100b/results/out`: `rows.jsonl` (the 78 pipeline rows), `outputs/<setting>/h100_p*.npz|json`,
  `h100_speed.json` and `summary.txt` (no verdicts yet), `env.json`, `progress`, `prof/`, `SHA256SUMS`, `END`.
  Committed copy: `out_h100/second_pod/`, without the binary files.
- `~/work/cascade_h100b/results/screen`: `first/` and `amend1/`, each with `screen_uniform.json`, `screen_depspam_fm.json`,
  and `h100_speed.json` and `summary.txt` with the verdicts; `SHA256SUMS`, `END`, `STATUS`. Committed copy:
  `out_h100/screen/`.
- On the branch: `jobs/h100_a1.sh`, `scripts/h100_screen_a1.sbatch`, `scripts/h100b_pull.sh`,
  `scripts/h100b_nsys_fetch.sh` and `scripts/h100b_nsys_wrapper.sh`. `h100_deploy.sh` takes `Q`, `A` and `INPUTS`, and
  `h100_setup_venv.sh` takes `Q`.

## The committed outputs (11 October)

the review of phase C's results (finding 1) found that the final H100 values and the screen's "58 of 58" could not be
checked from the repository. The report's revision committed the existing final outputs, unchanged, in `out_h100/`.
Nothing was measured or recomputed.

| directory | copied from (the Linux workstation) | contents |
|---|---|---|
| `first_pod/` | `~/work/cascade_h100t/results/out/` (the first pod's run, 10 October) | `rows.jsonl` (234 rows: 114 of the networks, 120 of the first lock's pipelines), `h100_speed.json` and `summary.txt` from the pod's `summarize`, `env.json`, `progress`, `END`, `SHA256SUMS`, `outputs/<setting>/h100_p*.json` |
| `second_pod/` | `~/work/cascade_h100b/results/out/` (the second pod's run, 11 October) | `rows.jsonl` (78 pipeline rows under Amendment 1), `h100_speed.json` and `summary.txt` from the pod's `summarize`, `env.json`, `progress`, `END`, `SHA256SUMS`, `outputs/<setting>/h100_p*.json`, and the profile's 13 text files in `prof/` |
| `screen/` | `~/work/cascade_h100b/results/screen/` (job CJ47) | `first/` and `amend1/`, each with `screen_uniform.json`, `screen_depspam_fm.json`, and `h100_speed.json` and `summary.txt` with the verdicts; `SHA256SUMS`, `END`, `STATUS` |

The binary files stay outside git: both pods' `outputs/<setting>/h100_p*.npz` (the H100 half of the screen, about
36 MB per pod) and, in `prof/`, `shots.npz`, `ref_C14.npz`, `ref_S55.npz`, `nsys_pipe.sqlite`, `nsys_pipe.nsys-rep`
and the `trace_*.json.gz` files. They remain on the Linux workstation and on Alpine
(`<cluster-scratch>/cascade_h100t/out` and `<cluster-scratch>/cascade_h100b/out`), and each pod's committed `SHA256SUMS`
lists their sha256. Before the commit, every committed file that a `SHA256SUMS` lists matched it (9 files in
`first_pod/`, 22 in `second_pod/`, 8 in `screen/`), and the three `SHA256SUMS` files were byte-identical to their copies
on Alpine, where the 8 screen files also verified.

Where each H100 number comes from:

- the networks' final values (the second table under "First rows" and the H100 columns of RESULTS.md section 4.1):
  `screen/first/summary.txt`, or `nets` in `screen/first/h100_speed.json`;
- the first repetition (the first table under "First rows"): the `rep: 1` rows of `first_pod/rows.jsonl`;
- the pipelines (the table under "Rows" and RESULTS.md section 4.2): `pipelines` in `screen/amend1/h100_speed.json`
  (second pod) and in `screen/first/h100_speed.json` (first pod), each value the slowest of 3 processes;
- the screen's 58 of 58 rows in each of four files: `rows`, each with `pass`, in
  `screen/{first,amend1}/screen_{uniform,depspam_fm}.json`;
- the profile: `second_pod/prof/summary.md`.

The screened `h100_speed.json` files hold the same values as the pods' own (`first_pod/h100_speed.json`,
`second_pod/h100_speed.json`) apart from the verdicts, and each records in `rows_sha256` the sha256 of its pod's
`rows.jsonl` (50c7310c… for the first pod, d0886cfd… for the second).
