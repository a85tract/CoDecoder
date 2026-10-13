# Phase C: the run

Phase C's run up to the lock (10-11 October 2026, on the Linux workstation) followed, at d9b0b899, a runbook that is not
part of this release, step by step, and submitted C4b and C7 on the lock. The study lead stopped it at 01:06 PDT on 11
October so that it could not submit the confirmation reads on that lock; at 01:20 PDT the authors chose to amend the
selection rule. The amendment's run then made Amendment 1 (`PHASE_C_PLAN.md` 4.3), ran the rule again on the same
development data, committed the new lock and submitted every remaining job on it. No confirmation shot (seed kind 11)
had been read when the new lock was committed, and none had been read when this file was written.

The jobs chain by themselves through Slurm dependencies. A last job writes
`<cluster-scratch>/cascade_w4/PHASE_C_JOBS_DONE` (one line per job: id, name, final state, exit code, elapsed
time) once every other job has ended, whatever the outcome. The report is written from there.

Times are PDT. Slurm, the logs and the files on Alpine use MDT, an hour later. `$R` is
`<cluster-scratch>/cascade_w4`.

## Checked before the first job

- Branch `ising-cascade-phasec` at d9b0b899, cloned from `~/work/repo.git`; the Cascade tree of `ising-cascade-net` at
  64b2e25d.
- Deployment at 17:47: `$R/code.20261010_174740`, whose `DEPLOYED` reads `net 64b2e25d phasec d9b0b899` (not dirty).
  The 310 deployed files carry the git blob hashes of 64b2e25d (the paths `deploy.sh` ships) and of d9b0b899
  (`cascade_compare/phase_c` without `dry/`): the sorted lists of blob hash and path, computed on Alpine with
  `git hash-object` and here with `git ls-tree`, have the same md5 (010fc0a3abba0237ea3a5a87e5a2b253). Nothing was
  redeployed.
- Weights: the sha256 of all nine checkpoints match plan 2.1 (its first 16 hex digits). C6' (`C6p`,
  `$R/ckpt/C6_dfm/final_ema.pt`, written at 16:58) has sha256
  5fd3bca606953ee88b09c1378bc996846313698474a97829cf2ba0199ef5c1c9, and `sha256sum -c` passes against the
  `SHA256SUMS` beside it.
- Pins: bb's `venv_cu128` has torch 2.11.0+cu128 and bb's CPU venv has relay-bp 0.2.2. Every GPU log starts with
  `NVIDIA RTX PRO 6000 Blackwell Server Edition, 590.48.01, Disabled`: a full card with MIG off, driver 590.48.01.
- Alpine's Slurm runs with `kill_invalid_depend`: a job whose `afterok` dependency fails is cancelled at once, so the
  last (`afterany`) job cannot be left waiting.

## Amendment 1 (11 October)

The registered rule (plan 4.3, step 3) took the speed-feasible candidate with the fewest development failures and sent
only exact ties to the lower predicted L1. The development curves are flat above a small routed fraction, so it picked
large routed fractions for one or two fewer failures: in `uniform` at 0.3 % the first lock took S55 -> C14 at q1 = 40 %
(1,263 failures on 500,000 shots, L1 802 us) where 5 % had 1,276 failures at 177 us and Cascade alone 1,264 at
1,783 us. At 01:20 PDT the authors chose to take the configuration with the largest overall gain, without putting
accuracy or speed first. Rule version A, for S -> B at every p and for S -> B -> R at 0.2 % and 0.3 %, in both
settings: among the speed-feasible candidates (all of them if none is feasible), F is the fewest development failures;
the candidates with at most F + sqrt(F) failures are tied, and the rule takes the tied candidate with the lowest
predicted L1 (then fewer failures). S comes out of the S -> B pick as before, and S -> R keeps that S. Everything else
stays as registered: the candidates, feasibility, the Holm family of 132, the sizing algorithm and its caps, the
non-inferiority margins, the check's tolerance and the timing protocol.

The note at the end of plan 4.3 gives the reason, both rules and the disclosures of the gate review finding 1: the amendment
is outcome-informed. It was made before any confirmation read, after the development curves had been seen and, on
another branch, the H100 failure counts of the first lock's configurations on kind 12, chunks 10-19 (the check's
shots). The reviewer preferred a paired band for S -> B and the accuracy-first rule for S -> B -> R; the authors chose version A
for both. The amendment was frozen before the new check, with no later tuning to obtain a pass. The gate review on the
amendment gave seven findings (PROCEED WITH CHANGES) with dispositions, which are not part of this release.

In the code only the pick in `pc_route.py select` changed (commit 08542cd8). Each pick in the lock now records `rule`,
`F`, `band_bound` and `tied` (S, q1, r, score, failures, L1 and A of every tied candidate).

## Deployments

- 17:47 PDT, 10 October (phase C's run up to the lock): `$R/code.20261010_174740`, `DEPLOYED` `net 64b2e25d phasec
  d9b0b899`. C1, C1b, C2, C3, C4 and the first lock's C4b ran on it.
- 01:30 PDT, 11 October (the amendment's run): `$R/code.20261011_013018`, `DEPLOYED` `net 64b2e25d phasec 08542cd8` (not
  dirty); `$R/code` points at it. Hash check as in the first deployment: the sorted lists of blob hash and path of the
  313 deployed files, computed on Alpine with `git hash-object` and here with `git ls-tree`, have the same md5
  (55a62ffa5752c0966c28ca92cac6358d). The 313 are the 310 of the first deployment and the three files of the first
  lock that 1e8d1c6d added under `phase_c/out/`, which no job reads. Against d9b0b899 the deployed `phase_c` differs
  only in `pc_route.py` and those three files. Every job from the second C4 on runs this tree.

## Commits (branch `ising-cascade-phasec`)

- 1e8d1c6d, 21:53 PDT on 10 October: the first lock, now superseded. `locked_uniform.json`
  4644fbcc1b65a34c40fe60410c48859b6d1d60f4503371288e2b3d03f11be760, `locked_depspam_fm.json`
  02b19c451c6a38f6854a6be665b9bdcc87f8feb53d86841309833a70f9be0e81, `speed_components.json`
  1ec6b2563060f0042f866f19841631e8e1499872e6c15ed77dae7a01c8c811f5 (a copy of `$R/out/speed/speed_components.json`).
  Both locks froze that sha256, hold 16 configurations each and name C3's card, driver and torch (NVIDIA RTX PRO 6000
  Blackwell Server Edition, 590.48.01, 2.11.0+cu128). Their confirmation blocks: 1,750 chunks of 10,000 shots in
  `uniform`, 2,100 in `depspam_fm`.
- 08542cd8, 01:30 PDT on 11 October: Amendment 1 (`pc_route.py select` and `PHASE_C_PLAN.md` 4.3).
- 629ca1c4, 01:33 PDT: the new lock, `cascade_compare/phase_c/out/locked_uniform.json`
  1748c2d4d3b94fbc2df427df4f9bd6a1026c5ba2163266f5e57691f9704da6b1 and `locked_depspam_fm.json`
  398d02c60e08535a16a1d37a325e3aec9537f8a3608c794c985f471c0be1c41d (on Alpine in `$R/out/amend1`).
  `speed_components.json` is unchanged.
- The commit that adds this file also adds the first lock's check as its record,
  `phase_c/out/lock_1e8d1c6d/check_uniform.json` (dc381b3262cc46252b8ae80abc831346b62da7f5457d15e9807459489a770298) and
  `check_depspam_fm.json` (f5dfcdc1e44dcd23ed1f310dd950dadb683dfd1d052f367f7cff988d3b2cb4f7), copies of
  `$R/out/check_<setting>.json`.
- The check commit of the new lock had not been made when this file was written.

## The new lock

The second C4 (job CJ37, on acpu, 51 s) ran `select.sh` with `LOCKD=$R/out/amend1` on the first lock's inputs:
`$R/out/dev`, `$R/out/relay_dev` and `$R/out/speed/speed_components.json`. Compared field by field with jq, the new
locks equal the first ones except for the picks and what follows from them (`configs`, `joint_power`, the timings of
the run): every candidate row (156 in `uniform`, 310 in `depspam_fm`), the confirmation sizes (`conf_chunks`, again
1,750 and 2,100 chunks), the non-inferiority margins, the networks and their read precisions, the frozen paths, the
card, driver and torch.

The picks changed in 21 of the 32 configurations: 19 of the 24 S -> B and S -> B -> R picks, and the two S -> R
configurations of `depspam_fm` at the claim points, whose S follows the S -> B pick (S55p, that is S55', instead of
S55p_deep). Every changed configuration, with the development failures (300,000 to 10,000,000 shots
per p) and the predicted latencies in us. "S q1" gives q1 in %, and "/r score" gives r in % and B's score. "Cascade" is
Cascade alone on the same development shots. F, bound and tied: the fewest failures among the feasible candidates, F +
sqrt(F), and the number of tied candidates.

| setting | type | p (%) | Cascade | first lock: S q1/r score | fail | L1 | A | new lock: S q1/r score | fail | L1 | A | F; bound; tied |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| uniform | S-B | 0.1 | 238 | S55 2 | 236 | 124 | 5.6 | S55 1 | 237 | 106 | 4.9 | 236; 251.4; 6 |
| uniform | S-B | 0.2 | 372 | S55 2 | 369 | 124 | 5.6 | S55 1 | 371 | 106 | 4.9 | 369; 388.2; 6 |
| uniform | S-B-R | 0.2 | 372 | S55 40/0.1 softplus | 454 | 905 | 39.2 | S55 1/0.1 minabs | 471 | 202 | 10.9 | 454; 475.3; 12 |
| uniform | S-B | 0.25 | 496 | S55 20 | 496 | 445 | 18.5 | S55 2 | 512 | 124 | 5.6 | 496; 518.3; 5 |
| uniform | S-B | 0.3 | 1264 | S55 40 | 1263 | 802 | 32.8 | S55 5 | 1276 | 177 | 7.7 | 1263; 1298.5; 4 |
| uniform | S-B-R | 0.3 | 1264 | S55 40/0.1 minabs | 1110 | 1189 | 57 | S55 5/0.1 minabs | 1133 | 577 | 32.7 | 1110; 1143.3; 7 |
| uniform | S-B | 0.35 | 1615 | S55 20 | 1611 | 445 | 18.5 | S55 10 | 1618 | 267 | 11.3 | 1611; 1651.1; 3 |
| uniform | S-B | 0.4 | 3438 | S55 40 | 3439 | 802 | 32.8 | S55 20 | 3445 | 445 | 18.5 | 3439; 3497.6; 2 |
| uniform | S-B | 0.45 | 6318 | S55 40 | 6317 | 802 | 32.8 | S55 20 | 6365 | 445 | 18.5 | 6317; 6396.5; 2 |
| depspam_fm | S-B | 0.1 | 561 | S55p 2 | 548 | 124 | 5.6 | S55p 1 | 550 | 106 | 4.9 | 548; 571.4; 12 |
| depspam_fm | S-B | 0.15 | 750 | S55p 10 | 737 | 266 | 11.3 | S55p 1 | 740 | 106 | 4.9 | 737; 764.1; 12 |
| depspam_fm | S-B | 0.2 | 701 | S55p_deep 2 | 696 | 167 | 7.7 | S55p 1 | 720 | 106 | 4.9 | 696; 722.4; 12 |
| depspam_fm | S-B-R | 0.2 | 701 | S55p_deep 5/0.1 minabs | 657 | 334 | 17 | S55p 2/0.1 minabs | 670 | 238 | 12.8 | 657; 682.6; 22 |
| depspam_fm | S-R | 0.2 | 701 | S55p_deep q 0.25 | 942 | 413 | 23.9 | S55p q 0.25 | 1042 | 379 | 22.3 | a priori (S from S -> B) |
| depspam_fm | S-B | 0.25 | 945 | S55p_deep 2 | 936 | 167 | 7.7 | S55p_deep 1 | 957 | 149 | 7 | 936; 966.6; 10 |
| depspam_fm | S-B | 0.3 | 2305 | S55p_deep 10 | 2297 | 309 | 13.4 | S55p 5 | 2343 | 177 | 7.7 | 2297; 2344.9; 8 |
| depspam_fm | S-R | 0.3 | 2305 | S55p_deep q 2 | 3180 | 4623 | 287 | S55p q 2 | 3452 | 5013 | 312 | a priori (S from S -> B) |
| depspam_fm | S-B | 0.35 | 2910 | S55p_deep 20 | 2903 | 487 | 20.6 | S55p_deep 5 | 2943 | 220 | 9.8 | 2903; 2956.9; 7 |
| depspam_fm | S-B | 0.4 | 5752 | S55p_deep 20 | 5745 | 487 | 20.6 | S55p_deep 10 | 5813 | 309 | 13.4 | 5745; 5820.8; 5 |
| depspam_fm | S-B | 0.45 | 10677 | S55p_deep 40 | 10667 | 844 | 34.9 | S55p_deep 20 | 10680 | 487 | 20.6 | 10667; 10770.3; 3 |
| depspam_fm | S-B | 0.5 | 18716 | S55p_deep 40 | 18714 | 844 | 34.9 | S55p 40 | 18779 | 801 | 32.8 | 18714; 18850.8; 2 |

Unchanged: `uniform` S -> B at 0.15 % (the fewest failures there also had the lowest L1), `uniform` S -> B at 0.5 and
0.55 % and `depspam_fm` S -> B at 0.55 % and S -> B -> R at 0.3 % (each the only candidate in its band), and the other
six S -> R configurations.

At the claim points the new lock plans these powers against Cascade (the first lock's in brackets): `uniform` S -> B ->
R 0.054 (0.068) at 0.2 % and 1.0 (1.0) at 0.3 %; `depspam_fm` S -> B -> R 0.99975 (1.0) and 1.0 (1.0), S -> R 0.977
(0.994) and 0.999 (1.0). S -> B has no planned power at either claim point in either lock (its discordant rate is below
10 % of Cascade's failure rate). Joint powers: `uniform` S -> B -> R 0.054 (0.068); `depspam_fm` S -> B -> R 0.99975
(1.0), S -> R 0.976 (0.994); the others are unchanged.

## Jobs

`$G` is `-p artxpro6000 --gres=gpu:rtx_pro_6000:1`. Each job text is on Alpine as `$R/jobs/<name>.sh` (those on the new
lock also in a `jobs_a1/` directory on the Linux workstation, submitted by a `submit_rest.sh` there), and each log is
`$R/logs/<name>_<job>.out`. Every job on the new lock has `_a1` in its name.

| step | job | name | resources | waits for | ran (PDT) | state when this file was written |
|---|---|---|---|---|---|---|
| C1 | CJ28 | pc_c1_uniform | `$G -c 16 --mem=96G -t 06:00:00` | - | 17:48-19:30 | COMPLETED, 1:42:00 |
| C1 | CJ29 | pc_c1_depspam | `$G -c 16 --mem=96G -t 06:00:00` | - | 17:48-19:30 | COMPLETED, 1:42:12 |
| C3 | CJ30 | pc_c3_speed | `$G -c 16 --mem=96G -t 06:00:00` | - | 19:30-21:38 | COMPLETED, 2:07:50 |
| C1b | CJ33 | pc_c1b_late | `$G -c 16 --mem=96G -t 03:00:00` | - (C6''s weights were on Alpine) | 19:30-19:49 | COMPLETED, 0:19:12 |
| C2 | CJ31 | pc_c2_relay | `$G -c 32 --mem=96G -t 01:00:00` | afterok CJ28, CJ29 | 19:50-19:56 | COMPLETED, 0:05:52 |
| C4 | CJ32 | pc_c4_select | acpu, `-c 8 --mem=64G -t 01:00:00` (18 CPUs allocated) | afterok CJ31, CJ30 | 21:51-21:52 | COMPLETED, 0:00:51 |
| first lock | | | | C4 | commit 1e8d1c6d, 21:53 | superseded |
| C4b | CJ34 | pc_c4b_check | `$G -c 32 --mem=96G -t 03:00:00` | the first lock's commit | 00:37-01:09 | COMPLETED, 0:31:45; every row passed in both settings (the first lock's record) |
| C7 | CJ35 | pc_c7_pipe | `$G -c 24 --mem=96G -t 08:00:00` | the first lock's commit | never started | held, then CANCELLED by the amendment's run at 01:30 |
| C4 again | CJ37 | pc_c4_select_a1 | acpu, `-c 8 --mem=64G -t 01:00:00` (18 CPUs allocated) | the deployment of 08542cd8 | 01:31-01:32 | COMPLETED, 0:00:51 |
| new lock | | | | C4 again | commit 629ca1c4, 01:33 | |
| C4b again | CJ38 | pc_c4b_check_a1 | `$G -c 32 --mem=96G -t 03:00:00` | the new lock's commit | | PENDING (Priority; Slurm's estimate 07:10) |
| C7 again | CJ39 | pc_c7_pipe_a1 | `$G -c 24 --mem=96G -t 08:00:00` | the new lock's commit | | PENDING (Priority; Slurm's estimate 10:10) |
| C5 | CJ40 | pc_c5_uniform_a1 | `$G -c 16 --mem=96G -t 06:00:00 --kill-on-invalid-dep=yes` | afterok CJ38 | | PENDING (Dependency) |
| C5 | CJ41 | pc_c5_depspam_a1 | `$G -c 16 --mem=96G -t 06:00:00 --kill-on-invalid-dep=yes` | afterok CJ38 | | PENDING (Dependency) |
| C6 | CJ42 | pc_c6_conf_a1 | `$G -c 32 --mem=96G -t 02:00:00 --kill-on-invalid-dep=yes` | afterok CJ40, CJ41 | | PENDING (Dependency) |
| C8 | CJ43 | pc_c8_report_a1 | acpu, `-c 8 --mem=64G -t 01:00:00 --kill-on-invalid-dep=yes` | afterok CJ42, CJ39 | | PENDING (Dependency) |
| done | CJ44 | pc_jobs_done_a1 | acpu, `-c 1 --mem=2G -t 00:15:00` | afterany CJ38, CJ39, CJ40, CJ41, CJ42, CJ43 | | PENDING (Dependency) |

The texts of the jobs on the new lock (one command per line in each file):

| name | text |
|---|---|
| pc_c4_select_a1 | `mkdir -p $R/out/amend1`, `LOCKD=$R/out/amend1`, `source $PC/jobs/select.sh` |
| pc_c4b_check_a1 | `LOCKD=$R/out/amend1`, `source $PC/jobs/check.sh` |
| pc_c7_pipe_a1 | `LOCKD=$R/out/amend1`, `source $PC/jobs/speed_pipe.sh` |
| pc_c5_uniform_a1, pc_c5_depspam_a1 | `SETTING=<setting>`, `LOCKED=$R/out/amend1/locked_<setting>.json`, a `grep -q` that stops the job unless `$R/out/amend1/check_<setting>.json` records the sha256 of `$LOCKED` as `locked_sha256`, an `echo`, `source $PC/jobs/confirm.sh` |
| pc_c6_conf_a1 | `LOCKD=$R/out/amend1`, `source $PC/jobs/confirm_relay.sh` |
| pc_c8_report_a1 | `ADDON=1`, `O=$R/out/amend1`, `SP=$R/out/speed`, `SPL=$R/out/speed_late`, `PAPER=$R/out/paper_timing.json`, `RO=$R/out/report`, `source $PC/jobs/report.sh` |
| pc_jobs_done_a1 | `sacct` of the 15 phase-C jobs above, from CJ28 to CJ43, into `$R/PHASE_C_JOBS_DONE` |

`report.sh` reads the development curves from `$O/dev` and `$O/dev_late`; `$R/out/amend1/dev` and
`$R/out/amend1/dev_late` are links to `../dev` and `../dev_late`. `pc_report.py` matches every check and confirmation
to its lock by sha256 and never by path.

## Where the files are

| what | path on Alpine |
|---|---|
| the new locks, the new check, the confirmation analyses | `$R/out/amend1/locked_<setting>.json`, `check_<setting>.json`, `confirm_<setting>.json` |
| the first lock and its check (the record; nothing there was moved or replaced) | `$R/out/locked_<setting>.json`, `$R/out/check_<setting>.json` |
| the check's reads (kind 12, chunks 10-19, both checks) | `$R/out/check/<setting>/read` and `eager` |
| the development reads and Relay-BP's development runs | `$R/out/dev`, `$R/out/dev_late`, `$R/out/relay_dev` |
| the confirmation reads and Relay-BP on them | `$R/out/conf/<setting>`, `$R/out/relay_conf/<setting>` |
| the speed components and rows (C3's, and C7's pipelines tagged with the new lock's sha256) | `$R/out/speed` |
| the report | `$R/out/report` |

## The check on the new lock

The second C4b runs `check.sh` with the default `CKD` (`$R/out/check`), so it reuses the reads of the first C4b. That
is valid because the computation and the provenance are the same. `check.sh` always calls the read wrapper
`curves.sh`, which skips a group whose marker `done_<p,...>` exists. All four read directories (`uniform` and
`depspam_fm`, `read` and `eager`) hold `done_0.002,0.003`, written by the first C4b. Those reads
are seed kind 12, chunks 10-19, of every network (the eager ones of C6 and C6'), made by tree d9b0b899, whose read code
(`curves.sh`, `pc_eval.py`, `pc_common.py`, `pc_fused.py`) is byte for byte the new tree's. The reads do not depend on
the lock. `pc_speed.py check` then recomputes every row against the new lock: each network alone on its frozen paths,
the compiled reads against the eager ones, and every locked configuration in single and batched mode at both claim
points, with one Relay-BP decision per shot of the new routed union. It writes only
`$R/out/amend1/check_<setting>.json`.
The first C4b's outputs stay in `$R/out` (all 48 rows passed in each setting, `decoders_failing` empty), and the
second C4b writes only into `$R/out/amend1`.

## Deviations from the runbook

1. Amendment 1 and the second lock, as above. C1, C1b, C2 and C3 were not rerun; C4 ran again on their outputs.
2. The new locks, checks and confirmation analyses are in `$R/out/amend1`. The jobs take `LOCKD` or `LOCKED`, and C8
   takes `O` (with the two links), `SP`, `SPL`, `PAPER` and `RO`, so that the paper's timings and the report keep the
   runbook's paths.
3. The held C7 (CJ35) never started; it was cancelled and submitted again on the new lock and tree (CJ39), with
   the same resources.
4. C5 was submitted before the check commit, waiting for the second C4b with `afterok`. At 01:34 Slurm estimated the
   check's start at 07:10 PDT. No full RTX PRO 6000 card was free, a job of higher priority was waiting for one, and
   the earliest scheduled release was at 04:09. The study lead's instructions allowed this when the check could not
   start within about 2 hours. C5 still reads nothing before the check has ended without error, and each C5 job also
   stops unless its setting's check comes from the new lock.
5. C5, C6 and C8 request `--kill-on-invalid-dep=yes`. Phase C's run up to the lock noted that Alpine runs with
   `kill_invalid_depend`, but the option sits in `SchedulerParameters` (`DependencyParameters` is empty), and a job of
   another user has been waiting in `DependencyNeverSatisfied` on a failed `afterok` dependency, so the jobs do not rely
   on the system default. `scontrol show job` gives `KillOInInvalidDependent=Yes` for all four.
6. The check commit was left until the jobs had ended.

## Hours

GPU hours used so far, all on full RTX PRO 6000 cards: C1 3.4, C3 2.1, C1b 0.3, C2 0.1 (its GPU idle) and the first
C4b 0.5, about 6.5 in all. The CPU jobs (C4 twice) took 51 s each.

[The instructions to the report's author are not part of this release.]

## The H100 screen job

The H100 screen job CJ36 (`h100t_screen`, branch `ising-cascade-h100t`, acpu) waited for the first C4b. It started when
that check ended and FAILED after 4 s (01:09 PDT). The amendment's run left it alone.
