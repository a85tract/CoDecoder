# Gross (BB144) stage 4: the registered test of cells G-4R and G-4F (plan and freeze record)

2026-10-07, branch `ising-bb-gross2`. This document is binding from its commit, the freeze commit,
which also holds the code (eac7ab90, unchanged since), the frozen numbers (`data/bb/stage4_gross/cell.json`), fusedt's
CAL read with its margins (`cal_fusedt.npz`) and the size calculation (`sizes.json`). The user asked for this stage on
10-07. Its rules: a fresh, unseen test set; the method frozen before the first test read; equal conditions; one
analysis, its command committed before the run and run once; everything reproducible. There are no guards, tickets or
pre-run reviews.

Terms. CAL: the calibration split (cal chunks 0-9, 100,000 shots at p = 0.004). Margin: the score gap between the
network's best and second-best logical class. tau: the routing threshold, set as the q-quantile of the 100,000 CAL
margins (`bb.route`'s rule), so that a shot goes to the fallback iff its margin < tau. fusedt: the frozen network run
by `bb.fused`'s Triton kernels with the launch settings tuned for the H100 (16 taps per iteration, 4 stages, no
programmatic dependent launch; `bb.fast.FUSEDT`, GROSS_FAST_ENGINE.md). Sector Relay-BP: relay-bp 0.2.2
`RelayDecoderF32` on the X-detector sector with the authors' gross parameters (`bb.opp relay`, G-4's fallback).
Full-DEM Relay-BP: the same decoder and parameters on the full detector error model, both sectors (`bb.opp
relay_full`). Composite: a hybrid's time per shot = GPU term (the network, batch 1) + handoff + CPU term (the routed
share x the fallback's mean time per routed shot). test4: this stage's new test split.

## 0. The frozen method

| part | frozen value | plain meaning |
|---|---|---|
| network | g2_D12k113c10 `final_ema.pt`, sha256 `eb380295a6eccfde022d8e61ad1c3f2b9154d54b3c04cc5eea84e3f91c224692` | G-4's network, unchanged |
| engine | fusedt on an NVIDIA H100 80GB HBM3: torch 2.11.0+cu126, CUDA 12.6, Triton 3.6.0, `engine.py` ca6c91f3b6c29c97, `net.py` 36e2b562e3acad8c, `fused.py` and `fast.py` as in the fast-engine study; signature digest 9bab38ece6d00140 (`cell.json` "engine") | the fast-engine study's fusedt, signature included |
| G-4R | margin < tau1 = 2.2055678367614746 goes to sector Relay-BP | tau1 = the 1 % quantile of fusedt's CAL margins |
| G-4F | margin < tau2 = 6.272581100463867 goes to full-DEM Relay-BP | tau2 = the 2 % quantile |
| beside, not gated | G-4's old q: margin < tau0 = 0.5653958916664124 goes to sector Relay-BP | the 0.5 % quantile of fusedt's margins |
| fallbacks | sector Relay-BP: alpha 1.0, gamma0 0.15, pre_iter 80, num_sets 300, set_max_iter 60, gamma_dist_interval (-0.22628432386414646, 0.6216020925981884), stop_nconv 5; full-DEM Relay-BP: the same on the full DEM | a routed shot takes the fallback's prediction from its test4 read |
| records | `cell.json` (every frozen number and the analysis command), `cal_fusedt.npz` (fusedt's CAL read: margins, decisions, signature), `sizes.json`, `pod/` (the pod's scripts, environment and pre-freeze records), `mac/` (the Mac mini's session script) | committed with this plan |

Evidence on CAL (`cell.json` "cal"; in-sample for every tau):

| quantity | value | plain meaning |
|---|---|---|
| network alone | 717 failures | |
| G-4R (q 1 %) | 259 (1,000 routed) against sector Relay-BP's 490; paired 218 both / 41 only G-4R / 272 only Relay-BP | 47 % fewer |
| G-4F (q 2 %) | 156 (2,000 routed), with the full-DEM CAL study's read of the 3,000 least confident CAL shots (every routed shot is among them) | full-DEM Relay-BP alone: about 268 (95 % 199-415), at least 179 (GROSS_FULLDEM_CAL.md) |
| G-4F, worst case against full-DEM Relay-BP | 146 both / 10 only G-4F / 33 only full-DEM | full-DEM Relay-BP counted right on every CAL shot the full-DEM CAL study did not decode |
| G-4's old q (0.5 %) | 406 (graph32: 409) | |

fusedt's CAL read on this stage's pod equals the fast-engine study's read bit for bit (100,000 margins; `cell.json` "cal_read_vs") and
differs from stage 2's graph32 read in 3 decisions (the fast-engine study's result).

### 0.1 Changes against stage 3 (cell G-4)

| item | stage 3 | now | plain meaning |
|---|---|---|---|
| engine | graph32 (3.43 ms) | fusedt (0.257 ms, section 8) | same network, 13x faster on the GPU side |
| routing | q 0.5 % to sector Relay-BP | G-4R: q 1 % to sector Relay-BP; G-4F: q 2 % to full-DEM Relay-BP | the CAL studies' recommendations (GROSS_BPOSD_FALLBACK.md 4, GROSS_FULLDEM_CAL.md 5) |
| test set | test, 300,000 shots, read once | test4, 150,000 fresh shots | G-4's test set has been analysed |
| speed gate before the freeze (Q3) | session A on CAL | none | on CAL's numbers G-4R's composite is about 9.7 ms against cudaq-qec's 15.7 ms (section 4.3) |
| full-DEM Relay-BP | first 10,000 test shots, reported | every test4 shot; G-4F's gating reference | the paired test needs it on every shot |
| CPU reads | one Derecho job, 128 cores | two Derecho jobs, 128 + 64 cores (the 192 allowed) | about 560 core-hours, mostly full-DEM Relay-BP |

## 1. Cells, claims and alpha

| cell | claim (each comparison at alpha = 0.05 / 3) | speed condition |
|---|---|---|
| G-4R | fewer block failures than each of sector Relay-BP, BP-OSD ms80-CS60 (ldpc) and cudaq-qec GPU BP-OSD ms80-CS60 | composite <= the fastest of these three on the same hosts (Relay-BP and BP-OSD on the Mac mini, cudaq-qec on the H100) |
| G-4F | fewer block failures than full-DEM Relay-BP alone and than each of the three gating opponents | composite <= full-DEM Relay-BP's own mean time per shot on the idle Mac mini (the weaker condition, as BB72-3F); the fastest gating opponent is reported beside, not gated |

- Code [[144,12,12]], p = 0.004, 12 rounds (the qnd circuit, X basis), as every gross split.
- A cell's claim needs all its comparisons (intersection-union, no correction within the cell) and its speed condition.
- Each cell spends a new share of 0.05 / 3, as Amendments F and S did. The study-wide union bound goes from 7 x 0.05 / 3
  = 0.117 to 9 x 0.05 / 3 = 0.15.
- Reported beside, not gated: the network alone, G-4's old q (tau0) with sector Relay-BP, every decoder's failures
  with a Wilson 95 % interval, the routed shares and every time.
- Not claimed: best-known accuracy, other codes, noise levels, noise models or hardware.

## 2. Test set

| split | p | chunks x shots | TAG | plain meaning |
|---|---|---|---|---|
| test4 (code bb144) | 0.004 | 15 x 10,000 = 150,000 | b"ising_compare bb test4 G-4R G-4F\n" (`bb.shots.TEST4_TAG`) | one fresh split for both cells |

Size (`bb/scripts/stage4_gross_sizes.py`, output `sizes.json`). PLAN 3.3's rule with `bb.route.power`: the full
registered decision (exact McNemar p < 0.05 / 3 and the paired Newcombe interval at 1 - 0.05 / 3 below 0), 20,000
draws, seed 20261006, target power 1 - 0.10 / k per comparison for a cell with k comparisons, multiples of 10,000 from
100,000 to 300,000. The size is 150,000 (GROSS_FULLDEM_CAL.md section 5) unless G-4R's rule needs more.

| cell, comparison (per 100,000: both fail / only the cell / only the opponent) | n by the rule | power at 150,000 | plain meaning |
|---|---|---|---|
| G-4R vs sector Relay-BP, CAL: 218 / 41 / 272 | 100,000 | 1.000 | |
| G-4R vs sector Relay-BP, gap one standard error smaller: 218 / 50 / 263 | 100,000 | 1.000 | |
| G-4R vs sector Relay-BP, no shared failure: 0 / 259 / 490 | 100,000 | 1.000 | the worst pairing |
| G-4R vs BP-OSD, cudaq-qec, no shared failure: 0 / 259 / 2,615 and 2,575 | 100,000 | 1.000 | stage-1 dev rates |
| G-4F vs full-DEM Relay-BP, worst case on CAL: 146 / 10 / 33 | 160,000 | 0.974 | below its target 0.975 by 0.001 |
| G-4F vs sector Relay-BP, CAL: 64 / 92 / 426; the other rows as G-4R's | 100,000 | 1.000 | |

n = 150,000. G-4R needs 100,000. G-4F's weakest comparison has power 0.974 at 150,000 in its worst case; on the
estimate of full-DEM Relay-BP's CAL rate (about 268 against 156) its power is higher. Nothing is extended after a
test4 shot exists.

Seeds (stage 3's construction, PLAN 3.5): seed(TAG, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j
as 2 bytes big-endian)) mod 2^62 for chunks j = 0..14. R = the 32 randomness bytes of drand mainnet (`icmp.t2`, at
least two endpoints agreeing). Round = the first one at least 600 s after max(this commit's committer time, the time
this commit was seen public on GitHub). Record: `data/bb/stage4_gross/beacon.json` (`python -m bb.stage4_gross
beacon`).

Sampler and host: `bb.shots`'s call per chunk (`python -m bb.shots --split test4`), Stim 1.16.0, NumPy 2.2.6, Python
3.12.14 in the pod's g2venv, on the pod's x86-64 CPU (Intel Xeon Platinum 8480+, avx2). `bb.stage3_gross hostcheck`
runs first: cal chunk 0 (sampled on Alpine) and D-bank chunk 10 (sampled on the gross2 pod) regenerated bit for bit;
it passed before the freeze (`pod/hostcheck.json`). If it fails at sampling time, sampling moves to Derecho CPU with
the same check. The 15 manifest rows (file, sha256, shots, seed, host, cpu, stim) and the beacon record are committed
and pushed before any test4 read; `shots.load` checks each file's sha256 on every read.

## 3. Reads

| decoder | configuration | test4 read | host | estimate |
|---|---|---|---|---|
| the network | section 0, `python -m bb.stage4_gross infer --split test4 --chunks 0-14` | all 150,000 | H100 pod | about 1 min |
| cudaq-qec GPU BP-OSD ms80-CS60 | cudaq-qec 0.8.0 nv-qldpc-decoder: min-sum scale 1.0, 80 iterations, OSD-CS order 60, X sector; `bb.gpu_bposd read`, one whole read | all 150,000 | H100 pod | about 10 min |
| sector Relay-BP | section 0 | all 150,000 | Derecho CPU, job A | 6 core-h |
| BP-OSD ms80-CS60 | ldpc 2.4.1 `BpOsdDecoder`, min-sum scale 1.0, 80 iterations, parallel schedule, OSD-CS order 60, X sector | all 150,000 | Derecho CPU, job A | 5 core-h |
| full-DEM Relay-BP | section 0 | all 150,000: chunks 0-9 in job A, 10-14 in job B | Derecho CPU | about 550 core-h (13.1 s per shot, GROSS_FULLDEM_CAL.md 4) |

Derecho: PBS develop queue (cpudev), account <account>, the qldpc venv (relay-bp 0.2.2, ldpc 2.4.1) as in stage 3;
`bb/scripts/derecho/cpu_s4_a.pbs` (128 cores: `bb.opp --dec relay,bposd` on all chunks, then `bb.stage4_gross
fullread` on chunks 0-9) and `cpu_s4_b.pbs` (64 cores: `fullread` on chunks 10-14), each with 6 h of walltime and
about 3 h expected. `fullread` gives the processes one shot at a time across all its chunks, saves every minute and
resumes when resubmitted. Code: a git archive of the commit with the test4 manifest rows. The CPU opponents' accuracy
does not depend on the host, so no read is repeated elsewhere.

## 4. Timing (stage 3's protocol; no time is converted between hosts)

### 4.1 GPU side (the H100 pod, session `H100 session s4`)
- `nvidia-smi -q` and the pod record before the first timed process (`pod/`).
- Acceptance per timed process (`bb.timing.gpu_watch`, as stage 3): just before it, two device-wide samples with
  utilization 0 % and less than 1,024 MiB used; during it, one compute app and every 100-ms SM clock sample at least
  97 % of the maximum. Nothing else runs on the pod meanwhile.
- GPU term and handoff: `python -m bb.fast lat --kind fusedt` (bb.latency's protocol: 200 warm-up replays, 5 x 100
  timed replays with CUDA events), 3 accepted fresh processes; counted: the slowest GPU term, and the larger of 1.1 us
  and the slowest handoff. Measured before this commit (section 8).
- cudaq-qec: its batch-1 call on the first 300 test4 shots (`bb.gpu_bposd time`), 3 accepted fresh processes;
  counted: the fastest.

### 4.2 CPU side (the idle Mac mini, Apple M4, session `M4 session s4`, script `mac/mac_s4.sh`)
- One single-threaded process at a time from one non-interactive shell, thread counts 1, default QoS. Before each
  process `ps` must show no Python process at 1 % CPU or more and no process at 50 % or more (otherwise it waits, up
  to 30 min); the load average and the three busiest processes are logged before and after each process. Each process
  first times a fixed 0.1-s integer loop, recorded with its row (an efficiency core shows). Versions: Python 3.12.14,
  relay-bp 0.2.2, ldpc 2.4.1, Stim 1.16.0, NumPy 2.2.6.
- Warm-up, untimed: each decoder once over the first 100 shots of the set it will time (10 for full-DEM Relay-BP,
  whose shots take seconds each).
- Timed, three fresh processes each:
  1. sector Relay-BP and BP-OSD on the first 1,000 test4 shots; counted: the fastest;
  2. full-DEM Relay-BP on the first 300 test4 shots (G-4F's bar); counted: the fastest;
  3. G-4R's CPU term: sector Relay-BP on the first 1,000 routed test4 shots in split order; counted: the slowest;
  4. G-4F's CPU term: full-DEM Relay-BP on the first 100 routed test4 shots; counted: the slowest.
  CPU term = the routed share of all 150,000 test4 shots x the mean time per timed routed shot (`bb.stage4_gross
  cputerm`).

### 4.3 Speed conditions and expected values
- G-4R: GPU term + handoff + CPU term <= the fastest of sector Relay-BP and BP-OSD (Mac mini) and cudaq-qec (H100).
- G-4F: GPU term + handoff + CPU term <= full-DEM Relay-BP's mean time per shot on the Mac mini.
- Measured inequalities; no statistical test applies to them.

| quantity | expected | source | plain meaning |
|---|---|---|---|
| GPU term + handoff | 0.279 ms | section 8 | |
| G-4R's CPU term | 1 % x about 0.94 s = 9.4 ms | Relay-BP on routed CAL shots, Alpine 2.18 s per shot x the M4/Alpine ratio of stage 3's session A (0.43) | |
| G-4R composite against its bar | about 9.7 ms against about 15.7 ms (cudaq-qec; Relay-BP about 61 ms and BP-OSD about 79 ms on the M4) | stage 3's test4-like numbers | about 6 ms to spare |
| G-4F's CPU term | 2 % x about 17 s = 0.33 s | full-DEM Relay-BP on routed CAL shots, 55.7 s on loaded Derecho x 0.30 | |
| G-4F composite against its bar | about 0.34 s against about 4 s | full-DEM Relay-BP's own mean, 13.1 s on loaded Derecho x 0.30 | about 12-fold |
| Mac mini session | about 3.4 h | 1,000 x 0.94 s and 100 x 17 s three times each, 300 x 4 s three times | |

## 5. Statistics: one analysis, run once

- A cell's prediction per test4 shot: the network's decision if its margin >= the cell's tau, otherwise its
  fallback's test4-read prediction. A block failure: any of the 12 logical observables wrong.
- Per comparison, paired on the same 150,000 shots: the exact two-sided McNemar test at p < 0.05 / 3 and the paired
  Newcombe interval (method 10, `qldpc_neural/bench/paired.py`) at level 1 - 0.05 / 3 for LER(cell) - LER(opponent)
  entirely below 0 give "better" (`bb.route.paired`).
- A claim holds iff all the cell's comparisons are "better", its speed condition holds and the test4 read carries the
  frozen engine signature.
- The command, committed here: `python -m bb.stage4_gross analyze --out data/bb/stage4_gross`. It runs once, on the
  Mac mini (no compute on the shared Linux workstation), after every read and the timing session, from the records commit, which
  is mirrored to GitHub first. `results.json`, `RESULTS_stage4_gross.md`
  and `gross_stage4.png` are committed and pushed as produced. No failure count of any decoder on test4 is printed or
  looked at before this run (every test4 read stores predictions only). If the analysis crashes, only the defect is
  fixed (no rule, threshold or input changes), it reruns on the same inputs and the fix is disclosed.
- Published on the hub (this branch) and then in the release repository's staging branch `bb144-stage4` (from its
  main, scrubbed as SANITIZE.txt says; not pushed to GitHub): this plan, the code, `cell.json`, fusedt's CAL read, the
  beacon record, the test4 manifest, every decoder's per-shot test4 predictions, the network's margins, every timing
  row and the analysis outputs.

## 6. Order of steps

| # | step | where |
|---|---|---|
| 1 | code (eac7ab90); the pod: environment from gross2's lock and stage 3's cudaq-qec freeze, host check, fusedt's CAL read, GPU term | Linux workstation, pod |
| 2 | this freeze commit: plan, `cell.json` (`python -m bb.stage4_gross freeze`), `cal_fusedt.npz`, `sizes.json`, `pod/`, `mac/`; mirrored to GitHub | Linux workstation |
| 3 | beacon (`bb.stage4_gross beacon`), the 15 chunks sampled on the pod; manifest rows and beacon record committed and pushed | Linux workstation, pod |
| 4 | pod: the network's test4 read, cudaq-qec's batch-1 timing and test4 read; copy-back with sha256; the pod stopped | pod |
| 5 | Derecho jobs A and B (section 3); predictions copied back with sha256 | Derecho |
| 6 | the Mac mini session (4.2) | Mac mini |
| 7 | records commit (blind), mirrored to GitHub; the one analysis; outputs pushed | Linux workstation, Mac mini |
| 8 | per-shot outputs packed; release rows | Linux workstation |

Stops: a failed host check moves sampling to Derecho CPU; a busy Mac mini makes the timing wait; a Derecho job that
does not finish is resubmitted and resumes. On any stop of the pod's work the pod is stopped at once.

## 7. Hours and dollars

| item | amount | cost |
|---|---|---|
| H100 pod | about 1.5 h (setup, CAL read and GPU term 0.2 h; freeze, mirror and beacon wait 0.5 h; sampling and reads 0.4 h; copy-back and slack 0.4 h) | about $5 at $3.49 per hour; cap $15 |
| Derecho CPU | about 560 core-hours (128 + 64 cores for about 3 h) | allocation <account> (QEC's cap 10,000 core-hours, about 1,300 used before) |
| Mac mini | about 3.4 h of one core | $0 |

## 8. Freeze record

- The RunPod pod (created 18:08 PDT): one NVIDIA H100 80GB HBM3 (MIG off), Intel Xeon Platinum 8480+;
  `<pod-home>/g2venv` = gross2's lock exactly (LOCK_OK), `<pod-home>/cqvenv` = stage 3's cudaq-qec freeze exactly (CQLOCK_OK);
  code eac7ab90 (`pod/`: scripts, `allocation.txt`, environments, `nvidia_smi_q_summary.txt`).
- Host check passed: cal chunk 0 and D-bank chunk 10 regenerated bit for bit; avx2 present; Stim loads its SSE2 module
  by its own dispatch, as on the stage-3 pod.
- fusedt's CAL read (`python -m bb.fast eq --kind fusedt --label s4`, 26 s): equal bit for bit to the fast-engine study's read; against
  graph32 3 decisions and 6 routings at G-4's old tau differ, network alone 717 / 717, sector hybrid 406 / 409.
- Taus (`python -m bb.stage4_gross freeze --tag g2_D12k113c10_fusedt_s4`): tau1 = 2.2055678367614746 (q 1 %),
  tau2 = 6.272581100463867 (q 2 %), tau0 = 0.5653958916664124 (q 0.5 %, reported).
- GPU term (session H100 session s4, 3 processes, all accepted at clock ratio 1.0 with one compute app): 0.2569
  ms (the slowest; the others 0.2561 and 0.2560 ms), handoff 0.0219 ms (the slowest). Rows in `rows_gpu.csv` with the
  post-sampling cudaq-qec rows, committed with the records.
- The study lead mirrored this commit and the records commit to GitHub, before the beacon and before the analysis.
