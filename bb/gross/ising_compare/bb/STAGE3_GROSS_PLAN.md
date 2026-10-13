# Gross (BB144) stage 3: the registered test of cell G-4 (plan)

2026-10-07, branch `ising-bb-gross2` (on d71bc563). Status: plan. It becomes binding with the freeze
commit (step 6), which adds the code and the measured pre-freeze numbers (`data/bb/stage3_gross/cell.json`) and changes
no rule of this plan. This is the fresh registration that `PLAN_gross2_stage2.md` section 3 asks for after a passed
gate. Like `bb/PLAN.md` (version 3, 01755dde) and BB72's `bb/STAGE3_PLAN.md` (ising-bb, 151bebab), whose structure it
copies, it is a registration document in the repository (Q7). The network, tau and q were chosen on the D-bank and
CAL only; no test shot exists. No review before the run: the authors streamlined reviews on 10-07.

User rules for this stage (10-07): an unseen fresh test set; the method frozen before any test read; equal
conditions; one analysis (its command committed and pushed before the reads, run once, outputs pushed); public
reproducibility; no guards, locks, tickets or arbiter layers. Opponents are never timed under worse conditions than
ours: our times count the slowest of the fresh processes, theirs the fastest, and CPU timings run on an idle host.
RunPod at most $25.

Terms used below. CAL: the calibration split (cal chunks 0-9, 100,000 shots at p = 0.004), used only to set tau.
D-bank: 50,000 stored monitor shots at p = 0.004 (chunks 10-14), used to pick the network. Margin: the network's
confidence in its decision. tau: the routing threshold; q: the share of CAL shots below it. Composite: the hybrid's
time per shot = GPU term (the network, batch 1) + handoff + CPU term (routed share x the fallback's mean time per
routed shot).

## 0. The frozen hybrid

| part | frozen value | plain meaning |
|---|---|---|
| network | `ckpt/g2_D12k113c10/final_ema.pt`, 10,938,879 bytes, sha256 `eb380295a6eccfde022d8e61ad1c3f2b9154d54b3c04cc5eea84e3f91c224692` (Linux workstation: `~/work/stage_gross2/pod_out/ckpt/g2_D12k113c10/`, checked 10-07 14:20) | EMA weights of the gross2 D12 trunk at 113,238,016 samples plus its 1e7-sample cooldown (run id 53); the best of stage 2 on the D-bank (350 / 50,000) |
| engine | `graph32`: fp32, batch 1, CUDA graph; torch 2.11.0+cu126, CUDA 12.6, cuDNN 91002, `engine.py` ca6c91f3b6c29c97, `net.py` 36e2b562e3acad8c, GPU "NVIDIA H100 80GB HBM3" | the policy's engine signature, unchanged; the stage-3 pod is the same GPU model (H100 SXM) with gross2's locked environment |
| routing | a shot goes to the fallback iff its margin < tau = 0.5629201531410217 (`bb.route`) | tau = the 0.5 % quantile of CAL's margins (q = 0.005), the fine frontier's policy |
| fallback | sector Relay-BP: relay-bp 0.2.2 `RelayDecoderF32` on the X-detector sector, alpha 1.0, gamma0 0.15, pre_iter 80, num_sets 300, set_max_iter 60, gamma_dist_interval (-0.22628432386414646, 0.6216020925981884), stop_nconv 5 (`bb.opp relay`) | the opponent's own configuration (the authors' gross-code parameters); a routed shot takes sector Relay-BP's prediction from its test read, as on CAL |
| records | `data/bb/stage3_gross/frontierf_g2_D12k113c10.json` (the policy; sha256 1f5e2ad79eb48d64..., binds the CPU-term rows) and `selection_d.json` (stage 2's gate: pass) | committed with this plan |

Evidence on CAL (`data/bb/stage3_gross/sizes.json`): network alone 717 failures, sector Relay-BP 490, hybrid 409 (only
the hybrid fails on 217 shots, only Relay-BP on 298, both on 192; exact McNemar p = 0.0004); composite 15.12 ms
against the provisional 15.8 ms limit. tau is in-sample on CAL. On the D-bank, which played no part in setting tau,
tau routes 0.524 % of shots (262 of 50,000), and the network fails 256 per 100,000 on the shots it keeps (CAL: 297), so its side
of the hybrid shows no optimism; Relay-BP was not read there, and only the test measures the hybrid. q is not
re-chosen with the stage-3 hosts' times.

### 0.1 Disclosed changes against PLAN.md version 3

| item | PLAN v3 | now | plain meaning, effect |
|---|---|---|---|
| network | replicas r21 / r22, the probe gate (2.3), the two-replica dev gate (2.6) | one network from the exploratory gross2 route (`PLAN_gross2.md`, `PLAN_gross2_stage2.md`), stage 2's gate passed on CAL at 14:09 on 10-07; PLAN 2.3's probe stays failed | the registered cell, claim and alpha are kept; the network is new |
| engine, GPU (Q9) | the fused engine on an RTX PRO 6000 Blackwell | `graph32` on one RunPod H100 80GB HBM3, the GPU of its policy (gross has no fused rung) | GPU term and both GPU decoders on one card |
| dev split | dev reads and their gate | none: the D-bank (selection) and CAL (tau, frontier) took those roles | the test is the only out-of-sample paired evidence |
| test size (3.3) | the power rule on dev paired counts | the rule on CAL's paired counts, then the cap: 300,000 (section 2) | CAL's counts are in-sample for tau |
| quiet CPU host (Q8) | an exclusive Alpine acpu node | an Apple M4 Mac mini, launched as in BB72's stage 3 | no time is converted between hosts |
| Q3's timing before the freeze | on dev | on CAL: its 500 routed shots and its first 1,000 shots (session A) | the registered pre-freeze speed gate, on the split that exists |
| sampling (3.5) | Alpine | the pod's x86-64 CPU (AVX2) after a host check; Derecho CPU if the check fails | Stim's bits depend on the host |
| CPU opponents' test reads | Alpine acpu | Derecho CPU (one node); the Mac mini as fallback (section 3) | accuracy does not depend on the host |

## 1. Cell, claim and alpha

| cell | code, noise | test shots | claim (each comparison at alpha = 0.05 / 3) | speed condition |
|---|---|---|---|---|
| G-4 (PLAN 1) | [[144,12,12]], p = 0.004, 12 rounds (the qnd circuit, X basis, as every gross split) | split test, 300,000 shots | the hybrid's block failure rate is lower than each of sector Relay-BP, BP-OSD ms80-CS60 (ldpc) and cudaq-qec GPU BP-OSD ms80-CS60 | composite batch-1 time on the frozen hosts <= the fastest of these three |

- p = 0.004 is the only gross noise level. p = 0.003 is not a cell: sector Relay-BP fails about 3.4 times per 10,000
  shots there, so a paired test would need about 10^6 shots (PLAN 1).
- Alpha: 0.05 / 3 is G-4's registered Bonferroni share (PLAN 3.2). The claim needs all three comparisons
  (intersection-union, no correction within the cell). That share is already counted in the three registered cells'
  family-wise 0.05, which BB72's stage 3 kept, so this registration spends no new alpha.
- Reported beside, not gated: full-DEM Relay-BP on the first 10,000 test shots (PLAN 3.2), the network alone, the
  routed share, every time.
- Not claimed: best-known accuracy, fastest decoding in general, other codes, p, noise models or hardware.

## 2. Test set

| split | p | chunks x shots | TAG | plain meaning |
|---|---|---|---|---|
| test (code bb144) | 0.004 | 30 x 10,000 = 300,000 | b"ising_compare bb test G-4\n" | one fresh split for G-4 |

Size. PLAN 3.3's rule with `bb.route.power` (the full registered decision: exact McNemar p < 0.05 / 3 and the paired
Newcombe interval at 1 - 0.05 / 3 below 0; 20,000 draws, seed 20261006), floor 100,000, cap 300,000 and a target
power of 1 - 0.10 / 3 = 0.967 per comparison. Script `bb/scripts/stage3_gross_sizes.py`, output
`data/bb/stage3_gross/sizes.json`:

| comparison: counts per 100,000 (both fail / only hybrid / only opponent) | n by the rule | power at 150,000 | power at 300,000 | plain meaning |
|---|---|---|---|---|
| sector Relay-BP, CAL: 192 / 217 / 298 | 150,000 | 0.974 | 1.000 | CAL's point estimate |
| sector Relay-BP, gap one standard error smaller: 192 / 228 / 287 | 270,000 | 0.773 | 0.982 | if CAL's 81-failure gap is 22 too high |
| sector Relay-BP, no shared failure: 0 / 409 / 490 | 250,000 | 0.819 | 0.988 | the worst pairing |
| BP-OSD (ldpc), no shared failure: 0 / 409 / 2,615 | 100,000 | 1.000 | 1.000 | stage-1 dev rate 523 / 20,000 |
| cudaq-qec, no shared failure: 0 / 409 / 2,575 | 100,000 | 1.000 | 1.000 | stage-1 dev rate 515 / 20,000 |

n = 300,000, the cap. The rule on CAL's counts alone gives 150,000, but those counts are in-sample for tau, and this
network was taken after stage 2 had missed its gate once, so CAL's gap may be optimistic. At the cap the power stays
at or above 0.98 in all three Relay-BP rows; the extra 150,000 shots cost about 11 CPU core-hours and 15 pod-minutes.
Nothing is extended after a test shot exists.

Seeds (PLAN 3.5, BB72's construction): seed(TAG, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2
bytes big-endian)) mod 2^62 for chunks j = 0..29. R = the 32 randomness bytes (SHA-256 of the signature) of drand
mainnet (League of Entropy, chain 8990e7a9..., 30-s period; `icmp.t2.fetch_beacon`, at least two endpoints agreeing on
round, signature and previous signature). Round = the first one whose time is at least 600 s after the freeze commit
is public on GitHub (the push time is recorded). Record: `data/bb/stage3_gross/beacon.json`.

Sampler and host: exactly `bb.shots.sample`'s call per chunk (qnd circuit, X basis; one
`compile_detector_sampler(seed=s).sample(10,000, separate_observables=True)`; the same packing), Stim 1.16.0, NumPy
2.2.6, Python 3.12.14 (the pod's gross2 venv), on the pod's x86-64 CPU. Host check first: the CPU flags include avx2
and the Stim module that loads is recorded; CAL chunk 0 (sampled on Alpine, AMD EPYC 7543) and D-bank chunk 10
(sampled on the gross2 stage-1 pod, Intel Xeon Platinum 8470), regenerated there, equal the stored arrays bit for bit. Any
mismatch: Derecho CPU with the same check; never arm64. Each test chunk's row (file, sha256, shots, seed, code, p,
split "test", chunk, host, cpu, stim) goes into `data/bb/shot_manifest.csv`; the rows and the beacon record are
committed and pushed before any read, and `shots.load` checks each file's sha256 on every read.

## 3. Decoders, reads and platforms

| decoder | configuration | test read | host | estimate | plain meaning |
|---|---|---|---|---|---|
| ours (the hybrid's network) | section 0; `bb.route infer --split test` | all 300,000 | H100 pod | about 18 min (3.4 ms per shot) | margins and decisions, blind |
| cudaq-qec GPU BP-OSD ms80-CS60 | cudaq-qec 0.8.0 nv-qldpc-decoder: min-sum scale 1.0, 80 iterations, OSD-CS (osd_method 3) order 60, X sector; `bb.gpu_bposd read`, calls of 1,000 shots | all 300,000, one whole read | H100 pod | 5-15 min (1,202 shots per s on the A100 at batch 256) | |
| sector Relay-BP | section 0's configuration | all 300,000 | Derecho CPU | 11.8 core-h | gating opponent; also the fallback's predictions |
| BP-OSD ms80-CS60 | ldpc 2.4.1 `BpOsdDecoder`: min-sum scale 1.0, 80 iterations, parallel schedule, OSD-CS order 60, X sector (`bb.opp bposd`) | all 300,000 | Derecho CPU | 10.9 core-h | |
| full-DEM Relay-BP | section 0's parameters on both detector sectors (`bb.opp relay_full`) | the first 10,000 (chunk 0) | Derecho CPU | 12.0 core-h | reported beside |

Core-hours are on the loaded-Alpine basis of the stage-1 reads (DESIGN 6: 143.2, 130.5 and 4,310 ms per shot; CAL's
Relay-BP read 141.4 ms): 34.6 core-h in all.

Where the CPU reads run:

| host | processes | speed per process | wall time for the 34.6 core-h | plain meaning |
|---|---|---|---|---|
| Derecho, one CPU node (2 x AMD EPYC 7763, 128 cores), PBS develop queue as in stage 1, account <account>, the qldpc venv with relay-bp 0.2.2 and ldpc 2.4.1 checked before the job | 128 (`bb.opp --procs 128`; `--slice 10` for full-DEM) | about loaded Alpine (same EPYC generation) | 30-40 min with the slowest slices; about 1 h with copying, setup and queue (stage 1's develop jobs queued for seconds) | faster, and the Mac mini stays idle |
| the Mac mini, Apple M4 (4 performance + 6 efficiency cores) | 6 (about 4 on performance cores, 2 on efficiency cores) | 1.7-2.4 x loaded Alpine on a performance core (BB72: Relay-BP 3.82 ms on loaded Alpine and 1.60 ms on the M4; BP-OSD 2.36 and 1.27 ms) | about 3.5 h (3-5 h) | no timing session can run meanwhile |

Derecho is about three times faster for these counts and leaves the Mac mini free for session B, so the reads go
there. If its job has not started 30 minutes after submission, the Mac mini reads instead, after session B.

## 4. Timing (PLAN 2.7's protocol on the frozen hosts; no time is converted)

### 4.1 GPU side (the H100 pod)
- At the start of each GPU session: `nvidia-smi -q` (product name, UUID, MIG disabled, full memory visible, compute
  mode), the pod id and its GPU count (one).
- Acceptance per timed process, ours and cudaq-qec's alike (`bb.timing.gpu_watch`, BB72 4.1): just before it starts,
  two device-wide samples show utilization 0 % and memory.used below 1,024 MiB; during its window exactly one compute
  app and every 100-ms SM clock sample at least 97 % of the maximum. A rejected process is replaced at most twice. No
  read or sampling runs on the pod during a timed process.
- Before the freeze: the GPU term, `bb.latency` in five fresh processes, each the mean of 5 x 100 graph replays after
  200 warm-up replays; counted: the slowest. The handoff from the same processes; counted: the larger of 1.1 us and
  the slowest. cudaq-qec's batch-1 call (`bb.gpu_bposd time`, 300 calls on the first CAL shots) in three fresh
  processes; counted: the fastest.
- After sampling: cudaq-qec's batch-1 call on the first test shots, three fresh processes; counted: the fastest.

### 4.2 CPU side (the idle Mac mini)
- Idle: nothing else runs on the Mac mini during a session (no study-4, BB72 or read process; `ps` checked before each
  session); load average and the busiest processes are recorded before and after each timed process. A busy Mac mini
  means the timing waits.
- Launch as BB72's stage 3: one single-threaded process at a time, from the same non-interactive shell, default QoS,
  no `taskpolicy` clamp, thread counts 1. Each process first times a fixed 0.1-s integer loop, recorded with its row,
  so a process placed on an efficiency core shows. Rows name "one M4 core as scheduled by macOS" and the versions
  (Python 3.12.14, relay-bp 0.2.2, ldpc 2.4.1 there, as in BB72's stage 3).
- Warm-up: each session starts with one untimed pass of every decoder it times, over the first 100 shots of the set
  that decoder will time (BB72 used 1,000; routed gross shots take about 1 s each, so 1,000 would add about 17 min).
- Session A, before the freeze (about 40 min, at most about 1 h):
  1. CPU term on CAL: sector Relay-BP on the 500 CAL shots that tau routes (the frozen engine's H100 read; local CAL
     copies, sha256 equal to the manifest's), three fresh processes; counted: the slowest. CPU term = 0.005 x the mean
     time per routed shot.
  2. Sector Relay-BP and BP-OSD on the first 1,000 CAL shots, three fresh processes each; counted: the fastest.
  3. Speed gate (Q3): GPU term + handoff + CPU term <= the fastest of (2) and cudaq-qec on CAL (4.1). A fail stops
     G-4 before the freeze; nothing is sampled.
- Session B, after the network's test read (about 65 min, at most about 2 h):
  1. CPU term on the test: sector Relay-BP on the first 1,000 routed test shots in split order (all if fewer), three
     fresh processes; counted: the slowest. CPU term = the test's routed share x the mean time per timed routed shot.
  2. Sector Relay-BP and BP-OSD on the first 1,000 test shots, three fresh processes each; counted: the fastest.

### 4.3 Speed condition
- Composite = GPU term (4.1) + handoff + CPU term (session B). Condition: at most the fastest of cudaq-qec on the
  H100 (test shots), sector Relay-BP and BP-OSD on the M4 (session B). It is a measured inequality; no statistical
  test applies to it.
- Every row names its host and CPU or GPU model, its session (date and host), its process and its binding (the policy
  file's sha256 for CPU-term rows, the engine-signature digest for GPU-term and handoff rows).
- Expected values, estimated from stage 2's numbers with BB72's measured host ratios (the M4 1.9-2.4 x faster than
  loaded Alpine for Relay-BP; the H100 1.1-1.35 x faster than the Derecho A100 for cudaq-qec). Session A measures both
  sides before the freeze.

| quantity | stage 2 (provisional) | expected on the frozen hosts | plain meaning |
|---|---|---|---|
| GPU term + handoff | 3.43 ms | 3.43 ms | same GPU model |
| CPU term | 0.5 % x 2.34 s = 11.7 ms | 0.5 % x 0.98-1.23 s = 4.9-6.2 ms | Relay-BP's routed shots on loaded Alpine, then on one M4 core |
| composite | 15.12 ms | 8.3-9.6 ms | |
| bar | 15.8 ms (cudaq-qec, Derecho A100) | 11.7-14.4 ms (cudaq-qec, H100); Relay-BP 59-74 ms and BP-OSD 69-77 ms on the M4 | the fastest gating opponent |
| margin | 0.68 ms | about 2-6 ms | the 0.68 ms came from the loaded CPU basis |

## 5. Statistics: one analysis, run once

- The hybrid's prediction per test shot: the network's decision if its margin >= tau, otherwise sector Relay-BP's
  test-read prediction (`bb.route apply`'s rule). A block failure: any of the 12 logical observables wrong.
- Per comparison, paired on the same 300,000 shots: the exact two-sided McNemar test (binomial on the discordant
  pairs) at p < 0.05 / 3 and the paired Newcombe interval (method 10, `qldpc_neural/bench/paired.py`) at level
  1 - 0.05 / 3 for the failure-rate difference (hybrid minus opponent) entirely below 0 give "better"
  (`bb.route.paired` with ALPHA_TEST).
- The claim holds iff all three comparisons are "better" and the speed condition holds.
- Reported, not gated: each decoder's failures with a Wilson 95 % interval; the network alone; full-DEM Relay-BP
  against the hybrid on the first 10,000 test shots (the same paired statistics, for information); the routed share;
  session A's gate; all times.
- The command, committed and pushed in the freeze commit: `python -m bb.stage3_gross analyze --out
  data/bb/stage3_gross`. It runs once, on a Linux workstation, after every read and session B; `results.json`,
  `RESULTS_stage3_gross.md` and the figure are committed and pushed as produced. No failure count of any decoder on
  the test is printed or looked at before this run. If the analysis crashes, only the defect is fixed (no rule,
  threshold or input changes), it reruns on the same inputs, and the fix is disclosed.
- Published (the hub, branch `ising-bb-gross2`, then a `gross/` folder in the release repository as for `bb72/`, which
  the study lead mirrors to GitHub): PLAN.md v3, both gross2 plans, this plan, the freeze commit's code, the
  checkpoint (sha256 above), the policy and gate records, the CAL reads behind `sizes.json` (network margins, sector
  Relay-BP predictions), the beacon record, the test manifest, every decoder's per-shot test predictions and failures
  (packed, with sha256), the network's test margins, every timing row and the analysis outputs.

## 6. Order of steps

| # | step | where | needs |
|---|---|---|---|
| 1 | this plan, the size script, `sizes.json` and the policy and gate records committed and pushed | Linux workstation | |
| 2 | code, one commit, pushed: `origin/ising-bb` merged into `ising-bb-gross2` (BB72's stage-3 code: the test split in `shots`, `timing` on macOS, `gpu_watch`, `--gpu`), with `engine.py` and `net.py` unchanged (their hashes are in the engine signature); `shots`: TAG and beacon file for bb144; `bb/stage3_gross.py` for G-4 (host check on bb144, CPU term, session A's gate, analyze), BB72's `bb/stage3.py` untouched; a Derecho CPU job script for `bb.opp`; tested on stand-in data (CAL chunks as "test") | Linux workstation | 1 |
| 3 | the study lead creates the H100 pod (H100 80GB HBM3, one GPU) when the Mac mini is free | | 2 |
| 4 | M4 session A (4.2) | Mac mini | 2; the Mac mini idle |
| 5 | pod: environment (gross2's lock: torch 2.11.0+cu126, Stim 1.16.0, NumPy 2.2.6, Python 3.12.14; the cudaq-qec 0.8.0 venv as on BB72's pod), a clean clone of step 2's commit, the checkpoint copied and its sha256 checked; GPU record; host check; engine check: the 10 CAL chunks re-read through the frozen engine, every decision equal to stage 2's H100 read and the routed set at tau equal (largest margin difference recorded); GPU term, handoff and cudaq-qec's CAL rows (4.1) | pod | 2, 3 |
| 6 | freeze commit, pushed: the code and `data/bb/stage3_gross/cell.json` with every frozen number (network sha256, engine signature, tau, q, the fallback's and opponents' configurations and versions, hosts, n = 300,000, session A's and the GPU results, the gate's verdict) and the analysis command; then the study lead's check that it is public on GitHub | Linux workstation | 4, 5 |
| 7 | beacon (the first drand round at least 600 s after the freeze commit is public on GitHub); the 30 chunks sampled on the pod; manifest rows and beacon record committed and pushed | pod, Linux workstation | 6 |
| 8 | pod: the network's test read; cudaq-qec's batch-1 timing on the first test shots (quiet pod); cudaq-qec's test read; copy-back with sha256; then the study lead stops the pod | pod | 7 |
| 9 | Derecho: test chunks copied (sha256 checked); one CPU job: `bb.opp --dec relay,bposd` on chunks 0-29 and `relay_full` on chunk 0; predictions copied back (sha256) | Derecho | 7 |
| 10 | M4 session B (4.2) | Mac mini | 8's network read |
| 11 | the one analysis (section 5); outputs pushed | Linux workstation | 8, 9, 10 |
| 12 | release-repo rows for gross (data, code, README, figure), report | Linux workstation | 11 |

Stops. If the engine check fails, nothing is sampled and no test runs on that pod; the study lead decides on a new
pod. If the host check fails, sampling moves to Derecho CPU with the same check. If session A's gate fails, G-4 stops
before the freeze and the report says why. If the Mac mini is busy, the timing waits; nothing is timed on a loaded
host. If the Derecho job has not started within 30 minutes, the reads move to the Mac mini after session B. On any
stop the pod is stopped at once and the reason recorded, so billing ends on failure as on success.

## 7. Hours and dollars

| item | amount | cost | plain meaning |
|---|---|---|---|
| H100 pod | about 2.6 h: setup 0.4, CAL engine check and GPU numbers 0.5, freeze, GitHub mirror and beacon wait 0.5, sampling 0.1, reads and cudaq-qec timing 0.6, copy-back 0.1, slack 0.4 | about $9 at $3.49 per hour; hard ceiling 6 h = $20.94 | stopped as soon as step 8 ends or at any stop |
| Derecho CPU | 34.6 core-h of reads on one 128-core node (about 0.5 node-hours) | allocation <account>, no dollars | the CPU opponents' test reads |
| Mac mini timing | sessions A and B, about 1.8 h of one core (at most 3 h) | $0 | quiet host |
| Mac mini reads (fallback) | about 3.5 h (3-5 h) with 6 processes | $0 | only if Derecho does not start |
| working time | about 6 h | | code 2 h, runs 3 h, analysis and publication 1 h |

## 8. Timeline (PDT; T0 = the Mac mini free after d = 17 and BB72, expected 17:00-18:00 on 10-07; times for T0 = 17:30)

| time | what |
|---|---|
| 14:50 | this plan pushed (1) |
| 15:00-17:30 | code commit (2) |
| 17:30-18:10 | session A on the Mac mini (4); the pod created (3) |
| 17:35-18:25 | pod setup, host check, CAL engine check, GPU numbers (5) |
| 18:30 | freeze commit pushed and mirrored to GitHub (6) |
| 18:45-19:00 | beacon, sampling, manifest pushed (7) |
| 19:00-19:45 | pod reads and cudaq-qec timing; pod stopped (8; about 2.3 pod-hours) |
| 19:05-20:15 | Derecho CPU reads (9) |
| 19:25-20:35 | session B on the Mac mini (10) |
| 20:45-21:00 | the analysis, outputs pushed (11) |
| 21:00-22:00 | release rows, report (12) |

A later T0 shifts every row by the same amount. If Derecho cannot run, the Mac mini reads from about 20:35 until
about 23:30-01:30 and the analysis follows by about 02:00, still well before the morning.

## 9. Freeze record (added with the freeze commit; no rule above changes)

Pre-freeze numbers and records are in `data/bb/stage3_gross/`: `cell.json` (the frozen parameters, `prefreeze`, the
analysis command), `rows_gpu.csv` and `rows_cpu.csv` (the timing rows, to which the post-sampling rows are appended),
`pod/` and `mac/` (scripts, environments, logs). Code: 8bd5b815 (step 2), unchanged by this commit.
- Pod gpu-pod, created 14:55 PDT: one NVIDIA H100 80GB HBM3 (MIG disabled, the policy's GPU model), Intel Xeon
  Platinum 8480+. `<pod-home>/g2venv` is gross2's lock exactly (torch 2.11.0+cu126); `<pod-home>/cqvenv` holds cudaq-qec 0.8.0
  with BB72's pod packages (`pod/freeze_cqvenv.txt`).
- Host check passed on the pod: cal chunk 0 (Alpine) and D-bank chunk 10 (the gross2 stage-1 pod) regenerated bit for bit;
  avx2 present; Stim 1.16.0 loads its SSE2 module there by its own dispatch, as on BB72's pod.
- Engine check passed: the frozen engine re-read all 100,000 cal shots with every decision equal and every margin
  bitwise equal to stage 2's H100 read; tau routes the same 500 shots.
- GPU side (session h100-gpu-pod-pre, every process accepted at clock ratio 1.0 with one compute app): GPU term
  3.4296 ms (the slowest of 5; the others 3.3555 to 3.4081 ms), handoff 0.0226 ms (the slowest). cudaq-qec's batch-1
  call on the first 300 cal shots: 16.294 ms (the fastest of 3: 16.29, 17.29, 18.02 ms). Disclosed: 20 earlier
  cudaq-qec processes stopped at import before any timing, because the relay testdata module that builds the X-sector
  DEM and its dependencies were missing on the pod; they were added at the versions of BB72's pod and the runs were
  repeated. Those attempts left only idle-check rows.
- Session A (M4 session gA, idle Mac mini; `mac/load_A.log`): the CPU term on cal's 500 routed shots 5.031 ms (the
  slowest of 5.007, 5.031, 5.019 ms; 1.006 s per routed shot), sector Relay-BP 63.07 ms and BP-OSD 76.95 ms per shot on
  the first 1,000 cal shots (the fastest of 3 each).
- Speed gate (Q3): composite 3.4296 + 0.0226 + 5.031 = 8.483 ms against the fastest gating opponent, cudaq-qec at
  16.294 ms: pass, so G-4 is tested (`cell.json` status "tested").
- The beacon round is the first at least 600 s after max(this commit's time, the time this commit is seen public on
  GitHub).
