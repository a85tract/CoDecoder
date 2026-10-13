# BB72 stage 3: the registered test, run on 2026-10-07 (plan)

2026-10-07. Status: plan for the study lead's check. It becomes binding with the freeze commit (step 6
below), which adds the code and the measured pre-freeze numbers (`data/bb/stage3/cells.json`) and changes no rule of
this plan. Like `bb/PLAN.md` (version 3, file at 01755dde) this is a registration document in the repository (Q7). It
also registers Amendment F (`bb/AMENDMENT_F_draft.md`, committed verbatim with this revision) as cells BB72-2F and
BB72-3F, with the changes listed in 0.1. All their choices were made on cal; no test shot exists yet.

Revision 1 (11:30 PDT) answers a quick review of df742488 (REVISE, no critical; [review text
not included]): BB72-2F's screen no longer scales CPU times between decoders or hosts and uses the H100's
own routing of the cal split (findings 1, 2); the changes to Amendment F are registered in 0.1 (3); the M4 core claim
is qualified and all timed processes get the same QoS (4); whole-card checks on the H100 (5); a preflight of the GPU
acceptance rule (6); routing edge cases (7); sampling details (8); the size calculation is committed (9); paired
Newcombe named, the 0.083 bound kept (10); what is published (11); the pod stops on failure too (12).

User rules for this stage (2026-10-07): an unseen fresh test set; the method frozen before any test read; equal
conditions; one analysis, its command committed and pushed before the reads, run once, outputs pushed; public
reproducibility; no new guards, locks or tickets. Budget for BB72: at most $40 (H100 at $3.49 per hour).

## 0. Disclosed changes against PLAN.md version 3

Alpine is down until 2026-10-08 05:30 PDT, so everything that needed Alpine moves today.

| item | PLAN v3 | now | plain meaning, effect |
|---|---|---|---|
| quiet CPU host (Q8, 2.7) | one exclusive 64-core Alpine acpu node (AMD EPYC) | an Apple M4 Mac mini (4 performance + 6 efficiency cores, 16 GB), one single-threaded process at a time, otherwise idle, one untimed warm-up pass per decoder first (user-approved) | DESIGN 5.4 still holds: every CPU decoder, opponents and fallback alike, is timed on the same machine in one session with the same launch settings, so no conversion is made and "Apple M4" is named in every row. macOS cannot pin a process to a core type, so the claim is "one M4 core as scheduled by macOS": every timed process is started the same way (from the same non-interactive shell, default QoS, no `taskpolicy` clamp, thread counts 1), and each first times a fixed 0.1-s integer loop, recorded with its row, so a process placed on an efficiency core (about 2 x slower) shows. Load average and the busiest other processes are recorded before and after each timed process |
| GPU (Q9, 2.8) | a full RTX PRO 6000 Blackwell on Alpine | one RunPod H100 80GB pod (created by the study lead) for our engine's test reads, its GPU term and handoff, cudaq-qec's test reads and batch-1 timing | Both GPU decoders run on the same card under the same acceptance rule (4.1). Engine code, torch 2.11.0+cu128 and Triton 3.6.0 stay as frozen; the engine signature may differ from the policies' only in the field "gpu" (recorded in every output). The registered engine validation (2.4, fused rung) is re-run on the H100 before any test read. Stage 2's cal and dev reads (RTX card) are not redone, which PLAN 2.8 would ask for after a model switch: the registered cells' policies route nothing (q = 0), and the F cells apply their frozen tau to H100 margins (H100-vs-RTX agreement on all cal chunks is reported) |
| test-set sampling (3.5) | on Alpine | on the pod's x86-64 CPU, in the dev/cal sampler environment; Derecho if the host check fails; never arm64 | Stim's output depends on the instruction set. Host check first (section 2): reference chunks regenerated on the pod must equal the shot manifest's sha256 bit for bit |
| CPU opponents' test reads | Alpine acpu | the Mac mini, at most 6 processes, at least 6 GB free (Python 3.12.14, relay-bp 0.2.2, ldpc 2.4.1, Stim 1.16.0 there); Derecho CPU only if the Mac mini cannot finish today | accuracy does not depend on the host; versions recorded with each read |
| test sizes (3.3) | `bb.route size` on the dev apply records | the floors, from the published dev counts (section 2) | the apply records are on Alpine; the worst case of the registered rule already gives the floor |
| cells | BB72-3, BB72-2 (G-4 ended at its probe gate, PLAN 2.3) | plus BB72-2F and BB72-3F (Amendment F, with 0.1) | section 1 |

### 0.1 Changes to Amendment F's draft, registered here

1. BB72-2F's speed screen before the freeze. The draft compared a provisional composite with LIMIT_MS = 0.63 ms (a
   stage-1 cudaq-qec time on a Derecho A100, against CPU times on the Alpine cal-read basis) and asked for a
   re-measurement on an exclusive host. No measured scaling links routed full-DEM Relay-BP on another CPU to that basis
   (review finding 1), so the screen now compares like with like on the frozen hosts: the composite of r12's routed cal
   workload on the H100 (frozen tau applied to r12's H100 cal margins; the routed fraction is the H100's, not 0.25 %)
   against the fastest gating opponent measured before the freeze on the same hosts (4.2, session A). No time is
   converted anywhere in this stage. The policy itself cannot change: r12's hybrid cal failures rise with q (106, 123,
   144, 156, 161, 163 for q = 0.25, 0.5, 1, 2, 3, 5 %; r11 119 to 163), so q = 0.25 % and replica r12 stay the rule's
   picks under any bar that admits q = 0.25 %. A failed screen drops BB72-2F before the freeze.
2. BB72-2F's dev-side CPU term (the draft's Q8 row on about 750 routed dev shots) is not measured: the dev shots and
   r12's dev read are on Alpine. The cal-side screen of item 1 (same hosts, same protocol) takes its place before the
   freeze, and the test-side term (4.2, session B) decides the claim.
3. BB72-3F's timing: the reference is full-DEM Relay-BP's own mean time per shot on the first 2,000 test shots of split
   BB72-3 (the draft: 10,000), three fresh processes, the fastest counted; the CPU term is timed on the first 1,000
   routed test shots (all of them if fewer; PLAN 2.7's rule would take up to 5,000). This keeps session B near 25
   minutes; the cal ratio of the two sides is 140 / 8.35 = 17.
4. Unchanged from the draft: replicas, q, tau, fallback configuration, claims, alpha_F = 0.05 / 3 per F cell, test
   sizes, and the 0.083 bound over all claims of the study.

## 1. Cells

| cell | code, p | hybrid (frozen) | test shots | claim (each comparison at alpha = 0.05 / 3) | speed condition |
|---|---|---|---|---|---|
| BB72-3 (registered) | [[72,12,6]], 0.003, 6 rounds | r11 through the fused engine; policy q = 0: nothing routed, so the network alone | split BB72-3, n = 200,000 | lower block failure rate than each of sector Relay-BP, BP-OSD ms80-CS60 (ldpc), cudaq-qec GPU BP-OSD ms80-CS60 | composite <= the fastest gating opponent |
| BB72-2 (registered) | [[72,12,6]], 0.002 | r12, q = 0 | split BB72-2, n = 300,000 | as BB72-3 | as BB72-3 |
| BB72-2F (Amendment F) | [[72,12,6]], 0.002 | r12; shots with margin < tau = 2.979396343231201 (q = 0.25 % on cal) go to full-DEM Relay-BP (`bb.opp relay_full`) | split BB72-2 (the same 300,000 shots) | lower than full-DEM Relay-BP and than each of the three gating opponents (four comparisons) | composite <= the fastest gating opponent; exists only if its pre-freeze screen passes (0.1 item 1) |
| BB72-3F (Amendment F, weaker speed claim) | [[72,12,6]], 0.003 | r12; margin < tau = 3.6329734325408936 (q = 2 % on cal) to full-DEM Relay-BP | split BB72-3 (the same 200,000 shots) | lower than full-DEM Relay-BP | composite <= full-DEM Relay-BP's own quiet-host mean time per shot; the fastest gating opponent's time is reported beside it, not gated |

- Cal evidence (`data/bb/fullroute/bb72_cal.json`, 7c324df2; 200,000 cal shots per p). BB72-2F: hybrid 106 failures
  against full-DEM Relay-BP's 169 (only hybrid 10, only full-DEM 73, exact McNemar p 6e-13). BB72-3F: hybrid 900
  against 1,117 (80 / 297, p 2e-30). Replica and q follow Amendment F's rules (fewest hybrid cal failures among
  admissible q; ties to the smaller q, then to the smaller composite, then replica 1).
- Each cell reads only its own replica: r11 on split BB72-3; r12 on splits BB72-2 and BB72-3.
- Routing on the test is never truncated: every shot with H100 margin < tau goes to full-DEM Relay-BP for accuracy, and
  every CPU term uses the test's own routed fraction. Zero routed shots: CPU term 0 and the hybrid equals the network.
- Multiplicity: the three registered cells keep their family-wise 0.05 (0.05 / 3 each, G-4 included though it ended).
  Each F cell also takes 0.05 / 3, so the union bound over all claims of the study is 0.05 x 5 / 3 = 0.083, stated in
  the report.
- Not claimed: best-known accuracy, fastest decoding in general, other codes, p, noise models or hardware.

## 2. Test sets

| split | p | chunks x shots | seeds | plain meaning |
|---|---|---|---|---|
| BB72-3 | 0.003 | 20 x 10,000 = 200,000 | TAG = b"ising_compare bb test BB72-3\n" | cells BB72-3 and BB72-3F |
| BB72-2 | 0.002 | 30 x 10,000 = 300,000 | TAG = b"ising_compare bb test BB72-2\n" | cells BB72-2 and BB72-2F |

- Seeds (PLAN 3.5): seed(TAG, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian))
  mod 2^62. R = the 32 randomness bytes (SHA-256 of the signature) of drand mainnet (League of Entropy, chain
  8990e7a9..., 30-s period; the chain and endpoints of `icmp.t2`), round = the first round whose time is at least 600 s
  after the freeze commit's committer time, fetched from at least two endpoints that agree on round, signature and
  previous signature (`icmp.t2.fetch_beacon`). Record: `data/bb/stage3/beacon.json`.
- Sampler: exactly `bb.shots.sample`'s call per chunk (qnd circuit, X basis; one
  `compile_detector_sampler(seed=s).sample(10,000, separate_observables=True)`; the same packing), Stim 1.16.0, NumPy
  2.2.6, Python 3.12.14, on the pod. Host check before sampling: the CPU has AVX2 and Stim loads its AVX2 module
  (recorded), and dev chunk 0 and the monitor chunk at both p, regenerated there, all equal the manifest's sha256;
  any mismatch means no sampling on that host. Each test chunk's row (file, sha256, shots, seed, code, p, split
  "test", chunk, host, cpu, stim) goes into `data/bb/shot_manifest.csv`; the rows and the beacon record are committed
  and pushed before any read. `shots.load` checks each file's sha256 on every read.
- Sizes. PLAN 3.3 takes, per gating opponent, the smallest n from the floor (dev size) at which the decision passes
  with probability >= 1 - 0.10 / 3, simulated from the paired dev counts. Those paired counts are on Alpine, but the
  marginal dev counts are recorded: BB72-3 r11 1,327 of 200,000 against 2,747 / 3,308 / 3,366; BB72-2
  r12 269 of 300,000 against 773 / 911 / 895. The worst case for the decision is no shared failure (largest discordant
  total, no pairing credit in the Newcombe interval). `bb.route.power`, the complete registered decision, on these
  worst-case counts gives power 1.0000 at the floor for all six comparisons (`data/bb/stage3/sizes_worstcase.json`,
  `bb/scripts/stage3_sizes.py`, run on the Mac mini; `bb.route` as at 01755dde), so n = 200,000 and 300,000 whatever the
  pairing. F cells: `bb72_cal_sizes.json` (7c324df2) gives the floors, 300,000 (BB72-2F) and 200,000 (BB72-3F).
  Nothing is extended after a test shot exists.

## 3. Decoders and platforms

| decoder | configuration | reads (test) | host | estimate |
|---|---|---|---|---|
| ours, fused engine | r11 (sha256 58e60897...), r12 (f2821c32...); the policies' engine signature except "gpu"; `bb.route infer --split test` | r11 on BB72-3; r12 on BB72-3 and BB72-2 (700,000 shots) | H100 pod | about 2 min |
| cudaq-qec GPU BP-OSD ms80-CS60 | cudaq-qec 0.8.0, `bb.gpu_bposd` | both splits in full | H100 pod | about 15 min |
| sector Relay-BP | relay-bp 0.2.2, the authors' tuned parameters, `bb.opp relay` | both splits in full | Mac mini | about 0.25 core-h |
| BP-OSD ms80-CS60 | `bb.opp bposd` (ldpc 2.4.1 on the Mac mini) | both splits in full | Mac mini | about 0.4 core-h |
| full-DEM Relay-BP | `bb.opp relay_full` (all detectors) | both splits in full (500,000 shots): BB72-2F and BB72-3F need them; the registered reference is reported on the first 200,000 of each split as PLAN 3.2 says | Mac mini | about 8 core-h (74 / 46 ms per shot at p 0.003 / 0.002 on the loaded Mac mini) |

CPU reads: 6 processes, about 85 min of wall time (up to 2 h with the efficiency cores).

## 4. Timing

### 4.1 GPU side (H100, PLAN 2.7)
- Allocation, recorded at the start of each GPU session: `nvidia-smi -q` (product name, UUID, MIG mode disabled, the
  full memory visible, compute mode) and the pod's id and GPU count (one whole H100; RunPod does not share a GPU
  between pods).
- Acceptance per timed process, ours and cudaq-qec's alike (`bb.timing.gpu_watch`): just before it starts, two
  device-wide samples show utilization.gpu 0 % and memory.used below 1,024 MiB (no other context on the card); during
  its window exactly one compute app is listed (the pod's PID namespace hides host PIDs; the gross study's pod used the same
  count) and every 100-ms SM clock sample is at least 97 % of the maximum; the device-wide utilization and memory.used
  samples of the window are recorded with the row. A rejected process is replaced at most twice. No CPU read runs on
  the pod during timed processes.
- Preflight, before the freeze: one `bb.latency` process and one cudaq-qec timing process with the window logged
  (start, end, number of samples inside it, smallest clock ratio). If the rule rejects a process on an idle card only
  because of startup samples or because too few samples fall inside a short window, the freeze commit fixes the window
  for both arms alike (from the first warm-up call to the last timed one, at least 3 samples, more warm-up if needed)
  before any number below is taken.
- Before the freeze: `bb.latency`, five fresh processes per network (r11, r12), each the mean of 5 x 100 graph replays
  after 200 warm-up replays; counted: the slowest. Handoff measured in the same processes; counted: the larger of
  1.1 us and the slowest. cudaq-qec's batch-1 call (`bb.gpu_bposd time`), 300 calls on the first dev shots of each p
  (dev chunk 0), three fresh processes per p; counted: the fastest (the bar of 0.1 item 1).
- After sampling: cudaq-qec's batch-1 call on the first test shots of each split, three fresh processes per split;
  counted: the fastest (the bar of the claims).

### 4.2 CPU side (the Mac mini; Q8 and stage 3)
Two quiet sessions. One single-threaded process at a time, launched as in section 0; each session starts with one
untimed pass of every decoder it times over 1,000 shots.

Session A, before the freeze (about 15 min):
1. Q8 for BB72-3 and BB72-2: sector Relay-BP and BP-OSD on the first 10,000 dev shots (dev chunk 0 at each p,
   regenerated on the pod and checked against the manifest), three fresh processes each; counted: the fastest. The
   registered policies route nothing, so their CPU term is 0. Stage-2 speed verdict 2.6 (ii) per cell: composite
   0.086 ms (the RTX session on Alpine, as registered) against the fastest of these and cudaq-qec on the RTX card
   (2.362 / 1.075 ms); the same with the H100's numbers is reported.
2. BB72-2F's screen (0.1 item 1): full-DEM Relay-BP on every cal shot of p 0.002 that r12's H100 read routes at tau
   (about 500; the first 5,000 in split order if more), from the local cal copies (sha256 = Alpine's), three fresh
   processes; counted: the slowest. Pass iff GPU term (H100) + handoff + (H100 routed fraction of the cal split) x
   (mean time per routed shot) <= the fastest of sector Relay-BP and BP-OSD (item 1, p 0.002) and cudaq-qec (4.1, dev,
   p 0.002).

Session B, after the networks' test reads (about 25 min):
1. Sector Relay-BP and BP-OSD on the first 10,000 test shots of each split, three fresh processes each; counted: the
   fastest.
2. BB72-2F's CPU term: full-DEM Relay-BP on the routed test shots of split BB72-2 (H100 margin < tau, in split order;
   about 750; the first 5,000 if more), three fresh processes; counted: the slowest; CPU term = the test's routed
   fraction x mean time per timed routed shot.
3. BB72-3F's CPU term: the same on the first 1,000 routed test shots of split BB72-3 (about 4,000 are routed; all if
   fewer than 1,000). BB72-3F's bar: full-DEM Relay-BP on the first 2,000 test shots of split BB72-3, three fresh
   processes; counted: the fastest (0.1 item 3).

### 4.3 Basis and gate
- Every speed condition compares times on the same hosts: GPU parts on the H100, CPU parts on the M4, with no
  conversion (DESIGN 5.4).
- Composite = GPU term + CPU term + handoff. Gate: at most the fastest gating opponent (cudaq-qec on the H100, sector
  Relay-BP and BP-OSD on the M4); BB72-3F: at most full-DEM Relay-BP's own time. A measured inequality, no test.
- Every row names its host and CPU or GPU model, its session (a label with date and host, in place of a Slurm job),
  its process and its binding (policy file sha256 or engine-signature digest).

## 5. Statistics: one analysis, run once

- Per comparison, paired on the same test shots: exact two-sided McNemar test (binomial on the discordant pairs),
  p < 0.05 / 3, and the paired Newcombe interval (method 10, `qldpc_neural/bench/paired.py`) at level 1 - 0.05 / 3 of
  the failure-rate difference entirely below 0 gives "better" (`bb.route.paired` with ALPHA_TEST).
- A cell's claim holds iff all its comparisons are "better" (intersection-union, no correction within a cell) and its
  speed condition holds.
- Reported, not gated: each decoder's failures with a Wilson 95 % interval; full-DEM Relay-BP beside the registered
  cells (first 200,000 shots of each split, and all 300,000 of split BB72-2); routed fractions; the H100-vs-RTX
  agreement of r11 and r12 on cal; the stage-2 speed verdicts; all times.
- The command, committed and pushed in the freeze commit: `python -m bb.stage3 analyze --out data/bb/stage3`. It runs
  once, after every read has finished; `results.json`, `RESULTS_stage3.md` (the tables) and the figure are committed
  and pushed as produced. No failure count of any decoder on the test is printed or looked at before this run. If the
  analysis itself crashes, only the defect is fixed (no rule, threshold or input changes), it reruns on the same
  inputs, and the fix is disclosed.
- Published (hub now, release repository next): PLAN.md v3 (01755dde), the Amendment F draft and this plan as
  committed, the freeze commit's code, both checkpoints (sha256 above), the beacon record, the test manifest, every
  decoder's per-shot test predictions and failures (packed, with sha256), the networks' margins, every timing row,
  and the analysis outputs.

## 6. Order of steps

| # | step | where | needs |
|---|---|---|---|
| 1 | this plan committed and pushed | Linux workstation | |
| 2 | study lead: review; creates the H100 pod | | 1 |
| 3 | code, one commit, tested on stand-in data (cal chunks): `shots` split "test" with section 2's seeds; `bb/stage3.py` (beacon, sample with the host check, analyze); `--gpu` in `route infer` and `timing gate` (the signature may differ from the policy's in "gpu" only, which is recorded); `timing` on macOS (CPU name via sysctl, session label, the 0.1-s loop) and 4.1's acceptance in the pod; `route infer` prints no failure count on the test split; F-cell parameters in `data/bb/stage3/cells.json` | Linux workstation, Mac mini | 1 |
| 4 | pod: environment (torch 2.11.0+cu128, Triton 3.6.0, cudaq-qec 0.8.0, Stim 1.16.0, NumPy 2.2.6, Python 3.12.14), clean clone, DEPLOYED files as `deploy.sh` writes them; allocation record; host check; engine validation (2.4, fused rung) on the H100; r11 and r12 on all cal chunks (agreement with the RTX reads; r12's routed cal set at each tau); GPU preflight; GPU-term session and cudaq-qec's dev batch-1 rows (4.1) | pod | 2, 3 |
| 5 | M4 session A (4.2): Q8 and BB72-2F's screen | Mac mini | 3; dev chunk 0, r12's H100 cal margins and the GPU numbers from 4 |
| 6 | freeze commit, pushed: code, `cells.json` with every frozen number (taus, sizes, engine signature with "gpu", versions, hosts, session-A and GPU results, BB72-2F's verdict), the analysis command | Linux workstation | 4, 5 |
| 7 | beacon (first drand round >= commit time + 600 s); sample both splits on the pod; manifest rows and beacon record committed and pushed | pod, Linux workstation | 6 |
| 8 | pod: the networks' test reads; cudaq-qec batch-1 timing on a quiet pod; cudaq-qec test reads. Then the study lead stops the pod | pod | 7 |
| 9 | M4 session B (4.2) | Mac mini | 8's network reads |
| 10 | CPU test reads on the Mac mini: sector Relay-BP, BP-OSD, full-DEM Relay-BP on both splits | Mac mini | 9 |
| 11 | the one analysis (section 5); outputs pushed | Linux workstation | 8, 10 |
| 12 | release-repo rows for BB72 (data, code, README, figure), report | Linux workstation | 11 |

Stops: the engine fails its validation on the H100 -> no GPU platform change, the test waits for Alpine's RTX card
(10-08), nothing is sampled. The pod fails the host check -> sample on Derecho with the same check. BB72-2F fails its
screen -> dropped before the freeze. Mac mini busy -> the timing waits; nothing is timed on a loaded host. On any stop
the pod is stopped at once and the reason recorded, so billing ends on failure as on success.

## 7. Hours and dollars

| item | amount | cost | plain meaning |
|---|---|---|---|
| H100 pod | about 2.5 h expected (setup 0.5, pre-freeze GPU work 0.5, freeze and beacon wait 0.4, sampling and reads 0.5, timing 0.2, slack 0.4) | about $9; hard ceiling 11 h = $38.39 | stopped as soon as step 8 ends or at any stop |
| Mac mini timing | sessions A and B, about 40 min of one core | $0 | quiet host |
| Mac mini CPU reads | about 8.7 core-h, 6 processes, 1.5-2 h | $0 | opponents' and full-DEM test reads |
| Derecho | none planned | $0 | fallback only |
| working time | about 5 h | | code 1 h, runs and checks 3 h, publication 1 h |

## 8. Timeline (PDT, 2026-10-07)

| time | what |
|---|---|
| 11:30 | revision 1 committed |
| 11:30-12:20 | Review re-check, pod created (step 2) |
| 12:20-13:05 | code (3); pod setup, host check, validation, cal reads, preflight, GPU numbers (4) |
| 12:55-13:15 | M4 session A (5) |
| 13:25 | freeze commit (6) |
| 13:35-13:50 | beacon, sampling, manifest pushed (7) |
| 13:50-14:20 | pod reads and cudaq-qec timing; pod stopped (8) |
| 14:00-14:30 | M4 session B (9) |
| 14:30-16:30 | Mac mini CPU reads (10) |
| 16:30-16:50 | the analysis, outputs pushed (11) |
| 16:50-17:50 | release rows, report (12) |

Completion today depends on step 4 passing (validation, host check, preflight); a failure there postpones the test.
About six hours of slack remain before midnight for queue or setup delays.

## 9. Freeze record (added with the freeze commit; no rule above changes)

Pre-freeze numbers and records are in `data/bb/stage3/` (`cells.json` holds the frozen parameters, `prefreeze`,
`versions`, `execution_notes` and the analysis command; rows, host check, validation and agreement files beside it).
- Host check on the pod passed: cal chunks 0 and 19 at both p regenerated bit for bit (arrays; np.savez stamps file
  times, so file hashes cannot be compared). Stim 1.16.0 loads its SSE2 module there by its own dispatch.
- H100 engine validation (on cal chunk 0; the monitor chunk is stored only on Alpine): 4 / 4 pass. r11 and r12 on all
  800,000 cal shots: no decision and no margin differs from the RTX reads; routed sets equal (500 and 4,000 shots).
- GPU side (slowest of 5 accepted processes): GPU term 0.0794 ms (r11), 0.0793 ms (r12); handoff 0.0225 / 0.0227 ms.
  The preflight passed both arms under the unchanged acceptance rule.
- Q8 (first 10,000 cal shots in place of dev chunk 0, which is stored only on Alpine; fastest of 3): p 0.003 sector
  Relay-BP 1.598 ms, BP-OSD 1.266 ms (M4), cudaq-qec 0.965 ms (H100); p 0.002: 1.002, 0.511, 0.474 ms. Stage-2 speed
  verdicts 2.6 (ii): BB72-3 and BB72-2 pass (RTX session: 0.086 ms against 1.266 / 0.511 ms; H100: 0.102 ms against
  0.965 / 0.474 ms).
- BB72-2F's screen: composite 0.3267 ms (GPU 0.0793 + handoff 0.0227 + CPU term 0.2247, the slowest of 3; the H100
  routes 500 cal shots, 89.9 ms each on the M4) against the fastest gating opponent 0.4744 ms (cudaq-qec): pass, so
  BB72-2F is tested.
