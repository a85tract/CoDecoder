# ConvLines-BB: plan for stage 2 (training + dev evaluation) and stage 3 (a registered test)

A versioned registration document (Q7). Version 1 (6f7ff034, 2026-10-06 00:40 PDT) was the stage-1 proposal; version
2 (56133614 and c2a01ed4) answered the independent stage-2 gate review and the study lead's decisions Q1-Q7. This is
version 3 (2026-10-06): the re-check of c2a01ed4 (verdict REVISE, no critical finding, 4 HIGH, 1 MED) is
answered in code (DESIGN 11), and the study lead's decisions Q8 and Q9 are recorded; it is committed before any
training, cal read or dev read. After the second re-check (of de3e17ad: REVISE, no critical, 1 HIGH, 1 MED), and
before any cal read, the speed gate was made to enforce 2.7's CPU-term population and cudaq-qec's configuration and
device (DESIGN 12); no registered rule changed. The design and the stage-1 evidence are in `bb/DESIGN.md`. A rule here changes only
through a new dated version committed before the data it governs exist; nothing is re-tuned after a dev or test read.

## 0. Decisions (study lead, 2026-10-06, confirmed by the gate review)

| id | decision | where it binds |
|---|---|---|
| Q1 | Full-DEM Relay-BP is a reported accuracy reference, never a gate. Every claim names the three gating configurations: sector Relay-BP, ldpc BP-OSD ms80-CS60, cudaq-qec GPU BP-OSD ms80-CS60 | 1, 2.6, 3.2; `bb.route` GATING |
| Q2 | Training and everything else on Alpine. No Derecho job in stages 2 and 3: its 12.93 A100-h stay study 4's margin | 4, 5 |
| Q3 | Provisional speed estimates (source and rule frozen in 2.7) screen policies only; quiet-host timings are required before the test freeze and a failed or missing one means stop; the composite must not exceed the fastest gating opponent, with the CPU term from the routed shots' own fallback costs; an empty feasible set means stop | 2.5, 2.7; `bb.route` LIMIT_MS, `bb.timing` |
| Q4 | Gross runs as a bounded probe with enforced stops: the early stop in `bb.train`, then a calibration screen (accuracy and routing admissibility) before replica 2 | 2.3; `bb.train --early-stop` |
| Q5 | A bounded port of a fused engine, with an early timing check | 2.4 |
| Q6 | No distillation, no BB72 -> gross transfer | 2.1-2.3 |
| Q7 | `bb/DESIGN.md` and `bb/PLAN.md` are versioned registration documents in this repository | this header |
| Q8 | The quiet-host CPU side runs on one exclusive 64-core Alpine acpu node (about 1.7 h). The study lead is told at least a day before it is submitted, so that another project using these nodes can be told | 2.7, 5 |
| Q9 | Deployed reads, the engine validation gate and all GPU-side timing, ours and cudaq-qec's, run on a full RTX PRO 6000 Blackwell allocated alone (Alpine partition artxpro6000); A100-40GB is the fallback. Committed in this version, before any cal read | 2.4, 2.7, 2.8; `bb.engine` signature |

Stage-2 resources (study lead): at most 3 Alpine GPUs at a time (BB72's two replicas first, then gross) and 64
cores in all; the RTX PRO 6000 is this route's after another study's jobs end (about 11:00 PDT on 10-06; those
jobs are never touched); Alpine stops for maintenance on 2026-10-07 at 05:30 PDT (2.8).

## 1. Cells, opponents and fixed settings

| cell | code, noise | why this p | opponents' dev failures (stage 1) | plain meaning |
|---|---|---|---|---|
| BB72-3 (primary) | [[72,12,6]], p = 0.003, 6 rounds | the closed route's setting; failures common enough to test | of 20,000: sector Relay-BP 269 (1.345 %), BP-OSD 330 (1.650 %), cudaq-qec 327 (1.635 %); full-DEM Relay-BP 111 (0.555 %) | the main question |
| BB72-2 | [[72,12,6]], p = 0.002 | the brief's second p | of 30,000: 72 (0.240 %), 87 (0.290 %), 87 (0.290 %); full-DEM 27 (0.090 %) | read with BB72-3's networks unless 2.2's fine-tune replaces one |
| G-4 | [[144,12,12]], p = 0.004, 12 rounds | at p = 0.003 sector Relay-BP fails about 3.4 per 10,000 shots, so a paired test would need about 10^6 shots; at 0.004 it fails 0.5 % | of 20,000: 103 (0.515 %), 523 (2.615 %), 515 (2.575 %); full-DEM 1 of 640 | where failures are measurable within budget |

Gating opponents, each paired with ours on the same shots (Q1): sector Relay-BP (IBM's relay-bp 0.2.2 with the
authors' tuned parameters, `bb.opp relay`), ldpc BP-OSD ms80-CS60 on CPU (`bposd`) and cudaq-qec's GPU BP-OSD
ms80-CS60 (`gpu_bposd`; NVIDIA Ising has no qLDPC model, so this is the NVIDIA opponent). Full-DEM Relay-BP
(`relay_full`) is reported beside them; it was 2.4-2.7 times more accurate than sector Relay-BP on BB72 in stage 1 at
27 times its mean time per shot. The fallback for routed shots is sector Relay-BP (DESIGN 5.2).
 Data (fresh, this route's namespace, DESIGN 9): per cell a calibration split (cal: tau and the screens only), a dev
split (the gates) and a monitor chunk (training curves and the engine's validation). Sizes: BB72-3 cal 200,000 / dev
200,000; BB72-2 cal 200,000 / dev 300,000; G-4 cal 100,000 / dev 100,000, with about 2,700, 720 and 500 expected sector
Relay-BP dev failures, enough for the screens (dev sizes guarantee no power). Stage 2 extends the stage-1 splits
(chunks j = 2, 3, ...). The stage-1 routing smoke read BB72-3's dev chunks 0-1 once with a 9-minute network; dev is a
screen, and the claims rest on stage 3's untouched beacon-seeded test. Full-DEM Relay-BP reads the BB72 dev splits in
full and gross's first 10,000 dev shots (4.3 s per shot).

## 2. Stage 2: training, the engine, calibration, dev

### 2.0 Order of work (registered)
1. Train (2.1-2.3). 2. Port and validate the deployed engine; choose it per code (2.4). 3. Calibration reads through
the deployed engine, provisional speed estimates, policies and screens (2.5, 2.7). 4. Commit all policies of a code in
one commit, push, redeploy the clean tree. 5. Only then the code's dev reads and the paired evaluation (2.6). 6.
Quiet-host timing and the speed gate (2.7). 7. Stage 3 only for cells that pass, after the study lead's approval.

Read-order rule: no failure count of any decoder on a dev split is printed or recorded before step 4. `bb.opp` and
`bb.gpu_bposd` read dev blind (predictions stored, no count printed or written), so the opponents may decode dev
during training; the network's dev read and every paired evaluation go through the commit guard of `bb.route`, which
refuses unless the deployed tree is a clean commit and every policy file of the code in `data/bb/policies` has the
sha256 that `deploy.sh` recorded from that commit (DEPLOYED_POLICIES; a policy created or edited after the deployment
never passes), both policies of the cell passed their screen, the checkpoint is the policy's network and the engine
signature is the policy's. `bb.route frontier` refuses any split but cal.

### 2.1 BB72 (cells BB72-3 and BB72-2)
- Two replicas, run ids 11 and 12 (independent streams): ConvLines-BB b3, width 128, bottleneck 32, 4 heads, Tanner
  block taps (483,357 parameters, 118 M MACs per shot), true labels at p = 0.003, 2 x 10^8 samples each, DESIGN 7's
  recipe, torch.compile, monitor every 10^7 samples, the final EMA = the network.
- Throughput: 25,031 samples per second with torch.compile on a Derecho A100-SXM4-40GB in stage 1 (19,736 eager), so
  about 2.3 GPU-h per replica on an A100. On an Alpine H200 MIG 2g.35gb slice the stage-2 smoke trained it at 14,846
  samples per second with a warm compile cache (7,054 with a cold compile), about 3.7 h per replica; the Alpine
  A100-40GB rate is not measured yet (its queue was full).

### 2.2 BB72-2 fine-tuning: mandatory when its trigger fires
- Trigger, per replica, on its BB72-2 calibration read through the deployed engine: the network alone fails more than
  twice as often as sector Relay-BP on BB72-2's cal split.
- Then the fine-tune runs, no discretion: run id 13 from replica 1's final EMA (14 from replica 2's), p = 0.002,
  5 x 10^7 samples, peak learning rate 2 x 10^-4, 500 warm-up steps, cosine to 2 %, a fresh optimizer and EMA, its own
  stream (seeds `chunk_seed(13 or 14, 0.002, c)`), BB72-2's monitor chunk; about 0.6 GPU-h each.
- Replacement: replica r's BB72-2 network becomes the fine-tuned one if its network-alone BB72-2 cal failures are
  fewer than the original's, else the original stays. BB72-3 always uses the original. A fine-tune that does not
  complete leaves the original in place.

### 2.3 Gross (cell G-4): a probe with enforced stops
Stage-1 evidence (DESIGN 8.2): at p = 0.005 from scratch the network stayed at chance for 2.2 M samples (7 minutes);
with a ramp from 0.001 to 0.004 over 2.1 M samples it still failed 4,952 of 5,000 p = 0.004 monitor shots at 3 M; at
p = 0.001 it learned. So gross trains with the curriculum `--p-ramp 0.001:0.5` (p = 0.001 rising on the circuit grid
to 0.004 over the first half: about 8 M samples per grid step, then 5 x 10^7 at 0.004): b3, width 128 (821,085
parameters, 487 M MACs per shot), true labels, 10^8 samples, DESIGN 7's recipe, torch.compile; 5,290 samples per
second eager on an A100 in stage 1, so about 5 GPU-h per replica; 3,624 samples per second compiled on an H200 MIG
2g.35gb slice in the stage-2 smoke, so 7.7 h there.
- Probe = replica 1, run id 21, with `--early-stop 5e7:0.20`: at the first chunk boundary at or after 5 x 10^7 samples
  (the ramp's end) `bb.train` evaluates the EMA network on the p = 0.004 monitor chunk and stops the run if it fails
  more than 20 % of those shots (status early_stop; no final network, so nothing downstream can read it). Gross then
  ends with its numbers reported.
- Probe gate, on the cal split only: replica 1's screen (2.5) must pass, that is calibration accuracy (the policy
  fails fewer times than sector Relay-BP; the network alone at most 3 times as often) and routing admissibility (a
  non-empty feasible set under 2.7's provisional estimates). Only then does replica 2 (run id 22, the same
  configuration, its own stream) start. Fail: gross ends; no gross dev read (the commit guard needs replica 2's
  policy).

### 2.4 The deployed engine (Q5): bounded port, early timing check, validation
Every calibration, dev and test read of a network and every timing of it runs through one deployed engine (`bb.engine`):
batch 1, so no padding; fixed precision; one captured CUDA graph; cuDNN deterministic, TF32 off; reads on the
registered GPU model only (2.8, Q9: a full RTX PRO 6000 Blackwell; A100-40GB the fallback). The engine signature
(kind, batch, precision, GPU model, torch / CUDA / cuDNN versions, the network's and the engine code's sha256) is
stored with every read; policies record it, and `bb.route` refuses to mix reads with different signatures, so the
margins that set tau are the deployed decoder's own.
- Rungs: (0) `graph`, the PyTorch module (bf16 autocast for the trunk and read-out, fp32 vote and margin); (1)
  `inductor`, the same computation compiled by torch.compile in its default mode (no autotuning); (2) a fused Triton
  engine in arch.engine's style (gather-GEMM tap convolutions, the read-out pooling and MLP, the vote and top-2 in a
  few kernels), the port proper; and `graph32`, rung 0 entirely in fp32 (TF32 off), for a network whose bf16 rungs
  fail validation. Under bf16 the stage-2 smoke network (a few minutes of training, 41 % cal failures) agreed with the
  fp32 module on 99.6 % of shots, short of the bar below; the speed study's bf16 engine agreed on 99.98-100 % on
  trained surface-code networks (speed/README.md).
- Bounds of the port: at most 8 working hours from the start of training and 2 GPU-h. Its GPU work starts when a GPU
  is free under the cap (after BB72's training at the latest); its timing check and the validation run on the
  registered GPU model.
- Early timing check: the fused engine must run end to end on BB72 (random weights) within 4 working hours of the
  port's start, and its batch-1 GPU term (2.7's provisional rule) must then be at most 40 % of the faster of rungs 0
  and 1; otherwise the port stops and rung 0 or 1 is deployed.
- Validation of any rung before it is deployed, on the cell's 10,000 stored monitor shots (never cal or dev): two fresh
  processes give bitwise equal outputs; decisions agree with the fp32 module on at least 99.9 % of shots and the margin
  differs by at most 5 % of the median margin on 99 % of shots. The thresholds catch implementation bugs and say
  nothing about accuracy, since the policy is fitted to the engine's own margins.
- Choice: per code, the validated rung with the smallest provisional GPU term, fixed before the code's first cal read
  and recorded in every policy. No accuracy-based choice.

### 2.5 Calibration, policies, screens and the commit
- Per replica and cell: `bb.route infer` on the whole cal split through the deployed engine; the fallback's cal read
  (`bb.opp relay`); `bb.route frontier` with 2.7's provisional estimates. Policy rule (DESIGN 5.1): among the
  admissible q of the grid, the smallest whose cal failures are at most M - 2 sqrt(2 M), M = round(0.90 x sector
  Relay-BP's cal failures), else the admissible q with the fewest (ties to the smaller q); an empty feasible set gives
  no policy.
- Screen per replica and cell: a policy exists, its cal failures are fewer than sector Relay-BP's, and the network
  alone fails at most 3 times as often as sector Relay-BP. A cell whose either replica fails its screen stops before
  dev (the commit guard enforces it).
- Commit: all policy files of a code (BB72: 2 cells x 2 replicas; gross: 2 replicas), screened or not, go into
  `data/bb/policies/<code>_p<p>_rep<r>.json` in one commit with the test replica of each cell (3.1), pushed to the hub
  and redeployed before any dev read of that code.

### 2.6 Dev reads and the two-replica gate (per cell, as arch README 15.2 / 15.5)
After the commit: each replica's network read on the whole dev split, then `bb.route apply` against every opponent.
A cell passes iff both replicas completed, their checks pass, and each replica's committed policy (i) is better than
each of the three gating opponents on the dev split (exact two-sided McNemar p < 0.05 and the Newcombe 95 % interval
of the failure-rate difference entirely below 0) and (ii) passes the speed gate of 2.7 on the quiet hosts. A cell that
fails is reported with its numbers; nothing is re-tuned on dev.

### 2.7 Timing: provisional estimates and the quiet-host gate
Provisional estimates (frozen now; used only to screen policies and to gate the gross probe):
- GPU term: `bb.latency` with the deployed engine and the final network on the registered GPU model, three fresh
  processes in one job; each process's value is the mean of 5 x 100 timed graph replays after 200 warm-up replays;
  counted: the slowest process.
- CPU term: the fallback's per-shot wall times recorded by `bb.opp` in the cal read (32 single-threaded processes on a
  shared acpu node), summed over the routed cal shots and divided by the number of cal shots.
- Handoff: the larger of the 1.1 us allowance and the slowest handoff measured by the same three `bb.latency`
  processes (`bb.route frontier --handoff-us`; 12.4 us per shot on an H200 MIG slice in the smoke).
- Limit: the fastest gating opponent's stage-1 batch-1 time (DESIGN 6; cudaq-qec on a Derecho A100): 1.06 ms (BB72-3),
  0.63 ms (BB72-2), 15.8 ms (G-4), coded as `bb.route.LIMIT_MS`.

Quiet-host protocol (registered; after the dev reads, before any test freeze; `bb.latency`, `bb.timing`,
`bb.gpu_bposd --timing-out`):
- GPU side, one full card of the registered model allocated alone (2.8): per network five fresh `bb.latency`
  processes (counted: the slowest), the handoff measured in the same processes inside the same watched window
  (counted: max of 1.1 us and the slowest measurement), and cudaq-qec's batch-1 call (`bb.gpu_bposd time`, which
  writes no predictions), 300 calls on the first dev shots in three fresh processes per cell (counted: the fastest).
  Acceptance per process, ours and cudaq-qec's (`bb.timing.gpu_watch`): no other process on the GPU and every 100-ms
  SM clock sample at least 97 % of the maximum; a rejected process is replaced, at most twice per network or cell,
  and a third rejection fails the session.
- Records: every row names its session (the Slurm job), its process (a fresh id per process) and its binding (the
  digest of the policy's engine signature for the GPU term and the handoff, the policy file's sha256 for the CPU
  term). `bb.timing gate --gpu-session --cpu-session` counts only those two sessions' rows bound to the policy, one
  row per process, the handoff rows from the GPU-term processes; a third rejection gives a failed session. A CPU-term
  row counts only if it was measured on the gated split (dev here, test in stage 3) over its complete routed
  population: every chunk of that split in the shot manifest, the routed set recomputed from the policy network's
  read at tau, the first 5,000 (BB72) or 1,000 (gross) routed shots timed; a cudaq-qec row only in the registered
  configuration (ms80_cs60) on the policy's GPU (the device in its engine signature).
- CPU side (Q8), one exclusive 64-core Alpine acpu node (about 1.7 h; announced to the study lead at least a day
  before it is submitted), one single-threaded process at a time, pinned to one core: per policy the
  fallback on its routed dev shots in split order (at most 5,000 on BB72, 1,000 on gross), three fresh processes,
  counted: the slowest, CPU term = routed fraction x their mean time per shot; sector Relay-BP and BP-OSD on the first
  10,000 (BB72) or 1,000 (gross) dev shots, three fresh processes each, counted: the fastest.
- Speed gate (2.6 (ii), `bb.timing gate`): composite = GPU term + CPU term + handoff, at most the fastest gating
  opponent. A failed gate, or a session that cannot be completed before the test freeze, means the cell stops.

### 2.8 Infrastructure, maintenance, resume and retries (frozen)
- Resume: `bb.train` checkpoints at chunk boundaries at every monitor point (10^7 samples) and at every pause, with the
  weights, the EMA, the optimizer, the step (which fixes the schedule and the stream position: chunk c has the seed
  `chunk_seed(run_id, p_c, c)`), the sampler environment and the log; `--resume` continues exactly and refuses changed
  settings and any other sampler environment (Stim, NumPy and Python versions, the venv and the hash of its package
  list, a fingerprint of Stim's output at the run's p: the stream is regenerated, so it must come out bit for bit the
  same). Every run either finishes before Alpine drains or pauses and resumes this way, in its own environment (ising's
  venv_gpu in the pyfull container, on any card but a Blackwell one).
- Maintenance: Alpine's reservations pm-10.7-* run from 2026-10-07 05:30 PDT to 2026-10-08 05:30 PDT on every partition.
  04:30 PDT is our drain buffer: every job ends by then (`gpu.sbatch` pauses training 10 minutes before the earlier of
  the job's end and 04:30), and 04:30 is never a start or restart time. Paused runs resume only after service is
  restored (partitions up and the reservations gone). Walltimes are the expected run time x 1.3, capped at the drain,
  with `--time-min 1:00:00` so the scheduler can fit a job before the reservation.
- Infrastructure failures (node or driver failure, preemption, out-of-memory kill, time limit, filesystem error, a
  system cancellation): a training run resumes from its last checkpoint; a read reruns from scratch (network reads are
  deterministic for a fixed engine signature; Relay-BP and cudaq-qec reads are not (DESIGN 5.2), and every rerun
  replaces the whole read: a cudaq-qec read stores its id in every chunk and replaces the stored read only once
  complete, and `bb.opp.load_preds` refuses chunks of different reads); at most 3 retries per job, a fourth failure
  fails that run or read.
- A non-finite loss fails the replica (`bb.train` exits 4 and marks its checkpoint terminal); no restart.
- Bugs: a fix that changes no registered setting (model, data, seeds, schedule, rules, thresholds, engine numerics) is
  committed and logged, then the run resumes or the read reruns; a fix that would change one stops the work until the
  study lead decides.
- No retry, resume, rerun or fine-tune is ever triggered by a monitor, cal or dev number, except the registered
  triggers of 2.2, 2.3 and 2.5.
- GPU types: training on any Alpine GPU (A100-40GB / 80GB, H200, L40, or an H200 MIG slice of 2g.35gb or more),
  whichever starts first. A job waiting more than 30 minutes ends the session with a handoff.
- Registered GPU model (Q9): the deployed reads of every split (monitor validation, cal, dev, test), the engine
  validation, every GPU term and handoff, and cudaq-qec's reads and timing run on a full NVIDIA RTX PRO 6000 Blackwell
  allocated alone (artxpro6000, `--gres=gpu:rtx_pro_6000:1`; torch 2.11.0 built for CUDA 12.8, `$R/venv_cu128`, since
  the training venv's CUDA 12.6 build has no Blackwell kernels). Fallback, by availability only and never by an
  accuracy or speed number: A100-40GB (aa100) for a code if the engine or cudaq-qec cannot run on Blackwell, or if no
  full RTX PRO 6000 has started a job of ours within 3 hours of the first submission of that code's engine
  validation. The model is fixed per code at that validation, before the code's first cal read, and recorded;
  the engine signature names it, so a code's reads and timing never mix models, and a later switch would
  mean making every read of that code again.

## 3. Stage 3: a registered test (pre-specified now, before any dev read)
Only for cells that pass 2.6, after the study lead's approval.
1. Test replica: per cell, the replica whose committed policy has fewer cal failures; ties go to the smaller
   provisional composite, then to replica 1 (`rank_key` in the policy; `bb.route size` checks it). Fixed by the commit
   of 2.5, before any dev read. Only that replica's frozen artifact is read on the test.
2. Claim and multiplicity: per cell one claim: the hybrid (the test replica's network through the frozen engine, its
   committed policy, sector Relay-BP as fallback) has a lower block failure rate than each of sector Relay-BP, ldpc
   BP-OSD ms80-CS60 and cudaq-qec GPU BP-OSD ms80-CS60 on the cell's test shots, and its composite batch-1 time on the
   frozen hosts is at most the fastest of these three's. Accuracy: each of the three comparisons by the exact two-sided
   McNemar test at alpha = 0.05 / 3 and the Newcombe (1 - 0.05 / 3) interval entirely below 0; the claim needs all
   three (an intersection-union test, so no correction within a cell; `bb.route apply --split test` uses
   ALPHA_TEST = 0.05 / 3 for both); 0.05 / 3 is each registered cell's Bonferroni
   share, fixed whether or not every cell reaches the test, so the family-wise error over the three cells' claims is at
   most 0.05. Speed: a measured inequality under 2.7's protocol, without a statistical test. Not claimed: the
   best-known accuracy (full-DEM Relay-BP, reported beside on the first 200,000 BB72 test shots and the first 10,000
   gross test shots), universally fastest decoding, or other noise models, codes, p or hardware.
3. Size (`bb.route size`): per gating opponent o, n_o = the smallest multiple of 10,000 from the cell's floor up to its
   cap at which 3.2's decision passes with probability at least 1 - 0.10 / 3, simulated (20,000 draws, fixed seed)
   from the test replica's paired dev counts against o; n = the largest n_o, so all three comparisons pass with
   probability at least 0.90 (union bound). Floors = the dev sizes (200,000 / 300,000 / 100,000); caps 1,000,000 (BB72
   cells) and 300,000 (gross, CPU budget). Cap handling, deterministic: if any n_o exceeds the cap the cell is not
   tested and is reported as underpowered with each comparison's power at the cap. Nothing is extended after any test
   shot exists.
4. Freeze (one commit, then a review via the study lead): network artifacts (final EMA, sha256), each policy's tau
   at full precision, the engine code, venv and signature (with the GPU model), the fallback's and every opponent's
   configuration and versions, the timing hosts and protocol, n per cell, the analysis code; nothing changes after the
   freeze.
5. Seeds: after the freeze commit, the first drand round at least 10 minutes later; seeds = SHA-256(randomness || TAG ||
   chunk index) mod 2^62 + 2^62 with TAG = "ising_compare bb test <cell>\n" (studies 3-4's construction, one tag per
   cell); sampled once on Alpine, stored with sha256, read-only.
6. Reads: each decoder once on the test shots (ours through the frozen engine and router, every opponent with its
   frozen configuration), every read logged with its hash; then 3.2's tests, and 2.7's quiet-host protocol on the
   routed test shots for the composite.

## 4. Budget

| item | where | amount | plain meaning |
|---|---|---|---|
| BB72 replicas 11, 12 | Alpine GPU | 2 x 2.3 = 4.6 GPU-h | training, 2 x 10^8 samples each |
| BB72-2 fine-tunes 13, 14 (only if 2.2 triggers) | Alpine GPU | up to 1.2 GPU-h | 5 x 10^7 samples each |
| gross probe 21 | Alpine GPU | 5.3 GPU-h (2.7 if the early stop fires) | 10^8 samples |
| gross replica 22 (only after the probe passes) | Alpine GPU | 5.3 GPU-h | |
| engine port, early check, validation | registered GPU (2.8) | at most 2 GPU-h | bounded (2.4) |
| deployed reads: monitor, cal, dev | registered GPU | about 1 GPU-h | batch 1 through the engine, about 0.6 ms per shot |
| provisional GPU terms | registered GPU | about 0.3 GPU-h | three processes per network |
| cudaq-qec dev reads and batch-1 rows | registered GPU | about 0.5 GPU-h | the NVIDIA opponent |
| quiet-host GPU side | registered GPU | about 1 GPU-h | 2.7 |
| CPU opponents on cal and dev | Alpine acpu, at most 32 cores | about 40 core-h | 35 core-h from stage 1's per-shot times; full-DEM Relay-BP is 21 of them |
| quiet-host CPU side | one exclusive 64-core acpu node (Q8) | about 1.7 h = 110 core-h | the whole node is charged |
| smoke tests (stage-2 revisions) | Alpine GPU | under 2.5 GPU-h | version 2's code checks, and version 3's on one Blackwell MIG slice |

Stage 2 asks for about 21 GPU-h on Alpine (6.5 of them conditional on 2.2 and 2.3; about 5 on the registered GPU)
and about 150 core-h; no Derecho hours (Q2). GPU-h are hours of one GPU (A100-equivalent); core-h are CPU
core-hours. Stage 3 adds the test reads (about 50 core-h at the size caps), the quiet-host protocol on the test (about
1 GPU-h and 110 core-h) and the engine's test reads (under 1 GPU-h).

## 5. Schedule (PDT; T0 = 2026-10-06 about 05:00, the start of version 3's training)

| when | what | needs |
|---|---|---|
| T0 | version 3 committed and deployed; BB72 replicas 11, 12 and the gross probe 21 (`s2_train.txt`) on H200 MIG slices, the only GPUs free at T0 (`-t` 5:00 / 5:00 / 10:00, `--time-min 1:00:00`); the splits extended (`s2_shots.txt`, about 15 min, 16 cores), the manifest rows committed, the tree redeployed; then the CPU opponents on cal and dev, dev blind (`s2_opp.txt`, about 1.5 h, 32 cores) | 3 GPUs, CPU |
| T0 + 4 h | BB72 training done (about 3.8 h on a 2g.35gb slice); the gross probe's early-stop check (5 x 10^7 samples) between T0 + 3 h and T0 + 4 h | |
| from about 11:00 | the RTX PRO 6000 (after the other study's jobs): the engine port's GPU work and early timing check, then the validation; cudaq-qec's blind dev reads (`s2_cudaq.txt`) | registered GPU, the probe |
| T0 + 6 h to T0 + 8 h | the gross probe done | |
| by T0 + 11 h | engine validated and chosen per code (the port's 8 working hours end at T0 + 8 h); BB72 cal reads, provisional GPU terms, frontiers, screens; the BB72-2 fine-tune trigger (runs 13, 14 if it fires, about 1 h each on a slice) | registered GPU, up to 2 others |
| T0 + 12 h | gross probe gate; replica 22 starts if it passes (5.3 h on an A100, 7.7 h on a 2g.35gb slice) | 1 GPU |
| T0 + 13 h | BB72 commit (4 policies), redeploy, BB72 dev reads and `apply`; BB72's quiet-host GPU side | registered GPU |
| T0 + 20 h | gross replica 22 done, or paused at the 04:30 drain and resumed after the maintenance; its cal read, the gross commit, gross dev reads, gross's quiet-host GPU side | registered GPU |
| 10-07 04:30 to 10-08 05:30 | drain and maintenance: nothing runs | |
| after the maintenance | quiet-host CPU side on an exclusive node, announced to the study lead at least a day ahead (Q8; announced at T0, so 10-08 06:00 at the earliest); then each cell's speed gate (2.6 (ii)) | 64 cores |

Every time after T0 assumes no further queue wait. A run that reaches its walltime pauses (exit 3) and resumes when
`s2_train.txt` is submitted again (2.8); reads wait for the registered GPU.

## 6. Risks and stop conditions

| risk | what we know | rule |
|---|---|---|
| Relay-BP leaves little room | BB72-3: sector Relay-BP 1.345 %, BP-OSD 1.65 % on dev; the 9-minute smoke hybrid beat sector Relay-BP on stage-1 dev (0.945 %), full-DEM Relay-BP is at 0.555 % | screens (2.5) stop a cell before dev; the gate (2.6) needs both replicas |
| failures are rare at low p | BB72-2: Relay-BP 0.24 %, BP-OSD 0.29 %; gross at 0.003: 0.05 % | BB72-2 reads 300,000 dev shots; gross sits at p = 0.004; 3.3's size rule and cap |
| gross may not learn at p = 0.004 | closed route: 2.47 % against Relay-BP's 0.034 % at p = 0.003; stage 1: a 7-minute run from scratch at p = 0.005 and a 3 M-sample curriculum to 0.004 both failed | 2.3: curriculum, enforced early stop, probe gate before replica 2 |
| routing cost | routed shots are the hard ones: 17 ms per routed BB72 shot at q = 3 % against Relay-BP's 3.8 ms mean (loaded node); on gross the mean is 143 ms and routed shots will cost more | the admissibility filter (2.5) and the quiet-host gate (2.7); an empty feasible set stops the cell |
| speed | the smoke policy's provisional composite (one-process median GPU term of 541 us, PyTorch graph) was level with cudaq-qec's 1.06-1.30 ms; better training may route fewer shots, but the registered GPU term is the slowest of five processes and the GPU term alone is half the BB72-3 limit and most of the BB72-2 limit (0.63 ms) | the engine port (2.4) with its early check; the speed gate decides |
| GPU queue | at 03:45 PDT on 10-06 the gpu-normal QOS had every A100-40GB it may use busy (150 jobs pending on aa100) and all 16 full RTX PRO 6000 cards were allocated; H200 MIG 2g.35gb slices were idle | training on any GPU of 2.8; `--time-min`; resume across interruptions; a wait over 30 minutes ends the session with a handoff; the registered GPU's availability fallback (2.8) |
| Blackwell software | ising's venv_gpu is torch 2.11.0 for CUDA 12.6, without sm_120 kernels: its test computation failed on a Blackwell MIG slice (s2smoke4, on Alpine), where `$R/venv_cu128` ran every engine rung and cudaq-qec 0.8.0 read and timed | the engine on `$R/venv_cu128` (gpu.sbatch picks it on Blackwell); A100-40GB if the engine or cudaq-qec cannot run on the full card (2.8) |
| maintenance | 2026-10-07 05:30 PDT to at most 10-08 05:30 PDT | 2.8: drain at 04:30, faithful resume after service returns |

## 7. Open questions for the study lead

| id | question | proposal |
|---|---|---|
| Q10 | When the other study's jobs end, do their RTX PRO 6000 cards pass to this route, or does our job queue with the others (70 jobs pending on artxpro6000 at 03:45 PDT, every full card allocated)? | submit the engine work to a full card when BB72's training ends and queue normally; 2.8's fallback after 3 hours |

## 8. Deviation log (disclosed; how events were counted, no registered rule changed)

- 2026-10-06, r21's retries under 2.8. Infrastructure failure 1: Alpine's power event (10:16 PDT) killed a training
  segment. Failure 2: when compute returned at 15:38 PDT the resume chain died at launch (exit 0:53, before its script
  ran) on gpu-node-03, which had come back without the GPFS scratch file system; the resubmission at 15:45 PDT
  went to the same node and died the same way, with nothing run. The study lead counted the two launch failures as
  one (one node, one cause), so the chain that completed r21 (15:48-16:35 PDT) was retry 2 of 3; it
  had first been counted as retry 3 of 3.
- 2026-10-06, gross's engine job: its first submission (16:34:59 PDT) died the same way on gpu-node-06
  (retry 1 of 3); the resubmission with that node excluded, the 3-h window unchanged, started at 16:39:23 PDT on
  another full RTX PRO 6000, so gross's registered model is the full RTX PRO 6000 under either reading of 2.8.
