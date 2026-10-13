# Cascade behind ConvLines on the surface code: the plan (2026-10-10)

Exploratory, not registered. On 2026-10-10 the authors asked for three things on the rotated surface code:
- reproduce Cascade's surface-code network (Gu, Bonilla Ataides, Lukin and Yelin, arXiv:2604.08358);
- route the least confident shots of our small registered ConvLines network to it, and from it to correlated
  PyMatching;
- report accuracy and speed against Cascade alone.

The claim under test is the authors' framing. ConvLines' core is the small network plus confidence routing, and strong
decoders such as Cascade sit behind it as fallbacks. Phase C tests the same claim on BB72
(`cascade_compare/PHASE_C_PLAN.md` on `ising-cascade-phasec`). This plan reuses phase C's routing rule, screens, speed
protocol and statistics, and section 4.6 lists where it departs from them.

Status: the plan was reviewed before the training run (twelve findings, all accepted and folded in here), and the
evaluation code before any development or confirmation read (seven findings, all fixed). CS13 finished on 12
October (`DONE` at 03:39 UTC, no observed improvement at step 81,130), and the evaluation reads its
`final_ema.pt`. The results are in `RESULTS.md` beside this file.

## 0. Summary

| item | choice |
|---|---|
| distance | d = 13 first. d = 17 only if d = 13 shows the expected pattern (section 1) |
| circuits | the release's: NVIDIA Ising-Decoding legacy circuit, rotated code, X memory, R = d; DEPOLARIZE1(p) idle, DEPOLARIZE2(p) after CX, flips 2p/3 at reset and measurement; last round noiseless |
| test p | 0.6 % (the registered regime) and p_low = 0.35 %, the smallest of {0.20, 0.25, 0.30, 0.35, 0.40} % at which a confirmation within the cap gives enough expected Cascade failures, judged from correlated PM's measured failure rate (section 2.2) |
| training | the paper's three-stage curriculum to p2 = 0.7 %, batch 3,328, 80,000 steps (2.66e8 samples, `budget_ema.pt` kept), then the floor learning rate up to 120,000 steps unless the monitor shows no observed improvement (section 3.3) |
| network | Cascade-surface, L = 13 (= d, the paper's rule), H = 256 (not stated by the paper; inferred from its surface examples, section 3.2), 2.53 M parameters, the BB pilot's literal recipe (section 3): an explicitly inferred, strong local reproduction, with no claim of the paper's accuracy |
| configurations | C1 ConvLines -> Cascade -> corr. PM; C2 Cascade alone; C3 corr. PM alone; C4 ConvLines -> corr. PM (registered); C5 ConvLines -> Cascade |
| thresholds | phase C's rule on a development block at each p (quantile candidates, a predicted-speed cap of 0.9 x Cascade with matching's time over each candidate's own routed shots, the fewest development failures), locked and committed before the confirmation read |
| statistics | every computation that a claim rests on is read once on one fresh confirmation block per p, in both modes (single shot and batched; section 4.4); paired exact McNemar tests, Newcombe intervals, non-inferiority at 10 % of Cascade's failure rate; one Holm family of 40 hypotheses at alpha = 0.05 |
| claim | a chain is "faster within the predefined accuracy tolerance" than Cascade alone (non-inferior and not worse), or "faster and more accurate" (better); section 7 |
| speed | phase C's final protocol: every GPU step of the evaluation (development, confirmation, timing) on one GPU model in one venv, a full RTX PRO 6000 with bb's `venv_cu128`; PyMatching on the same node's CPU; Cascade alone timed beside the chains in the same harness and job |
| compute | CS13 training: 49.4 RTX PRO 6000-hours for the 2.66e8-sample budget, 24.7 hours wall on two cards with DDP (measured; DDP saves wall time, not GPU-hours), up to another 24.7 GPU-hours with the extension; evaluation and speed about 12-23 GPU-hours, most of it Cascade's batch-1 confirmation read (section 6.2). All on Alpine; RunPod is a contingency of at most about 90 USD, not planned |

## 1. The distance

d = 13 comes first:
1. Our registered networks cover d = 7-17. d = 13 is the largest distance of study 2 with a released fp32 network
   (`d13_b5_r775`; d = 17 exists only as deploy tables for the GPU engine), and it has a registered routing to
   correlated PM: q = 2 %, tau = 1.8486117124557495. On 1,000,000 T2 test shots ConvLines -> corr. PM failed 5,627
   times and corr. PM alone 6,549 times (release, `results/s2_test_eval.csv`).
2. Cost. The paper's d = 19 model takes about 200 H200 GPU-hours. A training step costs (d + 1)^3 nodes x L ~ d
   blocks, so d = 13 costs 14^3 x 13 / (20^3 x 19) = 0.23 of d = 19 per sample, about a fifth. The pilot measures
   1,505 samples/s at d = 13 on a full RTX PRO 6000 (section 8). That is 49 GPU-hours on that card for the paper's
   2.66e8 samples, and the H200 is faster (section 6).
3. Separation. Cascade's advantage over correlated matching grows with distance (Lambda 8.4 against 7.8 at p = 0.2 %
   in the paper's Fig. 3c). Extrapolated with the paper's own fit, Cascade fails about 0.6 times as often as
   correlated matching at d = 13 ((7.78 / 8.38)^7 = 0.59), against 0.74 at d = 7 ((7.78 / 8.38)^4), where the two
   curves of Fig. 3 nearly meet. d = 13 is thus the smallest registered distance at which the fallback should be
   clearly stronger than correlated PM. The ratio leaves out the fits' prefactors, so it motivates the choice of d
   and sizes nothing (section 2.2; review finding 7).

d = 17 follows as a second step only if d = 13 shows the expected pattern on the confirmation block, with the labels
of section 4.5:
- (P1) Cascade-surface alone is "better" than correlated PM alone (C2 against C3) at both test p, in both modes.
  Without it the fallback is not strong, and a larger d would only add cost;
- (P2) one and the same chain X, C1 or C5, (a) is "faster within the predefined accuracy tolerance" (or "faster and
  more accurate") than Cascade alone at p = 0.6 %, the registered regime (section 7), and (b) at p_low is "better" than
  or "non-inferior" to Cascade alone in both modes. A "level", "inconclusive" or "worse" result at p_low does not
  count, nor does a pass at p_low by the other chain (review finding 4).

d = 17 is the largest registered ConvLines distance (study 4), and there the separation between Cascade and
correlated PM should be larger again. It costs 18^3 x 17 / (14^3 x 13) = 2.8 times d = 13 per sample; the pilot
measured 540 samples/s at d = 17 on the RTX PRO 6000, 2.78 times slower, so its training alone takes about 137 RTX
PRO 6000-hours (section 6.1), which an underpowered result at p_low does not justify. It needs the release's deployed
engine for ConvLines, since d = 17 ships as deploy tables only. If (P1) holds and (P2) fails, the report says so and
d = 17 is not proposed. If (P1) fails, the reproduction is examined first.

## 2. The setting

### 2.1 Circuits

The release's verdict circuits: `circuits/d13_legacy.stim` (sha256 86f1d9e4...), read through the release's own
loader `convlines.circuits.circuit`, which checks the sha256. This is NVIDIA Ising-Decoding's `MemoryCircuit`
(commit 3ce4379):
- rotated surface code, X-basis memory, code rotation XV, R = d rounds, every detector plus the boundary detectors;
- the legacy single-parameter noise: DEPOLARIZE1(p) idle noise on every qubit, DEPOLARIZE2(p) after each CX layer,
  X_ERROR / Z_ERROR(2p/3) after resets and before measurements;
- in the file, the last syndrome round and the data readout carry no noise.

The files exist at p = 0.006 only. `surface_layout.circuit_at(d, p)` rewrites every DEPOLARIZE argument to p and every
flip to 2p/3, after checking that each equals 0.006 or 0.004. A unit check confirms that p = 0.006 gives the pinned
text back exactly; the same check passes at d = 17.

The paper's circuits and noise are not stated beyond "circuit-level depolarizing noise" generated with Stim. Our noise
differs at least in SPAM (2p/3), idle noise and the noiseless last round. The comparison is therefore between decoders
on our circuits; the paper's Lambda values and timings appear as context, and no verdict is judged against them.

### 2.2 Test p

- p = 0.6 %, the registered regime of ConvLines and its routing.
- p_low = 0.35 %, closer to the paper's 0.2 %, chosen by a rule. p_low is the smallest p in
  {0.20, 0.25, 0.30, 0.35, 0.40} % at which 0.6 x r_M(p) x 4e7 >= 1,200, where:
  - r_M(p) is correlated PM's block failure rate on 1,000,000 fresh shots at p (seed kind 6);
  - 0.6 is the paper's Cascade-to-correlated-matching ratio extrapolated to d = 13 (section 1);
  - 4e7 is the confirmation cap at p_low (section 4.4);
  - 1,200 is about the number of Cascade failures at which the non-inferiority test of section 4.5 reaches power 0.9
    when the discordant rate is about 0.7 of Cascade's failure rate.

  The condition is r_M >= 5e-5. The rule uses correlated PM only, with no network output involved. It ran during
  planning (job AJ18, 5.3 minutes on 16 cores; `surface/out/plow_d13.json`):

  | p (%) | correlated PM failures / 1e6 | r_M (Wilson 95 %) | plain PM failures | correlated PM µs per shot (EPYC 9534, one thread) | 0.6 x r_M x 4e7 |
  |---|---|---|---|---|---|
  | 0.20 | 1 | 1.0e-6 (0.2e-6, 5.7e-6) | 11 | 76 | 24 |
  | 0.25 | 5 | 5.0e-6 (2.1e-6, 1.2e-5) | 63 | 98 | 120 |
  | 0.30 | 30 | 3.0e-5 (2.1e-5, 4.3e-5) | 216 | 121 | 720 |
  | 0.35 | 114 | 1.14e-4 (0.95e-4, 1.37e-4) | 680 | 149 | 2,736 |
  | 0.40 | 283 | 2.83e-4 (2.52e-4, 3.18e-4) | 1,531 | 169 | 6,792 |

  So p_low = 0.35 %. At 0.30 % a confirmation of 4e7 shots would hold about 720 Cascade failures, too few for the
  non-inferiority test; reaching 0.30 % would take about 7e7 shots. The review kept the rule's constants.

  The constants are planning assumptions that only pick p_low (review finding 7). The ratio 0.6 extrapolates the paper's
  fitted Lambda values without their fitted prefactors, so it cannot establish Cascade's failure rate on our circuits;
  1,200 failures give power 0.9 only if the discordant rate is about 0.7 of Cascade's failure rate. The confirmation
  is sized from the discordances measured on the development block (section 4.4); these constants do not enter it. A
  comparison that the cap leaves underpowered is reported as inconclusive (section 4.5). With the family of forty
  hypotheses (4.5) the same power needs about 11 % more failures, about 1,330; 0.35 % still meets that (2,736) and
  0.30 % still does not (720), so p_low stays.

ConvLines was trained at p = 0.6 % only, so at p_low it runs outside its training regime; the report says so.
Cascade-surface sees 0.2-0.7 % during its curriculum, and the paper evaluates its 0.7 % models at 0.2 %.

### 2.3 Training p

The paper's curriculum: a brief stage at a low p1, a linear anneal to the target p2 = 0.7 % (the surface codes'
target), then p2; the first two stages take at most 2 % of the steps. p1 and the split are not stated. As on BB72:
p1 = 0.2 % for the first 1 % of the steps, then the anneal over the next 1 % (step 800 to 1,600 of 80,000).

On our circuits, 0.7 % is near correlated PM's threshold. At 0.6 % its Lambda is only about 1.5 (the release's d = 11,
13, 15 and 17 counts: 9,325, 6,549, 4,311 and 2,943 per 1,000,000), and at d = 13 it fails on 1.8 % of the shots at
0.7 % (355 of 20,000 in the pilot's monitor) against 0.62 % at 0.6 % (123 of 20,000). Training there follows the
paper and the authors' instruction. The monitor tracks both p.

## 3. The network

### 3.1 What is reproduced

`cascade_compare/surface/surface_net.py`, from the BB re-implementation (`ising_compare/bb/cascade_net.py` on
`ising-cascade-net`, 64b2e25d). It keeps the BB embedding, block, residual and read-out, and uses the paper's
surface-code convolution in place of the Tanner-graph one. Every choice:

| item | the paper (Methods, Extended Data Fig. 1, ED Fig. 4, SI) | Cascade-surface | fidelity |
|---|---|---|---|
| input | "a 3D tensor of shape (R, d+1, d+1)"; a learned embedding of each detection event | the release's grid (`weights/grid_d13.npz`, the geometry ConvLines reads): (T, d+1, d+1) with T = R + 1 = 14. X and Z alternate on the plaquette checkerboard; t = 0 holds X only, t = 13 the boundary X layer from the data readout. Tokens (X, 0), (X, 1), (Z, 0), (Z, 1) and 'absent' (a place without a detector) through `nn.Embedding(5, H)` | implementation choices: T = R + 1 = 14 instead of R, so every detector has its own place (the paper's own d = 19 count, 6,840 = 19 x 360 positions, equals the detector count of this layout), and the absent token (ours, from the BB re-implementation) |
| convolution | "a standard 3D convolution with zero padding at boundaries", K = 27 | 3 x 3 x 3, stride 1, zero padding, bias-free, at width H/4 inside the bottleneck | literal |
| block | BN -> SiLU -> project H -> H/4 -> BN -> SiLU -> conv -> BN -> SiLU -> project H/4 -> H | the same; projections are bias-free 1 x 1 x 1 convolutions; channels-last 3D tensors | literal |
| residual | "weighted by 1/sqrt(2L)", drawn on the skip path | h / sqrt(2L) + f(h) (`--res skip`) | literal; BB pilot's choice |
| read-out | "a convolution scatters the check-node representations to data qubits", the average over the observable's support, a two-layer MLP (hidden 2H), one logit | data qubit (r, c) reads the four plaquettes around it at times t and t + 1 (a 2 x 2 x 2 kernel, H -> H). The average runs over the 13 data qubits of the observable (data row 0, derived from the circuit's gates and checked against the release's line geometry) and all 14 layers; then H -> 2H -> 1 with SiLU. Computed as pooling then one product (equal to the literal form to 4e-10) | structure literal; the eight-tap scatter (2 x 2 x 2) is an implementation choice (not stated). The 4e-10 equality checks our pooled form against our literal form: it validates the optimisation, not the choice's fidelity to the paper |
| depth | "L ~ d"; L = d in ED Fig. 4 and the SI | L = 13 | literal |
| width | not stated for the Fig. 3 models; H = 256 in the SI's d = 19 roofline and in the ED Fig. 2b ablation (d = 15, L = 8); "saturates ... for H >~ 64" in the capacity study (d = 15, L = 8, data-level noise); H = 128 as a "deployment-sized" example | H = 256 | inferred (3.2) |
| optimisers | Muon 3e-3 for the convolution weights, Lion 2e-4 for the rest, weight decay 3e-3, EMA 0.9998, MuP mentioned | the same rates for every parameter, no width scaling (`--mup-base 0`); Muon on the blocks' 1 x 1 x 1 and 3 x 3 x 3 weights flattened as (out, in k^3) | literal rates; MuP disabled, an implementation choice (as on BB72, the BB pilot's literal arm) |
| schedule | warm-up 1,000 steps, cosine to 1/10 of the peak at 50,000, batch 3,328, 80,000 steps, "converge within 3 x 10^8" | the same, then 1/10 of the peak up to 120,000 steps (section 3.3) | literal; the extension is ours |
| precision | bf16 mixed precision, gradient clipping | bf16 autocast, fp32 weights and residual stream, clipping at norm 1.0 | clip value inferred |
| normalisation statistics | not stated | BN per micro-batch of 832 shots (per step two micro-batches on each of two GPUs, or four on one) | implementation choice (not stated) |
| initialisation, augmentation | not stated | fan-in normal for convolutions, zero last MLP layer; no augmentation | ours (BB re-implementation) |
| data | Stim, R = d, circuit-level depolarizing noise, on the fly | Stim on our circuits, on the fly, 2 sampler processes (123,000 shots/s each, far above the GPU's need) | circuits differ (section 2.1) |

The review found the architecture and recipe broadly faithful (finding 2): L = 13, the zero-padded 3D bottlenecks,
the skip scaling, the pooling over the observable, Muon and Lion, the EMA, batch 3,328, bf16 and the curriculum are
supported by the paper. Five items are implementation choices, and the report lists them as such: T = 14 time
layers, the absent token, the eight-tap scatter, BatchNorm statistics per micro-batch, and MuP disabled. The read-out
equality check validates our optimisation of the read-out, not whether these choices match the authors' code.

### 3.2 The choice of L and H

L = 13 = d follows the paper's rule; the receptive field (2L + 1 = 27 in each direction) covers the 14 x 14 x 14 grid.

The paper does not state the width of its surface-code models, so H = 256 is inferred. The report calls
Cascade-surface an explicitly inferred, strong local reproduction of the paper's network. It does not claim that our
network is as accurate as the paper's or as a wider one would be (review finding 1). The evidence for the choice:
- The paper's own surface examples use H = 256: the SI's d = 19 roofline and the ED Fig. 2b ablation (d = 15, L = 8).
  It names H = 128 a "deployment-sized" model.
- Its capacity study (d = 15, L = 8, data-level noise) finds that the exponent of the failure rate saturates from
  about H = 64. Wider models still lower the prefactor, so the study does not show that H = 512 would be no more
  accurate than H = 256. A wider Cascade may fail less often than ours.
- Training cost cannot identify the authors' width. At d = 13 our network trains at 3,049, 1,505 and 703 samples/s
  for H = 128, 256 and 512 on a full RTX PRO 6000 (section 6.1). Extrapolated to d = 19, L = 19 (4.26 times d = 13's
  work per sample at the same arithmetic rate; d = 13 and d = 17 share that rate at H = 256) and converted with the
  estimated H200 ratio of 2.0-2.7 (section 6.1), 3e8 samples cost about 43-58 H200-hours for H = 128, 87-118 for
  H = 256 and 187-253 for H = 512. The paper's "approximately 200 H200 GPU hours" for d = 19 fits H = 512 at our
  efficiency or H = 256 at about half of it. The authors' efficiency is unknown, so either fits.

We keep H = 256, which the review judged defensible (H = 512 is not mandatory). It is the width of every surface
example the paper gives. G1 (section 3.3) requires it to beat correlated PM on the monitor block, so it is a strong
opponent, and it saves about 56 RTX PRO 6000-hours of training against H = 512 (49.1 against 105.1 per 2.66e8
samples). For the speed half of the claim it is the harder opponent, since H = 512 would run about twice as slow. For
the accuracy half it may be the easier one. No H = 512 arm is run. The report states the consequence: every verdict
is a verdict against this reproduction at H = 256, and a wider network might be more accurate and slower.

A short pilot cannot rank widths by accuracy: a few thousand steps do not predict the converged order, the lesson of
the BB pilot. The pilot therefore checked that H = 256 trains and measured the cost of each width; it did not select H
by accuracy.

### 3.3 The training run (CS13)

- Run `CS13`, run id 1 (training seeds `train_seed(13, 1, rank, step)`), `--d 13 --L 13 --H 256 --res skip
  --mup-base 0 --compile --workers 2`, two micro-batches of 832 per rank on two GPUs (or four on one). The job file
  `surface/jobs/cs13_job.sh` sources `surface/jobs/cs13.sh`, which holds the arguments and the chain's contract.
- The paper's schedule over `--steps 80000`; `budget_ema.pt` keeps the EMA at the fixed budget (2.66e8 samples). The
  run then continues at the floor learning rate and p2 up to `--max-steps 120000` (4.0e8 samples), a bounded extension.
- The monitor: a fixed block (seed kind 1: chunks 100-129 at 0.6 % = 300,000 shots, 100-109 at 0.7 % = 100,000), read
  with the EMA every 1e7 samples, at the budget and at the end. The raw weights are read too, but only below 8e7
  samples (`--raw-until 8e7`: the points k = 1-7, up to step 21,034). Up to there they are the early-learning
  diagnostic, because the EMA still holds much of the initial weights (0.9998^step: 1.5 % at step 21,034, 0.8 % at the
  first EMA-only point, step 24,039). After that they add nothing, and the monitor reads the EMA alone (review finding
  11). That saves about 0.3 wall-hour (0.6 allocated GPU-hour) up to the budget and more with the extension. Correlated
  PM decodes the block once (`--monitor-pm`).
- The stopping rule, `--no-improvement 3e7:0.02` (BB72's C14 rule with a smaller cap): past the budget, at every
  monitor point, the run stops when the EMA failures fell by less than 2 % at both p since the point 3e7 samples
  earlier. A rise also stops it, so its result is "no observed improvement" (status `no_improvement`), and the report
  does not call it convergence (review finding 10). Otherwise the run ends at 120,000 steps (status `completed`). The cap
  is 120,000 steps rather than BB72's 160,000 because the paper converges within 3e8 samples and each step costs 1.1 s
  on two RTX PRO 6000.
- Two full RTX PRO 6000 of one node with DDP (world size 2, two micro-batches of 832 shots per rank, so the BN batch is
  the one-GPU run's). The planning check (job AJ20) measured 2,997 samples/s, 1.99 times one card, at 63 GB per
  GPU, and a two-GPU run paused at step 30, resumed at step 30 and paused again at step 60. 80,000 steps take 24.7 hours
  wall and 49.4 GPU-hours: DDP saves wall time, not GPU-hours (review finding 10). The extension can add up to another
  24.7 GPU-hours.
- The chain, submitted on 2026-10-10 with `scripts/chain.sh` (tree `code.20261010_180639`, commit b9488934): three
  gpu-normal jobs of 24 hours on two full RTX PRO 6000 of one node (`-p artxpro6000 --gres=gpu:rtx_pro_6000:2 -c 32
  --mem=200G`), each after the previous one ended successfully (`--dependency=afterok`, `--kill-on-invalid-dep=yes`).
  gpu-long (7 days) shares one cap of five RTX PRO 6000 among all its users, so the run does not rely on it. Before
  s1 started, its job file was re-pinned to tree `code.20261010_212737` (commit 917ad20b). The trainer, `jobs/cs13.sh`
  and `scripts/gpu.sbatch` are the same in both trees; the new one adds a guard to the chain check, `stream_shots` and
  a size limit in `eval_shots` that the monitor's blocks stay below (`surface/W5FIX.md`).

  | job | name | what |
  |---|---|---|
  | AJ25 | cs13_s1 | the chain check (`jobs/cs13_check.sh`, below), then CS13 from step 0 until 10 minutes before the job's end |
  | AJ26 | cs13_s2 | resumes, passes the budget and ends the run (stopping rule or 120,000 steps) |
  | AJ27 | cs13_s3 | spare; cancelled when DONE is written |

  Expected end: at submission (19:14 MDT on 10 October) Slurm put s1's start at 23:32 MDT that day. At 22:23 MDT,
  after another user's six one-card jobs of higher priority entered the queue, it moved s1 to 23:34 MDT on 11 October
  (node gpu-node-06). s1 runs the check (about 20 minutes) and then about 75,000 steps. s2 resumes when two cards of
  one node are free after s1's end (about 23:30 MDT on 12 October at the earliest), passes the budget about 1.6 hours
  later and stops at the first point with no observed improvement (step 81,130 at the earliest) or at 120,000 steps,
  about 14.5 hours after its start. If s1 starts as planned, the run thus ends between about 01:30 and 14:00 MDT on
  13 October, later by any wait between the segments.
- The contract in the run directory `$R/ckpt/CS13` (review finding 9; `jobs/cs13.sh`). The trainer writes `status.json` at
  every end (`paused`, `completed`, `no_improvement` or `nonfinite`, with the start and end step, the job and the host)
  and never writes a checkpoint from a non-finite state: before every write it checks the loss accumulated since the
  last log line, the weights and the EMA. A segment runs only if neither `DONE` nor `ATTENTION` exists, and resumes
  `last.pt` only after an intentional pause (`status.json` says `paused`). Any other end writes `ATTENTION` with the
  reason, the status and the job log's last lines, cancels the chain's later jobs and exits 1, so that their afterok
  dependencies lapse as well: a non-finite loss, a crash or a kill that left no status, a pause without progress, an
  inconsistent exit code. A run that has ended (`completed` or `no_improvement`, `final_ema.pt` written) gets `DONE`,
  with the checkpoints' sha256, and the spare is cancelled. `segments.log` records each segment's start and end.
  torchrun turns every non-zero exit into 1, so the chain decides from the status file alone. A failing interval
  therefore costs at most one segment; the review estimated up to one allocated GPU-hour per retry without the contract.
- The checks of the contract. Two Slurm chains ran on a MIG slice (world size 1, run id 903, the monitor on dry-run
  shots of seed kind 13; jobs AJ21-AJ22 and AJ23-AJ24). The first paused at step 30, resumed and paused
  at 60, met a forced non-finite loss at step 65, wrote ATTENTION and cancelled its fourth job before it started. The
  second paused, then stopped with no observed improvement, wrote DONE and cancelled its third job. s1 repeats the
  checks on the production path (two cards, torchrun) before CS13's first step: pause, resume, the forced non-finite
  loss, refusals with and without ATTENTION, a crash on rank 0, and DONE. If an expectation fails, s1 writes
  ATTENTION in CS13's run directory and the chain stops before CS13 starts. `surface/W5FIX.md` has the records.
- Fallback: world size 1, four micro-batches of 832 (63 GB), on whichever full H200 or RTX PRO 6000 starts first, but
  only if a one-card chain is expected to reach the budget sooner. One card needs about 24 hours more than two, so its
  first segment must be expected to start about a day before the two-card chain's. The earlier trigger (two cards
  not started within 6 hours of submission) ignored that difference. At 22:24 MDT on 10 October `sbatch --test-only`
  put a full H200 at 20 October, and a one-card RTX PRO 6000 job queued behind s1 at 23:32 MDT on 12 October; the
  two-card chain stays. Segments may move between the two models. This concerns the training only; every evaluation
  step runs on one GPU model (section 5). Both node types sample identical shots (the same Stim fingerprint,
  21335d99f1b86f0b, on EPYC 9534 and 9555), so a resumed segment continues the same stream. The world size chosen at
  the start stays for the whole run; `surface_train` refuses a resume with another. To switch before s1 starts: cancel
  the chain, move `$R/ckpt/CS13/chain` aside, and submit a one-card chain with `W=1` in the installed job file.
- The network for every read: `final_ema.pt`, the EMA at the run's end (stopped with no observed improvement, or at
  120,000 steps). All reported results use the EMA weights, as in the paper; `budget_ema.pt` is described beside it.
- The pilot's checkpoint is not reused. CS13 starts at step 0 with its own run id, so that one configuration defines
  its shot stream and schedule from the start.

Gate G1, after training and before any development shot: on the monitor block, Cascade-surface's EMA must fail less
often than correlated PM at both 0.6 % and 0.7 %. If it does not, the run is reported to the authors and nothing
further is read; the experiment's premise (a strong fallback) would not hold.

## 4. The chain, the baselines, the thresholds and the statistics

### 4.1 Decoders

- S: ConvLines `d13_b5_r775` (release `weights/d13_b5_r775.safetensors`, sha256 cbd2b84a...). It is read and timed
  through the release's deployed Triton engine (`timing/engine.py`: the path behind the registered decisions, fp16
  matmul inputs, fp32 accumulation, detector bits (B, m), so batched too). Confidence |c|, decision c > 0. Each of the
  engine's programs works on one shot, and the read-out sums the lines in a fixed order, so its outputs should not
  depend on the batch size; both batch sizes are read all the same (4.4).
- B: Cascade-surface `final_ema.pt` of CS13: eval mode (BN running statistics), bf16 autocast, on the frozen paths of
  section 5 (CUDA graphs at batch 1, compiled at larger batches). Confidence |z|, decision z > 0. bf16 tiles depend on
  the batch size, so B at batch 1 and B at a large batch are two computations (phase C saw 34 of 20,000 decisions
  change between batch 1 and 8,192), and each gets its own confirmation outputs (4.4).
- M: correlated PyMatching 2.4.0 through the release's matcher. The DEM of `circuit_at(13, p)` with
  `decompose_errors=True, approximate_disjoint_errors=True`, `Matching.from_detector_error_model(dem,
  enable_correlations=True)`; `decode(det, enable_correlations=True)` one shot per call in single mode,
  `decode_batch(det, enable_correlations=True)` in batched mode.
- Context only, no verdict: plain PyMatching (the same DEM without correlations).

Routing sends a shot to the next stage iff its confidence is strictly below the threshold. The float32 score is
compared in float64, as in the release's `decoder.route` and in phase C, and a shot exactly at tau stays.

### 4.2 Configurations, each at both test p

| | configuration | thresholds |
|---|---|---|
| C1 | S -> B -> M | tau1 on S's |c|; tau2 on B's |z| among the shots S routes (rule 4.3) |
| C2 | B alone (Cascade alone) | none |
| C3 | M alone (correlated PM alone) | none |
| C4 | S -> M, the registered configuration | the registered tau = 1.8486117124557495 at both p |
| C5 | S -> B | tau1 (rule 4.3) |

C4 keeps the registered threshold at p_low as well: it is the registered decoder as deployed, and the report gives its
routed fraction there.

### 4.3 The threshold rule (fixed here; fitted on the development block of each p)

1. Candidates for S's threshold: tau1(q1) = the q1-quantile of S's |c| over the development block (its development
   read, 4.4; numpy.quantile, linear), for q1 in {1, 2, 5, 10, 20, 40, 60} %. Phase C stops at 40 %; 60 % is added
   because ConvLines costs about a fifteenth of Cascade per shot at batch 1 here, so heavier routing can still be
   faster (the cost cap of step 4 removes it otherwise).
2. For C1 only, Cascade's threshold: for each q1 and each r in {0.1, 0.25, 0.5, 1, 2} % with r < q1, tau2(q1, r) is
   the quantile of B's |z| over the shots tau1(q1) routes at which a fraction r of all development shots reaches M.
   This is phase C's construction (`pc_route.py`), with r extended to 2 % because correlated PM costs 0.15-0.3 ms per
   shot (section 2.2; the release's 0.28 ms at 0.6 %) where Relay-BP cost 80 ms.
3. Predicted cost of each candidate X (phase C's model, with matching's cost taken over X's own routed shots):
   - single shot: L1 = t1(S) + f_B t1(B) + f_M t_M(X);
   - amortized: A = a(S) + f_B a(B) + f_M t_M(X) / W.

   Here t1 is the mean single-shot latency on the frozen single path and a the time per shot on the frozen batched
   path (C3, section 5); f_B and f_M are the fractions of the development block that X's thresholds route to B and to
   M (measured, so shots tied at a threshold count as routed); t_M(X) is correlated PM's mean single-threaded decode
   time over the shots that X routes to it, from the per-shot times of the development read on the timing node's CPU
   (4.4). Matching's mean over all shots would understate it: the routed shots are the least confident ones, with more
   detection events, and take longer to match, which can change a candidate's feasibility (review finding 8; phase C
   takes Relay-BP's time over each candidate's own routed set the same way). W = 16 is the number of CPU workers of
   the batched pipeline. The model only ranks candidates; every speed verdict rests on the end-to-end measurement of
   section 5.
4. Choice, phase C's rule. Among the candidates with L1 <= 0.9 L1(B) and A <= 0.9 A(B), take the one with the fewest
   development failures; ties go to the lower L1. If no candidate is feasible, take the fewest failures and flag it
   "predicted slower than Cascade". The choice is made for C5 over q1 and for C1 over (q1, r), separately at each p.
5. The lock, `cascade_compare/surface/out/locked_d13.json`, holds:
   - every candidate's development counts, routed fractions, t_M(X), predicted costs and feasibility;
   - the chosen thresholds;
   - the margins delta_p (4.5), the planned confirmation sizes, every comparison's planned power and cap-limited flag,
     and each chain's joint power (4.4);
   - the frozen timed paths (with B's batch b*), the computations of 4.4 with their code paths and settings, the GPU
     model, driver and venv of section 5, and every network's and input's sha256.

   It is committed to the hub before any confirmation shot is drawn.

### 4.4 Blocks, reads and the outputs behind each claim

Every read streams its block (review finding 12). `surface_data.stream_shots` yields one chunk of 10,000 shots at a time,
sampled from its seed; the decoders decode it and its per-shot outputs are written before the next chunk is drawn, with
a few chunks (under 0.1 GB) in memory. No read holds a block as one array: the detector bits of a 4e7-shot block take
87 GB (2,184 bytes per shot), about 175 GB while concatenating, against 22 MB per chunk. `eval_shots`, which
concatenates, refuses more than 1,000,000 shots. The batched reads carry S's batches, B's queue and C2's batches
across chunk boundaries, so that they form exactly the batches of the pipelines.

- Development block, seed kind 10: 1,000,000 shots at 0.6 % (chunks 0-99 of 10,000) and 10,000,000 at p_low = 0.35 %
  (chunks 0-999). The expected failures: about 4,000 for Cascade at 0.6 %, and at 0.35 % about 1,140 for correlated PM
  and about 700 for Cascade at the ratio 0.6. Each decoder reads it once through its batched computation (S on the
  engine at 4,096, B at its frozen batch b*), and M's `decode` times each shot single-threaded on the timing node's CPU
  (t_M, 4.3). The development block chooses thresholds and sizes the confirmation; no claim rests on it. Its per-shot
  outputs go into the lock's inputs.
- Confirmation block, seed kind 11, read once after the lock commit. Its size at p is the smallest n on phase C's chunk
  grid (10, 20, ..., 2,000 chunks of 10,000, extended by 3,000 and 4,000 at p_low) for which:
  - the non-inferiority tests of C1 and of C5 against C2 (4.5), at the Bonferroni share alpha / 40 one-sided, have
    power >= 0.9;
  - the power is computed for a true difference of zero and the discordant rate measured on the development block
    between the chain and C2 (review finding 7), simulated as in phase C's `bb.route.power` (its fixed seed) with the
    non-inferiority decision in place of the superiority test. The development read is batched, and the single-mode
    tests use the same rate, since the two modes differ only by bf16 rounding;
  - n lies between the development size and the cap: 4,000,000 shots at 0.6 % and 40,000,000 at p_low.

  A comparison whose power stays below 0.9 at the cap is flagged cap-limited in the lock, with the power reached, and
  if nothing is rejected its label is "inconclusive" (4.5). Each p is sized for 0.9 on its own. The two p are
  independent blocks, so a chain planned at 0.9 at both has a planned probability of about 0.81 of passing both, which
  the headline needs (section 7); the lock records each chain's joint power, the product of its two planned powers.
  The two modes read the same shots and add no factor.

The outputs behind each claim (review finding 5). Phase C's final design evaluates each decoder through the path it is
timed on; here that is carried to every computation. A claim about a configuration in a mode rests only on
confirmation outputs of the computations that mode times. A computation is fixed by its decoder, code path, batch size
and batch composition, precision, settings, GPU model and venv (section 5). The confirmation reads:

| computation | mode | what runs | shots |
|---|---|---|---|
| S1 | single | S on the engine at batch 1 | every shot |
| SB | batched | S on the engine in batches of 4,096 consecutive shots | every shot |
| B1 | single | B on its frozen single path (CUDA graph at batch 1) | every shot |
| Bb | batched | B on its frozen amortized path, in batches of b* consecutive shots | every shot |
| BQ1, BQ5 | batched | B at 4,096 on the queue of shots that SB routes under C1's (C5's) tau1, in order, the last batch padded | those shots |
| MS | single | M's `decode`, one shot per call | every shot |
| MB | batched | M's `decode_batch` | every shot |

Each configuration's decisions in a mode are assembled as its pipeline makes them, with the locked thresholds applied
in float64 to that mode's confidences:

| configuration | single mode | batched mode |
|---|---|---|
| C1 | S1; B1 on the shots S1 routes; MS on the shots B1 routes | SB; BQ1; MB on the shots BQ1 routes |
| C2 | B1 | Bb |
| C3 | MS | MB |
| C4 | S1; MS on the shots S1 routes | SB; MB on the shots SB routes |
| C5 | S1; B1 on the shots S1 routes | SB; BQ5 |

- Screens are diagnostics; they do not replace these outputs. On a 100,000-shot screen at p_low Cascade fails about
  seven times, and phase C's count tolerance (`agrees`) would accept up to nine more failures of a timed computation
  (9 <= 0.07 + 3 sqrt(9)), far more than the non-inferiority margin allows.
- Where two computations of one decoder agree bitwise on every confirmation shot (S1 and SB should, and so should MS
  and MB, since PyMatching decodes each shot on its own), the report says so. Each mode still uses its own outputs.
- A computation must be reproducible: its outputs on the same shots are bitwise equal in every process, so that its
  read and its timings describe one computation. The paths run with TF32 off, cuDNN's autotuning off
  (`cudnn.benchmark = False`) and torch.compile in its default mode, and C3 freezes only paths whose outputs agree
  bitwise across their three processes (section 5).
- Each pipeline process (C7) saves its decisions and confidences on its timed shots, and the analysis re-runs the same
  computations on those shots. If any shot differs, the mode is flagged and its claims are withheld (the timed
  computation is not the confirmed one). The comparison is bitwise.
- Per-shot outputs are saved as npz per computation with their sha256, and one analysis runs (`confirm`). It refuses
  outputs whose network sha256, computation, GPU model, venv or thresholds differ from the lock.
- Cost (review finding 12): B1 is the large read, B at batch 1 on up to 4.4e7 shots. At the pilot's ungraphed 948 µs per
  shot it would take 11.6 GPU-hours. C3 measures the graphed rate, the lock records the budget, and the read may be
  split by chunk ranges over two cards of the one model. The batched reads of B over the 5.5e7 development and
  confirmation shots take about 2.1 GPU-hours (section 6.2).

### 4.5 Statistics: one alpha budget

Paired on the same shots. At each p and in each mode (single shot, batched; each on its own outputs, 4.4) the family
holds the ten hypotheses below, forty over both p and both modes, judged by one Holm procedure at alpha = 0.05:

| | hypothesis | test |
|---|---|---|
| N1 | C1 non-inferior to C2 | one-sided; H0: P(C1) - P(C2) >= delta_p |
| N2 | C5 non-inferior to C2 | as N1 |
| S1-S8 | no difference between: C1-C2, C5-C2, C4-C2, C3-C2, C1-C4, C5-C4, C1-C5, C4-C3 | two-sided exact McNemar on the discordant pairs |

- delta_p = 10 % of B's block failure rate on the development block at p (phase C's margin), frozen in the lock, the
  same in both modes. It is the predefined accuracy tolerance: a non-inferior chain may fail up to 10 % more often
  than Cascade alone.
- The non-inferiority p-value inverts Newcombe's matched-pair interval (method 10, phase C's `bb.route.paired`). It is
  the one-sided level at which the interval's upper end equals delta_p, found by bisection, so the decision equals "the
  upper end at the Holm level lies below delta_p".
- Labels of X against Y (review finding 3):
  - "better": S rejected by Holm, and the upper end of Newcombe's interval for P(X) - P(Y) at the Holm level lies
    below zero;
  - "worse": S rejected by Holm, and the interval's lower end lies above zero. A "worse" X gets no other label: a
    chain that is statistically worse than Cascade is never "non-inferior" or positive, even when its N is rejected;
  - "non-inferior": N rejected by Holm, and not "worse";
  - "inconclusive": a cap-limited comparison of a chain with C2 (4.4) in which nothing was rejected;
  - "level": otherwise, meaning neither direction was shown (it does not mean equivalence).
- Phase C keeps non-inferiority in a separate family. Here it enters the one family, because accuracy within the
  tolerance is half of the authors' claim (section 7).
- Also reported, without tests: Wilson 95 % intervals on every P_block; P_L per round by the paper's Eq. (1) with
  k = 1 and R = 13, P_L = (1 - (1 - 2 P_block)^(1/13)) / 2. On the shots S keeps: B's failures that S repairs and S's
  failures that B would not have made. On the shots M decides: M's repairs and regressions.

### 4.6 Departures from phase C

- No accuracy curves over a p grid. Two test p, chosen as in section 2.2.
- Non-inferiority enters the one Holm family (above), and the family covers both modes, each on its own outputs.
- q1 extends to 60 % and r to 2 % (4.3).
- The confirmation is sized for non-inferiority, not for a 10 % improvement. At equal discordance the two sizes are
  close.
- The claim is "faster within the predefined accuracy tolerance", or "faster and more accurate" where the chain is
  "better" (section 7); phase C's is "better and faster".
- Every computation behind a claim gets its own confirmation outputs (4.4). Phase C's screen and claim-point check
  compare timed paths with the read by counts; here the screen is a diagnostic, and the claim-point check is replaced
  by the confirmation reads and the bitwise check of the pipelines' outputs.

## 5. Speed

Phase C's final protocol (`PHASE_C_PLAN.md` section 5 at d9b0b899, `pc_speed.py`) with the surface decoders:
- One GPU model and one software environment for every GPU step of the evaluation (review finding 6): the speed
  screen C3, the development reads, the confirmation reads, the pipelines C7 and every diagnostic run on a full RTX
  PRO 6000 Blackwell Server Edition (96 GB; partition `artxpro6000`, GRES `gpu:rtx_pro_6000`, never a MIG slice) with
  bb's `venv_cu128` (torch 2.11.0+cu128) in the pyfull container, in jobs that hold the whole card. There is no H200
  path: bf16 tiles and compiled kernels depend on the GPU, so a read and a timing describe one computation only on one
  model. Every output records the card, its driver and torch, and the analysis refuses a read or a timing made on
  another card or venv. TF32 off, cuDNN autotuning off (4.4). M runs on the same node's CPU (AMD EPYC 9555),
  single-threaded per decode, `OMP_NUM_THREADS=1`, in the same container and venv, for its development and
  confirmation decodes as well as its timings. (CS13's training may run elsewhere; it only produces the network.)
- Single shot: the GPU term is CUDA-graph replay at batch 1 (200 warm-up replays, then 5 x 100 timed with CUDA
  events). The host round trip is measured over 10,000 shots after 200 warm-up: pinned H2D copy, replay, D2H copy,
  synchronize, `perf_counter`. Reported: mean, p50, p99, p99.9 and max. M single-shot is `decode(syn,
  enable_correlations=True)` in-process (the release's `timing/cpu_term.py`).
- Amortized: batches 1, 16, 64, 256, 1,024 and 4,096, each timed for at least 2 s after 10 warm-up replays, as
  wall time / (batches x B), for resident and host-to-host data. Per round = per shot / 13.
- Repetition: three fresh processes per network or pipeline, interleaved; the value is the slowest process's
  mean.
- The speed screen (C3, before the development reads): every path of S and B at every batch size, in three processes.
  It freezes, for each network, the fastest reproducible paths (outputs bitwise equal across the three processes,
  4.4): single (batch 1) and amortized (the lowest host-to-host time per shot over the batch sizes up to 4,096; its
  batch is b* for B). As a diagnostic it also compares every path with its network's path at 4,096 on 20,000 to
  100,000 kind-12 shots: the decisions and confidences that differ, and phase C's `agrees` (|only_t - only_e| <=
  0.01 n_e + 3 sqrt(only_t + only_e)) on the failures at 0.6 % and on the routed sets at each candidate quantile.
  These numbers are reported; they flag bugs before the lock and establish no accuracy (review finding 5).
- Decoders alone: C3 is M alone, single shot in-process on one core, batched over the 16 worker processes of the
  pipelines (wall time / shots).
- Pipelines (C7: C1, C4, C5 and Cascade alone, C2, at both p), on fresh shots (kind 12), in one job and one rep-major
  loop, three fresh processes per configuration and mode:
  - single mode: 20,000 shots after 200 untimed. S at batch 1, then B at batch 1 if routed, then M in-process if
    routed. C2: B alone on its frozen single path.
  - batched mode: 500,000 shots. S runs in batches of 4,096; routed rows queue on the GPU and B runs them at 4,096,
    the last batch padded; M runs in 16 worker processes with pre-built matchers. C2: B alone on its frozen
    amortized path, batches of b*, its best eligible batch size.
  - Cascade alone is timed beside the chains in the same harness and job, as in phase C's final design (review finding
    6), and each chain is compared with that C2 (section 7). It adds about 0.15 GPU-hour.
  - Included: transfers, thresholding, queueing, fallback time. Excluded: Stim sampling and matcher construction,
    reported separately.
  - Each process saves its decisions and confidences on its timed shots for the bitwise check of 4.4.
- No claim-point check by counts: the confirmation reads every computation that a claim rests on (4.4).
- Context columns: the paper's single-shot "~40 µs per cycle" on an H200 (its precision and method not stated).
  The release's ConvLines composite of 57.96 µs per shot (H100 SXM, batch 1, its study-2 protocol).

## 6. Compute

### 6.1 Measured (section 8; batch 3,328, p = 0.7 %, compiled, 2-4 Stim processes, 20 timed steps)

Full RTX PRO 6000 Blackwell (96 GB), job AJ17:

| d | L | H | micro-batches | samples/s | GPU-hours per 2.66e8 samples | per 3e8 | memory |
|---|---|---|---|---|---|---|---|
| 13 | 13 | 256 | 4 | 1,505 | 49.1 | 55.4 | 63 GB |
| 13 | 13 | 128 | 2 | 3,049 | 24.2 | 27.3 | 63 GB |
| 17 | 17 | 256 | 8 | 540 | 136.7 | 154.2 | 86 GB |
| 13 | 13 | 512 | 8 | 703 | 105.1 | 118.5 | 63 GB (job AJ19) |

Full H200: not measured. The benchmark job (`pilot/jobs/w5_bench_h200.sh`) did not start within the
planning window; Slurm's start estimate moved to 05:37 the next morning, so it was cancelled. The evaluation has no
H200 path (section 5), so it is not rerun; the estimate below is for the training's fallback only. Estimate: the BB
stack measured an H200 at 4.1-4.4 times half an RTX PRO 6000, which is 2.0-2.2 times a full card if the card scales
linearly, and the memory-bandwidth ratio is 2.7 (4.8 against 1.8 TB/s). With 2.0-2.7 times the full RTX PRO 6000,
d = 13 at H 256 trains at about 3,000-4,100 samples/s on an H200: 18-25 H200-hours per 2.66e8 samples.

Inference, d = 13, L 13, H 256, compiled, bf16, random weights (RTX PRO 6000): 7,370 shots/s at batch 4,096
(136 µs per shot), 948 µs per shot at batch 1 without CUDA graphs (73 µs per round).

### 6.2 The parts

| part | where | hours |
|---|---|---|
| pilot and benchmarks (done) | RTX PRO 6000; acpu | 2.3 RTX PRO 6000-hours (the pilot job, 2 h 2 min; the H = 512 benchmark; the DDP check, 6 min on two cards); 0.4 + 2.5 core-hours (unit checks, p_low rule) |
| p_low rule: correlated and plain PM on 5 x 1,000,000 shots (done) | acpu, 16 cores | 9.5 minutes of a 16-core job (2.5 core-hours) |
| CS13 training to 80,000 steps (2.66e8 samples) | two RTX PRO 6000 with DDP; fallback one RTX PRO 6000 or H200 | 24.7 hours wall and 49.4 GPU-hours on two RTX PRO 6000 (measured; DDP saves wall time, not GPU-hours); 49 hours on one; 18-25 hours on one H200 (estimated); the extension to 120,000 steps adds up to half again (up to 24.7 GPU-hours on two cards) |
| chain checks (done, and s1's check) | MIG slices; s1's two cards | about 0.2 GPU-hours on MIG slices; about 0.3 hours of s1 on two cards (0.6 GPU-hours) |
| monitor during training (400,000 shots every 1e7 samples with the EMA; the raw weights too at the first seven points) | the training GPUs | about 0.5-1 hour wall in total (up to 2 allocated GPU-hours), inside the training time; reading the raw weights at the first seven points only saves about 0.3 wall-hour (0.6 GPU-hour) to the budget |
| speed screen of S and B (C3) | the one GPU model (section 5) | about 2 GPU-hours |
| development reads: 1.1e7 shots, S at 4,096 and B at b* on the GPU, M on the node's CPU with each shot timed | the one GPU model and its node's CPU | about 0.5 GPU-hours; about 1 core-hour |
| confirmation, batched computations (SB, Bb, BQ1, BQ5) on at most 4.4e7 shots | the one GPU model | Bb about 1.7 GPU-hours at 7,370 shots/s; BQ1 and BQ5 together at most 1.2 times that (q1 <= 60 %); SB under 0.5 |
| confirmation, single computations (S1, B1) on at most 4.4e7 shots | the one GPU model, split over two cards if needed | B1: 4.4e7 x t1(B), 11.6 GPU-hours at the ungraphed 948 µs per shot, less at the graphed rate that C3 measures; S1 about 0.7 (about 60 µs per shot) |
| confirmation, M's MS and MB on every shot | the node's CPU | about 4-8 core-hours (0.15-0.3 ms per shot, twice) |
| pipelines (C7): C1, C4, C5 and Cascade alone x 2 p x (single + batched) x 3 processes | the one GPU model and its CPU | about 3.2 GPU-hours (Cascade alone about 0.15 of it) |
| bitwise check of the pipelines' outputs (4.4) | the one GPU model | about 0.2 GPU-hours |
| d = 17, conditional | Alpine, 2-4 GPUs | training about 137 RTX PRO 6000-hours (2.8 times d = 13's); inference and speed about 2-3 times |

The evaluation thus takes about 12-23 GPU-hours on the one GPU model with the confirmation at its caps, most of it the
batch-1 read of Cascade (review finding 12; the earlier estimate of under 10 hours read each decoder once). Stim sampling
is never the bottleneck: 123,000 shots/s per core at d = 13. Shots are regenerated from their seeds and streamed
(4.4), not stored.

Critical path: training (24.7 hours wall on two RTX PRO 6000 to the budget, up to about 38 to 120,000 steps, plus any
wait between segments); in parallel, the experiment code and its dry runs on the pilot's checkpoint (kind 13 shots
only, on the one GPU model) and the fresh check of the evaluation fixes; then the speed screen, development reads,
lock, confirmation reads (B1 the longest, split over two cards), pipelines and report (about 1.5 more days).

### 6.3 Placement

- Alpine GPUs first: CS13 runs on two RTX PRO 6000 of one node with DDP (section 3.3), or on one full H200 or RTX PRO
  6000 as the fallback; every evaluation step runs on full RTX PRO 6000 cards (section 5). W5 uses at most two GPUs at
  a time, so that with phase C's jobs the account stays within four. M's development and confirmation decodes run on
  the CPUs of the RTX PRO 6000 nodes inside the GPU jobs (section 5); other CPU work runs on acpu within QEC's cap of
  80 cores (at most 32 cores for W5).
- Multi-GPU: the stack supports single-node DDP (`torchrun`, NCCL, a separate shot stream per rank, BN statistics
  per GPU). The global batch 3,328 = 2^8 x 13 allows 2 or 4 GPUs. The two-GPU pause-and-resume check passed in planning
  (job AJ20, `surface/jobs/ddp_check.sh`), at 1.99 times one card's throughput.
- RunPod (112 USD left) is used only where it shortens the critical path materially, and is not planned.
  Contingency: if CS13's next segment cannot start on Alpine within 12 hours, the remaining steps run on RunPod with
  the run's world size (one H200 per rank, community cloud, 3.59 USD per hour in October 2026). The whole 80,000-step
  run would take about 18-25 hours on one H200, about 65-90 USD; a remainder costs proportionally less. It shortens the
  path only against a single Alpine RTX PRO 6000 (49 hours), not against two of them with DDP (about 25 hours).
  Expected cost: 0 USD on the default path, at most about 90 USD in the contingency. d = 17 would exceed the budget on
  RunPod (2.8 times d = 13), so it stays on Alpine.

## 7. What the report will state

`cascade_compare/SURFACE_RESULT.md`, with every table generated from the saved per-shot outputs:
1. The setting and its fidelity: our circuits (not the paper's), Cascade-surface's choices (section 3.1 with the
   fidelity column and the implementation choices), its status as an explicitly inferred, strong local reproduction
   at H = 256 (3.2), the training run (steps, samples, monitor curve, and the stopping rule's result: no observed
   improvement, or the 120,000-step limit; never "converged"), G1.
2. For each test p, mode and configuration C1-C5 (plus plain PM as context): failures and shots, P_block with Wilson
   95 % intervals, P_L per round, the fractions routed to B and to M, the forty hypotheses with their Holm labels,
   Newcombe intervals and planned powers (cap-limited ones flagged), and the repairs and regressions of 4.5. Where two
   computations of a decoder agree bitwise, the report says so.
3. Speed for each decoder and pipeline, on the one GPU model and its CPU: mean single-shot latency per shot and per
   round, p50, p99, p99.9, and the amortized time per shot at each batch size, with Cascade alone from the pipelines'
   job beside each chain. The paper's and the release's timings go in separate context columns.
4. The verdict per chain (C1, C5) and p, decided as below, and the d = 17 decision of section 1.

A chain X (C1 or C5) is "faster within the predefined accuracy tolerance" than Cascade alone at p when, in each mode
(single shot and batched), all of the following hold (review finding 3):
1. accuracy, on that mode's confirmation outputs (4.4): X is "non-inferior" to C2 or "better" than C2 (4.5). The
   tolerance is the margin frozen in the lock: the upper end of the Holm-level Newcombe interval of P(X) - P(C2) lies
   below 10 % of Cascade's development failure rate, so X may fail up to 10 % more often than Cascade alone. A "worse"
   X never satisfies this condition, whatever its non-inferiority test says;
2. speed: X's measured pipeline is faster than Cascade alone in that mode: its mean single-shot latency (single mode)
   or its amortized time per shot (batched mode) is lower than C2's, on the one GPU model under the same protocol, C2
   timed in the same job, each value the slowest of three processes (section 5);
3. the bitwise check of that mode passed: the timed computations are the confirmed ones (4.4).

X is "faster and more accurate" than Cascade alone at p when 1-3 hold with "better" in both modes, that is, with the
upper end of the paired interval below zero. No other wording is used for a non-inferior chain: the report prints
"within the predefined accuracy tolerance (up to 10 % more failures than Cascade alone)" beside the label, with the
interval.

The headline "ConvLines with Cascade behind it is faster than Cascade alone within the predefined accuracy tolerance
on the d = 13 surface code" needs one chain to satisfy 1-3 at both p. "Faster and more accurate" in the headline needs
"better" at both p as well. The joint planned power of a chain over the two p is about 0.81 when each p is planned at
0.9 (4.4), and the report states it beside the headline. Otherwise the report lists the partial results: faster but
outside the tolerance, "worse", inconclusive (cap-limited), the reverse, or a mode withheld by the bitwise check.
Beside every verdict it prints p99.9 of the single-shot latency, and says "faster on average, slower in the tail"
where X's p99.9 exceeds Cascade's; a routed shot pays for S and B in sequence. Each verdict names the GPU model, the
venv and the precision of every path (bf16 for Cascade, fp16 inputs with fp32 accumulation for ConvLines' engine,
CPU for PyMatching), and that the opponent is Cascade-surface at H = 256, an inferred reproduction (3.2).

## 8. The pilot

Records: `cascade_compare/surface/pilot/PILOT.md` and its job logs.
- Run `pilot13`, run id 900 (W5's pilot namespace). d = 13, L 13, H 256, `--res skip --mup-base 0`, the paper's
  schedule over 80,000 steps with p2 = 0.7 %, eight micro-batches of 416 shots.
- Paused at step 3,000 (1.0e7 samples). Monitor: 20,000 fresh shots at 0.6 % and at 0.7 % (seed kind 5), raw and EMA
  weights every 500 steps, and correlated PM on the same shots.
- Result: the loss falls and the network learns on our circuits. 113 minutes at 1,510 samples/s; loss 0.035 at the end
  of the 0.2 % stage, then at the fixed 0.7 % from 0.205 (step 1,700) to 0.106 (step 3,000), monotonically. Raw-weight
  failures on the 0.6 % monitor were 8,324, 4,315, 1,196, 536, 401 and 324 of 20,000 at steps 500 to 3,000, against
  123 for correlated PM on the same shots; at 0.7 % the final figure is 762 against 355. At 3.75 % of the schedule this
  is no forecast of the converged accuracy. The EMA still holds 55 % of the initial weights and is not yet meaningful.
- The checkpoint (`ckpt/pilot13/last.pt`, sha256 92e1e334...) stays on Alpine for the experiment code's dry runs. CS13
  does not start from it (section 3.3).

## 9. Order of work and gates

1. An external review of this plan; the dispositions are recorded. Done: VERDICT REVISE, twelve findings, all
   accepted and folded in.
2. The p_low rule: applied during planning (section 2.2); the review kept it.
3. CS13 trains (section 3.3; submitted). Meanwhile the experiment code is written: the streamed reads of every
   computation of 4.4, rule 4.3 with t_M, `confirm` with the labels of 4.5, the speed protocol with Cascade alone in
   the pipelines' job, the bitwise check, the report. Its dry runs use the pilot's checkpoint and kind-13 shots only,
   on the one GPU model.
4. A fresh check of the evaluation fixes, before any development or confirmation read. First pass:
   the stage gate of 2026-10-12 on 2e9a655, REVISE, seven findings, fixed before any read.
5. G1 on the monitor block.
6. Speed screen (C3, freezing the reproducible paths), then the development reads, then the lock commit
   (`surface/scripts/lockpair.sh`, whose receipt every confirmation read requires).
7. The confirmation reads of every computation of 4.4 and the one analysis.
8. Pipelines (C7, Cascade alone in the same job), the bitwise check, the report, the d = 17 decision.

## 10. Seeds, files and records

W5's own namespace (`surface_data.py`), disjoint from every documented one: ising_compare and the release below 2^33;
interp, BB72 and BB144 at 2^40-2^46; W1 at 2^50 + ...; W3 and phase C in [2^55 + 1e15, 2^55 + 6e15); registered test
sets at 2^62 and above.

- `train_seed(d, run, rank, chunk) = 2^56 + 1e15 d + 1e12 run + 1e10 rank + chunk` (run < 1000, rank < 100);
- `eval_seed(d, kind, p, chunk) = 2^56 + 3e16 + 1e15 d + 1e13 kind + 1e7 round(1e6 p) + chunk` (kind < 100,
  chunk < 1e7);
- for d <= 29 everything lies in [2^56 + 3e15, 2^56 + 6e16), below 2^57.

Kinds:

| kind | use |
|---|---|
| 1 | training monitor |
| 2 | unit checks |
| 3 | sampler checks and the host fingerprint |
| 5 | the pilot's monitor |
| 6 | the p_low rule |
| 10 | development blocks |
| 11 | confirmation blocks |
| 12 | screens, checks and pipeline streams |
| 13 | dry runs |

Run ids: 1 = CS13, 2 = the d = 17 run, 900 = the pilot, 901 = the benchmarks, 902 = the DDP check
(`surface/jobs/ddp_check.sh`), 903 = the checks of the job chain. These checks and the dry runs read seed kind 13 only.

Code (`surface/` here, `cascade_compare/surface/` in the research tree; on the cluster under `<cluster-scratch>/cascade_w5`):

| file | content |
|---|---|
| `surface_net.py` | the network |
| `surface_layout.py` | circuits at any p, the grid, the observable's support |
| `surface_data.py` | seeds, the Stim stream, evaluation shots (streamed in chunks of 10,000 for the reads, `stream_shots`), correlated PM |
| `surface_train.py` | `bb/cascade_train.py` with the surface network; the loop, checkpoints, resume rules, monitor and benchmark, with the status file, finite-only checkpoints, the EMA-only monitor after early learning and the no-observed-improvement rule |
| `cascade_optim.py` | a byte-identical copy of `ising_compare/bb/cascade_optim.py` at 64b2e25d, git blob 572c22cd |
| `test_surface.py` | unit checks |
| `plow.py` | the p_low rule (section 2.2) |
| `bench_infer.py` | inference throughput |
| `scripts/` | deploy, submit (pins each job to one deployed tree), lockpair (the lock commit and its receipt), chain (the training's job chain), poll, the Slurm wrappers, the benchmark |
| `jobs/` | CS13's job file (`cs13_job.sh`), its arguments and contract (`cs13.sh`), the chain check (`cs13_check.sh`), the DDP check (run in planning, job AJ20) |
| `out/` | records (`plow_d13.json`; later the evaluation's lock, gate file, gate check and screen summary, from `scripts/lockpair.sh`) |
| `pilot/` | job files, logs, PILOT.md |
| `sv_common.py`, `sv_read.py`, `sv_speed.py`, `sv_route.py`, `sv_test.py` | the evaluation: reads, speed screen and pipelines, threshold rule, gate, `confirm`, report, unit checks |
| `eval/` | the evaluation's job files (`eval/jobs/`), and the dry runs' job logs and records |

The release files the code reads are checked by sha256 at load: circuit 86f1d9e4... (d = 13) and 5a17d282...
(d = 17); grid d1f651ac... (d = 13).
