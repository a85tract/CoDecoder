# Checkpoint lineage of the released networks

For each of the 38 released networks: what it was initialised from, every training stage with its samples, schedule
and data, and the cumulative budget. The first 16 trained with the code in this folder (each stage's command is named
below); `TRAINING.md` says how to run it. The output-head comparison's 21 networks trained
with `../head/` and CS13 with `../cascade/d13/` (last two sections). The sha256 values are those of the original checkpoints, which
`weights/surface/configs.json` (surface code), the BB checkpoints' records and the Cascade comparison's locks (`source_sha256`)
also give. Sample counts are whole batches: "2e8" means 195,312 steps of 1,024, that is 199,999,488 samples.

## Surface code: six networks, thirteen stages

Program: `surface/ising_compare`, `python -m arch.train --phase truth` (see `surface/SOURCE`). The following holds for
every stage:
- Data: fresh Stim samples of the verdict circuit of the stage's distance, `circuits/d{d}_legacy.stim` (NVIDIA
  Ising-Decoding's memory circuit with its legacy noise at p = 0.006). The noise is the same throughout; there is no
  noise curriculum. Run id R fixes the data: 8 virtual Stim streams with seeds 10^8 + 10^6·R + v (v = 0-7), read in
  chunks of 8,192 shots in a fixed round-robin order, so the shot order does not depend on the number of sampler
  processes.
- Schedule: torch seed R. AdamW (0.9, 0.99) with weight decay 0.01 and batch 1,024. The learning rate warms up
  linearly, then follows a cosine down to 2 % of the peak. Gradients are clipped at 1; training uses bf16 autocast.
- Batches: the shots come in pools of 32 batches. Each pool is sorted by fired-detector count, cut into batches, and
  the batches are shuffled with a generator seeded with R.
- Loss: BCE(c, y_0) + mean_j BCE(z_j, y_j), where c is the logit of the standard observable and z_j the logit of
  logical line j.
- Read-out: the EMA of the weights (0.999) at the last step (`final_ema.pt`).
- Growth (`--init-transfer`, `arch.transfer`):
  - Every distance-independent weight is copied.
  - The line read-out's per-line embedding `head.e` and the vote weights `w` are interpolated over the line position;
    `w` is also scaled by d_old/d_new. The time bias `head.b_t` is interpolated over time. The distance-bin bias
    `head.b_line` is padded with its edge values.
  - Added blocks start as the identity (zero-initialised up-projection). Before stages 774, 775, 782 and 787,
    `arch.g6_check` checked the mapping and the trainer's start. At an unchanged distance it also checked that the
    network equals its parent in fp32 and bf16.
- Optimiser and EMA: every stage starts with a fresh optimiser, schedule and EMA. `--init` (stage 782) loads the weights
  as they are, at the same distance.

| stage (run id R) | network | initialised from | samples | peak LR, warm-up steps | command (`surface/ising_compare/scripts/alpine_arch/args/`) | recorded wall time | final_ema.pt sha256 |
|---|---|---|---|---|---|---|---|
| 740 | d = 7, 1 block, width 64, bottleneck 16 | random | 199,999,488 | 1e-3, 1,000 | `d7b1w64_r740.txt` | 1,705 s | caca97c3… |
| 710 | d = 9, 2 blocks, width 128 | random | 199,999,488 | 1e-3, 1,000 | `g2_r710.txt` | 7,183 s | 353cdf7c… |
| 711 | d = 9, 2 blocks | random | 199,999,488 | 1e-3, 1,000 | `g2_r711.txt` | 7,187 s | 135ef76d… |
| 764 | d = 9, 3 blocks | 710, one block added | 199,999,488 | 1e-3, 1,000 | `g6_d9_b3t_r764.txt` | 16,555 s (MIG slice) | 462ffb29… |
| 732 | d = 13, 3 blocks | 710 transferred to d = 13, one block added | 199,999,488 | 1e-3, 1,000 | `d13b3t_r732.txt` | 24,245 s | 05150225… |
| 733 | d = 13, 3 blocks | 711 transferred to d = 13, one block added | 199,999,488 | 1e-3, 1,000 | `d13b3t_r733.txt`; it ran as `d13b3t_r733d.txt` (the same run, two segments on another cluster) | two segments | 68588b00… |
| 735 | d = 11, 3 blocks | 733 transferred to d = 11 | 199,999,488 | 1e-3, 1,000 | `d11b3t_r735.txt` | 15,574 s | 2941b48a… |
| 777 | d = 11, 5 blocks | 735, two blocks added | 199,999,488 | 1e-3, 1,000 | `g8_d11_b5t_r777d.txt` (two segments on another cluster) | two segments | 4b303acd… |
| 774 | d = 13, 5 blocks | 732, two blocks added | 199,999,488 | 1e-3, 1,000 | `g7_d13_b5t_r774.txt` | 32,179 s | eea02c35… |
| 775 | d = 13, 5 blocks | 733, two blocks added | 199,999,488 | 1e-3, 1,000 | `g7_d13_b5t_r775.txt` | 32,186 s | 44b131d2… |
| 778 | d = 15, 5 blocks | 774 transferred to d = 15 | 199,999,488 | 1e-3, 1,000 | `g9_d15_b5t_r778.txt` | 48,182 s | 346ea266… |
| 782 | d = 15, 5 blocks | 778's weights (`--init`) | 99,999,744 | 2e-4, 500 | `g12_d15_b5t_r782.txt` | 24,121 s | 3017a705… |
| 787 | d = 17, 5 blocks | 778 transferred to d = 17 | 199,999,488 | 2e-4, 500 | `g14_d17_b5t_r787.txt` | 69,414 s | 3627e743… |

Unless the table says otherwise, the width is 128, the bottleneck 32 and the read-out has 4 heads. The wall times come
from different GPUs: 764 ran on a MIG slice, and 733 and 777 ran in two segments each on another cluster's A100s.
They are a guide to cost, not a benchmark.

| released file | stages | cumulative samples |
|---|---|---|
| `weights/surface/d7_b1w64_r740.safetensors` | 740 | 199,999,488 |
| `weights/surface/d9_b3_r764.safetensors` | 710 → 764 | 399,998,976 |
| `weights/surface/d11_b5_r777.safetensors` | 711 → 733 → 735 → 777 | 799,997,952 |
| `weights/surface/d13_b5_r775.safetensors` | 711 → 733 → 775 | 599,998,464 |
| `weights/surface/d15_b5_r782.safetensors` | 710 → 732 → 774 → 778 → 782 | 899,997,696 |
| `weights/surface/d17_b5_r787_deployed.safetensors` (deploy tables) | 710 → 732 → 774 → 778 → 787 | 999,997,440 |

The pairs 710/711, 732/733 and 774/775 are replicas: the same recipe on different run ids, trained side by side. Each
chain above follows one member of a pair. In the network-alone failures, replicas differed by up to 20 %. At d = 13,
774 and 775 had 1,456 and 1,626 calibration failures. At d = 15, 778 and a replica of it that was not released
(779) had 1,726 and 2,063 validation failures. Use that spread as the tolerance when you compare a retrained network
with a released one.

## BB72 (bb/bb72/): r11 and r12

Program: `bb/ising_compare` (with `bb/qldpc_neural`), `python -m bb.train` (see `bb/SOURCE`). The command is
`bb/ising_compare/bb/scripts/alpine/args/s2_train.txt` with RID = 11 or 12.
- Data: fresh Stim samples of Relay-BP's BB72 X-memory circuit (6 rounds, uniform circuit noise, p = 0.3 %, Relay-BP at
  revision d185194). Chunk c of 8,192 shots has the seed `bb.shots.chunk_seed(run_id, p, c)`. There is no noise
  curriculum.
- Network: ConvLines-BB with 3 blocks, width 128, bottleneck 32 and 4 heads; its taps come from
  `data/bb/design_bb72.json`.
- Optimiser: AdamW (0.9, 0.99) with weight decay 0.01 on the matrices (not on the head's biases), batch 1,024,
  peak LR 1e-3 with 500 warm-up steps, then a cosine to 2 %. Gradients are clipped at 1; training uses bf16 and TF32.
- Read-out: the EMA (0.999) at the last step.
- Loss: cross-entropy over the 4,096 logical classes, plus the mean BCE over every translate's flips.

| released file (the original checkpoint) | initialised from | samples (= cumulative) | run id | sha256 | calibration failures, network alone (200,000 shots, p = 0.3 %) |
|---|---|---|---|---|---|
| `weights/bb/r11_final_ema.pt` | random | 199,999,488 | 11 | 58e60897… | 1,333 |
| `weights/bb/r12_final_ema.pt` | random | 199,999,488 | 12 | f2821c32… | 1,396 |

Each took about 3 hours on a MIG slice (2g) of an H200.

## Gross code (bb/gross/): g2_D12k113c10

The same program, on Relay-BP's BB144 X-memory circuit (12 rounds, uniform circuit noise, p = 0.4 %) with
`data/bb/design_bb144.json`, batch 1,024, width 128, bottleneck 32 and 4 heads. Four stages:

| stage | network | initialised from | samples | schedule | command |
|---|---|---|---|---|---|
| r21 (run id 21) | 3 blocks | random | 99,999,744 | Noise curriculum: p rises from 0.1 % to 0.4 % over the first half of the samples, on Relay-BP's circuit grid. LR 1e-3, 500 warm-up steps, cosine to 2 %. The early stop (`--early-stop 5e7:0.20`) did not fire. | `s2_train.txt`, RID = 21 |
| g2_A3D (run id 31) | 6 blocks | r21, three blocks added at zero | 19,996,672 | p = 0.4 %. LR 3e-4, 1,000 warm-up steps, peak held to 80 % of the steps (`--decay-start 0.8`), then cosine. | `gross/ising_compare/bb/PLAN_gross2.md`, tier A, "A3 D" |
| g2_D12 trunk (run id 51) | 12 blocks | g2_A3D, six blocks added at zero | 113,238,016 (paused there) | LR 3e-4, 1,000 warm-up steps, then flat (`--decay-start 1`). Its EMA network at the pause is `ema_113238016.pt`. | `gross/ising_compare/bb/PLAN_gross2_stage2.md`, "trunk", `--stop-at 113238016` |
| g2_D12k113c10 (run id 53) | 12 blocks | the trunk's EMA at 113,238,016 samples | 9,999,360 | LR 3e-4, 200 warm-up steps, then cosine to 2 % (`--decay-start 0`). | the same, "cooldown", `--samples 1e7` |

`weights/bb/g2_D12k113c10_final_ema.pt` is the original checkpoint (sha256 eb380295…). Its cumulative budget is
243,233,792 samples. With every warm start (`--init`), the step-0 decisions must equal the
initialising network's (bb.train checks this). Each stage has a fresh optimiser, schedule and EMA. The gross plans
ran these stages through a driver that is not included (`bb.gross2`: budget, ledger and selection on a rented GPU).
Its training calls are the `bb.train` commands quoted above, with the plans' common arguments
`--code bb144 --p 0.004 --design data/bb/design_bb144.json --compile --monitor 10-11 --monitor-shots 10000`. The
plans' speed options (`--lean`, `--chunk-h2d`) give bit-equal inputs and labels.

Expected results: 717 failures in 100,000 calibration shots and 350 in the 50,000-shot D-bank, both for the network
alone at p = 0.4 %.

## Cascade comparison on BB72 (cascade/bb72/): seven networks

These networks are all trained on BB72 at p = 0.55 %, either under uniform circuit noise ("uniform") or under the
fitted `depspam_fm` family (primed names). The noise curriculum is Cascade's three-stage one
(`cascade_optim.p_schedule`): p = 0.2 % for the first 1 % of the budget, a linear anneal to 0.55 % over the next 1 %,
then 0.55 %. The schedule is fixed by a budget; past the budget the LR stays at its floor until a convergence rule
stops the run (`--converge 3e7:0.02`). The rule fires when the EMA failures on the fixed monitor block fell by less
than 2 % at both monitor p over the last 3·10^7 samples. The released file is the EMA network at that stop
(`final_ema.pt`); the network at the budget itself (`budget_ema.pt`) is not released. Data seeds come from the
training stage's namespace, `cascade_data.train_seed(run_id, 0, c)`. The commands are in
`bb/ising_compare/bb/cascade/stage2/jobs/`.

| released file | program | network | initialised from | run id | budget | stopped at (= cumulative unless noted) | sha256 of the original | monitor failures, 200,000 shots at 0.3 % / 100,000 at 0.55 % |
|---|---|---|---|---|---|---|---|---|
| `weights/cascade_bb72/S55.pt` | `bb.train --w3-stream`, uniform (`w3b_S55.sh`) | ConvLines-BB, 3 blocks | random | 110 | 2e8 samples, LR 1e-3, warm-up 500, cosine from the warm-up's end | 210,001,920 samples | 87886c65… | 1,162 / 13,022 |
| `weights/cascade_bb72/S55p.pt` | the same, `--family depspam_fm` (`w3c_S55_dfm.sh`) | 3 blocks | random | 210 | the same | 200,007,680 | 22fa0c83… | 1,967 / 19,027 |
| `weights/cascade_bb72/S55p_deep.pt` | the same (`w3d_S55_dfm_deep.sh`) | 6 blocks | S55p, three blocks added at zero (`--init`) | 211 | the same | 200,007,680 own; 400,015,360 cumulative | 051370c3… | 1,335 / 13,998 |
| `weights/cascade_bb72/C14.pt` | `bb.cascade_train`, uniform (`w3b_C14.sh`) | Cascade, L 14, H 512 | random (init seed 0) | 101 | 80,000 steps of 3,328 (266,240,000 samples) | step 81,130 (270,000,640) | 0f44e614… | 500 / 6,321 |
| `weights/cascade_bb72/C6.pt` | the same (`w3b_C6.sh`) | Cascade, L 6, H 512 | random | 102 | the same | step 84,135 (280,001,280) | e20101ec… | 682 / 9,830 |
| `weights/cascade_bb72/C14p.pt` | the same, `--family depspam_fm` (`w3c_C14_dfm.sh`) | L 14 | random | 201 | the same | step 81,130 (270,000,640) | dd0c3b14… | 933 / 10,174 |
| `weights/cascade_bb72/C6p.pt` | the same (`w3d_C6_dfm.sh`, then `w3e_C6_dfm.sh`) | L 6 | random | 202 | the same | step 81,130 (270,000,640) | 5fd3bca6… | not in the comparison's table |

Cascade's recipe (`bb.cascade_train`) uses Muon at peak LR 3e-3 for the matrices and Lion at 2e-4 for the rest,
weight decay 0.003, a 1,000-step warm-up, then a cosine over 50,000 steps to a floor of 0.1. Gradients are clipped at
1, the batch is 3,328 (two micro-batches), and the run may go to 160,000 steps at most.

C6' trained first on the Windows workstation's RTX A6000 up to step 67,331; it then resumed exactly
(cascade_train's resume) on an RTX PRO 6000 on the cluster. C14, C6 and C14' trained on one GPU each: an RTX PRO
6000, a MIG slice and an H200, in that order. `S55'` and `S55'-deep` stopped by the rule right after the budget.

## Output-head comparison (head/): 21 networks

Program: `head/ising_compare` (with `head/qldpc_neural`), the comparison's versions of `arch.train` (surface code) and
`bb.train` (BB72), which add the `--head` choice. The arguments are `head/ising_compare/a1/jobs/args/common.sh`,
`train_surface.txt` and `train_bb72.txt`; `a1/pod/job.sh CODE ARM SEED [RAMP]` ran them on the pods.
- Every arm of seed s starts from the same registered trunk: `python -m a1.trunk` draws the stem and blocks after
  `torch.manual_seed(s)`, and its sha256 is in `head/ising_compare/data/a1/trunks.txt`. The head is drawn with seed
  s + 1,000. Every arm of a seed reads the same shots in the same order (`a1/seeds.py`).
- Surface code, d = 9, p = 0.6 %: the deployed recipe of d9_b3_r764, from the trunk. 3 blocks of width 128, AdamW
  (0.9, 0.99), weight decay 0.01, batch 1,024, peak LR 1e-3 with 1,000 warm-up steps and a cosine to 2 %, clip 1, EMA
  0.999, bf16, 2e8 samples.
- BB72, p = 0.3 %: the recipe of r11 and r12 above (`data/bb/design_bb72.json`, 500 warm-up steps), from the trunk,
  2e8 samples.
- The three ramp reruns (`a1ramp_*`, seed 1) raise p from 0.1 % to the target over the first quarter of the samples
  (`--p-ramp 0.001:0.25`); the other runs train at the target p throughout.
- Run ids: 1,100 + 10 s + i on the surface code (i = 0-4 in the order of the table) and 1,200 + 10 s + i on BB72
  (i = 0-3); a ramp rerun adds 50.
- Hardware: rented H100 SXM pods, one run per GPU at a time.

| arm | code | head flags | parameters | released files |
|---|---|---|---|---|
| S-VA | surface | `--head vote --line-weight 1` | 389,395 | `weights/head/surface/a1_S-VA_s{1,2}/`, `a1ramp_S-VA_s1/` |
| S-V | surface | `--head vote --line-weight 0` | 389,395 | `weights/head/surface/a1_S-V_s{1,2}/` |
| S-L | surface | `--head line --line-weight 0` | 388,235 | `weights/head/surface/a1_S-L_s{1,2}/` |
| S-P | surface | `--head pooled --line-weight 0` | 507,115 | `weights/head/surface/a1_S-P_s{1,2}/`, `a1ramp_S-P_s1/` |
| S-PA | surface | `--head pooled --line-weight 1` | 507,115 | `weights/head/surface/a1_S-PA_s{1,2}/`, `a1ramp_S-PA_s1/` |
| B-JT | BB72 | `--head joint --translate-weight 1` | 483,357 | `weights/head/bb72/a1_B-JT_s{1,2}/` |
| B-J | BB72 | `--head joint --translate-weight 0` | 483,357 | `weights/head/bb72/a1_B-J_s{1,2}/` |
| B-P | BB72 | `--head pooled --translate-weight 0` | 997,905 | `weights/head/bb72/a1_B-P_s{1,2}/` |
| B-PA | BB72 | `--head pooled --translate-weight 1` | 997,905 | `weights/head/bb72/a1_B-PA_s{1,2}/` |

Each directory holds `final_ema.pt`, an inference copy of the run's final EMA checkpoint: the configuration, the EMA
state dict byte for byte, the training record with its paths cut to file names, and `source_sha256`, the sha256 of the
original checkpoint that the reads used. The training logs are in `head/training_records/`, and the not-trained check
at 2e7 samples of every run is in `head/results/{surface,bb72}/train_checks.csv`. Seed 1's pooled surface head (S-P)
never trained (`head/A1_RESULT.md`, sections 0 and 1.6).

## Cascade at d = 13 (cascade/d13/): CS13

Program: `cascade/d13/surface/surface_train.py`, Cascade's recipe of `bb.cascade_train` on the surface network
(`surface_net.py`, L = 13, H = 256, 2,531,841 parameters), run as `pod/pod_cs13.sh` describes; the plan's section 3.3
(`cascade/d13/SURFACE_PLAN.md`) gives the arguments.
- Data: fresh Stim samples of the release's d = 13 circuit family at the training p, seeds
  `surface_data.train_seed(13, 1, rank, step)` (run id 1).
- Noise curriculum: p = 0.2 % for the first 1 % of the steps, a linear anneal to 0.7 % over the next 1 %, then 0.7 %.
- Optimiser: Muon at 3e-3 for the convolution weights and Lion at 2e-4 for the rest, weight decay 3e-3, EMA 0.9998;
  a budget of 80,000 steps of 3,328 shots (four micro-batches of 832 on one GPU), then the floor LR up to 120,000 steps
  at most. The stopping rule `--no-improvement 3e7:0.02` reads the EMA on a fixed monitor block (300,000 shots at
  0.6 %, 100,000 at 0.7 %) every 1e7 samples.
- From scratch on one rented H200, in two segments (steps 0-56,364 and 56,364-81,130, resumed exactly). It stopped
  with status `no_improvement` at step 81,130 (270,000,640 samples) on 12 October 2026.

`weights/cascade_d13/CS13/final_ema.pt` is an inference copy of the EMA at the stop (configuration, EMA state dict byte
for byte, step, status, arguments, and `source_sha256` 5351b27b…, the original checkpoint the evaluation read). The
training log, both segments' output and the end record are in `cascade/d13/training_records/`.
