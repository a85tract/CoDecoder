# Training ConvLines

This file covers the training side, in this order:

| step | what it does | time and hardware |
|---|---|---|
| 0 | A smoke test: train a tiny surface-code front end from a Stim circuit, then tune its threshold with PyMatching. | about 5 minutes on 16 CPU cores |
| 1 | Check that the frozen training programs rebuild every released network. | under a minute, CPU |
| 2 | Tune a released network's threshold and compare the result with the record. | 1-4 minutes at d = 7 and 9 on a CPU; a GPU above that |
| 3 | Train a front end for your own circuit and noise model, surface code or BB code, with one command. | hours on a GPU for a full run |
| 4 | Tune that front end's threshold with any strong decoder. | minutes |
| 5 | Retrain the released networks stage by stage with the frozen programs. | about 90 GPU-hours for the surface code; about 3 per BB72 network, more for gross and Cascade |
| 6 | Compare a retrained network with the released one. | minutes |

Run every command from the repository root unless a step says otherwise. Steps 0-4 and 6 use
`training/convlines_train/`, the release's training package. Step 5 uses `training/surface` and `training/bb`, frozen
copies of the programs that trained the released networks. `training/LINEAGE.md` gives every released network's stages:
initialisation, samples, schedule, data and cumulative budget. `training/surface/SOURCE` and `training/bb/SOURCE` give
the commits the frozen copies come from.

## Environment

The release's environment (`README.md`, "Installation"; `results/surface/SURFACE.md`, "Reproduction", section 1:
Python 3.12.14 and `requirements.txt`) runs `convlines_train` on a CPU or on one CUDA GPU. `python -m convlines_train...`
imports both packages, `training/convlines_train` and `evaluation/convlines`, so from the repository root set

```bash
export PYTHONPATH=$PWD/evaluation:$PWD/training
```

`training/smoke.sh` sets it itself. The steps that `cd` into `training/bb/ising_compare` or use
`training/surface/run_stage.sh` keep their own settings.

Some steps need more:

- BP-OSD as the strong decoder (`convlines_train.strong:bposd`; BB codes need it, since PyMatching cannot decode them):
  `pip install ldpc==2.4.1`.
- The frozen surface-code program (steps 1 and 5) builds its circuits with NVIDIA Ising-Decoding's code at commit
  3ce4379. Clone it as `results/surface/SURFACE.md`, section 3, does and set `ISING_DECODING_DIR` to the clone. Only
  `code/` is read, so the checkpoints and the extra packages are not needed here.
- The frozen BB programs and the released BB networks read Relay-BP's circuits at revision d185194.
  `QND_RELAY=/path/to/relay-bp bash training/bb/qldpc_neural/scripts/fetch_relay.sh` clones it there. Keep
  `QND_RELAY` set.
- Step 5 needs a CUDA GPU: the frozen programs compile the network with `torch.compile`, and Cascade runs Triton
  kernels.

For a given seed, Stim's samples depend on its version (1.16.0 here) and can depend on the CPU's vector width. Every
expected number below was measured on x86-64 (AMD EPYC 9534 or 7713P, 16 cores) with the pinned versions.

## Step 0. Smoke test

```bash
bash training/smoke.sh                     # optional: OUT directory (default runs/smoke), PY, SAMPLES, BATCH
```

The script does three things:
1. It writes Stim's rotated surface code (d = 5, 5 rounds, X basis, circuit noise p = 0.3 %) to
   `runs/smoke/d5_p003.stim`.
2. It trains a front end on 10^6 fresh samples (batch 256, 3,906 steps; `convlines_train.train` with its other
   defaults).
3. It runs `convlines_train.tune` on 100,000 new calibration shots (Stim seed 2), with PyMatching as the strong
   decoder on the least confident shots up to 20 %. The threshold follows the paper's rule, with
   M = round(0.9 × PyMatching's own failures).

Outputs:
- `runs/smoke/train/`: `final_ema.pt` (the front end), `log.jsonl` (a row every 250,000 samples), `run.json` and
  `last.pt`.
- `runs/smoke/tune.json` (the frontier, the rule and the choice) and `runs/smoke/tune_frontier.csv`.
- `runs/smoke/train.log` and `runs/smoke/tune.log`.

Our two runs on the same node type gave identical numbers; `training/smoke_record/` holds their logs. Another CPU type
or thread count changes the floating-point order, so the trained network and its counts differ somewhat:

| quantity | ours | expect elsewhere |
|---|---|---|
| EMA network on the 20,000 monitor shots, last row of `log.jsonl` | 113 failures | 80-160 |
| front end alone on the 100,000 calibration shots | 510 | 400-650 |
| PyMatching alone on the same shots | 381 | 381 when Stim gives the same shots, else 330-430 |
| the rule | M = 343 and limit 290.6; no q qualifies, so the fallback takes the fewest failures: q = 1 %, τ = 1.4665…, 369 failures | q at most 5 %, failures below the front end's alone |
| wall time | 293 s, 266 of them training | about 5 minutes on 16 cores; longer on fewer cores |

The front end alone fails more often than PyMatching here: 10^6 samples is a short run. With routing, the pair beats
both.

## Step 1. The frozen programs rebuild the released networks

```bash
ISING_DECODING_DIR=/path/to/Ising-Decoding QND_RELAY=/path/to/relay-bp \
    python -m convlines_train.check_frozen --surface 7,9,11,13,15,17 --bb
```

For each distance, the check confirms:
- `training/surface`'s circuit builder gives NVIDIA's circuit with the sha256 that every record uses.
- Its network class, built from that circuit, has the released grid, parity map A and parameter count.
- At d = 7-15 it loads the released state dict strictly. At d = 17, where only deploy tables are released, it compares
  A in engine order.

`training/bb`'s network classes then build each of the ten released BB networks from its own configuration and load
its weights strictly: r11, r12, g2_D12k113c10, S55, S55', S55'-deep, C14, C14', C6 and C6'. `bb.design` recomputes
BB72's taps.

Output: one JSON line per check, then `frozen trees: every check passed` (exit 0). Our runs took 17-23 s on a CPU.
`--surface ''` skips the surface code; leave out `--bb` to skip the BB codes.

## Step 2. Tune a released network and compare with the record

At d = 9, with study 2's calibration set and rule:

```bash
python -m convlines_train.tune --frontend d9_b3_r764 --chunks 60090000:20 \
    --strong convlines_train.strong:pymatching_correlated --grid 0,0.01,0.02,0.04,0.06,0.08,0.1 \
    --rule paper --target 2500 --out runs/released/tune_d9.json
```

Inputs: the released network, named as in `weights/surface/configs.json`, and its calibration shots. These are 20
chunks of 10,000 shots of `evaluation/circuits/d9_legacy.stim`, chunk j drawn with Stim seed 60,090,000 + j; the shots
are not stored.

Outputs: `tune_d9.json` and `tune_d9_frontier.csv`, with one row per q: τ, routed shots, failures, repaired and
introduced failures, mean time per shot.

Expected (the records are in `weights/surface/configs.json`, policy):

| quantity | record | our run |
|---|---|---|
| network alone | 2,484 | 2,484 |
| routed at q = 1 % | 2,188 | 2,188 |
| τ at q = 1 % | 0.6820775270462036 | 0.6813445055484771 |
| the rule's choice (M = 2,500, limit 2,358.6) | q = 1 % | q = 1 % |

The records' c came from the deployed engines (the FROZEN Triton engine; at d = 7 a CUDA kernel), whose arithmetic
differs slightly from fp32 PyTorch on a CPU. A shot near c = 0, or near the quantile, can therefore land on the
other side. Expect counts within 5 shots of the record and τ within 0.1 %; ours: 0.6813 against 0.6821.

For every distance, the calibration set and the record (200,000 shots; `weights/surface/configs.json`):

| `--frontend` | `--chunks` | network alone | correlated PM alone | q | routed at q | τ |
|---|---|---|---|---|---|---|
| d7_b1w64_r740 | 60070000:20 | 4,260 | 3,433 | 4 % | 3,204 | 1.3934324979782104 |
| d9_b3_r764 | 60090000:20 | 2,484 | 2,562 | 1 % | 2,188 | 0.6820775270462036 |
| d11_b5_r777 | 60110000:20 | 1,682 | 1,855 | 1 % | 1,408 | 1.0014245510101318 |
| d13_b5_r775 | 60130000:20 | 1,626 | 1,259 | 2 % | 1,082 | 1.8486117124557495 |
| d15_b5_r782 | 64150000:20 | 1,737 | 891 | 6 % | 806 | 3.9547388553619385 |

Our check ran d = 7 and 9 on a CPU. At d = 7 it found:
- network alone 4,262 (record 4,260);
- correlated PM alone 3,433, which matches the record exactly, so the shots match;
- routed at q = 4 %: 3,201 (record 3,204), with τ = 1.394216504096985 (record 1.3934324979782104).

Our d = 7 run used `--rule paper --target-strong 0.9` in place of `--target 2500`. It decodes every shot with the
strong decoder, so the result also gives correlated PM's count alone. Study 2 chose d = 7's q by another rule, so
read the q = 4 % row. Networks with five blocks take several milliseconds per shot on a CPU, so use a GPU
(`--device cuda`) at d = 11-15. The d = 17 network is released as deploy tables only, so this tool cannot run it.

`--target` is the study's planning count M. The paper's rule takes the smallest q whose calibration failures are at
most M − 2√(2M), and falls back to the q with the fewest failures. `--target-strong 0.9` sets
M = round(0.9 × the strong decoder's failures on every calibration shot) instead, as the BB72, gross and d = 17
studies did.

## Step 3. A front end for your own circuit

Surface code:

```bash
python -m convlines_train.train --circuit my_surface.stim --out runs/my_surface
```

The circuit is a rotated surface-code memory experiment with your noise model. It needs:
- d² data qubits, each measured once at the end, in d lines (`QUBIT_COORDS`);
- one observable that reads the d data qubits of one line;
- single-qubit measurements;
- detectors placed in space and time, either by `DETECTOR` coordinates (x, y, t) or through their ancilla records.

`training/convlines_train/surface.py` derives the geometry and the parity map A from the circuit. Before training it
checks on 20,000 fresh shots that A gives every line's flip, and it stops if not. Stim's
`surface_code:rotated_memory_x` circuit (the smoke test's) qualifies.

BB code, given in Bravyi et al.'s form (l, m, a1, a2, a3, b1, b2, b3) with A = x^a1 + y^a2 + y^a3 and
B = y^b1 + x^b2 + x^b3:

```bash
python -m convlines_train.train --circuit my_bb.stim --bb 6,6,3,1,2,3,1,2 --out runs/my_bb     # BB72's parameters
```

The circuit follows Relay-BP's conventions (`training/convlines_train/bbcode.py` lists them; Relay-BP's own BB circuits
qualify). The tool first runs `bb.design` on it: the stem's taps are every co-firing offset of the DEM. It also checks
every translate label on 2,000 shots, and writes both results to `runs/my_bb/design.json`.

The defaults come from our recipes and can all be changed:
- Network: for BB codes, 3 blocks of width 128 (bottleneck 32, 4 heads). For the surface code, 1 block of width 64
  (bottleneck 16) up to d = 7, 3 blocks of width 128 at d = 9, and 5 above.
- Data: 2·10^8 fresh samples of your circuit (`--samples`), batch 1,024.
- Optimiser: AdamW at peak LR 1e-3 with a warm-up of 1,000 steps (surface code) or 500 (BB), cosine to 2 %, weight
  decay 0.01, EMA 0.999.
- Noise curriculum: none for the surface code. BB: `--curriculum 0.36:0.01:0.01`, which scales every noise
  probability by 0.36 for the first 1 % of the samples, then ramps it linearly to 1 over the next 1 %. That is the
  S55 networks' 0.2 % → 0.55 %.

`python -m convlines_train.train --help` lists every option.

Outputs:
- `final_ema.pt`: the front end, with its configuration, geometry or code, circuit and training record.
- `log.jsonl`: every `--eval-every` samples, the LR, the training loss, and the failures of the EMA network and of
  the current weights on `--eval-shots` fixed shots.
- `run.json` and `last.pt`.

Both kinds were tested end to end on a CPU (`training/smoke_record/`), including the growth from d = 3 to d = 5 and a
BB run from 1 to 2 blocks. A run of 2·10^8 samples needs a GPU: on the studies' GPUs, the research programs took 2
hours (d = 9, 2 blocks) to 19 hours (d = 17, 5 blocks). On 16 CPU cores, the d = 5 smoke trained at 3,800
samples/s. This package's rate on a GPU was not measured for this release, so measure yours with `--samples 1e6`
before a long run.

- Interrupt and resume: `--max-minutes M` stops at a pool boundary after saving `last.pt` (exit status 3). The same
  command with `--resume` continues it exactly; every setting except `--max-minutes`, `--workers`, `--compile` and
  `--resume` must be unchanged.
- Growth: `--init-from` starts a larger network from a trained one, for example
  `python -m convlines_train.train --circuit my_d11.stim --out runs/my_d11 --init-from runs/my_d9/final_ema.pt`.
  For the surface code, the source may be a smaller distance of the same width. For BB codes, the source has the same
  code and fewer blocks (`--blocks`). Added blocks start as the identity; for BB this is checked at step 0. This is how the released
  networks grew (`training/LINEAGE.md`).

## Step 4. Tune your front end's threshold

```bash
python -m convlines_train.tune --frontend runs/my_surface/final_ema.pt --shots 200000 --seed 7 \
    --strong convlines_train.strong:pymatching_correlated --max-fraction 0.1 \
    --rule budget --front-us 2 --budget-us 3 --out runs/my_surface/tune.json
```

The tool first runs the front end on every calibration shot. It then runs the strong decoder, one call per shot with
its wall time measured, on the least confident shots up to `--max-fraction`. For each q of the grid (`--step`, or
`--grid` with an explicit list) it computes:
- τ(q), numpy's q-quantile of the confidence (the studies' rule);
- the shots routed (confidence < τ);
- the failures of the pair;
- the mean time per shot: the front end's time plus (the routed shots' strong-decoder time + `--handoff-us` per routed
  shot) / n.

Then it picks q by the rule you state:
- `--rule budget --budget-us B`: the fewest failures among the q whose mean time is at most B µs (ties: the smaller q).
- `--rule paper --target M` (or `--target-strong F`): the paper's rule from step 2. `--limit-us` first drops every q
  over a speed limit.

In this example the front end is charged 2 µs per shot, as a deployed engine might take, and the mean time may be at
most 3 µs. That leaves 1 µs per shot on average for the strong decoder, whose time is measured. Times are measured on
this host. `--front-us` replaces the front end's measured time (for example by your deployed
engine's), and `--timing batch` times a batch-only decoder once over all routed shots. The strong decoder is any
factory `module:function` that takes the `stim.Circuit` and returns a callable:
det (k, n_det) uint8 → predicted observables (k, n_obs). `convlines_train.strong` provides `pymatching`,
`pymatching_correlated` and `bposd`; in Python, `convlines_train.tune.tune()` takes the callable itself. The
calibration shots come from one of three sources:
- `--shots N --seed S`: fresh shots of the front end's circuit, or of `--circuit`;
- `--chunks BASE:K[:SIZE]`: K chunks with seeds BASE + j;
- `--calib FILE.npz`: your own shots, as `det` (packed or not) and `obs`.

Deploy the pair as follows: the front end decides every shot whose confidence is at least τ, and the strong decoder
decides the rest. Pick τ on shots that are not your test shots.

Outputs: `tune.json`, which holds the frontier, the rule with its numbers, the choice, τ, the strong decoder's
failures alone when it decoded every shot, and the sha256 of the front end's file and of the circuit;
`tune_frontier.csv`; with `--save-shots`, the per-shot arrays. The smoke test's `tune.json` is an example.

## Step 5. Retrain the released networks

The stages, their order and their recorded results are in `training/LINEAGE.md`. A retrained network is not
bit-for-bit the released one: the shots and their order are fixed by the run ids, but GPU arithmetic is not
deterministic. Step 6 compares the two.

### Surface code (`training/surface`, 13 stages)

```bash
export ISING_DECODING_DIR=/path/to/Ising-Decoding ICMP_CKPT=$PWD/runs/surface/ckpt ICMP_OUT=$PWD/runs/surface/out
bash training/surface/run_stage.sh d7b1w64_r740         # -> d7_b1w64_r740 (released d7_b1w64_r740)
bash training/surface/run_stage.sh g2_r710              # -> g2_r710 (d = 9, 2 blocks)
bash training/surface/run_stage.sh g6_d9_b3t_r764       # -> d9_b3t_r764 (released d9_b3_r764)
bash training/surface/run_stage.sh d13b3t_r732          # -> d13_b3t_r732
bash training/surface/run_stage.sh g7_d13_b5t_r774      # -> d13_b5t_r774
bash training/surface/run_stage.sh g9_d15_b5t_r778      # -> d15_b5t_r778
bash training/surface/run_stage.sh g12_d15_b5t_r782     # -> d15_b5t_r782 (released d15_b5_r782)
bash training/surface/run_stage.sh g14_d17_b5t_r787     # -> d17_b5t_r787 (released as d17_b5_r787's deploy tables)
bash training/surface/run_stage.sh g2_r711              # -> g2_r711
bash training/surface/run_stage.sh d13b3t_r733          # -> d13_b3t_r733
bash training/surface/run_stage.sh g7_d13_b5t_r775      # -> d13_b5t_r775 (released d13_b5_r775)
bash training/surface/run_stage.sh d11b3t_r735          # -> d11_b3t_r735
bash training/surface/run_stage.sh g8_d11_b5t_r777d     # -> d11_b5t_r777 (released d11_b5_r777)
```

Each command runs the stage's recorded `arch.train` line
(`training/surface/ising_compare/scripts/alpine_arch/args/<stage>.txt`) with three changes:
- the parent is read from `$ICMP_CKPT`;
- `--init-sha`, which pins the original parent's sha256, is dropped;
- `--no-val` is added, because the validation monitor reads stored shots that are not released and does not change
  the training.

Output: `$ICMP_CKPT/<run>/final_ema.pt`, plus `last.pt` and the logs. A stage needs its parent's `final_ema.pt`,
which the order above respects. Extra arguments pass through. `--workers` may be 1, 2, 4 or 8 without changing the
shots or their order.

The d = 17 stage carries `--max-hours 23.6 --max-hours-at-pool`, written for a 24-hour queue. Such a stop saves the
complete state as `state_p<pools>.pt`. Continue with
`bash training/surface/run_stage.sh g14_d17_b5t_r787 --resume $ICMP_CKPT/d17_b5t_r787/state_p<pools>.pt`.
Four stage files also run `arch.g6_check` before training: 774, 775, 782 and 787. It checks the mapping from the
parent, the line labels, the trainer's start and, where the distance is unchanged, that the added blocks are the
identity. It is optional, and it pins the original parent's sha256 (`--sha`), so drop that flag for a retrained
parent. Stages 774 and 775 also timed a random-weight network for the deployment's speed gate. That timing is not
part of the training, and its modules (`arch.latency`, `arch.artifact`, `arch.qualify`) are not in
`training/surface`.

### BB72, gross and the Cascade comparison (`training/bb`)

```bash
cd training/bb/ising_compare
export PYTHONPATH=.:../qldpc_neural QND_RELAY=/path/to/relay-bp BB_CKPT=$PWD/runs/ckpt BB_OUT=$PWD/runs/out \
       CASCADE_ROOT=$PWD/runs/cascade
# BB72 (weights/bb/): r11, r12
python -m bb.train --run r11 --run-id 11 --code bb72 --p 0.003 --design data/bb/design_bb72.json --samples 2e8 \
    --eval-every 1e7 --workers 6 --monitor 0 --compile
python -m bb.train --run r12 --run-id 12 --code bb72 --p 0.003 --design data/bb/design_bb72.json --samples 2e8 \
    --eval-every 1e7 --workers 6 --monitor 0 --compile
# gross (weights/bb/g2_D12k113c10_final_ema.pt): four stages
G="--code bb144 --p 0.004 --design data/bb/design_bb144.json --compile --workers 6"
python -m bb.train --run r21 --run-id 21 $G --samples 1e8 --p-ramp 0.001:0.5 --early-stop 5e7:0.20 \
    --eval-every 1e7 --monitor 0
python -m bb.train --run g2_A3D --run-id 31 $G --samples 19996672 --warmup 1000 --eval-every 5e6 --monitor 10-11 \
    --monitor-shots 10000 --init $BB_CKPT/r21/final_ema.pt --lr 3e-4 --decay-start 0.8 --blocks 6
python -m bb.train --run g2_D12 --run-id 51 $G --samples 1e9 --warmup 1000 --eval-every 5e6 --monitor 10-11 \
    --monitor-shots 10000 --init $BB_CKPT/g2_A3D/final_ema.pt --lr 3e-4 --decay-start 1 --blocks 12 --stop-at 113238016
python -m bb.train --run g2_D12k113c10 --run-id 53 $G --samples 1e7 --warmup 200 --eval-every 1e6 --monitor 10-11 \
    --monitor-shots 10000 --lr 3e-4 --decay-start 0 --init $BB_CKPT/g2_D12/ema_113238016.pt --blocks 12
# the Cascade comparison (weights/cascade_bb72/): ConvLines-BB S55, S55', S55'-deep
W="--code bb72 --p 0.0055 --design data/bb/design_bb72.json --samples 2e8 --eval-every 1e7 --workers 6 --monitor 0 \
   --compile --w3-stream --p-curriculum 0.002:0.01:0.01 --w3-monitor 0.003:100:120,0.0055:100:110 --extend-to 4e8 \
   --converge 3e7:0.02"
python -m bb.train --run S55 --run-id 110 $W
python -m bb.train --run S55_dfm --run-id 210 --family depspam_fm $W
python -m bb.train --run S55_dfm_deep --run-id 211 --family depspam_fm $W --blocks 6 --init $BB_CKPT/S55_dfm/final_ema.pt
# Cascade C14, C6, C14', C6' (our re-implementation; checkpoints under $CASCADE_ROOT/ckpt/<run>/)
C="--res skip --mup-base 0 --micro 2 --compile --workers 2 --log-every 200 --max-steps 160000 \
   --monitor 0.003:100:120,0.0055:100:110 --monitor-every 1e7 --converge 3e7:0.02"
python -m bb.cascade_train --run C14 --run-id 101 --L 14 --H 512 $C
python -m bb.cascade_train --run C6 --run-id 102 --L 6 --H 512 $C
python -m bb.cascade_train --run C14_dfm --run-id 201 --family depspam_fm --L 14 --H 512 $C
python -m bb.cascade_train --run C6_dfm --run-id 202 --family depspam_fm --L 6 --H 512 $C
```

These are the recorded commands, the same as in `training/bb/ising_compare/bb/scripts/alpine/args/s2_train.txt`,
`training/bb/ising_compare/bb/cascade/stage2/jobs/` and the gross plans, without the cluster's wrapper. Each run writes
`$BB_CKPT/<run>/final_ema.pt`; Cascade's runs write `$CASCADE_ROOT/ckpt/<run>/final_ema.pt`. Every run can be paused
and resumed exactly: add `--max-minutes M`, then rerun the same command with `--resume`. The gross trunk pauses by
itself at 113,238,016 samples (`--stop-at`) and writes its EMA network as `ema_113238016.pt`, which the cool-down
reads. The gross plans ran these stages through a driver that is not released. Its options `--lean` and `--chunk-h2d`
change only the speed, not the inputs or labels. r11 and r12 each took about 3 hours on a MIG slice of an H200.

## Step 6. Compare a retrained network with the released one

Surface code: run step 2's command with `--frontend $ICMP_CKPT/<run>/final_ema.pt`, which is an `arch.train`
checkpoint. Compare its counts with the released network's on the same calibration chunks. Replicas of the same recipe
on different run ids differed by up to 20 % in failures of the network alone (`training/LINEAGE.md`), so expect the
retrained network within 20 % of the record. Routing narrows the gap.

BB72 and gross: `convlines_train.tune` also reads `bb.train` checkpoints. Their circuit is Relay-BP's for their code
and p, so `QND_RELAY` must be set. Run these from the repository root with the environment section's `PYTHONPATH`
(step 5's BB block changed both). Compare yours with the released one on the same fresh shots:

```bash
python -m convlines_train.tune --frontend weights/bb/r11_final_ema.pt --shots 100000 --seed 5 \
    --strong convlines_train.strong:bposd --grid 0 --rule budget --budget-us 1e9 --out runs/released/tune_r11.json
python -m convlines_train.tune --frontend training/bb/ising_compare/runs/ckpt/r11/final_ema.pt --shots 100000 \
    --seed 5 --strong convlines_train.strong:bposd --grid 0 --rule budget --budget-us 1e9 --out runs/tune_r11_mine.json
```

`--grid 0` evaluates the network alone, which needs no strong decoder run. Ours, at p = 0.3 % on these 100,000
shots: r11 640 and r12 687 failures (0.64 % and 0.69 %; on the study's 200,000 calibration shots, 1,333 and 1,396).
The released gross network failed 36 of 5,000 shots at p = 0.4 % (seed 5; on the study's 100,000 calibration
shots, 717). The gross network is slow on a CPU, at 6 ms per shot. Expect a retrained network within 20 % of the
released one on the same shots.

Cascade comparison: each run's last monitor row is the failure count on the fixed monitor block (200,000 shots at
0.3 % and 100,000 at 0.55 %, seeds of the comparison's evaluation namespace). Compare it with `training/LINEAGE.md`:
C14 500 / 6,321, C6 682 / 9,830, S55 1,162 / 13,022, C14' 933 / 10,174, S55' 1,967 / 19,027,
S55'-deep 1,335 / 13,998. Expect the same 20 % spread. The comparison's own evaluation of the released networks is
`cascade/bb72/CASCADE.md`, "Reproduce".
