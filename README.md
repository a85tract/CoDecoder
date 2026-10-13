# ConvLines: code, weights and records

This repository holds the artifacts of the paper "A Small Network Makes Quantum Decoders Up to 100× Faster and More
Accurate" by Yicheng Guang, Yuliang Xiong, Xun Gao and Yueqi Chen (University of Colorado Boulder): the training method,
the trained networks and the test method, with the records that check every table and figure.

[`METHODS.md`](METHODS.md) states how the accuracy and the speed were measured: the shots and their seeds, the decoders
and their settings, the statistics, what was timed, on which hardware and in which session.

## What the paper shows

A quantum error-correction decoder reads the detector bits of one shot (one run of the error-correction circuit) and
decides which logical flip happened. The most accurate decoders, here called strong decoders, are slow. ConvLines is a
small convolutional network that decodes every shot and reports a confidence; the least confident shots, a fixed
fraction q chosen on calibration shots before any test, go to a strong decoder. Its output head predicts the flips of
many shifted copies of the logical operators and decides all logical qubits together.

The paper puts ConvLines in front of four strong decoders. Except where marked, the numbers here come from the two
newest registered tests, SURF-v2 and RETEST (next section). In SURF-v2, on the rotated surface code at distances d = 7
to 17, the pair (the network with its strong decoder) failed 10 % to 31 % less often than correlated matching
(PyMatching) alone and took 2.3 to 9.3 times less time per shot. At d = 9, 11 and 13 the registered rule chose q = 0, so
the network alone decided every shot. At every distance the pair was also more accurate and faster than plain matching,
NVIDIA's Ising pre-decoders followed by plain or correlated matching, and two BP-OSD settings.

RETEST repeated the other comparisons on fresh shots. In front of the belief-propagation decoder Relay-BP on all
detectors, the pair failed 23 % and 29 % less often on the bivariate-bicycle (BB) code BB72 [[72,12,6]] at p = 0.3 % and
0.2 %, and 50 % less often on the gross code [[144,12,12]] at p = 0.4 %. RETEST timed nothing again, so its times are
estimates from the registered timing sessions: the pair took 17 times less time per shot on BB72 at p = 0.3 % and 12
times less on the gross code. Relay-BP's own time at p = 0.2 % was measured only after the test, in a supplementary
session (`bb/bb72/ising_compare/data/bb/stage3/SUPPLEMENT_fulldem_time_p0.002.md`), and against it the pair took 111
times less time. In front of our two re-implementations of the neural decoder Cascade, the pair failed 1.1 % and 0.3 %
less often than Cascade alone on BB72 (p = 0.2 % and 0.3 %) and 1.2 % to 1.3 % less often on the surface code at d = 13
and p = 0.35 %, each significantly, and took 5.1 to 16.4 times (BB72) and 3.7 to 7.7 times (d = 13) less time per shot
on average. At d = 13 and p = 0.6 % it failed 0.4 % less often, which the re-test found non-inferior but not
significantly better. At d = 13 its 99.9th-percentile time was also longer than Cascade's (490 µs against 411 and
417 µs). On BB72, the same network with a conventional head that predicts each logical qubit separately failed 3.8 to
5.0 times as often when it decided every shot itself.

Most tests were registered: the networks, thresholds, test sets and analyses were fixed before the test shots existed,
and each test set was drawn once and read once. Both new registrations fixed in advance that the paper reports their
numbers, whatever they turned out to be, in place of the original tests' (`RETEST.md` section 5,
`surface_v2/surf_v2/REGISTRATION.md` "The paper's rows"). The original tests stay in this repository as earlier
registered tests, with their plans and records.

## Registered tests and how to verify them

RETEST and SURF-v2 each fixed everything their results depend on (the networks, their thresholds, the rule that makes
the test seeds, and the analysis) in a freeze commit and pushed it to GitHub before any test shot existed. GitHub
records the push in the repository's activity list. The test seeds come from the first round of the drand public
randomness beacon (mainnet, one round every 30 s) at least 600 s after that push, so nobody could know the test shots
when the freeze was made. Each test set was then drawn once, read once by every decoder and analysed once by the frozen
analysis.

RETEST (`RETEST.md`) re-ran the BB tests of Table I, both Cascade comparisons and the output-head comparison on fresh
shots, with the networks, thresholds, decoders and analyses frozen for their original runs; it timed nothing again.
SURF-v2 (`surface_v2/surf_v2/REGISTRATION.md`) tested the surface-code networks retrained without line supervision
(trained on the observable's flip alone) at d = 7 to 17 against eight baselines, with the deployed networks read
beside them. It also registered a secondary comparison with gap switching
(`surface_v2/surf_v2/GAPSW_REGISTRATION.md`), in which the pair failed less often than gap switching at every distance.

| test | freeze commit | GitHub push record of the freeze | beacon round | analysis commit | results |
|---|---|---|---|---|---|
| RETEST | `dd332c5` | `push` to `main` from `ab5cf49` to `dd332c5`, 2026-10-13T02:34:48Z, record 46295757992 | 6547616, released 2026-10-13T02:45:00Z; the record, the seeds and the draw manifests are in `390f2da` | one per test: `7c1aae0`, `d92e011`, `a3bd87d`, `4a85d65`, `87a1742`, `301e9e7`; the summary `3cc1258` | `RETEST_RESULTS.md`; each test's records and analysis outputs in `retest/run/` |
| SURF-v2 | `f94f4ed` | `branch_creation` of `surf-v2` at `f94f4ed`, 2026-10-13T11:49:59Z, record 46339453746 | 6548726, released 2026-10-13T12:00:00Z; the record and the test manifest are in `ce8780e` | `88d14de`; gap switching `3b41af7` | `surface_v2/surf_v2/analysis/tables.md`, `surface_v2/surf_v2/gapsw/analysis/gapsw_tables.md` |

- The push records are GitHub's JSON as its API returned it: `retest/run/github_push_activity.json`, and
  `surface_v2/surf_v2/beacon.json`, which keeps the record together with the whole activity list.
- `RETEST.md` as frozen is `git show dd332c5:RETEST.md`. The summary commit `3cc1258` recorded in it the run's one
  deviation, the job shape of the gross-code reads, and corrected or completed several commands and descriptions;
  `git diff dd332c5 3cc1258 -- RETEST.md` shows every change. After `f94f4ed`, SURF-v2's commits only add files.
- `surface_v2/surf_v2/REGISTRATION.md` names the branch `surf-v2`, to which the freeze was pushed (the
  `branch_creation` record above). That branch still exists, and `main` now holds the same commits.
- The plans of the earlier registered tests are listed in the Table IX row of [Tables and figures](#tables-and-figures).

### Checking a registration

The steps check SURF-v2 from `surface_v2/` (paths are relative to it) in the pinned environment of
[Installation](#installation), with the GitHub CLI `gh` and network access to drand. Each step ends with the same check
for RETEST, run from the repository root.

1. Read the push record.

   ```bash
   cd surface_v2
   gh api 'repos/a85tract/Convlines_Public/activity?per_page=100' > activity.json    # one page: under 100 records
   jq '.[] | select(.after == "f94f4edb843b4a0b4762c556d79782cf2a2167e2") | {id, activity_type, ref, timestamp}' activity.json
   ```

   The earliest record whose `after` is the freeze commit gives the push time T, so a later push of the same commit
   cannot move it: record 46339453746, `branch_creation` of `refs/heads/surf-v2`, 2026-10-13T11:49:59Z, the same
   record as `push_activity` in `surf_v2/beacon.json`. RETEST: select `dd332c5082f44c41d486327400d90c73c77c926a`, which
   gives record 46295757992, `push` to `refs/heads/main`, 2026-10-13T02:34:48Z.

2. Derive the drand round: the first round whose time, 1595431050 + 30 (round − 1) in UNIX seconds, is at least
   T + 600. T = 1791892199 (`date -u -d 2026-10-13T11:49:59Z +%s`) gives round 6548726, released at 1791892800
   (12:00:00 UTC). RETEST: T = 1791858888 gives round 6547616, released at 1791859500 (02:45:00 UTC).
   `convlines.beacon.beacon_round` computes the same.

3. Check the round against drand's public API. `convlines.beacon fetch` derives T and the round from your copy of the
   activity list, fetches the round from four drand endpoints, requires at least two answers that agree, checks them
   and writes a new record, whose round, signature and randomness must equal the committed record's:

   ```bash
   python -m convlines.beacon fetch --commit f94f4edb843b4a0b4762c556d79782cf2a2167e2 --activity activity.json \
       --push-source "verification" --out beacon_check.json
   jq -c '{round, randomness, signature}' surf_v2/beacon.json beacon_check.json
   ```

   The randomness is the SHA-256 of the round's signature, `930cdbaf...` for round 6548726, and drand serves the round
   without our code at
   `https://api.drand.sh/8990e7a9aaed2ffed73dbd7092123d6f289930540d7651336225dc172e51b2ce/public/6548726`. RETEST:
   `python retest/beacon.py --push-time 2026-10-13T02:34:48Z --commit dd332c5082f44c41d486327400d90c73c77c926a --out
   rt_beacon_check.json` fetches and checks round 6547616, to compare with `retest/run/beacon.json` (randomness
   `299266d1...`).

4. Re-derive the seeds and check them against the test manifest.

   ```bash
   python -m convlines.beacon check --beacon surf_v2/beacon.json
   python -m convlines.testset seeds --set sv2test --approval surf_v2/beacon.json
   ```

   The first checks the committed record offline: the chain, T as the earliest activity record of the freeze commit,
   the round, the endpoints' agreement, and randomness = SHA-256(signature). The second checks that
   `testsets/sv2test.json` names this record by its sha256 and re-derives all its seeds, 200 chunks of 10,000 shots at
   d = 7 and 100 at each other distance ("700 of 700 seeds re-derived"). RETEST: `python retest/seeds.py --verify
   retest/run/beacon.json retest/run/seeds.json` recomputes every set's seeds from the beacon, `retest/tags.json` and
   the historical seeds and prints `"equal": true`, and it prints the same with step 3's `rt_beacon_check.json` in
   place of the committed record. The draw manifests `retest/run/bb/manifest_bb72.csv`, `manifest_gross.csv` and
   `retest/run/head/rows.csv` list each chunk's seed and sha256.

5. Regenerate a test chunk and compare its sha256.

   ```bash
   python -m convlines.testset regen --set sv2test --d 9 --chunks 0 --out shots/sv2test
   ```

   It prints `d9_p0.006_X_sv2test_000.npz: seed 5867115360100734495 sha256 8c7f5d84... == manifest`. Bit-for-bit
   regeneration needs Stim 1.16.0 and numpy 2.2.6 on an x86-64 CPU with AVX2. RETEST's BB chunks are rebuilt through its
   shim, as `retest/jobs/bb_draw.sbatch` does: `retest/shim/rt_bb.py bb.shots ...` with `RT_BEACON` and `RT_SEEDS` set
   to `retest/run/beacon.json` and `retest/run/seeds.json`; the BB circuits come from Relay-BP's repository
   (Installation).

6. Rerun the analysis and compare it byte for byte.

   ```bash
   python -m surf_v2.analysis --freeze surf_v2/freeze.json --reads READS... --shots shots/sv2test \
       --cal-shots shots/sv2cal --sessions SESSION --cpu CPU --out run1
   diff -r run1 surf_v2/analysis
   ```

   Nothing in the outputs depends on the time, the host or the output directory, so `diff` must report only
   `sessions.log`, a one-line note of the timing session's pod and GPU that the analysis does not write. The test and
   calibration sets are rebuilt with `regen` (`--set sv2cal` for the calibration set). The other inputs (every decoder's
   per-shot reads, the timing session and the CPU-term records) are archived by the authors and are not in this
   repository; `surf_v2/analysis/analysis.json` lists each of them with its sha256 under `inputs`, and
   `surf_v2/args/alpine_analysis.txt` holds the command as it ran. The gap-switching analysis runs the same way
   (`surf_v2/args/alpine_gapsw_analysis.txt`), and `RETEST.md` section 3 gives each RETEST analysis command.

### NVIDIA's Ising decoders

The tables' Ising M1 and Ising M4 are NVIDIA's public pre-decoders Ising-Decoder-SurfaceCode-1-Fast (receptive field 9)
and Ising-Decoder-SurfaceCode-1-Accurate (receptive field 13). Each is followed by plain or by correlated matching on
the detectors it leaves, which gives the rows Ising M1 / M4 + PM and Ising M1 / M4 + correlated PM.

- Accuracy comes from Ising-Decoding's official PyTorch numerics at commit 3ce4379: NVIDIA's own inference module, with
  fp32 weights and TF32 as NVIDIA's runner enables it (`surface_v2/convlines/opponents.py`). SURF-v2 read all four Ising
  rows, with plain and with correlated PM, on every test shot (`surface_v2/surf_v2/args/pod_e1_test.txt`); their
  failures are in `surface_v2/surf_v2/analysis/tables.md` and `family_A.csv`.
- Speed comes from the registered composite times of Ising's TensorRT plans: the fastest validated FP16 or FP8 plan at
  batch 1 on an H100, timed with back-to-back CUDA-graph replays as NVIDIA's trtexec does, plus the matching share on
  the Grace basis (`results/surface/SURFACE.md`, section 4; `evaluation/timing/ising_trt.py` builds the plans). SURF-v2
  did not time Ising again. It compared against these times, copied into `surface_v2/surf_v2/config.json`
  (`registered_times_us`) from `results/surface/s2_composite.json` (Ising + PM at d = 7 to 13),
  `results_accuracy_speed_d15.json` (d = 15), `results_accuracy_speed_d17.json` (both rows at d = 17) and
  `controls_speed.json` (Ising + correlated PM at d = 7 to 15).

An Ising row's accuracy and its time thus come from two numerics of the same model, and no single run measured both.
The earlier surface-code tests also read every timed TensorRT plan for accuracy (`results/surface/s*_test_eval.csv`,
`r17_test_eval.csv`).

## Layout

| path | contents |
|---|---|
| `evaluation/` | the surface-code decoder (`convlines/`: network, reference decoder, routing, opponents, statistics, evaluation), the six verdict circuits (`circuits/`), the test-set manifests (`testsets/`), the deployed routing masks (`routing/`), the deployed GPU engines and timing harnesses (`timing/`), and `scripts/make_circuits.py` |
| `weights/` | the 38 released networks, `networks.csv` (name, file, parameter count), `MANIFEST.sha256` and `check_weights.py` |
| `training/` | `convlines_train/` (train and tune a front end for any circuit), the frozen programs that trained the released networks (`surface/`, `bb/`), `TRAINING.md`, `LINEAGE.md` (every network's stages) and the smoke test |
| `results/` | the surface-code records (`surface/`, with `SURFACE.md`), R17, the d = 17 re-registration (`r17/`), the analysis of the discordant shots (`discordant/`), the paper's figure scripts and figures (`figures/`), and a reproduction on a rented H100 (`reproduction/`) |
| `bb/` | the BB72 test (`bb72/`), the gross-code tests (`gross/`, `gross_stage4/`), each a frozen code tree with its plans and records, and `BB.md` |
| `cascade/` | ConvLines in front of our Cascade on BB72 (`bb72/`, with `CASCADE.md`) and on the surface code at d = 13 (`d13/`) |
| `head/` | the output-head comparison: its code, read-out rules, training records and results (`A1_RESULT.md`) |
| `RETEST.md`, `RETEST_RESULTS.md`, `retest/` | RETEST, the registered re-test on fresh shots: its plan, every re-tested number beside its original, and its beacon and seed code, shims, jobs, checks before the freeze (`retest/replay/`) and records (`retest/run/`) |
| `surface_v2/` | SURF-v2, the registered test of the retrained surface-code networks: the evaluation package it ran from, with the new networks in `weights/`, and `surf_v2/` with the registrations, their code, the freeze and beacon records and both analyses |
| `SANITIZE.md` | what was replaced before publication |

The frozen trees under `bb/`, `cascade/`, `head/` and `training/` are the programs as they ran, so their own documents
use the names of the research project: study 2 and 3 are the surface-code tests at d = 7-13 and 15 (test sets T2 and
T3), study 4 and R17 the two at d = 17 (T5 and T6), stage 3 and 4 are the BB tests, phase C and D1 the two Cascade tests on BB72, W5 the
comparison at d = 13, A1 the output-head comparison and X2 the analysis of the discordant shots. Table XI of the paper
maps every descriptive name to these short names, and `weights/networks.csv` does the same for the networks.

## Installation

Linux x86-64 with Python 3.12.14:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126    # or .../whl/cu128 (see below)
pip install -r requirements.txt
```

This installs numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, torch 2.11.0, safetensors 0.8.0 and matplotlib
3.11.2, which run the surface-code decoder, the weight check, the figures and the record checks. The surface-code
tests ran torch 2.11.0+cu126 (with Triton 3.6.0); the BB, Cascade and output-head work ran torch 2.11.0+cu128, which
NVIDIA Blackwell GPUs (RTX PRO 6000) need. A CPU wheel of torch runs everything that needs no GPU.

On a GPU, Triton (the deployed surface engine, the BB engines, and `torch.compile` in the training programs) builds a
small C helper at first use, which needs gcc and Python 3.12's development headers (`python3.12-dev` or
`python3.12-devel`). Where the headers are missing, point `C_INCLUDE_PATH` at a copy of them.

| part | also needs |
|---|---|
| BB codes and the Cascade comparison on BB72 | `pip install relay-bp==0.2.2`, and Relay-BP's repository at revision d185194 for its circuits: `QND_RELAY=/path/to/relay-bp bash training/bb/qldpc_neural/scripts/fetch_relay.sh`, then keep `QND_RELAY` set |
| the surface-code opponents Ising-Fast and Ising-Accurate | NVIDIA Ising-Decoding at commit 3ce4379 (tag v0.1.0) and its inference requirements (`results/surface/SURFACE.md`, section 3) |
| the BP-OSD opponent | `cudaq-qec-cu12==0.8.0` on an NVIDIA GPU, in a separate environment |
| Ising's TensorRT plans | the TensorRT stack listed in `results/surface/SURFACE.md`, section 4 |
| the frozen surface-code training program and the output-head comparison's surface arms | NVIDIA Ising-Decoding's `code/` at 3ce4379 (`ISING_DECODING_DIR` or `ICMP_NV`), which builds the circuits and the detector geometry |
| training with BP-OSD as the strong decoder | `pip install ldpc==2.4.1` |

Python packages are imported from the tree they belong to. The surface-code commands run from `evaluation/` (which
makes `convlines` and `timing` importable) and SURF-v2's from `surface_v2/`, the training package needs
`export PYTHONPATH=$PWD/evaluation:$PWD/training` from the repository root, RETEST's tools run from the repository root,
and each frozen tree runs from its `ising_compare/` directory with `PYTHONPATH=.:../qldpc_neural`, as its document says.

## Hardware

| work | hardware | time |
|---|---|---|
| the weight check, the record checks, the figures | any CPU | seconds to minutes |
| rebuilding a surface-code test set bit for bit | an x86-64 CPU with AVX2 and the pinned stim and numpy | minutes per distance; 0.4-0.6 GB per set |
| the surface-code accuracy of ours against matching | a CPU, or one CUDA GPU for speed | minutes to hours per distance |
| the deployed engines and the registered timings | an NVIDIA H100 (Hopper; the d = 7 kernel is built for sm_90a) and one exclusive CPU node | hours |
| reading the BB networks on test shots | an NVIDIA GPU; the commit guard of each BB tree checks the frozen engine signature (`bb/BB.md`) | minutes to hours |
| the Cascade comparison on BB72 and at d = 13 | one NVIDIA RTX PRO 6000; the evaluation code refuses other cards | hours |
| the output-head comparison's reads | one GPU of any model (H200 and RTX PRO 6000 MIG slices were used) | under an hour per code |
| the training smoke test | 16 CPU cores | about 5 minutes |
| retraining the released networks | CUDA GPUs | about 90 GPU-hours for the surface chain, about 3 per BB72 network, 22 hours on one H200 for CS13 |

## Tables and figures

Paths are relative to the repository root. The checks are described below the table. The records here are those of the
original tests. The paper's surface rows come from SURF-v2, and its BB, Cascade and output-head numbers from RETEST
([Registered tests and how to verify them](#registered-tests-and-how-to-verify-them)).

| paper | what it shows | records | check |
|---|---|---|---|
| Fig. 1, Fig. 2 | the pair; the output-head geometry (diagrams) | | |
| Fig. 3 | surface code: failure rate against composite time per shot, d = 7-17 | `results/surface/` (`s2_test_eval.csv`, `s3_test_eval.csv`, `s4_test_eval.csv`, `r17_test_eval.csv`, `s2_composite.json`, `results_accuracy_speed_d15.json`, `results_accuracy_speed_d17.json`, `r17_summary.json`, `controls_*`), `results/r17/` | surface code; figures |
| Table I | BB codes: failures and composite time | `bb/bb72/ising_compare/data/bb/stage3/` (`results.json`, `results_amendment_S.json`), `bb/gross/ising_compare/data/bb/stage3_gross/`, `bb/gross_stage4/ising_compare/data/bb/stage4_gross/` | BB codes |
| Table II | pairs with our Cascade: BB72 under the fitted noise model, and d = 13 | `cascade/bb72/cascade_compare/phase_d/` (`D1_RESULT.md`, `out/`), `cascade/d13/` (`RESULTS.md`, `out/`) | Cascade |
| Table III | the output-head comparison | `head/results/{surface,bb72}/` (`standalone.csv`, `pair_curves.csv`), `head/rule/`, `head/A1_RESULT.md` | output heads |
| Fig. 4 | where the pairs gain (exploratory) | `results/discordant/` (`X2_RESULT.md`, `data/`) | figures |
| Fig. 5 | routing frontiers on calibration shots | `results/surface/frontiers/`, `results/r17/policy_d17b5_r787_q08.json`, `bb/bb72/ising_compare/data/bb/fullroute/bb72_cal.json`, `bb/bb72/ising_compare/data/bb/policies/`, `bb/gross/ising_compare/data/bb/stage3_gross/frontierf_g2_D12k113c10.json`, `bb/gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md` | figures |
| Table IV | paired accuracy against the closest baseline | `results/surface/s2_test_eval.csv`, `s3_test_eval.csv`, `s4_test_eval.csv`, `r17_test_eval.csv` | surface code |
| Fig. 6 | BB codes: block failure rate against composite time | the records of Table I and `bb/bb72/ising_compare/data/bb/stage3/SUPPLEMENT_fulldem_time_p0.002.md` | figures |
| Table V | where the composite time goes | `results/surface/s2_composite.json`, `results_accuracy_speed_d15.json`, `results_accuracy_speed_d17.json`, `results/r17/R17_RESULT.md`; the BB rows in the records of Table I | surface code; BB codes |
| Table VI | decoder settings and timing | the settings in `evaluation/convlines/opponents.py`, `bb/bb72/ising_compare/bb/opp.py` and `route.py`, `cascade/bb72/cascade_compare/phase_c/pc_common.py`, `cascade/d13/surface/sv_common.py` | |
| Table VII | the first registered test with our Cascade on BB72 | `cascade/bb72/cascade_compare/phase_c/` (`RESULTS.md`, `out/confirm_*.json`, `out/report/`) | Cascade |
| Fig. 7 | failures of each output-head arm's pair against q | `head/results/{surface,bb72}/pair_curves.csv` | figures |
| Table VIII | every comparison of the output-head study | `head/results/{surface,bb72}/comparisons.csv` | output heads |
| Table IX | provenance of the claims | the plans: `results/r17/PLAN_R17.md`; `bb/bb72/ising_compare/bb/PLAN.md`, `STAGE3_PLAN.md`, `AMENDMENT_S.md`; `bb/gross/ising_compare/bb/STAGE3_GROSS_PLAN.md`; `bb/gross_stage4/ising_compare/bb/STAGE4_GROSS_PLAN.md`; `cascade/bb72/cascade_compare/PHASE_C_PLAN.md`, `phase_d/PHASE_D1_PLAN.md`; `cascade/d13/SURFACE_PLAN.md`; the test-set manifests in `evaluation/testsets/` | |
| Table X | medians of the shot descriptors (exploratory) | `results/discordant/data/tables.md`, `data/surface/summary.json` | |
| Fig. 8 | matching's class gap at d = 13 (exploratory) | `results/discordant/data/surface/fig_d13.npz` | figures |
| Fig. 9 | where the network fails, on shots read before any test | `bb/bb72/ising_compare/data/bb/fullroute/bb72_cal.json`, `policies/bb72_p0.003_rep2.json`, `bb/gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md`, `results/surface/development/a2_summary.json` | figures |
| Table XI | names in the paper and in this repository | `weights/networks.csv`, this README | |

### Weights

```bash
python weights/check_weights.py
```

checks every file of `weights/MANIFEST.sha256`, then builds each of the 38 networks with the code that uses it, loads
its weights strictly and compares its parameter count with `weights/networks.csv` (CPU, a few minutes).

### Surface code

From `evaluation/`:

```bash
cd evaluation
python -m timing.composite                                                      # every registered speed score from the recorded components
python -m convlines.testset regen --set T2 --d 9 --chunks 0-99 --out shots/T2    # rebuild T2 at d = 9 bit for bit
python -m convlines.evaluate --set T2 --d 9 --chunks 0-99 --shots shots/T2 \
    --opponents "plain PM" "correlated PM" --out d9.json                         # ours against matching, with the statistics
```

`timing.composite` recomputes the composite times and checks every speed cell against the records. `convlines.evaluate`
gives the failures, the paired test and the interval of each comparison, which `results/surface/s2_test_eval.csv`
records. `results/surface/SURFACE.md` covers the rest: the other distances and test sets, the Ising and BP-OSD
opponents, the deployed engines and their timing, the CPU term on the Grace basis, the two post hoc controls, study 4
and R17 (whose statistics it recomputes from the recorded counts), and what is not established.

SURF-v2 ran from its own copy of this package, `surface_v2/`, which adds the retrained networks, the set manifests
`sv2cal`, `sv2dry` and `sv2test`, and `surf_v2/` (the registration, its code and its records); run SURF-v2's commands
there. The copy in `evaluation/` is an earlier state of the package, and the commands above for the released networks
use it. [Registered tests and how to verify them](#registered-tests-and-how-to-verify-them) shows how to check SURF-v2.

### BB codes

`bb/BB.md` gives, for BB72 and both gross-code tests, how each test ran, its disclosed changes, and the commands that
rebuild its test shots, read the networks and the opponents, and analyse them. Each test's records hold every decoder's
per-shot predictions (`test_outputs_*.npz`) and the timing rows, so the statistics can be checked without decoding
again.

### Cascade

```bash
python cascade/bb72/repro/recompute_numbers.py --out /tmp/cascade_numbers        # BB72: every number of the comparison from its records
python cascade/d13/surface/scripts/results_tables.py cascade/d13/out             # d = 13: the tables of cascade/d13/RESULTS.md
```

`recompute_numbers.py` compares each recomputed number with its record and with the value printed in the paper at the
time it was written (CPU, under a minute). `cascade/bb72/CASCADE.md` describes the BB72 comparison: how it ran, where
each number comes from, the per-shot files and the GPU check of the networks (`cascade/bb72/repro/check_networks.py`).
`cascade/d13/RESULTS.md` and `cascade/d13/SURFACE_PLAN.md` do the same for d = 13.

### Output heads

`head/A1_RESULT.md` gives every verdict with its counts, and `head/results/{surface,bb72}/` the analysis outputs
(`analysis.json`, `standalone.csv`, `pair_curves.csv`, `comparisons.csv`, `references.csv`, `sizes.csv`,
`train_checks.csv`). The runs that drew the test sets, read the networks and the fallbacks, and analysed them are the
argument files `head/ising_compare/a1/jobs/args/draw_sets.txt`, `reads_main.txt`, `fallback_pm.txt`,
`fallback_relay.txt` and `analysis_main.txt`; they run from `head/ising_compare` under `a1/jobs/job.sbatch` with your
own directories for the networks, the sets and the reads. The sets' seeds and digests are in
`head/ising_compare/data/a1/sets_manifest.csv`.

### Figures

```bash
python results/figures/make_figs.py        # Figs. 3, 5, 6 and 9
python results/figures/make_head_fig.py    # Fig. 7
python results/figures/make_why_fig.py     # Figs. 4 and 8, from results/discordant/data
```

Each script writes its PDFs next to itself and names, beside every block of numbers, the record it was copied from.

## Training

`training/TRAINING.md` is the guide. In short, from the repository root:

```bash
bash training/smoke.sh                     # train a tiny d = 5 surface-code front end and tune its threshold (CPU, 5 minutes)
export PYTHONPATH=$PWD/evaluation:$PWD/training
python -m convlines_train.train --circuit my_circuit.stim --out runs/mine      # a front end for your own Stim circuit
python -m convlines_train.tune --frontend runs/mine/final_ema.pt --shots 200000 --seed 7 \
    --strong convlines_train.strong:pymatching_correlated --max-fraction 0.1 \
    --rule budget --front-us 2 --budget-us 3 --out runs/mine/tune.json       # its threshold, for any strong decoder
```

The released networks were trained by the frozen programs in `training/surface` (13 stages for the surface chain,
`bash training/surface/run_stage.sh STAGE`) and `training/bb` (BB72, gross and the Cascade comparison on BB72), by the
output-head comparison's code in `head/` and by `cascade/d13/surface/surface_train.py` (CS13). `training/LINEAGE.md`
gives each network's initialisation, stages, samples, schedule, learning rates and data seeds; `TRAINING.md` gives the
commands, and how to check that the frozen programs rebuild the released networks (`convlines_train.check_frozen`) and
how a retrained network compares with the released one. The threshold-tuning rule of the paper is `--rule paper` of
`convlines_train.tune`.

## Checks already run

- The Cascade comparison on BB72 was checked from fresh clones on the cluster (`cascade/bb72/REPRODUCTION_CHECK.md`):
  every manifest and archived per-shot file matched; the networks' GPU check passed; of 192 recomputed numbers, the 70
  that have a record all equal it, and of the 146 compared with the paper, one was not as printed (5.1 against 5.2, a
  speed-up's low end), which the paper has since corrected.
- R17's record check (`results/r17/REPRODUCTION_CHECK.md`) and a full re-run of the surface-code reproduction steps on a
  rented H100 (`results/reproduction/REPRODUCTION.md`).
- RETEST's replay, seed and platform checks before its freeze (`RETEST.md` section 6, records in `retest/replay/`).
- The verification of this repository's first commit from a fresh clone (next section).

## Verification of this repository

On 12 October 2026 this repository's first commit, `ab5cf49`, was cloned fresh on the Alpine cluster (CU Boulder
Research Computing) and checked there with the commands above.

- Installation: a new virtual environment from `requirements.txt` with torch 2.11.0+cu128, which brought Triton 3.6.0
  and cuDNN 9.19, plus relay-bp 0.2.2 and ldpc 2.4.1. Every pin installed as listed. The nodes' Python was 3.12.12
  (pinned: 3.12.14), and the GPU node needed `C_INCLUDE_PATH` for the Python headers (Installation).
- Weights: all 45 files match `weights/MANIFEST.sha256`, and all 38 networks build, load strictly and have the
  parameter counts of `weights/networks.csv`. `convlines_train.check_frozen` confirmed that the frozen training
  programs rebuild the 16 networks of the surface chain, BB72, the gross code and the Cascade comparison on BB72.
- Code and documents: every Python file compiles, and the 42 bash blocks of `README.md`, `results/surface/SURFACE.md`,
  `bb/BB.md`, `cascade/bb72/CASCADE.md` and `training/TRAINING.md` parse.
- Records: `timing.composite` reproduced every registered composite time and speed cell. `recompute_numbers.py`
  recomputed the 192 numbers of the Cascade comparison on BB72: none of the 70 that have a record differs from it, and
  of the 146 compared with the paper's values when the script was written, the one not as printed is the 5.1 that the
  paper now prints. `results_tables.py` printed the d = 13 tables, and the three figure scripts redrew every figure.
- Surface code, on a dry set (T2 at d = 9, chunks 0 and 99): both chunks were rebuilt with the manifest's sha256, and the
  reference decoder failed 94 and 124 times, as the earlier release's verification recorded. On chunk 0,
  `convlines.evaluate` gave ours 94 failures against 233 for plain PM and 124 for correlated PM (better than both), and
  the deployed Triton engine on an H200 MIG slice gave the same 94 failures, with the routing of every shot equal to the
  published mask.
- SURF-v2's options in `evaluation/`: `convlines.evaluate --save` wrote every decoder's per-shot arrays and record for
  the same chunk, on the CPU and through the deployed engine, with the same counts and the network's sha256 equal to the
  manifest's; `convlines.beacon` checked BB72's beacon record offline and re-derived its 50 test seeds.
- BB72, on a dry set (test chunk 0 at p = 0.3 %, 10,000 shots): the shots were rebuilt from their seed, and the second
  network's read on an H200 MIG slice, through the commit guard, matched the published per-shot outputs exactly: all
  10,000 decisions equal and every margin bitwise equal.
- Training: `training/smoke.sh` (4·10^4 samples) and `convlines_train.train --bb` on BB72 (2·10^4 samples) ran on a
  CPU; the frozen BB program (16 steps of `bb.train`) and the frozen surface program (20 steps of the d = 7 stage, with
  `torch.compile`) ran on an H200 MIG slice.

## What is not included

- The per-shot outputs of the test reads and the test shots themselves. The shots are rebuilt from the manifests'
  seeds; the per-shot outputs are archived by the authors.
- The fp32 module of the d = 17 surface network: it is released as its deploy tables, which the deployed engine reads.
- NVIDIA's Ising-Decoding code and checkpoints, and Relay-BP's repository: fetch them from their public releases.
- The research project's internal plans that the frozen trees cite but that no result depends on, the review texts,
  and the logs of runs that no table reads.

## License

The code and the trained network weights are licensed under the Apache License, Version 2.0 (`LICENSE`, `NOTICE`).
NVIDIA's Ising-Decoding code and checkpoints are only referenced and never redistributed; they keep their own terms
(the code under Apache 2.0, the checkpoints under their model cards on Hugging Face). PyMatching, Stim and Relay-BP keep
their own licenses.

## Citation

```bibtex
@misc{guang2026convlines,
  title  = {A Small Network Makes Quantum Decoders Up to 100x Faster and More Accurate},
  author = {Guang, Yicheng and Xiong, Yuliang and Gao, Xun and Chen, Yueqi},
  year   = {2026},
  note   = {Code, weights and records: \url{https://github.com/a85tract/Convlines_Public}}
}
```

`CITATION.cff` holds the same.
