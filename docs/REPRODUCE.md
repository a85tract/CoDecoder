# Reproducing the paper

Part of the [CoDecoder release](../README.md).

## Repository layout

| path | contents |
|---|---|
| `evaluation/` | the surface-code decoder (`convlines/`: network, reference decoder, routing, opponents, statistics, evaluation), the six verdict circuits (`circuits/`), the test-set manifests (`testsets/`), the deployed routing masks (`routing/`), the deployed GPU engines and timing harnesses (`timing/`), and `scripts/make_circuits.py` |
| `weights/` | the 38 released networks, `networks.csv` (name, file, parameter count), `MANIFEST.sha256` and `check_weights.py` |
| `training/` | `convlines_train/` (train and tune a front end for any circuit), the frozen programs that trained the released networks (`surface/`, `bb/`), `TRAINING.md`, `LINEAGE.md` (every network's stages) and the smoke test |
| `results/` | the surface-code records (`surface/`, with `SURFACE.md`), R17, the d = 17 re-registration (`r17/`), the analysis of the discordant shots (`discordant/`), the paper's figure scripts and figures (`figures/`), and a reproduction on a rented H100 (`reproduction/`) |
| `bb/` | the BB72 test (`bb72/`), the gross-code tests (`gross/`, `gross_stage4/`), each a frozen code tree with its plans and records, and `BB.md` |
| `cascade/` | CoDecoder in front of our Cascade on BB72 (`bb72/`, with `CASCADE.md`) and on the surface code at d = 13 (`d13/`) |
| `head/` | the output-head comparison: its code, read-out rules, training records and results (`A1_RESULT.md`) |
| `RETEST.md`, `RETEST_RESULTS.md`, `retest/` | RETEST, the registered re-test on fresh shots: its plan, every re-tested number beside its original, and its beacon and seed code, shims, jobs, checks before the freeze (`retest/replay/`) and records (`retest/run/`) |
| `surface_v2/` | SURF-v2, the registered test of the retrained surface-code networks: the evaluation package it ran from, with the new networks in `weights/`, and `surf_v2/` with the registrations, their code, the freeze and beacon records and both analyses |
| `speed_rt/` | SPEED-RT, the registered speed re-test: its registrations (`SPEED_RT.md`, `STAGE2_*.md`), its timing code, sets, beacon records and manifests, stage 1's records and outputs (`cascade/records/`, `cascade/results/`) and stage 2's (`s2/<family>/records/`, `s2/<family>/results/`) |
| `d9_shallow/` | D9-SHALLOW, the registered test of the one-block network at d = 9: its plan, code, freeze, beacon and shot records, reads and results |
| `cascade_trt_reg/` | CASCADE-TRT-REG, the registered speed re-test with Cascade on TensorRT: its rules, code, beacon and seed records, timing records and results |
| `joint/` | the joint decision of CoDecoder and Cascade, held out and not registered: the fits of its weight on fresh shots, its counts on the test shots of the Cascade comparisons, the records of its timing session, the checks of the timed pipelines, an exploratory joint decision for the surface code, and `check_paper_numbers.py` |
| `SANITIZE.md` | what was replaced before publication |

The frozen trees under `bb/`, `cascade/`, `head/` and `training/` are the programs as they ran, so their own documents
use the names of the research project: study 2 and 3 are the surface-code tests at d = 7-13 and 15 (test sets T2 and
T3), study 4 and R17 the two at d = 17 (T5 and T6), stage 3 and 4 are the BB tests, phase C and D1 the two Cascade tests on BB72, W5 the
comparison at d = 13, A1 the output-head comparison and X2 the analysis of the discordant shots. Table XIV of the paper
maps every descriptive name to these short names, and `weights/networks.csv` does the same for the networks. The
joint decision's studies under `joint/` use names of their own, which `joint/README.md` maps.

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
| the deployed engines and the earlier tests' timings | an NVIDIA H100 (Hopper; the d = 7 kernel is built for sm_90a) and one exclusive CPU node | hours |
| rerunning SPEED-RT's analyses on its published records ([How to verify](SPEED.md#how-to-verify)) | a Linux x86-64 machine with Python 3.12, where the reruns were checked; numpy for Decoder Switching | seconds |
| re-timing as SPEED-RT did | H100 80GB HBM3 SXM machines for the GPU terms, Ising's plans and CUDA-Q; one RTX PRO 6000 and one H100 for Cascade; for the CPU terms, a Linux machine whose timed cores run nothing else, their SMT siblings offline (SPEED-RT used 8 cores of an AMD Ryzen 9 9950X) | hours |
| reading the BB networks on test shots | an NVIDIA GPU; the commit guard of each BB tree checks the frozen engine signature (`bb/BB.md`) | minutes to hours |
| the Cascade comparison's reads on BB72 and at d = 13 | one NVIDIA RTX PRO 6000; the evaluation code refuses other cards (SPEED-RT's launchers also admit the H100, for timing) | hours |
| the output-head comparison's reads | one GPU of any model (H200 and RTX PRO 6000 MIG slices were used) | under an hour per code |
| the training smoke test | 16 CPU cores | about 5 minutes |
| retraining the released networks | CUDA GPUs | about 90 GPU-hours for the surface chain, about 3 per BB72 network, 22 hours on one H200 for CS13 |

## Tables and figures

Paths are relative to the repository root. The checks are described below the table. The records here are those of the
original tests. The paper's surface accuracy comes from SURF-v2 (D9-SHALLOW at d = 9), its BB, Cascade and output-head
accuracy from RETEST
([Registered tests and how to verify them](REGISTERED_TESTS.md#registered-tests-and-how-to-verify-them)), and every speed number from
SPEED-RT, whose outputs the speed entries name ([How the speed was measured](SPEED.md#how-the-speed-was-measured-speed-rt)).
The rows of the joint decision, at the end of the table, number the tables and sections as the paper's version of
19 October 2026 does, in which the Cascade table is Table I. The rows above them follow an earlier numbering, in which
it is Table II.

| paper | what it shows | records | check |
|---|---|---|---|
| Fig. 1, Fig. 2 | CoDecoder and a strong decoder as a composition; the output-head geometry (diagrams) | | |
| Table I | BB codes: failures and composite time | `bb/bb72/ising_compare/data/bb/stage3/` (`results.json`, `results_amendment_S.json`), `bb/gross/ising_compare/data/bb/stage3_gross/`, `bb/gross_stage4/ising_compare/data/bb/stage4_gross/`, RETEST's reruns in `retest/run/bb/`; speed: `speed_rt/s2/bb/results/` (rows B1 to B6) | BB codes; [SPEED-RT](SPEED.md#how-to-verify) |
| Fig. 3 | surface code: failure rate against composite time per shot, d = 7-17 | `surface_v2/surf_v2/analysis/tables.md`, `d9_shallow/results/` (d = 9); speed: `speed_rt/s2/surface/results/` (rows S1 to S4), `speed_rt/s2/gapsw/results/` | surface code; figures; [SPEED-RT](SPEED.md#how-to-verify) |
| Table II | CoDecoder + Cascade: BB72 under the fitted noise model, and d = 13 | `cascade/bb72/cascade_compare/phase_d/` (`D1_RESULT.md`, `out/`), `cascade/d13/` (`RESULTS.md`, `out/`), RETEST's reruns in `retest/run/`; speed: `speed_rt/cascade/results/` (Cascade alone's fastest process, each composition's slowest) | Cascade; [SPEED-RT](SPEED.md#how-to-verify) |
| Table III | the output-head comparison | `head/results/{surface,bb72}/` (`standalone.csv`, `pair_curves.csv`), `head/rule/`, `head/A1_RESULT.md` | output heads |
| Fig. 4 | where the compositions gain (exploratory) | `results/discordant/` (`X2_RESULT.md`, `data/`) | figures |
| Fig. 5 | routing frontiers on calibration shots | `results/surface/frontiers/`, `results/r17/policy_d17b5_r787_q08.json`, `bb/bb72/ising_compare/data/bb/fullroute/bb72_cal.json`, `bb/bb72/ising_compare/data/bb/policies/`, `bb/gross/ising_compare/data/bb/stage3_gross/frontierf_g2_D12k113c10.json`, `bb/gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md` | figures |
| Table IV | surface code: failures and composite time of every decoder | `surface_v2/surf_v2/analysis/tables.md`, `d9_shallow/results/` (d = 9); speed: `speed_rt/s2/surface/results/` (row S1) | surface code; [SPEED-RT](SPEED.md#how-to-verify) |
| Fig. 6 | BB codes: block failure rate against composite time | the records of Table I; speed: `speed_rt/s2/bb/results/` | figures; [SPEED-RT](SPEED.md#how-to-verify) |
| Table V | times on the raw and on the Grace basis | `speed_rt/s2/surface/results/table.md` (row S4x), `speed_rt/s2/bb/results/numbers.json` (`sensitivity_raw_xeon`), `d9_shallow/results/table.md` | [SPEED-RT](SPEED.md#how-to-verify) |
| Table VI | where the composite time goes | `speed_rt/s2/surface/results/table.md` (row S6), `speed_rt/s2/bb/results/table.json` (row B4), `d9_shallow/results/table.md` | [SPEED-RT](SPEED.md#how-to-verify) |
| Table VII | decoder settings and timing | the settings in `evaluation/convlines/opponents.py`, `bb/bb72/ising_compare/bb/opp.py` and `route.py`, `cascade/bb72/cascade_compare/phase_c/pc_common.py`, `cascade/d13/surface/sv_common.py`; the timing in `speed_rt/SPEED_RT.md` section 1, `STAGE2_surface.md` and `STAGE2_bb.md` | |
| Table VIII | the speed-ups of Table II from mean times | `speed_rt/cascade/results/numbers.json` (every repetition) | |
| Table IX | the first registered test with our Cascade on BB72 | `cascade/bb72/cascade_compare/phase_c/` (`RESULTS.md`, `out/confirm_*.json`, `out/report/`); speed: `speed_rt/cascade/results/` | Cascade |
| Table X | the registered speed re-test with Cascade on TensorRT | `cascade_trt_reg/results_alpine/` (`tables.md`, `numbers.json`), `cascade_trt_reg/RESULTS.md` | |
| Fig. 7 | failures of each output-head arm's composition against q | `head/results/{surface,bb72}/pair_curves.csv` | figures |
| Table XI | every comparison of the output-head study | `head/results/{surface,bb72}/comparisons.csv` | output heads |
| Table XII | provenance of the claims | the plans: `results/r17/PLAN_R17.md`; `bb/bb72/ising_compare/bb/PLAN.md`, `STAGE3_PLAN.md`, `AMENDMENT_S.md`; `bb/gross/ising_compare/bb/STAGE3_GROSS_PLAN.md`; `bb/gross_stage4/ising_compare/bb/STAGE4_GROSS_PLAN.md`; `cascade/bb72/cascade_compare/PHASE_C_PLAN.md`, `phase_d/PHASE_D1_PLAN.md`; `cascade/d13/SURFACE_PLAN.md`; `d9_shallow/PLAN.md`; `cascade_trt_reg/RULES.md`; the test-set manifests in `evaluation/testsets/` | |
| Table XIII | medians of the shot descriptors (exploratory) | `results/discordant/data/tables.md`, `data/surface/summary.json` | |
| Fig. 8 | matching's class gap at d = 13 (exploratory) | `results/discordant/data/surface/fig_d13.npz` | figures |
| Fig. 9 | where CoDecoder fails, on shots read before any test | `bb/bb72/ising_compare/data/bb/fullroute/bb72_cal.json`, `policies/bb72_p0.003_rep2.json`, `bb/gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md`, `results/surface/development/a2_summary.json` | figures |
| Table XIV | names in the paper and in this repository | `weights/networks.csv`, this README | |
| Table I (the Cascade table), rows "CoDecoder + Cascade, joint decision" and note a | the joint decision's failures, times and speed-ups (held out, not registered) | failures: `joint/table1_compiled/out/` (`bb72_counts.json`, `d13_counts.json`) and, at d = 13, `joint/table1/out/table1.json`; times: `joint/timing/` (`timing.json`, `timing_cells.md`, `records/`) | joint decision |
| Sec. III B, "Joint decision on routed shots" | the rule, and its weight fitted on separate fresh shots and frozen | `joint/README.md`, `joint/fit_bb72/out/analysis/summary.json`, `joint/fit_d13/out/analysis/summary.json` | joint decision |
| Sec. IV D, "A joint decision recovers routed failures" | the joint decision against Cascade alone, and the time it adds | `joint/table1_compiled/out/`, `joint/timing/` (`timing.json`; the equal-clock run in `records/RTX/out/order_rtx_AJ56/`) | joint decision |
| Sec. VI A | the joint decision's repairs and breaks, and its answers that neither decoder chose | `joint/table1_compiled/out/` | joint decision |
| App. E, "The joint decision." and "Timing the joint decision." | the fits, the share of shots whose decisions differ, the held-out recovery, the Holm procedure; the timing session | `joint/fit_bb72/`, `joint/fit_d13/`, `joint/table1_compiled/out/`, `joint/timing/` (`timing.json`, `stage1.json`) | joint decision |
| Sec. VIII, "Better scores for the joint decision" and "Training the network for its fallback" | exploratory: a joint decision scored by matching's complementary gap; a perfect choice between the two answers at d = 13 | `joint/soft_output_pm/out/arb5_results.json`, `joint/fit_d13/out/analysis/summary.json` | joint decision |
| Table XII, row "Joint decision on the routed shots" | its status: held out, not registered | `joint/README.md`, `joint/fit_bb72/PLAN.md`, `joint/fit_d13/PLAN.md` | |
| abstract, Sec. IX, Sec. X | 14-18 % fewer failures than Cascade alone on BB72 with the joint decision | `joint/table1_compiled/out/bb72_counts.json` | joint decision |

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
python -m timing.composite                                                      # the earlier tests' speed scores from their recorded components
python -m convlines.testset regen --set T2 --d 9 --chunks 0-99 --out shots/T2    # rebuild T2 at d = 9 bit for bit
python -m convlines.evaluate --set T2 --d 9 --chunks 0-99 --shots shots/T2 \
    --opponents "plain PM" "correlated PM" --out d9.json                         # ours against matching, with the statistics
```

`timing.composite` recomputes the earlier tests' composite times and checks their speed cells against the records;
SPEED-RT supersedes these times. `convlines.evaluate` gives the failures, the paired test and the interval of each
comparison, which `results/surface/s2_test_eval.csv` records. `results/surface/SURFACE.md` covers the rest: the other
distances and test sets, the Ising and BP-OSD opponents, the deployed engines and their earlier timing, the CPU term on
the earlier Grace basis, the two post hoc controls, study 4 and R17 (whose statistics it recomputes from the recorded
counts), and what is not established.

SURF-v2 ran from its own copy of this package, `surface_v2/`, which adds the retrained networks, the set manifests
`sv2cal`, `sv2dry` and `sv2test`, and `surf_v2/` (the registration, its code and its records); run SURF-v2's commands
there. The copy in `evaluation/` is an earlier state of the package, and the commands above for the released networks
use it. [Registered tests and how to verify them](REGISTERED_TESTS.md#registered-tests-and-how-to-verify-them) shows how to check SURF-v2.

### BB codes

`bb/BB.md` gives, for BB72 and both gross-code tests, how each test ran, its disclosed changes, and the commands that
rebuild its test shots, read the networks and the opponents, and analyse them. The records of these three tests hold
every decoder's per-shot predictions (`test_outputs_BB72-2.npz` and `test_outputs_BB72-3.npz` in
`bb/bb72/ising_compare/data/bb/stage3/`, `test_outputs_G-4.npz` in `bb/gross/ising_compare/data/bb/stage3_gross/` and
`test4_outputs_G-4RF.npz` in `bb/gross_stage4/ising_compare/data/bb/stage4_gross/`) and the timing rows, so the
statistics can be checked without decoding again. RETEST's reruns of the BB72 test and of the gross code's stage-4
test keep theirs in `retest/run/bb/stage3/outputs/` and `retest/run/bb/stage4/outputs/`. The other tests' full
per-shot predictions are not in this repository ([What is not included](CHECKS.md#what-is-not-included)).

### Cascade

```bash
python cascade/bb72/repro/recompute_numbers.py --out /tmp/cascade_numbers        # BB72: every number of the comparison from its records
python cascade/d13/surface/scripts/results_tables.py cascade/d13/out             # d = 13: the tables of cascade/d13/RESULTS.md
```

`recompute_numbers.py` compares each recomputed number with its record and with the value printed in the paper at the
time it was written (CPU, under a minute). `cascade/bb72/CASCADE.md` describes the BB72 comparison: how it ran, where
each number comes from, the per-shot files and the GPU check of the networks (`cascade/bb72/repro/check_networks.py`).
`cascade/d13/RESULTS.md` and `cascade/d13/SURFACE_PLAN.md` do the same for d = 13. The speed numbers of both comparisons
in the paper come from SPEED-RT's stage 1 (`speed_rt/cascade/results/`), not from these records.

### Joint decision

```bash
python3 joint/check_paper_numbers.py        # every number the paper prints about the joint decision, from joint/
```

The script reads only the records in `joint/` and compares each number with the paper's text (standard library, under
a second). `joint/README.md` describes the rule, the code that carries it (commits 824afab, 0e9321e, 5462f4e and
f03266d), the seven studies behind the numbers, the commands that rebuild their outputs, and what is not included.
Each study's own README says which numbers it backs and how to rerun it.

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
