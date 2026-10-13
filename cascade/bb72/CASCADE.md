# ConvLines in front of our re-implementation of Cascade on BB72

The third kind of fallback is a large neural decoder. We re-implemented Cascade (A. Gu, J. P. Bonilla Ataides, M. D.
Lukin and S. F. Yelin, arXiv:2604.08358) on our BB72 circuits with 14 blocks of width 512 (19,687,937 parameters) and
trained one copy per noise setting: C14 under our uniform circuit noise and C14′ under `depspam_fm`, a variant with
extra depolarizing noise after resets and before measurements and a noisy final measurement, fitted to the Tesseract,
BP-OSD and Relay-BP failure rates that Cascade's paper reports. Below, Cascade means this re-implementation, not the
published network or its numbers. In front of it run three-block ConvLines-BB networks (S55 and S55′, 483,357
parameters) and, in the fitted setting, a six-block network (S55′-deep, 841,053 parameters). The six-block Cascade
networks C6 and C6′ (12,335,617 parameters) were secondary opponents.

Everything for this comparison is in [`cascade/bb72/`](./): the re-implementation for inference with its eager and
compiled paths, the seven networks (`weights/cascade_bb72/`), the BB72 front ends' engines and the multi-shot engine, the
comparison's code (routing, evaluation, the Relay-BP chain, timing), the plans, locks, amendment and freeze, and every
record behind a reported number, with two scripts that recompute the numbers and re-decode recorded shots. The
training code and the training records are not part of this release, and neither is the surface-code Cascade
network, whose evaluation is not done. The exploratory curve sets, which set BB72's networks against Cascade's
published curves, are in `cascade/bb72/ising_compare/bb/cascade/`, with their records in
`cascade/bb72/ising_compare/data/bb/cascade/`.

Three tests ran, all on one RTX PRO 6000 for timing:

- Phase C, the first registered test, fixed every configuration in a lock on development shots and judged each
  accuracy verdict once on fresh confirmation shots, under one Holm procedure over 132 comparisons at α = 0.05. An
  amendment changed the lock's selection rule after the development curves had been seen. Under the amended lock in
  the fitted setting, the pair with the three-block network was level with Cascade at p = 0.2 % and worse at 0.3 %;
  the chain that sent the routed shots Cascade was least sure of on to full-DEM Relay-BP was better at both points.
- An exploratory look evaluated 52 configurations on phase C's confirmation reads, without adjusting for multiplicity.
  It found the six-block network's pairs, which the lock had picked before the amendment, level or better.
- D1, a registered follow-up, froze its plan, its lock and its code in one commit before drawing any of its 28,000,000
  fresh shots and tested those two configurations. All four of its comparisons were "better": the pair failed less
  often than Cascade by about 0.75 % (p = 0.2 %) and 0.35 % (0.3 %) of Cascade's block failure rate, and was faster in
  both modes.

The table shows the fitted setting (`depspam_fm`). Routed is the percentage of all shots that the network sends to
Cascade and, for a chain, then the percentage that goes on to full-DEM Relay-BP. The times are the mean single-shot
latency and the batched cost per syndrome round (6 rounds per shot), each the slowest of three processes with Cascade
alone in the same job.

| test, p, shots | decoder | routed (%) | failures | verdict | single shot (µs) | batched (µs/round) |
|---|---|---|---|---|---|---|
| phase C, 0.2 %, 5,000,000 | Cascade alone (C14′) | | 3,858 | | 1,696.5 | 11.98 ᶜ |
| | pair (S55′ → C14′) | 1 | 3,896 | level | 105.5 | 0.89 |
| | chain (S55′ → C14′ → Relay-BP) | 2, 0.1 | 3,596 | better | 254.3 ᵃ | 1.70 ᵃ |
| phase C, 0.3 %, 1,000,000 | Cascade alone | | 4,459 | | 1,700.8 | 11.98 |
| | pair | 5 | 4,550 | worse | 178.2 | 1.38 |
| | chain | 5, 0.25 | 3,957 | better | 882.4 ᵃ | 10.66 ᵃ |
| D1, 0.2 %, 11,000,000 | Cascade alone | | 7,912 ᵇ | | 1,759.8 ᶜ | 12.00 ᶜ |
| | pair (S55′-deep → C14′) | 2 | 7,853 ᵇ | better | 163.2 | 1.355 |
| | the same, multi-shot engine | | | | 138.8 ᶜ | 0.734 |
| D1, 0.3 %, 17,000,000 | Cascade alone | | 76,676 ᵇ | | 1,763.8 ᶜ | 12.01 |
| | pair | 10 | 76,409 ᵇ | better | 300.5 | 2.332 |
| | the same, multi-shot engine | | | | 276.6 | 1.710 |

ᵃ Timed with one Relay-BP decoder per process, seeded once with relay-bp's default seed 0. The accuracy reads built a
new decoder for every routed shot, seeded with the shot's index in the read, so the chain's accuracy and its time
come from two stochastic implementations of it. ᵇ On Cascade's compiled single path. On its compiled batched path the
pair and Cascade fail 7,858 and 7,918 times at 0.2 % and 76,445 and 76,711 times at 0.3 %; on its eager path, 7,855 and
7,922, and 76,469 and 76,730. ᶜ At least one of its processes ran below the 97 % clock rule.

- Phase C: the excess block failure rate of the pair stayed below the frozen margin (10 % of Cascade's development
  rate) at both points, the plan's secondary statement. At 0.3 % the network broke 155 of Cascade's correct decisions
  on the shots it kept and repaired 64. Pairs with full-DEM Relay-BP as the fallback failed more often than Cascade
  alone at both points in both settings. In the uniform setting the pair was level at 0.2 % and worse at 0.3 %, the
  chain worse at 0.2 % and better at 0.3 %, and no tested decoder was better than Cascade at both points.
- D1: on the single path the pair failed less often than Cascade by 5.36e-6 per shot at 0.2 % (95 % interval
  [3.59, 7.16]e-6) and by 1.57e-5 at 0.3 % ([1.33, 1.81]e-5). Cascade's failures on the shots the network kept were
  4.0 % and 3.0 % of all its failures, so routing could not gain more than that. The pair's 99.9th-percentile
  single-shot latency was 3.5-7.7 % above Cascade's.
- Multi-shot engine, batch 4,096: a three-block network takes 1.883 µs per shot host to host (0.314 µs per round), 2.20
  times faster than the BB engine on the same batches; Cascade alone takes 11.96 µs per round. Phase C's pairs take
  0.510 and 0.999 µs per round at 0.2 and 0.3 %, 23.5 and 12.0 times less than Cascade alone in the fitted setting
  (23.6 and 12.0 in the uniform one).
- On an H100 (secondary, speed only) Cascade alone takes 40.4-41.2 µs per shot in batches and a three-block network
  2.82 µs per shot at 4,096 (1.88 on the RTX PRO 6000); phase C's pairs are 7.6-11.4 times faster than Cascade alone
  there, and the chains at 0.3 % slower than it. Every batched H100 value of the pairs and of Cascade involves
  processes below the clock rule, and the card's clock samples record its 700 W power cap as the only throttle reason
  besides idle.
- Curve sets (exploratory, `ising_compare/bb/cascade/CASCADE_CURVES.md`): BB72's registered networks r11 and r12
  (`weights/bb/`), alone and with r12's least confident shots routed to full-DEM Relay-BP, on fresh shots at ten
  rates from 0.1 to 0.55 %, set against the values digitised from Cascade's Fig. 2(a). At 0.1 % the least confident
  0.29 % of 10,000,000 shots held 344 of r12's 348 failures; full-DEM Relay-BP failed on 469 of them, so routing
  raised the failures to 473.

## How the comparison was run

- Shots: Stim samples of relay-bp's BB72 X-memory circuit at revision d185194 (uniform) or of its `depspam_fm`
  rewrite (`ising_compare/bb/cascade/noise.py`), at any p, from `bb.cascade_data.eval_seed(kind, p, chunk, family)`:
  kind 10 for the development curves, 11 for phase C's confirmation, 12 for timing and the claim-point check, 13 for
  dry runs and 14 for D1. Every read records each chunk's seed and digest and the circuit's sha256.
- Phase C (`cascade_compare/PHASE_C_PLAN.md`, `phase_c/RUN.md`, `phase_c/RESULTS.md`): development reads of every
  network gave the curves and fixed the lock (`phase_c/out/locked_<setting>.json`): each configuration, its
  thresholds and the non-inferiority margin. Amendment 1 replaced the selection rule after the development curves had
  been seen and before any confirmation shot was read. The claim-point check (`out/check_<setting>.json`) tested that
  the timed paths decide as the reads do, and passed every row; its records were committed after the confirmation
  read, while job dependencies and lock hashes kept the order of the steps. One read of the confirmation shots
  (`out/confirm_<setting>.json`) decided each verdict with the exact McNemar test and Newcombe's interval; the report
  is `out/report/phase_c_summary.json` and `phase_c_tables.md`.
- Timing (phase C and D1): end to end, from shots in host memory to decisions on the host, on Alpine's RTX PRO 6000
  (driver 590.48.01, torch 2.11.0+cu128, Triton 3.6.0). Single mode decodes 20,000 shots one at a time after 200
  untimed ones; batched mode decodes 500,000 shots with the networks at batch 4,096 and Relay-BP in 16 worker
  processes, routed shots queuing for Cascade until 4,096 wait. Each value is the slowest of three fresh processes in
  a rep-major loop with Cascade alone in the same job and loop, reported whether or not the SM clock stayed at or
  above 97 % of its maximum; every row records its lowest clock (`phase_c/out/speed/rows.jsonl`,
  `phase_d/out/speed/rows.jsonl`).
- The exploratory look (`phase_c/EXPLORE_DEEP.md`, `phase_c/explore/`) is post hoc and not registered.
- D1 (`phase_d/PHASE_D1_PLAN.md`, `phase_d/D1_RESULT.md`): the plan, the lock (`phase_d/d1_locked_depspam_fm.json`)
  and the code were frozen in commit F = 310ce8e0f8bfb9d3b97b27186547a836d29fc9ca, which reached the hub before any
  read job was submitted. The shots come from seed kind 14 at chunk offsets fixed by F's hash (from chunk 6,433,102).
  Cascade was read on its eager accuracy path and on the two compiled paths that the timings run (compiled@1 and
  compiled@4096, `d1_compiled.py`); each compiled comparison joins the single and the batched comparison as an
  intersection-union test, and one Holm procedure at 0.05 judged the two eager and the two compiled comparisons
  (`d1_confirm.py`, `phase_d/out/confirm/run_1/`). The timing ran on both ConvLines engines.
- Multi-shot engine (`cascade_compare/batchopt/`, `M2.md`, `mfused.py`): a second Triton engine runs the BB72
  networks with every kernel serving several shots, with launch settings in six tiers by batch size. In 88,750,000
  shot evaluations of five networks at 14 batch sizes from 1 to 4,100, no decision and no FP32 margin bit pattern
  differed from the BB engine's (repeated evaluations of the same shot sets). Its times are the slowest of six
  processes (`out_m2/`; the per-process rows are in `out_m2/rows/`).
- H100 (secondary): two rented H100 pods timed phase C's pipelines (`phase_c/H100_TIMING.md`, `phase_c/out_h100/`),
  and an H100 session timed the multi-shot engine (`M2.md` section 9, `batchopt/out_h100/20261011_204455/`).
- Noise-model fit (`cascade/bb72/noise_fit/`, `CASCADE_CAL.md`): Tesseract, BP-OSD and Relay-BP decoded the circuits of
  the candidate noise models on the Mac mini and on Alpine; `table.py` turns the run rows in `runs/` into
  `cascade_cal_counts.json` and `table.md`. In the fitted setting the published rates fall inside our 95 % intervals,
  a fit that rests on 17-20 Tesseract failures. `ising_compare/bb/cascade/noise.py` is the fit's `noise.py` with one
  optional argument added for the comparison's circuits at any p.
- Curve sets (`ising_compare/bb/cascade/`: `CASCADE_CURVES.md`, `curves.py`, its job scripts and `args/`; records in
  `ising_compare/data/bb/cascade/`): Stim drew fresh shots from the curve sets' own seeds at each p (`shots_w2.csv`
  lists every chunk with its seed and sha256), r11 and r12 read them in FP32 on a MIG slice of an RTX PRO 6000,
  full-DEM Relay-BP decoded the shots r12 routed (and, at four rates, the first chunks on its own), and
  `curves.py report` turned the per-shot predictions into `curves_counts.json` and `curves.png`. Nothing was frozen or
  registered in advance.

## Disclosed changes and limits

- Amendment 1 changed the lock's selection rule after the development curves had been seen, so phase C's selection
  was not preregistered. D1's hypotheses came from the exploratory look, after phase C's results were known.
- D1 read only the fitted setting, at two points, on one GPU model. The H100 values are secondary and speed only.
- Cascade runs in BF16. The published network's numbers, including the FP8 H200 times of its paper, enter no
  comparison.
- The released networks are inference copies of the checkpoints the comparison read: every tensor is byte for byte
  the original's, and each file records the original's sha256 (`source_sha256`), which is the sha256 that the locks,
  reads and timing rows give. The locks here differ from the hub's in paths, so their sha256 is not the one the
  records hold (`cascade/bb72/SANITIZE.txt`).
- The training program is not released. `bb.cascade_eval` takes its two statistics helpers from `bb/cascade_stats.py`
  (copied from the training program), `bb.cascade_data` keeps only the evaluation shots, and `cascade_eval
  --bn-recal`, which no read of the comparison used, does not run.
- The multi-shot engine's per-process rows (`batchopt/out_m2/rows/`) and D1's job logs (`phase_d/out/logs/`) were
  not on the hub; they come from the cluster (`cascade/bb72/SOURCE`).

## Where each number comes from

`python cascade/bb72/repro/recompute_numbers.py --out DIR` recomputes every number below from the records with the
release's own code (`pc_report`, `pc_common.holm`, `d1_confirm.speed`, `m2.py report`, `noise_fit/table.py`,
`bb/cascade/curves.py`) and prints it beside the record's value and the paper's (`DIR/numbers.md`).
`python cascade/bb72/repro/check_networks.py` (one GPU) gives the parameter counts and the six tiers. Paths are under
`cascade/bb72/cascade_compare/` unless they start with `cascade/` or `weights/`.

| paper | numbers | records |
|---|---|---|
| Abstract, introduction | 23 times the parameters; 0.75 % and 0.35 %; 0.73 and 1.71 µs per round against 12.0; 28 million shots | `weights/cascade_bb72/`; `phase_d/out/confirm/run_1/d1_result.json`; `phase_d/out/speed/rows.jsonl` |
| Our Cascade | 19,687,937, 483,357 and 841,053 parameters | `weights/cascade_bb72/` |
| | 17-20 Tesseract failures; the published rates within errors | `cascade/bb72/noise_fit/runs/`, `cascade_cal_counts.json` |
| | published P_L inside our 95 % intervals at 0.1-0.3 %, below them at 0.4 % | `phase_c/out/report/phase_c_summary.json` (curves); `phase_c/pc_report.py` (PAPER) |
| Phase C | 132 comparisons; 1 % and 5 % routed; level and worse; 4,550 against 4,459; the margin at both points; 155 and 64; the uniform verdicts; Relay-BP pairs worse | `phase_c/out/confirm_<setting>.json`, `phase_c/out/report/phase_c_summary.json` |
| | 9.5-16.1 and 8.7-13.5 times | `phase_c/out/speed/rows.jsonl` |
| Relay-BP chain | 3,596 against 3,858; 3,957 against 4,459 | `phase_c/out/confirm_depspam_fm.json` |
| Exploratory look | 52 configurations; 3.5 to 11 times; 3,822 against 3,858 (p = 2.1e-7); 4,452 against 4,459 (p = 0.12) | `phase_c/explore/explore_deep.json` |
| D1 | 11,000,000 and 17,000,000 shots; four verdicts; 5.36e-6 [3.59, 7.16]e-6 and 1.57e-5 [1.33, 1.81]e-5; 0.75 % and 0.35 %; 4.0 % and 3.0 % | `phase_d/out/confirm/run_1/d1_result.json` |
| | 5.9-12.7 and 5.2-16.4 times (recomputed 5.15-16.36); the clock flags; 3.5-7.7 % | `phase_d/out/speed/rows.jsonl` |
| Table of the comparison | every failure count, verdict and time | as for phase C and D1 above |
| Multi-shot engine | six tiers; 88,750,000 evaluations of five networks at 14 batch sizes from 1 to 4,100 | `batchopt/mfused.py`; `batchopt/out_m2/rows/m2_CJ55/screen.jsonl` |
| RQ2 | 1.883, 0.314, 2.20, 0.448 and 11.96 µs; 0.734, 1.710, 12.00, 12.01, 16.4 and 7.0; 0.510, 0.999, 1.74, 1.38, 23.5 and 12.0; about 0.04, 17 and 7 | `batchopt/out_m2/summary.json` (rebuilt from `out_m2/rows/`); `phase_d/out/speed/rows.jsonl` |
| | within 4.3 % of the predictions | `phase_d/PHASE_D1_PLAN.md` section 4.3 (the printed predictions; 4.8 % against its formula unrounded), `d1_result.json` |
| RQ3, discussion | 97 % of Cascade's failures on the routed 10 %; 3.0 % left | `phase_d/out/confirm/run_1/d1_result.json` |
| | BB72 at p = 0.1 % (exploratory): 0.29 % of 10,000,000 shots; 344 of 348; 469; 473 | `cascade/bb72/ising_compare/bb/cascade/curves_counts.json` |
| | 129.6 ms against 1.72 ms | `phase_c/out/speed/rows.jsonl` |
| | about 40 µs; 2.82 against 1.88 µs; 7.6-11.4 times; chains slower; the clock flags; the 700 W cap | `batchopt/out_h100/20261011_204455/` (`summary.json`, `env/clock_samples.csv`) |
| Methods | 20,000 and 500,000 shots; 16 Relay-BP workers; 94.1-94.7 % (D1); 96.3-96.6 % and the accepted processes (phase C) | `phase_c/out/speed/rows.jsonl`, `phase_d/out/speed/rows.jsonl` |
| Implementation | the training shots and hours | the training records, not part of this release |

## Per-shot files

The reads' per-shot outputs are not in git. Phase C's 68 files are listed with their sha256 in
`cascade/bb72/cascade_compare/phase_c/out/npz_sha256.txt` (paths relative to the run directory's `out/`) and are archived
on the cluster with the project's other per-shot files (the archive's manifest is `revision/ARCHIVE_MANIFEST.md` on
the research branch `ising-rev-archive`). D1's four files are listed in `phase_d/out/npz_sha256.txt`; they are in the
same archive directory, which the manifest at 8aa1b171 does not list yet, and the release check
(`cascade/bb72/REPRODUCTION_CHECK.md`) verified their sha256
there. `check_networks.py` re-decodes recorded shots from both.

The curve sets' 1,285 shot chunks and 7,310 prediction files (r11's and r12's outputs per chunk, Relay-BP's per
shard) are archived as the manifest's `cascade_w2/` entries. `cascade/bb72/ising_compare/data/bb/cascade/shots_w2.csv`
gives each chunk's seed and sha256, and `preds_sha256.txt` beside it the sha256 of each prediction file.
`check_networks.py --curves` re-decodes two of the chunks, and `curves.py report` rebuilds `curves_counts.json` from
the predictions.

<details>
<summary>D1's per-shot files</summary>

```
ceb3ab405dc490dea9599ae02aa5095f923d066a9649c79e2517babe59d779a3  out/read/p0.00200.npz
360e9fb38c0df4c49d219aecad47a1ba52f417992393c7e6f0a0cfe6db001321  out/read/p0.00300.npz
d90acc56b62e7651cde2659d33fdd43fefc8f55de3b0f74380b04dcf9ebc367f  out/compiled/p0.00200.npz
5ae624ca683e6525c583952107aeda4c26856352e5072206942ff3c79e652340  out/compiled/p0.00300.npz
```

</details>

## Reproduce

The comparison ran with Python 3.12, torch 2.11.0+cu128, Triton 3.6.0, Stim 1.16.0, PyMatching 2.4.0, SciPy 1.18.1
and NumPy 2.2.6 (`cascade/bb72/cascade_compare/batchopt/req_cu128.txt` pins them), plus relay-bp 0.2.2 for the Relay-BP
chain. `QND_RELAY` must point at relay-bp's repository at d185194, whose `tests/testdata` holds the circuits.

```bash
# every verdict, count, ratio and time from the records (CPU, no shots): DIR/numbers.md
python cascade/bb72/repro/recompute_numbers.py --out DIR

# the seven networks, and the curve sets' r11 and r12, on recorded shots (one RTX PRO 6000; ARCHIVE holds the
# per-shot files above)
QND_RELAY=/path/to/relay-bp python cascade/bb72/repro/check_networks.py \
    --archive ARCHIVE/cascade_w4/out --d1 ARCHIVE/cascade_d1/out --curves ARCHIVE/cascade_w2 --out DIR

# the curve sets' counts, tables and plot from their archived predictions (CPU, needs matplotlib; reads
# ARCHIVE/cascade_w2/preds and out/taus.json, which is data/bb/cascade/taus.json byte for byte)
cd cascade/bb72/ising_compare
W2_DIR=ARCHIVE/cascade_w2 PYTHONPATH=.:../qldpc_neural QND_RELAY=/path/to/relay-bp \
    python -m bb.cascade.curves report --out DIR
cd ../../..

# a read with the comparison's evaluation path (phase C's pc_eval.py: Cascade in BF16, ConvLines on the fused engine),
# here confirmation chunks 0-9 at p = 0.3 % in the fitted setting; --save writes the per-shot arrays
cd cascade/bb72/ising_compare
PYTHONPATH=.:../qldpc_neural:../cascade_compare/phase_c QND_RELAY=/path/to/relay-bp \
    python ../cascade_compare/phase_c/pc_eval.py --ckpt ../../../weights/cascade_bb72/C14p.pt,../../../weights/cascade_bb72/S55p_deep.pt \
    --family depspam_fm --p 0.003 --kind 11 --chunks 0:10 --save DIR
```

`pc_eval.py` refuses any card but the RTX PRO 6000 (phase C's GPU). C6 and C6′ were read under torch.compile
(`pc_common.EVAL_COMPILED`); `check_networks.py` sets that up for the released files. The steps that drew and judged
the shots (`pc_route.py`, `d1_confirm.py`, the job scripts in `phase_c/jobs/` and `phase_d/jobs/`) are here as they
ran; they expect the cluster's directories and the hub's locks.
