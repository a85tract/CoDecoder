# CoDecoder: code, weights and records

This repository holds the artifacts of the paper "CoDecoder: Make Quantum Decoders Up to 100× Faster and Even More
Accurate" by Yicheng Guang, Yuliang Xiong, Xun Gao and Yueqi Chen (University of Colorado Boulder),
[arXiv:2610.28748](https://arxiv.org/abs/2610.28748): the training code, the trained networks, the test sets and the
records behind every table and figure.

## What CoDecoder is

CoDecoder is a small neural network that works in front of an existing strong decoder and leaves that decoder
unchanged. It decodes every shot (one run of the error-correction circuit) and reports a confidence. A shot whose
confidence falls below a frozen threshold goes to the strong decoder, the fallback, whose answer replaces the
network's. We write CoDecoder for the network alone and CoDecoder + X for its composition with the strong decoder X.
The network's output head scores the flips of many shifted copies of the code's logical operators; on
bivariate-bicycle (BB) codes it decides all logical qubits together.

CoDecoder was called ConvLines while the experiments ran. Code names (`convlines/`, `convlines_train/`), seed tags,
study documents and records keep that name, because the frozen commands, seeds and sha256 manifests refer to them.

## Main results

Each composition against its own strong decoder alone, on the same fresh test shots (paper Section IV and Table I):

| strong decoder X | code and noise | failures of CoDecoder + X | time of X over CoDecoder + X |
|---|---|---|---|
| correlated PyMatching (PM) | rotated surface code, d = 7 to 17, p = 0.6 % | 6.4 % to 31.4 % fewer | 2.3 to 10.3 |
| full-DEM Relay-BP | BB72 [[72,12,6]] at p = 0.2 % and 0.3 %, gross code [[144,12,12]] at p = 0.4 % | 22 % to 50 % fewer | 8.9 to 124 |
| our Cascade for BB72, joint decision | BB72 under a fitted noise model, p = 0.2 % and 0.3 % | 18 % and 14 % fewer | 5.14 to 10.08 (RTX PRO 6000), 3.05 to 9.68 (H100) |
| our Cascade for the surface code, joint decision | d = 13, p = 0.35 % and 0.6 % | 1.0 % to 2.0 % fewer | 3.56 to 7.65 (RTX PRO 6000), 3.30 to 7.06 (H100) |

- Every speed-up compares the slowest run of the composition with the fastest run of the strong decoder alone. The
  Cascade ranges cover single shots and batches.
- The Cascade rows use the [joint decision](#joint-decision) on the routed shots. Its weight was fitted on separate
  fresh shots and frozen before the test shots, so these rows are held out, not registered, and they were timed in a
  later session against Cascade alone timed again there. Without the joint decision, the composition is better than
  Cascade alone or non-inferior to it in all four settings and 3.17 to 10.23 (BB72) and 3.33 to 7.66 (d = 13) times
  faster.
- With Cascade compiled by TensorRT, every composition still took 2.3 to 16.3 times less time per shot than Cascade
  alone ([CASCADE-TRT-REG](docs/REGISTERED_TESTS.md#cascade-trt-reg-cascade-on-tensorrt)).
- On BB72, the same network with a conventional output head fails 3.8 to 5.0 times as often.
- Cascade's code and trained networks are not public, so we re-implemented it from its paper. [Our Cascade against the published one](#our-cascade-against-the-published-one) compares the two.

### CoDecoder alone already beats the usual fallbacks

**Without any fallback, the small network is already more accurate and faster than the decoders that earlier
hierarchical designs fall back on.** Pre-decoders such as Clique, Promatch, LATTE, NVIDIA's Ising and Chongder pass
their remaining shots to minimum-weight matching, which is PyMatching in our tests. From the paper's surface-code
table (p = 0.6 %) and BB-code table:

| compared with | CoDecoder alone |
|---|---|
| PyMatching, surface code d = 7 to 17 | 1.5 to 4.8 times fewer failures at every distance (3,296 against 15,726 at d = 17), in 1.1 to 3.5 times less time per shot |
| NVIDIA's Ising-Accurate pre-decoder + PyMatching, d = 11 to 17 | 2.2 to 2.5 times fewer failures |
| correlated PyMatching, d = 11, 13 and 15 | 8 to 26 % fewer failures |
| sector Relay-BP and BP-OSD (ldpc and CUDA-Q), BB72 at p = 0.2 % and 0.3 % | 49 to 69 % fewer failures, in 6 to 37 times less time |
| full-DEM Relay-BP, BB72 at p = 0.2 % | level with it (269 against 288 failures in 300,000 shots, p = 0.33), in 1/590 of its time |

So the fallback does not rescue a weak frontend. The frontend is itself a strong decoder, and the fallback sees only
the at most 10 % of shots it is least sure of. Where the network alone is not ahead (correlated PyMatching at d = 7, 9
and 17, full-DEM Relay-BP at p = 0.3 %, sector Relay-BP on the gross code), its composition with that decoder is. At
d = 9 the row uses the one-block network of the second d = 9 test, as the paper's table does. The counts are in
[`surface_v2/surf_v2/analysis/tables.md`](surface_v2/surf_v2/analysis/tables.md),
[`d9_shallow/RESULTS.md`](d9_shallow/RESULTS.md) and [`RETEST_RESULTS.md`](RETEST_RESULTS.md), and the times in
[`docs/SPEED.md`](docs/SPEED.md) and [`speed_rt/s2/`](speed_rt/s2/).

### Below 1 µs per syndrome round

In batches of 4,096 shots:

- CoDecoder + correlated PM decodes the surface code at p = 0.6 % in 0.187, 0.698 and 0.822 µs per round at d = 7, 9
  and 11, about the round period of current superconducting processors, against 3.22, 6.45 and 10.8 µs for correlated
  PM alone (one CPU process on the Grace basis; registered SPEED-RT stage 3, [`docs/SPEED.md`](docs/SPEED.md)).
- CoDecoder + Cascade on the multi-shot engine decodes BB72 at p = 0.2 % in 0.733 and 0.891 µs per round on an
  RTX PRO 6000 and an H100, against 12.00 and 7.34 µs for Cascade alone (paper Section VII B).
- To our knowledge, CoDecoder is the first open-source design that brings a strong decoder below 1 µs per round at a
  physical error rate as high as 0.6 % while reducing its failures.

[`docs/RESULTS.md`](docs/RESULTS.md) gives every number with its source file.

### Our Cascade against the published one

Both of our Cascades follow its paper (A. Gu, J. P. Bonilla Ataides, M. D. Lukin and S. F. Yelin,
[arXiv:2604.08358](https://arxiv.org/abs/2604.08358)). Where the paper gives a setting we used it, and where it does not,
the table says what we chose.

| | Cascade's paper | our re-implementation |
|---|---|---|
| BB72 network | its largest model has L = 14 blocks of width H = 512 (stated for the [[144,12,12]] code), no BB72-specific size | 14 blocks of width 512, 19,687,937 parameters |
| surface-code network | L ≈ d blocks, width not stated (256 in its scaling and ablation examples, 128 called deployment-sized) | d = 13: 13 blocks of width 256, 2.53 million parameters |
| training | batch 3,328, Muon and Lion, a three-stage error-rate curriculum up to p = 0.7 % on surface codes, convergence within 3 × 10^8 examples | the same batch, optimizers and curriculum, and the d = 13 network stopped at 81,130 steps, about 2.7 × 10^8 shots |
| precision | bfloat16 training, FP8 inference | BF16 |
| single-shot time per round, surface code | about 40 µs per cycle on an H200 | 29.6 µs on an RTX PRO 6000 and 33.1 µs on an H100 at d = 13 |

On BB72 the paper does not fully specify its noise model, so we fitted one (`depspam_fm`: extra depolarizing noise
after resets and before measurements, and a noisy final measurement) to the baseline failure rates the paper reports.
The rates below are logical errors per logical qubit per round (the paper's Eq. 1). The paper's values are read off its
Fig. 2(a), and ours carry 95 % intervals.

| p | decoder | Cascade's paper | ours |
|---|---|---|---|
| 0.2 % | Tesseract | 8.13e-06 | 8.17e-06 (5.29e-06 to 1.26e-05) |
| 0.2 % | Relay-BP | 2.00e-05 | 1.74e-05 (1.18e-05 to 2.56e-05) |
| 0.2 % | BP-OSD | 3.14e-04 | 3.59e-04 (2.96e-04 to 4.36e-04) |
| 0.3 % | Tesseract | 4.73e-05 | 4.73e-05 (2.95e-05 to 7.58e-05) |
| 0.3 % | Relay-BP | 9.50e-05 | 1.14e-04 (8.43e-05 to 1.55e-04) |
| 0.3 % | BP-OSD | 1.30e-03 | 1.31e-03 (1.07e-03 to 1.61e-03) |
| 0.1 % | Cascade | 8.17e-07 | 7.79e-07 (7.17e-07 to 8.46e-07) |
| 0.2 % | Cascade | 1.04e-05 | 9.74e-06 (9.04e-06 to 1.05e-05) |
| 0.3 % | Cascade | 6.58e-05 | 6.42e-05 (6.16e-05 to 6.69e-05) |
| 0.4 % | Cascade | 2.19e-04 | 2.69e-04 (2.62e-04 to 2.76e-04) |

- Every published value lies inside our 95 % interval except Cascade at 0.4 %, where the published rate is 19 % below
  ours. The Cascade rows are phase C's confirmation read. Pooled over all our reads (paper Appendix E), the published
  rates lie 3.7, 2.0 and 5.2 % above ours at p = 0.1, 0.2 and 0.3 %.
- Under our plain uniform circuit noise, Tesseract failed only 0.54 to 0.59 times as often as the paper reports, which
  is why the comparison uses the fitted model. The fit rests on 17 to 20 Tesseract failures.
- Our Cascades are not the authors' networks, and each was trained once per setting. A matching curve does not show
  that our network makes the same errors as theirs. At d = 13 the width is our choice and the circuit is our
  registered test's, so the paper has no d = 13 accuracy to compare with.
- Sources: [`cascade/bb72/noise_fit/table.md`](cascade/bb72/noise_fit/table.md) for the baselines,
  [`cascade/bb72/cascade_compare/phase_c/RESULTS.md`](cascade/bb72/cascade_compare/phase_c/RESULTS.md) for the
  Cascade rates and [`cascade/d13/`](cascade/d13/) for the surface-code network.

## How the numbers were measured

- **Accuracy.** Each registered test fixed its networks, thresholds, test-set rule and analysis and pushed them to
  GitHub before its test shots existed. The test seeds come from the drand public randomness beacon, and each test set
  was drawn once and read once ([`docs/REGISTERED_TESTS.md`](docs/REGISTERED_TESTS.md)).
- **Speed.** On the surface and BB codes, a composition's time is the network on an H100, plus the handoff, plus the
  fallback on the routed shots timed on one CPU core and converted to one core of NVIDIA's Grace CPU. The Cascade
  comparisons are timed end to end on one card, an RTX PRO 6000 or an H100 ([`docs/SPEED.md`](docs/SPEED.md)).
- [`METHODS.md`](METHODS.md) states the shots and seeds, the decoders and their settings, the statistics and the
  hardware.

## Joint decision

A routed shot discards the network's prediction, which is already computed. When the fallback also reports logits, as
our Cascades do, the joint decision lets both decoders decide together the routed shots where their decisions differ
(paper Section III B). On BB72 it answers the class that maximizes CoDecoder's class score plus w times Cascade's
log-likelihood of that class, so it can answer a class that neither decoder chose. At d = 13 it answers the sign of the
network's logit plus w times Cascade's. The weight w was fitted on separate fresh shots and frozen before it was
applied to the test shots of the Cascade comparisons, so these results are held out, not registered.

- On the same test shots, CoDecoder + Cascade with the joint decision failed 18 % (p = 0.2 %) and 14 % (p = 0.3 %) less
  often than Cascade alone on BB72, and 1.0 % to 2.0 % less often at d = 13. One Holm procedure over its eight exact
  McNemar tests against Cascade alone rejected all eight.
- Timed later on fresh shots by the rules of SPEED-RT's stage 1, the composition took at most 2.4 % longer with the
  joint decision than without it in 15 of the 16 timings. The 16th ran at a 5 % lower GPU clock, and a separate run of
  it at equal clock found 0.2 %.
- On the held-out halves of the fit shots, the joint decision recovered 59 % to 61 % of what a perfect choice between
  the two answers would recover on BB72, and 2 % to 4 % at d = 13.

[`joint/README.md`](joint/README.md) gives the code, the studies behind each number and how to rerun them.
`python3 joint/check_paper_numbers.py` prints every number the paper gives for the joint decision, from the records.

## Quick start

Linux x86-64 with Python 3.12:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
bash training/smoke.sh      # trains a tiny d = 5 surface-code CoDecoder and tunes its threshold (CPU, 5 minutes)
```

CoDecoder for your own Stim circuit, in front of a strong decoder of your choice:

```bash
export PYTHONPATH=$PWD/evaluation:$PWD/training
python -m convlines_train.train --circuit my_circuit.stim --out runs/mine
python -m convlines_train.tune --frontend runs/mine/final_ema.pt --shots 200000 --seed 7 \
    --strong convlines_train.strong:pymatching_correlated --max-fraction 0.1 \
    --rule budget --front-us 2 --budget-us 3 --out runs/mine/tune.json
```

[`docs/REPRODUCE.md`](docs/REPRODUCE.md) has the full installation (BB codes, Ising, BP-OSD, TensorRT), the hardware,
the commands behind every table and figure, and training; [`training/TRAINING.md`](training/TRAINING.md) is the
training guide.

## Layout

| path | contents |
|---|---|
| `evaluation/` | the surface-code decoder, its GPU engines and timing harnesses, the circuits and test-set manifests |
| `weights/` | the 38 released networks and their manifest |
| `training/` | training and threshold tuning for any circuit, and the frozen programs that trained the released networks |
| `results/` | the surface-code records, the figure scripts and the figures |
| `bb/`, `cascade/`, `head/` | the BB-code tests, the Cascade comparisons and the output-head comparison, each with its records |
| `retest/`, `surface_v2/`, `speed_rt/`, `d9_shallow/`, `cascade_trt_reg/` | the registered tests RETEST, SURF-v2, SPEED-RT, D9-SHALLOW and CASCADE-TRT-REG |
| `joint/` | the joint decision of CoDecoder and Cascade: its fits, counts, timing records and checks |
| `docs/`, `METHODS.md` | the details below; [`docs/REPRODUCE.md`](docs/REPRODUCE.md) has the full layout |

## Documentation

| file | contents |
|---|---|
| [`docs/RESULTS.md`](docs/RESULTS.md) | every result with its source file: the other baselines, the BB modes, latency tails, Decoder Switching |
| [`docs/SPEED.md`](docs/SPEED.md) | SPEED-RT, the registered speed re-test: stages, what a time is, hosts, the Grace basis, statistics, how to verify |
| [`docs/REGISTERED_TESTS.md`](docs/REGISTERED_TESTS.md) | D9-SHALLOW, CASCADE-TRT-REG, every registered test and how to check its registration, NVIDIA's Ising decoders |
| [`docs/REPRODUCE.md`](docs/REPRODUCE.md) | installation, hardware, the commands and files behind every table and figure, training |
| [`docs/CHECKS.md`](docs/CHECKS.md) | the checks already run, how to verify this repository, what is not included |
| [`METHODS.md`](METHODS.md) | how accuracy and speed were measured |

## License

The code and the trained network weights are licensed under the Apache License, Version 2.0 (`LICENSE`, `NOTICE`).
NVIDIA's Ising-Decoding code and checkpoints are only referenced and never redistributed; they keep their own terms
(the code under Apache 2.0, the checkpoints under their model cards on Hugging Face). PyMatching, Stim and Relay-BP keep
their own licenses.

## Citation

```bibtex
@misc{guang2026codecoder,
  title         = {CoDecoder: Make Quantum Decoders Up to 100x Faster and Even More Accurate},
  author        = {Guang, Yicheng and Xiong, Yuliang and Gao, Xun and Chen, Yueqi},
  year          = {2026},
  eprint        = {2610.28748},
  archivePrefix = {arXiv},
  primaryClass  = {quant-ph},
  note          = {Code, weights and records: \url{https://github.com/a85tract/CoDecoder}}
}
```

`CITATION.cff` holds the same.
