# CoDecoder: code, weights and records

This repository holds the artifacts of the paper "CoDecoder: Make Quantum Decoders Up to 100× Faster and Even More
Accurate" by Yicheng Guang, Yuliang Xiong, Xun Gao and Yueqi Chen (University of Colorado Boulder): the training code,
the trained networks, the test sets and the records behind every table and figure.

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

Each composition against its own strong decoder alone, on fresh test shots, in registered tests:

| strong decoder X | code and noise | failures of CoDecoder + X | time of X over CoDecoder + X |
|---|---|---|---|
| correlated PyMatching (PM) | rotated surface code, d = 7 to 17, p = 0.6 % | 6.4 % to 31.4 % fewer | 2.3 to 10.3 |
| full-DEM Relay-BP | BB72 [[72,12,6]] at p = 0.2 % and 0.3 %, gross code [[144,12,12]] at p = 0.4 % | 22 % to 50 % fewer | 8.9 to 124 |
| our Cascade for BB72 | BB72 under a fitted noise model, p = 0.2 % and 0.3 % | 1.1 % and 0.3 % fewer | 5.1 to 16.3 (RTX PRO 6000), 3.1 to 9.1 (H100) |
| our Cascade for the surface code | d = 13, p = 0.35 % and 0.6 % | 0.4 % to 1.3 % fewer | 3.6 to 7.6 (RTX PRO 6000), 3.3 to 7.1 (H100) |

- Every speed-up compares the slowest run of the composition with the fastest run of the strong decoder alone.
- With Cascade compiled by TensorRT, every composition still took 2.3 to 16.3 times less time per shot than Cascade
  alone ([CASCADE-TRT-REG](docs/REGISTERED_TESTS.md#cascade-trt-reg-cascade-on-tensorrt)).
- On BB72, the same network with a conventional output head fails 3.8 to 5.0 times as often.
- Cascade's code and trained networks are not public, so we re-implemented it from its paper.

[`docs/RESULTS.md`](docs/RESULTS.md) gives every number with its source file.

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
  title  = {CoDecoder: Make Quantum Decoders Up to 100x Faster and Even More Accurate},
  author = {Guang, Yicheng and Xiong, Yuliang and Gao, Xun and Chen, Yueqi},
  year   = {2026},
  note   = {Code, weights and records: \url{https://github.com/a85tract/CoDecoder}}
}
```

`CITATION.cff` holds the same.
