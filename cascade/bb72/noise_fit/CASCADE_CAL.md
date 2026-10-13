# Do our BB72 circuits and noise match Cascade's Fig. 2(a)?

Exploratory comparison (not registered), run 2026-10-08/09. Cascade is the convolutional decoder of Gu, Bonilla
Ataides, Lukin and Yelin (arXiv:2604.08358). Its Fig. 2(a) plots, for [[72,12,6]] under "circuit-level depolarizing
noise" with R = d = 6 rounds, the logical error rate per logical qubit per round of Cascade and three reference
decoders. The paper states the configuration of two of these exactly (Tesseract, BP+OSD) but gives neither its
circuits nor its noise model. I decoded fresh shots of our frozen BB72 setting with those two decoders, configured as
the paper states, and compared the results with the study lead's digitisation of the figure.

## Verdict: not aligned

Under our frozen setting Tesseract's P_L is 0.54x the paper's at p = 0.2 % and 0.59x at 0.3 %. Both 95 % intervals
exclude the paper's value (ratio intervals 0.34-0.86 and 0.38-0.91), so Tesseract fails the criterion at both p.
BP+OSD agrees within 20 % (0.88x and 0.94x). Our setting is therefore easier than the paper's: Tesseract fails
about 1.75 times less often in it.

I tried three alternative noise models on the same circuits:
- A noisy final data measurement alone (`uniform_fm`) changes BP+OSD by at most 6 % and cannot close the gap.
- Measurement and reset flips at 2p (`spam2_fm`) overshoot. Tesseract comes out at 1.71x at 0.3 % (interval
  1.19-2.45) and BP+OSD at 1.30x at 0.2 % (1.09-1.56).
- Adding DEPOLARIZE1(p) after every reset and before every measurement, on top of each flip p and with the final
  data measurement noisy as well (`depspam_fm`, an effective flip of about 5p/3), reproduces the paper. Tesseract
  gives 1.01x and 1.00x, BP+OSD 1.14x and 1.01x, at 0.2 % and 0.3 %. Our full-DEM Relay-BP gives 0.87x and 1.20x
  of the paper's Relay curve under it.

The matches have wide intervals: the Tesseract points under `depspam_fm` rest on 20 and 17 failures (ratio intervals
0.65-1.55 and 0.62-1.60). I picked `depspam_fm` after seeing the uniform and 2p models bracket the paper, so the fit
shows only that one of three plausible conventions reproduces the figure. The paper's actual noise model remains
unknown, and a different syndrome-extraction schedule, which I did not test, could produce the same shift.

The paper's "Relay" curve lies between our full-DEM Relay-BP (0.63-0.80x, registered results) and our sector Relay-BP
(1.88-1.99x), closer to the full-DEM one, and our full-DEM Relay-BP matches it within 20 % under `depspam_fm`. On our
circuits, decoding the X-detector sector alone makes Tesseract 5-6x worse than the paper's Tesseract, so the paper's
decoders very likely used both detector sectors.

## Setting

The circuit is `qnd.circuits.load('bb72', 'X', p)` of the frozen code (qldpc_neural), i.e. relay-bp's reference file
`tests/testdata/bicycle_bivariate/circuit=bicycle_bivariate_72_12_6_memory_X,distance=6,rounds=6,error_rate=<p>,
noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim` at relay revision d185194, sha256-checked against
`qldpc_neural/data/circuit_manifest.csv` (p = 0.002: `4abcb5ad...`, p = 0.003: `ac17ceae...`). It is an X-basis
memory over 6 noisy rounds of Bravyi et al.'s depth-8 cycle, with 144 qubits, 2,592 CX, 432 detectors (both sectors)
and 12 X-type observables. A block failure is any of the 12 observables predicted wrong.

The noise is the circuit file's own ('uniform'), the model of Bravyi et al. (arXiv:2308.07915), all at strength p:
DEPOLARIZE2 after every CX; DEPOLARIZE1 on idle data qubits (in each round the L data in step 1, the R data in step 7
and all data in step 8; the ancillas are never idle; 792 slots in all); X_ERROR after R and Z_ERROR after RX (ancilla
resets every round, the data preparation); flip probability p on every ancilla measurement. The final data
measurement is noiseless.

The DEM is `circuit.detector_error_model(decompose_errors=False)`, 432 detectors x 16,200 mechanisms. Tesseract and
BP+OSD read this full DEM (all detectors).

Eq. (1) converts the block failure rate: P_L = (1 - (2 (1 - P_block)^(1/k) - 1)^(1/R)) / 2 with k = 12, R = 6. The
95 % interval is Wilson's on P_block, mapped through Eq. (1) (monotone). The paper's values are the study lead's
digitisation of the figure's SVG. Their P_block equivalents are Tesseract 5.85e-4 / 3.40e-3, BP+OSD
2.23e-2 / 8.91e-2, Relay 1.44e-3 / 6.82e-3 and Cascade 7.49e-4 / 4.73e-3 at p = 0.2 % / 0.3 %.

## Decoders

### Tesseract, short beam

This is tesseract-decoder's own preset `tesseract-short-beam` from
`tesseract_decoder.make_tesseract_sinter_decoders_dict()`. Its README calls this one of "the two sets of parameters
used in the Tesseract paper" (arXiv:2503.10988, App. B: "a beam of 15 with beam climbing combined with an ensemble of
16 different detector orderings, and a pqlimit of 200,000", with the no-revisit-detections heuristic and no detection
penalty). I built it directly as

    TesseractConfig(dem, det_beam=15, beam_climbing=True, no_revisit_dets=True, merge_errors=True, pqlimit=200_000,
                    num_det_orders=16, det_order_method=DetectorOrderMethod.Index, seed=2384753, det_penalty=0.0)

and decoded shot by shot (`decode`, `low_confidence_flag`). The corresponding command-line flags are
`--beam 15 --beam-climbing --num-det-orders 16 --det-order-index --det-order-seed 2384753 --pqlimit 200000
--no-revisit-dets` (no `--at-most-two-errors-per-detector`, no `--det-penalty`); I did not run the binary itself.
With beam climbing the decoder runs max(beam + 1, orders) = 16 searches per shot, the b-th with beam b and the b-th
order (src/tesseract.cc). The `Index` method orders the detectors by increasing or decreasing index at random. The
Tesseract paper instead used random projections of detector coordinates (`Coordinate`), which would give nearly the
same two orders here: the relay circuits give each detector one coordinate, and it increases with the detector index
except at 5 places. As a check, on the first 50 shots of each pilot chunk (both p) the preset called through sinter
(`compile_decoder_for_dem(...).decode_shots_bit_packed`) and this per-shot decoder agree on every shot. No shot raised
`low_confidence_flag`.

### BP+OSD

`stimbposd.BPOSD(dem)` with every default, read from stimbposd 0.2.0's `config.py`: `DEFAULT_MAX_BP_ITERS = 30`,
`DEFAULT_BP_METHOD = "product_sum"`, `DEFAULT_OSD_ORDER = 60`, `DEFAULT_OSD_METHOD = "osd_cs"`. stimbposd passes these
to ldpc 2.4.1's `BpOsdDecoder` (parallel schedule). It builds its check matrix by merging mechanisms with the same
detector set (432 x 16,200 here) and caps the OSD order at columns - rows, which does not bind. BP did not converge on
77 % (p = 0.2 %) and 93 % (0.3 %) of the pilot shots, so OSD decides most shots.

### Relay-BP (optional)

The frozen BB72 configuration (`qnd.teachers` 'relay': relay-bp 0.2.2 `RelayDecoderF32`, alpha 1, gamma0 0.15,
pre_iter 80, 300 sets x 60 iterations, gamma interval (-0.226, 0.622), stop after 5 solutions) ran on the full DEM
(`relay_full`) and on the X-detector sector of relay's `filter_detectors_by_basis` (`relay_sector`, 252 detectors).
For the alternative noise models `cal.RelayFull` runs the same parameters on the rebuilt circuit's full DEM; on the
frozen circuit it reproduces `teachers.Decoder` exactly (300 shots, `check_relayfull.py`).

## Results on our setting (uniform noise)

| p (%) | decoder | shots | failures | P_block | P_L (Eq. 1) | P_L 95 % interval | paper P_L | ours / paper |
|---|---|---|---|---|---|---|---|---|
| 0.2 | Tesseract | 54,000 | 17 | 3.15e-4 | 4.37e-6 | 2.73e-6 - 7.00e-6 | 8.13e-6 | 0.54 |
| 0.2 | BP+OSD | 12,000 | 235 | 1.96e-2 | 2.75e-4 | 2.42e-4 - 3.12e-4 | 3.14e-4 | 0.88 |
| 0.3 | Tesseract | 10,000 | 20 | 2.00e-3 | 2.78e-5 | 1.80e-5 - 4.29e-5 | 4.73e-5 | 0.59 |
| 0.3 | BP+OSD | 3,000 | 253 | 8.43e-2 | 1.23e-3 | 1.08e-3 - 1.39e-3 | 1.30e-3 | 0.94 |

Tesseract's ratio intervals are 0.34-0.86 (0.2 %) and 0.38-0.91 (0.3 %); BP+OSD's are 0.77-0.99 and 0.83-1.07.

The Relay-BP table combines fresh shots (Alpine) with the registered BB72 results (300,000 / 200,000 test shots) on
the same circuits:

| p (%) | Relay-BP | shots | failures | P_block | P_L | P_L 95 % interval | paper Relay | ours / paper |
|---|---|---|---|---|---|---|---|---|
| 0.2 | full DEM, fresh | 19,000 | 20 | 1.05e-3 | 1.46e-5 | 9.47e-6 - 2.26e-5 | 2.00e-5 | 0.73 |
| 0.2 | full DEM, registered | 300,000 | 274 | 9.13e-4 | 1.27e-5 | 1.13e-5 - 1.43e-5 | 2.00e-5 | 0.63 |
| 0.2 | sector, fresh | 30,000 | 73 | 2.43e-3 | 3.38e-5 | 2.69e-5 - 4.25e-5 | 2.00e-5 | 1.69 |
| 0.2 | sector, registered | 300,000 | 812 | 2.71e-3 | 3.76e-5 | 3.51e-5 - 4.03e-5 | 2.00e-5 | 1.88 |
| 0.3 | full DEM, fresh | 11,000 | 57 | 5.18e-3 | 7.22e-5 | 5.57e-5 - 9.35e-5 | 9.50e-5 | 0.76 |
| 0.3 | full DEM, registered | 200,000 | 1,092 | 5.46e-3 | 7.61e-5 | 7.17e-5 - 8.07e-5 | 9.50e-5 | 0.80 |
| 0.3 | sector, fresh | 10,000 | 141 | 1.41e-2 | 1.97e-4 | 1.67e-4 - 2.33e-4 | 9.50e-5 | 2.08 |
| 0.3 | sector, registered | 200,000 | 2,704 | 1.35e-2 | 1.89e-4 | 1.82e-4 - 1.96e-4 | 9.50e-5 | 1.99 |

## Alternative noise models (same circuits, `noise.py`)

`noise.py` strips the frozen circuit's noise and puts it back at the same locations with other strengths. Its
'uniform' rebuild gives the frozen DEM exactly at both p (`python noise.py --check`). It defines these models:
- `uniform_fm`: uniform plus flips p on the final data measurement.
- `spam2_fm`: resets and all measurements (final included) at 2p.
- `depspam_fm`: uniform plus a final-measurement flip p, with DEPOLARIZE1(p) after every reset and before every
  measurement (final included).

Expected faults per shot (sum of DEM priors) at p = 0.2 % are 8.44 for uniform, 8.59 for uniform_fm, 9.92 for
depspam_fm and 10.59 for spam2_fm. `noidle_fm` and `si1000` are defined but were not run. Removing the idle noise
lowers P_L, the wrong direction. SI1000 puts 5p on every measurement and, because every idle slot of this schedule
falls in a measurement or reset step, 2p on every idle; that nearly doubles the expected faults per shot (15.16 at
0.2 %).

| model | p (%) | decoder | shots | failures | P_block | P_L | P_L 95 % interval | paper | ours / paper |
|---|---|---|---|---|---|---|---|---|---|
| uniform_fm | 0.2 | BP+OSD | 4,000 | 83 | 2.08e-2 | 2.91e-4 | 2.35e-4 - 3.61e-4 | 3.14e-4 | 0.93 |
| uniform_fm | 0.3 | BP+OSD | 1,000 | 84 | 8.40e-2 | 1.22e-3 | 9.85e-4 - 1.51e-3 | 1.30e-3 | 0.94 |
| spam2_fm | 0.2 | Tesseract | 22,000 | 16 | 7.27e-4 | 1.01e-5 | 6.22e-6 - 1.64e-5 | 8.13e-6 | 1.24 |
| spam2_fm | 0.2 | BP+OSD | 4,000 | 116 | 2.90e-2 | 4.09e-4 | 3.41e-4 - 4.91e-4 | 3.14e-4 | 1.30 |
| spam2_fm | 0.2 | Relay full | 20,000 | 40 | 2.00e-3 | 2.78e-5 | 2.04e-5 - 3.79e-5 | 2.00e-5 | 1.39 |
| spam2_fm | 0.3 | Tesseract | 5,000 | 29 | 5.80e-3 | 8.08e-5 | 5.63e-5 - 1.16e-4 | 4.73e-5 | 1.71 |
| spam2_fm | 0.3 | BP+OSD | 1,000 | 100 | 1.00e-1 | 1.47e-3 | 1.20e-3 - 1.78e-3 | 1.30e-3 | 1.13 |
| spam2_fm | 0.3 | Relay full | 5,000 | 46 | 9.20e-3 | 1.28e-4 | 9.63e-5 - 1.71e-4 | 9.50e-5 | 1.35 |
| depspam_fm | 0.2 | Tesseract | 34,000 | 20 | 5.88e-4 | 8.17e-6 | 5.29e-6 - 1.26e-5 | 8.13e-6 | 1.01 |
| depspam_fm | 0.2 | BP+OSD | 4,000 | 102 | 2.55e-2 | 3.59e-4 | 2.96e-4 - 4.36e-4 | 3.14e-4 | 1.14 |
| depspam_fm | 0.2 | Relay full | 20,000 | 25 | 1.25e-3 | 1.74e-5 | 1.18e-5 - 2.56e-5 | 2.00e-5 | 0.87 |
| depspam_fm | 0.3 | Tesseract | 5,000 | 17 | 3.40e-3 | 4.73e-5 | 2.95e-5 - 7.58e-5 | 4.73e-5 | 1.00 |
| depspam_fm | 0.3 | BP+OSD | 1,000 | 90 | 9.00e-2 | 1.31e-3 | 1.07e-3 - 1.61e-3 | 1.30e-3 | 1.01 |
| depspam_fm | 0.3 | Relay full | 5,000 | 41 | 8.20e-3 | 1.14e-4 | 8.43e-5 - 1.55e-4 | 9.50e-5 | 1.20 |

`uniform_fm` was not run with Tesseract. Its DEM differs from the uniform one by under 2 % in total prior weight and
BP+OSD moved by at most 6 %, so I infer, without a Tesseract run, that it cannot account for a 1.75x gap. The work
went in this order: uniform with all decoders; BP+OSD (cheap) on `uniform_fm` and `spam2_fm`; Tesseract and Relay-BP
on `spam2_fm`; then `depspam_fm`, chosen because uniform and `spam2_fm` bracketed the paper for all three decoders.

## Diagnostics

Sector-only decoding (relay's X-detector filter, 252 detectors) ran on the same 1,000 shots, Alpine chunk 501.
Tesseract fails 3 times at 0.2 % and 20 times at 0.3 %, P_L 4.17e-5 and 2.81e-4, which is 5.1x and 5.9x the paper's
Tesseract. stimbposd fails 3 and 20 times, 0.13x and 0.22x the paper's BP+OSD. Neither of the paper's curves matches
sector-only decoding on our circuits.

The preset with BFS detector orders (`tesseract_bfs`, same seed) fails 5 times on the same 2,000 shots at 0.3 % where
the `Index` preset fails 4 times, with 3 failures shared. That shows no large sensitivity to the order method, but
the sample is small.

On identical shots, Tesseract and BP+OSD share 3 failures at 0.2 % (12,000 shots), with 232 failures BP+OSD's alone
and 3 Tesseract's alone; at 0.3 % (3,000 shots) the counts are 3, 250 and 2. About half of Tesseract's failures on
these shots are shots that BP+OSD decodes correctly, so the short beam search is not exhaustive here.

Every chunk row stores a digest of its observables. Chunks read by several decoders on one architecture have equal
digests, including Alpine chunks read on EPYC 9534 and 7713P nodes.

## Shots, seeds, hosts, versions

Chunk j of point i is 1,000 shots from `circuit.compile_detector_sampler(seed=2^50 + 10^9 + 10^6 i + j)` (`cal.py`).
Stim's samples for a seed depend on the host's SIMD width, so the Mac mini read chunks below 500 and Alpine chunks
500 and up. Decoders reading the same chunk on the same architecture read the same shots. All shots are fresh; no
registered test set was read. One check (`check_relayfull.py`) used 300 shots of point 1, chunk 999.

| point i | model | p | chunks decoded |
|---|---|---|---|
| 0 | uniform | 0.002 | Tesseract 0-39, 200-212 (Mac), 500 (Alpine timing); BP+OSD 0-11; Relay full 500-509, 526-534; Relay sector 500-529; sector diagnostics 501 |
| 1 | uniform | 0.003 | Tesseract 0-8, 100; BFS 0-1; BP+OSD 0-2; Relay full 500-505, 511-515; Relay sector 500-509; sector diagnostics 501 |
| 2 / 3 | uniform_fm | 0.002 / 0.003 | BP+OSD 500-503 / 500 |
| 6 / 7 | spam2_fm | 0.002 / 0.003 | Tesseract 0-21 / 0-4; BP+OSD 500-503 / 500; Relay full 500-519 / 500-504 |
| 10 / 11 | depspam_fm | 0.002 / 0.003 | Tesseract 0-21, 200-211 / 0-4; BP+OSD 500-503 / 500; Relay full 500-519 / 500-504 |

Points 4 / 5 (`noidle_fm`) and 8 / 9 (`si1000`) are defined in `cal.py` but were not run.

| host | CPU | used for |
|---|---|---|
| Mac mini, macOS 26.2 arm64, at most 4 processes | Apple M4 | Tesseract (all), BP+OSD on the uniform points, BFS check |
| Alpine acpu, Debian container `pyfull.sif` | AMD EPYC 9534 / 7713P | timing, Relay-BP, sector diagnostics, BP+OSD on the alternatives |

Both hosts ran the same versions: Python 3.12.14 (Mac) / 3.12 (container), stim 1.16.0, tesseract-decoder
0.1.1.dev20260910235247, stimbposd 0.2.0, ldpc 2.4.1, relay-bp 0.2.2, beliefmatching 0.2.0 (needed by relay's
basis filter), numpy 2.2.6, scipy 1.18.1. Each JSONL file starts with a header row holding the versions, the decoder
configuration, the circuit's sha256 and the DEM size.

Commands (from `ising_compare/bb/cascade`, with `QND_RELAY` set to relay's clone at d185194):

    python cal.py --point 0 --dec tesseract --chunks 0 --out runs/pilot_tesseract_i0.jsonl --sinter-check 50
    python cal.py --point 0 --dec tesseract --chunks 1-199 --out runs/tesseract_i0_mac_a.jsonl    # stopped after 39
    python cal.py --point 1 --dec tesseract --chunks 1-99 --out runs/tesseract_i1_mac_a.jsonl     # stopped after 8
    python cal.py --point 0 --dec bposd --chunks 1-11 --out runs/bposd_i0_mac.jsonl
    python cal.py --point 10 --dec tesseract --chunks 0-199 --out runs/tesseract_i10_mac_a.jsonl  # likewise 6, 7, 11
    python noise.py --check
    sbatch -p acpu --qos=cpu-normal -c 6 --export=ALL,CMD=relay_runs.txt alpine/multi.sbatch     # one cal.py line per core
    python table.py runs/mac/*.jsonl runs/alpine/*.jsonl --json cascade_cal_counts.json > table.md

The Alpine argument files are `alpine/{relay_timing,relay_runs,sector_timing,alt_bposd,alt_relay,alt3,alt_bposd2}.txt`
and the environment setup is `alpine/setup.sbatch`.

## Compute

The work used about 36 core-hours against a budget of about 40. The Mac mini spent 24.2 core-hours in completed
chunks (Tesseract 21.3, BP+OSD 1.9, BFS check 1.0) and about 0.4 more in partial chunks dropped at stops. Alpine used
11.25 allocated core-hours over 8 jobs. On an M4 core, short-beam Tesseract took 0.32 s per shot at 0.2 % and 1.36 s
at 0.3 % (uniform), and 0.43-0.50 s and 2.0-2.1 s on the alternatives, with single shots up to 26 s. stimbposd took
0.45-0.53 s per shot. Alpine's EPYC 9534 decoded Tesseract 2.3x slower than the M4 (0.66 s per shot at 0.2 %), so
every Tesseract run went to the Mac mini.

## Caveats on the runs

- Tesseract has fewer than 100 failures per point: 17 (0.2 %) and 20 (0.3 %) on the uniform points, 16-29 on the
  alternatives. Our Tesseract fails about every 3,200 shots at 0.2 % and every 500 at 0.3 %. At the measured speeds
  100 failures would cost about 28 and 19 M4 core-hours, 47 for the two uniform points alone. I stopped once the
  verdict was settled at both p and spent the rest on the alternatives. BP+OSD has 235 and 253 failures on the
  uniform points.
- Every stop was set by time or budget; none depended on the results so far. Most processes were stopped seconds
  after a chunk boundary, and the chunk just begun was dropped. Three partial chunks were dropped later into their
  decoding: point 1 chunk 101 (Tesseract, about 20 minutes in), point 1 chunk 2 (BFS, about 2 minutes), and the
  Relay-BP chunks in progress when Alpine job CJ01 was cancelled. Dropped chunks were never
  recorded.
- The rows of chunk 500 of `relay_full` at points 0 and 1 come from an earlier `cal.py` that marked every Relay-BP
  shot as flagged. Relay-BP's `flagged` field is meaningless; the counts are unaffected.

[A section on what this fit means for comparisons with the registered BB72 numbers, which went beyond the fit's own
question and discussed other work, is not part of this release.]

## Files

- `cal.py` covers circuits, seeds and decoders and writes one JSON row per chunk; `noise.py` holds the alternative
  noise models; `table.py` does the aggregation, Wilson intervals, Eq. (1) and paired counts; `check_relayfull.py` is
  the RelayFull check; `alpine/` has the job scripts and argument files.
- `runs/mac/*.jsonl` and `runs/alpine/*.jsonl` hold the raw rows: per chunk the seed, shots, failures, failing shot
  indices, flagged shots, decode seconds and observable digest, after a header row per process.
- `cascade_cal_counts.json` holds the aggregated counts per (point, decoder): shots, failures, P_block and P_L with
  intervals, the paper's value and ratio, chunk ranges, seeds, hosts, failing indices per chunk. It also holds every
  paired comparison and every header row. `table.md` is the same table as text.
