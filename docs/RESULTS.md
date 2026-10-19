# CoDecoder results in detail

Part of the [CoDecoder release](../README.md).

## What the paper shows

The accuracy numbers come from the newest registered tests: SURF-v2 for the surface code, with [D9-SHALLOW](REGISTERED_TESTS.md#d9-shallow-the-network-at-d--9) at
d = 9, and RETEST for the BB codes, Cascade and the output heads. Every speed number comes from SPEED-RT, the
registered speed re-test ([How the speed was measured](SPEED.md#how-the-speed-was-measured-speed-rt)), except the composition at
d = 9, which D9-SHALLOW timed in its own registered session. Every comparison times the composition by its slowest process
and the strong decoder alone by its fastest. The table compares each composition with its own strong decoder alone.
Percentages and speed-ups are rounded down.

| strong decoder behind the network | code and noise | the composition's failures against the strong decoder alone | the strong decoder's time per shot over the composition's |
|---|---|---|---|
| correlated PyMatching (PM) | rotated surface code, d = 7 to 17, p = 0.6 % | 6.4 % to 31.4 % fewer (SURF-v2; D9-SHALLOW at d = 9) | 2.3 to 10.3 (SPEED-RT stage 2; D9-SHALLOW at d = 9) |
| Relay-BP on all detectors (full-DEM Relay-BP) | BB72 [[72,12,6]] at p = 0.3 % and 0.2 %, the gross code [[144,12,12]] at p = 0.4 % | 22 %, 29 % and 50 % fewer (RETEST) | 17.5, 124.9 and 8.9 (stage 2) |
| our Cascade for BB72 | BB72 under a fitted noise model, p = 0.2 % and 0.3 % | 1.1 % and 0.3 % fewer, each significant (RETEST) | 5.1 to 16.3 on an RTX PRO 6000, 3.1 to 9.1 on an H100 (stage 1) |
| our Cascade for the surface code | d = 13, p = 0.35 % and 0.6 % | 1.2 % to 1.3 % fewer at 0.35 %, each significant; 0.4 % fewer at 0.6 %, non-inferior but not significantly better (RETEST) | 3.6 to 7.6 on the RTX PRO 6000, 3.3 to 7.1 on the H100 (stage 1) |

The speed column comes from `speed_rt/s2/surface/results/table.md` (row S2, SURF-v2's networks),
`speed_rt/s2/bb/results/table.json` (row B5) and `speed_rt/cascade/results/verdicts.json` (claim H3-cascade, whose
twelve comparisons are rows C1, C2 and C8 of `table.json`). On the surface and BB codes a composition's time is a composite:
the network's time on an H100 GPU, plus the handoff, plus the strong decoder's time on the routed shots, measured on one
CPU core and converted to one core of NVIDIA's Grace CPU; the strong decoder alone is measured and converted the same
way. A Cascade time is measured end to end on one card, with the composition and Cascade alone in the same job, one shot at a
time and in batches. Stage 1's registered analysis took the slowest of three processes on both sides; the Cascade
speed-ups above are its sensitivity rows, in which Cascade alone takes its fastest process. This lowers no speed-up
by more than 3.2 % and changes no outcome (`speed_rt/cascade/results/table.json`). With Cascade compiled by TensorRT,
every composition the frozen analysis could compare still took 2.3 to 16.3 times less time per shot than Cascade alone on its
faster engine ([CASCADE-TRT-REG](REGISTERED_TESTS.md#cascade-trt-reg-cascade-on-tensorrt)).

On the surface code the registered rule chose q = 0 at d = 11 and 13, so CoDecoder alone decided every shot there. At
d = 9 the composition is [D9-SHALLOW](REGISTERED_TESTS.md#d9-shallow-the-network-at-d--9)'s one-block network with correlated PM at q = 6 %. Against the other
decoders (plain PM, NVIDIA's Ising pre-decoders Fast and Accurate followed by plain or correlated PM, and two BP-OSD
settings), the composition was more accurate at every distance (at d = 9 D9-SHALLOW compared it with plain and correlated
PM on its own shots; the other decoders' d = 9 counts are SURF-v2's). It was also faster than each of them at every distance
except against plain PM at d = 7, where the composition took 11.942 µs per shot and plain PM 11.890 µs, 0.995 of the composition's
time (with the deployed network, which SPEED-RT timed as well, the composition took 12.480 µs and plain PM 0.952 of that). At
d = 9 the composition took 21.234 µs against plain PM's 26.297 µs. From d = 11 on, plain PM took 1.25 to 2.97 times as long
as the composition (1.20 to 1.77 with the deployed networks). These numbers are in `speed_rt/s2/surface/results/table.md`
(rows S1 and S4), the `ratio` column of `rows.csv` beside it and `d9_shallow/results/table.md`.

On BB72 and the gross code CoDecoder runs in three modes (Table IV): alone, balanced (CoDecoder + sector Relay-BP, which
decodes one type of check) and accurate (CoDecoder + full-DEM Relay-BP); the table above gives the accurate mode. On the Grace basis, CUDA-Q's BP-OSD on the GPU, timed host to host, was the fastest decoder other than ours at all
three points, and the network alone took less time than it at each: 0.09072 against 0.8909 ms per shot on BB72 at p =
0.3 %, 0.08941 against 0.5396 ms at 0.2 %, and 0.2555 against 13.70 ms on the gross code, 0.019 of CUDA-Q's time. A composition
with a fallback is not always faster than CUDA-Q. On the gross code the balanced composition took 20.35 ms per shot against
CUDA-Q's 13.70 ms, while sector Relay-BP alone took 6.2 times as long as the composition. The accurate composition took 4.864 against
0.8909 ms on BB72 at p = 0.3 % and 427.5 against 13.70 ms on the gross code. The balanced composition on BB72 at both p and the
accurate composition at p = 0.2 % took less time than CUDA-Q. These numbers are rows B1 to B6 of
`speed_rt/s2/bb/results/table.json`.

In the Cascade comparisons, the composition's 99.9th-percentile latency in single-shot mode is longer than that of Cascade
alone on both cards, since a routed shot runs the network before Cascade. At d = 13 it was 486.2 and 499.5 µs against
404.3 and 422.4 µs on the RTX PRO 6000 (p = 0.6 % and 0.35 %), and 626.4 and 573.9 µs against 564.9 and 554.5 µs on the
H100. On BB72 the composition's was 7.1 % to 9.9 % above Cascade's on the RTX PRO 6000 and 8.4 % to 17.8 % above on the H100
(rows C8 and C10 of `speed_rt/cascade/results/table.json`).

On BB72, the same network with a conventional head that predicts each logical qubit separately failed 3.8 to 5.0 times
as often when it decided every shot itself (RETEST).

SURF-v2 also compared the composition with gap switching, our re-implementation of Decoder Switching, which uses plain PM as
the front end, routes by its complementary gap and falls back to correlated PM; the composition failed less often at every
distance (`surface_v2/surf_v2/GAPSW_REGISTRATION.md`). SPEED-RT timed gap switching on the CPU: 57.41 µs per shot at d =
7 up to 2,402.31 µs at d = 17 on the Grace basis, against the composition's 11.943 to 81.580 µs. This comparison is descriptive
and has no verdict (`speed_rt/s2/gapsw/results/gs_tables.md`).

Most tests were registered: the networks, thresholds, test sets and analyses were fixed before the test shots existed,
and each test set was drawn once and read once. The three newest registrations fixed in advance that the paper reports
their numbers, whatever they turned out to be, in place of the earlier ones (`RETEST.md` section 5,
`surface_v2/surf_v2/REGISTRATION.md` "The paper's rows", `speed_rt/SPEED_RT.md`). D9-SHALLOW (`d9_shallow/PLAN.md`)
fixed in advance that its composition becomes the paper's d = 9 row if its claim holds, and CASCADE-TRT-REG
(`cascade_trt_reg/RULES.md`) fixed its engines, rules and analysis before any of its timing shots existed. The
original tests and the earlier timing sessions stay in this repository with their plans and records.
