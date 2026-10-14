# CASCADE-TRT-REG: the paper's Cascade comparisons re-timed with Cascade on TensorRT

This directory is the freeze (F) of a registered speed re-test. Stage 1 of SPEED-RT (`speed_rt/SPEED_RT.md`,
`speed_rt/cascade/`) timed every Cascade comparison of the paper with Cascade compiled by `torch.compile`. This test
times the same comparisons with Cascade on the faster of two engines, on both of stage 1's cards at the same time. The
rules below are the authors' and are final. There is no separate plan and no plan review (the authors' decision,
2026-10-14).

## Settings
- Every Cascade comparison the paper reports, on both cards: the RTX PRO 6000 (Alpine, partition `artxpro6000`, one
  whole card) and an H100 80GB HBM3 SXM (one rented secure pod).
  - BB72 under the fitted noise: the follow-up's deep pairs (S55p_deep -> C14p) at p = 0.2 % and 0.3 %, and the first
    registered test's pairs (S55p -> C14p) at 0.2 % and 0.3 %.
  - The d = 13 surface code: the pair (ConvLines -> CS13) at p = 0.6 % and 0.35 %.
- Modes: single (batch 1, host round trip) and batched (the registered batch sizes).
- Engines: on each card, in each mode and for each Cascade (C14p, CS13), Cascade alone is timed with both engines, the
  registered compiled path (`torch.compile` in default mode, BF16, CUDA graph) and TensorRT. The engine with the lower
  Cascade-alone time is that card and mode's Cascade engine, and the pair runs with it: the table's pair time and
  speed-up are the ones with that engine. Both engines' Cascade-alone times are reported. Pairs timed with the other
  engine, if any, are reported apart and decide nothing.
- Statistic, against our claim: Cascade alone takes the fastest of 3 fresh processes, and every pair the slowest of 3.
  Speed-up = Cascade alone on its engine / the pair. Each comparison's Cascade alone is the one stage 1 divides by.
- Stage 1's acceptance and clock rules are kept unchanged (`speed_rt/SPEED_RT.md` sections 1.4 to 1.6;
  `speed_rt/cascade/srtlib.py` `Window` and `drive.py`): every timed window judged in-process, a process counted only
  if all its windows are accepted, a rejected or failed process replaced by a fresh one at most twice, and a slot
  without an accepted process after three attempts not established, with every comparison that uses it withheld.

## TensorRT engines
Built on each card, before any timing, from the ONNX export of `code/trt_build.py` (the whole evaluation step, fp32
graph, opset 17, a fixed batch per engine) with these builder settings: TensorRT 10.16.1.11 (`tensorrt-cu12`), BF16,
builder optimisation level 3, TensorRT's default workspace, and at d = 13 every convolution held to BF16
(`--conv-prec`). The engine call is captured in a CUDA graph behind the harnesses' graph interface
(`code/trt_common.py`). The sha256 of every ONNX file and every plan is recorded. Building is not timing.

## Timing shots
Fresh shots by stage 1's mechanism with new tags. T is the time of the earliest record in GitHub's activity list of
this repository whose `after` is F. The beacon is the first drand mainnet round at least 600 s after T, fetched and
checked by `speed_rt/beacon.py` with `--sets cascade_trt_reg/sets_ctrt.json`. The seeds are `speed_rt/sets.py`'s, run
on that sets file:
seed(set, j) = 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG(set) || j as 2 bytes big-endian)) mod 2^62,
TAG(set) = "convlines speed-rt <set>" and a newline, R = SHA-256 of the round's signature. The four sets
(`ctrt-cas-dfm-p2`, `ctrt-cas-dfm-p3`, `ctrt-cas-d13-p6`, `ctrt-cas-d13-p35`) have stage 1's kinds, p, chunks and
chunk sizes under new names, so every tag and every seed is new; chunk j of a set replaces chunk j of the original
stream, as in stage 1. The beacon record and the seed manifest are committed as a direct child of F (B). Nothing is
timed before B is public.

## Decision check (reported, not a gate)
On fresh beacon shots (chunks 10 to 19 of each Cascade's two sets: 100,000 shots per p), TensorRT's decisions and
failures against the compiled path's, for each Cascade on each card, as `code/trt_check.py` computes them: agreement and
the exact McNemar test (two-sided binomial on the shots exactly one engine fails). The paper's accuracy claims stay with
the registered tests.

## Output
One table per card: Cascade alone with both engines, every pair, the speed-ups against Cascade on its engine, and the
registered stage-1 values beside them; the decision checks; the engine chosen in each case; every departure from this
file; the costs (cap: 15 USD for the H100 pod; Alpine costs nothing). The analysis runs twice on a CPU node, and the two
outputs must be byte-identical.

## Code
`code/` is the exploratory code as it ran in one unregistered H100 session on 2026-10-14, byte for byte, minus three
scripts that reached the rented pod through an internal relay. Its launcher `trt_launch.py` runs the registered
harnesses with the Cascade network on either engine; `trt_build.py` exports and builds; `trt_check.py` is the decision
check. The scripts that run this test from it (B's seeds in place of stage 1's, both cards, stage 1's windows and
replacement loop) and the analysis are written after F and published with B, before any timing. Every difference from
this file is listed in the results as a departure.
