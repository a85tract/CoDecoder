# BB72: full-DEM Relay-BP's own time at p = 0.002 (supplementary timing T1)

This supplementary measurement was made on 2026-10-09, after the registered test. It is not a registered result, and
no registered file or result changed (`RESULTS_stage3.md`, `results.json` and `rows_cpu.csv` are as committed). Every
number below, the commands and the session's five files verbatim are in `SUPPLEMENT_fulldem_time_p0.002.json`.

The registered test timed full-DEM Relay-BP only at p = 0.003 (session B, 58.5963 ms per shot, the bar of BB72-3F),
because only that cell's speed gate needed it. Without a time at p = 0.002, the poster and the paper draw it there as a
dashed accuracy line. This session timed it at p = 0.002 with session B's method, after re-running session B's
p = 0.003 measurement as a control.

## Result

On one Apple M4 core, full-DEM Relay-BP takes 36.4087 ms per shot at p = 0.002: the mean over the first 2,000 test
shots of split BB72-2, the fastest of three fresh processes. The control re-ran the registered p = 0.003 measurement
and got 58.0535 ms against the registered 58.5963 ms, a ratio of 0.9907.

| p | shots | process | mean (ms per shot) | median (ms) | 99th percentile (ms) | core check (s) |
|---|---|---|---|---|---|---|
| 0.003 (control) | first 2,000 of test chunk 0 (split BB72-3) | 1 | **58.0535** | 39.52 | 344.3 | 0.0503 |
| | | 2 | 58.4872 | 39.81 | 353.4 | 0.0502 |
| | | 3 | 58.2623 | 39.61 | 349.0 | 0.0502 |
| 0.002 | first 2,000 of test chunk 0 (split BB72-2) | 1 | 36.5871 | 27.48 | 178.7 | 0.0494 |
| | | 2 | **36.4087** | 27.36 | 176.7 | 0.0500 |
| | | 3 | 36.5811 | 27.54 | 177.4 | 0.0503 |
| 0.003 (registered, session B, 2026-10-07) | the same shots | 1 | **58.5963** | 39.85 | 354.4 | 0.0512 |
| | | 2 | 58.9964 | 40.07 | 354.7 | 0.0499 |
| | | 3 | 59.2071 | 40.36 | 352.9 | 0.0507 |

The counted value in bold is the fastest of the three processes, as in the registration. The control's fastest is
0.93 % below the registered fastest, and the mean of its three processes (58.268 ms) is 1.13 % below the registered
three's mean (58.933 ms). Within each p the three processes lie within 0.75 % (p = 0.003) and 0.49 % (p = 0.002) of
each other. At p = 0.002 the time is 0.627 of the control's at p = 0.003.

## How it was measured

- Host: the Mac mini of the registered sessions (Apple M4, `m4-host` in the release), macOS 26.2 (25C56). It had been
  up since 2026-10-05 18:17 PDT, so this session ran in the same boot as the registered sessions of 2026-10-07.
- Session `mac-mini-m4-supp-T1`, 2026-10-09 00:50:39 to 01:01:05 PDT. The script, `~/work/bb72_supp_t1/mac_T1.sh`
  (verbatim in the JSON), was launched from the box with `nohup` and wrote only to that directory. It ran one warm-up
  process, then the three control processes, then the three p = 0.002 processes, one at a time, each a fresh
  single-threaded Python process.
- Code: session B's own code copy (`~/work/bb72s3/code`) with session B's environment: `PYTHONPATH`,
  `BB_SHOTS`, `QND_RELAY`, one thread each for OpenMP, OpenBLAS, MKL, vecLib and Rayon, run from
  `code/ising_compare`. Every Python file under `ising_compare/bb` and `qldpc_neural/qnd` in that copy matches the
  freeze commit `19368aea` byte for byte, except `bb/stage3.py` (amendment S's `cputerm`, not used here) and the added
  `bb/scripts/stage3_pack.py`.
- Versions: Python 3.12.14 (session B's venv), relay-bp 0.2.2, numpy 2.2.6, stim 1.16.0, scipy 1.18.1, with Relay-BP's
  test-data circuits at `d185194`. Since session B, only `fontTools` and `__pycache__` have changed among the top-level
  entries of the venv's site-packages.
- Decoder: `bb.opp`'s `relay_full`, that is `qnd.teachers.Decoder('relay', 'bb72', 'X', p, 'all')`:
  `relay_bp.RelayDecoderF32` on the full detector error model (both detector sectors) with the frozen parameters
  (alpha 1.0, gamma0 0.15, pre_iter 80, 300 sets of 60 iterations, gamma interval [-0.2263, 0.6216], stop after
  5 solutions).
- Shots: the first 2,000 shots of test chunk 0 at each p, `bb72_p0.003_test0000.npz` (sha256 `1f0c3e13...`) and
  `bb72_p0.002_test0000.npz` (sha256 `e32497e2...`). The warm-up used the first 1,000 shots of
  `bb72_p0.003_test0001.npz`, as session B's did. All three files match `data/bb/shot_manifest.csv`, and `bb.shots.load`
  checks each one against the manifest again in every process.
- Timing: `bb.timing opponent` records the wall time of each single-shot decode call. The row value is the mean over
  the 2,000 shots, and the row also records the median, the 99th percentile and the core-check loop (about 0.05 s on a
  performance core, about twice that on an efficiency core).

The commands (`$PY` is `~/work/.venv/bin/python`, `$O` is `~/work/bb72_supp_t1`):

```bash
# warm-up, untimed, as in session B
$PY -m bb.timing opponent --code bb72 --p 0.003 --dec relay_full --split test --chunks 1 --first 1000 --out $O/warmup_T1.csv
# control: session B's command, for k = 1, 2, 3
$PY -m bb.timing opponent --code bb72 --p 0.003 --dec relay_full --split test --chunks 0 --first 2000 --proc $k --out $O/rows_cpu_T1.csv
# p = 0.002: the same command with --p 0.002, for k = 1, 2, 3
$PY -m bb.timing opponent --code bb72 --p 0.002 --dec relay_full --split test --chunks 0 --first 2000 --proc $k --out $O/rows_cpu_T1.csv
```

## Idle checks

Before each of the seven processes, the script ran session S's check as narrowed in the release (`mac_S.sh`): wait in
15-second steps until no process is at 20 % CPU or more in `ps`, no Python process is at 1 % CPU or more, and the
1-minute load is below 3. The check passed at once every time. All 15 lines of the idle log are in the JSON.

- Across the logged lines the 1-minute load ranged from 1.24 to 2.20; session B's relay_full lines show 1.80 to 2.12.
  The logged top processes were macOS's own: the aerial wallpaper extension (at most 10.1 % CPU), WindowServer and
  VTDecoderXPCService, the same three as in session B's relay_full lines, and at times opendirectoryd and dasd.
- The line logged before control process 2 (00:53:31) shows launchd at 28.2 % CPU. That sample was taken just after
  the check's own sample had passed, at the moment process 1 exited; launchd handles process exits and launches.
  Process 2 (58.4872 ms) is the slowest of the three controls, 0.75 % above the fastest, and is not the counted one.
- The only Python process besides ours was the Command Line Tools' system Python 3.9, idle since boot at 0.0 % CPU.
  This is the process that led to the narrowed check in session S.
- The six timed processes' core checks took 0.0494 to 0.0503 s, which is performance-core speed (the release gives 0.049
  to 0.051 s, to three decimals, for registered sessions A and B). The warm-up's took 0.0477 s.

## Differences from session B

- Session B logged the load and the top three processes before and after each process, without waiting. This session
  ran the narrowed idle check above before each process and logged the same fields after it.
- Session B warmed up relay, bposd and relay_full in turn, since it timed all three. This session warmed up relay_full
  only, on the same 1,000 shots.
- In session B the three relay_full processes followed about 15 minutes of other single-threaded timing (the opponents
  and the F cells' CPU terms). Here they followed only the 55-second warm-up, and the p = 0.002 processes followed the
  control.
- The session label is `mac-mini-m4-supp-T1`, and the outputs went to `~/work/bb72_supp_t1` rather than session
  B's `~/work/bb72s3/out`, so no registered record was touched.
- Session B had no p = 0.002 step. Here it is the same command with `--p 0.002`, on split BB72-2's first 2,000 test
  shots.
