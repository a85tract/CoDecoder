# speed_rt/cpu_ws: the CPU half of SPEED-RT on the Windows CPU host

Every CPU part of stage 2 is timed on one Windows Server machine, inside a WSL2 distro. This folder holds the setup,
the parallel harness and the plans.

| part (plans.py) | what one process times | code it runs |
|---|---|---|
| `surf_base` | plain and correlated PM per block, d = 7-17, timing chunk 0 (10,000 shots) | `timing.cpu_term ref [--plain]` |
| `surf_term` | our pairs' CPU terms: correlated PM on the routed shots of 13 networks | `timing.cpu_term term` |
| `surf_share` | Ising's PM shares: plain and correlated PM on the residual detectors of Ising M1's and M4's counted plans, d = 7-17 (24 processes) | `timing.cpu_term share [--plain]` |
| `gref` | the Grace fit's 18 points: plain PM at d = 13, 21, 31; correlated PM at d = 5, 9, 13, 17, 21, 31; p = 0.3 % and 0.6 % | `speed_rt.gref time` |
| `gs_fe`, `gs_term` | Decoder Switching's front end and its routed correlated PM, d = 7-17 | `speed_rt.gapsw.gs_time fe / term` |
| `bb`, `bb_warm` | BB72 (p = 0.3 %, 0.2 %) and gross (0.4 %): sector Relay-BP, full-DEM Relay-BP, ldpc BP-OSD, and our Relay-BP terms (cells 3S, 2S, 3F, 2F, G-4R, G-4F) | `speed_rt/bb/srt_bb.py` with `bb.timing opponent`, `bb.stage3 cputerm`, `bb.stage4_gross cputerm` |

The BB "CPU BP-OSD" baseline is the one `bb/` timed: `ldpc.BpOsdDecoder` (min-sum, 80 iterations, OSD-CS order 60).
`bb_jobs.json` sets `BB_PIDNS=1`, as `speed_rt/bb/cpu_item.sh` does. Only the GPU watch reads it, and no CPU job calls
that watch.

## Files

| file | what |
|---|---|
| `setup_env.sh` | the pinned environment in the tmpfs (`build`, `sync`, `restore`, `check`) |
| `prep.sh` | puts code, environment and data into the tmpfs (`env`, `code`, `relay`, `val`, `surface`, `gs`, `bb`, `handoff_dry`) |
| `tmpfs_libs.sh` | copies the C libraries into the tmpfs (`/dev/shm/srt/lib`) |
| `pinlibs.py` | keeps the disk copies of those shared objects locked in memory, for any tool a family's code starts |
| `harness.py` | the timed windows and their checks (`topo`, `disk`, `freq`, `sessionref`, `run`), the hand-off import and the saving; the Linux host mode ("Linux host" below) |
| `fetch.sh` | the fetcher: pulls a session's saved records from the Windows folder to the Linux workstation |
| `cfg_box.json`, `fetch_box.sh`, `handoff_pull.sh` | the Linux host: its config, its fetcher (local save folder to the collecting machine) and its hand-off copy (the GPU side's folder to the local hand-off folder) |
| `export.py`, `export_surf.py`, `export_bb.py`, `export_gs.py` | a run directory into each family analysis's layout |
| `plans.py` | the plan: jobs packed into windows of 8, sides interleaved |
| `cfg_ws.json` | the harness config (paths, the reference process, the filler, slots, limits) |
| `filler.py`, `clk.c` | the untimed filler of a slot whose job ended; the clock probe |
| `bb_jobs.json` | every BB CPU process (8 warm-ups, 15 timed), with dry overrides |
| `summarize.py` | one line per window and per job, from a run's records |
| `dry_all.sh`, `mkmask_dry.py`, `mkres_dry.py`, `bbpreds_dry.py`, `bbdraw_dry.sh` | DRY RUNS ONLY: the whole dry run, synthetic routing masks (surface), Ising residuals and predictions (BB), and the BB dry draw without the cluster's host check |

Paths: the tmpfs is `/dev/shm/srt`; persistent files (logs, the environment's cache) are under `$QRT`
(default `$HOME/qrt`, on the distro's disk).

## The host

- Windows Server 2022 Datacenter, build 20348.3328; Intel Xeon Gold 6326 at 2.90 GHz, 16 cores, 32 logical
  processors; 127.6 GB RAM; a mechanical HDD and no SSD. The account that runs the work is not an administrator.
- WSL 2.4.12.0, kernel 5.15.167.4-microsoft-standard-WSL2; the distro is Ubuntu 24.04.4 LTS, without systemd and with
  interop on. Other distros on the machine stay stopped.
- `%USERPROFILE%\.wslconfig` (there was none before): `[wsl2] memory=110GB swap=0 processors=32`, then `wsl --shutdown`.
  The distro then shows MemTotal 113,325,824 kB, SwapTotal 0 kB and 32 vCPUs, with 55 GB in `/dev/shm`.
- The account may not use WMI there (access denied) and cannot read the CPU counters through PDH either.
  `powershell.exe` started through WSL interop exits with code 66 and no output; `cmd.exe` works, so the harness
  reads the Windows facts (power plan, CPU, version, `.wslconfig`, VM processes, users) and the profile folder through
  `cmd.exe`. A detached run's own interop socket dies with the login session that started it, so the harness uses a
  socket that answers (the one of the `wsl.exe` that keeps the distro up).

## Keeping the distro alive

A WSL2 distro instance ends about 15 s after its last `wsl.exe` session closes, whether background processes run or
not, and its tmpfs goes with it. Windows' OpenSSH server also ends every process of a session when the session closes.
Its job object allows breakaway, so one `wsl.exe` started with `CreateProcess(..., CREATE_BREAKAWAY_FROM_JOB, ...)`
outlives the session and keeps the distro up. It runs `sleep` inside the distro. Long jobs start detached with
`setsid -f` and are polled. A plain `cmd &` at the end of a login session can be killed before it leaves the session.

## The HDD rule

- `/dev/shm/srt` (tmpfs) holds the interpreter and its standard library (a copy of Ubuntu's CPython 3.12.3), the venv
  with every package, the code, the circuits (each process builds its DEMs in memory), the shots, the masks and every
  output.
- The C libraries that the interpreter and the extension modules link (libc, libm, libz, libcrypto, libstdc++, ...)
  are copied into `/dev/shm/srt/lib` by `tmpfs_libs.sh`, which `prep.sh env` runs. Every process gets
  `LD_LIBRARY_PATH=/dev/shm/srt/lib`, `LC_ALL=C.UTF-8` and `TZ=UTC0` (`env` in cfg_ws.json), so it loads them from the
  tmpfs and reads no locale or zone file either. Only the ELF loader named in the interpreter's header
  (`/lib64/ld-linux-x86-64.so.2`) stays the distro's file: rewriting the header with patchelf made the interpreter
  crash at start. `pinlibs.py` keeps the loader, and the disk copies of the other libraries, locked in memory. The
  script ends with a process that imports every timed module and prints what it maps outside the tmpfs: the loader
  and the C.UTF-8 locale files, all locked by `pinlibs.py`, which also locks `/usr/bin/uname` (`platform.processor()`
  runs `uname -p` in every cpu_term, gref and gapsw process) and git. In the first dry run, before the libraries were locked or
  copied, a window 20 minutes after the previous one read about 6 MB from the disk.
- `harness.py` reads `/proc/diskstats` (reads and sectors of every whole disk) and `/proc/vmstat` (pgmajfault,
  pswpin) before a window's first reference segment and after its last. A window with any read, major fault or
  swap-in is rejected.
- The environment (`setup_env.sh`): numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2,
  ldpc 2.4.1, stimbposd 0.2.0 and sinter 1.16.0 (imported by Relay-BP's `tests/testdata` helper, which loads the BB
  circuits). The host reaches PyPI but not GitHub, so Python is Ubuntu's 3.12.3, the version the GPU pods ran.

## Placement: one physical core per timed process

Inside the VM, `lscpu` shows 1 socket, 16 cores and 2 threads per core, and `thread_siblings_list` pairs vCPU 2k with
2k+1. Hyper-V on Windows Server 2022 uses the core scheduler by default: a VM's sibling vCPU pair runs on the two
logical processors of one physical core, together, and nothing else shares that core while it runs. A process pinned
to vCPU 2k with vCPU 2k+1 idle therefore has a physical core to itself, with its SMT sibling idle. The harness uses one
vCPU in each of 8 sibling pairs (vCPUs 2, 4, ..., 16). Pair 0 (vCPUs 0-1) is left to the kernel, and the harness, its
monitors and the clock samples run on the spare pairs (vCPUs 18-31). The per-window reference segments and clock
probes, kept in every window record, are the evidence. Native Windows timing with affinity masks was not needed, and
Windows has no tmpfs.

## A window (`harness.py run`)

1. Read the disk counters, then start the monitors: the clock probe on a spare vCPU every 2 s, plus the Windows
   counters through interop where they are readable.
2. Reference segment "before": `timing.cpu_term ref --plain --d 9` on validation chunk 0 (the registered chunk,
   sha256 `49d1bea9...`) on every slot at once, then a 0.03 s clock probe on each slot.
3. Start the timed processes together, one per slot. A slot whose process ends early runs the filler until the
   window's last process ends, so the load stays at 8 busy cores.
4. Reference segment "after", then stop the monitors and read the disk counters again.

A window passes when it saw no disk read, every reference value is at most 1.10 x the session reference, the spread
(largest / smallest) of each segment is at most 1.06 (1.03 before the High-performance dry runs below), and every
process exited 0 and left a record. A process gets a fresh attempt with the same arguments in a later window
(`r001`, ...) when its window was rejected or when it failed (a non-zero exit, `gs_time`'s exit 4 included, or no
record). `plans.pack` packs these replacements, so the two sides stay interleaved. A process has at most 3 attempts in all (STAGE2_common C5). After a lost machine, a new session runs
the same plan with `--prior OLD/win`, which skips the jobs already accepted and counts the attempts already used.
Every record is written once (`window.json` per window). The session reference (`harness.py sessionref`) is the median
of 3 segments at full concurrency (24 values). A reference that saw a disk read is taken again, at most 3 times in all,
and the record keeps the earlier attempts.

## The hand-off into the tmpfs (STAGE2_common C2)

The untimed GPU passes' outputs reach the Windows folder `%USERPROFILE%\speedrt_in\` by scp from the Linux
workstation (`handoff_from` in cfg_ws.json). Nothing started inside the distro brings them. Each family folder carries
a checksum file and, written last, a marker:

| unit | folder | checksums, marker | what the jobs read |
|---|---|---|---|
| `surface` | `surface\<role>\val_<network>.npz` | `surface\SHA256SUMS`, `surface\DONE` | surf_term's masks |
| `ising` | `ising\<role>\res_d<d>_<M>_val000.npz` | `ising\SHA256SUMS`, `ising\DONE` | surf_share's residuals |
| `bb/draw` | `bb\draw\shots\`, `manifest_bb72.csv`, `manifest_gross.csv` | `bb\draw\draw_SHA256SUMS`, `bb\draw\DRAW_DONE` | every BB job's shots |
| `bb` | `bb\preds\` (and `bb\draw\`) | `bb\SHA256SUMS`, `bb\DONE` | our BB terms' predictions |

The role is `low` for d = 7, 9 and 11 and `high` for d = 13, 15 and 17. At each window boundary `harness.py run`
copies every unit whose marker exists into `/dev/shm/srt/handoff/<unit>/` and checks every file its checksum file
names. A unit with a mismatch is not used and is copied again at the next boundary. Each import is recorded in
`RUN/handoff/`. A window starts once the units its jobs need are in. Both sides of a comparison need the same units,
so they stay interleaved: surf_base and surf_term need `surface`, surf_share needs `ising`, BB's baselines need
`bb/draw` and our BB terms need `bb`. gref and Decoder Switching need none and go first. Decoder Switching's processes
run in windows of their own, because `gs_time`'s own probe check (exit 4) rejects the window it runs in and should
cost no other part an attempt. While a unit is awaited the
harness runs the ready windows and any replacements, and waits at most `--handoff-wait` seconds (default 3 h).
`push.sh`, which uploads into the Windows folder, is not part of this folder.

## Saving and fetching (partial results)

After each window, outside every check's span, the harness copies the window's records to
`%USERPROFILE%\speedrt_out\<session>\`. It also copies topo.json, plan.json and sref/ when the run starts
(`save_to` in cfg_ws.json; the session is the run directory's name). The fetcher runs on the Linux workstation. It
reads only that folder, using the host's own tar.exe over ssh, and never starts anything inside the distro:

```
WIN_SSH="ssh HOST" bash fetch.sh S2 ~/work/s2/raw/cpu                # every 15 minutes; --once after the session
```

It logs each fetch's start, end and progress (windows saved, windows planned) in `DEST/fetch.log`. A lost link stops
neither the detached session nor the saving, and the next fetch picks up every window saved in the meantime.

## Running a session

Before the session, copy into the distro the frozen commit's trees (`git archive F surface_v2 evaluation bb speed_rt
retest/replay`, without `surface_v2/weights`), Relay-BP's `tests/testdata` at d185194 (the three BB circuits and the
helper), and the registered probe chunk `d9_p0.006_X_val_000.npz`. Then, inside the distro:

```
W=$QRT/cpu_ws; P="bash $W/prep.sh"
$P env && $P code $QRT/code_F.tgz <F's 40-hex hash> && $P relay $QRT/relay_testdata.tgz && $P val
cp d9_p0.006_X_val_000.npz /dev/shm/srt/data/val/ && cp d9_p0.006_X_val_000.npz /dev/shm/srt/data/probe/
```

Every shot set is drawn once, off this host, and comes through the hand-off too, so that the pods and this host read
the same bytes: unit `surface_shots` (`shots\` with the timing chunks, `refshots\` with the reference chunks,
`srt-surf.json`, `srt-ref.json`) and unit `gs` (`shots\`, `srt-gs.json` and, if sent, the registered probe chunk
`d9_p0.006_X_val_000.npz`), each with `SHA256SUMS` and `DONE`. The same files as `surf\` and `ref\` (or `gs\`
and `probe\`) subtrees of `/dev/shm/srt/data` are read too. `prep.sh shots RUN` waits for both, copies and checks them as the harness does
(`harness.py import`, records in `RUN/handoff/`), links them into `/dev/shm/srt/data` where the plans read them,
builds the surface view (`speed_rt.draw view`) and computes Decoder Switching's masks (`gs_time mask`, 8 workers,
untimed). `prep.sh code TGZ COMMIT` writes the frozen commit's full hash to `code/DEPLOYED`, which the Decoder
Switching records carry.

Then the session, detached:

```
cd /dev/shm/srt/cpu_ws && R=/dev/shm/srt/runs/S2 && mkdir -p $R && setsid -f bash -c "
  export LD_LIBRARY_PATH=/dev/shm/srt/lib LC_ALL=C.UTF-8 TZ=UTC0
  bash prep.sh shots $R &&
  ../venv/bin/python harness.py topo --out $R/topo.json &&
  ../venv/bin/python harness.py sessionref --cfg cfg_ws.json --out $R/warm --repeats 1 &&
  ../venv/bin/python harness.py sessionref --cfg cfg_ws.json --out $R/sref &&
  ../venv/bin/python plans.py --parts bb_warm,gref,gs_fe,gs_term,surf_base,surf_term,surf_share,bb --reps 3 --out $R/plan.json &&
  ../venv/bin/python harness.py run --cfg cfg_ws.json --plan $R/plan.json --sessionref $R/sref/sessionref.json --out $R/win
  " > $R/session.log 2>&1 < /dev/null
tail -5 $R/session.log                                                     # poll
../venv/bin/python summarize.py $R/win | tail -40
```

The first `sessionref` is the warm pass: one untimed segment, which loads the reference process's code path before
the session reference. The BB warm-ups (`bb_warm`, untimed) come first in the plan. The plan is rep-major
(`--order rep`): every job of repetition 1, then of 2, then of 3, in windows grouped by the hand-off units they need.
A dry plan's Decoder Switching jobs read 4 chunks, so they need masks made with `--chunks 4`.

## Export to the families' analyses

`export.py` turns a run directory, on the host or a fetched copy, into the layout each family's analysis reads. It
copies values and changes none. A process of a rejected window becomes a rejected attempt and does not count.

```
cd /dev/shm/srt/cpu_ws && export PYTHONPATH=/dev/shm/srt/code:/dev/shm/srt/code/surface_v2
../venv/bin/python export.py $R $R/export --only surface,gs                     # surface/records/cpu/, gs/{session,prescreen}/
../venv/bin/python -m speed_rt.analysis_surf fit --cpu $R/export/surface/records/cpu \
    --testsets /dev/shm/srt/data/ref --out $R/fit_s2.json                         # the stage's Grace fit (C-ref)
../venv/bin/python export.py $R $R/export --only bb --fit $R/fit_s2.json         # bb/cpu/ with grace.json
```

The surface part writes `cpu_session.json`, `cpu_attempts.json` and each record under `cpu/` (STAGE2_surface 1.14).
The BB part writes `rows.csv`, `accepted.json`, `session.json`, `host.json`, `grace.json` and `attempt_log.json`
(STAGE2_bb); BB's `collect.sh` takes the directory only with `grace.json`, so the order is the one above. The Decoder
Switching part writes drive_gs's session and prescreen directories (STAGE2_gapsw 1.10); on a fetched copy of the run,
pass `--gs-manifest` (B_2's `srt-gs.json`), because the path the processes read is on the host. Each part's missing
slots, conflicts and reasons are printed and kept in `export_summary.json`.

## Dry run (13 October, 16:44-16:58 PDT; dry shots, synthetic masks and predictions)

Every CPU part ran end to end through the harness, in 10 windows of 8 (`dry_all.sh`). Every process exited 0 except
Decoder Switching's: `gs_time`'s own probe check rejected 17 of 18 attempts (exit 4; see the clocks below). The disk
check saw no read in 9 of the 10 windows; the first one, the warm-up window, had 1 read and 1 major fault.

Time per repetition, one process each, on one vCPU at 8-way concurrency. A * marks values extrapolated from the dry
run's per-shot time to the registered shot count. The dry run routes the heaviest shots to the CPU (synthetic masks
and predictions), so its per-routed-shot times for the CPU terms are upper bounds.

| part | processes per rep | seconds per process | total per rep |
|---|---|---|---|
| surf_base (plain + correlated PM, d = 7-17, 10,000 shots) | 12 | 0.8 (plain d7) to 13.2 (corr d17) | 45 s |
| surf_term (13 networks, 200,000 shots) | 13 | 0.9 (tau 0) to 34.6 (d17 q08) | 104 s |
| gref (18 points, 10,000 shots) | 18 | 0.7 (corr d5) to 228* (corr d31 p6: 22.6 ms/shot) | 480 s* |
| gs_fe (d = 7-17, 200,000 shots) | 6 | not measured (rejected); from the pods' 8470 dry run x 1.3: 22 to 900* | 1,800 s* |
| gs_term (d = 7-17) | 6 | 10.7 at d13 with 40,000 shots (77.2 us term), so 15 to 90* | 250 s* |
| BB72 baselines: relay / bposd (10,000 shots), relay_full (2,000), p = 0.3 % and 0.2 % | 6 | 46, 14, 269 (p3); 35, 11, 193 (p2)* | 570 s* |
| BB72 our terms 3S, 2S, 3F, 2F | 4 | 15, 8, up to 450, up to 210* | 680 s* |
| gross baselines relay / bposd (1,000 shots), relay_full (300) | 3 | 116, 237, 935* | 1,290 s* |
| gross G-4R (1,000 routed) and G-4F (100 routed) | 2 | up to 2,690 and 5,360* (synthetic heaviest shots); at the M4's real routed costs x 1.1-1.5: 1,100-1,450 and 1,300-1,800 | 2,400-8,000 s* |

Per-shot times measured in the dry run:

- BB72: relay 4.59 / 3.48 ms (p3 / p2), bposd 1.36 / 1.08 ms, relay_full 134.3 / 96.4 ms.
- gross: relay 115.9 ms, bposd 237.0 ms, relay_full 3,117.5 ms.
- Per block at d = 7-17: plain PM 19.99, 41.65, 84.00, 135.98, 237.63 and 399.95 us; correlated PM 47.46, 101.45,
  224.5, 418.43, 743.69 and 1,185.2 us.
- Grace points, per shot at p = 0.3 % / 0.6 %:

| decoder | d = 5 | d = 9 | d = 13 | d = 17 | d = 21 | d = 31 |
|---|---|---|---|---|---|---|
| plain PM | | | 58.09 / 151.78 us | | 313.9 / 950.6 us | 1,641 / 4,659 us |
| correlated PM | 10.54 / 14.37 us | 51.73 / 100.92 us | 144.98 / 407.46 us | 425.27 / 1,088.4 us | 1,145.6 / 3,176.8 us | 8,771 / 22,561 us |

The registered plan is rep-major (`--order rep`, the default), so each repetition's longest BB processes get windows
of their own. To fit the session into one evening, the slowest items run on the first N shots, or the first N routed
shots, of the stage's drawn set: Decoder Switching's front end on 50,000 shots per d (its sets hold 50,000; 200,000
before), gross full-DEM Relay-BP on 75 shots (300 before), gross BP-OSD on 500 (1,000), gross G-4R on 300 routed
shots (1,000) and G-4F on 25 (100). The table above has the earlier counts. With the new counts the plan's estimates
(`plans.py --show`) come to about 27 min of windows per repetition and 80 min for the session (46 windows). With the
BB warm-up window (about 4 min) and about 10 s of reference segments per window, that is about 91 min, plus any
replacements.

### The clocks under Balanced (16:44)

The cores do not hold a steady clock. The clock probe (a chain of dependent adds) reads 2.0-2.5 GHz on an idle vCPU
and 1.0-2.6 GHz on the slots during the windows; nominal is 2.9 GHz and turbo 3.5 GHz. The reference segment (plain PM,
d = 9, per block) spreads by 20-45 % over the 8 slots of one segment, against a limit of 3 %. It spreads 31 % over 4
slots, and 8.7 % over 4 repeats on a single slot. Every dry window therefore failed the reference check, at every
concurrency tried.

Nothing else was visibly running on the host: one VM process, and no other user's VM. The active power plan is
Balanced, with a minimum processor state of 5 %. High performance (`8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`) is
available on the host, but switching to it is a machine-wide change and waits for the machine's owners
(`powercfg /setactive 8c5e7fda-e8bf-4a96-9a85-a6e23a8c635c`, then the dry run again). Until the clocks hold, the 3 %
spread rule cannot pass on this host.

## High performance (13 October, from 18:02 PDT)

The host's owners switched the power plan to High performance at 18:02 and turned turbo boost off at 18:33 (boost
mode 0). The reference segment's spread within each segment (largest / smallest, the registered check) was:

| run | turbo | 8 slots | 4 slots | 1 slot |
|---|---|---|---|---|
| dry3 session reference (18:16) | on | warm 1.027; 1.039, 1.026, 1.028 (36.59 us per block) | 1.049, 1.028, 1.008, 1.006 | one value per segment; 1.026 across 4 segments |
| dry2 windows (18:06-18:15, 8 windows) | on | before / after 1.013-1.053; 1 window of 8 passed every check | | |
| dry4 session reference (18:55) | off | warm 1.074; 1.221, 1.124, 1.060 (55.78 us) | 1.036, 1.044, 1.033, 1.040 (42.1 us) | 1.046 across 4 segments (53.4 us) |

With turbo on, the clock probe read 2.77-2.92 GHz and no value exceeded 1.041 x the session reference; about
half of the 8-slot segments were within 1.03. With turbo off, the probe read 1.9-2.6 GHz, below the nominal 2.9 GHz,
and the values rose to 1.15-1.47 x the turbo-on reference. `gs_time`'s own probe check (exit 4) fired in 6 of the 8
turbo-on windows, which is why Decoder Switching now runs in windows of its own.

Turbo has been on again since 19:05 (boost mode 2, Aggressive, the plan's default), and the segment limit is now 1.06.
A limit of 1.03 would have rejected about half the turbo-on windows; the 1.10 bound against the session reference, the
8 slots and the 3 attempts are unchanged.

## Linux host (fallback CPU host, 13 October, from 20:30 PDT)

When the Windows host cannot be reached, the same harness runs on a Linux workstation in its Linux host mode
(`cfg_box.json`, `"host": "linux"`). The plan, the reference segments, the clock probes, the limits (every value at
most 1.10 x the session reference, each segment's largest at most 1.06 x its smallest), the 3 attempts and the records
are those of the Windows host. What changes:

- The host: AMD Ryzen 9 9950X, 16 cores and 32 threads in two CCDs of 8 cores (each CCD has its own 32 MB L3 cache),
  123 GB RAM, NVMe, Ubuntu 24.04.1 with kernel 7.0. The cpufreq driver is amd-pstate-epp in active mode. For the
  timing the machine's owners set the governor to performance on every CPU (energy-performance preference
  performance, boost on) at 20:29; before, it was powersave with balance_performance. The harness changes no setting
  and records them in every window. The machine is shared: other accounts are logged in, and they are asked to run
  nothing on it during a session.
- The tmpfs is `/dev/shm/srt_box`: `SRT_TMPFS=/dev/shm/srt_box` for setup_env.sh, prep.sh and tmpfs_libs.sh, and
  `SRT_CFG=cfg_box.json` for `prep.sh shots`. It holds the same interpreter (Ubuntu 24.04's CPython 3.12.3, copied
  in), the venv with the pinned packages, the C libraries and everything else a timed process reads. pinlibs.py locks
  the loader and the locale files, as on the Windows host.
- Placement, read from /sys (`thread_siblings_list`, `cache/index3/shared_cpu_list`): the 8 slots are the 8 cores of
  CCD 0, one CPU each (CPUs 0-7), and their SMT siblings (CPUs 16-23) stay idle. The harness, its monitors, the spare
  clock probe, the fetcher and the hand-off copy run on CCD 1 (CPUs 8-15 and 24-31). `timed_ccd` in the config picks
  the CCD. It is CCD 0 because another account's long job was pinned to four CPUs of CCD 1 when the host was set up.
- The disk rule on a shared machine: other accounts' I/O does not count, so the check covers our own processes. The
  harness reads its own read_bytes from /proc/self/io before a window's first reference segment and after its last.
  That counter includes every child the harness has reaped (the reference processes, the timed processes, the fillers
  and the clock probes) and, through them, the children those reaped. The window passes the rule when read_bytes did
  not grow (`disk_delta`, with `rule` "own read_bytes"). Each process's own block reads, from its wait4 rusage with its
  children included, are kept beside its exit code (`read_bytes`). The machine's /proc/diskstats is still recorded, as
  `disk_box_delta`. The session reference uses the same rule.
- The monitor, per window and per session reference: the cpufreq governor, preference, limits and boost at both ends;
  every CPU's frequency every 2 s with the load average (`monitor.jsonl`); each CPU's busy share over the window
  (/proc/stat), with the slots' siblings' highest share as `sibling_busy_max`; and the CPU seconds of every process
  that is not the harness or one of its descendants, with its user, command and the CPUs it was seen on (`not_ours`).
- Saving and the hand-off use local folders: `save_to` is `~/srt_box_io/out` and `handoff_from` is `~/srt_box_io/in`.
  `handoff_pull.sh` copies the units from the folder the GPU passes fill (a local path or HOST:PATH over ssh), each
  unit's marker last. `fetch_box.sh` copies the saved records to the collecting machine every 15 minutes and once at
  the end, and logs each copy. Neither reads the tmpfs, and both run on CCD 1. There is no Windows step.

Setup and session, as the harness's user:

```
export SRT_TMPFS=/dev/shm/srt_box SRT_CFG=cfg_box.json QRT=$HOME/qrt_box
W=<this folder>; P="bash $W/prep.sh"
$P env && $P code $QRT/code_F.tgz <F's 40-hex hash> && $P relay $QRT/relay_testdata.tgz && $P val
cp d9_p0.006_X_val_000.npz /dev/shm/srt_box/data/val/ && cp d9_p0.006_X_val_000.npz /dev/shm/srt_box/data/probe/
setsid -f bash $W/handoff_pull.sh HOST:HANDOFF ~/srt_box_io/in > /dev/null 2>&1 < /dev/null
cd /dev/shm/srt_box/cpu_ws && R=/dev/shm/srt_box/runs/S2 && mkdir -p $R && setsid -f bash -c "
  export LD_LIBRARY_PATH=/dev/shm/srt_box/lib LC_ALL=C.UTF-8 TZ=UTC0
  bash prep.sh shots $R ~/srt_box_io/in &&
  ../venv/bin/python harness.py topo --cfg cfg_box.json --out $R/topo.json &&
  ../venv/bin/python harness.py sessionref --cfg cfg_box.json --out $R/warm --repeats 1 &&
  ../venv/bin/python harness.py sessionref --cfg cfg_box.json --out $R/sref &&
  ../venv/bin/python plans.py --parts bb_warm,gref,gs_fe,gs_term,surf_base,surf_term,surf_share,bb --reps 3 --out $R/plan.json &&
  ../venv/bin/python harness.py run --cfg cfg_box.json --plan $R/plan.json --sessionref $R/sref/sessionref.json --out $R/win
  " > $R/session.log 2>&1 < /dev/null
setsid -f bash $W/fetch_box.sh S2 HOST:DEST > /dev/null 2>&1 < /dev/null     # every 15 minutes; --once at the end
```

The validation chunk must be the registered probe chunk (sha256 `49d1bea9...`). A chunk redrawn with
`convlines.testset val` has other bytes and other shots, and `gs_time` then stops on its probe check (the first dry
run's Decoder Switching windows, exit 1). The export and the analyses run as on the Windows host (`--dry` on the fit
of a dry run). `analysis_bb.py` expects a Xeon in the CPU rows' `device` column and warns otherwise; on this host the
rows name the AMD Ryzen 9 9950X.

### Dry runs on the Linux host (13 October, 20:39-20:57 PDT)

Three dry runs, each one repetition of every CPU part at 8 slots, on dry shots and dry hand-off units drawn on the host
with prep.sh's dry steps (the families' code: the stage-2 draft 9e6ece9). "Spread" is a segment's largest value over
its smallest (limit 1.06); the session reference is the median of 3 segments of 8 values.

| run | host setup | session reference, us per block (its 3 segments' spreads) | window segments: median / largest spread; over 1.06 | windows rejected | length |
|---|---|---|---|---|---|
| boxdry1 (20:39) | siblings online, bluetoothd running | 17.251 (1.007, 1.007, 1.022) | 1.010 / 1.180; 1 of 32 | 3 of 16: 2 Decoder Switching (the redrawn probe chunk, above), 1 segment | 3.9 min |
| boxdry2 (20:44) | as boxdry1 | 17.320 (1.010, 1.004, 1.021) | 1.009 / 1.193; 2 of 36 | 2 of 18, both on a segment; their jobs passed in r001 and r002 | 6.5 min |
| boxdry3 (20:51) | siblings 16-23 offline, bluetoothd stopped, other processes off CCD 0 | 17.306 (1.010, 1.006, 1.009) | 1.007 / 1.018; 0 of 32 | none of 16 | 4.9 min |

The rejected segments of boxdry1 and boxdry2 had another process on the slot's core or on its sibling. In boxdry1's
w014, CPU 17 (the sibling of slot 1) was 26 % busy with bluetoothd, polkitd and a desktop session; in boxdry2's w007,
bluetoothd used 17.6 s of CPU in 81 s and ran on CCD 0. From 20:49 the host's owners took CPUs 16-23 offline, stopped
bluetoothd and kept every other process off CCD 0 (systemd AllowedCPUs 8-15,24-31 for the system slice and the other
users' slices), for the session only. Under that setup (boxdry3) every segment stayed within 1.018 and every value
within 1.016 x the session reference; the other processes' CPU time seen on CCD 0 was 0.84 s in 4.9 min, mostly
per-CPU kernel threads. The clock probe read 5.2-5.6 GHz on the slots, and no window saw a block read of our own
(read_bytes 0).

The export and the three analyses ran on boxdry2 (`--dry`), each with exit code 0: surface C-ref 18/18, C-base 12/12,
C-term 13/13 and C-share 24/24 (67 of 75 attempts accepted; the 8 of the rejected window were replaced), the fit
complete (plain 1.475, correlated 1.573, common 1.540); BB 15 of 15 slots, qualified, with grace.json; Decoder
Switching 12 slots with the pre-screen passed. Every comparison is withheld, as it should be: a dry run has no GPU side.

At the registered counts (each job's per-shot time from the dry runs, packed by plans.py), the session is 46 windows:
18.2 min for repetition 1 (with the 8 BB warm-ups) and 13.9 min for each of repetitions 2 and 3, 44 min of timed
windows plus about 2.2 s of reference segments and probes per window, so about 46 min, plus replacements. The dry
draw routes the heaviest shots to the CPU for our BB terms, so their times are upper bounds (per process: BB72 2F
514 s, gross G-4F 489 s, gross G-4R 313 s).
