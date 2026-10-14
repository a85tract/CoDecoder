# SPEED-RT stage 2, BB72 and gross: the BB comparisons on one GPU pod and the Linux workstation (STAGE2_bb)

This family's registration is part of freeze F_2 with `STAGE2_common.md` (the common
rules, cited as C1-C9), `STAGE2_surface.md` and `STAGE2_gapsw.md`. It changes no file that stage 1 froze. Sections follow `SPEED_RT.md` section 1.

## 1.1 What is timed

This stage re-times the BB comparisons of the paper (rows B1-B10 of the speed inventory and the BB parts of headlines H1, H2, H3 and H5; B11, the earlier gross test, is out of the re-test by decision): our pairs
and networks for BB72 (p = 0.3 % and 0.2 %) and the gross code (p = 0.4 %) against the CPU baselines (sector Relay-BP, full-DEM Relay-BP, ldpc
BP-OSD) and CUDA-Q's BP-OSD. The work is split by machine. The H100 pod times GPU work only: our networks' GPU terms and handoffs, CUDA-Q (host to
host, not converted) and the fp32 graph, and it makes one untimed GPU pass on the timing shots whose output is the hand-off to the CPU side. Every
CPU part (our pairs' Relay-BP terms, sector Relay-BP, full-DEM Relay-BP, ldpc BP-OSD) is timed on the Linux workstation by the stage's CPU session with this
stage's own timing code, and every CPU time is put on the Grace basis. Composite = GPU term + handoff + Grace-basis CPU term. No Alpine CPU, no
Apple M4, no CPU timing on a pod.

| side | items (fresh processes per item) | comparisons it feeds |
|---|---|---|
| GPU pod, one session | BB72: r11 (5), r12 (5): GPU term and handoff; CUDA-Q BP-OSD at p = 0.3 % (3) and 0.2 % (3). Gross: g2 (3): GPU term and handoff; CUDA-Q BP-OSD (3); graph32 (5, descriptive). 27 slots in 5 rounds | the GPU side of every pair (B1, B2, B4, B5, B6, B7, B9, B10), CUDA-Q as a baseline (B3, B6) |
| the Linux workstation, the stage's CPU session | baselines: sector Relay-BP and ldpc BP-OSD at BB72 p = 0.3 % and 0.2 % (first 10,000 shots of chunk 0), full-DEM Relay-BP at both p (first 2,000); gross: sector Relay-BP (first 1,000 shots of chunk 0), ldpc BP-OSD (first 500), full-DEM Relay-BP (first 75). Our CPU terms: BB72-3S, -2S (sector, first 1,000 routed), -3F (full-DEM, first 1,000 routed), -2F (full-DEM, up to 5,000 routed), gross G-4R (sector, first 300 routed), G-4F (full-DEM, first 25 routed). 15 items, 3 processes each | the CPU side of every pair, and the baselines (all rows above) |

- The fast mode (r11) and a network alone have no CPU term. The bar of a fast or balanced mode and of a network alone is the fastest of sector
  Relay-BP, ldpc BP-OSD and CUDA-Q; the bar of an accurate mode is full-DEM Relay-BP (`speed_rt/bb/analysis_bb.py`, as before).
- g2 is timed once (3 fresh processes) and serves every gross comparison. Each pair's two sides come from two machines by design (the rules of
  15:45), so no comparison has both sides on one machine except a CPU term against its own baseline: the Grace-basis ratio of a routed
  shot's mean time to the same decoder's baseline mean (B9) is computed inside the CPU session, with one factor on both sides.
- A CPU item is `bb.timing opponent` (a baseline: its mean time per shot on the first shots of test chunk 0, one decode call per shot) or
  `bb.stage3 cputerm` / `bb.stage4_gross cputerm` (the routed fraction x the mean time per shot over the first routed shots in shot order; a shot
  is routed when its margin in the GPU pod's read is below the cell's frozen tau). The exact command and environment of each item on the Linux workstation are in `speed_rt/cpu_ws/bb_jobs.json`
  (15 timed items, 3 repetitions each, and 8 untimed warm-up processes, one per CPU decoder, as the registered sessions ran); `speed_rt/bb/cpu_item.sh` calls the same modules for one item run alone
  (`--list` names the 15 items and their shot counts; `--warm` the warm-up processes).
- Rounds on the pod: 5; every round gives each item that has a slot in it one fresh process, in a fixed order (r11, r12, CUDA-Q at p = 0.3 %, CUDA-Q
  at p = 0.2 %, g2, CUDA-Q on the gross code, graph32; r11, r12 and graph32 have 5 slots, the others 3). On the Linux workstation: 3 slots per item, the two
  sides of each comparison interleaved across slots by the session's scheduler (STAGE2_common C1).
- The untimed GPU pass ("reads", the hand-off): r12 on BB72 p = 0.3 % chunks 0-19 and p = 0.2 % chunks 0-29 (`bb.route infer`, the H100 named as
  stage 3 did), g2 on the gross code chunks 0-14 (`bb.stage4_gross infer`), on the timing shots, blind. Its 65 margin files (`preds/*.npz`, 11 MB,
  `preds.sha256`) are what the CPU terms route with.

## 1.2 Code

- The frozen BB trees, unchanged: `bb/bb72/ising_compare` (DEPLOYED 2407497d, policies, the engine signature) and `bb/gross_stage4/ising_compare`, with
  the three networks (`weights/bb`, sha256 in `freeze.json`). Frozen taus: r12 3.6329734 (p = 0.3 %), 2.9793963 (p = 0.2 %), gross G-4R 2.2055678,
  G-4F 6.2725811.
- New, all in `speed_rt/`: `bb/pod_bb.sh` (the GPU session: setup, ensure, prescreen, reads, timed), `bb/prescreen_bb.py` (stage 1's pre-screen),
  `bb/srt_bb.py` (the shim: the beacon's seeds and the draw's manifest), `bb/attempts.py` (the attempt log), `bb/cpu_item.sh` (one CPU process), `bb/analysis_bb.py`,
  `bb/collect.sh`, `bb/draw.sh`, `bb/sets_bb.json` (this stage's sets file), `bb/hcvenv.lock`, `bb/cpuvenv.lock`; shared: `watch.py` (the window record of the GPU
  items, unchanged since 14:00; its host numbers are recorded and decide nothing on a pod), `sets.py` and `beacon.py` (frozen in F_1), `grace_fit.py` (the lead's).
- GPU pod environments, each equal to its lock (`uv pip freeze | sort` against the lock): s3venv (BB72: Python 3.12.14, torch 2.11.0+cu128, triton 3.6.0,
  numpy 2.2.6, scipy 1.18.1, stim 1.16.0, relay-bp 0.2.2, ldpc 2.4.1, cudaq-qec-cu12 0.8.0, cuda-quantum-cu12 0.16.0.post1: r11, r12, CUDA-Q at BB72, the
  reads); g2venv (torch 2.11.0+cu126, triton 3.6.0: g2, graph32, the gross reads); cqvenv (CUDA-Q on the gross code); hcvenv (the pre-screen's GPU
  checks and the attempt log: the g2 lock's torch, PyMatching 2.4.0, safetensors 0.8.0).
- CPU environment on the Linux workstation: `speed_rt/cpu_ws/setup_env.sh`'s (Python 3.12.3 (Ubuntu 24.04's CPython, copied into the tmpfs), numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2, ldpc 2.4.1, stimbposd 0.2.0, sinter 1.16.0 and beliefmatching 0.2.0, the versions of winserver's environment; no torch); the earlier registered CPU
  sessions ran the same package versions on CPython 3.12.14 (`bb/cpuvenv.lock`). Every CPU process runs single-threaded (OMP, OPENBLAS, MKL, NUMEXPR and RAYON threads 1), pinned to one
  core by the CPU session's harness. The Relay-BP decoder is the one of `qnd.teachers` (RelayDecoderF32, 80 pre-iterations, 300 sets x 60 iterations, stop
  after 5 solutions; sector "basis" for sector Relay-BP, "all" for full-DEM), and the CPU BP-OSD baseline `bposd_ms80_cs60` of `bb/opp.py`: ldpc 2.4.1 `BpOsdDecoder`, min-sum (scale 1.0), 80 iterations, OSD-CS order 60, X sector, one thread
  (stimbposd is not on the BB path), all as registered. CUDA-Q's `nv-qldpc-decoder` (ms80_cs60, batch 1; `bb/gpu_bposd.py`) runs on the GPU pod, its OSD on the pod's host, pinned to the timed core.

## 1.3 The timing shots

- The beacon is stage 2's (STAGE2_common C8): one record, R_2, from the round after F_2; the record names the stage's sets file
  `speed_rt/sets_s2.json`, the union of the three families' sets files, which holds this family's sets (`speed_rt/bb/sets_bb.json`) unchanged.
- Sets: `speed_rt/bb/sets_bb.json`: srt-bb72-p3 (bb72, p = 0.003, split "test", 20 chunks of 10,000 shots), srt-bb72-p2 (0.002, "test", 30), srt-gross-p4
  (bb144, 0.004, "test4", 15). seed(set, j) = 2^62 + (the first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes big-endian)) mod 2^62, TAG =
  "convlines speed-rt SET" and a newline (`speed_rt/sets.py`).
- Draw: once, on an Alpine x86-64 node with AVX2, Stim 1.16.0, numpy 2.2.6, by `speed_rt/bb/draw.sh` after the beacon record exists (no timing): the host
  check (the original test sets regenerated there equal the published arrays), the draw, the seed check (every manifest seed recomputed from the record, every
  file's sha256, every set complete). `beacon.json`, `manifest_bb72.csv`, `manifest_gross.csv`, `seedcheck.json` and `draw_SHA256SUMS` are committed with this
  stage's B commit. The 65 shot files (64 MB) go to the GPU pod and to the Linux workstation by sha256-checked copy, never regenerated: every BB step that loads shots checks the
  file against the manifest, so the two machines read byte-identical shots, and a test split has no seed without the beacon (`srt_bb.py`).
- Counts. Every CPU item times the first N shots, or the first N routed shots, of its chunk in shot order (N in the table of 1.1 and in
  `speed_rt/bb/cpu_item.sh --list`). The gross items' counts were cut before F_2 so that the CPU session fits one evening (the authors' decision):
  full-DEM Relay-BP 300 -> 75 shots, ldpc BP-OSD 1,000 -> 500 shots, G-4R 1,000 -> 300 routed shots, G-4F 100 -> 25 routed shots. Every other count is as before.
- The pre-screen and the environments never read a timing shot. The reads and everything after them start only after B is public.

## 1.4 The GPU session: pre-screen, window and acceptance

- Pre-screen (`prescreen_bb.py`, stage 1's rule, SPEED_RT.md 1.5), on the idle pod after setup: five GPU checks (`surf_v2.gpucheck`, each a separate process
  that holds 1 GiB and runs 3 s of fp16 products; a check passes when the median SM clock is at least 97 % of the maximum, every clock-event reason is GPU idle,
  applications clocks or the SW power cap, the enforced power limit is the card's default, and the GPU is alone), CPU steal at most 1 % over 5 s on the container's
  CPUs, and a container quota of at least 20 CPUs. No host-CPU busy check and no PyMatching reference on the pod. The card, driver, power limit, CPU model, visible
  CPUs, cpuset, quota and kernel are recorded. A pod that fails is replaced (section 7).
- The timed core: the highest-numbered core of the container's cpuset on the GPU's NUMA node, recorded in the pre-screen. Every timed process is pinned to it (for
  the launch latency of the GPU items and CUDA-Q's host part); the session's own script stays off that core and its SMT siblings.
- Before every process the idle-card check (`bb.timing idle`: two samples 100 ms apart show 0 % utilisation and under 1,024 MiB used; 20 tries 3 s apart).
- Within the window (`speed_rt/watch.py`, the whole process): `nvidia-smi` every 100 ms; no other compute process on the card before or after; the throttle reasons
  none or idle in every sample (rule "throttle": CUDA-Q's OSD runs on the host and the idle gaps drop the clock); the BB modules apply their own 97 % clock floor
  inside their timed spans and require exactly one compute application (`bb.timing.gpu_watch`, the registered clock rule, unchanged). The window's host numbers
  (the timed core's busy share and steal) are recorded and decide nothing.
- A software power cap that the workload itself causes is not an interference for the GPU terms; the rule is the registered one.
- An attempt counts only when the idle check, the module's own acceptance and the watch accept it and the process exits 0 with its rows. The analysis counts a
  row only if its process belongs to an accepted attempt.

## 1.5 The CPU parts on the Linux workstation and the hand-off

- The stage's CPU session runs one process per slot, each the command and environment of one job of `speed_rt/cpu_ws/bb_jobs.json` (`speed_rt/bb/srt_bb.py` with `bb.timing opponent`,
  `bb.stage3 cputerm` or `bb.stage4_gross cputerm`; `QND_RELAY`, `BB_SHOTS`, `BB_PREDS`, `SRT_MANIFEST` and `BB_SESSION` name the Relay-BP clone's circuits, the draw's shots, the hand-off,
  the draw's manifest and the label of the rows, all in the tmpfs; `--out` is the process's own record directory in its window, C1). The command line is the whole code path of the item: one decoder per
  process, the same frozen module the earlier sessions ran. `speed_rt/bb/cpu_item.sh ITEM PROC OUT.csv` (the environment variables BBW_REL, BBW_DRAW, BBW_PREDS, BBW_RELAY and BBW_PY) runs the same item alone.
- The session's machinery is `speed_rt/cpu_ws/`'s, registered in STAGE2_common C1: the tmpfs, pinning one process per physical
  core, the concurrency of at most 8, the reference segments (plain PyMatching per block before and after each window, within 1.10 x the session's reference
  with at most 6 % spread within a segment), the CPU frequency record, the interleaving of the two sides of a comparison, and the acceptance of each window. This stage fixes
  what runs in a slot and what is read from the slot's rows.
- A slot is established when one of at most three attempts is accepted by the session's windows; the session records the accepted process of each slot
  (`accepted.json`: item, round 1-3, the process id of the rows). An item is established when its 3 slots are.
- Grace basis (STAGE2_common C3): every CPU time is the raw time multiplied by the stage's applied factor. Relay-BP (sector and full-DEM), ldpc BP-OSD
  and our pairs' Relay-BP terms are neither matching decoder, so they take the common factor when the plain and the correlated PyMatching factors agree within
  their spread, and otherwise exp((ln f_plain + ln f_corr) / 2); C3 fixes this before any data exist. CUDA-Q is not converted. The analysis publishes every raw
  time beside its Grace-basis value, the factor, its spread and the residuals.

- After B is public the pod runs `reads` (about 2 min) and the hand-off folder is pulled to the Linux workstation: `handoff/bb/preds/` (the 65 npz files and
  `preds.sha256`) and `handoff/bb/draw/` (the 65 shot files, the two manifests, `draw_SHA256SUMS`), with `SHA256SUMS` over everything and a `DONE` file written last.
  The route is STAGE2_common C2: the folder goes to `~/work/s2/handoff/bb/` with `SHA256SUMS` and a last `DONE`, and
  into the tmpfs at a window boundary with `SHA256SUMS` checked; a CPU slot reads only tmpfs files, and the shots on both machines are byte-identical.
- The GPU session needs nothing from the Linux workstation and the CPU session nothing from the pod beyond the hand-off, so the two run at the same time after the reads.
  The draw part of the hand-off (`handoff/bb/draw/`, with `DRAW_DONE`) is pulled as soon as B is public; every baseline item reads only the shots, so the CPU session can start those
  before the reads exist, and only the routed CPU terms wait for `DONE`.
- Partial results: STAGE2_common C1 holds (the pod's driver writes every finished slot's rows, attempt record and window to its disk as it goes, and the fetcher
  pulls them every 15 minutes and once at the end; the fetched records decide nothing before the frozen analysis).

## 1.6 Replacement and failure: one rule

STAGE2_common C5 holds, with these particulars:
1. An attempt is rejected when the idle-card check refuses it (no process starts) or the module's or the watch's verdict after it refuses it; on the Linux workstation, when a
   window check of C1 refuses it. It is failed when it does not finish its work (a non-zero exit, no timing row, no watch record). A slot has at most three attempts.
2. A slot with no accepted attempt is not established. Every comparison that uses the slot's item is withheld; an item with a missing slot has no value, and no
   other process and no earlier value stands in. A Grace reference slot not established withholds every converted CPU time (C5.2).
3. A GPU session that does not reach its end (the pod is lost, the third attempt of a slot failed) runs once more as a whole on a newly pre-screened pod, only the new
   session's records count, and if that session also fails every comparison that uses a GPU item is withheld (C5.3). The CPU session follows C5.3.
4. Pods: at most five are screened for the GPU role, one of which runs (C1). 
5. Nothing runs again because of a value it produced; every attempt, watch record and session is kept and published (C5.4).

## 1.7 Statistics

- Stage 1's rule: ours the slowest accepted process (GPU terms: 5 fresh processes for r11, r12 and graph32, 3 for g2) and the slowest of three repetitions (CPU terms, Grace basis); every
  baseline the fastest of its 3 fresh processes. Handoff = the frozen host-side allowance of 1.1 us per shot (routing decision, packing, indexing; `bb/DESIGN.md` 5.3 (c)). The measured
  CPU-GPU transfer (the pinned-memory round trip of one shot's copies, `px.hbb`, about 16-22 us) is a transfer: neglected in the composite, as in NVIDIA's setting, and published beside
  it as context (every process's value and the slowest). Composite = GPU term + handoff + CPU term (the routed fraction of the timing set's read x the mean time per shot over the
  first routed shots).
- Baselines: the fastest of three accepted processes (sector Relay-BP, ldpc BP-OSD, full-DEM Relay-BP on the Grace basis; CUDA-Q BP-OSD as timed).
- Every process's value and the range are published beside each counted value; the sensitivity of the two statistics is not used for any verdict.

## 1.8 Verdicts

- "X takes less time than Y" holds when X's re-measured value is strictly below Y's on this stage's basis (GPU terms on the pod, CPU terms on the Grace basis, one core).
  No statistical test, as in the paper. The sentences of rows B1-B10 are restated from the re-measured values, whatever they are; a failing comparison is written as in STAGE2_common C7 ("At POINT, OURS took X against BASELINE's Y (us per shot; GPU terms from the pod, CPU terms from the Linux workstation
  on the Grace basis); the statement does not hold."), and a missing result as there.
- Descriptive, deciding nothing: graph32 (B10's comparison of g2 with the fp32 graph), the machines' host records, the windows' host numbers on the pod.
- Out of the re-test, by decision (16:30): B11 (the earlier registered gross test's times). The sensitivity check: every verdict is also computed on the raw basis (the Linux workstation's own times,
  no Grace factor) and published beside the Grace-basis verdicts (`sensitivity_raw_xeon`, a key named after the earlier Xeon host; it holds the raw basis); it decides nothing.

## 1.9 What each time includes

H100 GPU term (the fused engine's device time per shot at batch 1) + handoff (the 1.1 us host-side allowance; the CPU-GPU transfer is neglected) + the routed fraction x the time of one Linux workstation core for the fallback decoder
on the routed shots, on the Grace basis (NVLink transfer neglected, as NVIDIA's Ising paper does); against one Linux workstation core (sector, full-DEM Relay-BP, ldpc BP-OSD, each one decode call
per shot, Grace basis) or CUDA-Q on the GPU pod (a batch-1 host-to-host call, its OSD on the pod's host pinned to one core, not converted). Table I's CPU times are restated on the Grace
basis; the raw-basis times, the CPU model and the factor are printed beside them.

## 1.10 The analysis

`python speed_rt/bb/analysis_bb.py --records RECORDS --out numbers.json --table table.json` (from the tree root) reads `gpu/` (rows.csv, attempts.json, session.json,
prescreen/prescreen.json; layout: `speed_rt/bb/collect.sh`) and `cpu/` (rows.csv, accepted.json, session.json, grace.json, host.json; `speed_rt/cpu_ws/export.py` writes them from the CPU session's run directory, STAGE2_common C1), checks the sessions (the GPU session reached its end
and passed its pre-screen; the CPU session qualified), counts only accepted attempts and accepted processes, withholds every item with a missing slot, converts every CPU time with the factor, and
writes every quantity with every repetition, the raw and converted CPU times, every reason a quantity is missing, the machines' host records and the sha256 of every input, and the table of rows
B1-B10 (old, new, verdict, basis). Deterministic; it runs twice and the two outputs must be byte-identical. The records, outputs and run log are committed and pushed.
Every record whose sha256 the analysis lists, and every hash list of this stage (the draw's `SHA256SUMS`, `preds.sha256`, the hand-off's `SHA256SUMS`, `COLLECTED.sha256`), names files by relative path only: the attempt records give the command and log relative to the output directory or the tree, and `collect.sh` drops any absolute directory prefix from a copied text record before it lists the files, so replacing paths for publication changes no hash.

## 1.11 Order of events

1. F_2 is pushed (this section, the code, `speed_rt/bb/sets_bb.json`); its beacon (T, round, R) follows from GitHub's record. The GPU pod is requested at the push, so that its setup and
   pre-screen (no timing shot) overlap the beacon wait.
2. After the beacon's round: `draw.sh` on Alpine; the beacon record and manifests are committed and pushed (B_2); the draw goes to the pod.
3. After B_2 is public: on the pod `reads`, then `timed`; the hand-off folder is pulled to the box, the CPU host (C2); the CPU session runs its slots when the folder is in the tmpfs, after the stage's Grace references and Decoder Switching (C8). Every session's start (machine, CPU model, time) is logged.
4. The analysis runs twice; its outputs are committed and pushed.

## 1.12 Departures from the paper's BB method

- The paper's CPU times (Apple M4) are restated on the Grace basis from the Linux workstation's AMD Ryzen 9 9950X; the paper's CUDA-Q numbers were on a Xeon 8480+ pod host, unpinned; here pinned to one core, on the pod.
- The registered sessions used no host gate for the CPU parts; the CPU session applies the `cpu_ws` machinery's checks. The GPU pod applies stage 1's pre-screen and the registered clock rule; no
  host-CPU check.
- The earlier GPU clock probe (6 s of 8,192 x 8,192 fp16 products) is dropped: it holds the H100 at its power cap and refuses a healthy pod; the registered rule (the modules' 97 % floor inside their
  timed spans, the idle card, the five GPU checks) stays.
- Replacement and failure: one rule (section 7), the same shape as the Cascade stage's.

## 1.13 Durations

GPU session, final code, on pod p15 (H100 80GB HBM3, Xeon Platinum 8470, 208 CPUs visible, quota 22.1 CPUs, NUMA node 1, timed core 207), dry draw, 27 slots in 5 rounds:

| step | run A | run B (after a full rebuild of the four environments) |
|---|---|---|
| environments (`setup`; `ensure` when they already match their locks) | `ensure` 0 s | `setup` 19 s (uv cache warm; a new pod's first `setup` is not measured, expected about 10 min) |
| pre-screen (steal, quota, five GPU checks) | 55 s, passed (steal 0.000 %) | 61 s, passed |
| reads (65 files, 11 MB) | 95 s | 93 s |
| timed (every slot, idle-card checks and records) | 151 s, session qualified, exit 0 | 157 s, session qualified, exit 0 |

Run A also exercised the refusal paths: the idle-card check rejected the first attempt of r11 (a held GPU), the first attempt of g2 was forced to fail, and both slots were
established on their second attempt. The pre-screen's failure branch (exit 4) was not forced with this build. From `TIMING_GO` to `DONE_gpu` the session takes about 4.2 minutes (reads 1.6, timed 2.6), plus the draw's copy to the pod; the pre-screen runs before B_2 and overlaps the beacon wait.

CPU parts, plain Linux python (Alpine, AMD EPYC 7543, the cpuvenv.lock environment, every shot count capped at 20, one process per item, not a timing result): all 15 items and all 8
warm-up processes ran with exit 0 and wrote their rows; the batch took 17 min 41 s, of which the gross items dominated (`cpu_G-4F` 439 s, `cpu_G-4R` 48 s, `relay_full_p0.004` 142 s).
The 20 shots are 1/5 to 1/50 of the gross items' counts before the cut (1.3), so these durations are no forecast. Their per-shot means (EPYC, 20 shots, heavy tails) were 21.8 s per routed shot
for G-4F, 2.1 s for G-4R and 7.0 s per shot for the gross full-DEM Relay-BP, which put one process of each at roughly 35 minutes on that machine at the counts before the cut (9 to 11 minutes at the stage's counts); the CPU
session measures the real figures. The hand-off is 65 npz files (11 MB) plus the draw (59 MB in the dry draw; its pull from Alpine took 20 s).

Analysis: `analysis_bb.py` on the GPU records of run B and two dry CPU fixtures (the earlier Xeon dry rows, and the Alpine smoke rows), registered slot structure and `--dry`, run twice
each: numbers, table and standard output byte-identical in all eight pairs. `--selftest` (the paper's earlier registered rows through the new code): 79 rows, 15 reproduce
their printed value, 64 have no registered record, 0 differ. The dry values mean nothing about speed.

CPU wall on winserver, from the winserver dry run of 13 October (dry shots; synthetic routing masks and predictions that route the heaviest shots, so the routed-term durations are upper bounds; values marked *
are extrapolated from the dry run's per-shot time to the registered shot count). One process each, per repetition: BB72 baselines (relay, BP-OSD, full-DEM relay) 46, 14 and 269 s at p = 0.3 % and 35, 11 and 193 s at
p = 0.2 %* (about 570 s in all); BB72 terms 3S, 2S, 3F and 2F 15 s, 8 s, up to 450 s and up to 210 s* (about 680 s); gross baselines relay 116 s, BP-OSD 237 s and full-DEM relay 935 s* (1,290 s); G-4R and G-4F up to 2,690 s
and 5,360 s* on the synthetic routing, 1,100 to 1,450 s and 1,300 to 1,800 s* with the routed costs of the earlier M4 session times 1.1 to 1.5. Per-shot times measured in the dry run: BB72 relay 4.59 and 3.48 ms (p = 0.3 % and 0.2 %),
BP-OSD 1.36 and 1.08 ms, full-DEM relay 134.3 and 96.4 ms; gross relay 115.9 ms, BP-OSD 237.0 ms, full-DEM relay 3,117.5 ms. The nine long processes (G-4F, G-4R and the gross full-DEM Relay-BP, each three times) therefore take about
22 to 30, 18 to 24 and 16 minutes each on the M4's routed costs, all other items together about 80 core-minutes; the gross items are the critical path (queued first, and the baselines can start when the draw part of the hand-off is in the tmpfs).
These figures are at the counts before the cut (1.3). At the stage's counts, from the same per-shot times and routed costs, a gross process takes about 4 minutes
(full-DEM Relay-BP, 75 shots), 2 minutes (ldpc BP-OSD, 500), 5.5 to 7.3 minutes (G-4R, 300 routed) and 5.4 to 7.5 minutes (G-4F, 25 routed).
The stage's other families share the machine: all CPU parts of the stage with three repetitions take 6 to 8 core-hours, 50 to 70 minutes on 8 slots when packed longest first (`plans.py --order lpt`), plus about 10 s of reference segments
per window (counts before the cut). The session's plan, repetition by repetition, put the whole CPU session at about 131 minutes before the cut and about
91 minutes at the stage's counts on winserver, and the box's plan at about 50 minutes (STAGE2_common C1). In that dry run every BB process exited 0 and the disk check saw no read in 9 of the 10 windows (the first, the warm-up window, had 1 read and 1 major fault); no window passed the reference check (the host's clocks varied between
1.0 and 2.6 GHz, and the spread of the reference segment over the 8 slots was 20 to 45 %, against a limit of 3 % then and 6 % now, C1).
