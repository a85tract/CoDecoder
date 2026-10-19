# SPEED-RT stage 3: Figs. 3 and 4 in throughput mode (STAGE3_batched)

Stage 2 timed the decoders of the paper's Figs. 3 and 4 in reaction mode (batch one). This stage times the same
decoders in throughput mode (the paper's Section II B): the networks decode batches of 4,096 shots and the fallbacks
decode their routed shots in single-threaded processes. Stage 3 follows `SPEED_RT.md` section 0 (F_3, its beacon, B_3,
the analysis) and `STAGE2_common.md` C1-C8 except where this file says otherwise (section 9 lists every departure). It
never reads stage 2's timing records, except stage 2's Grace factor (section 6) and, for the d = 7 validation only,
stage 2's recorded logits (section 5). Code: `speed_rt/s3/` (`gpu3.py` and `run3.sh` on the pods, `cpu3.py`,
`plan3.py` and `win3.py` with `cfg_s3.json` on the workstation, `analysis3.py`, `draw3_bb.py`, `pod3.sh`,
`prescreen3.py`, `prep3.py`, `sanitize3.py`); sets: `speed_rt/s3/sets_s3.json`; the validation records: `speed_rt/s3/validation/`.
Every constant of the stage (networks, thresholds, counts, cuts, limits) is in this file and in that code.

## 1. What is timed

The networks, thresholds and fallbacks are those of Figs. 3 and 4 (the paper's Tables III, IV and VII). tau is the
frozen routing threshold; the routed fraction is stage 2's on its timing shots.

| point | network (sha256 prefix) | rule | tau | stage 2's routed fraction | fallback |
|---|---|---|---|---|---|
| surface d = 7, p = 0.6 % | `surface_v2/weights/d7_b1w64_sv2` (8a7a128f) | routed when abs(double(c)) < tau, else c > 0 | 0.90324866771698 | 2.05 % | correlated PM |
| d = 9 | `d9_shallow/weights/d9_b1w64_sh` (28d17a86) | same | 2.337686538696289 | 6.07 % | correlated PM |
| d = 11 | `surface_v2/weights/d11_b5_sv2` (f09e87d8) | same | 0.0 (q = 0) | 0 | none routed |
| d = 13 | `surface_v2/weights/d13_b5_sv2` (68069bda) | same | 0.0 (q = 0) | 0 | none routed |
| d = 15 | `surface_v2/weights/d15_b5_sv2` (bb16284c) | same | 1.9195706844329834 | 0.98 % | correlated PM |
| d = 17 | `surface_v2/weights/d17_b5_sv2` (e11d0847) | same | 2.2122440338134766 | 1.02 % | correlated PM |
| BB72, p = 0.2 % | `weights/bb/r12_final_ema.pt` (f2821c32) | routed when margin < tau, else the network's 12 bits | 2.979396343231201 | 0.25 % | balanced: sector Relay-BP; accurate: full-DEM Relay-BP |
| BB72, p = 0.3 % | r12, as above | same | 3.6329734325408936 | 1.94 % | as above |
| gross, p = 0.4 %, balanced | `weights/bb/g2_D12k113c10_final_ema.pt` (eb380295) | same | 2.2055678367614746 | 0.96 % | sector Relay-BP |
| gross, p = 0.4 %, accurate | g2, as above | same | 6.272581100463867 | 2.00 % | full-DEM Relay-BP |

Decoders timed, per point: our composition (network + fallback on its routed shots); CoDecoder alone (the network
deciding every shot, q = 0); each baseline of the figure alone: plain PM and correlated PM (surface), sector Relay-BP and
full-DEM Relay-BP (BB72 and gross). PM = PyMatching 2.4.0 built as stage 2 built it (`convlines.circuits.matcher`,
correlated with `enable_correlations=True`); Relay-BP = relay-bp 0.2.2 `RelayDecoderF32` with Table VII's settings, built
by `qnd.teachers.Decoder("relay", code, "X", p, sector)` (sector "basis" or the full DEM "all") as stage 2 built it.
Not timed: Decoder Switching, BP-OSD (ldpc and CUDA-Q), and Ising + PM: NVIDIA's batched path (its plans or PyTorch at
batch 4,096, validated against its PyTorch numerics as in `STAGE2_surface.md` 1.13) was not validated before this
freeze, so Ising + PM is out of this stage.

## 2. G: the network on the GPU, host to host

- G is the host-to-host wall time per shot of `speed_rt/s3/gpu3.py time`: per batch of 4,096 shots, the detector bits
  are copied from pinned host memory to the graph's input (a copy stream, two input slots), one CUDA graph runs the
  network, the per-shot outputs (surface: the fp32 logit c; BB: the fp32 margin and the 12 bits) are copied to pinned
  host memory, and the host applies the release's own rules (`convlines.decoder.route` and c > 0; `margin < tau` and
  the bits), collects the routed shots' indices and packs their detector bits into one contiguous host buffer for the
  fallback. Routing and packing are inside the timed loop. Mode `alone` (CoDecoder alone) is the same loop without
  routing or packing. A timing set whose size is not a multiple of 4,096 ends with a batch of the remainder, which has
  its own graphs.
- Engines, each the release's batched form of the deployed network in the deployed precision (fp16 matrix inputs, fp32
  accumulation): surface, the deployed Triton engine (`surface_v2/timing/engine.py` through `timing.deployed.engine`,
  FROZEN, the network's own stage count), run inside the graph on sub-batches of the largest power of two up to 4,096
  whose tensors fit the engine's int32 offsets (each shot's arithmetic does not depend on the batch); BB72, the
  multi-shot engine (`cascade/bb72/cascade_compare/batchopt/mfused.py`, the paper's Section VII B); gross, bb.fused's
  kernels batched by `cascade/bb72/cascade_compare/phase_c/pc_fused.Batched` (no multi-shot engine exists for g2).
  d = 7's deployed engine is a resident batch-one kernel with no batched form; its batched path is the Triton engine.
- A timed process: pinned to the core the pre-screen picks; it builds its engine and graphs, runs one untimed pass over
  the timing set, then whole passes until at least 10 s have passed between two `perf_counter` readings; G = that
  time over the shots decoded. Its window is stage 1's, applied as stage 1 applied it (`speed_rt/cascade/srtlib.py` Window,
  called by `gpu3.stage1_window`; `SPEED_RT.md` section 1.4): nvidia-smi samples the card every 100 ms and its compute
  applications every second; the window is rejected for another compute application, a gap over 1 s between samples, a
  clock-event reason other than GPU idle, applications clocks or the SW power cap, an enforced power limit below the
  default, or an SM clock below 90 % of the maximum in a sample without the SW power cap; the workload's own power cap
  counts as the card's speed under the workload (recorded). Its decisions and flags of its last pass must equal the
  decision pass's (section 5). A process counts only when its window is accepted, its outputs are equal and it exits 0
  (`run3.sh` logs every exit code); otherwise it is replaced under C5 (at most three attempts per repetition), and per
  repetition the first accepted attempt counts; a repetition with none withholds the slot.
- Processes: five fresh processes per (point, mode) on the surface code and BB72, three on the gross code (modes
  `route` and `alone`; the gross code's `route` once per tau), one at a time per pod, in the order route 1, alone 1,
  route 2, alone 2, and so on.
- The cards (the user's decision): G is timed on RunPod NVIDIA H100 80GB HBM3 (SXM) pods (required) and, if an RTX
  PRO 6000 Blackwell Server Edition is in hand before the GPU timing starts (a RunPod RTX PRO 6000 pod or the
  authors' cluster job holding one whole RTX PRO 6000, whichever is in hand first; on the cluster the card sits in a node
  shared with other jobs, and the node, its CPU and the other jobs' GPU count are recorded at the start; the same
  pre-screen applies), also on it; no other card. This depends on availability only, never on a result. Each card's
  G, composites and speed-ups are reported separately, never mixed. The CPU terms use the H100's decision pass. An
  RTX composite reuses the H100's C(N) only at a point where the RTX decision pass's routing flags equal the H100's on
  every timing shot; otherwise that point's RTX composites and speed-ups are withheld (its G and CoDecoder alone stay),
  and the number of differing flags is reported. A pod that arrives after F_3
  runs section 5's validation before its first timed process; a point that fails it is not timed on that pod. The networks may be timed on
  1 to 4 H100 pods, one network per process; with one pod, its work runs in the order: every decision pass, then the
  timed processes (point by point: surface d = 7-17, BB72 p = 0.2 % and 0.3 %, gross), while the CPU window runs.
- Pods: RunPod, secure cloud, pre-screened by stage 1's rule (`speed_rt/s3/prescreen3.py` runs
  `speed_rt.gpu_prescreen` unchanged) except that a pod whose container quota is below 20 CPUs is screened with a quota
  limit of 12 CPUs (recorded, decision D1); a pod that fails is replaced (at most five per role; if the fifth fails, the role runs on
  the candidate with the fewest failed checks, ties to the earlier, and its rows are descriptive). Roles: S-low (d = 7,
  9, 11), S-high (13, 15, 17), BB72 (both rates), gross; with fewer pods, one pod takes several roles. One environment
  on every pod (`speed_rt/s3/pod3.sh`): stage 2's BB72 pod lock (torch 2.11.0+cu128, Triton 3.6.0, numpy 2.2.6, Stim
  1.16.0) plus safetensors 0.8.0 and PyMatching 2.4.0, the environment of the validation (section 5); D9-SHALLOW's
  network is overlaid on the surface tree as its own pod script did. The H100 pod in hand (pod 1, Intel Xeon
  Platinum 8462Y+) has a container quota of 13.6 CPUs; by the orchestrator's decision D1 a pod with fewer than 20 CPUs
  is screened with a quota limit of 12 CPUs (a recorded departure: a timed GPU process uses one host thread and a copy
  stream), and its rows count like any other; the five GPU checks and the steal check are unchanged.

## 3. C(N) and B(N): the fallbacks on the workstation

- C(N): N single-threaded processes (`speed_rt/s3/cpu3.py`), each pinned to its own physical core of CCD0 (CPUs 0-7,
  slot k on CPU k), decode the routed shots of the timing set from memory with the decoder's batch entry point
  (PyMatching `decode_batch`; relay-bp `ObservableDecoderRunner.decode_observables_batch(parallel=False)`,
  `RAYON_NUM_THREADS=1`). The routed shots, in shot order, go to the N processes in equal contiguous chunks. Each
  process builds its decoder, loads its rows into memory and decodes 20 rows of the point's validation chunk untimed;
  then a barrier: each writes its ready time (CLOCK_MONOTONIC) into the group's directory, the common start T is the
  latest ready time plus 20 ms, each waits for T and makes one batch call. C(N) = (the last process's end minus T) /
  the number of timing shots (all shots, routed or not). A point with no routed shot has C(N) = 0 and no process.
- B(N), each baseline alone: the same harness decoding every timing shot (or the registered first shots, section 4),
  B(N) = wall time / shots decoded.
- N = 1 and N = 8 (every physical core of the timed CCD, the most C1 allows). Three runs of every (item, N).
- The settings and checks are C1's (the orchestrator applies them: CPUs 16-23 offline, governor performance, every
  other account's processes on CPUs 8-15 and 24-31). The window machinery is `speed_rt/cpu_ws/harness.py`'s, called by
  `speed_rt/s3/win3.py` with its configuration `speed_rt/s3/cfg_s3.json` (tmpfs root `/dev/shm/srt3_box`): before and
  after every window each used slot runs the reference segment (plain PM at d = 9 on validation chunk 0,
  `timing.cpu_term ref`), the harness's own read counter must not grow, every process must exit 0 and leave its
  record, every reference value must be at most 1.10 times the session's reference and each segment's spread at most
  1.06; slots whose jobs end early run the untimed filler until the window's last process ends. The session's
  reference (`harness.py sessionref`, the median of three full-concurrency segments) must lie within 1.10 times stage
  2's (17.306 us per block), because the Grace factor is stage 2's (section 6); if it does not, every Grace-basis value
  is withheld and the raw values stand.
- Windows (`speed_rt/s3/plan3.py`): per block and repetition, one N = 8 window whose eight processes (slots 0-7)
  decode the block's items one after another, each item with its own barrier and its own record; the block's N = 1
  runs (all three repetitions) are packed eight to a window, longest first, from stage 2's per-shot times. A rejected window runs again as a whole, at
  most three attempts (C5: a slot never established withholds every cell that uses it). Blocks run in this order,
  each with its repetitions 1, 2, 3 in turn: (a) surface, (b) BB72's compositions and sector Relay-BP baselines, (c)
  BB72's full-DEM baselines at N = 8, (d) BB72's full-DEM baselines at N = 1, (e) the gross code's compositions and N = 8
  baselines, (f) the gross code's N = 1 baselines.
- The window budget is 55 minutes from the first window's start (decision D3). A block starts only if 1.25 times its
  expected length (section 4) fits in what remains; otherwise it and every later block are not timed and are reported as not
  timed. The same deadline holds for every window and every repeat: a window, first or repeated, starts only if 1.25
  times its expected length fits before the deadline, otherwise it and everything after it is not timed; at a barrier a
  process waits at most 60 s for its peers (then it fails, and its window is rejected). This realizes the cut order: Ising + PM (already out), the gross code's N = 1 baselines, the gross code, BB72's
  N = 1 full-DEM baseline.

## 4. The timing shots

Fresh shots from this stage's beacon through `speed_rt/sets.py` with new tags (`speed_rt/s3/sets_s3.json`):
`s3-surf-d<d>` (d = 7-17, 20 chunks of 10,000: 200,000 shots each), `s3-bb72-p2` and `s3-bb72-p3` (10 chunks: 100,000
each), `s3-gross-p4` (1 chunk of 10,000 drawn; the gross code's timing set is its first 2,000 shots, by the orchestrator's
decision D3: stage 2 timed the gross composition on 25 routed shots per process, and 2,000 shots give about 40 routed
shots per run of the accurate composition). Surface chunks are drawn by `speed_rt.draw --kind surface --sets-file
speed_rt/s3/sets_s3.json`, BB chunks by `speed_rt/s3/draw3_bb.py` (`speed_rt/bb/srt_bb.py` with this stage's sets
file), on the workstation (x86-64, AVX2, Stim 1.16.0, numpy 2.2.6), once; the same bytes go to the pods; every chunk's
sha256 is in the manifests of B_3 and is checked by every process that reads it.

Registered cuts (the first shots of the set in shot order): BB72's full-DEM Relay-BP alone, the first 4,000 shots at
each rate; the gross code's sector Relay-BP alone, the first 1,000; its full-DEM Relay-BP alone, the first 40. Every
other item decodes its whole timing set (its routed shots, for a CPU term).

Expected window time (`speed_rt/s3/plan3.py` with stage 2's raw per-shot times, which come from per-shot loops; the
batch calls are faster): (a) surface 9.5 minutes (9 windows), (b) BB72's compositions and sector baselines 13.9 (6),
(c) BB72's full-DEM baselines at N = 8 3.0 (3), (d) at N = 1 3.9 (1), (e) the gross code's compositions and N = 8
baselines 15.6 (4), (f) its N = 1 baselines 1.9 (1): 47.8 minutes in all, inside the 55-minute budget.

GPU session on one pod (decision D4): 89 timed processes (surface 6 x 10, BB72 2 x 10, gross 9) and nine decision
passes; from the dry runs on pod 1 (one process: d = 17 33 s, d = 15 26 s, d = 7-13 13-20 s, BB72 13 s, gross 12 s;
a decision pass 3-13 s) about 28 minutes, under 40, so five fresh processes per (point, mode) stay (three on the gross
code).

## 5. Correctness

- Before F_3, on validation shots that are neither test shots nor this stage's timing shots (the first 8,192 shots of
  stage 2's timing set of the point, `srt-surf-d<d>` chunk 0, `srt-bb72-p<k>` chunk 0, `srt-gross-p4` chunk 0,
  regenerated from stage 2's beacon and equal to stage 2's manifests), the batched path's decisions and every tau's
  routing flags must equal the deployed reaction-mode pipeline's shot for shot on the same pod (`gpu3.py validate`;
  the deployed pipeline at batch one: the Triton engine at d = 9-17, `bb.engine.Deployed(model, "fused")` on BB72 and
  the gross code; at d = 7, whose resident kernel needs an H100 build, stage 2's recorded logits of that kernel on the
  same shots, `speed_rt/s2/surface/records/low/masks/val_d7_b1w64_sv2.npz`). The records are in
  `speed_rt/s3/validation/`; a point that fails is not timed. Result on pod 1 (H100): all nine points pass, 0
  decisions and 0 flags differ; the per-shot floats are bit for bit equal at every point but d = 7 (the Triton engine
  against the resident kernel: 246 of 8,192 logits bitwise equal, every decision and flag equal).
- The decision pass (`gpu3.py decide`, untimed, once per point on its pod after B_3): the batched path over the timing
  set gives the routing flags (the CPU terms' rows), the decisions and the per-shot outputs. Every timed GPU process's
  decisions and flags must equal it (a process that differs fails and is replaced under C5).
- Failure counts on the timing shots (decisions against the shots' observables; a composition takes its fallback's
  predictions on the routed shots, from the N = 1 run 1 processes) are reported descriptively only. Accuracy in the
  figures is the registered tests' (Figs. 3 and 4 unchanged); only the time axis is new.

## 6. The composite and the statistics

- T(N) = G + C(N) per shot, the paper's Eq. (1) with separately measured parts (GPU on the pod, CPU on the
  workstation). Primary = this sum. The pipelined bound max(G, C(N)) is reported beside it, descriptively.
- Cost per round = time per shot / rounds per shot: d on the surface code; 6 on BB72 and 12 on the gross code (the
  rounds of the paper's BB shots).
- Grace basis: every CPU time (C and B) times stage 2's common factor 1.5097605583565314
  (`speed_rt/s2/surface/results/fit_s2.json`, `applied`), read and checked by the analysis; GPU terms unconverted.
  Every value is also reported raw.
- Statistics (C6): our side takes the slowest accepted of its fresh processes (G: five, three for the gross code; C(N):
  three runs), every baseline the fastest of its three runs. Speed-up = baseline / ours at the same N and basis. "Less
  than" means strictly below. No statistical tests. Rounded against us: our times up, speed-ups down.

## 7. Claims and outputs

For each composition and N (1 and 8), on the Grace basis and raw: its cost per round; its speed-up over its fallback
alone (correlated PM; sector Relay-BP for balanced, full-DEM Relay-BP for accurate) and over the other baselines of its
figure at the same N (PM; the other Relay-BP); whether its cost per round is below 1 us (the round period the paper
cites). Plus CoDecoder alone's batched cost per round per network (G alone / rounds). `speed_rt/s3/analysis3.py`
writes `numbers.json` (every value and repetition), `table.md` and `claims.json`; it checks every record against this
freeze (the sha256 of the frozen files, the manifests and the seeds, the windows' acceptance) and withholds any value
whose slot is not established. It runs twice with byte-identical outputs.

## 8. Order of events

1. F_3 (this file, `speed_rt/s3/`, the validation records) committed and pushed to the public repository and GitHub.
2. T_3, the round (the first drand mainnet round at least 600 s after T_3) and R_3 by `speed_rt.beacon fetch --sets
   speed_rt/s3/sets_s3.json`; seeds by `speed_rt/sets.py`; the draw; B_3 (the beacon record and the manifests)
   committed and pushed. Nothing of stage 3 is timed before B_3 is public.
3. The decision passes on the pods; the flags go to the workstation (`SHA256SUMS`, C2). The GPU sessions; the CPU
   session in the window the orchestrator opens (C1 settings).
4. The analysis, twice; records and outputs committed and pushed.

## 9. Departures from STAGE2_common

- The Grace factor is stage 2's, not refitted (C3 asks a later stage for its own reference points); the session
  reference must match stage 2's within 1.10 (section 3) instead.
- Throughput mode: G is host to host at batch 4,096 (stage 2: batch one, input on the GPU, plus a handoff term); C(N)
  and B(N) are wall times of N processes with batch calls (stage 2: per-shot loops at 8-way concurrency).
- The pre-screen's quota limit is 12 CPUs on pods with fewer than 20 (decision D1; pod 1 has 13.6).
- The gross code's timing set is 2,000 shots (D3), the CPU window 55 minutes.
- The GPU window is stage 1's rule (SPEED_RT.md 1.4), as in stage 1's batched mode, not stage 2's watch rules.
- A rejected window runs again as a whole (an N = 8 run is one measurement of eight processes).
- The cut order of section 3 and the registered first-shot cuts of section 4.
