# SPEED-RT stage 2, the surface comparisons and Ising: GPU sessions on two pods, CPU parts on the Linux workstation (STAGE2_surface)

This family's registration is part of freeze F_2 with `STAGE2_common.md` (the
common rules, cited as C1-C9), `STAGE2_bb.md` and `STAGE2_gapsw.md`. It changes no file that stage 1 froze. Sections follow `SPEED_RT.md` section 1; Ising is section 1.13.
The pods of this family time GPU work only; every CPU part is timed on the Linux workstation (AMD Ryzen 9 9950X) by the stage's CPU session, from an untimed GPU pass of the pods
(section 1.14, STAGE2_common C2). The CPU items of section 1.14 run under C1's window machinery (`plans.py` parts `gref`, `surf_base` and `surf_term`); section 1.16 gives the durations of winserver's dry run.

## 1.1 What is timed

The surface speed comparisons of the paper (rows S1-S11, E1, E2 and E8 of the speed inventory, and the surface parts of headlines H2, H3 and H4):
ConvLines' pairs (a network on the H100, then correlated PyMatching on the routed shots on one core of the Linux workstation's Ryzen 9 9950X) of both network sets (the six
deployed networks and SURF-v2's six) against six baselines at d = 7, 9, 11, 13, 15 and 17: plain PyMatching (PM), correlated PM, Ising-Fast (M1) + PM,
Ising-Accurate (M4) + PM, and CUDA-Q's BP-OSD (min-LLR and BP 60); and two controls, Ising-Fast + correlated PM and Ising-Accurate + correlated PM. That is 8
opponents at each of the six distances: 48 speed comparisons per network set (the paper's "all 48"). The distances are split over two pods so
that each session is about half as long:

| pod role | distances | networks | what the pod holds |
|---|---|---|---|
| S-low | 7, 9, 11 | 6 (two per d) | G1-G6 and the hand-off for d = 7, 9, 11, G4 (the d = 7 Triton engine) |
| S-high | 13, 15, 17 | 6 (two per d) | G1-G6 and the hand-off for d = 13, 15, 17, the d = 17 network at study 4's tau (q = 10 %) |

**GPU side: every comparison's GPU items run on one pod in one session** (that pod's H100 and its Ising plans, CUDA-Q's BP-OSD with that pod's host
CPU). A pod never runs two timed processes at once: the session is one sequence of fresh single processes, each pinned to the session's timed core. **CPU
side: every CPU term is timed on the Linux workstation by the CPU session**, with this stage's own code, one code path per decoder (section 1.14); a comparison's CPU
terms and its GPU terms are composed in the analysis (section 1.10), and no comparison needs both sides on one machine, because the composite is a sum of
terms, as in the paper.

GPU items of a session (one slot = one process of an item in one round; `speed_rt/pod_s.py` lists every one):

| item | what is timed | processes per role (S-low / S-high) |
|---|---|---|
| G1 | the GPU term (a) of the role's six networks: `timing.gpu_term` over the 200,000 timing shots of the network's distance, five fresh processes each (rounds 1-5) | 30 / 30 |
| G2 | the handoff (c): five processes per d = 7 network (the d = 7 kernel's host round trip minus its device time, the largest of five), one per d = 9-17 network (max(1.1 us, the router's estimate)) | 14 / 6 |
| G3 | the routing masks of the role's networks on the timing shots at their frozen tau (untimed GPU pass, before the timed block; hand-off part 1); at S-high also the d = 17 mask at study 4's tau (q = 10 %) | - |
| G4 | S-low only: the fused Triton engine at d = 7, outputs bit for bit against its own reference, five processes | 5 / - |
| G5 | Ising (NVIDIA's Ising-Decoding 3ce4379, Fast = M1, Accurate = M4), the role's three distances: 12 plans per d (the eight registered plans and NVIDIA's four official plans; settings, timing-cache policy per kind and the official build's workspace in section 1.13.2); graph timing (one process per d and order, rounds 1 and 2) and NVIDIA's `trtexec` command (five repetitions per plan); the counted plan per model and d by `speed_rt/ising_select.py` (section 1.13.5); its residual detectors (untimed GPU pass after the timed block; hand-off part 2) | 186 / 186 (6 graph + 180 trtexec, when all 36 plans validate) |
| G6 | CUDA-Q QEC 0.8.0 BP-OSD (min-LLR and BP 60), batch 1, 200 host-to-host calls on chunk 0 of the timing set, three repetitions per d and arm (the pod's host CPU, not converted) | 18 / 18 |

253 timed slots at S-low and 240 at S-high with all plans validated (G1 30, G2 14 or 6, G4 5 or none, G5 graph 6 and trtexec 180, G6 18). CPU items, timed by the CPU session on the Linux workstation (the stage fixes what is
timed and how it is read; STAGE2_common fixes the machine's rules; the dry run's durations are in section 1.16): C-ref (the Grace fit's 18 reference points, the slowest of three), C-base
(the plain and the correlated PM per block at each distance on chunk 0 of the timing set, the fastest of three), C-term (our CPU term b: correlated PM on the
routed timing shots of each network, and at S-high the d = 17 mask at q = 10 %; three repetitions, the slowest counts), C-share (the plain and the
correlated PM shares after Ising M1 and M4 on the residual detectors of each model's counted plan, chunk 0; the fastest of three).

Order of a pod session: pre-screen, plan builds and checks (validation shots only), then after commit B: shots, masks (G3) and hand-off part 1, the GPU
block (rounds 1-5, each G1, G4, G5 graph in rounds 1-2, G5 trtexec, G6 in rounds 1-3), G2, the Ising residuals (untimed) and hand-off part 2.

## 1.2 Code

- The surface release's timing modules, unchanged: `timing.gpu_term`, `timing.deployed`, `timing.cpu_term`, `timing.ising_trt`,
  `surf_v2.session.networks` and `surf_v2.gpucheck`, with `convlines.circuits` (the verdict circuits and the DEM settings,
  `decompose_errors=True, approximate_disjoint_errors=True`).
- The stage's own files: `pod_s.py` (one pod session: prep, packm, timed, handoff, residuals, pack), `pod/run_s.sh` (its steps on a pod; `ROLE=low|high`),
  `gpu_prescreen.py` (the pod's pre-screen, stage 1's rule), `gref.py` (the reference circuits and the reference timing, run by the CPU session),
  `triton7.py`, `bposd_time.py`, `ising_d5.py`, `ising_time.py`, `ising_select.py`, `draw.py` and `alpine/draw_s.sh` (the draw), `analysis_s.py`,
  `analysis_surf.py`, `table_s.py` (the analysis), `sets_surf.json`, `config_surf.json`, `freeze_surf.json`, `circuits/` (the twelve reference circuits and
  `ref_circuits.json`), `pod/collect_s.sh`. Shared files imported unchanged: `watch.py` and `hostcheck.py` (the BB helper's: the clock window's samples,
  and the core pick and host record of the pre-screen) and `grace_fit.py` (the lead's); the first stage that uses them freezes them.
- Environment of the pods (the frozen pod recipe): Python 3.12.3, torch 2.11.0+cu126, Triton 3.6.0, TensorRT 10.14.1.48 and its `trtexec`, ModelOpt 0.47.0,
  Stim 1.16.0, PyMatching 2.4.0 (the Ising checks' accuracy reference), numpy 2.2.6, CUDA 12.1 `nvcc` for the d = 7 kernel (cubin sha256 8f4180d7...),
  CUDA-Q QEC 0.8.0 in its own venv. `pod/run_s.sh setup` checks every version. Environment of the CPU code on the Linux workstation: Python 3.12.3 (Ubuntu 24.04's CPython, copied into the tmpfs), numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2, ldpc 2.4.1, stimbposd 0.2.0, sinter 1.16.0 and beliefmatching 0.2.0, the versions of winserver's environment, no torch, Triton, TensorRT or GPU (checked: the CPU code imports with those modules blocked).

## 1.3 The timing shots

- Sets: `speed_rt/sets_surf.json`: `srt-surf-d7` ... `srt-surf-d17` (20 chunks of 10,000 shots each: the 200,000 timing shots of a distance) and twelve
  reference sets `srt-ref-d<d>-p<3|6>` (d = 5, 9, 13, 17, 21, 31; p = 0.3 % and 0.6 %; one chunk of 10,000 shots each; the CPU session's). The stage's beacon
  record (one for stage 2, STAGE2_common C8) names `speed_rt/sets_s2.json`, the union of this file with the other families'; seeds are `seed(set, j)` of `speed_rt/sets.py` from the stage's beacon.
- Draw: once, on an x86-64 CPU node with AVX2 (`speed_rt/alpine/draw_s.sh`, Stim 1.16.0, numpy 2.2.6), after the stage's beacon: the timing sets by Stim's
  detector sampler on the verdict circuit of d (`convlines`), the reference sets on NVIDIA's legacy circuit of (d, p). The manifests (`srt-surf.json`,
  `srt-ref.json`: every chunk's set, index, seed and sha256, per d the digest of the whole set, the first-200-shots digest for BP-OSD, each reference
  circuit's sha256) are committed with the stage's commit B. Every seed is recomputed from the record by the draw's own check.
- The reference circuits (NVIDIA's legacy memory circuits: rotated surface code, X basis, `code_rotation` "XV", d rounds, all detectors plus boundary
  detectors, idle, single- and two-qubit gate error p, SPAM 2p/3) are built by `python -m speed_rt.gref circuits --write` from NVIDIA's code at 3ce4379
  with the repository's builder (`evaluation/scripts/make_circuits.py`) and committed in F with their sha256 (`speed_rt/circuits/ref_circuits.json`). The
  p = 0.6 % circuits at d = 9, 13 and 17 equal the repository's verdict circuits byte for byte. The CPU session reads only these files and checks each sha256.
- Each pod receives only its role's timing chunks (`ship_shots.sh`: sha256-checked in transit) and reads them through a directory of symbolic links
  (`draw.py view --ds`), one per chunk, named as the validation chunk it replaces. The CPU session builds the same view of the role's distances from
  the draw's chunk files. Every routing mask records the digest of its shots, and every timed process and the analysis check it against the manifest.
- The plans' calibration and checks (G5) and the pre-screen read the studies' validation shots, never a timing shot.

## 1.4 Acceptance and the pre-screen

- Pre-screen of a pod (`python -m speed_rt.gpu_prescreen --out DIR`; stage 1's rule, SPEED_RT.md 1.5, on the idle pod before anything is timed on it): five
  GPU checks (`surf_v2.gpucheck --when prescreen`, 6 s apart; each passes when the median SM clock is at least 97 % of the maximum, every sample's clock-event
  reasons are GPU idle, applications clocks or SW power cap, the enforced power limit is the default, under 64 MiB were in use before and no other process
  used the GPU), all five passing; then the CPU steal at most 1 % over 5 s on the container's CPUs and the container's CPU quota at least 20 CPUs. The pod's CPU
  model, quota, driver, power limit and maximum SM clock and every check's samples are recorded (`prescreen.json`, copied into the session). **There is no
  host-CPU busy check on a pod**: the pod times GPU work only. The timed core (the quietest physical core of the GPU's NUMA node outside core
  0's, by `hostcheck.pick_core`) is a pin, not a gate.
- A pod that fails is released and replaced (section 1.5). One pre-screen per pod, as in stage 1.
- Within every timed process: `speed_rt.watch` pins it to the timed core and judges its clock window: for a GPU process the clock rule of its item ("floor"
  for the continuous Ising timings: every SM-clock sample at least 97 % of the maximum, throttle reasons none or idle; "throttle" for G1, G2, G4 and G6:
  throttle reasons none or idle, the modules also apply the 97 % floor inside their own timed spans) and no other compute process on the GPU. The host-side shares
  (the timed core's steal and busy share, its siblings', the load average) are recorded in every window and decide nothing. A software power cap that the
  workload itself causes is recorded as "workload power cap".
- A process counts only when its window and its own module accept it, and its record names the registered inputs (the network's and the shots' sha256, the
  code digest, the session's CPU model and timed core).

## 1.5 Pods

STAGE2_common C1 holds for the pods: two roles, one pod each, each an H100 80GB HBM3 SXM pod (RunPod, secure cloud), pre-screened by section 1.4, requested when
F_2 is pushed (S-high's earlier, since its plans are built before F_2: the builds' record, below), released as soon as its outputs are pulled, never idle more than 15 minutes; at most five pods are screened per role and the fifth failing leaves the
role descriptive, as C1 states. Nothing else runs on a pod during its session. The
two roles' pods may be different hosts; a comparison never mixes pods on its GPU side. Every pre-screen and release is in the run log and is published.

## 1.6 Replacement and failure: one rule

STAGE2_common C5 holds, with these particulars:
1. A process whose window is rejected (clock rule, another compute application) or which fails (a non-zero exit, a missing or foreign record, a module's own
   check) is replaced at once by a fresh process with the same arguments, at most twice: a slot has at most three attempts, on a pod and on the Linux workstation alike.
2. A slot whose three attempts are all rejected or failed is not established. On a pod, every comparison that uses the slot is withheld (no verdict); on
   the Linux workstation, a CPU slot not established withholds the comparisons that need it; no other process and no earlier value stands in. A reference slot (C-ref) not
   established leaves the stage's fit undefined, and then every Grace-basis CPU time of the stage is withheld, that is every comparison of this family.
3. A pod session that does not reach its end (a pod or process lost, the driver stopping) is a technical failure: the role runs once more as a whole, from its
   first slot, on a newly pre-screened pod, and only the new session's records count; if that session also fails, every comparison of that role is withheld.
   A lost CPU session follows C5.3.
4. Nothing runs again because of a value it produced. Every attempt, accepted or not, every pre-screen and every session is kept and published
   (`attempts.json`, `prescreen/`, the window records).

## 1.7 Statistics

Stage 1's rule for every term: a baseline takes the FASTEST of its three fresh processes and our side the SLOWEST.
- Ours: G1 (the GPU term), G2 (the handoff) and G4 (the Triton engine): the slowest accepted process of the five; our CPU term b (the Linux workstation): the slowest of
  the three repetitions.
- Baselines: BP-OSD (pod), the fastest of three accepted repetitions; Ising's plan time (pod), the fastest accepted candidate by `ising_select.py`; the
  plain and correlated PM baselines (C-base) and the PM shares after Ising (C-share) (the Linux workstation), the fastest of three fresh processes. The earlier draft's
  medians of three are gone.
- The Grace fit's reference points (C-ref): the slowest of three fresh processes per point (item 2). A larger Grace factor makes every converted CPU time
  larger, which favours our pairs against the all-CPU baselines; the slowest reference time gives the smaller factor, so the choice goes against the
  claim. State of the analysis: `analysis_s.cpu_quantities` takes `max` over a point's repetitions, `min` over C-base and C-share, `max` over C-term.
- Every repetition's value and the range are published beside each counted value.

## 1.8 The Grace conversion

The conversion is STAGE2_common C3: the stage's 18 reference points (C-ref: NVIDIA's plain and correlated PyMatching at the points of Tables VII and IX, timed once
on the Linux workstation at the session's concurrency, the slowest of three per point: section 1.7 gives the reason), one factor per decoder over all points, the common-factor rule, the
residuals. Which term takes which factor is fixed here, before any data exist. If the two decoders agree, every CPU time of this family takes the common factor. Otherwise:
- plain-PM terms (f_plain): the plain PM baseline and the PM shares after Ising M1 and M4;
- correlated-PM terms (f_corr): our CPU term b, the correlated PM baseline, the correlated shares after Ising (Ising + correlated PM).
This family has no other CPU decoder. Every CPU time on the Grace basis is the Linux workstation's own raw time x its applied factor; the references run in the same session and at the same
concurrency as the terms they convert, and one fit serves both roles. CUDA-Q's BP-OSD is timed host to host on the pod and is not converted. Published: C3's list (raw time,
converted value, factors, spreads, decision, per point the raw time, ratio and residual) and the per-distance ratios as a diagnostic.
- The d = 17 basis: at d = 17 the Grace basis is the fitted factor of the decoder, the same one as at every other distance (the common
  factor, or f_plain for plain-PM terms and f_corr for correlated-PM terms). There is no separate d = 17 anchor, interpolation or extrapolation of NVIDIA's
  table: correlated PM at d = 17, p = 0.6 % (NVIDIA 860.71 us per shot) is one of the fit's 18 points, and plain PM at d = 17, which NVIDIA does not report, is
  carried by the plain factor. Every d = 17 margin (S3, S4, S7, S8, E8) is stated on this basis; the residual of the d = 17 correlated-PM point is published beside
  them (row S8) and decides nothing.
- The earlier d = 13 anchors, the d = 17 interpolated basis and the plain-share proxy are gone. A Grace-basis number is an estimate; no Grace hardware is used and no claim about
  it is made.
The reference points are this family's `gref.py time` on `srt-ref-*` (one chunk of 10,000 shots per point); they serve every CPU part of the stage, Decoder Switching included, and the fit
file `fit_s2.json` is written by `python -m speed_rt.analysis_surf fit` (C3).

## 1.9 Verdicts and wording

- Every statement is judged from this stage's records only: a comparison at d takes its GPU terms from the pod of d (S-low d = 7, 9,
  11; S-high d = 13, 15, 17) and its CPU terms from the Linux workstation (the CPU session). "X takes less time than Y" holds when X's re-measured value is strictly below Y's on the Grace basis
  (the fitted conversion); no statistical test, as in the paper. A statement over all six distances (S2's range, S4's count and
  fractions, S6's range, S11's range) is made only when both pods and the CPU session established every cell it needs; otherwise it is withheld.
- **Network sets.** Both are reported, each against its own old values: the deployed networks (the paper's rows as they stand) and
  SURF-v2's six (the revision's rows). This re-test does not choose which set the paper prints.
- Every printed number, ratio and range of rows S1-S11, E1, E2 and E8 and of the surface parts of H2, H3 and H4 is replaced by its
  re-measured value, whatever it is. Ranges are recomputed by the paper's arithmetic and rounded in the direction that weakens the
  claim: speed-ups down, times and fractions of time up.

| row | what is re-measured and decided |
|---|---|
| S1 | Fig. 3's composite times: ours (a + b + c) per set and d, and the six baselines and two controls per d, Grace basis; BP-OSD unconverted |
| S2 | the ratios correlated PM / ours per set and d, and the range "LO-HI times less composite time than correlated PM" (LO and HI rounded down) |
| S3 | ours against correlated PM at d = 17: a strict comparison |
| S4 | the 48 speed comparisons per set (six baselines and the two Ising + correlated-PM controls at each of the six distances: 8 x 6; the controls at d = 7-15 are listed as S10 rows and counted here), each strict; the count that holds and the count withheld per set; ours / the fastest other decoder (the six baselines) per d and its range (rounded up) |
| S4x | sensitivity check: the same 48 comparisons per set on the raw basis (the Linux workstation's own time), every CPU part as timed and unconverted, with the count that holds; reported beside the Grace-basis verdicts, which decide |
| S5 | NVIDIA's own Grace values at d = 13, p = 0.6 %: labelled; they are two of the fit's 18 points, not re-measured here |
| S6 | Table V's rows: routed share, network term, routed fallback b (converted), handoff, total, CPU share; the range "the network term takes LO-HI % up to d = 15" (LO down, HI up) |
| S7 | the fastest baseline per d and its margin over ours |
| S8 | d = 17 on the fitted-factor basis of section 1.8 (the earlier separate d = 17 basis is gone): the routed CPU term b, how many of the eight comparisons hold, and the converted correlated PM at d = 9 and 17 beside NVIDIA's measured values (residuals of the fit, deciding nothing); E6's margin is labelled |
| S9 | the measured-CPU disclosure: ours (b raw) against plain PM, both as timed on the Linux workstation, unconverted; descriptive, decides nothing |
| S10 | Ising + correlated PM against ours: strict comparisons, at d = 7-15 post hoc controls, at d = 17 registered opponents (the ten at d = 7-15 are among S4's 48) |
| S11 | a routed shot's correlated PM over the per-block mean (the slowest C-term repetition's conditional mean / the C-base value, the fastest of three); its range |
| E1 | the resident d = 7 kernel's GPU term (deployed d7_b1w64_r740, slowest of five), headline H4's 8.44 us; SURF-v2's d = 7 network beside it |
| E2 | the d = 7 Triton engine (slowest of five) against PM's value at d = 7: the statement that the engine is the slower |
| E8 | the CPU time saved at d = 17 by routing 8 instead of 10 %, on the Grace basis and in the Linux workstation's own time |
| E3, E9 | not re-timed, by the authors' decision: E3, the screening times on MIG slices (the paper drops its number), and E9, the routing rule's frozen speed limit (kept in the paper, labelled as the value the rule was frozen with) |
| E4-E7, E10 | labelled, not re-measured (D7): development measurements or non-time statements or inputs of frozen rules |
| F1 | the Grace fit (one, from the Linux workstation's references): factors, spreads, the decision, every point's raw time, ratio and residual (diagnostic) |

- Wording as STAGE2_common C7: a comparison that fails is written "At d = D, OURS took X against BASELINE's Y (us per shot; GPU terms from pod POD, CPU terms from
  the Linux workstation on the Grace basis); the statement does not hold."; a missing result "ITEM was not established on POD (or on the Linux workstation) in SPEED-RT (REASON); the
  comparisons that need it are withheld, and no earlier value stands in for it."
- Descriptive, deciding nothing: the measured-CPU disclosure (S9), the raw-basis verdicts (S4x), the per-point ratios of the fit, the router's numbers.
- No deployment-latency claim is made from these numbers.

## 1.10 What each time includes

Composite time (ours): the H100 GPU term of the network (input already on the GPU) + the handoff (host-side routing) + the correlated PyMatching time of
the routed shots on one core of the Linux workstation's Ryzen 9 9950X, converted to the Grace basis by the fitted factor. The CPU-GPU transfer over NVLink is neglected, as
NVIDIA's Ising paper does; the handoff term stays. Baselines: plain and correlated PM = the Linux workstation's time per block (decode only, 10,000 timing shots)
x the factor; Ising = its TensorRT plan on the H100 (batch 1, CUDA graph, input on the GPU) + its PM share on the residual detectors, converted; BP-OSD =
CUDA-Q's batch-1 host-to-host call on the pod with the pod's host CPU, unconverted. No surface number is end to end, and none includes the detector transfer
into the GPU or the logical-observable return. The GPU terms come from the role's pod, the CPU terms from the Linux workstation; every raw time is published beside its
Grace-basis value.
The handoff term departs from STAGE2_common C4, which splits a transfer off: the surface family keeps G2 as measured: the d = 7 kernel's host round trip minus its device time, and max(1.1 us, the router's estimate) at d = 9-17. Whatever share of a CPU-GPU transfer G2 holds stays in our composites; it can only lengthen them..

## 1.11 The analysis

`python -m speed_rt.analysis_surf --records RECORDS --out OUT` (RECORDS laid out by `speed_rt/pod/collect_s.sh`: `low/`, `high/`, `cpu/`, `testsets/`) reads the
two pod sessions and the CPU session's session, checks them against the manifests, `circuits/ref_circuits.json` and `freeze_surf.json` (the surface part of the
earlier single freeze file, with the CPU recipe replaced; every network's sha256 and tau, the code digest, every mask's shots digest, every CPU record's
circuit, shots, code digest and CPU model), re-derives the quantities, fits the factors with `grace_fit.fit` from the C-ref records, and writes `analysis.json`
(every quantity with every repetition and every reason a quantity is missing, the fit point by point, the sha256 of every input), `table.md` and `rows.csv`. A pod
session establishes nothing when it did not reach its end (no `DONE.json`), when its pre-screen record (copied into the session) is missing or did not pass,
when it was a dry run or ran from a tree that is not a commit; the two pods must have run the same tree. The CPU session's session (`cpu_session.json`: done,
its CPU model; `cpu_attempts.json`: every slot's attempts and acceptance) establishes nothing unless it reached its end; a CPU slot counts only through an
accepted attempt, and every CPU record must name this stage's circuit, shots and code digest and be pinned. The analysis is deterministic; it runs twice and the
two outputs must be byte-identical. The records, the outputs and the run log are committed and pushed.

## 1.12 Order of events

1. F_2 is pushed (STAGE2_common C8). It holds this section, the code (including the twelve reference circuits, `freeze_surf.json` and `sets_surf.json`) and everything shared that
   it imports (the families' single F_2).
2. After F_2 is on GitHub: the stage's beacon record (T_k from GitHub's activity list, the round at least 600 s later, R_k from two agreeing endpoints;
   `speed_rt.beacon fetch --sets speed_rt/sets_s2.json` from F_2's own tree: the stage's one record, STAGE2_common C8, whose sets file holds the sets of
   `speed_rt/sets_surf.json` unchanged), then the draw from the deployed F_2 tree (`alpine/draw_s.sh`; the six timing sets
   and the twelve reference sets, 132 chunk files, every seed recomputed from the record by the draw's own check), and commit B_2 (the beacon record
   `testsets/beacon_surf.json`, `testsets/srt-surf.json`, `testsets/srt-ref.json`).
3. When F_2 is pushed the S-low pod is requested and pre-screened, set up, and its Ising plans built and checked (`pod_s_real.sh`; validation shots only);
   the S-high pod does the same before F_2, from files byte-identical to F_2's (the builds' record, below):
   no timing shot exists then and nothing is timed. A pod that fails its pre-screen is released and replaced (section 1.5).
4. After B_2 is public: each pod receives its role's timing chunks (every chunk checked by sha256 against the manifest), B_2's tree
   replaces F_2's (the code files are equal; B_2 adds the manifests), the pod builds the view, computes the masks (untimed GPU pass), hands them off, and runs
   its GPU session (`ROLE=low|high pod/run_s.sh shots prep packm timed handoff residuals pack`). The two sessions run in parallel, one on each pod. Every
   session's start (pod, GPU, CPU model, quota, time) is logged before it starts.
5. Hand-off part 1 (the masks) goes to the CPU session as soon as `packm` is done; part 2 (the counted plans' residuals) when the session ends (section 1.14).
   The CPU session times the CPU parts on the Linux workstation while the GPU sessions run.
6. Each pod's records are copied to Alpine (sha256-checked), the CPU session's records beside them, and the analysis runs there, twice, from the deployed B_2
   tree; its outputs are committed and pushed with the records.
7. Expected GPU wall times per pod: section 1.16.

## 1.13 Ising (G5): plans, checks, timing, the counted plan and the residuals

Ising is NVIDIA's pre-decoder (models M1 "Fast" and M4 "Accurate"), the baseline of every surface comparison "ours against
Ising M + PyMatching". Ising's time at (model, d) = the GPU time of the counted plan + the PyMatching share timed on that same
plan's residual detectors (Grace basis, as the other surface terms). This section freezes the GPU part (and is part of the stage's registration, not a separate one): what is built, checked
and timed on each pod, every setting, the counted plan's rule, the residuals, the statistics and the failure handling.

### 1.13.1 Where it runs
- One pod per distance group, the GPU side of every comparison on the same pod and session (the PM shares are timed on the Linux workstation, section 1.14): pod S-low d = 7, 9, 11; pod
  S-high d = 13, 15, 17. Every Ising command takes its pod's explicit list (`--d 7 9 11` or `--d 13 15 17`); a list is
  validated (only 7, 9, 11, 13, 15, 17; no repeats), sorted ascending, and built in that order. A plan never moves between
  pods. The two pods have separate work directories, so separate timing caches (below).
- Never a timing shot before the timed block: the ONNX export, the FP8 calibration, the builds and the checks read only the
  studies' validation chunk 0 (10,000 shots per d); the first shot of the timing set's chunk 0 is the only timing-set input
  that the plans see before the timed block, and it is written to a file by `speed_rt.ising_d5 input` after the beacon.

### 1.13.2 What is built (12 plans per d)
| kind | plans | profile | workspace | precision | optimisation level | verbosity | timing cache | calibration |
|---|---|---|---|---|---|---|---|---|
| registered (`timing.ising_trt build`) | 8 per d: M1, M4 x FP16, FP8 x level 3, 5 | fixed (1, m) | 4 GiB | FP16: `BuilderFlag.FP16` on the FP32 ONNX; FP8: STRONGLY_TYPED network of the Q/DQ ONNX | 3 and 5 | DETAILED | ONE persistent file `WORK/timing.cache` per pod: empty on a fresh pod, loaded before and saved after every registered build, `ignore_mismatch` False; shared by all registered plans of the pod in the build order below | FP8: validation chunk 0 shots 0-255 (256 rows; sha256 in `*_prep.json`) |
| official (`speed_rt.ising_d5 official`) | 4 per d: M1, M4 x FP16, FP8 | min (1, m), opt (2048, m), max (2048, m); 2048 = NVIDIA's public test batch | TensorRT's default memory-pool limit (no limit set; the value the API reports, `workspace_bytes_default`, is in every build record) | as NVIDIA's code: EXPLICIT_BATCH; FP16 plan: `BuilderFlag.FP16`; FP8 plan: STRONGLY_TYPED, no builder precision flag | TensorRT's default (3; recorded as `builder_optimization_level_default`) | DETAILED (the one recorded departure; it keeps layer and tactic names and does not change tactic choice) | none: no cache object is created, loaded, saved or passed; every plan has its own builder and builder config, so nothing carries over from one build to the next (the record keeps `timing_cache_object` and `disable_timing_cache_flag`) | same ONNX and calibration rows as the registered plans |
- One ONNX per (model, d), shared by both kinds: opset 18, dynamic batch, constant folding, dynamo off, traced on random bits of
  seed 0 as NVIDIA's export does; FP8 by ModelOpt 0.47.0 (`op_types_to_quantize` ["Conv"], `high_precision_dtype` "fp16").
  No ReLU substitution, no sparsity, no INT8. TensorRT 10.14.1.48 (pip wheel `tensorrt_cu12` 10.14.1.48.post1); trtexec 10.14.1
  build 48 from NVIDIA's apt repository, every TensorRT package pinned to 10.14.1.48-1+cuda12.9.
- Build count and order, per pod: `prep` (ONNX and FP8 Q/DQ) for every (d, model), d ascending, M1 then M4; then the registered
  plans, d ascending, per d M1 FP16 L3, M1 FP16 L5, M1 FP8 L3, M1 FP8 L5, M4 FP16 L3, M4 FP16 L5, M4 FP8 L3, M4 FP8 L5; then
  the official plans, d ascending, per d M1 FP16, M1 FP8, M4 FP16, M4 FP8. Each plan is built once. A plan that fails to build
  (or whose FP8 quantization failed) is recorded and never timed; the other plans go on.

### 1.13.3 Checks (before the timed block; a plan that fails is never timed)
- Every plan of every kind is checked once on the studies' validation chunk 0 (10,000 shots; `timing.ising_trt check` for the
  registered plans, `speed_rt.ising_d5 check` for the official ones; no timing shot) against the official PyTorch numerics
  (FP32 weights, TF32 as NVIDIA's runner enables it, followed by plain PyMatching): decision agreement at least 99.9 % (FP16)
  or 99 % (FP8); the paired test against the FP32 reference not "worse"; the residual-detector density within 5 % of the
  reference; for an FP8 plan, every convolution that the recipe quantized runs an FP8 (e4m3) tactic (the engine inspector's
  layer table); and the plan file is the one its build recorded (same sha256). The check record binds the plan's sha256.
- Both checks exit 1 when any plan fails or is missing (the session driver records "some plans FAIL" and goes on). Timing
  re-reads the record: `ising_time` and `ising_d5 trtexec` run a plan only when its check says PASS (FP8: also "FP8 execution
  confirmed") and the plan file's sha256 is the checked one; otherwise the row says why ("check FAIL", "check unchecked",
  "FP8 execution not confirmed", "the plan is not the checked one") and the other plans go on.
- Dry results (validation chunk 0): section 1.15.

### 1.13.4 Timing (on the pod, inside the timed block, every process pinned to the session's timed core; windows judged by the rules of `speed_rt/watch.py`, section 1.4)
- **Graph timing** (`speed_rt.ising_time`, one process per d and order; round 1 forward, round 2 reversed): every validated plan of
  d, in PLANS order (M1 then M4; FP16 then FP8; level 3, level 5, official; reversed in round 2), at batch 1 on the first shot of
  the timing set's chunk 0, inside a CUDA graph: 200 warm-up replays, then five repetitions of 100 replays between CUDA events
  (`graph_reps`, unchanged). Each plan's span is judged on the nvidia-smi samples inside it and the nearest sample on each side:
  every SM clock at least 97 % of the maximum, throttle reasons none or idle; no other compute process on the GPU (the host-side shares of the timed core and its siblings
  are recorded over the whole process and decide nothing: the pods time GPU work only). A plan whose window fails has
  `accepted` false and its values do not count; a process counts only when every plan it timed is accepted.
- **NVIDIA's command** (`speed_rt.ising_d5 trtexec`): one repetition per round r = 1-5 of every validated plan:
  `trtexec --loadEngine=PLAN --useCudaGraph --noDataTransfers --useSpinWait --loadInputs=dets:INPUT` (INPUT = the first shot of the
  timing set's chunk 0, one uint8 row of m bytes, written by `ising_d5 input`; never trtexec's random inputs), plus
  `--shapes=dets:1xm` for an official (dynamic) plan, and trtexec's own defaults otherwise: one inference stream, `--warmUp` 200 ms,
  `--duration` 3 s, `--iterations` 10 (the record keeps trtexec's printed options: Iterations 10, Duration 3s (+ 200ms warm up),
  Inference Streams 1, Data transfers Disabled, Spin-wait Enabled, CUDA Graph Enabled, Multithreading Disabled, and the input
  file it loaded). trtexec 10.14.1 build 48 (TensorRT v101401), the plans' own TensorRT version. A repetition's value is the
  "GPU Compute Time" mean it reports (never the minimum single inference); the window (rule floor) starts at trtexec's
  "Starting inference" line and covers its warm-up and its 3 s.
- Rounds follow the session's order (G1, G4, then G5's graph timing in rounds 1-2, then G5's trtexec repetition r of every
  validated plan of the role's distances); a rejected or failed process is replaced at once, at most twice per slot.

### 1.13.5 The counted plan (`speed_rt.ising_select`, shared by the pod and the analysis)
- Candidates: every accepted repetition (five per graph process, fwd and rev; one per trtexec repetition) of every validated
  plan in the session's counted attempts (`attempts.json`: accepted by the watch and the module; a replaced attempt never counts).
- Ising's GPU time per model and d = the smallest candidate value (the studies' "fastest repetition of the fastest validated
  plan"); the counted plan is that candidate's plan, whichever kind it is (registered L3, L5, or official; graph or trtexec).
- Ties go to the first plan in PLANS order (model, FP16 before FP8, level 3, level 5, official), then graph before trtexec, then
  forward before reversed, then the earlier repetition.
- A (model, d) with no accepted candidate has no Ising time: every comparison that needs it is withheld. `ising_select --d LIST`
  names the pod's distances and lists the missing (model, d) pairs.
- Every repetition's value, the range and the counted candidate's source are published beside the counted value.

### 1.13.6 Residuals
- The PyMatching shares (P3) are timed by the CPU session, on the Linux workstation, on the residual detectors that the counted plan itself produces on the timing set's
  chunk 0 (`ising_d5 residuals`, batch 1, plan checked and sha256-matched first): `res_d{d}_{M}_val000.npz` (pre_L, res, obs) and a
  record with the plan, its sha256 and the content digests. Ising's GPU time and its CPU share come from the same plan. A row
  that pairs one plan's time with another's accuracy is labelled "an envelope that favours the baseline", not one decoder.
- The files are made on the pod that counted the plan, from the plan on that pod, after the timed block (an untimed GPU pass: hand-off part 2, section 1.14).

### 1.13.7 Failure handling
- A plan that fails to build, fails its check or its FP8 audit is never timed and never counted; a (model, d) left without a
  counted plan has no Ising time and its comparisons are withheld. Nothing is rebuilt, re-checked or re-tried because of a result.
- A trtexec or graph process that fails or is rejected is replaced at once, at most twice per slot (the session rule). A slot
  still rejected is not established; the other plans' values stand.
- Builds are not repeated for a plan that built: a record with the same name makes the command skip it, so a rerun of the
  build step after a lost pod starts the role from scratch on a new pod (new work directory, new timing cache).
- The builds' record: after the checks the build step writes `BUILD_INPUTS.json` (`speed_rt/build_inputs.py`): the sha256 of every file of the
  deployed tree the builds read (the import closure of `surface_v2/timing/ising_trt.py` and `speed_rt/ising_d5.py`, `speed_rt/pod/run_s.sh`,
  `speed_rt/build_inputs.py`, NVIDIA's legacy circuits and the studies' test-eval tables), of NVIDIA's code and checkpoints (the SUMS file), of the
  validation chunks 0 and of every file built, and trtexec's version. The session copies it into its records (`session/BUILD_INPUTS.json`), and the
  analysis checks every listed tree file against its own tree, which is F_2's: without the record, or with any file different, the role's session
  does not qualify (the role runs again on a new pod, by the session rule).
- S-high's plans are built before F_2. Its pod is requested so that its 45 minutes of builds end about when B_2 is due, and it is set up from a
  commit that holds the files F_2 freezes. The builds compile the frozen networks on the validation chunks 0 and time nothing, and no timing shot
  exists before B_2, so building early changes nothing the stage's order protects. The builds' record binds the plans to F_2's files; if a file
  it lists changes before F_2, the plans are built again from F_2's tree. S-low's plans are built after F_2, from F_2's tree.
- If B_2 is not public within 15 minutes after S-high's builds end, the pod does not wait: every build output (ONNX files, plans,
  timing caches, the prep, build and check records, `BUILD_INPUTS.json`) is saved off the pod with its sha256 and the pod is released.
  About 15 minutes before B_2 a new S-high pod (the same GPU model and the same TensorRT, CUDA and torch versions) is set up from F_2,
  loads the saved outputs after checking their sha256, and runs the build step: it skips every plan whose record and file exist and
  re-runs every check on the timing pod, so a plan that does not load there fails its check and is rebuilt on that pod and recorded.
  The plans are then built before F_2 from F_2's bytes on an identical H100 and loaded on the timing pod, with their provenance by
  sha256; the first pod's `BUILD_INPUTS.json` is kept beside the timing pod's.

### 1.13.8 Durations
See section 1.16 (the dry runs' durations, in place of estimates).

## 1.14 The hand-off to the Linux workstation

Everything the CPU session needs from the pods is produced by two untimed GPU passes (G3 and the Ising residuals), pulled by `tools/surface/pull_handoff.sh` (pod
to box, every file's sha256 checked) into `~/work/s2/handoff/surface/<role>/` (masks) and `.../handoff/ising/<role>/` (residuals), each part with a
`SHA256SUMS` over its files and a `DONE` file when both roles are in. The pod writes them with `speed_rt.pod_s packm` (right after the masks) and `pack` (after
the residuals).
- Masks, one file per network and role: `val_<network>.npz`, networks `d{d}_...` of `freeze_surf.json` (twelve, plus `val_d17_b5_r787_q10.npz` at S-high);
  arrays `c` (float32, the network's logit per timing shot, 200,000), `routed` (bool, 200,000: |c| < tau), `tau`, `network`, `d`, `shots_sha256` (the digest of
  the timing set at d, equal to `srt-surf.json`'s), `kind`. `HANDOFF_MASKS.json`: per file its sha256, d, network, tau, routed count and shots digest.
- Residuals: `res_d{d}_{M}_val000.npz` (arrays `pre_L`, `res`, `obs`: the residual detectors of chunk 0 after the counted plan of model M = M1 or M4) and its
  `.json` record (the plan's name and sha256, the content digests), `selected.json` (the counted plan, its value and source per model and d), `RES.sha256`,
  `HANDOFF_RES.json`. The residuals exist only for (model, d) pairs that have a counted plan; the CPU session times a C-share slot only for those.
- Timing shots and reference shots are not on the pods' path: the CPU session takes the draw's chunk files from the draw directory (`d{d}_p0.006_X_srt_NNN.npz`, 20
  per d; `ref_d{d}_p{3|6}_000.npz`, 12) and the manifests `testsets/srt-surf.json`, `srt-ref.json` (B_2).
The CPU session runs this stage's code from the B_2 tree, on Python 3.12.3 (Ubuntu 24.04's CPython, copied into the tmpfs), numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2, ldpc 2.4.1, stimbposd 0.2.0, sinter 1.16.0 and beliefmatching 0.2.0, the versions of winserver's environment (no torch), with
`PYTHONPATH=<tree>/surface_v2:<tree>`; VIEW = `python -m speed_rt.draw view --shots <draw>/shots --out VIEW --ds 7,9,11,13,15,17` (links named as the validation
chunks); each process is one fresh process pinned to one core, writes exactly one JSON record into its own output directory, and is one repetition (rep = 1..3):

| item | slot key | command (one process) |
|---|---|---|
| C-ref | `<dec>_d<d>_p<3\|6>` (`dec` = plain or corr; the 18 points of `grace_fit.points()`) | `python -m speed_rt.gref time --dec DEC --d D --p P (0.003 or 0.006) --shots <draw>/refshots --manifest <B_2>/speed_rt/testsets/srt-ref.json --rep R --core C --out OUT` |
| C-base | `<corr\|plain>_d<d>` (the role's distances) | `python -m timing.cpu_term ref --d D --val VIEW --rep R --core C --out OUT [--plain]` |
| C-term | the network name (and `d17_b5_r787_q10`) | `python -m timing.cpu_term term --d D --val VIEW --mask <handoff>/surface/<role>/val_<network>.npz --rep R --core C --out OUT` |
| C-share | `<corr\|plain>_<M1\|M4>_d<d>` | `python -m timing.cpu_term share --d D --model M --res <handoff>/ising/<role>/ --rep R --core C --out OUT [--plain]` |

Returned to the analysis (`records/cpu/`, which `speed_rt/cpu_ws/export.py` writes from the session's run directory, STAGE2_common C1): `cpu_session.json` (`cpu`: the CPU model as the records name it, `done`, `dry`, the machine's record), `cpu_attempts.json` (a
list as the pods' `attempts.json`: item, key, round = repetition, attempt, accepted, record = the record's path relative to `cpu/`) and `cpu/<slot-directory>/<record>.json`.
The records carry their own `cpu`, `pinned`, `code_digest`, circuit and shots digests, which the analysis checks. C-term's `shots_sha256` and mask digest must equal
the hand-off's, and C-share's `residuals_content_sha256` the counted plan's.

## 1.15 Departures

- The pods time GPU work only: the plain and correlated PM baselines, the shares after Ising, our CPU term and the Grace
  references are timed on the Linux workstation by the CPU session, from the hand-off of section 1.14; the earlier Alpine EPYC 7713 and the pods' Xeon are no longer used.
- The Grace conversion is one fitted factor per reference decoder over all of NVIDIA's points, fitted from the Linux workstation's references and applied to every
  CPU time, replacing the d = 13 anchors and the d = 17 interpolated basis; its reference circuits for d = 5, 21, 31 and for p = 0.3 % are built from NVIDIA's code.
- The pods use stage 1's pre-screen (five GPU checks, steal, quota) and have no host-CPU busy check; the host-side shares are recorded in every window.
- Two pods (S-low and S-high) replace the single session S; no comparison's GPU side spans two pods. The composite sums GPU terms from a pod and CPU terms from
  the Linux workstation, as the paper's composite sums terms.
- The failure rule has no choice left (finding 6): replacement at most twice per slot, a slot not established withholds, a session that does not reach its end
  runs once more as a whole on a new pod.
- Statistics: baselines and PM shares the fastest of three, our side the slowest, the Grace reference points the slowest of three
  (the draft's medians are gone); the speed comparisons are 48 per set (8 opponents x 6 distances), the raw-basis verdicts are kept as a sensitivity check
  (rows S4x); the d = 17 basis is the fitted factor (section 1.8).
- The Ising plans' `trtexec` and graph timing are those of the Ising helper's modules; the shares after Ising are timed on the residuals of the counted plan.
- Each pod holds only its role's distances; the surface timing modules read the shots through the view, unchanged.

## 1.16 Dry runs and durations

The dry runs exercised every GPU step of the registered session on the reworked code (13 October, pods p17 = S-high, p21 = S-low; same code path as the
registered session, the dry draw's shots, one round, a failure injected into one G1 slot (one process killed) and one G2 slot that is never established).

| run | pod | pre-screen | prep (G3 masks, inputs) | timed (G1, G4, G5, G6) | G2 | residuals + pack | total |
|---|---|---|---|---|---|---|---|
| S-low, full size (200,000 timing shots, 200 BP-OSD calls), 1 round | p21 | 66 s, PASSED | 165 s | 382 s | 64 s | 20 s | 11.6 min |
| S-high, 20,000 timing shots, 1 round | p17 | 65 s, PASSED | 209 s | 603 s | 51 s | 20 s | 15.8 min |
| S-high, full size, 1 round (before the CPU parts left the pods; no pre-screen) | p17 | - | 202 s | 667 s | 182 s | 21 s | 18 min |

Per-slot durations (one process, launch included; full size): S-low G1 18.0 s, G2 8.0 s, G4 7.0 s, G5 graph 3.7 s, G5 trtexec 5.7 s, G6 5.3 s; S-high G1 15.3 s,
G2 36.4 s, G5 graph 3.3 s, G5 trtexec 5.5 s, G6 61.0 s. Plan builds and checks (validation shots only, before B_2): S-low 20.3 min (p21), S-high about 45 min (p17: prep
446 s, registered plans 455 s, official plans 1720 s, checks 87 s). The pre-screen of both pods passed on its first try (five GPU checks at 1980 of 1980 MHz,
no throttle reason, steal 0.000 %, quota 22.1 CPUs on p17). The injected failures took the failure rule through every branch: the killed process was replaced
and accepted, the never-established slot was recorded as such (4 of 61 S-low attempts injected; 57 accepted).

Expected wall times of the registered sessions, from these per-slot durations (5 rounds, 3 BP-OSD repetitions; replacements add at most 10 %, an estimate that the dry runs could not give):
S-low 35 min (pre-screen and prep 4 min, timed 29 min, G2 2 min), S-high 51 min (pre-screen and prep 4.5 min, timed 43 min, G2 3.6 min). Before B_2, each pod needs its
plan builds: S-low 20 min from F_2's tree, after about 10 min of setup; S-high 45 min, run before F_2 (the builds' record).

The CPU items on winserver (Xeon Gold 6326), dry run of 13 October: dry shots, synthetic routing masks (they route the heaviest shots, so the C-term durations are upper bounds), one process per slot, 8 at once, in windows
of 8 together with the stage's other CPU parts (10 windows in all), one repetition. Values marked * are extrapolated from the dry run's per-shot time to the registered shot count. C-base (12 processes: plain and correlated PM per block,
d = 7 to 17, 10,000 shots): 0.8 s (plain, d = 7) to 13.2 s (correlated, d = 17), 45 s per repetition. C-term (13 networks, 200,000 shots): 0.9 s (tau 0) to 34.6 s (d = 17, q = 8 %), 104 s per repetition. C-ref (18 points, 10,000 shots): 0.7 s (correlated,
d = 5) to 228 s* (correlated, d = 31, p = 0.6 %, 22.6 ms per shot), about 480 s* per repetition. Per-shot times of the dry run, d = 7, 9, 11, 13, 15, 17: plain PM per block 19.99, 41.65, 84.00, 135.98, 237.63, 399.95 us; correlated PM 47.46, 101.45, 224.5, 418.43,
743.69, 1,185.2 us. The reference points (per shot, us; p = 0.3 % / 0.6 %): plain d = 13 58.09 / 151.78, d = 21 313.9 / 950.6, d = 31 1,641 / 4,659; correlated d = 5 10.54 / 14.37, d = 9 51.73 / 100.92, d = 13 144.98 / 407.46, d = 17 425.27 / 1,088.4,
d = 21 1,145.6 / 3,176.8, d = 31 8,771 / 22,561. The C-share items were not part of that dry run. Every process of these parts exited 0; no window passed the reference check (the host's clocks varied between 1.0 and 2.6 GHz, and the spread
of the reference segment over the 8 slots was 20 to 45 %, against a limit of 3 % then and 6 % now, STAGE2_common C1). By winserver's dry-run times, a whole session of three repetitions of every CPU part of the stage is about 6 to 8 core-hours, 50 to 70 minutes on 8 slots when packed longest first.

The hand-off, tried end to end on the dry records (`tools/surface/pull_handoff.sh`, sha256-checked on arrival; S-high: 207 MB for masks and residuals): the CPU session's four
commands of section 1.14 ran on it with torch, triton, TensorRT, CUDA-Q and ONNX blocked (an ImportError stub ahead of PYTHONPATH) at every distance of both roles,
all exit 0. The analysis ran twice on the dry records of both pods plus a stand-in for the CPU session's records (479 rows, 99 comparisons), outputs byte-identical (section 1.11), and the fit file made by the stand-alone command equals the analysis' own. The draw script re-ran as a dry draw on a CPU node (identical chunk digests; its log, manifests and SHA256SUMS hold no absolute path), and the collection step replaced every pod path by a token before hashing.
