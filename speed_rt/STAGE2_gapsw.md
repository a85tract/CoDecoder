# SPEED-RT stage 2, Decoder Switching: gap switching timed on the Linux workstation (STAGE2_gapsw)

Its code is `speed_rt/gapsw/`, its sets file `speed_rt/gapsw/sets_gs.json`. It is part
of freeze F_2 with `STAGE2_common.md` (the common rules, cited as C1-C9), `STAGE2_surface.md` and `STAGE2_bb.md`, and it changes no file that
stage 1 froze (`SPEED_RT.md`, `sets.py`, `beacon.py`, `cascade/*`) and never changes `surf_v2/gapsw.py`. Sections follow `SPEED_RT.md`
section 1. Every CPU part of this family runs on the Linux workstation; no pod is used and no hand-off is needed (C2).

## 1.1 What is timed

Decoder Switching as the paper's Fig. 3 shows it: our re-implementation of gap switching (`surface_v2/surf_v2/gapsw.py`, sha256
`7c652b0c966af77911b20acb61a940d00780e8d5eb329e01e9f658aa6e3b1f94`, at the tau frozen in `surf_v2/gapsw/policy_d<d>.json`,
`GAPSW_REGISTRATION.md`) on the verdict circuits of SURF-v2 (rotated surface code, X-basis memory, d rounds, NVIDIA's legacy single-parameter
noise at p = 0.6 %), d = 7, 9, 11, 13, 15, 17. Per d two terms, each a fresh single-threaded process pinned to one core of the Linux workstation, on the
50,000 timing shots of the d (section 1.3):

- FE (the front end): the two matchings per shot, one shot at a time (`gapsw.loop`: plain PM's decode with its weight, then the split graph's
  decode in the other logical class; 50 untimed warm-up shots, `perf_counter` around each shot); the term is the mean over the 50,000 shots,
  in us per shot.
- ROUTED: correlated PM (`enable_correlations=True`, the loop of `timing.cpu_term.loop`) on the shots whose gap is below the frozen tau, one
  shot at a time; the term is the sum of the routed shots' times over all 50,000 shots, in us per shot.
- Gap switching's composite is the fastest of three repetitions of FE plus the fastest of three of ROUTED, each on the Grace basis (C3): Decoder Switching is a
  baseline of the paper's Fig. 3, so C6's rule gives it the fastest of its three (the slowest of three that `GAPSW_REGISTRATION.md` took stays a sensitivity row, 1.7).
  No handoff and no GPU: the whole pair runs on the CPU, so C4's composite is its two CPU terms.

Per repetition 12 slots (6 FE, 6 ROUTED); 36 in all. The Grace reference points that convert them are the stage's one set (C3: the surface family's `srt-ref-*` points, timed in the same
CPU session at the same concurrency); this family has none of its own. SURF-v2's analysis printed, on an exclusive EPYC
7713 node (each term converted by G(d) over that node's correlated-PM time at d), composites of 74.69, 146.11, 375.88, 723.76, 1,359.85 and
2,709.58 us at d = 7-17 (`gapsw/analysis/gapsw_times.csv`); this stage replaces them. Durations, measured on a Xeon Platinum 8470 pod (full
size then, 200,000 shots per d, one repetition): FE at d = 7, 9, 11, 13, 15, 17: 24, 49, 94, 190, 364, 666 s; ROUTED: 10, 12, 17, 21, 31, 74 s; in all about 4,650 s of
one core for the three repetitions. With eight processes at once the family takes at least the d = 17 front end's duration: 666 s on the pod, estimated at 900 s on winserver's Gold 6326 (the pod's duration times 1.3;
winserver's dry run did not measure the front ends, section 1.13), so 15 minutes or more there (estimates made for winserver, not rules). On winserver's Gold 6326 the estimate is 22 to 900 s per front end and 15 to 90 s per routed term, about
2,050 s of one core per repetition (front ends about 1,800 s, routed terms about 250 s), all at 200,000 shots. At the stage's 50,000 shots (1.3) each is about a
quarter: 6 to 225 s per front end and 4 to 23 s per routed term, about 510 s of one core per repetition; the d = 17 front end, about 225 s, is the family's longest process.

## 1.2 Code

- `surface_v2/surf_v2/gapsw.py` (never changed; `gs_time.py` refuses any other sha256), `surface_v2/convlines`, `surface_v2/timing/cpu_term.py` and `timing/deployed.py` (released code,
  read only), `speed_rt/grace_fit.py` (the lead's fit, read only).
- The family's own files (`speed_rt/gapsw/`): `sets_gs.json` (the six timing sets), `draw_gs.py` (the draw), `gs_time.py` (the processes: fe, term, mask), `hostwin.py` (the process's own
  host checks, section 1.4), `gslib.py`, `analysis_gs.py`, `pack.sh` (the deployable tree with its SUMS). The timed processes are run by `speed_rt/cpu_ws/` (C1): the family supplies the code
  of every decoder, `cpu_ws` the machinery (`plans.py` parts `gs_fe` and `gs_term`, which run `python -m speed_rt.gapsw.gs_time fe` and `term`). `drive_gs.py`, `drive_par.py`, `pod_g.sh` and `alpine_dry.sh` served the dry runs and are not the timed path. The family's
  own reference circuits, `refs.json`, `make_ref_circuits.py` and the `ref` mode of `gs_time.py` are gone: the stage's one Grace fit is the surface family's (C3).
- Environment on the Linux workstation: C1's (Python 3.12.3 (Ubuntu 24.04's CPython, copied into the tmpfs), numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2, ldpc 2.4.1, stimbposd 0.2.0, sinter 1.16.0 and beliefmatching 0.2.0, the versions of winserver's environment; no torch and no GPU library, because nothing here uses them); the draw runs in the studies'
  environment on Alpine (Python 3.12.14, the same three pins). `OMP_NUM_THREADS=1` and the other thread variables are 1.
- The deployed tree is `pack.sh`'s (every file with its sha256 in SUMS, `DEPLOYED` = F_2 or B_2); the CPU session checks SUMS before anything runs.
## 1.3 The timing shots

- Sets: `speed_rt/gapsw/sets_gs.json`: `srt-gs-d<d>` for d = 7, 9, 11, 13, 15, 17, each 5 chunks of 10,000 shots (50,000 shots) of the verdict circuit. Seeds by `speed_rt/sets.py` with the stage's sets file
  (`speed_rt/sets_s2.json`) from B_2's record; files in SURF-v2's format (`convlines.testset`), drawn once on Alpine (`draw_gs.py`: x86-64 with AVX2, Stim 1.16.0, numpy 2.2.6) with a
  manifest of every chunk's seed and sha256 and every set's digest. The CPU session checks every chunk against the manifest before use and every timed process checks its set's digest; the
  shots reach the Linux workstation by C2's route.
- Counts. The count was cut before F_2 so that the CPU session fits one evening (the authors' decision): 200,000 -> 50,000 shots per d, 5 chunks
  instead of 20. Every term uses all of its set's shots, chunk 0 to 4 in shot order; ROUTED, the routed shots among them.
- Own sets. The comparison is descriptive and pairs nothing: no test here uses two decoders' decisions on the same shots, and gap switching's
  times do not depend on the ConvLines networks' routing. The terms are means over 50,000 shots (relative standard error of the per-shot
  mean: 0.12 to 0.22 % for the front end, d = 7 to 17, and 0.7 to 1.1 % for the routed term, twice the 0.06 to 0.11 % and 0.35 to 0.55 % that
  the full-size dry session measured over 200,000 shots, as the error scales with one over the square root of the count), far below the
  6 % spread within a segment that C1's windows allow.
- The routed shots are fixed before the first timed window of the family: `gs_time.py mask` decodes the d's 50,000 shots with the front end in
  parallel shares (batch decode, the same functions) and keeps the gap, the weight, the answer and `routed = gap < tau`. It is untimed and
  all CPU, so it needs no hand-off. Each timed FE process checks its own loop's answers, weights and gaps bit for bit against the mask's and
  fails when they differ; each ROUTED process takes its shots from the mask.

## 1.4 The window and acceptance

A timed process is one job of a window of C1. Its window runs from before its 50 warm-up shots to its last shot (the shortest are ROUTED at d = 7, about 1.5 s), and is accepted under C1:
tmpfs reads only, no block read, the reference segments before and after the window of 8 processes within 1.10 times the session's reference with a spread of at most 1.06 within a segment, the frequency
recorded. Each process also keeps its own host checks (`hostwin.py`, from `/proc/stat` and its own CPU time inside the process, so that its record holds the raw counters and the analysis
re-derives every verdict), beside the harness's checks of C1:
- idle and window: the timed core's busy share by others (its busy ticks minus the process's own utime + stime, over the core's ticks) at most 5 % (plus two ticks of 10 ms) and its steal share
  at most 1 %, over the 5 s before the window and over the window itself;
- probe: plain PM per block (d = 9, validation chunk 0, 10,000 shots; one untimed pass, then three timed passes in the process) just before the window: every pass at most 1.10 times the
  session's plain-PM reference (`--probe-ref`, the harness's session reference) and the largest pass over the smallest at most 1.06, the limit of C1's segments
  (`hostwin.py` LIMITS probe_spread 0.06). The dry runs used 1.03, a limit set on Alpine and the pods; on winserver with turbo on (8 windows, 13 October) it rejected
  the process in 6 of the 8 windows, while C1's reference segments of the same windows spread 1.013 to 1.053.
A process that fails one of them exits 4 (rejected, nothing wrong with the process); the harness runs it again in an extra window (C1), within C5's limit of three attempts in all. A process is counted when it is accepted, `gapsw.py` is the frozen one, its
answers equal the mask's and its shots are the draw's. It fails (not rejected) on a check of its own: a digest, the mask, the code sha256.
## 1.5 The machine and the schedule

The Linux workstation as C1 states it (AMD Ryzen 9 9950X, native Ubuntu 24.04, tmpfs, at most 8 timed processes at once on separate physical cores of one CCD, windows packed by `plans.py`, parts `gs_fe` and `gs_term`).
Nothing in this family reads a disk inside a window: it loads every shot and builds every decoder before a window starts. The 36 slots, the stage's reference slots and the other families' CPU
slots share the eight cores; the references and this family start first (C8), the longest slots first (the order only sets the wall time), and a core whose job ended runs the harness's untimed
filler until the window's last process ends, so that every window sees eight busy cores. No pod and no Alpine node takes part.
## 1.6 Replacement and failure: one rule

C5 holds unchanged: a slot has at most three attempts; a slot whose three are all rejected or failed is not established; a lost machine follows C5.3. In this family:
- a Decoder Switching term is withheld unless all three repetitions of its slot are accepted: an FE or ROUTED slot not established withholds that d's term and composite (and no other d's);
- a missing or not established Grace fit (`--fit none`) withholds every converted value of this family; the raw-basis times stay published;
- no other value stands in for a missing one.
## 1.7 Statistics

- A cell is (term, d). A Decoder Switching term's value is the fastest accepted process of its three repetitions (C6: Decoder Switching is a baseline; `GAPSW_REGISTRATION.md`
  took the slowest, which is a sensitivity row); every repetition's value is published beside it, raw and on the Grace basis; a term with fewer than three accepted repetitions is withheld. The reference points take the slowest of three (C3, C6).
- Sensitivity rows (deciding nothing): the composite with the slowest of three (`GAPSW_REGISTRATION.md`'s statistic) and with the median of three in place of the fastest; the composite
  with FE converted by the plain-PM factor and ROUTED by the correlated-PM factor; the composite with FE converted by the
  correlated-PM factor (SURF-v2's earlier choice); every row on the raw basis (C7).
- Round = per shot / d is NVIDIA's convention: its per-round value times d is the per-shot value of the fit.

## 1.8 The Grace conversion and the verdicts

The conversion is C3's: the stage's one fit, the file `fit_s2.json` that `python -m speed_rt.analysis_surf fit` writes from the `srt-ref-*` records, read here as `analysis_gs --fit FIT.json`
(the fit is re-derived from its points with `grace_fit.fit`, and a file that does not re-derive is refused). Decoder Switching is one of the "other" CPU decoders of C3: both its terms, FE and
ROUTED, are converted by `grace_fit.to_grace(us, "other", fit)`, that is the common factor when the plain-PM and correlated-PM factors agree within their spread and otherwise
exp((ln f_plain + ln f_corr) / 2), fixed here before any data exist. Each PM decoder's own factor (FE as kind "plain", ROUTED as kind "corr", and FE as kind "corr") is a sensitivity row, 1.7.
The family decides no claim: everything is descriptive (`GAPSW_REGISTRATION.md`: no verdict), so C7's strict comparison is not used here. Every raw-basis time is published beside its
converted value, and every composite has its raw-basis twin (C7).

Rows (this family's own names: the speed inventory has no Decoder Switching row; `GAPSW_REGISTRATION.md` names the quantities):
- GS-fit: the stage's fit as C3 publishes it (factors, spreads, common or separate, every point's raw time, ratio and residual). "On
  the Linux workstation's CPU, NVIDIA's Grace time over ours for plain PM is F_P (spread S_P), for correlated PM F_C (spread S_C); one common factor F
  applies / each decoder keeps its own factor."
- GS-<d> for d = 7 ... 17: "On the Linux workstation's CPU, gap switching's front end took X us per shot at d = D (Y us on the Grace basis), correlated
  PM on its routed shots Z us (W us), and the composite V us (SURF-v2's earlier composite on the EPYC 7713 basis: U us)."
- A missing result: "ITEM was not established on the Linux workstation in SPEED-RT (REASON); the comparison that needs it is withheld, and no earlier value
  stands in for it."
Also shown: the routed share on the timing shots against the calibration's q, the front end over its own plain-PM decode per d, and the front
end over plain PM on the same circuit and session at d = 13. The ConvLines pairs' composites and the other decoders' are the surface family's
records of this stage (`STAGE2_surface.md`), on the same fit; the paper quotes them from there. No deployment-latency claim is made.

## 1.9 What each time includes

FE: per shot, plain PM's decode with the weight, setting node b's bit from the answer, the split graph's decode; from the shot's detector bits
in host memory to the gap. Excluded: reading the shots, building the matchings and the extended input buffers, the 50 warm-up shots, the
routing decision. ROUTED: correlated PM's decode of a routed shot, summed and divided by all 50,000 shots. No GPU, no handoff, no queueing. The CPU is the Linux workstation's AMD Ryzen 9
9950X; every time is published raw and on the Grace basis.

## 1.10 The analysis

`python -m speed_rt.gapsw.analysis_gs --session SESSION --prescreen PRESCREEN --manifest srt-gs.json --masks MASKS --fit FIT.json --beacon speed_rt/gapsw/beacon.json
--commits F_2 B_2 --out OUT`, run twice into two directories; the two must be byte-identical (`diff -r`). It refuses: records without the family's 36 planned
slots or their attempts; a record that is not the session's (its code digest, stage digest, deployed commit F_2 or B_2, manifest, sets,
circuits, policies; any difference of code, environment or deployment between two records of a session); a window whose `cpu_ws` verdict
the records do not support (the reference segments, the frequency, the block-read counters: `ref_before`, `ref_after`, `check_before`, `check_after`, `monitor`, `disk0`, `disk1` and `disk_delta` of the
window's `window.json`, C1); a shot digest other
than the draw's; a mask that is not the draw's or the policy's or that a process did not read; a manifest whose seeds are not the beacon's. It re-derives every process's verdicts from the raw values its record holds, with the frozen
limits of `hostwin.py` (the probe: every pass at most 1.10 times the session's reference, the largest over the smallest at most 1.06, 1.4), and refuses a record whose limits differ.
SESSION and PRESCREEN are the session directory and the pre-screen directory in `drive_gs`'s layout, which `speed_rt/cpu_ws/export.py` writes from
the CPU session's run directory (C1: `topo.json`, `sref/`, `plan.json`, `win/`). The exporter reads each process's own record through the `window.json`
that names its directory, marks it accepted only when its window passed, and takes every attempt from the windows' process lists; a process of a
rejected window does not count.
It writes `gs_numbers.json`, `gs_fit.csv`, `gs_times.csv` and `gs_tables.md` with every row above; the fit is the stage's, read from FIT.json and re-derived from its points (`gs_fit.csv` lists every reference point's raw time, NVIDIA's value, ratio and residual).

## 1.11 Order of events

C8 holds. In this family:
1. F_2 is pushed with this file, the code, `sets_gs.json`; B_2's record and manifest follow.
2. After F_2 and before B_2: the Linux workstation's setup and warm passes on dry shots only; nothing timed, and no timing shot exists.
3. After B_2: the draw on Alpine (`draw_gs.py`) and the transfer by C2's route; `gs_time.py mask` (untimed) for the six d before the stage's first timed window; then the family's 36 slots with the stage's Grace reference points, which start first on
   the Linux workstation.
4. The analysis twice, the records and outputs committed and pushed.

## 1.12 Departures from SURF-v2's gap-switching timing

- Timed on the Linux workstation, up to 8 processes at once on separate physical cores of one CCD, not on an exclusive EPYC 7713 node; three repetitions run in the
  stage's one session beside the stage's Grace reference points.
- Each process runs on the core its worker holds, not on one core fixed for the whole session.
- The Grace conversion is the stage's fitted one with the geometric mean of the two factors on both terms when they differ, not G(d) over the
  node's correlated PM per d; the terms take the fastest of three, not the slowest.
- Timing shots are beacon-drawn and fresh, not the studies' validation shots; the mask is made in the session by the batch decode (SURF-v2's
  `gapsw mask`, same functions) and every timed FE process checks its own loop against it.
- The host checks of C1 replace the node's exclusivity; the records' `attempts.json` and the tree's SUMS replace SURF-v2's `job.json`. The timed
  loops are `gapsw.loop` and `cpu_term.loop` themselves; `gs_time.py` wraps them (it does not call `gapsw.fe_term`, which has no window).
- The Linux workstation's venv holds only numpy, Stim and PyMatching at the registered pins.

## 1.13 The dry runs (13 October; dry beacon, never a timing shot; the records are kept)

Code as of the dry trees `dry-gs11`, `dry-gs13`, `dry-gs15` (their host checks, driver and pre-screen are the earlier ones, replaced by C1).
- Alpine, three repetitions, sets of 2 chunks, forced rejections, EPYC 7713P: 2,491 s, 89 of 90 slots accepted, the forced slot not
  established; the analysis twice byte-identical.
- Alpine, one repetition at full size (20 chunks, every set's digest checked against the manifest), EPYC 74F3: 1,614 s, 30 of 30 slots accepted;
  analysis twice byte-identical.
- A pod (Xeon Platinum 8470), one repetition at full size, host checks off: 2,038 s, 30 of 30 slots (reference points 488 s, front ends 1,387 s,
  routed terms 165 s); the per-slot durations of section 1.1 are from it.
- Alpine, three workers in parallel, 18 short slots plus forced rejections, EPYC 9534: 363 s, 17 of 18 slots, analysis twice byte-identical.
- winserver (Xeon Gold 6326), 13 October, eight workers, the `cpu_ws` machinery, dry shots, in windows of 8 together with the stage's other CPU parts (10 windows in all): the routed term at d = 13 with
  40,000 shots took 10.7 s (a term of 77.2 us). `gs_time`'s own probe check rejected 17 of 18 attempts (exit 4), so no front-end duration was measured there; the front-end durations of section 1.1 are the pod's
  times 1.3. In that run no window passed the reference check: the host's clocks varied between 1.0 and 2.6 GHz, and the spread of the reference segment over the 8 slots was 20 to 45 %, against a limit of 3 % then and 6 % now (C1).
