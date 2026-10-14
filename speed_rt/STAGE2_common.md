# SPEED-RT stage 2: the rules its three families share (STAGE2_common)

Stage 2 is
one freeze, F_2, holding three families: the surface comparisons with Ising (`STAGE2_surface.md`), BB72 and gross (`STAGE2_bb.md`) and Decoder
Switching (`STAGE2_gapsw.md`). This departs from section 0 of `SPEED_RT.md` ("one family per stage") because the three share one CPU
session and its one set of Grace reference points, and a converted CPU time may not rest on another stage's records; the families' claims stay
separate, and a comparison uses its own family's records and the stage's Grace factors. A family not ready at F_2 is registered as a later stage
with its own file, its own F_k and its own Grace reference points timed in its own session; it cites C1-C7 and never reads stage 2's records.
A family file adds its own rules and may not weaken one of these. The rest of section 0 (T_k, the round, R_k, B_k) holds; C8 applies it.

## C1. Machines

The authors moved the CPU session from winserver to the Linux workstation before F_2, because winserver failed its reference checks and its link on 13 October. Only the dry runs described in these files ran on winserver; the passages about those runs describe them and are kept as they are.

GPU timing runs on RunPod H100 80GB HBM3 SXM pods (secure cloud) and nowhere else. Every CPU part is timed on the Linux workstation and nowhere else:
never on a pod, on Alpine's EPYC nodes or on the Mac mini's M4. The CPU parts are the family files' lists of: plain and correlated
PyMatching (PM), our CPU terms, the Grace references, Relay-BP, full-DEM Relay-BP, ldpc BP-OSD, Decoder Switching.

- Pods. A pod is pre-screened before anything is timed on it by stage 1's rule (`SPEED_RT.md` section 1.5, `speed_rt/gpu_prescreen.py`; the BB pod's `bb/prescreen_bb.py` applies the same rule):
  five GPU checks on the idle GPU, all passing; CPU steal at most 1 % over 5 s; a container quota of at least 20 CPUs. There is no
  host-CPU busy check and no window of a pod is judged by a host-CPU share, because a pod times GPU work only. The pod's CPU model, quota,
  driver, power limit, maximum SM clock and every check's samples are recorded. A pod that fails is released and replaced; at most five
  pods are screened per role (stage 1's bound; a failed pod is released at once, so six H100s are never exceeded), and if the fifth fails the role runs on the failed candidate with the fewest failed checks (ties: the
  earlier pod) and its comparisons are descriptive, without a verdict. A GPU window is accepted by the clock rules of section 1.4 of
  `SPEED_RT.md` (`speed_rt/watch.py`, rules floor and throttle). One timed process at a time on a pod. A pod is requested when F_2 is
  pushed (the surface family's S-high pod earlier, for its plan builds: STAGE2_surface) and released when its outputs are fetched; an enforcer stops a pod that has no process of ours for 15 minutes (a stop is a lost
  machine, C5); at most six H100s exist at any time.
- The Linux workstation: an AMD Ryzen 9 9950X (16 cores, 32 threads) on Ubuntu 24.04, native Linux, running the code on the host itself. The kernel, the CPU count, the memory and the swap
  are checked and recorded with the build. The environment (`speed_rt/cpu_ws/setup_env.sh`): Python 3.12.3 (Ubuntu 24.04's CPython, copied into the tmpfs), numpy 2.2.6, scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, relay-bp 0.2.2, ldpc 2.4.1, stimbposd 0.2.0, sinter 1.16.0 and beliefmatching 0.2.0, the versions of winserver's environment, no torch.
- The Linux workstation's settings for the timing, set by the authors for the session and restored afterwards: CPU frequency governor `performance` on every online CPU
  (`amd-pstate-epp` in active mode, energy-performance preference performance) with boost on; the SMT siblings of CCD0 (core complex die 0: CPUs 0-7), CPUs 16-23,
  taken offline, so the online CPUs are 0-15 and 24-31; every process but the CPU worker's own confined to CCD1 (CPUs 8-15 and 24-31: systemd `AllowedCPUs` on
  `system.slice`, `init.scope` and every other account's slice), so that CCD0's CPUs 0-7 are left to the timed processes; the Bluetooth service stopped (in the dry runs
  its daemon took CPU time on CCD0); at most 8 timed processes at once, on CCD0, each on one CPU whose SMT sibling is offline; no other user runs anything on the box
  during the timing. Each reference segment's limit is 1.06 (below). Its evidence on the Linux workstation: in the dry run under the final settings (13 October, 16 windows of 8 slots) every window segment spread at most 1.018 (median 1.007) and every value lay within 1.016 times the session's reference (17.306 us per block); before those settings 3 of 68 segments exceeded 1.06 (1.09 to 1.19), each while another process ran on CCD0 or on a slot's SMT sibling. The dry runs on winserver (Windows power
  plan High performance, turbo boost on) spread 1.013 to 1.053 at 8 slots (largest over smallest), and 1.03 would have rejected about half of their windows; turbo boost off was tried there and
  dropped: it lowered the clock to 1.9-2.6 GHz, below the 2.90 GHz base, and widened the reference segments to 1.02-1.23. The harness records, per session, the governor, the energy-performance
  preference and the boost state of every CPU, and the clock probes.
- The session's length: the slowest CPU items' shot counts were cut before F_2 so that the CPU session fits one evening (the authors' decision;
  every count is in STAGE2_bb 1.3 and STAGE2_gapsw 1.3, each subset the first N shots, or the first N routed shots, of the drawn set in shot order).
  With `speed_rt/cpu_ws/plans.py`'s packing and the per-shot times of the dry runs on the box, the session then takes about 50 minutes in all (46 windows
  of at most 8 processes: repetition 1 about 18 minutes with the BB warm-ups, repetitions 2 and 3 about 14 each, plus the session reference and the shot preparation;
  winserver's plan had put it at about 91 minutes). It starts only when a short reference check reads within 1.10 times the session reference of the final dry run on the box (17.306 us per block), so that windows failing on a slow host do not spend attempts.
- The CPU parts run the families' own timing code, one code path per decoder (`speed_rt/bb/`, `speed_rt/gapsw/`, `speed_rt/gref.py`, the surface code, `speed_rt/grace_fit.py`).
  `speed_rt/cpu_ws/` supplies only the machinery: pinning, windows, reference segments, the block-read check, the frequency record, interleaving, saving and fetching. On the Linux workstation the harness runs in its Linux host mode: the same harness with its Windows parts replaced (`cfg_box.json`; `README.md`, "Linux host"): the topology read from `/sys` (SMT siblings, shared L3), the timed CCD from the configuration (CCD0), the disk rule on the harness's own read counter (below), and per window the governor, energy-performance preference, boost and online CPUs at both ends, each CPU's frequency every 2 s with the load average, each CPU's busy share and every process that is not ours (user, command, CPU seconds, CPUs)
- Everything a timed process reads (interpreter, venv, code, circuits, detector error models, timing shots, routing masks) sits in the tmpfs under `/dev/shm/srt_box` (`/dev/shm`: 62 GB) before the
  first window, the C libraries the interpreter links included; `pinlibs.py` locks the loader and the locale files.
  Before a window's first reference segment and after its last the harness reads its own read counter (`read_bytes` of `/proc/self/io`, which includes every process it
  started and reaped: the reference, timed, filler and clock-probe processes and their children); a window passes the disk check when that counter did not grow. Each
  process's own block reads (`wait4`, children included) are in its record. The whole disks' counters and `/proc/vmstat` are system-wide on a box that other users share:
  they are recorded and decide nothing. The session's reference has the same rule.
- At most 8 timed processes run at once, on one CCD (CCD0 = CPUs 0-7 with SMT siblings 16-23, CCD1 = CPUs 8-15 with siblings 24-31), each pinned to one CPU of CCD0 with its SMT sibling offline, so that
  it has a physical core to itself; slot k runs on CPU k (k = 0 to 7); the harness, its monitors, the spare clock probe, the fetcher and the hand-off copy run on CCD1 (CPUs 8-15 and 24-31). `plans.py` packs the jobs of repetition 1, then of repetition 2, then of 3 into windows of at most 8, the longest first,
  with the two sides of every comparison alternating across the slots and the order reversed in every other window, so that neither side keeps a slot or a higher load level; a slot whose job ends early runs an
  untimed filler until the window's last process ends, so that eight cores stay busy. The Grace references run in the same windows, at the same concurrency.
- Per window (up to 8 processes started together): before it and after it every slot runs the reference segment, plain PM per block (`timing.cpu_term ref --plain --d 9` on validation
  chunk 0), then a 0.03 s clock probe (a dependent-add chain); the CPU frequency is recorded during the window (the same probe on a spare CPU every 2 s, and the kernel's cpufreq readings when they are
  readable). The window passes when there is no block read, every reference value is at most 1.10 times
  the session's reference with a spread (largest over smallest within a segment) of at most 1.06, and every process exits 0. The session's reference is the median of 3 segments at
  full concurrency (24 values, `harness.py sessionref`); a reference whose disk check saw a read is taken again, at most three times in all,
  and the earlier attempts stay in its record (`earlier_attempts`). A process that exits 4 (a host check of its own, section 1.4 of `STAGE2_gapsw.md`) is run again by the harness in extra windows
  (`r001`, `r002`, ...), at most three attempts in all; a process of a rejected window, and any other failure, runs again in a later window (C5). Other users share the box but run
  nothing on it during the timing; these checks are the protection if one does.
- The session's records, each written once and never overwritten. `topo.json` (`harness.py topo`): the CPU groups (CCDs and SMT siblings), the slots and the spare CPUs, kernel, memory, swap, the tmpfs size, `lscpu`,
  and the host facts (CPU model, build, the governor, energy-performance preference and boost state of every CPU, other users' processes). `sref/sessionref.json` (`harness.py sessionref`): the 3 reference segments (per slot the value per
  block, the exit code and the clock probe), their median `reference_us`, the largest over the smallest value, and the disk and clock records. `plan.json` (`plans.py`): every window with its
  jobs in slot order. `win/<window>/` (`harness.py run`; windows `w001`, ..., and `r001`, ... for the repeats): `window.json`, `monitor.jsonl` and one directory `s<k>_<job>/` per process. `window.json`
  holds the window's id, repetition, start and end, the slots, the disk counters before and after (`disk0`, `disk1`) and their difference (`disk_delta`, rule "own read_bytes": passed when the harness's own read counter
  did not grow; the system-wide counters are kept beside it as `disk_box_delta` and decide nothing), both reference segments (`ref_before`, `ref_after`) with their checks (`check_before`, `check_after`: worst value over the session's reference, spread, median, passed),
  the clock monitor's summary (`monitor`: the means, minima and maxima of the CPU frequency and of the probe), for each process its slot, CPU, job, part, side, attempt, command line, exit code,
  wall time and the directory of its own record, the filler slots, and the verdict `passed`. A process's own record is what its family's code wrote into its directory. `speed_rt/cpu_ws/export.py` turns a run
  directory into the layout each family's analysis reads (STAGE2_surface 1.14, STAGE2_bb, STAGE2_gapsw 1.10). It copies values and changes none;
  a process of a rejected window becomes a rejected attempt and does not count.
- Partial results. At each window boundary, outside the block-read check's span, the harness copies the finished window's records from the
  tmpfs to the session's output folder on the box's disk; the fetcher reads only that folder (`fetch_box.sh`, on CCD1, copies it to the records folder every 15 minutes and once at the end). A pod's driver writes each finished slot's records to the container disk, and one rsync pulls them. The fetcher pulls the new
  records to the Linux workstation (`~/work/s2/raw/<machine>/`) every 15 minutes while a batch runs and once at its end, logging each fetch's start and
  end. Fetched records serve safekeeping and the progress count only: no number or verdict is computed from them before the frozen analysis,
  and nothing in the run changes because of them. A batch on the Linux workstation runs detached from any login session, keeps its state in files and
  resumes after a lost link; a lost link is not a lost machine.

## C2. The hand-off from the untimed GPU pass to the Linux workstation

Our pairs' CPU terms (correlated PM, Relay-BP or full-DEM Relay-BP on the routed shots) and Ising's residual PM need the GPU side's decisions on the
timing shots. For each GPU role, untimed passes run on the role's pod on the timing shots after B_2: the routing masks (the margin read, the frozen tau, the
routed shots) before the role's timed block, and, where the counted plan decides them (Ising's residual detectors), the counted plan's outputs after it. Nothing in
a pass is a timing record. The route does not touch Alpine: the outputs are pulled to the Linux workstation, to
`~/work/s2/handoff/<family>/`, with a `SHA256SUMS` file and a `DONE` file written last; once `DONE` exists they stay there, and nothing is started for them during a window; at a window
boundary the harness copies the folder into the tmpfs and checks `SHA256SUMS`. A CPU term never recomputes a routing decision. Every shot set is generated once, on Alpine, and the same bytes go to the pods and to the Linux workstation (a chunk regenerated on another machine has other bytes); every chunk's sha256 is
checked against B_2's manifest on every machine that reads it, so that the pods and the Linux workstation use byte-identical shots. A pass that fails is attempted at most three times; if it never completes, the CPU terms that need it are not established (C5). Each part is pulled when it is done.

## C3. The Grace conversion

Reference decoders: NVIDIA's PyMatching, plain and correlated, as NVIDIA ran them (Ising paper, arXiv 2604.12841 v1, Tables VII and IX), on
NVIDIA's legacy rotated-surface-code memory circuits at the 18 points of `speed_rt/grace_fit.NVIDIA_US_PER_ROUND`: plain PM at d = 13, 21, 31
and correlated PM at d = 5, 9, 13, 17, 21, 31, each at p = 0.3 % and 0.6 % (the twelve circuits of `speed_rt/circuits/ref_circuits.json`, sha256
checked). The stage times its 18 points once, in its CPU session, three repetitions each, at the session's concurrency, by `speed_rt/gref.py time` on the sets
`srt-ref-d<d>-p<3|6>` of `speed_rt/sets_surf.json` (one chunk of 10,000 shots per point); a point's time per shot is the slowest of its
three repetitions' means: the slowest time gives the smaller factor, and a larger factor favours our pairs (the baselines are all CPU, ours only a routed
fraction), so the choice goes against the core claim. The fit is `speed_rt/grace_fit.py` (`fit`, frozen with this stage): per point r(i) = NVIDIA's time per round x d / ours; one factor per decoder f = exp(mean
ln r(i)) over all of its points; its spread s = the standard deviation of ln r(i) (n - 1); the common factor is the geometric mean over all 18
points. The two decoders agree when |ln f_plain - ln f_corr| <= max(s_plain, s_corr). The fit file `fit_s2.json` is written by `python -m speed_rt.analysis_surf fit` from the
points' records; every family's analysis reads it (`--fit FIT.json`), re-derives it from its points with `grace_fit.fit` and refuses a file that does not re-derive; `--fit none`
means the fit was not established, and then every converted value is withheld. The fit is not established when one of its points is not, or when the CPU
session did not reach its end (`analysis_s.qualifies_cpu`); the fit file then says so (`complete` false, `fit` null), and every family withholds its converted values.

Frozen now, before any data exist: which factor a CPU time takes.
- The common factor, when the two agree, for every CPU time of the stage.
- Otherwise each PM decoder takes its own factor: plain-PM terms f_plain (the PM baseline, the PM share after Ising), correlated-PM terms f_corr (the correlated
  PM baseline, our correlated-PM terms, Ising + correlated PM); and every other CPU decoder (Relay-BP in every form, full-DEM Relay-BP, ldpc BP-OSD, Decoder
  Switching) takes exp((ln f_plain + ln f_corr) / 2). `grace_fit.fit` returns these as `applied["plain"]`, `applied["corr"]` and `applied["other"]`, and `to_grace(us, kind, result)` takes kind "plain", "corr" or "other".
- Converted time = raw time x the applied factor (the raw basis is the Linux workstation's own, unconverted time; "raw" means it throughout). Not converted: GPU terms, the handoff term, and CUDA-Q's GPU BP-OSD, which is
  timed host to host on the pod and names that host's CPU.
- Published: every raw time and repetition, every converted value beside its raw value, the factors, the spreads, the decision, and per point
  the raw time, the ratio (a diagnostic) and the residual ln(NVIDIA) - ln(applied factor x ours / d). A Grace-basis number is an estimate:
  no Grace hardware is used and no claim about it is made. If the fit is undefined (a reference slot not established), nothing is converted.

## C4. The composite

Composite = the GPU term (the pod's H100, input on the GPU) + the Grace-basis CPU term (the Linux workstation) + the handoff term, as registered. The CPU-GPU transfer is
neglected for every family, as in NVIDIA's setting; host-side work (routing decisions, packing, indexing) stays. The handoff term is therefore the host-side work only,
timed in the GPU session on the pod and not converted. A transfer that a family's earlier handoff included is split off (BB's pinned-memory round trip of about 16 to 22 us,
`px.hbb`; BB's handoff term is then the registered host-side allowance of its family file) and left out; the measured transfer times are reported in the outputs as context and decide nothing. Keeping the host-side work, which NVIDIA's setting does not count, only lengthens our times. A decoder with no GPU part is
its Grace-basis CPU term. The GPU and the CPU terms of a composite come from two machines, as the paper's composites always did, and the text beside every composite says so, names
the pod's GPU and the Linux workstation's Ryzen 9 9950X, and says that the CPU term is on the Grace basis. The baseline CUDA-Q is the pod's host to host time, its host named.

## C5. Replacement and failure: one rule

1. A process whose window is rejected, or which fails (a non-zero exit, a missing record, a check of its own), is replaced by a fresh
   process with the same arguments (at once on a pod; on the Linux workstation in a later window), at most twice: a slot has at most three attempts, on any machine.
2. A slot whose three attempts are all rejected or failed is not established. The session goes on. Every cell, comparison and composite that
   uses the slot is withheld (no verdict); no other process and no earlier value stands in. A reference slot not established withholds the
   fit and so every Grace-basis value of the stage; the raw times of accepted processes are still published.
3. A machine is lost when a pod is stopped, lost or past its time, or when the Linux workstation restarts, loses its tmpfs or its batch dies (a lost link is not a loss).
   - Pod: the lost session's records are kept and published and do not count, because a role's GPU terms come from one pod's one session; the role runs once more
     as a whole on a newly pre-screened pod (C1) and only the new session's records count. If that session is lost too, every comparison that uses the role's
     GPU terms is withheld.
   - Linux workstation: the records already fetched stand. The slots without an accepted record run in a new session, after a new warm pass and reference, with the
     attempts they have left (an attempt that was running when the machine was lost counts); the earlier session's records stand under their own session's
     reference, and the new session's reference must lie within 1.10 times of the first's, or its slots are not established. A second loss makes every slot
     without an accepted record not established.
4. Nothing runs again because of a value it produced. Every attempt, accepted or not, every session, pre-screen, window record and fetch is kept
   and published (`attempts.json`, the window records, the run log).

## C6. Statistics

- A cell is one (item, point, mode, statistic) of a family file. Every family uses one rule, the direction that goes against our claims: a baseline takes the
  fastest accepted of its three fresh processes, our side the slowest accepted of ours; no medians. The Grace reference points take the slowest of their three (C3).
  Decoder Switching and the Ising and PM terms of a baseline are baselines. A family file lists its items under this rule; it states a different statistic only for
  a quantity that is not a comparison's side, with its reason. The statistic is chosen on raw times and then converted; the factor is a constant, so the order does not matter.
- Every repetition's value and the range are published beside each cell, raw and on the Grace basis. Sensitivity rows decide nothing.
- Speed-up = the baseline's value / ours. Ranges are recomputed by the paper's arithmetic and rounded in the direction that weakens the claim:
  speed-ups down, times and fractions of time up. Per round = per shot / d (d rounds in a shot). No statistical test, as in the paper.

## C7. Verdicts and wording

- "X takes less time than Y" holds when X's re-measured value, on the Grace basis for CPU parts, is strictly below Y's. Every printed number,
  ratio and range of the stage's rows is replaced by its re-measured value, whatever it is.
- A comparison that fails is written: "At POINT, OURS took X against BASELINE's Y (UNIT; GPU terms from POD, CPU terms from the Linux workstation on the
  Grace basis); the statement does not hold." A missing result: "ITEM was not established on MACHINE in SPEED-RT (REASON); the comparisons that
  need it are withheld, and no earlier value stands in for it."
- Every verdict is also computed on the raw basis (no Grace conversion); these verdicts are published as a sensitivity check and decide nothing.
- The outputs carry, for every CPU time and every composite, the raw-basis value beside its Grace-basis value, and every speed-up and
  verdict on both bases, as table rows the paper prints in its appendix. The Grace basis decides; the raw basis is reported beside it.
- Descriptive rows (the family files list them) carry no verdict. No deployment-latency claim is made from these numbers.

## C8. The beacon and the order of events

1. F_2 holds the three family files, this file, the stage's code and sets files, `speed_rt/grace_fit.py` and the `speed_rt/cpu_ws/` harness,
   with `speed_rt/sets_s2.json`, pushed to the public repository and GitHub. T_2, the round (the first drand mainnet round at least 600 s after T_2) and R_2 follow
   `SPEED_RT.md` section 0 and `speed_rt/beacon.py`; the record names the stage's sets file `speed_rt/sets_s2.json`, the union of `speed_rt/sets_surf.json`,
   `speed_rt/bb/sets_bb.json` and `speed_rt/gapsw/sets_gs.json` (`beacon.py fetch --sets`); seeds follow `speed_rt/sets.py`. B_2, the beacon record and the
   shot manifests, is committed and pushed. Nothing of stage 2 is timed before B_2 is public.
2. After F_2 and before B_2 is public: pods are requested, pre-screened and set up (caches and engines built on dry shots, never a timing
   shot); the Linux workstation is set up (C1: tmpfs, warm passes on dry shots). No timing shot exists and no timed window runs. The surface family's
   S-high pod is requested, pre-screened and its plans built before F_2, from files byte-identical to F_2's, which its analysis checks
   (STAGE2_surface, the builds' record).
3. When B_2 is public the GPU sessions and the CPU session start together; the draw and the shots' transfer come first. On the Linux workstation the Grace
   reference points and Decoder Switching (its front end and routed PM, all CPU, no hand-off) start first. Each comparison of a pair against its
   backend alone starts once both of its sides' inputs (the shots and the pass's outputs, C2) are in the tmpfs, so that its two sides stay
   interleaved; the pods' timed blocks run meanwhile.
4. Every file whose hash the frozen code checks (locks, session records, manifests) names files by relative path only, so that publication changes no hash. Every session's start (machine, CPU model, time) is logged before it starts. The analysis runs twice with byte-identical output; the records,
   the outputs and the run log are committed and pushed. Once any stage's timing data exist, no file frozen by an earlier stage changes.

## C9. Checklist: where each requirement is met

| requirement | where |
|---|---|
| GPU timing on H100 SXM pods with stage 1's pre-screen, no host-CPU check, pods time GPU work only | C1 |
| every CPU part on the Linux workstation, never on a pod, Alpine EPYC or the M4; families' own code, cpu_ws machinery only | C1 |
| tmpfs, no block read by our processes in a window (the harness's own read counter), governor and boost settings, build recorded; at most 8 processes on physical cores of one CCD, sides interleaved, same concurrency for references; per-window reference and frequency, the session's own reference | C1 |
| Grace: factor per decoder over all points, spread, residuals, raw beside converted, common-factor rule fixed before data; every CPU time converted, BB and gross included, CUDA-Q not (15:55) | C3 |
| composite = GPU + Grace-basis CPU + handoff, transfer neglected for every family and only host-side work kept, two machines stated | C4 |
| baselines fastest of 3, ours slowest, Grace references slowest of 3, no medians; raw-basis verdicts as a sensitivity check | C3, C6, C7 |
| factor for other CPU decoders: common if the two agree, else the geometric mean | C3 |
| untimed GPU pass, hand-off route, SHA256, byte-identical shots; order on the Linux workstation | C2, C8 |
| one failure rule with no choice left; the Linux workstation's fetched records stand after a loss, a lost pod's role runs again (C5.3) | C5 |
| partial results every 15 minutes, from the output folder only | C1 |
| pods requested at F_2 (S-high earlier, for its builds), released when pulled, 15-minute idle stop, at most 6 H100s | C1, C8 |
| staged freeze, beacon after F_2, B_2 before timing, no later change; one freeze for three families (section 0); relative paths in hashed records | head, C8 |
| Decoder Switching on the Linux workstation, descriptive; all 48 surface comparisons registered; d = 17 converted with the same fitted factor (no separate d = 17 basis); B11, E3 and E9 out by decision | family files |
