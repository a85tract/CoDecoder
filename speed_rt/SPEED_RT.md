# SPEED-RT: the registered re-test of the paper's speed numbers

SPEED-RT times again every speed number of the paper that the release can run, on fresh timing shots drawn from a
public randomness beacon after the rules are public. The paper reports SPEED-RT's numbers in place of the earlier ones,
whatever they are. A claim whose re-measured speed condition fails is reported as failing; nothing is re-chosen, re-run
or re-tuned because of a result.

## 0. Stages

SPEED-RT is registered in stages, one family of comparisons per stage, so that each family is timed as soon as its own
rules are fixed. The families' claims are separate: no comparison uses the records of two stages. Stage k is:

1. **F_k**, the freeze commit: stage k's section of this file, its code and everything shared that it uses, pushed to
   the public repository and to GitHub.
2. **Its beacon.** T_k is the time of the earliest record in GitHub's activity list of the repository whose `after` is
   F_k (GitHub's own record of the push). The round is the first drand mainnet round at least 600 s after T_k, and R_k
   is the SHA-256 of the round's signature, fetched after the round's time from two agreeing endpoints
   (`speed_rt/beacon.py`, SURF-v2's fetch and checks unchanged). The record keeps the activity list as GitHub returned
   it and names the stage's sets file.
3. **B_k**: the beacon record (and the stage's shot manifests, if it draws files), committed and pushed. Nothing of
   stage k is timed before B_k is public.
4. The timing sessions; then the analysis, run twice with byte-identical output, committed and pushed.

This file and stage 1's code are fixed at F_1. Every later stage is registered in its own file,
`speed_rt/STAGE<k>_<family>.md`, with its own code, frozen by its own F_k; it never changes a file that an earlier stage
froze. Later commits add stages, records and analyses; each stage's analysis runs on the code of its own F_k. Once any
stage's timing data exist, nothing frozen in an earlier stage changes. The rules of this section hold for every stage.

| stage | family | registration |
|---|---|---|
| 1 | the Cascade comparisons, on the RTX PRO 6000 and on the H100 | section 1 below |
| 2 onwards | every other speed number of the paper (BB and gross; surface and Ising; Decoder Switching) | their own files |

## 1. Stage 1: the Cascade comparisons on two cards

### 1.1 What is timed

The Cascade comparisons of the paper (rows C1-C10 of the speed inventory and the Cascade parts of headlines H3 and H4):
our ConvLines decoders (networks alone, pairs and chains) against Cascade, end to end and host to host, both sides on
the same card in the same job. Four rounds, each one job of fresh processes in the order of the original loops
(`speed_rt/cascade/drive.py`, which lists every process; `--list` prints the plan):

| round | processes | the original loop | rows |
|---|---|---|---|
| R1 | 36: D1-S, setting depspam_fm, p = 0.2 % and 0.3 %: Cascade alone (single, batched) and the pair S55'-deep -> C14' on the BB engine and on the multi-shot engine (single, batched), 3 repetitions | `phase_d/jobs/d1_speed.sh` | C1, C2, C10, H3 |
| R2 | 36: phase C, depspam_fm: Cascade alone, the pairs (configs 2, 7) and the Relay-BP chains (configs 3, 8), single and batched, 3 repetitions | `phase_c/jobs/speed_pipe.sh` | C6, C7 |
| R3 | 33: the networks at batch 4,096 (S55 on both engines, S55'-deep on the multi-shot engine, Cascade's C14 and C14') and the batched pipelines of configs 2 and 7 on both engines with Cascade alone, 3 repetitions | `batchopt/m2.py drive` | C3, C4, H4 |
| R4 | 48: d = 13, the d = 13 lock's configurations C1 (the chain to correlated PM), C4 (ConvLines with correlated PM), C5 (the pair) and C2 (Cascade alone) at p = 0.6 % and 0.35 %, single and batched, 3 repetitions | `surface/eval/jobs/e7_pipes.sh` | C8, C9, C10, H3 |

Every round runs on each of two cards, with the same processes, arguments and timing shots:
- **RTX PRO 6000**: one Slurm job per round on Alpine, one full NVIDIA RTX PRO 6000 Blackwell Server Edition per job
  (MIG off), with the node's AMD EPYC 9555 host, as the paper measured the Cascade tables.
- **H100**: one rented H100 80GB HBM3 SXM pod (RunPod, secure cloud): R1, R2, R3 and R4 in this order, each round
  one job, one process at a time on the pod. The paper's H100 paragraph (C5) was descriptive; here the H100 is
  registered like the RTX PRO 6000, and C5's sentences are restated from the H100's values of R1-R4.

Single mode times the first 20,000 shots of the stream one at a time (200 untimed shots first) and reports the mean and
the 99.9th percentile of the per-shot latency; batched mode decodes 500,000 shots (the networks at batch 4,096, routed
rows queued on the GPU) and reports the wall time per shot. The networks of R3 report the host-to-host time per shot at
batch 4,096.

### 1.2 Code

- The pinned trees of the Cascade studies, unchanged: the D1 tree for R1-R3 (`retest/pins/d1_tree.sha256`) and the
  d = 13 tree for R4 (`retest/pins/d13_tree.sha256`). Their locks, speed components and checkpoints are the studies'
  originals; `speed_rt/cascade/analysis_cas.py` (FREEZE) holds their sha256, and the harnesses refuse any other.
- The launchers `srt_cas.py` (BB72 harnesses) and `srt_d13.py` (d = 13), with `srtlib.py`, change three things from
  outside the trees and nothing else: the shot seeds (section 1.3), a sha256 trace of every sampled chunk, and the clock
  window (section 1.4). On the H100 they also admit the H100 in the harnesses' card guards (the BB harness `m2.py` has
  its own H100 switch; the others name the RTX PRO 6000 only), and the d = 13 harness uses the pod's own kernel caches
  (section 1.5).
- RTX PRO 6000 environment: the studies' container (`pyfull.sif`, sha256 adba1be6...) with their venv (Python 3.12.14,
  torch 2.11.0+cu128, Triton 3.6.0, CUDA 12.8, numpy 2.2.6, Stim 1.16.0, PyMatching 2.4.0; Relay-BP 0.2.2), job-local
  copies of the studies' kernel caches (R1-R3) and the d = 13 lock's own caches (R4), as the original jobs ran.
- H100 environment (`speed_rt/cascade/h100_pod.sh setup`): Python 3.12.3 and a venv with the D1 tree's pins
  (`batchopt/req_cu128.txt`: torch 2.11.0+cu128, Triton 3.6.0, numpy 2.2.6, Stim 1.16.0, PyMatching 2.4.0) and Relay-BP
  0.2.2; the trees, checkpoints, locks and Relay-BP circuits at the cluster's paths, every file's sha256 checked on the
  pod.

### 1.3 The timing shots

- Sets: `speed_rt/cascade/sets_cas.json`, eight sets, one per (setting or code, p) of the speed streams. Chunk j of a
  set replaces chunk j of the original speed stream (seed kind 12), so every process reads the chunks its original loop
  read. No shot files are drawn: each process samples its chunks in the run with the harness's own sampler (Stim 1.16.0).
- Seeds: seed(set, j) = 2^62 + (the first 8 bytes, big-endian, of SHA-256(R_1 || TAG || j as 2 bytes big-endian)) mod
  2^62, TAG = "convlines speed-rt SET" and a newline (`speed_rt/sets.py`); `python speed_rt/cascade/srtlib.py seeds
  --beacon BEACON --out SEEDS` writes the seeds file, and every process recomputes each seed from the beacon record before it samples.
  A kind-12 chunk outside the sets stops the process.
- Every process writes the sha256 of every chunk it sampled (its set, index and seed). The analysis checks that every
  process read only the chunks of its sets and that a chunk has one digest in every process of a card; a chunk whose
  digest differs between the two cards' hosts is reported (Stim's output can depend on the CPU) and decides nothing,
  since each card's comparisons read one stream on both sides.

### 1.4 The clock window and acceptance (both cards)

- Every timed window (single mode: from just before the 200 untimed shots to the last decision; batched mode: from the
  first copy to the wall-time reading; networks: each timed path and batch) is sampled by `nvidia-smi` every 100 ms
  (SM clock, maximum SM clock, the active clock-event reasons, the enforced and the default power limit), and the
  card's compute applications are listed every second.
- A window is **rejected** for outside interference: another compute application on the card; a gap of more than 1 s
  between samples; a clock-event reason other than GPU idle, applications clocks or SW power cap (that is, any HW
  slowdown, sync boost, SW or HW thermal slowdown, HW power brake or display-clock reason); an enforced power limit below
  the card's default; or an SM clock below 90 % of the maximum in a sample without the SW power cap.
- The workload's own power cap (the SW power cap at the card's default limit, no other process on the card) is part of
  the card's speed under that workload: it is recorded ("workload power cap"), not rejected. App. D's flag, the samples
  below 97 % of the maximum, is recorded and published for every process, as the paper reported it, and decides nothing.
- No burn or other load runs before a window (the original loops had none).
- A process counts only when every window it has is accepted and its harness's own checks pass (the lock and identity
  checks, the shot trace). The analysis re-derives every window's verdict from the samples.

### 1.5 The H100 pod

- Pre-screen, before anything is timed on the pod: five GPU checks on the idle GPU (each a separate process that holds
  1 GiB and runs 3 s of fp16 matrix products, as `surface_v2/surf_v2/gpucheck.py` measures them), all passing; a check
  passes when the median SM clock is at least 97 % of the maximum, every sample's clock-event reasons are within those
  section 1.4 accepts (GPU idle, applications clocks, SW power cap), the enforced power limit is the default, under 64
  MiB were in use before it, and no other process used the GPU. Then the CPU steal at most 1 % over 5 s on the
  container's CPUs, and the container's CPU quota at least 20 CPUs (the batched chains use 16 worker processes). The
  pod's CPU model, quota, driver, power limit and maximum SM clock, and every check's samples, are recorded.
- A pod that fails is released and replaced. At most five pods are screened for stage 1, one of which runs. If the
  fifth fails, the rounds run on the failed candidate with the fewest failed checks (ties: the earlier pod), and the
  H100 comparisons are reported descriptively, without a verdict.
- The kernel caches are compiled for the H100 on the pod before the timed block, by passes of the rounds on the dry
  shots (never a timing shot), as the RTX PRO 6000 jobs start from the studies' caches. The d = 13 harness then runs
  under a lock derived from the study's lock (`srt_d13.py h100-lock`), in which only the card's identity and the
  kernel-cache digest are the session's; every timed d = 13 process checks the cache's digest at its start and end.
- One timed process at a time on the pod; the rounds run in the order R1, R2, R3, R4.

### 1.6 Replacement and failure: one rule

1. A process whose window is rejected, or which fails (a non-zero exit, a missing row, window or shot trace, or a chunk
   outside its sets), is replaced at once by a fresh process with the same arguments, at most twice: a slot has at most
   three attempts.
2. A slot whose three attempts are all rejected by the clock window is not established. The job goes on. On that card,
   every comparison that uses the slot is withheld (no verdict), and no other process and no earlier value stands in.
3. A job that does not reach its end is a technical failure: its node or pod is lost, its time limit is reached, or
   the last attempt of a slot fails (`drive.py` then stops). The round then runs once more as a whole, from its first
   slot, as a new job on a card of the same kind (a new Slurm job; on the H100, a newly pre-screened pod if the pod is
   lost, otherwise the same pod), and only the new job's records count. If the new job also does not reach its end,
   every comparison of that round on that card is withheld.
4. Nothing runs again because of a value it produced. Every attempt, accepted or not, and every job is kept and
   published (`attempts.json`, the window records, the shot traces).

### 1.7 Statistics

- A cell is one (card, round, decoder, p, mode, statistic). Its value is the slowest accepted process of its three
  repetitions, on both sides of every comparison: slowest against slowest, the paper's statistic for Cascade.
- Sensitivity rows (deciding nothing) repeat every Cascade comparison with Cascade's fastest accepted process.
- Every repetition's value and the range are published beside each cell.
- Per round = per shot / 6 on BB72 and / 13 at d = 13. The 99.9th percentile is each process's own, over its 20,000
  timed latencies (the harness's computation).
- Speed-up = Cascade's value / ours.

### 1.8 Verdicts, per card

- Every statement is judged on each card from that card's records only, and each card has its own verdict for each
  claim; neither card is primary. "X takes less time than Y" holds when X's re-measured value is strictly below Y's. For
  the tail statements (C7's, C8's and C10's 99.9th percentiles) the paper says the routed decoder's tail is higher, and
  that statement holds when the re-measured tail is higher. No statistical test, as in the paper.
- Every printed number, ratio and range of rows C1-C10, H3's Cascade range and H4's per-round time is replaced by its
  re-measured value on each card, whatever it is. Ranges are recomputed by the paper's arithmetic and rounded in the
  direction that weakens the claim: speed-ups down, times and fractions of time up.
- The rows, their formulas and every comparison are those of `speed_rt/cascade/analysis_cas.py`, which is part of this
  freeze; C4's Eq. (3) estimates and D1's prediction check are recomputed and stay descriptive.
- A comparison that fails is written: "On the CARD, OURS took X against Cascade's Y (UNIT); the statement does not
  hold." A missing result is written: "ITEM was not established on the CARD in SPEED-RT (REASON); the comparisons that
  need it are withheld, and no earlier value stands in for it."
- No deployment-latency claim is made from these numbers.

### 1.9 What each time includes

End to end and host to host on the named card: the shot's detector bits from pinned host memory, the network or
Cascade on the GPU, every routed shot's fallback (C14', CS13, full-DEM Relay-BP or PyMatching in their worker
processes), to the decision on the host. The host CPU is the RTX node's AMD EPYC 9555 or the H100 pod's Intel Xeon, and
the results name it.

### 1.10 The analysis

The records are one directory per card and round, `RECORDS/RTX/R1` ... `RECORDS/H100/R4`, each a `drive.py` output
directory as the job wrote it. `python speed_rt/cascade/analysis_cas.py --records RECORDS --out numbers.json --table
table.json --verdicts verdicts.json`
reads every job's records, checks them (the locks' and checkpoints' sha256, the shot traces against the beacon's seeds,
every window's verdict), and writes every cell with every repetition, every reason a value is missing, App. D's clock
statistics per card, and every row of C1-C10 with its old value, its new value on each card and each card's verdict.
It is deterministic; it runs twice, and the two outputs must be byte-identical. The records, the outputs and the run log
are committed and pushed.

### 1.11 Order of events

1. F_1 is pushed; its beacon record is fetched and committed as B_1 (`speed_rt/cascade/beacon.json`) with the seeds
   file's sha256; B_1 is pushed.
2. After F_1 and before B_1 is public, the H100 pod may be pre-screened and set up and its d = 13 caches compiled, and
   the RTX PRO 6000 jobs may be queued; no timed process starts before B_1 is public.
3. The four RTX PRO 6000 jobs and the H100 pod's four rounds run in parallel. Every job's start (card, host CPU, time)
   is logged in the run log before it starts.
4. The analysis runs twice, and its outputs are committed and pushed.

### 1.12 Departures from the paper's Cascade method

- The paper recorded the clock and gated nothing (App. D flagged the rows below 97 %); stage 1 rejects windows with
  outside interference, replaces them (section 1.6) and publishes App. D's flag.
- The H100 runs the RTX PRO 6000's four rounds, registered, in place of the earlier descriptive H100 session (C5); its
  harnesses admit the H100 and its d = 13 kernels are compiled on the pod.
- R2 runs the 36 processes of phase C that feed Table VII and App. E (the original job also timed the uniform setting
  and the S-R pipelines, which feed no printed number); R3 runs the networks and pipelines that feed C3 and C4.
- The beacon's seeds replace seed kind 12; single mode times 20,000 shots, the harnesses' count.
