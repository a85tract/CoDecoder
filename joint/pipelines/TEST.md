# ARB3: the joint decision in the timed pipelines and the reads

2026-10-19. The release is this repository at 94aaaa7, and the changes are on top of it: in this repository the
commits 824afab and 0e9321e (as run: `code/release_arb-prod.patch` of the research repository's branch arb-prod,
with the changed files under `code/release/` and their sha256 in `code/SHA256SUMS`; joint/README.md lists the one
docstring line that differs).
Checks ran on one rented **H100 80GB HBM3** pod (RunPod, secure cloud, 3.99 USD/h, about 66 minutes), because the RTX PRO
6000 was sold out. The pod's own driver was 580.126.09, with torch 2.11.0+cu128, Triton 3.6.0 and Python 3.12.14, which
are the D1 tree's pins (`PINS_EQUAL`). Nothing here is registered; every time below is a smoke time on the H100.

## 1. The rule and its fixed arithmetic

**BB72** (D1's composition S55'-deep -> C14' at p = 0.2 % and 0.3 %). On a routed shot (S's margin below the lock's tau1,
compared in float64 as before) where S's and Cascade's decisions differ, the answer is the class y that maximizes
t(y) = s_N(y) + w (y . z). Every other shot is answered as in the composition. The arithmetic is fp32 on the GPU, in this
order:
- s_N(y) is the score from bb.fused's vote, computed by `_k_jvote`. That kernel is pc_fused's `_k_vote` line for line,
  with the same tiles, loop, beta terms, 4 warps and num_stages. It runs only on rows whose decisions differ, and it
  stores the 4,096 scores.
- e(y) is the sum of z_j over the set bits of y, in increasing j, starting from +0.0, with one fp32 add per step.
- m(y) = w32 e(y) is one IEEE multiply rounded to nearest. It is written in inline PTX (`mul.rn.f32`) so that no compiler
  can fuse it into an FMA. w32 = float32(w).
- t(y) = s_N(y) + m(y) is one fp32 add.
- The answer is the lowest class index among the maxima of t.

`joint_ref.joint_np` (numpy only) computes the same answer bit for bit, and `joint_ref.a3_f64` is ARB's float64 rule.

**d = 13** (C5 = S -> B at p = 0.6 % and 0.35 %). On a routed shot where the decisions differ, the answer is 1 if
v = c + w32 z > 0, 0 if v < 0, and Cascade's answer if v = 0. Here c is S's logit and z is Cascade's logit in fp32. The
product is the same inline `mul.rn.f32` and the sum is one fp32 add. `sv_joint.joint_np` reproduces it.

## 2. What changed

| file | change |
|---|---|
| `cascade/bb72/cascade_compare/phase_d/joint.py` (new) | Kernels `_k_jvote`, `_k_jtop` and `_mul_rn`. `Step` is the joint step on fixed buffers. `JointB` captures Cascade's computation with the logits kept (`make_fwd_z`) and the joint step in one CUDA graph, behind pc_speed.Graph's interface. `GraphZ` and `compiled_z` give Cascade's compiled path with the logits kept, for the reads. `fused_engine`, `build_b`, `joint_scores_gpu` and `scores_gpu` are helpers. `record(w)` gives the joint record that each pipeline row carries. |
| `.../phase_d/joint_ref.py` (new) | The numpy references `joint_np` (fp32, bit for bit) and `a3_f64` (ARB), with `w32_of`. |
| `.../batchopt/m2.py` | `pipeline --joint-w W --save F` (an S -> B configuration with tail-b 0). In single mode, B is a `JointB` that reads S's static buffers (its pool output s and its decision bits) in place, so a routed shot still costs one replay. In batched mode, `Stream.run` gathers S's s rows and decision bits of the routed rows right after S's replay and queues them with the detector rows; `Stream.run_b` copies them into the `JointB` before each replay at 4,096. `--save` writes the shots, every decision and every route after the clock. Without `--joint-w` nothing changes. `d1_m2.py` passes the options through unchanged. |
| `.../phase_d/d1_compiled.py` | `read --joint`. Multi (compiled@1) and compiled@4096 run `joint.compiled_z`. The existing bitwise check against phase C's compiled@1 Graph (which has no logits) is unchanged. A new check compares the logit-keeping compiled@4096 graph with phase C's on the first slice (`z_vs_frozen`). The npz gains `z1R` (compiled@1's logits on the routed rows) and `zR` (compiled@4096's logits on the routed queue). `--dry` is for checks only: it admits kinds 13 and 12 and skips the S anchor. |
| `.../phase_d/joint_read.py` (new) | The eager read: pc_eval unchanged, plus C14''s float32 logits for every shot (`p<p>_z32.npy`, ARB's arb_read.py). |
| `.../phase_d/scored.py` (new) | ARB's `arb_fused.Scored`, copied unchanged. It is pc_fused.Batched plus one store of the class scores in the vote. |
| `.../phase_d/joint_scores.py` (new) | S55'-deep's class scores on the union of every path's disagreement rows (eager, compiled@1, compiled@4096 queue), from Scored rerun on the saved detector rows. It keeps ARB's bitwise checks and adds one: `_k_jvote` on Scored's pool output gives Scored's scores bit for bit. |
| `.../phase_d/joint_count.py` (new) | `count` gives per path the failures of S alone, C14' alone, the composition and the joint composition, plus the joint rows, changes, repairs, breaks and third answers, over all shots and by chunk-parity half. `compare` is the bitwise check of a timed pipeline's saved decisions against the read on its path. |
| `cascade/d13/surface/sv_joint.py` (new) | Kernel `_k_j13`. `JointB13` captures B's computation and the joint step in one CUDA graph on its own capture stream (single mode reads S's logit and decision in place). Also `joint_np`, `assemble_joint` (C5 from the reads with the joint step), `check` (one pipeline file against the reads), `count` (failures from the reads) and `record`. |
| `cascade/d13/surface/sv_speed.py` | `pipeline --joint-w W` (C5 only). In batched mode, S's logits and decisions queue with the routed rows and `run_b` copies them into `JointB13`. `bitwise --joint-w P=W,P=W` assembles C5 with `assemble_joint`. The lines that srt_d13.py anchors its clock window on are unchanged and each still occurs once. |
| `speed_rt/cascade/drive.py` | `--joint-w P=W,...` adds `--joint-w W --save <attempt>/pipe.npz` to every R1 S -> B slot. `--joint-w13 P=W,...` adds `--joint-w W` to every R4 C5 slot. |
| `speed_rt/cascade/srt_d13.py` | `session` now works on either card, where before only `h100-session` existed. `joint-lock` derives the study's lock for the joint tree: it takes the session's code digest and kernel caches, plus the card and venv fields on the H100. `SRT_KCACHE_BUILD=1` (dry only) now also applies on the RTX. |
| `speed_rt/cascade/srt_cas.py` | `joint_read.py` can run under the launcher's beacon seeds, to read the timing stream's own shots. |

The Cascade path of the composition with the joint step is Cascade's compiled graph with the logits as a third output.
Cascade alone (`--config -1`) still runs the frozen graph.

## 3. Checks (outputs in `checks/out/`)

### (a) ARB's disagreement rows: fp32 against ARB's float64

`checks/check_arb.py cpu` ran `joint_ref.joint_np` on every row of `~/work/arb/out/analysis/disagree_<p>.npz` with
`~/work/arb/out/read/p<p>_scores.npy`, at ARB's fitted w. It ran on the Linux workstation, niced, on one thread, taking 0.6 s and
5.9 s. The scores (1.7 GB) could not reach the pod, because the workstation's upload to it ran at 30 to 70 KB/s per stream.

| p | w (w32) | rows | fp32 != ARB's float64 | fp32 ties | smallest top-2 gap of t |
|---|---|---|---|---|---|
| 0.2 % | 6.309573444801933 (0x40c9e807) | 9,037 | **0** | 0 | 7.0e-4 |
| 0.3 % | 3.758374042884441 (0x40708933) | 94,184 | **0** | 0 | 3.8e-5 |

`check_arb.py gpu` then ran the GPU kernel `_k_jtop` on a subset of ARB's saved scores on the pod. The subset was 95 and 96
rows: the 64 rows with the smallest top-2 gap, 32 rows at random, and any row that differed (there were none). GPU against
numpy: 0 differences. GPU against ARB: 0 differences, at both p. (`out/check_arb_*.json`, `pod/out/arb/`.)

### (b) Timed pipelines against the reads on the same path

Each check sampled 500,000 dry shots (kind 13, chunks 0-49; single mode used the first 20,000) at each p. The steps were:
the pipelines with and without the joint step; the eager read with the logits; the compiled read with the logits;
the score export; and `joint_count.py compare` for every pipeline (`pod/run_bb.sh`, `pod/run_d13.sh`).

**BB72**, with w = ARB's w (`pod/out/bb/p*/pipe_*_compare.json`):

| p | mode (path) | S engine | shots | routes differing | decisions differing | joint rows | changed by the joint step | failures: pipeline / read with the joint step / read, composition |
|---|---|---|---|---|---|---|---|---|
| 0.2 % | single (compiled@1) | base, new | 20,000 | 0 | **0** | 15 | 3 | 12 / 12 / 14 |
| 0.2 % | batched (compiled@4096) | base, new | 500,000 | 0 | **0** | 392 | 128 | 285 / 285 / 343 |
| 0.3 % | single (compiled@1) | base, new | 20,000 | 0 | **0** | 112 | 25 | 71 / 71 / 79 |
| 0.3 % | batched (compiled@4096) | base, new | 500,000 | 0 | **0** | 2,706 | 785 | 1,924 / 1,924 / 2,218 |

The detector rows of every pipeline equal the read's. The pipelines without the joint step fail exactly as the read's
composition does (14, 343, 79 and 2,218). Other checks, all bitwise and all passing at both p:
- The score export: on 397 and 2,722 rows, every one of ARB's checks passes, and `_k_jvote`'s scores equal Scored's.
- Cascade with the logits kept, against phase C's frozen graphs: compiled@1 against phase C's compiled@1 Graph on the
  first 20,000 rows and on every 1,000th row; compiled@4096 against phase C's compiled@4096 Graph on all 500,000 rows.
  0 decisions and 0 confidences differ, so keeping the logits changes nothing that Cascade decides.
- The eager read's z32: float16 of it equals the read's z, and min |z| equals conf bitwise.
- S's in-process check against C4b's read (regenerated on the pod, 2,000 and 32,768 shots): 0 decisions and 0 margins
  differ. The regenerated read gives the same failure counts as the release's record of C4b
  (`phase_c/out/check/depspam_fm/read/rows_0.002,0.003.json`: C14' 444 and S55'-deep 637 at 0.3 %; S55'-deep 94 at
  0.2 %).

Per-path counts on these 500,000 dry shots (`joint_count.py count`; exploratory, not for claims):

| p | path | composition | with the joint step | repairs / breaks | third answers |
|---|---|---|---|---|---|
| 0.2 % | eager | 343 | 289 | 55 / 1 | 56 |
| 0.2 % | compiled@1 | 343 | 286 | 58 / 1 | 54 |
| 0.2 % | compiled@4096 | 343 | 285 | 59 / 1 | 53 |
| 0.3 % | eager | 2,215 | 1,924 | 311 / 20 | 349 |
| 0.3 % | compiled@1 | 2,216 | 1,923 | 311 / 18 | 343 |
| 0.3 % | compiled@4096 | 2,218 | 1,924 | 312 / 18 | 346 |

**d = 13** with w = 1.0, a placeholder: the parallel fit at d = 13 (ARB2) was not ready (`~/work/arb2/RESULTS.md` still had
`@@SUMMARY@@`). Its failure counts mean nothing; it only exercises the code. The pipeline was C5 under the pod's dry lock
(`podrun13.py --derive`: the study's lock with the pod's card, venv and code identity, dry kind 13). The reads were
`sv_read --dry --stage speed` (S1 and B1 on chunks 0-1; SB and BQ5 on chunks 0-49). The check was `sv_joint check`
(`pod/out/d13/p*/check_*.json`):

| p | mode | routed | joint rows | changed | decisions / stages / S logits / B logits differing | GPU against numpy |
|---|---|---|---|---|---|---|
| 0.6 % | single | 1,974 | 110 | 39 | **0 / 0 / 0 / 0** | 0 |
| 0.6 % | batched | 49,449 | 3,132 | 1,174 | **0 / 0 / 0 / 0** | 0 |
| 0.35 % | single | 183 | 2 | 1 | **0 / 0 / 0 / 0** | 0 |
| 0.35 % | batched | 5,035 | 50 | 22 | **0 / 0 / 0 / 0** | 0 |

The d = 13 lock derivation also ran on the pod: `srt_d13.py session` and `joint-lock` with SRT_CARD=h100 wrote
`pod/out/d13/srt_joint_lock.json`.

## 4. Smoke times (H100, one process each, not registered, no claims)

Rows from `pod/out/bb/p*/rows_{joint,base}.jsonl` and `pod/out/d13/p*/rows_*.jsonl`. Single mode reports the mean and
p99.9 latency in µs; batched mode reports µs per shot. The S engines are D1-S's base and new engines.

| decoder | p | mode | without the joint step | with it | routed |
|---|---|---|---|---|---|
| BB72, base | 0.2 % | single | 191.61 / 2139.2 | 192.47 / 2153.9 | 1.9 % |
| BB72, new | 0.2 % | single | 160.08 / 2156.6 | 159.09 / 2121.8 | 1.9 % |
| BB72, base | 0.2 % | batched | 10.1030 | 10.1322 | 2.0 % |
| BB72, new | 0.2 % | batched | 5.1312 | 5.1336 | 2.0 % |
| BB72, base | 0.3 % | single | 357.89 / 2159.5 | 358.55 / 2174.0 | 10.1 % |
| BB72, new | 0.3 % | single | 328.60 / 2140.2 | 326.26 / 2146.7 | 10.1 % |
| BB72, base | 0.3 % | batched | 13.3896 | 13.4063 | 10.0 % |
| BB72, new | 0.3 % | batched | 8.4007 | 8.4175 | 10.0 % |
| d = 13 C5 | 0.6 % | single (reps 2 and 3) | 126.30, 126.36 / 578.9, 577.4 | 127.26, 126.78 / 581.1, 580.1 | 9.9 % |
| d = 13 C5 | 0.6 % | batched | 26.6103 | 26.8731 | 9.9 % |
| d = 13 C5 | 0.35 % | single | 82.37 / 571.6 | 83.69 / 579.2 | 0.9 % |
| d = 13 C5 | 0.35 % | batched | 16.3820 | 16.4075 | 1.0 % |

Single-mode means differ by less than one process's noise. A routed shot costs about 2.1 ms on BB72 (Cascade's batch-1
path on this H100) and about 0.57 ms at d = 13. `checks/bench_joint.py` (`pod/out/bench/`) therefore measured the joint
step directly, with the same inputs, on graphs with and without it.

**Added time per routed shot:**
- BB72 single: +3.3 µs (p = 0.2 % run) and +4.0 µs (0.3 % run) of GPU time per replay of Cascade's compiled@1 graph
  (1.828 ms without the joint step), with the decisions differing. Host-measured round trips of the two graphs differ by
  less than their noise (±10 µs).
- BB72 batched, on one replay of 4,096 rows (166.5 and 166.9 ms without the joint step): +0.07 to +0.11 µs per queued
  row when no decision differs (the kernels' launch and early exit), +0.15 µs with 4.1 % of the rows differing, +0.14 µs
  with 5.5 %, and +0.6 µs when every row differs. In the dry shots, 3.9 % (0.2 %) and 5.4 % (0.3 %) of the routed rows
  differ.
- d = 13 single, from the pipelines at 0.6 %: +3.6 µs on the mean via-B latency of two replicate pairs (566.0 -> 569.7
  µs; medians 565.6 -> 569.2). The first pair (534.7 -> 557.3 µs) is left out, because its base process's median, 512
  µs, is far from every other process's. At 0.35 % (one pair, 183 routed shots) the median went from 566.3 to 570.0 µs,
  while the mean went from 567.8 to 594.9 µs, carried by one 4.8 ms shot. In one process, the graph pair differed by
  -2.5 ± 3 µs, so nothing was measurable there.
- d = 13 batched (b* = 16): within ±0.5 µs per routed shot in one process. End to end, the B stage took +1.0 µs per
  routed shot at 0.6 % (6.111 -> 6.162 s over 49,449 shots) and +1.1 µs at 0.35 % (0.644 -> 0.650 s over 5,035).

**Added time per incoming shot** (the added time per routed shot times the routed fraction, or measured end to end):
- BB72 single: about 0.06 µs at 0.2 % (3.3 µs × 1.9 %) and 0.4 µs at 0.3 % (4.0 µs × 10.1 %).
- BB72 batched: +0.029 / +0.002 µs at 0.2 % (base / new engine, end to end), and +0.017 / +0.017 µs at 0.3 %.
- d = 13 single: about 0.4 µs at 0.6 % and 0.03 µs at 0.35 %.
- d = 13 batched: +0.26 µs at 0.6 % and +0.03 µs at 0.35 % (end to end, one process each). The B stage accounts for
  +0.10 µs of the 0.26; the rest is outside it and was not replicated.

## 5. Commands for the registered confirmation

`T` is the D1 tree with the patch applied (the patch's `cascade/bb72/X` is `$T/X`). `T13` is the d = 13 tree with
`cascade/d13/surface` patched, and `SRT_ROOT` is the public tree with `speed_rt/cascade` patched. Every `analysis_cas.py`
FREEZE pin of these trees must be renewed by the registration (see section 6). The weights below are ARB's fit; use the
frozen values. `W6` and `W35` are the frozen d = 13 weights.

```
WB="0.002=6.309573444801933,0.003=3.758374042884441"      # BB72, per p
W13="0.006=W6,0.0035=W35"                                  # d = 13, per p
```

**Accuracy reads, BB72** (RTX PRO 6000, since pc_common's guard admits only that card; the environment is D1's
gpu.sbatch: `cd $T/ising_compare`, `PYTHONPATH=$T/ising_compare:$T/qldpc_neural:$PC:$PD:$BO`, `QND_RELAY`, the job's
caches). `SHIM.py` is the confirmation's seed shim for its fresh shots (ARB's arb_shim.py with new sets), and `C0:C1` its
chunks. Per p, with W its weight:
```
python SHIM.py $PD/joint_read.py --family depspam_fm --ckpt $C14P,$S55PD --prec bf16 --kind 14 --p P \
    --chunks C0:C1 --procs 8 --save R --save-det --out R/rows_P.json          # eager + z32
python $PD/d1_compiled.py read --read R --p P --out CR --joint                 # compiled@1, compiled@4096 + logits
python $PD/joint_scores.py --read R --compiled CR --p P --ckpt $S55PD         # class scores, all three paths
python $PD/joint_count.py count --read R --compiled CR --p P --w W --out count_P.json --answers answers_P.npz
```

**Accuracy reads, d = 13.** sv_read is unchanged. Its outputs already hold S's c and Cascade's z per shot on each path
(S1, SB, B1, Bb, BQ1, BQ5). The reads run under the joint lock below, because the code digest changed. Count with
`python -m surface.sv_joint count --lock LOCK --read READ --p P --w W --out F.json`.

**Timing R1 with the joint step**, on the RTX PRO 6000 (Alpine):
```
SRT_ROOT=... D1_TREE=$T PY=... RELAY_ROOT=... TRITON_SEED=... INDUCTOR_SEED=... SRT_SEEDS=... SRT_BEACON=... OUT=... \
DRIVE_ARGS="--joint-w $WB" sbatch $SRT_ROOT/speed_rt/cascade/r1.sh
```
On the H100: `h100_pod.sh run r1` with `D1_TREE=$T` and `DRIVE_ARGS="--joint-w $WB"`. The seed caches lack the new
kernels and graphs, which compile before the clock starts.

**Timing R4 with the joint step** (C5 slots). The same steps on each card, with SRT_CARD=rtx or h100 and `cd $T13`. Each
derived lock is written once:
```
python $SRT_ROOT/speed_rt/cascade/srt_d13.py session --out S0.json
python $SRT_ROOT/speed_rt/cascade/srt_d13.py joint-lock --lock $D13_LOCK --session S0.json --out L0.json
SRT_DRY=1 SRT_KCACHE_BUILD=1 python $SRT_ROOT/speed_rt/cascade/drive.py r4 --tree $T13 --out DRYOUT --lock L0.json \
    --joint-w13 "$W13" --reps 1 --n-single 2000 --n-batched 20000 --gate off     # untimed cache build (dry seeds)
python $SRT_ROOT/speed_rt/cascade/srt_d13.py session --out S1.json                 # after the build
python $SRT_ROOT/speed_rt/cascade/srt_d13.py joint-lock --lock $D13_LOCK --session S1.json --out L1.json
D13_LOCK=L1.json DRIVE_ARGS="--joint-w13 $W13" sbatch $SRT_ROOT/speed_rt/cascade/r4.sh        # RTX
SRT_SESSION=S1.json drive.py r4 --card h100 --tree $T13 --out OUT --lock L1.json --joint-w13 "$W13"   # H100
```

**Bitwise check of the timed pipelines**, on each card:
- BB72: read the timing stream's own shots under the round's beacon seeds, then compare every R1 S -> B attempt:
  ```
  python $SRT_ROOT/speed_rt/cascade/srt_cas.py $PD/joint_read.py --family depspam_fm --ckpt $C14P,$S55PD --prec bf16 \
      --kind 12 --p P --chunks 0:50 --procs 8 --save TR --save-det --out TR/rows.json
  python $PD/d1_compiled.py read --read TR --p P --out TCR --joint --dry
  python $PD/joint_scores.py --read TR --compiled TCR --p P --ckpt $S55PD
  python $PD/joint_count.py compare --pipe OUT/slots/<slot>/a<k>/pipe.npz --read TR --compiled TCR --p P --w W --out F.json
  ```
- d = 13: `sv_read --stage speed` reads of chunks 0-49 under L1, then
  `python -m surface.sv_speed bitwise --lock L1.json --expect c7 --pipes PIPES --read READ --joint-w "$W13" --out F.json`.
- Before any registered run, the dry pre-check of section 3(b) on the card: `pod/run_bb.sh P W` and `pod/run_d13.sh P W`
  (here through `podrun.py` / `podrun13.py`; on Alpine the cluster paths need neither).

## 6. Not done, and caveats

- **Card.** Every check and smoke time is from an H100 80GB HBM3, not an RTX PRO 6000. The cards' guards were lifted
  on the pod and recorded: `pod/podrun.py` sets `pc_common.GPU` to the pod's card, and `pod/podrun13.py` sets
  `sv_common.GPU`. The registered timing needs both cards.
- **Clock flag.** The release's clock monitor (`bb/timing.py`) marked every BB72 smoke row as not accepted. The pod
  environment did not set `BB_PIDNS=1` (drive.py sets it on the H100), so the process's own host PID counted as another
  compute app. The batched rows also saw the SM clock drop to 93-96 % of its maximum, below the 97 % bar. The d = 13
  rows carry no such flag.
- **Weights on the pod.** The pod read the release's re-saved weights. `podrun.py` maps their identity to the
  `source_sha256` each one carries, which is the D1 lock's sha256, after checking the file against
  `weights/MANIFEST.sha256`. sv_common.load_B already does the same for CS13.
- **Pod paths.** C4b's read was regenerated on the pod (kind 12, chunks 10-19), because its npz is not in the release.
  The compiled read ran with `--dry`, so without the S anchor.
- **d = 13 on the pod.** Kernel caches were not pinned (`podrun13.py`), and the lock was the pod's derived dry lock.
  w = 1.0 was a placeholder.
- **Not run on a GPU.** These are not part of the passing checks above:
  - Four edits came after the pod checks: d1_compiled's `--dry` admitting kind 12, srt_cas's `joint_read` entry,
    sv_speed `bitwise --joint-w` taking a weight per p, and `sv_joint count`. The pod ran joint.py and joint_ref.py at
    digest `fba4c42d49c4f548` (unchanged since) and sv_joint.py at `615166f3d132ebe1` (before `count` was added).
  - The full-stage `sv_speed bitwise` path was not run; the check used `sv_joint check` per file.
  - drive.py's joint plans were only listed (`--list`), and no SPEED-RT job ran.
  - `h100_pod.sh` still calls `h100-lock`, which refuses the joint tree's code digest. On the H100, use `session` and
    `joint-lock` as in section 5.
- **Analysis.** `speed_rt/cascade/analysis_cas.py` was not changed. Its FREEZE pins the unchanged trees, and its check of
  a derived lock allows only `D13_REPLACED`, while the joint lock also replaces `identity.code`. The registration must
  renew both.
- **Not done.** No registered test, no read of fresh registered shots, and no Relay-BP configuration: the joint step is
  refused for S -> B -> R and for tail-b > 0.
