# RETEST: the paper's BB, Cascade and output-head tests on fresh shots

Registered by this commit (the freeze). GitHub's server record of the push that brought this commit to GitHub fixes
its public time; the test shots' seeds come from the first drand round at least 600 s after that time, so nothing
below can have been chosen with the re-test's shots in view.

## 0. What this study is

A fresh-shot confirmation of frozen tests. Every test listed in section 3 is run again with the networks, thresholds,
decoders, decoder settings, sizes, statistics and analysis code that were frozen for its original run, on new shots.
Nothing is trained, selected, tuned or calibrated again; the study confirms (or fails to confirm) the original tests'
accuracy verdicts conditional on that frozen training and selection. It is not a reproduction of every conclusion of
the studies that produced the networks.

The confirmatory part is accuracy. No time is measured again: every time comes from the registered timing session of
the original test, and the only time input that depends on the test shots, the fraction of shots routed to a CPU
fallback, is taken from the re-test (section 4). Every re-test speed number is therefore an estimate from the
registered timing sessions, labelled as such.

The tests (section 3), with the records of their original runs in this repository and the research hub:

| test | paper | original records | re-test sets |
|---|---|---|---|
| BB72 at p = 0.3 % and 0.2 %: fast, balanced and accurate modes, the second network alone, sector Relay-BP, BP-OSD (ldpc), BP-OSD (cudaq-qec), full-DEM Relay-BP | Table I | `bb/bb72/` (stage 3, amendment S) | 200,000 and 300,000 shots |
| gross code at p = 0.4 %: the same modes and opponents | Table I | `bb/gross_stage4/` (stage 4) | 150,000 shots |
| Cascade on BB72 (follow-up D1, fitted noise depspam_fm) | Table cascade | `cascade/`, `RELEASE_NOTES_cascade.md`, hub `ising-cascade-d1` | 11,000,000 and 17,000,000 shots |
| Cascade on the surface code at d = 13 | Table cascade | hub `ising-cascade-surface` (lock bddd3f91) | 1,000,000 and 30,000,000 shots |
| output-head comparison (A1), every arm and seed | Tables head, headcmp | hub `ising-rev-a1` (freeze 2e69738e) | 500,000 BB72 and 1,000,000 surface d = 9 shots |

Not repeated: the earlier gross test G-4 (stage 3; the paper's appendix cites it as an earlier test), which stays a
historical result; its 0.5 % hybrid is reported on the re-test's gross shots as stage 4 reported it on its own.

## 1. Procedure

1. This commit is pushed to the hub and, after an independent review of the shims and replay checks (section 6), to
   GitHub. The time of that push comes from GitHub's server records (`retest/github_push_activity.json`,
   committed with the results).
2. Beacon: `python retest/beacon.py --push-time T --commit C --out retest/run/beacon.json` on Alpine's login node:
   the first drand mainnet round (chain 8990e7a9..., 30 s period) whose time is at least 600 s after T, fetched from
   at least two of four endpoints that agree on round, signature and previous signature; R = SHA-256(signature).
   Stage 3's code does the fetch and the checks (`bb/bb72/ising_compare/icmp/t2.py`).
3. Seeds: `python retest/seeds.py retest/run/beacon.json retest/run/seeds.json` (section 2), then
   `python retest/seeds.py --verify retest/run/beacon.json retest/run/seeds.json`, which recomputes the whole file
   from the beacon, the TAG table and the historical seeds and must report it equal (`retest/jobs/run_beacon.sh`
   runs both; the report goes into commit 1).
4. Draws: the BB sets on Alpine (`retest/jobs/bb_draw.sbatch`, after a host check) and the head sets
   (`retest/jobs/head_draw.sh`); the Cascade reads draw their shots inside the read from the seeds file, as their
   originals did.
5. Commit 1, before any read: the beacon, the seeds, the draw manifests and host checks.
6. Reads: every decoder reads every shot of its sets once, blind (no failure count is printed or recorded before the
   analysis), where section 3 says.
7. Analyses: each test's frozen analysis runs once, from the committed reads, as soon as its reads are complete; its
   outputs are committed as they come out.
8. After the confirmatory analyses: the exploratory X2 adapter (section 3.2), reported apart.

The plan was reviewed before this freeze. The review's ten findings, cited by number below, and where this document
takes each:

| finding | what it asked | where |
|---|---|---|
| 1 | reused timings give speed estimates, not a speed replication | section 4: timing is not re-measured, every re-test speed number is an estimate; accuracy is the confirmatory part |
| 2 | the unchanged analyses read the old CPU terms | section 4: a frozen adapter from the re-test's routed fraction, derived rows beside the original rows |
| 3 | no acceptance rule for CUDA-Q BP-OSD on another GPU | section 3.1: cudaq-qec runs on an H100 80GB HBM3, the original platform, in the original environment |
| 4 | D1's read guard must not be dropped whole | section 3.2: every check kept except the obsolete kind-14 freshness test |
| 5 | beacon indices are not native chunk indices | sections 2, 3.2, 3.4: head BB72 keeps native chunks 0-49 (beacon indices 20-69); D1 maps c0 + j to j |
| 6 | different TAGs can still give equal seeds; the freeze time must be checkable | section 2: the collision rule and the check against 7,034 historical seeds; section 1: GitHub's push record |
| 7 | X2 hard-codes the original inputs and counts | section 3.2: an adapter reads them from the committed re-test records; reported apart, exploratory |
| 8 | freeze the verdict-to-paper mapping | section 5 and each test's mapping in sections 3.2-3.4 |
| 9 | say what the study is | section 0: a fresh-shot confirmation of the frozen networks, thresholds and analyses, timing inherited |
| 10 | replay checks of the shims before the freeze | section 6 and each test's replay checks |

## 2. Seeds

- Seed of beacon index j under TAG: 2^62 + (first 8 bytes, big-endian, of SHA-256(R || TAG || j as 2 bytes
  big-endian)) mod 2^62, stage 3's formula (`bb.shots.test_seed`). Every chunk is 10,000 shots from one seed, as in
  every original test.
- One TAG per code and error rate; a different noise model counts as a different code (`retest/tags.json`):

| TAG (bytes, newline included) | set | native chunks | beacon indices |
|---|---|---|---|
| `convlines retest bb72 p0.003\n` | BB72-3 (Table I) | 0-19 | 0-19 |
| | head-bb72 (head comparison, BB72 test) | 0-49 (A1's native chunk numbers kept) | 20-69 |
| `convlines retest bb72 p0.002\n` | BB72-2 (Table I) | 0-29 | 0-29 |
| `convlines retest bb144 p0.004\n` | gross (Table I) | 0-14 | 0-14 |
| `convlines retest bb72 depspam_fm p0.002\n` | d1-0.002 (Cascade BB72) | c0 + 0 .. c0 + 1,099 (c0 = 6,433,102, D1's own) | 0-1,099 |
| `convlines retest bb72 depspam_fm p0.003\n` | d1-0.003 | c0 + 0 .. c0 + 1,699 | 0-1,699 |
| `convlines retest surface d13 p0.006\n` | d13-0.006 (Cascade d = 13) | 0-99 | 0-99 |
| `convlines retest surface d13 p0.0035\n` | d13-0.0035 | 0-2,999 | 0-2,999 |
| `convlines retest surface d9 p0.006\n` | head-surface (head comparison, surface test) | 0-99 | 0-99 |

  The beacon indices are those of the table unless the collision rule drops one.
- Collision rule (frozen): the sets of one TAG take their chunks in order of their first index; each chunk takes the
  smallest index no earlier chunk took; an index whose seed equals a seed already taken or a historical seed is
  dropped (recorded in `seeds.json`) and the next index is tried. Historical seeds (`retest/historical_seeds.txt`):
  every 19-digit integer in [2^62, 2^63) found in any manifest, seeds or beacon file on every branch of the research
  hub, in the release, in this account's records on Alpine (919 seeds, the same in the hub and on Alpine) and in the
  re-test's own dry-run and replay records (6,115 more): 7,034 seeds, sha256 f2c7eb444b6b418e... (the scans and their
  counts: `retest/replay/hist/`). Seeds from a TAG lie in [2^62, 2^63); the arithmetic
  seed rules of D1, d = 13 and A1 give seeds below 2^57, so they cannot collide with these.
- TAGs used before, all different from these: `ising_compare study 2 T2`, `study 3 T3`, `study 4 T4`,
  `ising_compare bb test BB72-3`, `bb test BB72-2`, `bb test G-4`, `bb test4 G-4R G-4F`.
- The shims read the seeds through `retest/seeds.py chunk_seeds(set)`, which recomputes every seed from the beacon
  record and refuses a seeds file that differs.

## 3. The tests

### 3.1 Table I: BB72 (stage 3, amendment S) and the gross code (stage 4)

Code: the repository's `bb/bb72/` and `bb/gross_stage4/` trees at this commit, unchanged, run through
`retest/shim/rt_bb.py`. The shim changes, each only when its environment variable is set: `bb.shots.seed_of` on the
test splits (chunk j of BB72-3, BB72-2 and gross gets `chunk_seeds(set)[j]`; every other split and code goes to the
frozen function, a test split without a re-test set is refused), the shot manifest that `shots.load` checks every
chunk against (the re-test's draw manifest), and the frozen-parameter files the analysis reads (the re-test's copies,
below). The splits keep their names ("test" in `bb/bb72/`, "test4" in `bb/gross_stage4/`), so every frozen command, record
and analysis reads them unchanged; shots, predictions and outputs live in their own directories.

Frozen and unchanged:

| item | BB72 | gross |
|---|---|---|
| networks | `weights/bb/r11_final_ema.pt` 58e60897c3f9302c2014e82257d0d08f80a6e8386d2dd5216a093230990b252f, `r12_final_ema.pt` f2821c327b4e233a4dd1a75c465ceb261448d981c2b55166862fc34a99ec3e96; fused engine, batch 1, the policies `data/bb/policies/bb72_p0.003_rep1.json` (r11), `bb72_p0.00{3,2}_rep2.json` (r12) | `weights/bb/g2_D12k113c10_final_ema.pt` eb380295a6eccfde022d8e61ad1c3f2b9154d54b3c04cc5eea84e3f91c224692, fusedt engine (`cell.json` "engine") |
| thresholds | BB72-3F and 3S: margin < 3.6329734325408936; BB72-2F and 2S: margin < 2.979396343231201 | G-4R: margin < 2.2055678367614746; G-4F: margin < 6.272581100463867; beside, G-4's old q: 0.5653958916664124 |
| fallbacks | F: full-DEM Relay-BP; S: sector Relay-BP | G-4R: sector Relay-BP; G-4F: full-DEM Relay-BP |
| opponents | sector Relay-BP and full-DEM Relay-BP (relay-bp 0.2.2, IBM's tuned gross-code parameters: pre_iter 80, 300 sets x 60 iterations, stop after 5 solutions), BP-OSD min-sum ms80 OSD-CS 60 (ldpc 2.4.1, X sector), cudaq-qec 0.8.0 nv-qldpc-decoder ms80 OSD-CS 60 (`bb.opp`, `bb.gpu_bposd`) | the same |
| cells and statistics | `data/bb/stage3/cells.json`, `amendment_S.json`: per comparison exact two-sided McNemar p < 0.05/3 and the paired Newcombe interval (method 10) at 1 - 0.05/3 entirely below 0; a cell's claim needs every comparison "better" and its speed condition | `data/bb/stage4_gross/cell.json`, the same rule |
| analysis | `python -m bb.stage3 analyze`, then `... analyze --amend amendment_S.json` | `python -m bb.stage4_gross analyze` |

Reads (each once, blind):

| read | where | why there |
|---|---|---|
| draw of BB72-3, BB72-2, gross | Alpine CPU node (`bb/venv`, Stim 1.16.0), after a host check: the original test sets regenerated on the node from their published beacons equal the published arrays (stage 3 and 4's host check) | Alpine produced every calibration and development chunk |
| r11, r12 | Alpine, one full RTX PRO 6000 (`bb/venv_cu128`: torch 2.11.0+cu128, cuDNN 91900); fallback: the rented H100 in stage 3's environment with stage 3's commands (`pod_bb72net.sh`) | the GPU of the policies' own engine signature; stage 3 read on an H100 as a disclosed substitute (H100 = RTX on all 800,000 calibration shots, bit for bit) |
| g2_D12k113c10 on fusedt | a rented H100 80GB HBM3, stage 4's environment rebuilt from `stage4_gross/pod/env.lock` | the frozen engine signature names this GPU and environment exactly |
| cudaq-qec | the same H100; gross: stage 4's environment from `cqvenv.lock`; BB72: stage 3's environment (`s3venv`, rebuilt by `pod_s3venv.sh` from stage 3's pod record, the environment BB72's original cudaq-qec reads ran in). Both hold cudaq-qec 0.8.0, cuda-quantum 0.16.0.post1, numpy 2.2.6 and Stim 1.16.0; their CUDA libraries are 12.9 (stage 4) and 12.8 (stage 3) | each code's original platform and environment (cudaq-qec's decisions vary between processes, so no cross-GPU check is possible); `pod_reads.sh` reads only on an NVIDIA H100 80GB HBM3 and only after each environment equals its record package for package |
| sector Relay-BP, BP-OSD, full-DEM Relay-BP, BB72 | Alpine CPU nodes, `bb/venv` (relay-bp 0.2.2, ldpc 2.4.1) | stage 3 read on an Apple M4 and a Ryzen workstation; accuracy does not depend on the host |
| sector Relay-BP, BP-OSD, full-DEM Relay-BP, gross | Derecho, regular main queue (stage 4: the develop queue), the qldpc venv (relay-bp 0.2.2, ldpc 2.4.1) | stage 4's platform |

`bb.opp` keeps one decoder per process, so a Relay-BP decision can depend on which shots the process decoded before;
the process layout was never part of the frozen protocol and differs from the original's.

Run book (every command in `retest/jobs/`, as run):

| step | command | where |
|---|---|---|
| draw and BB72 reads | `bash retest/jobs/bb_submit.sh` (`bb_draw.sbatch`, then `bb72_net.sbatch` and two `bb_cpu.sbatch` jobs after it) | Alpine |
| gross full-DEM, sector and BP-OSD | the code (this commit's `bb/gross_stage4/` and `retest/`) and the draw's gross shots with their manifest copied to Derecho; three `gross_full_derecho.pbs` jobs (chunks 0-4 with sector Relay-BP and BP-OSD on all 15 chunks first, 5-9, 10-14); the per-chunk predictions copied back to Alpine | Derecho |
| gross network, cudaq-qec | the code (`bb/bb72/`, `bb/gross_stage4/`, `weights/bb/`, `retest/`) and the draw's BB72 and gross shots with their manifests copied to the pod; `bash pod_setup.sh` and `bash pod_s3venv.sh` (unless the pod's environments were built from the same records and checked), `bash pod_reads.sh run`; the predictions copied back to Alpine | rented H100 |
| analyses | `sbatch --export=ALL,WHICH=bb72 retest/jobs/bb_analysis.sbatch`, then `WHICH=gross`, each once its reads are complete | Alpine |

Fallback (frozen, decided on start times only, never on any output): a Derecho job that has not started 90 minutes
after its submission is deleted (`qdel` by its id) and its chunk range is read on Alpine
with the same command (`gross_full_alpine.sbatch`); a read is never run twice on one chunk range. A GPU read whose
job has not started within an hour moves to a rented H100 where its frozen engine allows that GPU. Nothing is
submitted to Derecho and nothing is deleted there between 00:35 and 01:00 PDT: no `qsub` (also not of a cut read
submitted again) and no `qdel`.

Analysis, once per tree, after every read of the tree: `retest/jobs/bb_analysis.sbatch` (WHICH=bb72, WHICH=gross):

    python retest/shim/rt_bb.py --prepare OUT --gpu "NVIDIA RTX PRO 6000 Blackwell Server Edition"   (bb72; gross: no --gpu)
    python retest/shim/rt_bb.py bb.stage3 analyze --out OUT
    python retest/shim/rt_bb.py bb.stage3 analyze --amend OUT/amendment_S.json --out OUT
    python retest/shim/rt_bb.py bb.stage4_gross analyze --out OUT4

`--prepare` writes the re-test's frozen-parameter copies: the tree's `cells.json`, `amendment_S.json` (bb72) or
`cell.json` (gross) with exactly two changes, the CPU-term rows file (the derived rows of section 4) and, for BB72,
the field the analysis compares with every network read's engine signature ("gpu") set to the re-test's read GPU, the
frozen value kept as "gpu_frozen".

What the paper reports, from the original analyses (failures on the test shots; verdict of each comparison; claim =
accuracy and speed):

| cell (mode) | original failures | comparisons (all "better") | composite / bar (ms) | claim |
|---|---|---|---|---|
| BB72-3 (fast, 0.3 %), r11 alone | 1,341 of 200,000 | sector Relay-BP 2,704, ldpc 3,299, cudaq-qec 3,324 | 0.1019 / 0.9566 | yes |
| BB72-2 (fast, 0.2 %), r12 alone | 254 of 300,000 | 812, 903, 917 | 0.1020 / 0.5009 | yes |
| BB72-3S (balanced, 0.3 %) | 1,571 (3,960 routed) | 2,704, 3,299, 3,324 | 0.2602 / 0.9566 | yes |
| BB72-2S (balanced, 0.2 %) | 290 (721 routed) | 812, 903, 917 | 0.1147 / 0.5009 | yes |
| BB72-3F (accurate, 0.3 %) | 877 (3,960 routed) | full-DEM Relay-BP 1,092 | 3.4306 / 58.5963 | yes |
| BB72-2F (accurate, 0.2 %) | 171 (721 routed) | full-DEM 274, 812, 903, 917 | 0.3238 / 0.5009 | yes |
| G-4R (balanced, gross 0.4 %) | 387 of 150,000 (1,447 routed) | sector Relay-BP 716, ldpc 3,865, cudaq-qec 3,789 | 9.3584 / 13.8994 | yes |
| G-4F (accurate, gross 0.4 %) | 229 (2,983 routed) | full-DEM 484, 716, 3,865, 3,789 | 232.1868 / 2,774.7069 | yes |

Beside, as frozen: the second BB72 network (r12) alone at 0.3 % (1,387), full-DEM Relay-BP against the fast cells
(BB72-3: worse on the first 200,000 shots; BB72-2: level on the first 200,000 and on all 300,000), the gross network
alone (970) and G-4's old q (560).

### 3.2 Cascade on BB72, the follow-up D1 (Table cascade, BB72 rows)

#### What is re-tested

The paper's follow-up D1 compares a pair of decoders with our Cascade re-implementation alone on BB72 (X-memory, d = 6,
6 rounds) under the fitted noise model `depspam_fm`. S = S55'-deep (ConvLines-BB, six blocks, 841,053 parameters)
decides every shot whose margin is at least tau1; C14' (Cascade, 14 layers x 512, 19,687,937 parameters, BF16)
decides the rest; "Cascade alone" is C14' on every shot. The re-test reads the confirmation block again with new
shots and runs D1's analysis once. Networks, thresholds, sizes, statistics and analysis code are D1's frozen ones; nothing
chosen on the original shots is chosen again, and D1's own lock, tree and timing rows stay as they were. The timings are
inherited from D1's registered timing session (D1-S, job CJ73: speed shots of kind 12, so no timed input is a
confirmation shot): the re-test confirms D1's accuracy verdicts on fresh shots with inherited timing; it does not
repeat the timing session.

#### What is frozen

Tree: `<cluster-scratch>/cascade_d1/code.20261011_152939` (436 files besides Python's byte-code caches;
`DEPLOYED` = `net 64b2e25d d1 310ce8e0f8bfb9d3b97b27186547a836d29fc9ca`, `DEPLOYED_SHA` = F =
310ce8e0f8bfb9d3b97b27186547a836d29fc9ca, D1's freeze commit). Every read, compiled read and timing row binds that tag,
and `d1_confirm.py` refuses another one. The tag names a commit, not bytes, so the re-test pins the bytes too:
`retest/pins/d1_tree.sha256` lists every file of the tree but the byte-code caches with its sha256. The list on the
cluster, `<cluster-scratch>/retest/pins/d1_tree.sha256`, has sha256 46329005461cf149951e411d4594bedb66f5f74ea2abf93011b33799741ca07c; the copy here differs from it only
in masked names (the job ids of 33 job logs are `<job>`, the tool name in five review notes' names is `<review>`).
`retest/jobs/tree_pin.sh` checks the tree against that list before each D1 job is submitted (commands below): the
list's sha256, every file's sha256, and no file added or missing. The re-test runs from the tree and writes nothing
into it (`PYTHONDONTWRITEBYTECODE=1`); its own files live outside.

| item | file (under the tree unless absolute) | sha256 |
|---|---|---|
| D1's lock (status "frozen", kind 14, chunks, configurations, nets, speed_sha256, compiled_read, comparisons, power) | `cascade_compare/phase_d/d1_locked_depspam_fm.json` | 3ab39941df356883be395ef7bbf39672fdb546729187bd89fa523be22714f96c |
| first lock (the two configurations' source, configs 2 and 7) | `<cluster-scratch>/cascade_w4/out/locked_depspam_fm.json` | 02b19c451c6a38f6854a6be665b9bdcc87f8feb53d86841309833a70f9be0e81 |
| C14' (Cascade, EMA weights, step 81,130) | `<cluster-scratch>/cascade_w3/ckpt/C14_dfm/final_ema.pt` | dd0c3b1488eed58d02a1a54a7ac3a77d5d3521162c9f7db7da49fb0085c2e80a |
| S55'-deep (ConvLines-BB) | `<cluster-scratch>/cascade_w3/ckpt/S55_dfm_deep/final_ema.pt` | 051370c37d5aef08880d55a89490b3de2109a66c800635dc97583edcf4e281af |
| timing components (lock input) | `<cluster-scratch>/cascade_w4/out/speed/speed_components.json` | 1ec6b2563060f0042f866f19841631e8e1499872e6c15ed77dae7a01c8c811f5 |
| D1-S timing rows (analysis input, job CJ73) | `<cluster-scratch>/cascade_d1/out/speed/rows.jsonl` | 77889d11e194352f9a3fefaf1386810500f3566f0e4b0b144523d0ad9f52edd8 |
| analysis | `cascade_compare/phase_d/d1_confirm.py` | 0d9093cc031caf535d87de2eb1a1d09a99c934f014c52135b7cecc09a87a4f12 |
| compiled read | `cascade_compare/phase_d/d1_compiled.py` | 6446360c278d05c07d5a8dbaa7ccbe8ba13b0e12fbf19d6340e47b8c6113af1c |
| D1's helpers | `cascade_compare/phase_d/d1_common.py` | 38ec7e6bf695598f3f4e04ff995b36e4e5e7902431292b4ed2d128ed1a2b7308 |
| D1's read job (the steps the re-test keeps) | `cascade_compare/phase_d/jobs/d1_read.sh` | 018a3c51c79d87438f2bab53e703ad858733ed717db85825afd1a6ebfc9eeeb0 |
| D1's guard (finding 4: kept but for the kind-14 test) | `cascade_compare/phase_d/jobs/d1_guard.sh` | 1c8959f701f1fb0913f518c1402fb82d970ea4e92c81f91c6cfc483d20cd96de |
| D1's analysis job | `cascade_compare/phase_d/jobs/d1_confirm.sh` | 94683fe910b6422607867d5fb67b53bddb80b71daa5b3ffea7cc289fe04b0c74 |
| phase C's read driver | `cascade_compare/phase_c/jobs/curves.sh` | 4441fcb88fc9a8e4d7e942f8c2a15054567a6867204f1e4802a305d201439778 |
| the eager read (pc_eval.py, an extension of bb.cascade_eval) | `cascade_compare/phase_c/pc_eval.py` | 4e1a2f7b4c95352a66b4c9c4fe7465cc7aa86e2df510886b2687bf122af7e306 |
| phase C's constants and card guard | `cascade_compare/phase_c/pc_common.py` | 72e1dda3d637a47916f5934fc6e0ba8372f87f3d2e158c3c97f056e9d6d662a4 |
| seed function (patched by the shim) | `ising_compare/bb/cascade_data.py` | f3645907e8b4e6bdc97b10cf9f38d7188c4a6cbadb328c4fd7c1f416902e1977 |
| evaluation driver (sampling pool) | `ising_compare/bb/cascade_eval.py` | 3db3cb6cb5fa09a9375b5675bc7bd9c1319a3cf5676ec35b0beea8d325905be2 |
| D1's GPU and CPU job environments | `cascade_compare/phase_d/scripts/gpu.sbatch`, `cpu.sbatch` | bb96c95e4e2ba3cfb755eb098c29133b0153fe220b18a0db79e5690cbc64a6d3, ac6ffd131ca44ec9da5f00add6587e001c4b20e238af1a1d0d323c7a5c3d6687 |
| D1's own result (replay target) | `cascade_d1/out/confirm/run_1/d1_result.json`, `d1_tables.md` | f262c73de1e73498a807c78323a17fe2f4f65a718e657123c3063dc08adf3878, fb2bffe65319e65a81fd64c7380699a1f86dacc52c1fb13e15b3d13227e07ae8 |

Frozen values (the lock holds them; `d1_confirm.py` refuses a read that differs):

| point | q1 | tau1 (margin of S55'-deep, float64, routed iff margin < tau1) | chunks (10,000 shots each) | shots | circuit sha256 (`str(circuit_at(bb72, p, depspam_fm))`) |
|---|---|---|---|---|---|
| p = 0.2 % | 2 % | 7.3331888389587405 | 1,100: 6,433,102 .. 6,434,201 | 11,000,000 | dcaae19e2cebf3ce4fd0b3851061d88c093974e879b8b66e29f92413a815628d |
| p = 0.3 % | 10 % | 7.60439167022705 | 1,700: 6,433,102 .. 6,434,801 | 17,000,000 | 647fda0286c0407118c662291525ad3ac9215c2bd087a8151ec4faa4f2c92b11 |

Seed kind 14 (reserved for D1), family `depspam_fm`; c0 = int(F[:7], 16) mod 9,000,000 = 6,433,102 (`d1_common.c0_of`). The
circuits are Relay-BP's testdata at revision d185194ba0cb4101ced4340d82b2ee6d42f225f0 (base file sha256
ac17ceaed7d025a702c817de2200a9a65b2726ef733714d78c610b17df23a020) with W1's fitted noise. The lock also pins: the card (NVIDIA RTX PRO 6000 Blackwell
Server Edition, a full card; `pc_common.gpu_env` refuses every MIG slice and other card), torch 2.11.0+cu128 in `bb/venv_cu128`
(`d1_confirm.py` refuses records whose GPU or torch differ), read precisions (C14' BF16 eager, S55'-deep on the fused engine
at 4,096 shots per call), the compiled reads (32 streams x 16 batch-1 graphs, checks on the first 20,000 rows and every
1,000th row, S anchor on the first 32,768 rows of C4b's frozen read), the eager batches (8,192 + 1,808 per chunk), the
compiled@4096 and compiled@1 paths, the sampler (Stim 1.16.0 in 10,000-shot chunks, `--procs 8` spawn pool, detector rows
kept with `--save-det`), and the analysis (alpha 0.05, one Holm over H1e, H2e, H1c, H2c, the intersection-union compiled
rule). Triton and Inductor caches: each job copies D1's seed caches (`cascade_d1/cache/{triton,inductor}`, read only) into its
own directory, as D1's jobs did.

#### What changes: the seeds of the kind-14 chunks, and nothing else

D1's `bb.cascade_data.eval_seed(kind, p, chunk, family)` is arithmetic (2^55 + family offset + 2e15 + kind x 1e13 +
round(p x 1e6) x 1e7 + chunk). `retest/shim/rt_d1.py` (outside the tree) replaces that one function when `RT_SEEDS` is set:
for kind 14, family `depspam_fm`, p = 0.002 or 0.003, chunk c0 + j gets `seeds.chunk_seeds("d1-0.002" | "d1-0.003")[j]`,
the seed of beacon index j (`retest/seeds.py`; every seed is recomputed from the beacon record `RT_BEACON`, and any
difference raises). Any other chunk of kind 14 / `depspam_fm` (another p, a chunk outside the block) raises; every other
kind, family, p and chunk calls D1's function unchanged. Without `RT_SEEDS` the shim is an identity launcher.

How the patch reaches every process. `python rt_d1.py SCRIPT.py ARGS` runs the script by `exec` as `__main__`, with
`sys.modules["__main__"]` left as the shim. D1's eager read (`pc_eval.py`, which ends in `bb.cascade_eval.main()`)
samples in `mp.get_context("spawn").Pool(8)`; the workers run `bb.cascade_data._eval_chunk`, which calls `eval_seed` from its
module globals. A spawn worker starts a new interpreter and re-runs the parent's `__main__` file as `__mp_main__` before
unpickling a task, so each worker executes the shim's top level, which installs the patch, and not the target script
(`runpy.run_path` would put the target in `sys.modules["__main__"]` and the workers would run it unpatched, which is why the
shim execs). The main process (which writes each chunk's seed into the read's `chunk_meta`), the 8 sampling workers and
`d1_confirm.py` (which imports `eval_seed` inside `check_read`, after the patch, so that its check "every chunk's seed is
eval_seed(kind, p, j, family)" holds for the new seeds) therefore agree. pc_eval has no partial or resume path: a read is
written once at its end (`np.savez`, then `done_<p>`); a job that stops before that is run again from the start with the same
seeds, and a job that stopped after the eager read continues with the compiled read (see the guard below). The compiled read
draws no shots (its input is the saved detector rows), so `d1_compiled.py` runs as D1 ran it, without the shim. Stim's draws
depend on the host's CPU SIMD width; the read keeps the detector rows and each chunk's digest, as D1's did, and the read
job re-draws a sample of its chunks on the node (below).

Guard (review finding 4). `retest/jobs/d1_guard_rt.sh` keeps every check of D1's `d1_guard.sh`: the tree is F and clean
(`DEPLOYED_SHA` = F, `DEPLOYED` = `net 64b2e25d d1 F`) and a read already done in both parts (`done_<p>` in `read/` and
`compiled/`) is refused, now applied to the re-test's output directory. It drops only the freshness test (no JSON record
under the audited Alpine directories names kind 14), which D1's own reads fail by construction; the beacon takes its
place. Added, not replacing anything: the output directory may not lie inside D1's original outputs; a block record
(`read/block_<p>.json`) of other seeds or another beacon refuses a continued read; the dry seeds (drand round 6,540,000, a
past round) can only write under `/dry/` and only they may set `RT_DRY_CHUNKS`.

The read job (`retest/jobs/d1_read_rt.sh`) is D1's `d1_read.sh` with these differences and no others: (1) the guard above;
(2) F and the lock fixed to D1's (sha256 checked) and no `STREAMS` / `ROUNDS` override; (3) output `$RT_OUT/d1/<p>/{read,compiled}`
instead of `$A/out/{read,compiled}`, block record also holding the sha256 of the seeds file and the beacon; (4) the eager read's
`$PY .../pc_eval.py` calls (inside `curves.sh`) are `$PY retest/shim/rt_d1.py .../pc_eval.py`; (5) the shim, with `RT_D1_TRACE` set for the eager step, appends (pid, parent pid, p, chunk, seed) to a per-process file for every kind-14
seed it hands out, and after the eager read `jobs/d1_seedcheck_rt.py` (through the shim) checks that the read's kind, family and chunk
block are D1's, every chunk's recorded seed is the beacon's seed of its index and `eval_seed`'s, that the trace shows exactly one call
by the main process and one by a sampling worker per chunk, each with the beacon's seed (at least two different workers when there are
16 chunks or more), and, for the first chunk, the last and six in between, that the
saved detector rows equal a fresh draw from the recorded seed (and differ from a draw from D1's original seed of that
chunk); a failure stops the job before the compiled read; (6) `sha256.txt` of the four files the analyses read. The
analysis job (`jobs/d1_confirm_rt.sh`) links the two p's reads into `$RT_OUT/d1/confirm_in/{read,compiled}` (the directory pair
`d1_confirm.py` takes), checks their sha256 against the read jobs', writes `confirm/npz_sha256.txt` (the committed record of
the four npz hashes, as D1's `out/npz_sha256.txt`), and runs `d1_confirm.py` through the shim with D1's lock, F and D1-S's
timing rows (sha256 checked).

X2 (review finding 7, exploratory). `retest/shim/rt_x2.py` runs X2's BB72 analysis, hub branch `ising-rev-x2` commit
20179987 (`revision/x2/bb_d1.py`, then `bb_join.py`), from byte-identical copies in `retest/x2/` (sha256 in `x2/SOURCE.txt`,
checked at every start), after `d1_confirm.py`. `bb_d1.py` hard-codes D1's original inputs: the two npz sha256 (`NPZ_SHA`),
the outcome counts it asserts (`REG`: routed, S55'-deep failures, C14' failures, kept breaks, repairs, both-fail shots, eager
path) and the archive path. The adapter sets those names, and nothing else, before `bb_d1.main()` runs: `NPZ_SHA` from the
committed `confirm/npz_sha256.txt`, `REG` from `d1_result.json` of the confirmatory run (`points.<p>.paths.eager`), the
input directory from `confirm_in`. TAU1 stays X2's value and is checked equal to the lock's tau1 in the result; the eager path, the
circuit and detector-layout code, the chunk-digest check and `bb_join.py`'s percentile and binning code are untouched.
Its output (`$RT_OUT/d1/x2/bb/{p0.00200.npz, p0.00300.npz, summary.json, fig.npz, adapter.json}`) feeds the paper's Figure
"why" panel (b) (the shares 68 % and 75 % below the threshold, 18 % and 19 % above it) and is reported as exploratory, apart from
the four verdicts.

#### Commands after the beacon

```
export RT=<cluster-scratch>/retest/code/freeze
export RT_BEACON=<cluster-scratch>/retest/run/beacon.json
export RT_SEEDS=<cluster-scratch>/retest/run/seeds.json
export RT_OUT=<cluster-scratch>/retest/out               # the root of all tests, also the default; D1 writes $RT_OUT/d1
cd $RT/retest/jobs
PIN="bash $RT/retest/jobs/tree_pin.sh <cluster-scratch>/cascade_d1/code.20261011_152939 \
     <cluster-scratch>/retest/pins/d1_tree.sha256 46329005461cf149951e411d4594bedb66f5f74ea2abf93011b33799741ca07c"
$PIN && R1=$(sbatch --parsable d1_R1.sbatch)                    # p = 0.2 %, 11,000,000 shots
$PIN && R2=$(sbatch --parsable d1_R2.sbatch)                    # p = 0.3 %, 17,000,000 shots
$PIN && A1=$(sbatch --parsable --dependency=afterok:$R1:$R2 d1_confirm.sbatch)
$PIN && X2=$(sbatch --parsable --dependency=afterok:$A1 d1_x2.sbatch)   # exploratory, after the confirmatory analysis
```

`$PIN` runs before each submission and again once R1 and R2 have ended, before the analysis starts. If it does not
print `PIN_OK`, nothing more is submitted and the pending D1 jobs are cancelled by their ids. Its output lines go into
the next commit (the first ones into commit 1).

The sbatch files carry `--nice=1000` and D1's resources. R1 and R2: `-p artxpro6000 --qos=gpu-normal --gres=gpu:rtx_pro_6000:1
-N 1 -c 16 --mem=96G` (one full RTX PRO 6000 each, as D1's R1 and R2), `-t 2:00:00` and `-t 3:00:00` (1.5 times D1's wall times
of 1:17:22 and 2:00:13, which were 1.29 and 2.00 GPU-hours; D1's own limits were 3:00:00 and 4:00:00); the analysis `-p acpu
--qos=cpu-normal -c 4 --mem=32G -t 1:00:00` (D1-A: 32 s); X2 `-p acpu --qos=cpu-normal -c 2 --mem=40G -t 1:00:00` (X2's own job:
about 2 minutes). Logs go to `<cluster-scratch>/retest/logs/d1/`.

Each read job (`d1_gpu_job.sh` = D1's `gpu.sbatch` environment: D1's tree, `PYTHONPATH`, caches copied from D1's seed caches,
`PY` = `apptainer exec --nv -B <cluster-scratch-root> $I/pyfull.sif $B/venv_cu128/bin/python`) runs, for its p (`$PC` and `$PD` are
the tree's `cascade_compare/phase_c` and `phase_d`; chunks 6433102:6434202 at p = 0.002, 6433102:6434802 at p = 0.003):

```
d1_guard_rt "$F" P "$T" "$RT_OUT/d1/P"                  # tree = F and clean; read not done; output outside D1's; same seeds if continued
export RT_D1_TRACE=$RT_OUT/d1/P/read/trace           # the eager step only
$PY $RT/retest/shim/rt_d1.py $PC/pc_eval.py --family depspam_fm \
    --ckpt <cluster-scratch>/cascade_w3/ckpt/C14_dfm/final_ema.pt,<cluster-scratch>/cascade_w3/ckpt/S55_dfm_deep/final_ema.pt \
    --prec bf16 --kind 14 --p P --chunks 6433102:6434202 --procs 8 \
    --save $RT_OUT/d1/P/read --save-det --out $RT_OUT/d1/P/read/rows_P.json && touch $RT_OUT/d1/P/read/done_P     # curves.sh
$PY $RT/retest/shim/rt_d1.py $RT/retest/jobs/d1_seedcheck_rt.py --read $RT_OUT/d1/P/read --p P
$PY $PD/d1_compiled.py read --read $RT_OUT/d1/P/read --p P --out $RT_OUT/d1/P/compiled --lock $PD/d1_locked_depspam_fm.json
```

The analysis job (`d1_confirm_rt.sh`, D1-A's environment: the CPU venv `$I/venv_gpu`):

```
$PY $RT/retest/shim/rt_d1.py $PD/d1_confirm.py --lock $PD/d1_locked_depspam_fm.json \
    --F 310ce8e0f8bfb9d3b97b27186547a836d29fc9ca --read $RT_OUT/d1/confirm_in/read --compiled $RT_OUT/d1/confirm_in/compiled \
    --speed <cluster-scratch>/cascade_d1/out/speed/rows.jsonl --out $RT_OUT/d1/confirm
```

The X2 job (X2's environment: `PYX` = `apptainer exec -B <cluster-scratch-root> -B <cluster-projects> $I/py.sif $I/venv/bin/python`):

```
$PYX $RT/retest/shim/rt_x2.py run --inputs $RT_OUT/d1/confirm_in --result $RT_OUT/d1/confirm/run_1/d1_result.json \
    --npz-sha $RT_OUT/d1/confirm/npz_sha256.txt --out $RT_OUT/d1/x2
```

Outputs: `$RT_OUT/d1/{0.002,0.003}/{read,compiled}/`, `sha256.txt`, `$RT_OUT/d1/confirm_in/`, `$RT_OUT/d1/confirm/{npz_sha256.txt,
run_1/{d1_result.json,d1_tables.md}}`, `$RT_OUT/d1/x2/bb/{p0.00200.npz,p0.00300.npz,summary.json,fig.npz,adapter.json}`. A read job
that stops (time limit, node failure) is submitted again with the same variables: the guard lets it continue only under the same
seeds file and beacon (the block record), the eager read starts again from its first chunk when its `done_<p>` is missing (no
partial file is kept) with the same seeds, and the compiled read starts again when its `done_<p>` is missing. D1's
analysis is run once; a second run needs `RT_CONFIRM_RERUN=1`.

#### Replay checks before the freeze (review finding 10)

Run on Alpine against D1's tree and D1's original outputs (read only) with the bytes listed at the end of this section; records
in `retest/replay/d1/`. CPU jobs on `acpu` in D1's CPU environment (X2's own for X2), one GPU job on a full RTX PRO 6000. All
checks pass.

| check | what was run | result | record (job) |
|---|---|---|---|
| 1a. the original analysis through the shim, patch off | `rt_d1.py d1_confirm.py` with D1's lock, F, D1's original reads, compiled reads and D1-S rows, into a new directory | `d1_result.json` equal to D1's `run_1` (loaded JSON equal and byte-identical, sha256 f262c73d...); `d1_tables.md` byte-identical (fb2bffe6...) | `replay_confirm.json`, `compare_d1_*.json` (RJ01) |
| 1b. the shim switches on | the same with the dry seeds | exit 3, one "a chunk's seed is not eval_seed(kind, p, j, family)" refusal per p, nothing written | `replay_confirm.json` |
| 1c. X2 adapter on the original records | `rt_x2.py check` and `run` with D1's `npz_sha256.txt` and `d1_result.json`, on D1's original reads | the derived `NPZ_SHA` and `REG` equal the ones `bb_d1.py` hard-codes; per-shot files `p0.00200.npz` (e8d50e51...), `p0.00300.npz` (87d76004...), `fig.npz` (7fbe367b...) and `summary.json` (8c83effe...) byte-identical to X2's published outputs | `replay_x2.json`, `x2_check.json`, `compare_x2_*.json` (RJ02) |
| 1d. the eager read through the shim, patch off | D1's `curves.sh` / `pc_eval.py` on D1's first chunk (c0) at both p, on a full RTX PRO 6000 | every array of the read (detector rows, observables, both networks' predictions, confidences, logits, failures) equals the first 10,000 rows of D1's original read; chunk seed and digest and every model-signature field but the timings equal | `replay_read_off.json` (RJ03) |
| 2a. seeds, patch on (dry seeds, drand round 6,540,000) | `rt_d1_replay.py` through the shim | main process: the seeds of all 2,800 chunks (1,100 + 1,700) equal the beacon's, recomputed from R with the seed formula, lie in [2^62, 2^63) and differ from D1's original seeds; 9 other kind-14 requests (other p, chunks outside both blocks) raise `ValueError`; spawn workers started as `pc_eval` starts them (`chunks_in_order` over a spawn pool, 4 workers): the seed each worker computes for the first, second and last chunk of both blocks is the beacon's, and the shots it returns equal a direct draw in the main process from that seed and differ from the draw from D1's original seed (6 of 6) | `seeds_on.json` (RJ04) |
| 2b. the same pipeline, in every read | `RT_D1_TRACE` and `d1_seedcheck_rt.py` | per chunk exactly one seed call by the main process and one by a worker, both the beacon's; a sample of saved chunks equals fresh draws from the recorded seeds; seen in the dry run (below) | `dry/*/read/{trace,seedcheck_*.json}` |
| 2c. resume paths | the guard against D1's own, 24 cases; the read job run again on a finished eager read | the guard and D1's guard (kind-14 test emptied) give the same status on tree not F, tree not clean, completed read, fresh directory, eager read done only; D1's guard with D1's directory as `SEARCH` refuses ("records ... name kind 14"), which is the one check dropped; the added checks (output inside D1's outputs, other seeds or beacon in the block record, dry and real seeds mixed, missing files) each refuse for the stated reason. The read job run again on a copy of the dry eager read without its compiled read: guard passed, eager read skipped, seed check passed on the existing read, compiled read written, eager files unchanged. The eager read has no partial state to resume (written once at its end) | `guard.json` (RJ05), `replay_resume.json` (RJ03) |
| 3. other seeds unchanged | the same harness, patched function against the original loaded as a separate module | 22,288 cells (kinds 0-99, 8 p, 14 chunks, both families, minus the 112 cells of kind 14 / `depspam_fm`): 0 mismatches; kind 14 in the uniform family equal. Patch off: no patch attribute, all 22,400 cells equal, workers draw the original seeds, and the first three chunks at p = 0.2 % have the digests of D1's original read (informational: Stim's stream depends on the host's SIMD width, and this node's agrees) | `seeds_on.json`, `seeds_off.json` |

#### Dry run (the whole pipeline, dry seeds, one chunk per p)

Job RJ06 (`d1_R1.sbatch` with `RT_DRY_CHUNKS=1`, the dry beacon and seeds of round 6,540,000, both p in one job on one RTX PRO 6000,
node gpu-node-06, 1 min 45 s, 0.03 GPU-hours; two earlier runs of the same steps, jobs RJ07 and RJ08, differed from these bytes
only in the trace and the output path), from `RT=.../code/dry`, output `dry/out/d1`: guard passed; eager read through the shim, chunk
6,433,102 (10,000 shots) with seed 8005032492382299980 at p = 0.2 % and 7336132527545497238 at 0.3 %, each the seed of beacon index 0 of
its set; the trace holds one call by the main process and one by a worker per chunk; the seed check passed (saved rows equal a fresh
draw from the recorded seed, unlike a draw from D1's original seed); compiled reads: digests ok, S anchor passed, the first-20,000-rows
check (10,000 rows) and the every-1,000th check had 0 differing decisions or margin bits, on RTX PRO 6000 with torch 2.11.0+cu128. Dry
shots: routed 186 (1.86 %) and 1,003 (10.03 %), C14' failed on 9 and 46 shots, S55'-deep on 13 and 69; nothing here is a result.

The analysis as it will run (`d1_confirm_rt.sh`, job RJ09) refuses on these reads, as it must, with exit 3 and six reasons, all about
size: at each p "chunks [6433102, 6433103], not [6433102, 6434202|6434802] for F", "the chunks read are not c0 .. c0 + n - 1" and
"10000 shots, not 11000000|17000000". No refusal about the family, kind, seeds (`eval_seed` equals the recorded seeds under the shim),
network hashes, precisions, GPU, torch or tree tag, which `check_read` tests in the same call. The same job then ran `d1_confirm.py`
through the shim with a dry-only copy of D1's lock (blocks of 1 chunk; sha256 48865428...) and the timing rows re-bound to it: it ran to
its tables (all four comparisons "level", no discordant shots). X2's adapter then ran on that result (job RJ10, `--lock-sha` of the
dry lock): the derived input checks passed and `bb_d1.py` and `bb_join.py` wrote their outputs.

#### Cost

Build and checks before the freeze: 0.25 GPU-hours on one full RTX PRO 6000 (five short jobs, two of them superseded dry runs) and 5.0
core-hours allocated on `acpu` (including superseded runs). After the beacon: the reads as D1's, R1 1.29 and R2 2.00 GPU-hours (3.29 in
all, on two cards in about 2.0 h of wall time if both start at once), the analysis about 1 minute on 4 cores, X2 about 2 minutes on 2.

Files of the D1 re-test in `retest/` (sha256, first 16; byte-identical to the dry tree on Alpine that the checks above ran):

| file | sha256 |
|---|---|
| `retest/jobs/d1_confirm_rt.sh` | 2b9f2bf49ee64af3 |
| `retest/jobs/d1_confirm.sbatch` | 4b835c8b58c808d8 |
| `retest/jobs/d1_cpu_job.sh` | fc621289e2079d54 |
| `retest/jobs/d1_gpu_job.sh` | e6ecebb30bafa160 |
| `retest/jobs/d1_guard_rt.sh` | 569f25ed34be8e9a |
| `retest/jobs/d1_R1.sbatch` | 0f9bbe68ea47472d |
| `retest/jobs/d1_R2.sbatch` | fc69bb65a598de35 |
| `retest/jobs/d1_read_rt.sh` | dfeaac0b3f5cbe69 |
| `retest/jobs/d1_seedcheck_rt.py` | 71f36febb295e62c |
| `retest/jobs/d1_x2_rt.sh` | 4cba8e02da1f7f34 |
| `retest/jobs/d1_x2.sbatch` | 94db197150807fcf |
| `retest/replay/d1/d1_orig_npz_sha256.txt` | b3306041b24e42cf |
| `retest/replay/d1/d1_replay_gpu.sbatch` | c120a757549e9f56 |
| `retest/replay/d1/d1_replay_guard.sbatch` | 94467a779d1f806b |
| `retest/replay/d1/d1_replay.sbatch` | ba52d17c31210671 |
| `retest/replay/d1/d1_replay_x2.sbatch` | 156dcc6eac60a8e9 |
| `retest/replay/d1/dry_confirm.sh` | a285836935792f33 |
| `retest/replay/d1/dry_read.sh` | a3ef16add8d8daec |
| `retest/replay/d1/guard_tests.sh` | 9354f275f9cf715c |
| `retest/replay/d1/replay_confirm.sh` | 13a5f88fc24f5b19 |
| `retest/replay/d1/replay_read_off.py` | 45f1b5e3f79df1fd |
| `retest/replay/d1/replay_read_off.sh` | 415b10853f7815f1 |
| `retest/replay/d1/replay_seeds.sh` | 0cd2deae9fbab887 |
| `retest/replay/d1/replay_x2.sh` | 188911696feeda81 |
| `retest/replay/d1/rt_d1_compare.py` | 649b9f99f556ee73 |
| `retest/replay/d1/rt_d1_replay.py` | f9e2186aeb7dcec2 |
| `retest/shim/rt_d1.py` | 650d13975fb104c6 |
| `retest/shim/rt_x2.py` | 718b8ced2b068e43 |
| `retest/x2/bb_d1.py` | 2f336a4e8faf6658 |
| `retest/x2/bb_join.py` | 4008eccb6f24384f |
| `retest/x2/SOURCE.txt` | 5e6d3548cfb44c55 |

#### What the paper reports (v4, 9680d1d; Table "cascade", BB72 rows, main.tex 545-549, and the places that quote D1)

| setting | decoder | routed (%) | failures single / batched | verdict single / batched | time: single (us), batched (us per round) |
|---|---|---|---|---|---|
| BB72, p = 0.2 %, 11,000,000 shots | Cascade alone | | 7,912 / 7,918 | | 1,759.8; 12.00 |
| | pair | 2 | 7,853 / 7,858 | better / better | 163.2; 1.355 |
| BB72, p = 0.3 %, 17,000,000 shots | Cascade alone | | 76,676 / 76,711 | | 1,763.8; 12.01 |
| | pair | 10 | 76,409 / 76,445 | better / better | 300.5; 2.332 |

(The footnote: "On BB72 the counts come from Cascade's compiled single and batched paths".) The original analysis
(`d1_result.json`, sha256 f262c73d..., D1's `run_1`): all four comparisons "better" in one Holm family at 0.05:
H1e (0.2 %, eager) McNemar p = 2.30e-11 at the Holm level 0.025; H2e (0.3 %, eager) 3.10e-39 at 0.0167; H1c (0.2 %,
compiled) 1.15e-09 at 0.05; H2c (0.3 %, compiled) 2.77e-40 at 0.0125. Shots where only the pair failed / only Cascade alone
failed, and the pair's difference in logical error rate per round (Newcombe 95 % interval; the compiled comparison needs
both paths' intervals below zero at its Holm level):

| p | path | Cascade alone / pair failures | only pair / only Cascade | exact McNemar p | difference (x 1e-6) |
|---|---|---|---|---|---|
| 0.2 % | eager | 7,922 / 7,855 | 19 / 86 | 2.30e-11 | -6.091 [-7.962, -4.245] |
| 0.2 % | single | 7,912 / 7,853 | 19 / 78 | 1.15e-09 | -5.364 [-7.164, -3.586] |
| 0.2 % | batched | 7,918 / 7,858 | 19 / 79 | 7.13e-10 | -5.455 [-7.264, -3.668] |
| 0.3 % | eager | 76,730 / 76,469 | 81 / 342 | 3.10e-39 | -15.35 [-17.74, -12.98] |
| 0.3 % | single | 76,676 / 76,409 | 81 / 348 | 1.71e-40 | -15.71 [-18.11, -13.31] |
| 0.3 % | batched | 76,711 / 76,445 | 81 / 347 | 2.77e-40 | -15.65 [-18.05, -13.26] |

Routed to Cascade: 220,058 shots (2.0005 %) and 1,701,544 (10.0091 %). The claim "more accurate and faster" (`claims`) held
at both points on both engines (today's `base` and milestone 2's `new`): compiled verdict "better", the pair's mean
single-shot latency and its batched time per shot below Cascade alone's in the same timing job, and all of the pair's S
checks zero. The lock's planned power for each comparison was 0.92-0.96 (design scenario: half the exploratory gain; Holm of 4,
single and batched taken as identical or as independent), 0.84-0.86 that all four reject; the observed gain was larger.

Where the paper's other BB72 numbers come from (`d1_result.json` paths, each recomputed from the re-test's own result file):
"about 0.75 % and 0.35 % less often" = `points.<p>.paths.single|batched.diff_share_of_C` (0.7457 / 0.7578 % and 0.3482 /
0.3468 %); the appendix's 5.36e-06 [3.59, 7.16]e-6 and 1.57e-05 [1.33, 1.81]e-5 = `paths.single.stats_95` diff_ler, diff_lo,
diff_hi; "4.0 % and 3.0 % of Cascade's failures on the kept shots" = `C_fail_kept` / `stats_95.fail_b` on the compiled paths;
RQ3's "96-97 %" routed share of Cascade's failures, "repairs 86 and 342", "breaks 19 and 81" = `paths.eager.C_fail_routed`,
`kept_repairs`, `kept_breaks`; "all four were better" = `comparisons[].verdict`. Figure "why" panel (b)'s 68 % / 75 % and
18 % / 19 % are not in `d1_result.json`: they come from the exploratory X2 analysis (X2's `bb_join.py` summary, cells
"S only fails, routed" and "kept"), which the X2 adapter reruns. The timing columns and the speed-up ratios (5.9-12.7 per single
shot, 5.1-16.4 per batch, 138.8 us and 276.6 us on milestone 2's engine) are D1-S's, not re-measured.

#### From the re-test's labels to the paper

Fixed before any shot is drawn.
- Unit of comparison: one point (p) carries one compiled verdict, H1c at 0.2 % or H2c at 0.3 %: the joint verdict of the single
  and batched comparisons as the lock defines it (intersection-union: the larger of the two exact McNemar p-values, "better"
  only if both paired Newcombe intervals at the comparison's Holm level lie below zero, "worse" only if both lie above, otherwise
  "level"; one comparison if the two paths fail on exactly the same shots). The two verdict cells of a BB72 row both show that one
  verdict. D1's joint compiled verdict stays one verdict: it is not split into a per-path verdict for either mode.
- The paper's "better" is repeated only by "better" from `d1_confirm.py` on the re-test's shots, under one Holm procedure at
  0.05 over the four comparisons (H1e, H2e, H1c, H2c), the same family as D1. "Level" (shown in the paper as n.s.) is not
  equality and does not refute the original; "worse" contradicts it. A "better" at 0.2 % and a "level" at 0.3 % (or the reverse) are
  reported as two separate outcomes; no sentence of the paper is kept that needs the failed one.
- The eager comparisons (H1e, H2e) are reported beside the compiled ones as D1 reported them, with their own labels; they are
  in the Holm family and the claim does not read them.
- "More accurate and faster" at a point is repeated only when the re-test's compiled verdict at that point is "better" and
  the inherited timing conditions hold (`claims.<p>.more_accurate_and_faster.base|new`: faster in single-shot mean and in
  batched time per shot, and the S checks zero, all from D1-S's rows, which the analysis reads unchanged). Timing is
  inherited and never re-measured: every time in the BB72 rows, every speed-up ratio and the clock-flag appendix stay D1-S's,
  labelled as inherited from the registered session; no composite time is recomputed (D1's pair has no CPU fallback).
- Counts, routed percentages, "less often" shares, differences with their intervals, kept repairs and breaks and the routed
  share of Cascade's failures are recomputed from the re-test's result file and carry no verdict of their own.
- The X2 numbers (Figure "why" panel (b)) are exploratory: they come from the adapter run after the confirmatory analysis,
  are labelled exploratory, and have no say in any verdict above.

### 3.3 Cascade on the surface code at d = 13 (Table cascade, surface rows)

#### What is re-tested

The paper's d = 13 test (thresholds and analysis locked in advance, lock commit bddd3f91) compares our Cascade (CS13) with ConvLines (d13_b5_r775) and with correlated
PyMatching on fresh surface-code shots: S is ConvLines, B is Cascade alone, M is correlated PyMatching 2.4.0. The five
decoders are C1 = S -> B -> M ("the chain"), C2 = B alone ("Cascade alone"), C3 = M alone, C4 = S -> M ("ConvLines with
correlated PM") and C5 = S -> B ("the pair"). The re-test reads the confirmation block again with new shots and
runs the frozen analysis once. Networks, thresholds, matcher settings, sample sizes, statistics and the analysis code
are the frozen ones; what was chosen on the original shots is not chosen again. The timings are inherited from the
registered timing session (jobs e7 and e7r, kind-12 shots; no input of a timing is a confirmation shot). The re-test is
therefore a fresh-shot confirmation of accuracy with inherited timing; the timing session is not repeated.

#### What is frozen

Tree: `<cluster-scratch>/cascade_w5/code.20261012_060258` (`DEPLOYED`: commit
a43103277aca60e82d959255fd8e514d28f20cc8, release 8d65844a1dae1530ed158eac2cf1f75c1dcc955a, clean; 87 files besides
Python's byte-code caches, code digest `b15af28f3e162378`). Every record binds this digest and `sv_read` / `sv_route`
refuse another one. The re-test also pins every file of the tree but the byte-code caches: `retest/pins/d13_tree.sha256`
(the list on the cluster, `<cluster-scratch>/retest/pins/d13_tree.sha256`, has sha256 a4abb5b0cb09791e338f7cedb5956b7fe6418b958036c1b988aaf1c6c94ca227; the copy here masks
one job log's job id as `<job>`), checked by `retest/jobs/tree_pin.sh` before each d = 13 job is submitted (commands
below). The re-test's own files live outside the tree.

| item | file | sha256 |
|---|---|---|
| lock (commit bddd3f91, lock id 5767020407ed11f3) | `cascade_w5/eval/locked_d13.json` | 5767020407ed11f3ec3a2ef985f2c41e7d0ac93a489cb7929e16a8159b5a98f1 |
| gate (all_pass true, 16 rows) | `eval/gate_d13.json` | 7ba1fa223037e1b3d2ed1173d4e2c6e56b356ef9221d7f71512f0033af76e43a |
| receipt of the lock commit | `eval/lock_commit.json` | e40ef84815252cfdb07abfb48b3cb3964f76d13ea0973fdbe7308699ac312972 |
| speed summary (lock input) | `eval/speed.json` | 94059051cc8dff871c27c20914d6fda981345d6b6df2f1a400df62e82d1a34cf |
| e7 timing rows (analysis input) | `eval/pipes.jsonl` | e7d75b4cf0cd12295e1e3bb81b5194ba80bba83718acf56037db87122a387440 |
| e7 bitwise check (analysis input) | `eval/bitwise_c7.json` | 81ea56576c66882a21eb4c0a2988cc68d09b36079b9b5df91afe28a46bc9aaa1 |
| Cascade CS13, EMA at step 81,130 | `cascade_w5/ckpt/CS13/final_ema.pt` | 5351b27b449145ae6c461260b6320925cebba56d2b0449f5a23884d9f787a659 |
| ConvLines d13_b5_r775 | `release/weights/d13_b5_r775.safetensors` | cbd2b84a47dd6bf71c62eb0cae4d660e2811db5efb9b55b695dd9337d141f12a |
| circuit, X memory, R = 13, legacy noise | `release/circuits/d13_legacy.stim` | 86f1d9e436a788b3cc1cb141268f064ed5e4afe7c0f8da3639852bc8ec68397e |
| grid | `release/weights/grid_d13.npz` | d1f651ac189280fcd036996b8bb298ec1407d35a7eed44f5e9597699c45601db |
| container | `ising/pyfull.sif` | adba1be692e958e5b9ab872f9bbf34c577b8b51b166ca87498f3182e3f37ff43 |

The lock also pins: the GPU (NVIDIA RTX PRO 6000 Blackwell Server Edition, a full card; every read refuses a MIG slice
or another card), torch 2.11.0+cu128 in `bb/venv_cu128` (package-list hash `6fd0f5933f01c59c288002bd9e9c623e718068be18c143b78995bb6c0f24a964`), the frozen paths
(S: engine@1 and engine@4096; B: compiled@1 and compiled@16, b* = 16), the kernel-cache digest `46c6069894fa4844`
(1,248 files in `eval/kcache`; the cache directory name is part of the digest, so the re-test reads it in place), Stim
1.16.0, PyMatching 2.4.0 (16 workers), and every threshold:

| p | decoder | tau1 (S, abs confidence) | tau2 (B, abs z) | margin delta_p |
|---|---|---|---|---|
| 0.6 % | C5, the pair (q1 10 %) | 5.242445659637451 | - | 0.0005102 |
| 0.6 % | C1, the chain (q1 40 %, r 0.5 %) | 9.554975891113282 | 0.8671875 | 0.0005102 |
| 0.6 % | C4, S -> M | 1.8486117124557495 | - | |
| 0.35 % | C5, the pair (q1 1 %) | 8.111059150695802 | - | 7.5e-6 (stored 7.499999999999999e-06) |
| 0.35 % | C1, the chain (q1 60 %, r 0.1 %) | 13.637489700317381 | 4.59375 | 7.5e-6 |
| 0.35 % | C4, S -> M | 1.8486117124557495 | - | |

The margins are 10 % of Cascade alone's failure rate on the development shots (5,102 of 1,000,000 and 750 of
10,000,000); they stay fixed and are not recomputed from the new shots. The confirmation block is 1,000,000 shots at
p = 0.6 % (chunks 0-99 of 10,000) and 30,000,000 at p = 0.35 % (chunks 0-2999); the plain-PM line MP is context only.

#### What changes: the seeds of the kind-11 chunks

In the pinned tree `surface_data.eval_seed(d, kind, p, chunk)` is arithmetic and the confirmation block is kind 11.
`retest/shim/rt_surface.py` (outside the tree) replaces that function when `RT_SEEDS` is set: for kind 11 at d = 13
chunk j at p gets `seeds.chunk_seeds("d13-0.006")[j]` (p = 0.6 %) or `seeds.chunk_seeds("d13-0.0035")[j]` (p = 0.35 %),
the seeds of `retest/seeds.py` from the beacon round after the public freeze, each recomputed from the beacon record
(`RT_BEACON`); a kind-11 request for another d, p or a chunk beyond the sets raises. Every other kind (speed shots
kind 12, development 10, dry 13, ...) calls the pinned function unchanged. The patch is installed at import at the top
level of the shim, and the pinned tree's pools start with `multiprocessing.get_context("spawn")` (the sampling pool of
`surface_data.stream_shots`, and `sv_read`'s 16 M workers), which re-run the main script as `__mp_main__` in every
worker, so the main process, the sampling pool, the PyMatching workers and the resume path (`prime_queues`) draw the
same seeds. The reads also compare the digest of every chunk drawn by the GPU path with the one drawn by the M workers
(a mismatch stops the read), and store each chunk's digest in `obs_*.npz`; the optional check `d13_replay.py verify`
(below) recomputes some of them from the seeds. The shot arrays themselves are never stored.

The reads, as the originals (`eval/jobs/e5_conf_batched.sh`, `e6_conf_single.sh`, with `common.sh`: E = `cascade_w5/eval`,
the kernel cache, the staged venv, the guard against an unpinned or dirty tree), keep every argument and write a new
directory. e5 and e6 write the same directory `$RT_OUT/d13/conf`, as the originals wrote `$E/conf`, because `confirm`
needs all nine computations of a p in one directory and the observable files are published once and compared by the
second reader. One full RTX PRO 6000 per job, `-c 24 --mem=100G --qos=gpu-normal --nice=1000`; time limits 3:15:00 (e5)
and 2:10:00 (e6), 1.5 times the originals' 2:06:48 and 1:22:03 (limits 4:00:00 and 2:30:00).

| file in `retest/` | sha256 (first 16) |
|---|---|
| `shim/rt_surface.py` | a63015155daa232f |
| `seeds.py` | 828c5c3d53232419 |
| `tags.json` | a6ef241680ccf9dc |
| `jobs/d13_env.sh` | 426a5bbe83abc203 |
| `jobs/d13_read.sh` | f54c80e628180a34 |
| `jobs/d13_e5.sbatch` | f83d9ef3eea2f7db |
| `jobs/d13_e6.sbatch` | f68fae06b7612bac |
| `jobs/d13_analysis.sh` | 13fbfa0940960ece |
| `jobs/d13_replay.sbatch` | 5d06168e23c304c2 |
| `jobs/d13_dry_checks.sbatch` | 755a1c0d3e05c69c |
| `replay/d13/d13_replay.py` | 6375622a515088a3 |

#### Commands after the beacon

```
export RT=<cluster-scratch>/retest/code/freeze
export RT_BEACON=<cluster-scratch>/retest/run/beacon.json
export RT_SEEDS=<cluster-scratch>/retest/run/seeds.json
export RT_OUT=<cluster-scratch>/retest/out
module load slurm/alpine
PIN="bash $RT/retest/jobs/tree_pin.sh <cluster-scratch>/cascade_w5/code.20261012_060258 \
     <cluster-scratch>/retest/pins/d13_tree.sha256 a4abb5b0cb09791e338f7cedb5956b7fe6418b958036c1b988aaf1c6c94ca227"
$PIN && E5=$(sbatch --parsable --nice=1000 $RT/retest/jobs/d13_e5.sbatch)
$PIN && E6=$(sbatch --parsable --nice=1000 $RT/retest/jobs/d13_e6.sbatch)
$PIN && sbatch --parsable --nice=1000 --dependency=afterok:$E5:$E6 $RT/retest/jobs/d13_analysis.sh
```
`$PIN` runs before each submission and again once e5 and e6 have ended, before the analysis starts; without `PIN_OK`
nothing more is submitted and the pending d = 13 jobs are cancelled by their ids. Its output lines go into the next
commit (the first ones into commit 1).

Each read job runs, per p, the original command through the shim (shown for e5; `$PY` is the container with the staged
venv; `$E` is `<cluster-scratch>/cascade_w5/eval`, `$LOCK` and `$GATE` are its `locked_d13.json` and `gate_d13.json`):

```
$PY $RT/retest/shim/rt_surface.py surface.sv_read --stage conf --p P --chunks 0-100|0-3000 \
    --comp SB,Bb,BQ1,BQ5,MB,MP --lock $LOCK --gate $GATE --out $RT_OUT/d13/conf          # e5, P = 0.006 then 0.0035
$PY $RT/retest/shim/rt_surface.py surface.sv_read --stage conf --p P --chunks 0-100|0-3000 \
    --comp S1,B1,MS --lock $LOCK --gate $GATE --out $RT_OUT/d13/conf                      # e6
```

The analysis job runs, through the same shim on a CPU node (no GPU needed: the analysis identity is venv, packages and
code), the original e8's last two steps:

```
$PY $RT/retest/shim/rt_surface.py surface.sv_route confirm --lock $LOCK --gate $GATE --conf $RT_OUT/d13/conf \
    --out $RT_OUT/d13/route/confirm_d13.json
$PY $RT/retest/shim/rt_surface.py surface.sv_route report --lock $LOCK --confirm $RT_OUT/d13/route/confirm_d13.json \
    --speed $E/speed.json --pipes $E/pipes.jsonl --bitwise $E/bitwise_c7.json --out $RT_OUT/d13/route/SURFACE_RESULT_retest.md
```

A read that stops (time limit, node failure) continues once its job has ended with `RESUME=1` and the same variables
(`RESUME=1 sbatch --nice=1000 $RT/retest/jobs/d13_e5.sbatch`); the resume path re-samples from the same seeds the chunks that hold the shots left in the queues at the cut and refuses
unless every chunk digest equals the stored one. e6 can be split by p or
by chunk ranges at multiples of 100 with `P=` and `CH=`, as in the original. The analysis is submitted only after both
reads completed. Optional check of the seeds actually used (CPU, a few chunks of each part recomputed with Stim):
`PYTHONPATH=$PYTHONPATH:$RT/retest/replay/d13 $PY $RT/retest/shim/rt_surface.py d13_replay verify $RT_OUT/d13/conf OUT.json`.

#### Replay checks (review finding 10) and the dry run

Run on Alpine before the freeze, on the dry tree `<cluster-scratch>/retest/code/dry` with the files listed above
(the GPU runs used the shim, `seeds.py`, `tags.json`, `d13_env.sh`, `d13_read.sh`, `d13_e5.sbatch` and `d13_e6.sbatch` with the hashes above); results in
`retest/replay/d13/*.json`, scripts `jobs/d13_replay.sbatch`, `jobs/d13_dry_checks.sbatch` and `replay/d13/d13_replay.py`.
The dry seeds are drand round 6540000, a past round that is never a test seed.

| check | what ran | result |
|---|---|---|
| 1. original inputs reproduce the original analysis | `d13_analysis.sh` through the shim with the patch off on the original e5 / e6 reads (`cascade_w5/eval/conf`, read in place), compared with the original e8 outputs (`replay_analysis.json`) | `confirm_d13.json` equal to the original in every field except `secs` (the run time; sha256 differs for that reason only); `SURFACE_RESULT` report `.md` byte-identical (870764982e7f558a...) and its `.json` byte-identical (0be37e0ba4bfc849...); same labels, headline [C5], d = 17 "propose" |
| 2. the main process, the spawned sampling pool and the spawned M workers draw the intended seeds | `d13_replay.py seeds` through the shim with the dry seeds: `stream_shots` (spawned pool, 2 processes) and `sv_read._m_task` (spawned pool, 2 workers, worker pids differ from the main pid) sample 3 chunks of 10,000 shots at p = 0.6 % and 0.35 %, and the sha256 digests equal those of a direct Stim sample of the seeds recomputed from the beacon and `tags.json`; all 3,100 chunk seeds of both sets equal the expected ones in the main process; the pool's digest differs from the one the original seed gives (`replay_seeds_on.json`) | pass |
| 2b. the checks are not vacuous | the same probe with the patch installed in the main process only, and with no patch (`replay_seeds_control_main-only.json`, `replay_seeds_control_none.json`) | fails exactly where expected: main-only fails the spawned pool and the M workers, none fails the main process, the pool, the workers and the resume expression (`control_ok` true in both) |
| 2c. the resume path | the call sites of `eval_seed` in `sv_read.py` are lines 68 (M workers) and 211 (`prime_queues`, the resume path), both through `sc.sdata.eval_seed`; line 211's expression in the patched main process gives the intended digest; and a real interrupted read: e5's arguments at p = 0.35 %, chunks 0-199, dry seeds, job RJ11 cancelled after part 1 was journaled, resumed with `RESUME=1` as job RJ12 on another node: `resume.from_chunk` 100, queues rebuilt (SB 576 shots, BQ1 376, BQ5 21 verified bit for bit), every re-sampled chunk digest equal to the stored one, complete, kernel-cache digest unchanged (`dry_run.json`); `d13_replay.py verify` recomputes 22 stored chunk digests of both parts from the dry seeds: all equal (`dry_verify_resume_p0.0035.json`) | pass |
| 3. every other kind unchanged | 13,332 values: kinds 0-99 except 11 at d = 3, 13, 17, 29, five p and five chunks up to 9,999,999, plus kinds 1, 10, 12, 13 at d = 13, p = 0.6 % and 0.35 % over chunks 0-2999 step 7: the patched function equals the pinned function and the formula; kind 11 for another d or p, or a chunk beyond the set, raises; the pinned function's own checks still run first | pass |
| 3b. patch off | `d13_replay.py seeds` without `RT_SEEDS`: `eval_seed` is the pinned function object, kind 11 equals the formula for every chunk of both sets, the pools draw the formula's seeds (`replay_seeds_off.json`) | pass |

Dry run (the originals' command lines with `--chunks 0-1`, dry seeds, one full RTX PRO 6000 at a time; `RT_DRY_CHUNKS=0-1
PS=0.006` for e5 and `P=0.006` for e6): e5 job RJ13 (56 s) and e6 job RJ14 (29 s) read 10,000 shots at p = 0.6 %
and finished complete with the kernel-cache digest `46c6069894fa4844` at start and end (failures: SB 94, Bb 69, BQ1 68, BQ5 65,
MB 72, MP 224; S1 94, B1 68, MS 72). `eval/kcache` has the same 1,248 files, sizes and times after all dry jobs.
`dry_verify_1chunk_e5_e6.json`: the stored chunk digest equals the dry seed's. `d13_read.sh` refuses `RT_DRY_CHUNKS` without
dry seeds and dry seeds without `RT_DRY_CHUNKS` (exit 1 before anything runs). `confirm` as it runs after the beacon refuses
on size on this read ("parts cover 0-1, not 0-100"); with `RT_DRY_CONFIRM=0-1` (refused unless the seeds are dry) it
runs through load, assembly, the 20 hypotheses at p = 0.6 %, Holm and labels on the 10,000 dry shots; `report` then refuses
because the original e7 rows cover both p (`dry_analysis.json`). The real report ran in check 1. The attempt with chunks 0-199 at p = 0.6 % stopped at chunk 100
with the shim's refusal (the dry set has 100 chunks at that p), a second demonstration of the guard.


#### What the paper reports (v4, 9680d1d; Table "cascade", surface rows, main.tex 549-556 and 512-526)

| cell | Cascade alone, failures single / batched | pair (C5), failures single / batched | routed to Cascade | verdict single / batched | time, Cascade alone then pair: single (us), batched (us per round) |
|---|---|---|---|---|---|
| d = 13, p = 0.6 %, 1,000,000 shots | 4,963 / 4,953 | 4,936 / 4,926 | 10 % | better / better | 397.0 then 107.4; 10.48 then 2.34 |
| d = 13, p = 0.35 %, 30,000,000 shots | 2,406 / 2,411 | 2,373 / 2,376 | 1 % | better / better | 397.1 then 69.4; 10.48 then 1.37 |

The text states that the pair failed less often than Cascade alone at both p in both modes (0.54 % fewer at 0.6 %; 1.4 %
fewer in single mode and 1.5 % in batches at 0.35 %), was faster on average (3.70 and 5.72 times per single shot, 4.49 and
7.67 times in batches), and was slower in the tail (99.9th percentile about 490 us against 411 and 417 us).

The original `confirm_d13.json` (sha256 09869a77ec8e8d1108731a5cdc823ad13f48b55f77440ff369ca63dcedf53b31): 40
hypotheses, 31 rejected at the Holm level 0.05; the nine that were not rejected are all at 0.35 % (C1-C2, C4-C2, C1-C4,
C1-C5 in both modes, and non-inferiority of C1 in single mode, p = 0.0083). Failures per decoder (single / batched):

| p | C1 chain | C2 Cascade alone | C3 corr. PM | C4 S -> M | C5 pair | plain PM (context) |
|---|---|---|---|---|---|---|
| 0.6 % | 4,372 / 4,368 | 4,963 / 4,953 | 6,320 / 6,320 | 5,388 / 5,388 | 4,936 / 4,926 | 21,061 |
| 0.35 % | 2,506 / 2,499 | 2,406 / 2,411 | 2,909 / 2,909 | 2,502 / 2,502 | 2,373 / 2,376 | 18,936 |

Pair against Cascade alone, exact McNemar (shots where only the pair failed / only Cascade alone failed): 0.6 % single
13 / 40 (p = 2.7e-4), batched 13 / 40; 0.35 % single 0 / 33 (p = 2.3e-10), batched 0 / 35 (p = 5.8e-11); all four rejected.
Labels (`sv_route.label`), single / batched: 0.6 %: C1 better / better, C5 better / better, C4 worse / worse, C3 worse /
worse; 0.35 %: C1 level / non-inferior, C5 better / better, C4 level / level, C3 worse / worse. Verdicts of
`sv_route.verdicts`: headline chains [C5], C5 "faster and more accurate" at both p, P1 true, P2 [C5], d = 17 "propose"
(the original deferred that decision). The lock's planned power for "C5 better than C2" is 0.342 at 0.6 % and 0.999 at
0.35 %, so a fresh draw has about a one-in-three planned chance of showing "better" again at 0.6 %.

#### From the re-test's labels to the paper

The rules below are fixed before any shot is drawn.

- Unit of comparison: one cell, (p, mode, pair). The paper's "better" in a cell is repeated only by the re-test's label
  "better" (exact McNemar rejected at the Holm level of the same 40-hypothesis family, with fewer pair-only than
  Cascade-only failures). "Non-inferior" is not "better": it says the pair is shown to be at most delta_p worse, and it
  does not repeat the paper's claim. A non-significant result ("level") is not equality and is not reported as "the pair
  equals Cascade alone"; it does not repeat the claim, and it does not refute it either. "Worse" contradicts it.
- The sentence "the pair failed less often than Cascade alone at both p in both modes" counts as repeated only when all
  four cells are "better"; otherwise the paper reports the cells that repeated and the ones that did not, with their
  labels, counts and p-values. Percentages (0.54 %, 1.4 %, 1.5 %) are recomputed from the new counts and carry no verdict.
  The chain, ConvLines with correlated PM and correlated PM alone are reported with their labels in the same way.
- The non-inferiority margins stay the development-derived ones in the lock, 0.0005102 (p = 0.6 %) and 7.5e-6 (p = 0.35 %).
  They are not recomputed from the re-test shots and not rescaled.
- Timing is inherited, never re-measured. The verdicts "faster within the predefined accuracy tolerance" and "faster and
  more accurate" are those `sv_route report` computes from the re-test's labels and the original e7 rows (`pipes.jsonl`,
  `bitwise_c7.json`): the slowest of three e7 processes against Cascade alone's from the same job, and the e7 bitwise
  check. The e7r values (Cascade alone batched, 10.48 us per round, from the clean repeat of eight flagged timings) are
  display values for the paper's table only; no verdict reads them, and they stay separate from e7's verdict timings.
  Every re-test speed number is labelled as inherited from the registered timing session.

#### Cost

e5 takes 2.11 h and e6 1.37 h on one full RTX PRO 6000 each, as the originals did: 3.48 GPU-h, about 2.1 h of wall time
with both side by side. CPU use was 5.3 h in the originals, 83.5 core-h allocated (24 cores times the wall time of both);
the analysis takes about 0.1 CPU-h on `acpu`. The dry runs and replay checks before the freeze used five GPU jobs on one
full card at a time, 937 s in all (0.26 GPU-h: 56 s, 29 s, 259 s, 316 s, 277 s), and three CPU jobs of 9, 9 and 4 minutes
(about 2.6 core-h allocated).

### 3.4 Output-head comparison (A1; paper Tables head and headcmp)

Replay and dry-run records: `retest/replay/head/`.

#### What is re-tested

A fresh-shot confirmation of A1's output-head comparison: the 23 networks A1 trained and froze, read on new test shots, with
A1's thresholds, A1's fallbacks and A1's analysis, unchanged. Surface code d = 9, p = 0.6 %, 1,000,000 new test shots (100
chunks of 10,000); BB72 p = 0.3 %, 500,000 new test shots (50 chunks). Arms a-e (S-VA, S-V, S-L, S-P, S-PA) and f-i (B-JT, B-J,
B-P, B-PA), each with seeds 1 and 2, the three ramp reruns (seed 1: S-VA, S-P, S-PA), the deployed d = 9 network r764, and the
second BB72 network r12. Nothing is retrained, reselected or re-thresholded.

Changed, and only this: the seeds of the two test kinds (surface `test`, BB72 `a1test`) in `a1.sets.seed_of`, and the
directories (new sets, new reads, new analysis output). Every other seed kind (cal, dev, train, dry, ...) is A1's own. The
code is A1's pinned tree `rev_a1/code_at/2e69738e74be71c5cfbcb6661ecfa28a54048877` (PINSUMS
`0098f4b42c153b69d34d7594c85898da96ab1c89285d0442c767a458a5d4b94e`; `pin_check` runs at the start of every job and the pinned
tree is never modified). Reads: `a1.read`, fp32, TF32 off, batch 4,096, the first 2,000 shots again at batch one (a differing
decision fails the read). Fallbacks: correlated PyMatching on every surface test shot; full-DEM Relay-BP (relay-bp 0.2.2,
`RelayDecoderF32` with `qnd.teachers.CONFIGS['relay']`, a new decoder per shot seeded with the shot's index in its set, 10,000 c + r
for native chunk c, row r) on every BB72 test shot. Devices as in A1: surface reads on two H200 2g.35gb slices, BB72 reads on one
RTX PRO 6000 1g.24gb slice (A1 recorded the device in every read; its analysis does not check it).

Seeds. Surface test chunk j (0-99) takes `seeds.chunk_seeds("head-surface")[j]` (TAG `convlines retest surface d9 p0.006`, beacon
indices 0-99). BB72 native chunk j (0-49, kept so that Relay-BP's per-shot seeds stay as frozen; review finding 5) takes
`seeds.chunk_seeds("head-bb72")[j]` (TAG `convlines retest bb72 p0.003`, beacon indices 20-69; indices 0-19 belong to Table I's
BB72-3 set). A collision drops an index and takes the next (the frozen rule of `retest/seeds.py`).

Thresholds. A1 froze the rule, not the numbers: tau_q is the linear numpy quantile of each network's confidence on 200,000
calibration shots, computed inside `a1.analyze`. The re-test gives the analysis A1's own calibration reads (23 files, symlinked
read-only into the new reads directories, sha256 in `retest/head/cal_reads.sha256`, checked by every job) and the new test reads
only. No calibration shot is drawn or read. The new `pair_curves.csv` must reproduce the 18 original tau (and the full q grid)
digit for digit; `head_analysis.sh` checks it (`retest/head/check_taus.py` against `retest/head/taus.csv`, 126 rows) and fails
the job on any difference. The pinned tree has no set manifest (A1 committed `sets_manifest.csv` after the freeze), so the 40 calibration
chunks' rows (sha256 per chunk) come from `retest/head/cal_chunks.csv`, taken from A1's `rows.csv` (sha256 `82d2004a...`), and the shim lists
no row of an original test chunk.

#### The 23 networks (sha256 of the file each read loads, `final_ema.pt`; list: `retest/head/networks.sha256`, checked by every reads job)

Verified on Alpine against the disk on 2026-10-12 (23 of 23). Paths are under <cluster-scratch>/.

| network | sha256 of final_ema.pt | file |
|---|---|---|
| a1_S-VA_s1 | 50c2b30d79c03c1987d9f63386285a120cdcf6c328091205d75a15704c8ea5bf | rev_a1/ckpt/surface/a1_S-VA_s1/final_ema.pt |
| a1_S-VA_s2 | c35976a52d40aa8448d6523359f1000d9dcf7f4d996fbe0ae274572dc1b0c763 | rev_a1/ckpt/surface/a1_S-VA_s2/final_ema.pt |
| a1_S-V_s1 | bda0ca608ab854d84783f2611084de587c588e5be32120f7341ccb83af9ac43e | rev_a1/ckpt/surface/a1_S-V_s1/final_ema.pt |
| a1_S-V_s2 | abbbec7b819ed66c9644764d8ece78568729a69b6a9ed25f703ee88ae7f01048 | rev_a1/ckpt/surface/a1_S-V_s2/final_ema.pt |
| a1_S-L_s1 | ece896119a0701383764854f16fb8a812973d9f4db6f91196c411de162532547 | rev_a1/ckpt/surface/a1_S-L_s1/final_ema.pt |
| a1_S-L_s2 | 58649e88aa327b771fdfed69e02c360f3cb1c1a06614777667aedb083e12dd72 | rev_a1/ckpt/surface/a1_S-L_s2/final_ema.pt |
| a1_S-P_s1 | 4328c5373321c8c20236ba483a47739ae452821fc119ec7b7a3e65614a885aef | rev_a1/ckpt/surface/a1_S-P_s1/final_ema.pt |
| a1_S-P_s2 | 2c5e5c40015e50f1660c050b5d0b2a0a2fa058160e59056a6183afebc57049cf | rev_a1/ckpt/surface/a1_S-P_s2/final_ema.pt |
| a1_S-PA_s1 | e3cb8d8f06a49414a277a2040548299a4077b2a527524bceabb8a3e5f73be3f1 | rev_a1/ckpt/surface/a1_S-PA_s1/final_ema.pt |
| a1_S-PA_s2 | 06c377fdecebabc6c2a7cfe3de5abac2955395e268a3364e3e9537d4fcafa73e | rev_a1/ckpt/surface/a1_S-PA_s2/final_ema.pt |
| a1ramp_S-VA_s1 | 207398cdd135798f83b92e579aa44af215061cccff6bdb48c9d5ea37acbf9e26 | rev_a1/ckpt/surface/a1ramp_S-VA_s1/final_ema.pt |
| a1ramp_S-P_s1 | 485ffd92b00a6b709ae0ccf2f26b4391a6174ac416d7bebf061fe776486e56b2 | rev_a1/ckpt/surface/a1ramp_S-P_s1/final_ema.pt |
| a1ramp_S-PA_s1 | fbcecad020f83bbccfdd721f9991390576bec2d09b30d8850e3da64045834a4e | rev_a1/ckpt/surface/a1ramp_S-PA_s1/final_ema.pt |
| d9_b3t_r764 | 462ffb294ab2022bc1d4b276227a62270e6e4dde4fab186958c11881fcfffb40 | ising_arch/ckpt/d9_b3t_r764/final_ema.pt |
| a1_B-JT_s1 | 832ac2f7ca742debd4fda13c2925740896a7d96a1ce2e0308a022042b30426bd | rev_a1/ckpt/bb72/a1_B-JT_s1/final_ema.pt |
| a1_B-JT_s2 | a3dd2d6a092dbb7bab9f1ff4c7d47c9d6eff8735706703298fe78319b198b8d6 | rev_a1/ckpt/bb72/a1_B-JT_s2/final_ema.pt |
| a1_B-J_s1 | faa7df4e7b63a324ffca9f0c1ae19befb3e2c30c6774a4d80f7efe82ec4ca759 | rev_a1/ckpt/bb72/a1_B-J_s1/final_ema.pt |
| a1_B-J_s2 | 034b59ac16914ea6d035a2c6a7051545365a42f75d42bccadfd7d1ab37ff0449 | rev_a1/ckpt/bb72/a1_B-J_s2/final_ema.pt |
| a1_B-P_s1 | 22fe7bfddd424189b05389f4605b0d746a00040838f977894192c6ba21261b09 | rev_a1/ckpt/bb72/a1_B-P_s1/final_ema.pt |
| a1_B-P_s2 | df2020dd4bfce4f20ba91f75b7226479dcf770a21bfef55af928c743a817c82b | rev_a1/ckpt/bb72/a1_B-P_s2/final_ema.pt |
| a1_B-PA_s1 | dc3ce74f3b11c5344909f6a2b8788438bb66b82e367a0d985ebc90db91340f67 | rev_a1/ckpt/bb72/a1_B-PA_s1/final_ema.pt |
| a1_B-PA_s2 | 8d77b206e487f903844fe64a92e0764d880f116e479c774ac59961775bef0a90 | rev_a1/ckpt/bb72/a1_B-PA_s2/final_ema.pt |
| r12 | f2821c327b4e233a4dd1a75c465ceb261448d981c2b55166862fc34a99ec3e96 | bb/ckpt/r12/final_ema.pt |

#### The 18 thresholds (tau, full precision; from A1's `pair_curves.csv`; the re-test's must be identical)

The quantile levels q are 0.5, 1, 2, 4, 8, 16 % (surface) and 0.25, 0.5, 1, 2, 3, 5 % (BB72); q = 0 has tau = -inf (nothing
routed). The 3 ramp reruns and the 2 references use their own calibration reads inline; their tau are not stored in A1's outputs and
are recomputed from the same linked reads.


surface: tau at q = 0.5 %, 1 %, 2 %, 4 %, 8 %, 16 %

| network | q = 0.5 % | q = 1 % | q = 2 % | q = 4 % | q = 8 % | q = 16 % |
|---|---|---|---|---|---|---|
| S-VA seed 1 | 0.27803234070539473 | 0.5436882919073105 | 1.0681998443603515 | 2.0069144630432127 | 3.364422664642334 | 5.149870300292969 |
| S-VA seed 2 | 0.2998588015139103 | 0.5925747323036193 | 1.148993935585022 | 2.1265892601013183 | 3.566755657196045 | 5.338747615814209 |
| S-V seed 1 | 0.42339451745152473 | 0.8288900077342987 | 1.5361034297943115 | 2.656285228729248 | 4.143643932342529 | 5.9632103157043455 |
| S-V seed 2 | 0.40853380888700486 | 0.795220462679863 | 1.5086983561515808 | 2.670178394317627 | 4.14296932220459 | 5.939687843322754 |
| S-L seed 1 | 0.24145673744380475 | 0.49369946151971816 | 0.9830702769756318 | 1.8205783939361573 | 3.131095123291016 | 4.835130653381348 |
| S-L seed 2 | 0.27265773445367814 | 0.5576058578491211 | 1.1030781888961791 | 2.0352292442321778 | 3.39927131652832 | 5.108875160217285 |
| S-P seed 1 | 0.0010830852016806604 | 0.002144472226500511 | 0.004261773228645325 | 0.008501394689083099 | 0.01655760109424591 | 0.03118885487318039 |
| S-P seed 2 | 0.305126419365406 | 0.6170193231105805 | 1.1946486449241638 | 2.1595022678375244 | 3.580737714767456 | 5.379673442840576 |
| S-PA seed 1 | 0.2836950650811195 | 0.573679501414299 | 1.113765025138855 | 2.0334872436523437 | 3.4286936473846437 | 5.19013484954834 |
| S-PA seed 2 | 0.27101158171892165 | 0.5441883444786072 | 1.0372558975219728 | 1.9235230016708373 | 3.302904977798462 | 5.071369342803955 |

bb72: tau at q = 0.25 %, 0.5 %, 1 %, 2 %, 3 %, 5 %

| network | q = 0.25 % | q = 0.5 % | q = 1 % | q = 2 % | q = 3 % | q = 5 % |
|---|---|---|---|---|---|---|
| B-JT seed 1 | 0.42453680992126464 | 0.9231317043304443 | 1.958809986114502 | 3.7261102294921877 | 4.9190000247955314 | 6.532889080047608 |
| B-JT seed 2 | 0.43603260278701783 | 0.9391989421844482 | 1.967349328994751 | 3.696219310760498 | 4.942891883850097 | 6.584147977828979 |
| B-J seed 1 | 0.4297345304489136 | 0.8530500602722167 | 1.8069829368591308 | 3.5298247146606445 | 4.67027063369751 | 6.19082350730896 |
| B-J seed 2 | 0.38675675868988035 | 0.8747403335571289 | 1.885595169067383 | 3.604345932006836 | 4.727169618606568 | 6.249257421493531 |
| B-P seed 1 | 0.010280008986592292 | 0.022622090484946965 | 0.05615326855331659 | 0.17060681194067 | 0.4153361001610756 | 1.5282581806182862 |
| B-P seed 2 | 0.011110587446019053 | 0.023602476501837374 | 0.05451095040887594 | 0.15555013239383697 | 0.36722200125455795 | 1.4377883791923525 |
| B-PA seed 1 | 0.013846821028273552 | 0.030318295881152153 | 0.07215999446809292 | 0.23609422534704208 | 0.6296982812881463 | 2.4751815438270586 |
| B-PA seed 2 | 0.014986583408899605 | 0.031606292631477115 | 0.07794452980160713 | 0.27878439366817476 | 0.7932258403301238 | 2.874535393714905 |

#### Commands, as they will run after the beacon

Files (all under `retest/`, deployed with the freeze to `$RT`): `shim/rt_a1.py`, `jobs/head_job.sbatch` (A1's `job.sbatch` with
its `pin_check`, environment and venv choice, plus the re-test's paths and `--nice=1000`), `jobs/head_common.sh`,
`jobs/head_{draw,reads,fallback_pm,fallback_relay,analysis}.sh` (A1's `draw_sets.txt`, `reads_main.txt`, `fallback_pm.txt`,
`fallback_relay.txt`, `analysis_main.txt` through the shim), `jobs/head_submit.sh`, `jobs/head_replay.sh`, `head/{networks.sha256,
cal_reads.sha256,cal_chunks.csv,taus.csv,check_taus.py}`, `replay/head/*`.

On the Alpine login node, after `retest/run/{beacon,seeds}.json` exist:

```
export RT=<cluster-scratch>/retest/code/freeze
export RT_BEACON=<cluster-scratch>/retest/run/beacon.json RT_SEEDS=<cluster-scratch>/retest/run/seeds.json
export RT_OUT=<cluster-scratch>/retest/out
bash $RT/retest/jobs/head_submit.sh draw      # T + 5 min: head_draw (acpu, -c 4, 16 GB, 30 min), about 1 min
# ... commit 1 (beacon, seeds, draw manifest $RT_OUT/head/sets/rows.csv) before any read ...
bash $RT/retest/jobs/head_submit.sh reads     # T + 20 min: the 7 other jobs below, analyses depend on reads + fallback (afterok)
```

`head_submit.sh reads` submits (every job `--nice=1000`; env passed with `--export=ALL,RT=..,RT_BEACON=..,RT_SEEDS=..,RT_OUT=..,CMD=..`):

| job | resources | CMD | what runs |
|---|---|---|---|
| head_reads_s0 | `-p ah200 --qos=gpu-normal --gres=gpu:h200_2g.35gb:1 -c 8 --mem=48G -t 01:00:00` | `head_reads.sh` CODE=surface PARTS=2 PART=0 | 7 of the 14 surface networks, test set |
| head_reads_s1 | same | PART=1 | the other 7 |
| head_reads_b | `-p artxpro6000 --qos=gpu-normal --gres=gpu:rtx_pro_6000_1g.24gb:1 -c 8 --mem=48G -t 00:45:00` | `head_reads.sh` CODE=bb72 | the 9 BB72 networks, test set |
| head_pm | `-p acpu --qos=cpu-normal -c 32 --mem=64G -t 00:30:00` | `head_fallback_pm.sh` | correlated PM on 1,000,000 shots |
| head_relay | `-p acpu --qos=cpu-normal -c 32 --mem=48G -t 01:30:00` (RELAY_CPUS, RELAY_T to change) | `head_fallback_relay.sh` | Relay-BP on 500,000 shots |
| head_an_s | `-p acpu --qos=cpu-normal -c 4 --mem=32G -t 00:30:00`, afterok reads_s0, reads_s1, pm | `head_analysis.sh` CODE=surface | analysis, output `$RT_OUT/head/analysis_surface/` |
| head_an_b | same, afterok reads_b, relay | `head_analysis.sh` CODE=bb72 | output `$RT_OUT/head/analysis_bb72/` |

Inside the jobs (A1's own commands, each through the shim: `$PYS` = `apptainer exec --nv -B <cluster-scratch-root>
$I/pyfull.sif $V/bin/python $RT/retest/shim/rt_a1.py`, `$I` = <cluster-scratch>/ising, `$V` = `$I/venv_gpu`, or bb's
`venv_cu128` on a Blackwell card, as A1's `job.sbatch`; the Relay-BP job uses bb's `venv/bin/python` without `--nv`):

```
# head_draw.sh (FREEZE = PIN = 2e69738e...; CH_surface=0-99, CH_bb72=0-49)
cal_links                        # symlink A1's 40 calibration chunks into $RT_OUT/head/sets and its 23 cal reads into head/reads/<code>/
$PYS a1.sets draw --code surface --kind test  --chunks 0-99 --freeze $PIN --rows $RT_OUT/head/sets/rows.csv
$PYS a1.sets draw --code bb72    --kind a1test --chunks 0-49 --freeze $PIN --rows $RT_OUT/head/sets/rows.csv
# head_reads.sh (R=$RT_OUT/head/reads/$CODE; C = A1's checkpoint root; P=_p$PART for the surface parts)
cal_check; net_check             # 23 cal reads and 23 networks equal their frozen sha256
$PYS a1.preflight --code $CODE --decisions $A/rule --ckpt-root $C --reads $R --out $R/reads$P.json   # plans test reads only (cal kept)
$PYS a1.read --code $CODE --many $R/reads$P.json
$PYS a1.preflight ... --out $R/reads_again$P.json            # must find nothing pending
# head_fallback_pm.sh / head_fallback_relay.sh
$PYS a1.fallback pm --kind test --chunks 0-99 --out $R/fallback_test.npz --workers $SLURM_CPUS_PER_TASK          # R=head/reads/surface
$PYR $SHIM a1.fallback relay --code bb72 --kind a1test --chunks 0-49 --out $R/fallback_test.npz --workers $SLURM_CPUS_PER_TASK
# head_analysis.sh CODE
$PYS a1.preflight ... --out $R/reads_final.json              # no read pending
$PYS a1.check checks --code $CODE --runs <a1_<arm>_s<seed> and a1ramp_...> --out $R/checks.json
$PYS a1.analyze --config data/a1/analysis_$CODE.json --reads $R --checks $R/checks.json --out $RT_OUT/head/analysis_$CODE
python check_taus.py $CODE $RT_OUT/head/analysis_$CODE/pair_curves.csv head/taus.csv .../taus_check.json     # the 18 tau, digit for digit
```

The shim refuses (message `rt_a1: ...`, no output): a path of `--out`, `--reads`, `--rows`, `--many`, `--checks` outside
`$RT_OUT/head`; any `--kind` but the two test kinds; a `--many` job list with a calibration read; `--sets dry`; an analysis config
whose sha256 is not A1's frozen one (surface `328e54c0...`, BB72 `652dfe39...`); a dry beacon without `RT_DRY` and `RT_DRY` without a
dry beacon. A finished fallback output is kept, never replaced; the analysis refuses to run twice into one directory.

#### Replay checks (review finding 10; `jobs/head_replay.sh` under `head_job.sbatch`, Alpine job RJ15, 9 min 35 s, status 0; records `retest/replay/head/replay{1_analysis,2_seeds,3_kinds}.json`)

The job runs with the dry seeds (a past drand round, flagged `"dry": true`), RT_DRY=1 and its own `RT_OUT`
(`<cluster-scratch>/retest/replay/head/out`); nothing it writes lies in A1's tree or in the real output directory. The shim
file replayed has sha256 `f590fe89b4f1cc96...` (full hash in `notes/build_head.md`).

1. The original inputs through the shim, patch off (RT_SEEDS unset; the shim is an identity launcher): `a1.analyze --config
   data/a1/analysis_<code>.json --reads $A/reads/<code> --checks $A/reads/<code>/checks.json` into a new directory, then compared with
   A1's own output directory `$A/analysis/<code>`. Surface: 11 of 11 files byte-equal (analysis.json, comparisons.csv,
   pair_curves.csv, pair_curves.png, ramp.csv, references.csv, sizes.csv, standalone.csv, summary.md, train_checks.csv, and
   reads_meta.json regenerated with analysis_main.txt's own line). BB72: 10 of 10 (no ramp.csv). `pair_curves.csv` carries all 70 (surface,
   10 networks x 7 q) and 56 (BB72, 8 networks x 7 q) tau digit for digit against `retest/head/taus.csv`; the 14 + 10 verdict rows
   (7 x 2 and 5 x 2) are A1_RESULT's; the sha256 prefixes of analysis.json, comparisons.csv and pair_curves.csv are the published
   ones (surface 10efcd2a / 6dabba9e / fd62a616, BB72 55ef1f85 / d315de5a / 6cb1c181). No field differs, not even a timestamp or a path.
2. With dry seeds, the patch on:
   - Main process and spawned workers: a module run by `rt_a1.py` with a 3-worker spawn pool: the main process and all 3 workers
     (3 distinct pids, 9 tasks) return, for surface `test` chunks 0-99 and BB72 `a1test` chunks 0-49, exactly the seeds of the rule
     recomputed from the dry beacon (re-implemented in the checker, not imported: surface beacon indices 0-99, BB72 20-69, none
     dropped), none of A1's old seeds; every process has the patch.
   - Set generation: the production command `a1.sets draw` (2 chunks per code) stores in each chunk, in the rows file and in the
     arrays the seed of the rule (surface chunks 0, 1: beacon indices 0, 1; BB72 chunks 0, 1: beacon indices 20, 21, under the file
     names a1_bb72_a1test_000/001); the stored det/obs arrays equal stim sampled directly with that seed, and the seed is not A1's.
   - Resume path: draw again with rows present: 4 of 4 chunks KEPT, none drawn; rows file removed: 4 of 4 RECOVERED from the saved
     bytes (their seed checked against the patched function), chunk sha256 unchanged; rows file removed and the patch off: refused
     (`ValueError: a1_surface_test_000.npz is not chunk 0 ... (got, want): {'seed': (8724833424388466509, 72090000)}`), so the resume path is
     bound to the intended seeds.
   - Relay-BP fallback: `a1.fallback.relay` (the function the command calls) on 200 dry shots, the first 100 of native chunks 0 and 1,
     4 spawned workers, a recorder wrapped round `relay_bp.RelayDecoderF32` in every process (the replay launcher imports the shim,
     as every spawned worker does): the recorded decoder seeds are exactly 0-99 and 10,000-10,099 (the shot's index in its set,
     10,000 c + r), although the beacon indices in play are 20 and 21; a second run with 2 workers and other task sizes gives identical
     answers (the shot's index fixes the answer).
   - The shim's guards (11 runs): each refusal comes from the shim and exits non-zero: `--kind cal`, a rows path outside the re-test,
     A1_SETS pointing at A1's sets, a dry beacon without RT_DRY, RT_DRY with a non-dry beacon, a `--many` list with a calibration read,
     a read whose output lies under A1's tree, `--sets dry`, an analysis config that is not A1's frozen one; the two controls (the
     frozen config, a valid draw) pass the guard.
3. Every other seed kind: `a1.sets.seed_of` with the patch on equals the original function for 1,111 values (surface cal, cal13, test13,
   dry; BB72 dev, cal, smoke, monitor, a1cal, a1dry, test, test4 and their errors; chunks 0-99 and 9,999), the two test kinds equal the
   rule's seeds (and chunk 9,999 is refused, never an old seed); with the patch off the two test kinds equal A1's formulas for
   every chunk and the 190 rows of A1's `rows.csv` (the git manifest's equal), and `a1.seeds.surface_seed`, `bb.shots.seed_of` are unchanged.

#### Dry run (dry seeds, one chunk per code; eight Alpine jobs, record `retest/replay/head/dry_run.json`)

`RT=<cluster-scratch>/retest/code/dry`, `RT_BEACON`/`RT_SEEDS` = `retest/dry/{beacon,seeds}_dry.json`, `RT_DRY=1`,
`RT_OUT=<cluster-scratch>/retest/dry/head_out`, then `bash $RT/retest/jobs/head_submit.sh all` (the real run's script, with the draw as
a dependency instead of the commit in between). The outputs of a first chain were deleted by mistake before they were checked; this is the second.
All of A1's pipeline ran, with all 23 networks, not a sample:

| step | result |
|---|---|
| draw (acpu, 4 cores) | surface `a1_surface_test_000.npz` seed 8724833424388466509 (beacon index 0), BB72 `a1_bb72_a1test_000.npz` seed 7820737992942139394 (beacon index 20), both the dry seeds file's; 40 calibration chunks and 23 calibration reads symlinked in; `CAL CHECK` passed; 14 s |
| reads (H200 2g.35gb x 2, RTX PRO 6000 1g.24gb) | preflight planned only test reads: surface 14 pending and 14 kept, BB72 9 pending and 9 kept; after the reads 0 pending (28 and 18 kept); 14 + 9 networks read (7 per H200 slice), the batch-one check 0 differing decisions (largest confidence difference 3.7e-5 surface, 7.8e-5 BB72); devices as A1 (MIG 2g.35gb, MIG 1g.24gb); 39-77 s per job |
| fallbacks (acpu, 32 cores) | correlated PM on 10,000 shots: 138 failures, single and batch answers equal; Relay-BP on 10,000 shots: 66 failures, relay-bp 0.2.2, mean 0.150 s per shot (A1 0.130), 5 of 32 workers busy; MaxRSS 2.4 GB (PM) and 0.9 GB (Relay-BP), well inside 64 and 48 GB; wall 5.3 and 6.4 min, most of it the start-up of 32 spawned workers on a loaded node |
| analysis, both codes | ran to the end, no refusal on size (10,000 test shots): every table (standalone, pair_curves, comparisons, references, ramp (surface), train_checks, sizes), the figure, `reads_meta.json` (29 and 19 files), Holm over 14 and 10 tests; the cal digests are A1's (`6313c8e9...`, `fd05bc8b...`); `taus_check` 70 of 70 and 56 of 56 rows digit for digit |

The dry numbers are not results (10,000 shots, dry seeds): the verdicts differ from A1's, as 10,000 shots would. A1's tree: no file
newer than the run's start (`find`, logs excluded); the symlinks point into `rev_a1/reads` and `rev_a1/sets`, which no job writes.
What can only be seen at full size: the real draw (150 chunks), Relay-BP at 500,000 shots with every worker busy, and the analyses' refusal paths
on the real sets' digests; none is expected to differ in kind.

#### What the paper reports (v4 `9680d1d`, from A1's frozen outputs `revision/a1_results/`)

Tables head (failures per million fresh test shots, alone at q = 0 and the pair at the registered q, two seeds) and headcmp, the
text numbers they feed and the verdicts, all to be replaced by the re-test's values whatever they are (RETEST.md section 7).

Table head (surface: 1,000,000 shots, q = 1 %, correlated PM alone 13,080 failures; BB72: 500,000 shots, q = 2 %, Relay-BP alone
5,382 per million = 2,691 failures). Verdict column of the paper: a -, b better than a, c worse than a and b, d worse than b and
inconclusive against a, e inconclusive against a and d; f -, g inconclusive against f, h worse than f and g, i worse than f and
better than h. Dagger: d seed 1 did not train (train_checks ratio 1.000016).

Surface code, d = 9, p = 0.6 %, registered q = 1 %

| arm | A1 name | alone seed 1 | alone seed 2 | pair seed 1 | pair seed 2 |
|---|---|---|---|---|---|
| a | S-VA | 15,252 | 13,975 | 13,239 | 12,217 |
| b | S-V | 10,638 | 10,881 | 9,942 | 10,069 |
| c | S-L | 17,088 | 15,059 | 14,628 | 13,076 |
| d | S-P | 478,846 | 13,750 | 474,111 | 12,017 |
| e | S-PA | 14,917 | 15,719 | 12,981 | 13,610 |

BB72, p = 0.3 %, registered q = 2 %

| arm | A1 name | alone seed 1 | alone seed 2 | pair seed 1 | pair seed 2 |
|---|---|---|---|---|---|
| f | B-JT | 7,044 | 6,864 | 4,268 | 4,306 |
| g | B-J | 7,046 | 7,110 | 4,310 | 4,262 |
| h | B-P | 34,048 | 35,360 | 18,524 | 19,644 |
| i | B-PA | 28,146 | 26,216 | 12,816 | 11,046 |

Table headcmp (per row: X - Y in failures per million with Newcombe 95 %, adjusted p = Holm over 14 surface and 10 BB72 tests,
q = 0 and the registered q as one family; "readout" = the paper's "head", "whole design" = "both"):

Surface code:

| id | X vs Y | factor | q (%) | seed 1, X - Y [95 %] | seed 2, X - Y [95 %] | adjusted p | verdict |
|---|---|---|---|---|---|---|---|
| R1 | d vs b | readout | 0 | +468,208 [+467,209, +469,206] | +2,869 [+2,693, +3,046] | 3.5e-225 | worse |
| R1 | d vs b | readout | 1 | +464,169 [+463,172, +465,165] | +1,948 [+1,800, +2,097] | 1.8e-147 | worse |
| R2 | c vs b | readout | 0 | +6,450 [+6,235, +6,667] | +4,178 [+3,981, +4,376] | <1e-300 | worse |
| R2 | c vs b | readout | 1 | +4,686 [+4,499, +4,874] | +3,007 [+2,839, +3,176] | 1.1e-275 | worse |
| R3 | e vs a | readout | 0 | -335 [-532, -138] | +1,744 [+1,548, +1,940] | 1 | inconclusive |
| R3 | e vs a | readout | 1 | -258 [-428, -88] | +1,393 [+1,224, +1,562] | 1 | inconclusive |
| U1 | b vs a | supervision | 0 | -4,614 [-4,808, -4,421] | -3,094 [-3,271, -2,918] | 3.6e-264 | better |
| U1 | b vs a | supervision | 1 | -3,297 [-3,465, -3,130] | -2,148 [-2,301, -1,996] | 6.7e-171 | better |
| U2 | e vs d | supervision | 0 | -463,929 [-464,935, -462,923] | +1,969 [+1,771, +2,167] | 1 | inconclusive |
| U2 | e vs d | supervision | 1 | -461,130 [-462,132, -460,128] | +1,593 [+1,422, +1,764] | 1 | inconclusive |
| W1 | c vs a | whole design | 0 | +1,836 [+1,611, +2,062] | +1,084 [+883, +1,285] | 3.3e-25 | worse |
| W1 | c vs a | whole design | 1 | +1,389 [+1,194, +1,584] | +859 [+687, +1,032] | 1.2e-21 | worse |
| W2 | d vs a | whole design | 0 | +463,594 [+462,587, +464,600] | -225 [-414, -36] | 1 | inconclusive |
| W2 | d vs a | whole design | 1 | +460,872 [+459,870, +461,874] | -200 [-361, -39] | 1 | inconclusive |

BB72:

| id | X vs Y | factor | q (%) | seed 1, X - Y [95 %] | seed 2, X - Y [95 %] | adjusted p | verdict |
|---|---|---|---|---|---|---|---|
| R1 | h vs g | readout | 0 | +27,002 [+26,547, +27,463] | +28,250 [+27,787, +28,719] | <1e-300 | worse |
| R1 | h vs g | readout | 2 | +14,214 [+13,874, +14,560] | +15,382 [+15,031, +15,739] | <1e-300 | worse |
| R2 | i vs f | readout | 0 | +21,102 [+20,698, +21,511] | +19,352 [+18,965, +19,744] | <1e-300 | worse |
| R2 | i vs f | readout | 2 | +8,548 [+8,277, +8,824] | +6,740 [+6,492, +6,993] | <1e-300 | worse |
| U1 | g vs f | supervision | 0 | +2 [-175, +179] | +246 [+71, +421] | 1 | inconclusive |
| U1 | g vs f | supervision | 2 | +42 [-47, +131] | -44 [-133, +44] | 1 | inconclusive |
| U2 | i vs h | supervision | 0 | -5,902 [-6,306, -5,499] | -9,144 [-9,545, -8,745] | 1.3e-182 | better |
| U2 | i vs h | supervision | 2 | -5,708 [-6,085, -5,333] | -8,598 [-8,971, -8,228] | 2.3e-196 | better |
| W1 | h vs f | whole design | 0 | +27,004 [+26,550, +27,464] | +28,496 [+28,032, +28,966] | <1e-300 | worse |
| W1 | h vs f | whole design | 2 | +14,256 [+13,916, +14,602] | +15,338 [+14,987, +15,695] | <1e-300 | worse |

Verdicts of A1 (identical at q = 0 and at the registered q): surface R1 (d vs b) worse, R2 (c vs b) worse, R3 (e vs a)
inconclusive, U1 (b vs a) better, U2 (e vs d) inconclusive, W1 (c vs a) worse, W2 (d vs a) inconclusive. BB72 R1 (h vs g) worse, R2
(i vs f) worse, U1 (g vs f) inconclusive, U2 (i vs h) better, W1 (h vs f) worse. Derived text numbers (computed from the same tables): 3.8-5.0 times
(abstract, RQ2) = ratio of the fail_x / fail_y columns of the BB72 R1 and R2 rows at q = 0 (R1 4.83, 4.97; R2 4.00, 3.82), 2.6-4.6
times at q = 2 % (R1 4.30, 4.61; R2 3.00, 2.57); 4,262-4,310 = the BB72 pairs of f and g at q = 2 %, about 20 % below Relay-BP alone
5,382; 23-24 % = 1 - pair(b, q = 1 %)/13,080 for the two seeds (9,942 and 10,069); 12,293 and 10,976 = r764 alone and at q = 1 %,
7,126 and 4,336 = r12 alone and at q = 2 % (references.csv); ramp reruns 1,170 / 682 / 1,027 / 531 (ramp.csv diff_pm of S-P and S-PA
against the ramp rerun of S-VA at q = 0 and 1 %); routed shares within 0.1 pp of q for every network (largest deviation 0.0987 pp,
surface S-L seed 1 at q = 16 %; 0.0788 pp, BB72 B-J seed 2 at q = 5 %: only just inside, so the re-test's shares may exceed 0.1 pp
and the sentence is then reworded to the measured value); appendix sizes (params and read-out multiply-adds belong to the networks and stay as they are); the train checks and the rule for the second seed (training and validation records: not re-tested, unchanged).

#### Verdict-to-paper rule for the head comparison (frozen with the freeze, review finding 8)

- The two verdicts of every comparison, at q = 0 and at the registered q, are reported separately: one verdict never stands in
  for the other, and a sentence that holds "at both q" in the paper may be kept only if the re-test gives the same verdict at both.
- "Better" or "worse" is repeated only by the same verdict (Holm-adjusted p < 0.05, both seeds rejecting in the same direction).
  "Inconclusive" does not repeat "better" or "worse" and is not equality: where the paper says "better" and the re-test is
  inconclusive, the paper's claim reads "not confirmed on the fresh shots". Where the paper says inconclusive and the re-test
  says "better" or "worse", that is a new result, reported with the original named as an earlier registered test.
- A verdict that flips sign (better to worse) is reported as such, with both tests' numbers.
- The paper's claims that depend on the comparison are re-checked one by one in RETEST_RESULTS.md: "3.8-5.0 times", "about 20 %
  below Relay-BP alone", "23-24 % less often than correlated PM alone", the routed-share sentence, the dagger (d seed 1 did not
  train: the re-test reads the same network, so the same run fails to train, and its comparisons are reported as they come).
- Descriptive extras (ramp table, references r764 and r12, the pair-curve figure) are replaced by the re-test's values and stay
  outside the Holm family. The original test stays in the repository as the earlier registered test.

#### Cost

GPU, real run (A1's timings, test reads only; cal reads reused): surface 2 x about 17 min on H200 2g.35gb slices (A1: 20.9 min
with the calibration reads) = 0.57 slice-hours; BB72 about 8 min on one RTX PRO 6000 1g.24gb slice = 0.13 slice-hours (A1 9.6
min); about 0.7 slice-hours in all. CPU, real run: Relay-BP 18.1 core-hours (0.1300 s per shot x 500,000), correlated PM 0.44,
draw about 0.1, two analyses about 0.2, preflights and checks about 0.1: about 19 core-hours; wall about 40-60 min after the
reads' submission if the Relay-BP job starts at once (32 cores). Build, replay and dry runs (not part of the re-test): about 0.23 slice-hours of GPU (first dry chain 0.19, second 0.04) and about 11 allocated core-hours (two replay runs 8 cores x 9.6 min = 2.6; two dry chains about 8.4, most of it 32-core fallback jobs that start 32 workers for 5 busy ones).

## 4. Timing: inherited from the registered sessions

No time is measured again; the paper's times stay. The composites of the F, S, G-4R and G-4F cells change through
the routed fraction only, by a frozen adapter (`retest/shim/rt_bb.py --prepare`, review findings 1-2):
for every CPU-term row of the test split, one per registered timing process,

    value_ms = registered value_ms x (re-test routed fraction / registered routed fraction)

that is, the re-test's routed fraction times that process's full-precision mean fallback time per routed shot; the
analysis keeps the slowest process, as frozen. The derived rows (`rows_cpu_retest.csv`, `rows_cpu_S_retest.csv`) sit
beside the untouched originals, and each derived row's note keeps the registered value and fraction. With the
registered shots the adapter reproduces every original row bit for bit (section 6).

| test | reused, unchanged | recomputed |
|---|---|---|
| BB72 | GPU term and handoff (H100 session pre); cudaq-qec bars (H100 session post); sector Relay-BP and ldpc bars and full-DEM Relay-BP's own mean (M4 session B); mean fallback time per routed test shot of each process (session B for F, session S for S) | each F and S cell's CPU term |
| gross | H100 session s4 (GPU term, handoff, cudaq-qec bar); M4 session s4 (sector, ldpc, full-DEM own mean; per routed test shot for G-4R and G-4F) | the CPU terms of G-4R and G-4F |
| Cascade BB72 | D1-S, job CJ73 (36 processes on one RTX PRO 6000, speed shots of kind 12) | nothing |
| Cascade d = 13 | e7 (48 pipelines on kind-12 speed shots) for the verdicts; e7r (Cascade alone batched) for the paper's display values, kept apart | nothing |
| head comparison | no time is reported | |

Why the reuse holds: a time per shot is a property of the decoder, its settings and the host, averaged over shots of
one circuit and error rate; the registered sessions averaged over shots of the same distribution, and their own
calibration-against-test repeats moved by a few percent (for example 1.598 against 1.617 ms, full-DEM per routed shot
89.76 against 91.28 ms), far inside every speed margin (2F holds unless more than about 0.43 % of shots is routed,
G-4R unless more than about 1.45 %).

## 5. How a re-test verdict maps onto the paper (frozen)

- "Better" is repeated only by "better", from the frozen analysis on the re-test's shots. A cell's or comparison's
  original "better" is repeated or not repeated; nothing else counts as a repeat.
- Non-inferiority is not "better", and a non-significant result ("level", "inconclusive", "not better") is not
  equality.
- Table I: each comparison's verdict is reported against the original; a cell's accuracy claim is repeated iff every
  comparison is "better" again; its speed condition is an estimate from the registered sessions (section 4) and is
  reported as such, so a cell's claim is "repeated (accuracy confirmed, speed estimated)" at most.
- D1: the joint compiled verdict (single and batched modes together) stays one verdict; H1e, H2e, H1c, H2c keep their
  one Holm family.
- d = 13: the fixed development-derived non-inferiority margins stay as frozen; each of the 40 hypotheses keeps its
  label from the one Holm family; e7's verdict timings stay apart from e7r's display values.
- Head comparison: separate verdicts at q = 0 and at the registered q, per code, each from its frozen Holm family.
- The paper reports the re-test's numbers in place of the originals where it reports these tests, whatever they are,
  with each original test named as an earlier registered test and its numbers kept in the repository;
  `RETEST_RESULTS.md` lists each number and verdict with its original and whether it repeats.

## 6. Checks before this commit (results in `retest/replay/`)

Replay checks (patch off, the original inputs through the shims):
- Table I (`retest/jobs/bb_replay.sbatch`, records `retest/replay/bb/`): the original test sets regenerated on an
  Alpine CPU node from the published beacons equal the published arrays (BB72 50 of 50 chunks, gross test4 15 of 15);
  the published per-shot outputs, unpacked into per-chunk reads, through `--prepare` give CPU-term rows equal to the
  registered rows (`rows_cpu.csv`, `rows_cpu_S.csv`, gross `rows_cpu.csv`: every row, every column); stage 3's and
  Amendment S's `results.json` and reports are identical to the published ones; stage 4's report is identical and its
  `results.json` identical except two McNemar p values that differ in the 16th significant digit
  (4.122867683822274e-61 against ...273e-61, 3.853538655020656e-79 against ...658e-79: scipy's binomial test on
  another host; counts, intervals and verdicts equal).
- Table I seeds with the patch on (dry seeds; `retest/replay/bb_seedcheck.py`): every re-test chunk's seed in the
  main process and in forked workers equals the seeds file's (BB72-3 20, BB72-2 30, gross 15 chunks); `seed_of` is the
  frozen function on every other split, code and error rate (560 combinations checked); a chunk outside a set and a
  test split without a set are refused. No Table I worker draws shots: every read and the resumable full-DEM read
  (`fullread`, which resumes from its part file) load stored chunks checked against the draw's manifest.
- Cascade BB72 (section 3.2, `retest/replay/d1/`), Cascade d = 13 (section 3.3, `retest/replay/d13/`) and the head
  comparison (section 3.4, `retest/replay/head/`): each test's checks 1-3 of review finding 10, all passed.

Platform checks:
- BB72's networks on Alpine's RTX PRO 6000 (`retest/jobs/bb72_platform.sbatch`): r11 and r12 at p = 0.3 % and r12
  at 0.2 % on the original test sets, 700,000 shot-reads: 0 decisions and 0 margins differ from the published H100
  reads (bit for bit).
- The gross network on the rented H100 80GB HBM3 (`pod_reads.sh check`), in stage 4's environment rebuilt from its
  lock (the rebuilt environments' package lists equal `env.lock` and `cqvenv.lock`): 150,000 shot-reads on the
  original test4 set, 0 decisions and 0 margins differ from the published read.
- Every network file a read loads against this repository's copy (`retest/replay/weights_check.py`, 30 files): 5
  byte-identical, 25 in another container with equal tensors (the public copy equals the checkpoint's "ema" entry for C14' and CS13 and its "state" entry for the 23 others).
- BB72's networks on the rented H100 in stage 3's environment rebuilt from its record (`pod_s3venv.sh`; package list
  equal to `stage3/pod/pod_pip_freeze.txt`; the fallback of the run book): the same 700,000 shot-reads, 0 decisions
  and 0 margins differ from the published H100 reads.
- The Cascade and head reads run on the GPU models of their originals (full RTX PRO 6000; H200 2g.35gb and RTX PRO
  6000 1g.24gb slices), checked by their own frozen guards (lock identity, kernel-cache digest, batch-1 agreement).

Dry run (a past drand round, round 6,540,000, never a test seed): every draw, every read on reduced sets and every
analysis table, end to end.
- Table I (chunks of 100 shots over every frozen chunk range: 5,000 BB72 and 1,500 gross shots): the draw on Alpine
  after the host check; BB72's networks on the RTX PRO 6000; the gross network and cudaq-qec (both codes) on the rented
  H100; BB72's CPU opponents on Alpine; gross's sector Relay-BP, BP-OSD and full-DEM Relay-BP on Derecho's main queue
  (job RJ16: 128 processes, 8.3 s per full-DEM shot) and with the Alpine fallback; every transfer; then `--prepare`,
  stage 3's, Amendment S's and stage 4's analyses and the pack scripts ran to their outputs (no engine refusal: the
  RTX reads match the policies' signature, the H100 read the frozen fusedt signature).
- Cascade BB72, Cascade d = 13, head comparison: sections 3.2, 3.3 and 3.4.

## 7. Differences from the original runs

Each change below leaves the frozen computation as it was; the checks that show it are in section 6.

1. BB72's networks are read on Alpine's full RTX PRO 6000, the GPU of the policies' own engine signature; stage 3
   read them on an H100 as a disclosed substitute. The re-test's `cells.json` copy names, in the field the analysis
   compares, the GPU that the network reads' own signatures name (`rt_bb.py --prepare --gpu auto`: the RTX, or the
   frozen H100 after the fallback, which leaves the field as it is), the frozen H100 kept as "gpu_frozen"; the shim
   refuses any copy that differs from the frozen file elsewhere than in this field and the CPU-term rows. Checked: the RTX reads of the original test sets equal
   the published H100 reads, every decision and every margin bit.
2. The gross network and cudaq-qec run on a rented H100 80GB HBM3, as in stages 3 and 4, in environments rebuilt from
   the original records: the gross reads in stage 4's (`env.lock`, `cqvenv.lock`), BB72's cudaq-qec reads in stage
   3's (`stage3/pod/pod_pip_freeze.txt`, the environment BB72's original cudaq-qec reads ran in). The two hold the
   same cudaq-qec 0.8.0, cuda-quantum 0.16.0.post1, numpy 2.2.6 and Stim 1.16.0 but different CUDA libraries (12.9
   in stage 4's, 12.8 in stage 3's), so each code keeps its own. Every rebuilt environment equals its record package
   for package, checked when the pod is set up and again by `pod_reads.sh` before its reads.
3. The BB test sets are drawn on an Alpine CPU node (stage 3 and 4 drew on their pods), after a host check on the
   node: the original test sets regenerated there equal the published arrays.
4. BB72's CPU decoders run on Alpine (stage 3: an Apple M4 and a Ryzen workstation) and gross's on Derecho's regular
   main queue (stage 4: Derecho's develop queue), with the same package versions; their process layout differs.
5. No time is measured; the timing records are inherited (section 4).
6. The Cascade and head tests run from their original deployed trees on Alpine, not from this repository's copies,
   which were assembled from those trees for publication with absolute paths, host and account names and some wording
   replaced. The trees' bytes are bound by their records (D1's DEPLOYED tag, d = 13's code digest, A1's PINSUMS) and,
   for D1 and d = 13, by the re-test's pin lists (`retest/pins/`, checked before every job).
   `retest/replay/tree_compare.json` compares every file at the same relative path; it records the counts and, for
   each changed file, its sha256 on both sides and the number of changed lines, not the lines themselves
   (the original lines hold the replaced strings):
   D1's tree 436 files, of which the copy `cascade/bb72/` holds 143 (38 byte-identical, 105 changed, 64 of them JSON
   records); d = 13's 87 files, `cascade/d13/` holds 27 (17 identical, 10 changed: paths in 5 Python files and 5 job
   scripts); A1's 2,019 files, `head/` holds 61 (45 identical, 16 changed: default paths and docstrings in 12 Python
   files, a job script, a manifest's host column and two records).
7. D1's read guard keeps every check except its kind-14 freshness test, which the original D1 reads now fail by
   construction; the beacon takes its place.
8. This repository's `bb/` code differs from the trees that ran stages 3 and 4 only in sanitized defaults and labels
   (the default shot and prediction directories, one session label in `bb/gross_stage4/.../stage3.py`, comments); the engine
   sources the signatures hash are byte-identical, and every re-test command sets the directories explicitly.

## 8. Files

The re-test's files follow this repository's `SANITIZE.md`. Paths and accounts are placeholders: `<cluster-scratch>`
(the cluster directory of the account that ran the jobs), `<cluster-scratch-root>` (the file system the containers
bind, as the original jobs did), `<cluster-projects>`, `<cluster-work>`, `<account>`, `<pod-home>`. A node keeps the
label this repository already gives it (one label per node; the original A1 draws' node, for example, is cpu-node-13
here as in `head/`), a node new to the repository is `rt-cpu-node-NN` or `rt-gpu-node-NN`, and the re-test's jobs
are `RJNN` (the original jobs keep their labels:
CJ73 for D1-S, AJ32-AJ36 for d = 13's e5, e6, e7, e8 and e7r). The copies deployed on the cluster, on Derecho and on
the pod differ from these files only in the placeholders of `retest/`: every deployed file with its values replaced
by the placeholders again is byte-identical to this commit's, checked at each deployment. The scripts that copied
code and shots between the hosts are not included (as `SANITIZE.md` says of the earlier ones).

- `retest/beacon.py`, `seeds.py`, `tags.json`, `historical_seeds.txt`: the beacon and the seeds.
- `retest/shim/`: `rt_bb.py` (Table I), `rt_d1.py` (Cascade BB72), `rt_surface.py` (Cascade d = 13), `rt_a1.py`
  (head comparison), and the X2 adapter.
- `retest/jobs/`: every job and argument file, as run.
- `retest/replay/`: the replay, seed and platform checks and their results.
- `retest/pins/`: the content pins of D1's and d = 13's deployed trees (checked by `retest/jobs/tree_pin.sh`;
  its checks in `retest/replay/pins/pin_check.log`).
- `retest/run/` (after the beacon): `beacon.json`, `seeds.json`, the draw manifests; then each test's results.
