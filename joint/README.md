# The joint decision of CoDecoder and Cascade

Part of the [CoDecoder release](../README.md).

A routed shot discards the network's prediction, which is already computed. When the fallback also reports logits, as
our Cascades do, both decoders can decide together the routed shots where their decisions differ (paper Section III B,
"Joint decision on routed shots"). On BB72 the joint decision answers the class y that maximizes S(y) + w (y . l) over
the 4,096 classes, where S(y) is CoDecoder's class score and l holds Cascade's 12 logits, so the answer can be a class
that neither decoder chose. At d = 13 it answers the sign of c + w l, where c is the network's logit and l Cascade's,
and Cascade's answer on a tie. Every other shot keeps the composition's answer.

w was fitted on separate fresh shots and frozen before it was applied to the test shots of Table I (the Cascade table).
The result is therefore held out, not registered (Table XII, row "Joint decision on the routed shots").

| setting | rule | w, frozen | fitted on |
|---|---|---|---|
| BB72, p = 0.2 % | A3 | 6.309573444801933 = 10^(32/40) | 5,500,000 fresh shots, the even chunks of 11,000,000 |
| BB72, p = 0.3 % | A3 | 3.758374042884441 = 10^(23/40) | 8,500,000 fresh shots, the even chunks of 17,000,000 |
| d = 13, p = 0.6 % | J | 7.86574441929562 | 2,000,000 fresh shots, the even chunks of 4,000,000 |
| d = 13, p = 0.35 % | J | 13.842311471679352 | 15,000,000 fresh shots, the even chunks of 30,000,000 |

## The code

The timed pipelines and the reads of the Cascade comparisons carry the joint decision as options, and without these
options nothing changes. Commits `824afab` and `0e9321e` add the decision itself, and `5462f4e` and `f03266d` add what
its timing session needed.

- BB72: `cascade/bb72/cascade_compare/batchopt/m2.py pipeline --joint-w W --save F` and `phase_d/joint.py`. On the rows
  whose decisions differ, the kernels rerun the network's vote to keep all 4,096 class scores in fp32. They form
  w (y . z) as one rounded multiply and break ties toward the lowest class. `phase_d/joint_ref.py` is a numpy reference
  that gives the same answers bit for bit. `joint_read.py`, `joint_scores.py`, `joint_count.py` and `scored.py` are the
  reads, the score export, the counts and the bitwise checks, and `d1_compiled.py read --joint` keeps Cascade's logits
  on the compiled paths.
- d = 13: `cascade/d13/surface/sv_joint.py`, `sv_speed.py pipeline --joint-w W` and
  `sv_speed.py bitwise --joint-w P=W,P=W`.
- The timing harness: `speed_rt/cascade/drive.py` gains `--joint-w`, `--joint-w13`, `--joint-both` and `--engines`, and
  `srt_d13.py` gains `session` and `joint-lock`. `srtlib.py` reads `SRT_SETS_FILE`, and `relmap.py` applies with
  `SRT_RELMAP=1`. `sets_arb3.json` and `seeds_arb3.json` hold the timing session's own sets and seeds.

## The studies

| directory | run name | what it holds | paper numbers |
|---|---|---|---|
| [`fit_bb72/`](fit_bb72/) | ARB | the fit of w on BB72 (11,000,000 and 17,000,000 fresh shots), held-out halves | Sec. III B and App. E: w = 6.31 and 3.76, 5,500,000 and 8,500,000 shots, 59-61 % |
| [`fit_d13/`](fit_d13/) | ARB2 | the fit of w at d = 13 (4,000,000 and 30,000,000 fresh shots), held-out halves | Sec. III B and App. E: w = 7.87 and 13.84, 2,000,000 and 15,000,000 shots, 2-4 %, three times; Sec. VIII: 30 % |
| [`table1/`](table1/) | ARB4 | the frozen joint decision on Table I's test shots: BB72 on Cascade's eager path, d = 13 on both paths | Table I rows a at d = 13 (counted again in `table1_compiled/`); the eager BB72 counts that `table1_compiled/` reproduces |
| [`table1_compiled/`](table1_compiled/) | ARBCP | the same on Cascade's compiled paths, whose failures Table I prints | Table I rows a: the eight failure counts, the routed shares and "better"; Sec. IV D: 18 %, 14 %, 1.0-2.0 %; Sec. VI: 14-18 %, 14 to 49, a third, 249 and 218; App. E: 0.08, 0.56, 0.62, 0.01 %, a third, the Holm procedure, 0.033, 289 and 239; abstract, Sec. IX, Sec. X: 14-18 % |
| [`pipelines/`](pipelines/) | ARB3 | the joint decision built into the timed pipelines, and the bitwise checks against the reads | App. E: the timed BB72 pipelines rerun the vote, fp32, ties to the lowest class |
| [`timing/`](timing/) | ARB3 | the timing session: every slot's records, the summary and its tables | Table I rows a: 16 times and 16 speed-ups; Sec. IV D and App. E: 2.4 %, 15 of 16, 6.7 %, 5 %, 0.1-0.3 % and 0.2 %, 27-28 % |
| [`soft_output_pm/`](soft_output_pm/) | ARB5 | an exploratory joint decision for the surface code, scored by matching's complementary gap | Sec. VIII: 5-18 %, 17-27 % |

`check_paper_numbers.py` recomputes each of these numbers from the records in this directory and compares it with the
paper's text.

## Checking

From the repository root, with Python 3.12 and its standard library:

```bash
python3 joint/check_paper_numbers.py    # every joint-decision number of the paper from the records; exit 0 if all match
```

The script prints 84 lines, each with the paper's value, the value from the records and its source, and three notes.
In the related work, "removed a further 14-18 %" holds against Cascade alone's failures. Against the composition's
failures the range is 13.9-17.5 %.
At p = 0.3 % the equal-clock run's two joint processes took 0.146 % and 0.170 % longer, and the paper prints their
mean, 0.2 %.
The 30 % of the discussion is 30.4 % and 30.0 % on the whole fresh reads at d = 13, and 30.2 % and 29.2 % on their
evaluation halves.
Each study's README gives the commands that rerun its outputs. These run on any CPU in seconds and rebuild published
files byte for byte:

```bash
T=$(mktemp -d)
python3 joint/fit_bb72/code/arb_seeds.py check . joint/fit_bb72/seeds/arb_seeds.json    # the fits' seeds from RETEST's
python3 joint/fit_d13/code/arb2_seeds.py check . joint/fit_d13/seeds/arb2_seeds.json    # beacon
SRT_SETS_FILE=$PWD/speed_rt/cascade/sets_arb3.json python3 speed_rt/cascade/srtlib.py seeds \
    --beacon speed_rt/cascade/beacon.json --out $T/seeds_arb3.json \
  && cmp $T/seeds_arb3.json speed_rt/cascade/seeds_arb3.json                             # the timing session's seeds
(cd joint/timing && python3 summarize.py --rec RTX=records/RTX/out/bb_rtx_AJ52 \
    RTX=records/RTX/out/d13_rtx/drive_20261019_035343 H100=records/H100/out/bb_h100 \
    H100=records/H100/out/d13_h100/drive_20261019_100326 --out $T/timing.json --md $T/timing_cells.md > /dev/null \
  && cmp $T/timing.json timing.json && cmp $T/timing_cells.md timing_cells.md)            # the timing summary
```

The verification of SPEED-RT ([How to verify](../docs/SPEED.md#how-to-verify)) still reproduces stage 1's seeds and
analysis byte for byte with the code of these commits.

## What is not included

- The per-shot reads of the two fits: 5.96 GB on BB72 (C14''s and S55'-deep's outputs for every shot, Cascade's float32
  logits, the network's class scores on the rows that differ) and 9.74 GB at d = 13 (every shot's detector rows and
  both logits). Their sha256 are in `fit_bb72/out/pod_out.sha256` and `fit_d13/out/pod_out.sha256`, which list every
  file of each run. The per-disagreement-row files `analysis/disagree_<p>.npz` that the runs wrote are listed there too
  but were not kept in the research repository. The seeds, the TAGs and the commands that rebuild them are in each
  study's README.
- The re-test's per-shot reads, on which `table1/` and `table1_compiled/` count. The authors archive them, and RETEST
  rebuilds them from its seeds (`RETEST.md` sections 3.2 and 3.3).
- The network's class scores on Table I's shots that `table1/` exported (their sha256 are in
  `table1/out/pod/pod_read.sha256`), and the per-routed-shot arrays and class scores of `table1_compiled/` (1.5 GB at
  p = 0.3 %) and `soft_output_pm/`, whose sha256 were not recorded. Each study's scripts rebuild them from the re-test's
  reads, or from SURF-v2's registered reads for `soft_output_pm/`.
- The pipelines' saved decisions (`pipe.npz`) of the timing session and of the checks, which were not copied back.
- The coordination files of the runs: their task texts, launchers, polling and copy-back scripts, and an ssh helper.
  ARB3's `ref_arb/` and `code/` are also left out, because they equal `fit_bb72/code/` and the four commits above.

## What changed for publication

The studies are copies of the research repository's branches `arb-exp`, `arb-d13`, `arb-table1`,
`arb-table1-compiled`, `arb-prod` and `arb-pm`. Their text was sanitized as `SANITIZE.md` describes, and no number,
rule or result changed.

- The cluster's scratch paths (`/scratch/alpine/<account>`, `/gpfs/...`) became `<cluster-scratch>`, the cluster account
  `<user>`, and the pods' home directory `<pod-home>`. The internal git host became `<hub-host>` and the research
  repository `<research-repo>`.
- The RTX PRO 6000 node became `gpu-node-07`, the label this repository already uses for it. The memory node became
  `mem-node-01`, as in `results/r17/`. The two CPU nodes of `soft_output_pm/` became `cpu-node-24` and `cpu-node-25`,
  and the pods' container host names became `gpu-pod`.
- The ten Slurm jobs became `AJ47` to `AJ56`, in the order of their ids, in text and in file and directory names:
  `AJ47` and `AJ48` (ARB5), `AJ49` to `AJ51` (ARBCP), and `AJ52` (the BB72 timing), `AJ53` (the d = 13 kernel caches),
  `AJ54` (the d = 13 timing), `AJ55` (the benchmark) and `AJ56` (the order test). The timing tables call the order
  test "job AJ56".
- The workstations became "the Linux workstation" and "the Windows workstation", and home work directories are
  written `~/work` (`~/work/arb` and so on). Coordination wording and the name of the tool that reviewed a plan were
  reworded, and the review is cited by its finding numbers.
- 130 files have their sha256 in another published record, and these stay byte for byte as written.
  They include every file of the fits' `out/` (in their `pod_out.sha256`), the seed history files and the d = 13
  locks and sessions. 21 of them keep identifiers, the same kind of exception as stage 1's `H100/R4` files
  (`speed_rt/cascade/results/README.md`). Seventeen keep a rented pod's container host name or its home directory
  `/root`: the fits' `env.json`, setup logs and read records, and the d = 13 sessions and locks of the pods
  (`pipelines/checks/out/pod/out/d13/`, `timing/records/H100/out/d13_h100/`). Four keep the cluster account in their
  paths, and three of them the RTX node's host name: `timing/records/RTX/out/d13_rtx/session_0.json`,
  `session_1.json`, `session_settled.json` and `lock_timed_20261019_035343.json`. The derived locks record the
  sessions' sha256, and the timed slots record the timed lock's sha256.
- One line of code differs from the code that ran: the docstring of `cascade/bb72/cascade_compare/phase_d/scored.py`
  named the workstation. Its sha256 therefore differs from the digest `04e1370b2ac76065` that the checks' score
  exports record (`pipelines/checks/out/pod/out/bb/p0.00*/read/p0.00*_jscores.json`). The commits also differ from the
  ones that ran in their messages.
- In `fit_bb72/out/manifest.sha256`, the run's own list, `logs/run.nohup` fails, because the log grew after the list
  was written. It failed the same way in the research repository. The copy-back list `pod_out.sha256` matches it.

## Names

The studies' documents use the names of the research project. Table XIV of the paper maps the decoders and tests
(S55'-deep is the six-block CoDecoder network of the Cascade comparison, C14' our Cascade on BB72, d13_b5_r775 the
d = 13 network, CS13 our Cascade at d = 13, C5 the d = 13 composition and C2 Cascade alone there, D1 the follow-up on
BB72). The joint decision's own names are these:

| name in these records | meaning |
|---|---|
| ARB, ARB2 ("ARB at d = 13"), ARB3, ARB4, ARBCP, ARB5 | the studies, as in the table above |
| arbiter, arbitration, rule | a way to decide the routed shots where the two decisions differ; the joint decision is A3 on BB72 and J at d = 13 |
| A1m, A1s, A2 (BB72); P, T (d = 13) | the other rules the fits compared, which the paper does not use |
| routed oracle | a perfect choice between the two answers on the routed shots |
| third answer | an answer of the joint decision that neither decoder chose (BB72) |
| `arb3-cas-*` | the timing session's shot sets, which stage 1's `srt-cas-*` sets renamed so that their seeds are new |
