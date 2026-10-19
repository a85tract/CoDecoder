# Timing the joint decision

Part of the [joint decision's records](../README.md). One session on 19 October 2026 timed the joint decision by the
rules of the speed re-test (SPEED-RT stage 1, which gives Table I's times) on fresh shots and on both of Table I's
cards: an RTX PRO 6000 Blackwell Server Edition on a cluster node of the type in Table I, and an H100 80GB HBM3 on a
rented pod. Per setting, each card ran Cascade alone, the composition and the composition with the joint decision in
one job, each in three fresh processes, and each composition process was followed by a joint one on the same shots.
The session was not registered. Its own documents call it ARB3 (see the [name map](../README.md#names)), and its shot
sets are stage 1's, renamed so that their seeds are new (`speed_rt/cascade/sets_arb3.json` and `seeds_arb3.json`,
added by commit 5462f4e).

## Paper numbers

`../check_paper_numbers.py` prints all of these from `timing.json`, `stage1.json` and the slot records:

- Table I, the rows "CoDecoder + Cascade, joint decision": the 16 times (2 cards, 4 settings, single and batched
  mode) and their speed-ups (`timing.json`, `timing_cells.md`).
- Section IV D: the joint decision took at most 2.4 % longer than the composition in 15 of the 16 timings, and the
  16th took 0.2 % longer at equal clock.
- Appendix E ("Timing the joint decision."): on the H100, Cascade alone took 27-28 % longer per single BB72 shot than
  in the speed re-test (`timing.json` against `stage1.json`), so the joint decision's single-mode speed-ups on BB72
  there exceed the composition's in Table I. The joint decision took as long as the composition within 2.4 % in 15 of
  the 16 timings. In the 16th, single mode at p = 0.3 % on BB72 on the RTX PRO 6000, its processes ran at a 5 % lower
  GPU clock (2,295 against 2,422 MHz, the clock records in the slots' `rows.jsonl`) and took 6.7 % longer. A separate
  run at equal clock (the order test, `records/RTX/out/order_rtx_AJ56/`) found 0.1-0.3 % at p = 0.2 % and 0.2 % at
  0.3 %.

## Files

- `TIMING.md`: the results, how the session was timed, where it departs from Table I's timing, the commands as run,
  and what was not done. Its paths `timing/X` are this directory's `X`.
- `records/RTX/out/` and `records/H100/out/`: every timed slot (`attempts.json`, and per attempt `rows.jsonl`,
  `trace.jsonl` with the chunk sha256, `windows.jsonl` with the clock samples and `log.txt`), the d = 13 sessions and
  locks, the order test (`order_rtx_AJ56/`) and the RTX benchmark's outputs and logs (`bench_rtx/`).
  `records/RTX/logs/`: the cluster jobs' logs.
- `timing.json`, `timing_cells.md`: per-process values and the summary cells. `stage1.json`, `stage1_cells.md`: the same
  summary of stage 1's records (`speed_rt/cascade/records/`), which reproduces Table I's other rows. `tables.md`: the
  tables of `TIMING.md`.
- `bench/RTX/`, `bench/H100/`: the joint step's micro-benchmark (`../pipelines/checks/bench_joint.py`). The RTX's comes
  from cluster job AJ55, the H100's from the pod of `../pipelines/`.
- `summarize.py`, `make_tables.py`: the summaries. `bb_job.sh`, `d13_job.sh`, `bench_job.sh` and `arb3_rtx_*.sbatch`:
  the jobs. `dry/`: the dry beacon and seeds of the untimed passes.

## Rerunning it

From the repository root, on any CPU, in seconds, the summaries come back byte for byte:

```bash
cd joint/timing && T=$(mktemp -d)
python3 summarize.py --rec RTX=records/RTX/out/bb_rtx_AJ52 RTX=records/RTX/out/d13_rtx/drive_20261019_035343 \
    H100=records/H100/out/bb_h100 H100=records/H100/out/d13_h100/drive_20261019_100326 \
    --out $T/timing.json --md $T/timing_cells.md \
  && cmp $T/timing.json timing.json && cmp $T/timing_cells.md timing_cells.md
S=../../speed_rt/cascade/records
python3 summarize.py --rec RTX=$S/RTX/R1 RTX=$S/RTX/R4 H100=$S/H100/R1 H100=$S/H100/R4 \
    --out $T/stage1.json --md $T/stage1_cells.md \
  && cmp $T/stage1.json stage1.json && cmp $T/stage1_cells.md stage1_cells.md
python3 make_tables.py timing.json stage1.json bench > $T/tables.md && cmp $T/tables.md tables.md
cd ../..
```

To time it again, use this repository from commit f03266d on, which holds the four commits of the joint decision.
`TIMING.md` section 4 gives the commands as run. On the cluster these are the `arb3_rtx_*.sbatch` jobs. On a pod they
are `bb_job.sh`, then `d13_job.sh` with `PHASE=prep` (the kernel caches, untimed) and `PHASE=timed`. The timed runs
take `SRT_SETS_FILE=sets_arb3.json`, `SRT_SEEDS=seeds_arb3.json` and stage 1's beacon, and the frozen w:
`WB="0.002=6.309573444801933,0.003=3.758374042884441"` and `W13="0.006=7.86574441929562,0.0035=13.842311471679352"`.
For fresh shots, give the sets new names in place of `arb3-cas-*`.

## Not included

- The decisions that the joint processes saved (`pipe.npz`). They were not copied back: the RTX's stayed on the
  cluster's scratch and the H100's went with the pod. No bitwise check of this session's joint decisions against reads
  was run (`TIMING.md` section 6). `../pipelines/` checks the same pipelines on its own shots.
- The H100 benchmark of this session, which was lost with the pod. `bench/H100/` is the build stage's.
- `timing/code/` of the research branch, which equals commits 5462f4e and f03266d.

Four records of the d = 13 RTX session, `records/RTX/out/d13_rtx/session_0.json`, `session_1.json`,
`session_settled.json` and `lock_timed_20261019_035343.json`, are published byte for byte, because other records
hold their sha256. They still name the cluster's paths and node.
