# The frozen joint decision on Table I's compiled paths

Part of the [joint decision's records](../README.md). This study counted the joint decision, with the frozen w, on the
two compiled paths whose failures Table I prints: single mode (Cascade compiled at batch 1) and batched mode (compiled
at batch 4,096 on BB72, at batch 16 at d = 13). It used Table I's shots, the network's decisions, margins and routing
from the re-test's reads, and changed only Cascade's path. On BB72 the re-test's compiled files keep Cascade's
decisions but not its 12 logits, so one GPU job reran Cascade's compiled graphs on the routed shots with the logits as
one more output (`code/cp_run.py`). At d = 13 the re-test's read already holds Cascade's logit on both paths
(`code/cp_d13x.py`, no GPU). The study's own documents call it ARBCP (see the [name map](../README.md#names)).

## Paper numbers

`out/bb72_counts.json` and `out/d13_counts.json` hold every count, and `../check_paper_numbers.py` prints these from
them:

- Table I, the rows "CoDecoder + Cascade, joint decision": the eight failure counts (6,597 and 6,588 at BB72
  p = 0.2 %, 65,189 and 65,185 at 0.3 %, 4,886 and 4,884 at d = 13 p = 0.6 %, 2,306 and 2,308 at 0.35 %, single and
  batched) and the routed shares, on the same shots as Cascade alone and the composition.
- Appendix E: one Holm procedure over the eight exact McNemar tests against Cascade alone rejects all eight. The
  largest p, 0.033, is d = 13 at 0.6 % in single mode, with 289 shots failing only with Cascade alone and 239 only
  with the joint decision. The routed shots where the decisions differ are 0.08 % and 0.56 % of the BB72 shots and
  0.62 % and 0.01 % of the d = 13 shots.
- Section IV D, Section VI, the abstract, the related work and the conclusion: 18 % and 14 % fewer failures than
  Cascade alone on BB72 (14-18 %), and 1.0-2.0 % at d = 13.
- Section VI: on BB72 the joint decision repairs 14 to 49 shots for each one it breaks, and about a third of its
  repairs return a class that neither decoder chose. At d = 13 and p = 0.6 % in single mode it repairs 249 shots and
  breaks 218.

## Files

- `RESULTS.md`: the table, whether the recomputed Cascade outputs equal the re-test's, what was run, and how it was
  counted.
- `code/`: `cp_bb72.sbatch` and `cp_run.py` (the BB72 GPU job), `cp_d13x.py` (the d = 13 extraction), `cp_count.py`
  and `cp_count_d13.py` (the counts), `cp_tables.py` (the tables), and `arb_fused.py` (the network's engine with its
  class scores, a copy of `../fit_bb72/code/arb_fused.py`).
- `out/bb72_counts.json`, `out/d13_counts.json`: the counts. `out/bb72/p<p>.json` and `out/d13/d13_<p>.json` are the
  jobs' records (settings, environment, checks and the kept shots' counts). `out/bb72/`, `out/d13/` and
  `out/alpine_logs/` also hold the cluster jobs' logs. AJ49 is a first submission that stopped at once, because the
  Relay-BP circuit path was unset.

## Rerunning it

From the repository root, on any CPU, in seconds:

```bash
python3 joint/table1_compiled/code/cp_tables.py --bb72 joint/table1_compiled/out/bb72_counts.json \
    --d13 joint/table1_compiled/out/d13_counts.json        # RESULTS.md's tables, line for line
```

The counts themselves need the re-test's per-shot reads, which the authors archive and RETEST rebuilds from its seeds
(`RETEST.md` sections 3.2 and 3.3). The BB72 job (`cp_bb72.sbatch`) needs one RTX PRO 6000 Blackwell Server Edition
for about 12 minutes, the follow-up D1's environment and its kernel caches. `RESULTS.md` ("What was run")
gives the details. The d = 13 extraction and both counts run on a CPU:

```bash
python cp_d13x.py --read RETEST/out/d13/conf --lock cascade/d13/out/locked_d13.json --out OUT/d13
python cp_count.py --dir OUT/bb72 --p 0.002,0.003 --json bb72_counts.json
python cp_count_d13.py --dir OUT/d13 --w 0.006:7.86574441929562,0.0035:13.842311471679352 --json d13_counts.json
```

## Not included

The per-routed-shot arrays that the jobs wrote: Cascade's 12 float32 logits, decision and confidence on both compiled
paths (`p<p>_logits.npz`), the classes of the observables, the network, Cascade and the joint decision on each path
(`p<p>_routed.npz`, `d13_<p>.npz`), and the network's class scores (1.5 GB at p = 0.3 %). Their sha256 were not
recorded. `cp_run.py` and `cp_d13x.py` rebuild them from the re-test's reads.
