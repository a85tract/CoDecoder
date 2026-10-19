# Fitting w at d = 13

Part of the [joint decision's records](../README.md). This study drew 4,000,000 fresh surface-code shots at d = 13 and
p = 0.6 %, and 30,000,000 at 0.35 %, from new seeds. It decoded them with the network and Cascade of Table I's d = 13
rows, fitted w on the even chunks (the fit half) and looked once at the odd chunks (the held-out half). The joint
decision at d = 13 is the rule these documents call J, answer 1 iff c + w z > 0 (c the network's logit, z Cascade's),
and its w was frozen for Table I's shots and for the timing. The study compared two other rules, P and T, which the
paper does not use. Its own summary carried P forward, but the paper's joint decision is J. The study's documents call
it "ARB at d = 13" or ARB2 (see the [name map](../README.md#names)).

## Paper numbers

- Section III B and Appendix E: w = 7.87 at p = 0.6 % and 13.84 at 0.35 %, fitted by the fewest failures over every
  breakpoint of the fit half's objective on 2,000,000 and 15,000,000 fresh shots.
- Appendix E: on the held-out half, the joint decision recovered 2-4 % of what a perfect choice between the two answers
  would recover. On the held-out routed shots where the decisions differ, the network was wrong three times as often
  as Cascade.
- Section VIII ("Training the network for its fallback"): a perfect choice between the two answers on the routed shots
  would remove 30 % of the composition's failures (30.4 % and 30.0 % on the whole fresh reads).

All of these come from `out/analysis/summary.json`, and `../check_paper_numbers.py` prints them.

## Files

- `PLAN.md`: the plan as reviewed before the run. `RESULTS.md`: the results, how the run went, the seeds and their
  checks, and the definitions.
- `code/`: the read (`arb2_read.py`, run through the seed shim `arb2_shim.py`), the analysis (`arb2_analysis.py`),
  the seeds (`arb2_seeds.py`, `hist_scan.sh`), the checks, `arb2_tables.py`, and the pod scripts (`pod_env.sh`,
  `pod_setup.sh`, `pod_dry.sh`, `pod_read.sh`, `pod_run.sh`).
- `seeds/`: `arb2_seeds.json` (the two sets of chunk seeds) and `hist_extra.txt` (the historical seeds that the
  collision rule adds to `retest/historical_seeds.txt`).
- `out/`: `analysis/summary.json` (every fitted value and count, with J's whole curve over w on both halves) and
  `analysis/tables.md`. `read/` holds each read's settings, timings, part journal, seed check, dry-run comparison and
  the shim's trace, but no per-shot data. Also `dry/` (the dry run, never reported), `logs/`, `env.json`,
  `status.txt`, and the manifests `manifest.sha256`, `pod_out.sha256` and `pod_out.sizes`.

## Rerunning it

From the repository root, on any CPU, in seconds:

```bash
python3 joint/fit_d13/code/arb2_seeds.py check . joint/fit_d13/seeds/arb2_seeds.json    # the seeds from RETEST's beacon
T=$(mktemp -d) && python3 joint/fit_d13/code/arb2_tables.py joint/fit_d13/out/analysis/summary.json > $T/tables.md \
  && cmp $T/tables.md joint/fit_d13/out/analysis/tables.md                               # the tables, byte for byte
```

The fit itself ran on one RTX PRO 6000 Blackwell Server Edition for about 2 h 15 min. Lay out its work directory `W`
(default `/workspace/arb2`, see `code/pod_env.sh`) as the run did:

- `$W/release`: this repository at 94aaaa7, the commit the fit ran on (`git archive 94aaaa7 | tar -x -C $W/release`).
  `pod_setup.sh` links Cascade's checkpoint `weights/cascade_d13/CS13/final_ema.pt` from it.
- `$W/code` and `$W/seeds`: copies of this directory's `code/` and `seeds/`.

Then run `bash $W/code/pod_setup.sh`, `bash $W/code/pod_dry.sh` and `bash $W/code/pod_run.sh graph` (Cascade through
its CUDA graph, as in the run). The seeds come from the TAGs `convlines arb surface d13 p0.006\n` (400 chunks of
10,000 shots) and `convlines arb surface d13 p0.0035\n` (3,000 chunks), with R = the re-test's beacon
(`retest/run/beacon.json`, drand round 6,547,616). Chunk j takes seed j, and the first seeds are 7125939203913419922
and 5068027980986011364. The run writes `out/` in the layout above. `out/pod_out.sha256` gives the sha256 of every
file it wrote.

## Not included

The run's 42 per-shot files, 9.74 GB in all. In `out/read/` they are the parts of the reads, one per 1,000,000 shots
(`p0.0060/part_*.npz`, `p0.0035/part_*.npz`), with every shot's detector rows, both decoders' logits and decisions.
`out/analysis/disagree_<p>.npz` holds one row per shot where the decisions differ, and `out/dry/` holds the same
files for the dry run. `out/pod_out.sha256` lists the sha256 of all 42, and each read's journal
(`out/read/p<p>/progress.jsonl`) lists those of its parts. The fit rebuilds them from the seeds. The research
repository did not keep `disagree_<p>.npz` either.
