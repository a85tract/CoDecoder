# Fitting w on BB72

Part of the [joint decision's records](../README.md). This study drew 11,000,000 fresh BB72 shots at p = 0.2 % and
17,000,000 at 0.3 %, from new seeds, and decoded them with the network and Cascade of Table I's BB72 rows. It fitted
the joint decision's weight w on the even chunks (the fit half) and looked once at the odd chunks (the held-out half).
The fitted w was then frozen for Table I's shots and for the timing. The study's own documents call it ARB, the network
S55'-deep and Cascade C14' (see the [name map](../README.md#names)). The joint decision is the rule they call A3.

## Paper numbers

- Section III B and Appendix E: w = 6.31 at p = 0.2 % and 3.76 at 0.3 % (w = 10^(k/40) with k = 32 and 23, the
  logarithmic grid), fitted by the fewest failures on 5,500,000 and 8,500,000 fresh shots.
- Appendix E: on the held-out half, the joint decision recovered 59-61 % of what a perfect choice between the two
  answers would recover.

Both come from `out/analysis/summary.json`, and `../check_paper_numbers.py` prints them.

## Files

- `PLAN.md`: the plan as reviewed before the run. `RESULTS.md`: the results, how the run went, the seeds and their
  checks, and the definitions.
- `code/`: the read (`arb_read.py`, run through the seed shim `arb_shim.py`), the class-score export (`arb_scores.py`
  with the engine `arb_fused.py`), the analysis (`arb_analysis.py`), the seeds (`arb_seeds.py`, `hist_scan.sh`), the
  checks, `results_tables.py`, and the pod scripts (`pod_env.sh`, `pod_setup.sh`, `pod_dry.sh`, `pod_read.sh`,
  `pod_run.sh`).
- `seeds/`: `arb_seeds.json` (the two sets of chunk seeds) and `hist_extra.txt` (the historical seeds that the
  collision rule adds to `retest/historical_seeds.txt`).
- `out/`: `analysis/summary.json` (every fitted value and count) and `analysis/tables.md`. `read/` holds each read's
  signature, totals, seed check, dry-run comparison and the shim's trace, but no per-shot data. Also `dry/` (the dry
  run, never reported), `logs/`, `env.json`, `status.txt`, and the manifests `manifest.sha256`, `pod_out.sha256` and
  `pod_out.sizes`.

## Rerunning it

From the repository root, on any CPU, in seconds:

```bash
python3 joint/fit_bb72/code/arb_seeds.py check . joint/fit_bb72/seeds/arb_seeds.json    # the seeds from RETEST's beacon
python3 joint/fit_bb72/code/results_tables.py joint/fit_bb72/out/analysis/summary.json  # RESULTS.md's tables
```

The second command prints the tables of `RESULTS.md` from `summary.json`. `RESULTS.md` wraps their long lines.

The fit itself needs one RTX PRO 6000 Blackwell Server Edition (the BB72 harness's GPU guard admits only that card) for
about 1 h 45 min. Lay out its work directory `W` (default `/workspace/arb`, see `code/pod_env.sh`) as the run did:

- `$W/release`: this repository at 94aaaa7, the commit the fit ran on (`git archive 94aaaa7 | tar -x -C $W/release`).
- `$W/code` and `$W/seeds`: copies of this directory's `code/` and `seeds/`.
- `$W/relay_min`: Relay-BP's repository at revision d185194, for its BB72 circuit (see `docs/REPRODUCE.md`).

Then run `bash $W/code/pod_setup.sh`, `bash $W/code/pod_dry.sh` and `bash $W/code/pod_run.sh`. The seeds come from the
TAGs `convlines arb bb72 depspam_fm p0.002\n` (1,100 chunks of 10,000 shots) and `convlines arb bb72 depspam_fm
p0.003\n` (1,700 chunks), with R = the re-test's beacon (`retest/run/beacon.json`, drand round 6,547,616). The chunks
are numbered from 6,433,102, and the first seeds are 7800278565957140538 and 7639215094236668584. The run writes
`out/` in the layout above. `out/pod_out.sha256` gives the sha256 of every file it wrote.

## Not included

The run's 20 per-shot files, 5.96 GB in all. In `out/read/` they are the two reads (`p0.00200.npz`, `p0.00300.npz`),
Cascade's float32 logits (`p<p>_z32.npy`) and the network's class scores on the shots where the two decisions differ
(`p<p>_scores.npy`, `p<p>_scores_rows.npy`). `out/dry/` holds the same files for the dry run, and
`out/analysis/disagree_<p>.npz` holds one row per shot where the decisions differ. `out/pod_out.sha256` lists the
sha256 of all 20, and the fit rebuilds them from the seeds. The research repository did not keep
`disagree_<p>.npz` either.
