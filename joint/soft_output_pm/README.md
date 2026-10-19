# A joint decision with matching's complementary gap (exploratory)

Part of the [joint decision's records](../README.md). On the surface code the fallback, correlated PM, reports no
class score, so the joint decision of Section III B does not apply there. This exploratory read asked what a joint
decision gains when the fallback's score is the complementary gap: the weight of the best correction in the other
logical class minus that of the best correction. It drew no new shots. It reused the registered surface-code test
sets and the networks of Table III (SURF-v2's `sv2test` at d = 7, 11-17 and D9-SHALLOW's `d9stest` at d = 9) and
decoded the routed shots again on CPUs, with a second correlated decode forced into the other class. On the routed
shots where the network and correlated PM differ, the rule answers by the sign of c + w s, with s the signed gap. w is
fitted on the even chunks, and every count is from the odd chunks. Nothing here is registered. The study's own
documents call it ARB5 (see the [name map](../README.md#names)).

## Paper numbers

Section VIII ("Better scores for the joint decision"):

- With 2 % of the shots routed, a joint decision with this gap failed 5-18 % less often than CoDecoder + correlated
  PM in Table III at d = 11-17.
- The second decode runs only on the routed shots where the decisions differ, 17-27 % of them.

Both come from `out/arb5_results.json`, and `../check_paper_numbers.py` prints them.

## Files

- `RESULTS.md`: the question, the definitions, what was read and computed, the results per distance, the gap's CPU
  time, the checks, and what was not done.
- `code/arb5_alpine.py`: the per-shot scores (plain PM's answer and gap, correlated PM's answer, weight and gap) and
  their CPU times. `code/arb5_job.sbatch` and `code/arb5_time.sbatch`: the two cluster jobs (AJ47 for the scores and
  AJ48 for the times, on the paper's CPU, an AMD EPYC 7713).
- `code/arb5_analysis.py`: the fit and the tables, written to `out/arb5_results.json` and `out/arb5_tables.md`.
  `code/fill_results.py` puts those tables into `RESULTS.md`.
- `out/arb5_results.json` (every number of `RESULTS.md`), `out/arb5_tables.md`, `out/arb5_d<d>.json` and
  `out/t7713/time.json` (the jobs' records: checks, thresholds, hosts and versions), and `out/alpine_logs/` (the jobs'
  logs).

## Rerunning it

`out/arb5_tables.md` holds `RESULTS.md`'s tables, and every line of it appears in `RESULTS.md`. The scores need the
per-shot outputs of the registered surface-code tests, which the authors archive (`docs/CHECKS.md`). The shots
themselves rebuild from the manifests' seeds. With those reads in place, on a CPU node:

1. `arb5_alpine.py sv2 --d D --core K --out OUT` for d = 7, 11, 13, 15 and 17, and `arb5_alpine.py d9s ...` for d = 9
   (`code/arb5_job.sbatch`, one pinned core per distance; the job took 7 minutes). `code/arb5_time.sbatch` times
   the gaps.
2. `python arb5_analysis.py --out OUT --release RELEASE` for the fit and the tables, then
   `python fill_results.py RESULTS.md OUT/arb5_tables.md`.

## Not included

The per-shot arrays of the two jobs (`arb5_d<d>.npz` and `t7713/time_d<d>.npz`, 6-12 MB each), which no branch of
the research repository holds. The authors keep them. Their sha256 were not recorded, and step 1 rebuilds them from the
registered reads.
