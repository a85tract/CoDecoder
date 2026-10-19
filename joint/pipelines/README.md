# The joint decision in the timed pipelines: checks

Part of the [joint decision's records](../README.md). Commits 824afab and 0e9321e add the joint decision to the timed
pipelines and to the reads of the Cascade comparisons. On BB72 the timed pipelines rerun the network's vote on the
routed shots where the decisions differ, to keep all 4,096 class scores, and decide in fp32 with ties to the lowest
class. This directory holds the checks that these pipelines decide exactly as the fits' float64 analysis and the
reads do, and one GPU pod's smoke times, which back no number. Its documents call this build stage ARB3 (see the
[name map](../README.md#names)).

## Paper numbers

No number of the paper comes from this directory. It backs two statements:

- Appendix E: the timed BB72 pipelines rerun the network's vote on these shots to keep all 4,096
  class scores, in fp32 with ties to the lowest class. On every row of the BB72 fit where the decisions differ (9,037
  at p = 0.2 % and 94,184 at 0.3 %), the fp32 rule gives the fit's float64 answer, with no ties
  (`checks/out/check_arb_<p>.json`).
- The timed pipelines with the joint step decide as the reads on the same path: on 500,000 dry shots per p (20,000 in
  single mode), 0 decisions differ on BB72 (`checks/out/pod/out/bb/`, at the fitted w) and at d = 13
  (`checks/out/pod/out/d13/`, at w = 1.0, a placeholder, since the d = 13 fit was not done yet). The joint decision's
  timing in Table I (`../timing/`) runs these pipelines.

## Files

- `TEST.md`: the rule and its fixed arithmetic, what changed in the code, the checks, the smoke times (one H100 pod,
  not registered), the commands that a registered confirmation would run, and what was not done.
- `checks/check_arb.py`: check (a), the fp32 rule against the fit's float64 answers, on a CPU and then on the GPU
  kernel. `checks/bench_joint.py`: the joint step's micro-benchmark.
- `checks/out/`: the check records. `check_arb_<p>.json` is check (a) on the CPU. `pod/out/` is the pod's output:
  `arb/` (check (a) on the GPU), `bb/` and `d13/` (check (b): each pipeline's comparison with the reads, and the rows
  of the smoke times), `bench/`, `logs/` and the environment record.
- `pod/`: the pod scripts. `pod_env.sh` and `pod_setup.sh` set up the environment, `run_bb.sh P W` and
  `run_d13.sh P W` run check (b) at one p, `stage2.sh` chains the d = 13 checks and the benchmark, and `rep_d13.sh`
  repeats a smoke time. `podrun.py` and `podrun13.py` start the harnesses.

## Rerunning it

Check (a) needs the BB72 fit's per-shot files (`analysis/disagree_<p>.npz` and `read/p<p>_scores.npy`; see
`../fit_bb72/README.md`), which this repository does not hold. With them in `ARB_OUT`:

```bash
python checks/check_arb.py cpu --arb ARB_OUT --p 0.002 --w 6.309573444801933 --out a_0.002.json --subset s_0.002.npz
python checks/check_arb.py cpu --arb ARB_OUT --p 0.003 --w 3.758374042884441 --out a_0.003.json --subset s_0.003.npz
```

Check (b) draws its own dry shots. It needs one GPU pod (`TEST.md` used an H100 80GB HBM3, which `pod/podrun.py`
lets through the BB72 harness's GPU guard), this repository at the four commits of the joint decision as
`$W/release`, `pod/` as `$W/pod`, and Relay-BP's repository at revision d185194 as `$W/relay_min` (see
`docs/REPRODUCE.md`). Then run `bash $W/pod/pod_setup.sh` and, per p with its w, `bash $W/pod/run_bb.sh P W` and
`bash $W/pod/run_d13.sh P W` (`TEST.md` section 3). The pipelines' saved decisions (`pipe.npz`) were not copied back;
each comparison's record keeps its counts of differing routes, decisions and failures.
