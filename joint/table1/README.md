# The frozen joint decision on Table I's shots, from the reads

Part of the [joint decision's records](../README.md). This study counted the joint decision, with the w that the fits
froze (`../fit_bb72/`, `../fit_d13/`), on the test shots of Table I: the re-test's reads of 11,000,000 and 17,000,000
BB72 shots at p = 0.2 % and 0.3 %, and of 1,000,000 and 30,000,000 d = 13 shots at 0.6 % and 0.35 %. On BB72 it
used Cascade's eager accuracy path, the only path whose read keeps Cascade's 12 logits. `../table1_compiled/` then
counted the compiled paths whose failures Table I prints. The study's own documents call it ARB4 (see the
[name map](../README.md#names)).

## Paper numbers

- Table I, joint decision at d = 13: 4,886 and 4,884 failures at p = 0.6 % (single and batched), 2,306 and 2,308 at
  0.35 %, on the same shots as Cascade alone and the composition (`out/table1.json`). `../table1_compiled/` counts the
  same four numbers again, and the check script reads them from there.
- Appendix E: at d = 13 and p = 0.6 % in single mode, the exact McNemar test against Cascade alone gives p = 0.033,
  the largest of the eight (`out/table1.json`; `../table1_compiled/` gives the discordant counts, 289 and 239).
- On BB72 the paper prints the compiled paths' counts (`../table1_compiled/`). The eager counts here, 6,580 and 65,166,
  are the cross-check that `../table1_compiled/` reproduces exactly.

## Files

- `RESULTS.md`: the table, how the rule was applied, how the run went, and its commands.
- `code/`: `alpine_extract_bb.py` and `prep_bb.py` (the routed BB72 shots taken from the re-test's reads, checked and
  reduced), `run_scores.py` (the network's class scores on the pod, through `../fit_bb72/code/arb_scores.py`),
  `a3_pod.py` (the joint decision's answers) and `a3_bound.py` (how far the float16 storage of Cascade's logits can
  move them), `pod_run.sh`, and `table1.py` (the counts and the McNemar tests).
- `out/table1.json` and `out/table1.md`: the counts.
- `out/pod/`: the pod's outputs. `read/a3_summary.json` and `read/a3_bound.json` hold the counts on the rows where the
  decisions differ and the float16 bound: at 0.2 % no answer can change, and at 0.3 % the failures lie in
  [65,164, 65,171]. Per p, `read/p<p>/p<p>_a3.npz` holds each such row's answers (Cascade's, the network's, the
  joint decision's, the truth) with the score gap and its float16 bound. `p<p>_a3_open.npy` says per row whether the
  bound leaves the answer open and whether it may then repair or break, `p<p>_scores.json` holds the export's checks
  and `sel_info.json` the selection. `pod_read.sha256` gives the sha256 of the pod's per-shot files, and `pod_run.log`
  and `out/` hold the run's log, environment and logs.

## Rerunning it

The counts need the re-test's per-shot reads, which this repository does not hold. The authors archive them, and
RETEST rebuilds them from its seeds (`RETEST.md` sections 3.2 and 3.3). With the reads in place, `RESULTS.md`
("Commands") gives every step:

1. Extract and check the routed BB72 shots (`alpine_extract_bb.py`, `prep_bb.py`).
2. On one H100 80GB HBM3 (with the BB72 harness's GPU guard set to that card, as `run_scores.py` does), export the
   network's class scores and form the joint decision's answers (`pod_run.sh`, then `a3_bound.py`).
3. Count on any CPU: `table1.py --w13 0.006=7.86574441929562 0.0035=13.842311471679352`.

`out/pod/pod_read.sha256` gives the sha256 to compare the exported scores and answers against.

## Not included

The network's class scores on the routed BB72 shots where the decisions differ (`p<p>_scores.npy` and
`p<p>_scores_rows.npy`, 1.7 GB for the two p) and the reduced reads made from the re-test's reads.
`out/pod/pod_read.sha256` lists the sha256 of the class scores, and step 2 rebuilds them.
