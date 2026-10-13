# BB72 stage 3, Amendment S: the fast (sector) Relay-BP as the fallback

Requested by the authors on 2026-10-07 at about 12:20 PDT and registered here before any test read of BB72 stage 3 is
analyzed. Stage 3's freeze is commit 19368aea (12:00 PDT); its test sets were sampled at 12:11 PDT (commit 2407497d).
This amendment adds two cells and changes nothing in the registered cells, their order or their analysis.

## Cells

| cell | code, p | hybrid | test shots | claim | speed condition |
|---|---|---|---|---|---|
| BB72-2S | [[72,12,6]], 0.002 | r12; shots with margin < tau = 2.979396343231201 (BB72-2F's frozen tau, q = 0.25 % on cal) go to sector Relay-BP (the registered opponent configuration) | split BB72-2 (the same 300,000 shots) | lower block failure rate than each of the three gating opponents: sector Relay-BP, BP-OSD ms80-CS60 (ldpc), cudaq-qec GPU BP-OSD ms80-CS60 | composite <= the fastest gating opponent |
| BB72-3S | [[72,12,6]], 0.003 | r12; margin < tau = 3.6329734325408936 (BB72-3F's frozen tau, q = 2 % on cal) to sector Relay-BP | split BB72-3 (the same 200,000 shots) | as BB72-2S | as BB72-2S |

- Statistics as Amendment F: each comparison at alpha = 0.05 / 3, the exact paired McNemar test, paired Newcombe
  intervals; a cell's claim holds only if all its comparisons and its speed condition hold.
- Accuracy comes from the registered test reads only: the network's margins and predictions, and sector Relay-BP's
  predictions on every test shot. A routed shot takes sector Relay-BP's prediction. No new decoding outcome is needed.
- Speed: composite = the network's registered GPU term + (the routed fraction on test) x sector Relay-BP's mean time
  per routed shot, timed on the quiet Mac mini under the registered protocol (one process, warm-up, idle machine) on the
  first 1,000 routed test shots of each split, with the same statistic as Amendment F's CPU term.
- Why these taus: they are BB72-2F's and BB72-3F's frozen taus, so S and F route the same shots and differ only in the
  fallback decoder. Nothing is tuned and no test data is used to set them.
- Expectation, disclosed in advance: in stage 2 the routing rule chose q = 0 on cal for all four policies with sector
  Relay-BP as the fallback, so these cells may not beat the network alone. They are reported whatever the outcome.
- The analysis: a second command, committed and pushed before it runs, run once after the registered analysis; its
  outputs pushed.

---
Copy note (2026-10-07): the text above this line is the file registered as commit 1be1daab on the GitHub branch
bb72-amendment-s at 12:19:37 PDT, before any test analysis of stage 3, with the host name in its speed bullet
replaced for publication (`bb72/SANITIZE.txt`); the registered file's sha256 is
0d7e5f0c8a108b505648d7717b29cb11d794adf327984b9b4f86ba307bb40e84. The analysis command is in
`data/bb/stage3/amendment_S.json`.
