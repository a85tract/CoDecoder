# SPEED-RT stage 1: results

`numbers.json`, `table.json`, `verdicts.json` and `stdout.txt` are the outputs of `speed_rt/cascade/analysis_cas.py` at
the freeze F_1 (73ca9f9), run twice on the records of the eight stage-1 jobs; the two runs' outputs were byte-identical.
`../records/` holds those records: `RTX/R1` to `RTX/R4` (Alpine's RTX PRO 6000) and `H100/R1` to `H100/R4` (the H100
pod), each a `drive.py` output folder as the job wrote it. Every slot of every round was established on both cards.

Records and outputs were sanitized for publication as `SANITIZE.md` describes. Cluster paths became `<cluster-scratch>`,
the cluster account `<user>` and the pod's home directory `<pod-home>`. The two RTX nodes became `gpu-node-06` and
`gpu-node-08`, the pod's host name `gpu-pod`, and the four Slurm jobs `AJ37` to `AJ40`. No number changed. Each round's
`SHA256SUMS` was recomputed for the sanitized files.

Two files of `H100/R4` name cluster paths in their checkpoint entries: the d = 13 lock derived for the H100
(`derived_lock.json`) and the session record it was derived from (`session.json`). Their sanitized copies therefore have
other sha256 values than the originals, which the analysis checked against the freeze and against each other. Run on
these published records, the analysis reports those checks for the H100's d = 13 rows and withholds them. The outputs
here come from the original records, in which every check passed.
