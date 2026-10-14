# SPEED-RT stage 1: results

`numbers.json`, `table.json`, `verdicts.json` and `stdout.txt` are the outputs of `speed_rt/cascade/analysis_cas.py` at
the freeze F_1 (73ca9f9), run twice on the records of the eight stage-1 jobs; the two runs' outputs were byte-identical.
`../records/` holds those records: `RTX/R1` to `RTX/R4` (Alpine's RTX PRO 6000) and `H100/R1` to `H100/R4` (the H100
pod), each a `drive.py` output folder as the job wrote it. Every slot of every round was established on both cards.

Records and outputs were sanitized for publication as `SANITIZE.md` describes. Cluster paths became `<cluster-scratch>`,
the cluster account `<user>` and the pod's home directory `<pod-home>`. The two RTX nodes became `gpu-node-06` and
`gpu-node-08`, the pod's host name `gpu-pod`, and the four Slurm jobs `AJ37` to `AJ40`. No number changed. No checksum
list is published with the records: git fixes the published bytes, and the frozen analysis checks the hashes it needs.

Two files are the only ones published unsanitized, byte for byte as the H100 job wrote them: `H100/R4/derived_lock.json`,
the d = 13 lock derived for the H100, and `H100/R4/session.json`, the session record it was derived from. The frozen
analysis checks their sha256 against the freeze and against each other, so they keep what the others had replaced: the
cluster paths of the lock's checkpoint entries (in `derived_lock.json`), the pod's home directory (in both) and the pod's
host name (in `session.json`). Run on these published records, F_1's analysis reproduces the outputs here byte for byte,
the H100's d = 13 rows included.
