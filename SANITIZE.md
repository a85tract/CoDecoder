# What changed for publication

The trees under `bb/`, `cascade/`, `head/`, `results/r17/`, `results/discordant/`, `training/surface/` and `training/bb/`
are copies of the programs and records as they ran. Before publication, identifying details in their text were
replaced. No number, rule, argument or result changed, and no tensor of any network changed.

- Hosts. Cluster node names became `gpu-node-NN`, `cpu-node-NN` and `mem-node-01`, one label per node across the
  whole repository. Container host names of rented GPU pods became `gpu-pod`, and the workstations' nicknames became
  "the Linux workstation", "the Windows workstation" (`windows-workstation`) and `mac-mini`.
- Paths and accounts. Cluster directories became `<cluster-scratch>`, `<cluster-projects>` and `<cluster-sw>`, home
  work directories `~/work`, a pod's home directory `<pod-home>`, the cluster account `<user>`, other accounts
  `<account>`, the internal git host `<hub-host>` and the research repository `<research-repo>`. Job scripts and a
  few defaults in the code keep these placeholders: they show what ran, and need your own paths to run again.
- Jobs. Cluster job ids became labels: `AJ01` onward here, and `J01`-`J30`, `CJ01`-`CJ74` and `CC01`-`CC16` from the
  earlier release, each used for one job throughout.
- People and tools. Internal notes, the names of the people and tools that ran the studies, and coordination files
  were removed, or the sentence was reworded around them. Reviews of the plans and results are cited by their finding
  numbers; the review texts are not included. A cut passage leaves a note in square brackets.
- Scripts. The scripts that copied code trees to the cluster or to rented pods and submitted jobs through an internal
  relay were removed. The job scripts they submitted are kept.

The per-tree files `bb/bb72/SANITIZE.txt`, `bb/gross/SANITIZE.txt`, `bb/gross_stage4/SANITIZE.txt`,
`cascade/bb72/SANITIZE.txt`, `results/r17/SANITIZE.txt`, `training/surface/SANITIZE.txt` and `training/bb/SANITIZE.txt`
list the earlier release's replacements file by file. Where one of them says that node names or job ids stay, this
repository has since replaced them as above.

Hash lists recorded before these replacements keep their values. The in-tree lists that cover files of this repository
(the H100 timing records under `cascade/bb72/cascade_compare/` and the BB72 policies in `bb/bb72/DEPLOYED_POLICIES`) were
checked after the replacements: no file fails that did not already fail in the earlier release, where the per-tree
`SANITIZE.txt` explains it. The network files are checked by `weights/MANIFEST.sha256`, which was written for this
repository.

The released BB and gross networks under `weights/bb/` are the original checkpoints byte for byte, because the frozen
code checks their sha256. The gross network's training record keeps one path of the rented pod it trained on
(`/workspace/g2/ckpt/...`), which names no host or user. No other checkpoint's metadata holds an absolute path, a host
name or an id.

The SPEED-RT stage-1 records under `speed_rt/cascade/records/` follow these rules with two exceptions, published byte for
byte: `H100/R4/derived_lock.json` and `H100/R4/session.json`. They keep their cluster paths (the lock's checkpoint
entries), the pod's home directory and, in `session.json`, the pod's host name, because the frozen analysis checks their
sha256; `speed_rt/cascade/results/README.md` says more.

The SPEED-RT stage-2 records under `speed_rt/s2/` follow these rules, with one map of replacements for all three
families. The stage's analyses then ran on the sanitized records, so their published outputs carry the same labels;
`speed_rt/s2/README.md` says more. The labels new there: the Linux workstation's host name became `linux-workstation`,
the GPUs' UUIDs `<uuid-1>` to `<uuid-3>` (`GPU-<uuid-N>` where the record keeps NVIDIA's prefix), a GPU's serial number
`<gpu-serial-1>`, the root file system's UUID `<fs-uuid>` and the BB draw's Slurm job `AJ41`. In the hosts' process
lists, the process name of a tool that ran the studies became `<tool>`. The files whose sha256 the frozen analyses
check are published byte for byte, and none of them held an identifier.

The joint decision's studies under `joint/` follow these rules and continue the labels: their Slurm jobs became `AJ47`
to `AJ56`, in file and directory names too, and two CPU nodes new here became `cpu-node-24` and `cpu-node-25`. The
files whose sha256 another of their records holds are published byte for byte, and 21 of them keep identifiers.
Seventeen keep a rented pod's container host name or its home directory (the fits' environment records, setup logs
and read records, and the pods' d = 13 sessions and locks). Four, `joint/timing/records/RTX/out/d13_rtx/session_0.json`,
`session_1.json`, `session_settled.json` and `lock_timed_20261019_035343.json`, keep the cluster account in their paths
and, in the three session files, the RTX node's host name, because the timing session's later records hold their
sha256; `joint/README.md` names them. One docstring line of `cascade/bb72/cascade_compare/phase_d/scored.py` named
the workstation and was reworded, so that file's sha256 differs from the digest the joint decision's checks recorded
(`joint/README.md`).
