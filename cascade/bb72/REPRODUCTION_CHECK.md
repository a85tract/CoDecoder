# The Cascade comparison: the release check

On 2026-10-12 the Cascade comparison was checked on Alpine, each job in a fresh clone of the hub's branch
`cascade-results` at commit c263f7c ("Add the curve sets behind RQ3's p = 0.1 % numbers to cascade/"). Three CPU jobs
ran and passed. The GPU job, which loads the networks and decodes recorded shots again, ran later the same day and
passed too (its result is at the end). This file records what ran and what it printed. Paths are the release's
layout of the time: its `cascade/` is `cascade/bb72/` here, `cascade/weights` is `weights/cascade_bb72` and
`bb72/weights` is `weights/bb`.

## What ran

- Clone: the branch was cloned from the hub and bundled (sha256
  `6ad82ed87ddfccaa199eac7b5b9188444ccd5c9260dff6efb24d6dd2b21e007a`), the bundle was copied to the cluster with the
  same sha256, and each job cloned it again, so every job read the files of commit c263f7c and nothing else of this
  repository.
- Per-shot files: the archive directory on the cluster that holds the project's per-shot files (`ARCHIVE` below;
  manifest `revision/ARCHIVE_MANIFEST.md` on the research branch `ising-rev-archive`): `cascade_w4/out` (phase C),
  `cascade_d1/out` (D1) and `cascade_w2` (the curve sets).
- `CLONE` below is the job's clone, `DATA` the clone's `cascade/ising_compare/data/bb/cascade`, and `W2`, `REPORT`,
  `NUMBERS` and `OUT` are scratch directories. Each job also printed its node and the packages it ran with.

The manifest job (CPU), from the clone's `cascade/`. These are the GPU job's hashing steps, run on a CPU while that
job waits:

```bash
sha256sum -c --quiet MANIFEST.sha256
(cd weights && sha256sum -c --quiet MANIFEST.sha256)
(cd ../bb72/weights && sha256sum -c --quiet MANIFEST.sha256)
(cd ARCHIVE/cascade_d1 && sha256sum -c CLONE/cascade/cascade_compare/phase_d/out/npz_sha256.txt)
```

The archive job (CPU):

```bash
(cd ARCHIVE/cascade_w4/out && sha256sum -c CLONE/cascade/cascade_compare/phase_c/out/npz_sha256.txt)
cd ARCHIVE/cascade_w2
sha256sum -c --quiet DATA/preds_sha256.txt
awk -F, 'NR > 1 {print $2 "  shots/" $1}' DATA/shots_w2.csv | sha256sum -c --quiet -
for f in taus.json circuit_check.json pilot.json; do cmp out/$f DATA/$f; done
cmp out/rows_shots.csv DATA/shots_w2.csv
cmp out/report/curves_counts.json CLONE/cascade/ising_compare/bb/cascade/curves_counts.json
mkdir -p W2/out && ln -s ARCHIVE/cascade_w2/preds W2/preds && cp DATA/taus.json W2/out/
cd CLONE/cascade/ising_compare
W2_DIR=W2 QND_RELAY=RELAY PYTHONPATH=.:../qldpc_neural python -m bb.cascade.curves report --out REPORT
cmp REPORT/curves_counts.json bb/cascade/curves_counts.json; cmp REPORT/curves.png bb/cascade/curves.png
```

The numbers job (CPU), from the clone's root:

```bash
python cascade/repro/recompute_numbers.py --out NUMBERS
```

The GPU job (one RTX PRO 6000), from the clone's root, repeats the manifest job's four steps and then runs (it also
prints the driver and relay-bp's commit):

```bash
export QND_RELAY=RELAY PC_GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
python cascade/repro/check_networks.py --archive ARCHIVE/cascade_w4/out --d1 ARCHIVE/cascade_d1/out \
    --curves ARCHIVE/cascade_w2 --out OUT
python cascade/repro/recompute_numbers.py --out NUMBERS
```

## Results

With `--quiet`, `sha256sum -c` prints only the files that fail, so `rc=0` means that every listed file matched.

### The manifest job

```
node cpu-node-13, 2026-10-12T03:11:22-06:00
bundle sha256 6ad82ed87ddfccaa199eac7b5b9188444ccd5c9260dff6efb24d6dd2b21e007a
clone at c263f7ca9e1a3aff9de3f0e9503d39c73faba9fe
cascade/MANIFEST.sha256: rc=0 of 401 listed
cascade/weights/MANIFEST.sha256: rc=0 of 7 listed
bb72/weights/MANIFEST.sha256: rc=0 of 2 listed
out/read/p0.00200.npz: OK
out/read/p0.00300.npz: OK
out/compiled/p0.00200.npz: OK
out/compiled/p0.00300.npz: OK
D1 per-shot files: rc=0
done 2026-10-12T03:11:38-06:00
```

### The archive job

```
node cpu-node-13, 2026-10-12T02:57:41-06:00
bundle sha256 6ad82ed87ddfccaa199eac7b5b9188444ccd5c9260dff6efb24d6dd2b21e007a
clone at c263f7ca9e1a3aff9de3f0e9503d39c73faba9fe
phase C per-shot files: rc=0      68 OK
curve sets' predictions: rc=0 of 7310 listed
curve sets' shots: rc=0 of 1285 listed
taus.json: as archived
circuit_check.json: as archived
pilot.json: as archived
shots_w2.csv: as archived (out/rows_shots.csv)
curves_counts.json: as archived (out/report)
python 3.12.14 numpy 2.2.6 matplotlib 3.11.2
curves.py report (the release's taus.json, the archived predictions): rc=0
rebuilt curves_counts.json: identical to the release's
rebuilt curves.png: identical to the release's
done 2026-10-12T03:12:48-06:00
```

### The numbers job

```
node cpu-node-13, 2026-10-12T03:08:21-06:00
bundle sha256 6ad82ed87ddfccaa199eac7b5b9188444ccd5c9260dff6efb24d6dd2b21e007a
clone at c263f7ca9e1a3aff9de3f0e9503d39c73faba9fe
python 3.12.14 numpy 2.2.6 scipy 1.18.1
paper differs: D1 timing (RTX PRO 6000): batched speed-up over Cascade alone (two engines, two points): recomputed 5.1–16.4, paper 5.2–16.4
192 numbers; 70 compared with their records, 0 differing; 146 compared with the paper, 1 not as printed; 0 sections failed to run.
rc recompute_numbers=0, 2026-10-12T03:08:56-06:00
```

None of the 70 numbers that have a record differs from it. The one number not as printed is the low end of D1's
batched speed-up range over Cascade alone: 5.15 to two places (the README's table), so 5.1 to the paper's one place,
where the paper prints 5.2.

### The GPU job

It ran later on 12 October on one RTX PRO 6000. Both steps returned 0: `check_networks.py` (which exits 1 if any
network's check fails) and `recompute_numbers.py`. The numbers step printed the same counts as the CPU job: 192
numbers recomputed; 70 compared with their records, 0 differing; 146 compared with the paper, 1 not as printed. That
one is the same low end of D1's batched speed-up over Cascade alone, 5.1 against the 5.2 that the paper then
printed; the paper now prints 5.1.
