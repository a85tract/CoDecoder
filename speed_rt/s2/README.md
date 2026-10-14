# SPEED-RT stage 2: results

Stage 2 timed the three families that F_2 (7988e04) registered: the surface comparisons with Ising
(`speed_rt/STAGE2_surface.md`), the BB72 and gross comparisons (`speed_rt/STAGE2_bb.md`) and Decoder Switching
(`speed_rt/STAGE2_gapsw.md`), on the shots drawn from the beacon round that B_2 (ea1525c) records. The GPU terms ran on
rented H100 80GB HBM3 pods, one pod per role: S-low (surface, d = 7, 9 and 11), S-high (d = 13, 15 and 17) and BB. The
CPU terms of all three families ran in one session on the Linux workstation (AMD Ryzen 9 9950X, Ubuntu 24.04), with
the timed processes on CPUs 0-7, their SMT siblings offline and every other process kept off those CPUs
(`speed_rt/STAGE2_common.md` C1). Every slot of every family was established, and the CPU session accepted all 290 of
its processes at their first attempt.

Each family's folder holds `records/`, the records its analysis reads, and `results/`, the analysis's outputs:

- `surface/`, from `speed_rt/analysis_surf.py`: `analysis.json`, `table.md`, `rows.csv` and `fit_s2.json`, the stage's
  Grace fit, which the other two families read. Records: `low/` and `high/` (the two GPU roles' sessions), `cpu/` (the
  CPU session's surface slots and reference points) and `testsets/`.
- `bb/`, from `speed_rt/bb/analysis_bb.py`: `numbers.json`, `table.json` and `stdout.txt`. Records: `gpu/` (the BB
  pod's session), `cpu/` (its CPU processes) and `draw/` (the beacon record, the manifests and the checks of the draw).
- `gapsw/`, from `speed_rt/gapsw/analysis_gs.py`: `gs_numbers.json`, `gs_tables.md`, `gs_times.csv` and `gs_fit.csv`.
  Records: `session/` (the family's windows and processes in the CPU session), `prescreen/` (the session's host
  checks and reference) and `masks/` (each distance's routing mask, computed from the drawn shots).

Records were sanitized for publication as `SANITIZE.md` describes: identifiers were replaced in the records, with one
map for all three families, and no number changed. The files whose sha256 the frozen analyses check are published byte
for byte: in `surface/records/`, the CPU processes' records (`cpu/cpu/*/*.json`) and the pods' `masks/` and `res/`
folders; in `bb/records/`, `gpu/preds.sha256`; in `gapsw/records/`, `masks/*.npz`. None of them held an identifier. The
checksum lists inside the records were recomputed after the replacements. Each family's analysis then ran twice on
these published records; the two runs' outputs were byte-identical, and they are the results here. Put through the
same replacements, the outputs of the runs on the unsanitized records equal them except for the sha256 digests of the
input files that the replacements changed, which the outputs record.

## One amendment after the timing

Decoder Switching's analysis runs F_2's code with one line changed, in commit 54d9eec. Line 208 of
`speed_rt/gapsw/analysis_gs.py` checks that the manifest of the timing shots is the full draw. At F_2 it read

    assert man["files"].keys() == {G.ds_file(d, j) for d in G.DS_DISTANCES for j in range(20)}, "the manifest is not the full draw"

and since the amendment it reads

    assert man["files"].keys() == {G.ds_file(d, j) for d in G.DS_DISTANCES for j in range(G.sets()[G.ds_set(d)]["chunks"])}, "the manifest is not the full draw"

The authors cut the draw from 20 chunks of 10,000 shots per distance to 5 before the freeze, and this check was not
updated with it; F_2's registration text (`STAGE2_gapsw.md` 1.3) and its sets file (`speed_rt/gapsw/sets_gs.json`) say
5, and the draw, the hand-off and the CPU session used 5. The frozen code therefore refused the registered draw: on
these records it stops at line 208 with "the manifest is not the full draw" and writes no output. The amended line reads
the count from the frozen sets file, so the check enforces the registered 5 chunks per distance. No other line of
`analysis_gs.py` or `speed_rt/gapsw/gslib.py` depends on the old count of 20, and no Decoder Switching number existed
before the amendment. The surface and BB analyses ran F_2's code unchanged.
