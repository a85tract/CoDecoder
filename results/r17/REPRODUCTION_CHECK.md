# R17: a reader's run of README section 8

On 2026-10-10 the R17 commands of README section 8 ("R17 (d = 17 on T6)") ran once from a fresh clone of this
repository, as a reader would run them, on Alpine CPU nodes and in the environment of README section 1. This file
records what ran and what it printed.

The README meant here is the earlier release's; its sections 1 and 8 are now sections 1 and 8 of
`results/surface/SURFACE.md`. Paths below are the earlier release's: its `r17/`, `results/` and `testsets/` are
`results/r17/`, `results/surface/` and `evaluation/testsets/` here.

## Environment

- Clone: commit f31209f ("Merge R17: d = 17 re-registered at q = 8 % on test set T6"), the main branch before this
  check. `git status` stayed clean throughout (`.venv/` and `shots/` are in `.gitignore`).
- Python: Alpine's system Python 3.12 (3.12.15 on the login node that made the venv, 3.12.13 on the compute node that
  ran the blocks). pip installed `requirements.txt` without torch, which no step of section 8 imports: numpy 2.2.6,
  scipy 1.18.1, stim 1.16.0, PyMatching 2.4.0, safetensors 0.8.0 and matplotlib 3.11.2.
- Jobs: each block of section 8 was copied from the README into a file and run with `bash` from the clone's root, in a
  Slurm job on Alpine's acpu partition: T6 in one job (2 cores, 16 GB), the statistics and the composite in a second
  (1 core, 8 GB). Both jobs ran on node cpu-node-16, an AMD EPYC 9534 with AVX2.

The venv, made in the clone's root, and the T6 job without its log lines:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
grep -v '^torch' requirements.txt > ../req_notorch.txt && pip install -r ../req_notorch.txt
```

```bash
#SBATCH -p acpu --qos=cpu-normal -N 1 -n 1 -c 2 --mem=16G -t 00:30:00
cd <cluster-scratch>/rel17b/convlines_release && . .venv/bin/activate
bash ../jobs/t6_readme.sh > ../logs/t6_readme.log 2>&1      # section 8's first block, copied verbatim
```

## 1. T6

Section 8's first block, unchanged since R17 was added. The job printed:

```
host cpu-node-16
model name	: AMD EPYC 9534 64-Core Processor
avx2
avx512f
Python 3.12.13
numpy==2.2.6 stim==1.16.0 
f31209f Merge R17: d = 17 re-registered at q = 8 % on test set T6
start 2026-10-10T14:09:26-06:00
exit 0
end 2026-10-10T14:10:55-06:00
lines 100, True 100, False 0
files 100, 471M
```

The block itself printed one line per chunk file (name, seed, and whether its sha256 equals the one in
`testsets/T6.json`):

```
d17_p0.006_X_test6_000.npz 91170000 True
d17_p0.006_X_test6_001.npz 91170001 True
d17_p0.006_X_test6_002.npz 91170002 True
d17_p0.006_X_test6_003.npz 91170003 True
d17_p0.006_X_test6_004.npz 91170004 True
d17_p0.006_X_test6_005.npz 91170005 True
d17_p0.006_X_test6_006.npz 91170006 True
d17_p0.006_X_test6_007.npz 91170007 True
d17_p0.006_X_test6_008.npz 91170008 True
d17_p0.006_X_test6_009.npz 91170009 True
d17_p0.006_X_test6_010.npz 91170010 True
d17_p0.006_X_test6_011.npz 91170011 True
d17_p0.006_X_test6_012.npz 91170012 True
d17_p0.006_X_test6_013.npz 91170013 True
d17_p0.006_X_test6_014.npz 91170014 True
d17_p0.006_X_test6_015.npz 91170015 True
d17_p0.006_X_test6_016.npz 91170016 True
d17_p0.006_X_test6_017.npz 91170017 True
d17_p0.006_X_test6_018.npz 91170018 True
d17_p0.006_X_test6_019.npz 91170019 True
d17_p0.006_X_test6_020.npz 91170020 True
d17_p0.006_X_test6_021.npz 91170021 True
d17_p0.006_X_test6_022.npz 91170022 True
d17_p0.006_X_test6_023.npz 91170023 True
d17_p0.006_X_test6_024.npz 91170024 True
d17_p0.006_X_test6_025.npz 91170025 True
d17_p0.006_X_test6_026.npz 91170026 True
d17_p0.006_X_test6_027.npz 91170027 True
d17_p0.006_X_test6_028.npz 91170028 True
d17_p0.006_X_test6_029.npz 91170029 True
d17_p0.006_X_test6_030.npz 91170030 True
d17_p0.006_X_test6_031.npz 91170031 True
d17_p0.006_X_test6_032.npz 91170032 True
d17_p0.006_X_test6_033.npz 91170033 True
d17_p0.006_X_test6_034.npz 91170034 True
d17_p0.006_X_test6_035.npz 91170035 True
d17_p0.006_X_test6_036.npz 91170036 True
d17_p0.006_X_test6_037.npz 91170037 True
d17_p0.006_X_test6_038.npz 91170038 True
d17_p0.006_X_test6_039.npz 91170039 True
d17_p0.006_X_test6_040.npz 91170040 True
d17_p0.006_X_test6_041.npz 91170041 True
d17_p0.006_X_test6_042.npz 91170042 True
d17_p0.006_X_test6_043.npz 91170043 True
d17_p0.006_X_test6_044.npz 91170044 True
d17_p0.006_X_test6_045.npz 91170045 True
d17_p0.006_X_test6_046.npz 91170046 True
d17_p0.006_X_test6_047.npz 91170047 True
d17_p0.006_X_test6_048.npz 91170048 True
d17_p0.006_X_test6_049.npz 91170049 True
d17_p0.006_X_test6_050.npz 91170050 True
d17_p0.006_X_test6_051.npz 91170051 True
d17_p0.006_X_test6_052.npz 91170052 True
d17_p0.006_X_test6_053.npz 91170053 True
d17_p0.006_X_test6_054.npz 91170054 True
d17_p0.006_X_test6_055.npz 91170055 True
d17_p0.006_X_test6_056.npz 91170056 True
d17_p0.006_X_test6_057.npz 91170057 True
d17_p0.006_X_test6_058.npz 91170058 True
d17_p0.006_X_test6_059.npz 91170059 True
d17_p0.006_X_test6_060.npz 91170060 True
d17_p0.006_X_test6_061.npz 91170061 True
d17_p0.006_X_test6_062.npz 91170062 True
d17_p0.006_X_test6_063.npz 91170063 True
d17_p0.006_X_test6_064.npz 91170064 True
d17_p0.006_X_test6_065.npz 91170065 True
d17_p0.006_X_test6_066.npz 91170066 True
d17_p0.006_X_test6_067.npz 91170067 True
d17_p0.006_X_test6_068.npz 91170068 True
d17_p0.006_X_test6_069.npz 91170069 True
d17_p0.006_X_test6_070.npz 91170070 True
d17_p0.006_X_test6_071.npz 91170071 True
d17_p0.006_X_test6_072.npz 91170072 True
d17_p0.006_X_test6_073.npz 91170073 True
d17_p0.006_X_test6_074.npz 91170074 True
d17_p0.006_X_test6_075.npz 91170075 True
d17_p0.006_X_test6_076.npz 91170076 True
d17_p0.006_X_test6_077.npz 91170077 True
d17_p0.006_X_test6_078.npz 91170078 True
d17_p0.006_X_test6_079.npz 91170079 True
d17_p0.006_X_test6_080.npz 91170080 True
d17_p0.006_X_test6_081.npz 91170081 True
d17_p0.006_X_test6_082.npz 91170082 True
d17_p0.006_X_test6_083.npz 91170083 True
d17_p0.006_X_test6_084.npz 91170084 True
d17_p0.006_X_test6_085.npz 91170085 True
d17_p0.006_X_test6_086.npz 91170086 True
d17_p0.006_X_test6_087.npz 91170087 True
d17_p0.006_X_test6_088.npz 91170088 True
d17_p0.006_X_test6_089.npz 91170089 True
d17_p0.006_X_test6_090.npz 91170090 True
d17_p0.006_X_test6_091.npz 91170091 True
d17_p0.006_X_test6_092.npz 91170092 True
d17_p0.006_X_test6_093.npz 91170093 True
d17_p0.006_X_test6_094.npz 91170094 True
d17_p0.006_X_test6_095.npz 91170095 True
d17_p0.006_X_test6_096.npz 91170096 True
d17_p0.006_X_test6_097.npz 91170097 True
d17_p0.006_X_test6_098.npz 91170098 True
d17_p0.006_X_test6_099.npz 91170099 True
```

All 100 files have the manifest's sha256, and every seed is 91,170,000 + j. The files take 613,076,792 bytes (the
second job's `du --apparent-size`; the 471M above is the cluster file system's allocation at that moment).

## 2. The statistics

Section 8's second block, added with this check. It rebuilds each row's per-shot failures from the counts in
`results/r17_test_eval.csv`, recomputes the row with `convlines.stats.paired`, and compares p, the LER difference,
both intervals and phi with the file; δ comes from study 4's correlated-PM row in `results/s4_test_eval.csv`. It
printed:

```
M1_fp16_L3_B1         2708  12917  p 0  [-0.010467, -0.009955]  better
M1_fp16_L3_B1_corr    2708  12509  p 0  [-0.010055, -0.009550]  better
M1_fp16_L5_B1         2708  12954  p 0  [-0.010504, -0.009991]  better
M1_fp16_L5_B1_corr    2708  12502  p 0  [-0.010048, -0.009543]  better
M1_fp8_L3_B1          2708  12840  p 0  [-0.010389, -0.009878]  better
M1_fp8_L3_B1_corr     2708  12405  p 0  [-0.009950, -0.009447]  better
M1_fp8_L5_B1          2708  12886  p 0  [-0.010435, -0.009924]  better
M1_fp8_L5_B1_corr     2708  12452  p 0  [-0.009997, -0.009494]  better
M1_torch              2708  12961  p 0  [-0.010511, -0.009998]  better
M1_torch_corr         2708  12524  p 0  [-0.010070, -0.009565]  better
M4_fp16_L3_B1         2708   8109  p 0  [-0.005610, -0.005195]  better
M4_fp16_L3_B1_corr    2708   7939  p 0  [-0.005438, -0.005027]  better
M4_fp16_L5_B1         2708   8110  p 0  [-0.005610, -0.005196]  better
M4_fp16_L5_B1_corr    2708   7952  p 0  [-0.005451, -0.005040]  better
M4_fp8_L3_B1          2708   8072  p 0  [-0.005572, -0.005158]  better
M4_fp8_L3_B1_corr     2708   7902  p 0  [-0.005400, -0.004991]  better
M4_fp8_L5_B1          2708   8070  p 0  [-0.005570, -0.005156]  better
M4_fp8_L5_B1_corr     2708   7877  p 0  [-0.005375, -0.004966]  better
M4_torch              2708   8106  p 0  [-0.005606, -0.005192]  better
M4_torch_corr         2708   7955  p 0  [-0.005454, -0.005043]  better
pm                    2708  15658  p 0  [-0.013227, -0.012676]  better
pm_corr               2708   2943  p 9.9e-17  [-0.000299, -0.000172]  better
bposd_minllr           502   2114  p 7.7e-287  [-0.008593, -0.007541]  better
bposd_bp60             502   3141  p 0  [-0.013839, -0.012567]  better
q = 8 % vs 10 %: 2708 vs 2698 failures, p 0.54, interval [-2.33e-05, 4.33e-05], delta 8.169e-05 (file 8.169e-05), preserved True (file True)
accuracy cells held: 8 of 8
largest relative difference from the file: 0.0e+00
```

A p of 0 is below the smallest double (the README writes < 10⁻³⁰⁰). For the preservation row `convlines.stats` would
name the verdict "not shown" where R17's analysis wrote "level", so the block prints the registered decision instead:
preserved iff the upper end of the 97.5 % interval lies below δ.

## 3. The composite

Section 8's third block, added with this check. It recomputes (b) from the CPU job's records in
`r17/run/records/cpu_J23/` and adds study 4's (a) and (c). It printed:

```
(b) = 73.038 x 714.05 / 780.54 = 66.816 us; the q = 10 % term beside it: 81.704 us
ours = 66.660 + 66.816 + 1.543 = 135.019 us (r17_summary.json: 135.019)
16.33 us below study 4's 151.35: 14.89 same-node CPU saving + 1.44 (study 4's CPU session 83.14 vs this node at q = 10 %)
  M1             161.33 us, margin        26.31 us, holds True (file True)
  M4             165.97 us, margin        30.95 us, holds True (file True)
  M4_corr        275.79 us, margin       140.78 us, holds True (file True)
  M1_corr        297.51 us, margin       162.49 us, holds True (file True)
  pm             325.23 us, margin       190.21 us, holds True (file True)
  pm_corr        714.05 us, margin       579.03 us, holds True (file True)
  minllr      444502.44 us, margin    444367.43 us, holds True (file True)
  bp60        619209.00 us, margin    619073.98 us, holds True (file True)
```

## What did not run

- R17's GPU job on the RunPod H100 pod, and its CPU term: both need H100 numerics. The GPU job's reads depend on the
  deployed engine's c and Ising's TensorRT plans on the H100, and the CPU term's q = 8 % mask comes from the H100
  engine's c on V17b in study 4's development record, which is not published (README section 8).
- T6's BP-OSD shares and R17's analysis: they run `r17.py` from study 4's frozen tree, and the analysis also reads the
  per-shot outputs of the reads; neither is in this release.

## Fixes

No command needed a fix: the T6 block ran as written. Before this check the statistics and the composite had no
commands in section 8, only their description; the two blocks above went into section 8 byte for byte as they ran
here.
