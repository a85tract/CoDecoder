# The BB-code tests: BB72 and the gross code

## BB72: a registered test on the [[72,12,6]] bivariate bicycle code

A separate study tested the same idea on a qLDPC code: IBM's [[72,12,6]] bivariate bicycle code (BB72), X-basis
memory over 6 rounds under uniform circuit noise (the circuits of relay-bp's repository at revision d185194). The
network, ConvLines-BB, reads every detector of a shot and returns the 12 logical bits and one confidence margin. Two
trained copies are frozen: r11 decides the registered cell at p = 0.003, and r12 the one at p = 0.002 and every hybrid.
In the two registered cells the network decides every shot. In the two cells of amendment F, the shots whose margin is
below a frozen threshold τ go to full-DEM Relay-BP on the CPU. Amendment S, registered later the same day before any
test analysis, sends those shots to sector Relay-BP instead. Everything for this study is in [`bb/bb72/`](bb72/): the
frozen code, the two networks, the registration documents (`bb/bb72/ising_compare/bb/PLAN.md`, `STAGE3_PLAN.md`,
`AMENDMENT_F_draft.md`, `AMENDMENT_S.md`) and every record of the test (`bb/bb72/ising_compare/data/bb/stage3/`).

All six claims hold on the test sets. Without routing, the network fails less often than each of the three gating
opponents at both noise strengths, and its composite (0.102 ms) is below the fastest opponent's batch-1 time (0.957 ms
at p = 0.003, 0.501 ms at p = 0.002). Full-DEM Relay-BP, reported beside these two cells, is more accurate than the
network alone at p = 0.003 (1,092 against 1,341 failures) and level with it at p = 0.002 (274 against 254). Routing
the 0.24 % lowest-margin shots to full-DEM Relay-BP (BB72-2F) beats full-DEM Relay-BP itself, 171 against 274
failures, at a composite of 0.324 ms. At p = 0.003, routing 1.98 % of the shots (BB72-3F) also beats it, 877 against
1,092, at 3.43 ms, about 6 % of full-DEM Relay-BP's own 58.6 ms per shot. Amendment S sends the same shots to sector
Relay-BP. Both of its hybrids beat the three gating opponents, at composites of 0.115 ms (BB72-2S) and 0.260 ms
(BB72-3S), yet each fails more often than r12 alone on the same shots (290 against 254 failures at p = 0.002, 1,571
against 1,387 at p = 0.003), which the amendment expected from stage 2.

| cell | what | test shots | ours: failures (rate) | opponent | its failures | McNemar p | difference in failure rate, 98.33 % interval | verdict |
|---|---|---|---|---|---|---|---|---|
| BB72-3 | p = 0.003, network alone (registered) | 200,000 | 1,341 (0.670 %) | sector Relay-BP | 2,704 (1.352 %) | 1.8e-177 | [-0.741, -0.623] pp | better |
| | | | | BP-OSD (CPU) | 3,299 (1.649 %) | 4.0e-313 | [-1.044, -0.915] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 3,324 (1.662 %) | 5.8e-319 | [-1.057, -0.927] pp | better |
| BB72-2 | p = 0.002, network alone (registered) | 300,000 | 254 (0.085 %) | sector Relay-BP | 812 (0.271 %) | 6.0e-98 | [-0.209, -0.164] pp | better |
| | | | | BP-OSD (CPU) | 903 (0.301 %) | 5.4e-120 | [-0.240, -0.193] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 917 (0.306 %) | 4.9e-123 | [-0.245, -0.198] pp | better |
| BB72-2F | p = 0.002, hybrid with full-DEM Relay-BP fallback (amendment F) | 300,000 | 171 (0.057 %) | full-DEM Relay-BP | 274 (0.091 %) | 7.8e-22 | [-0.044, -0.026] pp | better |
| | | | | sector Relay-BP | 812 (0.271 %) | 1.9e-137 | [-0.237, -0.192] pp | better |
| | | | | BP-OSD (CPU) | 903 (0.301 %) | 1.8e-164 | [-0.268, -0.221] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 917 (0.306 %) | 2.8e-171 | [-0.273, -0.226] pp | better |
| BB72-3F | p = 0.003, hybrid with full-DEM Relay-BP fallback (amendment F, weaker speed claim) | 200,000 | 877 (0.439 %) | full-DEM Relay-BP | 1,092 (0.546 %) | 2.5e-31 | [-0.131, -0.085] pp | better |
| BB72-2S | p = 0.002, hybrid with sector Relay-BP fallback (amendment S) | 300,000 | 290 (0.097 %) | sector Relay-BP | 812 (0.271 %) | 7.9e-143 | [-0.193, -0.156] pp | better |
| | | | | BP-OSD (CPU) | 903 (0.301 %) | 3.9e-132 | [-0.227, -0.183] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 917 (0.306 %) | 2.9e-133 | [-0.232, -0.188] pp | better |
| BB72-3S | p = 0.003, hybrid with sector Relay-BP fallback (amendment S) | 200,000 | 1,571 (0.785 %) | sector Relay-BP | 2,704 (1.352 %) | 6.0e-282 | [-0.609, -0.525] pp | better |
| | | | | BP-OSD (CPU) | 3,299 (1.649 %) | 6.9e-319 | [-0.922, -0.807] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 3,324 (1.662 %) | 4.2e-322 | [-0.935, -0.819] pp | better |

| cell | composite (ms) | GPU term + handoff (ms) | CPU term (ms) | routed shots | bar (ms) | bar | speed | claim |
|---|---|---|---|---|---|---|---|---|
| BB72-3 | 0.102 | 0.102 | 0.000 | 0 (0.00 %) | 0.957 | cudaq-qec BP-OSD (GPU) | holds | holds |
| BB72-2 | 0.102 | 0.102 | 0.000 | 0 (0.00 %) | 0.501 | BP-OSD (CPU) | holds | holds |
| BB72-2F | 0.324 | 0.102 | 0.222 | 721 (0.24 %) | 0.501 | BP-OSD (CPU) | holds | holds |
| BB72-3F | 3.431 | 0.102 | 3.329 | 3,960 (1.98 %) | 58.596 | full-DEM Relay-BP's own mean | holds | holds |
| BB72-2S | 0.115 | 0.102 | 0.013 | 721 (0.24 %) | 0.501 | BP-OSD (CPU) | holds | holds |
| BB72-3S | 0.260 | 0.102 | 0.158 | 3,960 (1.98 %) | 0.957 | cudaq-qec BP-OSD (GPU) | holds | holds |

- BB72-3 against full-DEM Relay-BP (first 200,000 shots; reported, not part of the claim): 1,341 against 1,092 failures of 200,000, McNemar p 4.1e-11, worse.
- BB72-2 against full-DEM Relay-BP (first 200,000 shots; reported, not part of the claim): 166 against 173 failures of 200,000, McNemar p 6.9e-01, level.
- BB72-2 against full-DEM Relay-BP (all shots; reported, not part of the claim): 254 against 274 failures of 300,000, McNemar p 3.0e-01, level.
- BB72-2S against r12 alone on the same shots (reported, not part of the claim): 290 against 254 failures of 300,000.
- BB72-3S against r12 alone on the same shots (reported, not part of the claim): 1,571 against 1,387 failures of 200,000.

Speed is each cell's measured composite (GPU term on the H100 + handoff + the routed CPU part on one Apple M4 core) and
each opponent's time measured in the same session (sector Relay-BP and BP-OSD (ldpc) on one M4 core, cudaq-qec on the
H100, full-DEM Relay-BP's own time at p = 0.003), as measured, not converted to the Grace basis. Full-DEM Relay-BP's
own time was not measured at p = 0.002. Full-DEM Relay-BP is the fallback and the opponent of the claim in the F cells
and an accuracy reference elsewhere.

### How the test was run

- The networks (final EMA weights `weights/bb/r11_final_ema.pt` and `weights/bb/r12_final_ema.pt`, sha256 in
  `weights/MANIFEST.sha256`), every policy and τ, the
  opponents' configurations, the timing hosts and protocol, the test sizes and the analysis command were committed
  before the test set existed (freeze commit `19368aea`, 2026-10-07 12:00:15 PDT, mirrored to GitHub before the
  analysis).
- The test seeds come from drand mainnet round 6532307 (12:10:30 PDT), the first round at least 600 s after the
  freeze commit: seed = 2^62 + (the first 8 bytes of SHA-256(R ‖ TAG ‖ chunk index)) mod 2^62, with R the round's
  randomness and one TAG per split (`beacon.json`, `test_rows.csv`). Stim 1.16.0 sampled 200,000 shots at p = 0.003
  and 300,000 at p = 0.002 once, on an x86-64 host whose regenerated calibration chunks matched the stored ones bit for
  bit. Every decoder read the test once, and each analysis (the registered one and amendment S's) ran once.
- Each opponent was paired with ours on the same shots: sector Relay-BP (relay-bp 0.2.2 with the authors' tuned
  parameters), BP-OSD on the CPU (ldpc 2.4.1; min-sum, 80 iterations, OSD-CS order 60), NVIDIA cudaq-qec 0.8.0's GPU
  BP-OSD in the same configuration, and full-DEM Relay-BP (all detectors).
- Each comparison uses the exact two-sided McNemar test at α = 0.05/3 and the paired Newcombe interval (level 1 − α)
  of the difference in failure rate, which must lie entirely below 0. A cell's claim needs all of its comparisons and
  its speed condition. The three registered cells keep a family-wise 0.05 (the third, gross at p = 0.004, stopped at
  its probe gate before any test); with the two F cells the union bound over all claims of the study is 0.0833, and with
  the two S cells as well it is 0.1167.
- Speed is the composite: GPU term (the network through a fused Triton engine at batch 1, the slowest of 5 processes)
  plus CPU term (the routed fraction times the fallback's mean time per routed test shot, the slowest of 3 processes)
  plus handoff (the slowest measured, at least 1.1 µs). The bars come from the same hosts: cudaq-qec's batch-1 call on
  the H100, and sector Relay-BP and BP-OSD on one Apple M4 core, the fastest of 3 processes each. No time is converted
  between hosts, and the speed condition is a measured inequality with no statistical test.
- Amendment S was registered at 12:19:37 PDT as GitHub commit `1be1daab`, after the test set was sampled and before any
  test analysis. Its cells use amendment F's frozen τ values, so they route the same shots, and they take their accuracy
  from the registered reads. Its analysis command was pushed before it ran (`da8e6a00`). The CPU term was timed
  afterwards in a third quiet session on the M4 (14:48 PDT): sector Relay-BP on the first 1,000 routed test shots of
  each split, the slowest of 3 processes, with an idle check before each process (`load_S.log`).

### Disclosed changes and limits

- The registration named a full RTX PRO 6000 Blackwell for the GPU side and an exclusive Alpine CPU node for the CPU
  timing. Alpine was down on the test day, so the GPU side ran on one RunPod H100 80GB and the CPU timing on an Apple
  M4, one single-threaded process at a time on an otherwise idle machine. macOS cannot pin a core type; every CPU row
  records a fixed integer loop, which took 0.049-0.051 s in all 36 timed processes of the two registered sessions and
  0.049-0.053 s in the 6 of amendment S's session (an efficiency core would take about twice as long). The engine
  passed its validation again on the H100, and on all 800,000 calibration shots the H100 and RTX reads agree in every
  decision and every margin bit.
- Pre-freeze timing used the first 10,000 calibration shots where the plan named the first dev shots, because the dev
  split is stored only on Alpine.
- The F cells were added by an amendment registered before the test, with every choice made on the calibration split.
  BB72-3F's composite only has to beat full-DEM Relay-BP's own time; the fastest gating opponent's time is reported
  beside it.
- The composite is a benchmark score: the input is already on the GPU and the CPU part is an average, so it does not
  measure end-to-end latency. Nothing is claimed for other codes, noise strengths, noise models or hardware.
- Amendment S's idle check, as written in `amendment_S.json`, required that no python process run on the M4. A system
  Python idle since boot (0.0 % CPU) kept it false, so the check was narrowed to no python process at 1 % CPU or more
  before any timed process ran (`execution_notes` in that file).

### Reproduce

From `bb/bb72/ising_compare`, with `PYTHONPATH=.:../qldpc_neural`, `QND_RELAY` pointing at relay-bp's repository at
d185194 (its `tests/testdata`), and `BB_SHOTS`, `BB_PREDS` set to working directories:

```bash
python -m bb.shots --code bb72 --p 0.003 --split test --chunks 0-19 --rows rows.csv   # x86-64; reads data/bb/stage3/beacon.json
python -m bb.shots --code bb72 --p 0.002 --split test --chunks 0-29 --rows rows.csv
python -m bb.route infer --split test --code bb72 --p 0.003 --tag r11 --chunks 0-19 \
    --policy data/bb/policies/bb72_p0.003_rep1.json --ckpt ../../../weights/bb/r11_final_ema.pt --gpu "<your GPU's name>"
# likewise r12 with bb72_p0.003_rep2.json (chunks 0-19) and bb72_p0.002_rep2.json (--p 0.002, chunks 0-29)
python -m bb.opp --code bb72 --p 0.003 --split test --chunks 0-19 --dec relay,bposd,relay_full --procs 6 --out opp.csv
python -m bb.gpu_bposd read --code bb72 --p 0.003 --split test --chunks 0-19 --out gpu.csv
python -m bb.stage3 analyze --out results/
python -m bb.stage3 analyze --amend data/bb/stage3/amendment_S.json --out results/   # amendment S, from the stored rows
```

The network reads go through the commit guard: `bb/bb72/DEPLOYED` and `bb/bb72/DEPLOYED_POLICIES` hold the commit and the
policies' sha256 (of the policy files as published here; `bb/bb72/SANITIZE.txt`), and the engine must report the frozen signature (torch 2.11.0+cu128,
Triton 3.6.0, cuDNN 9.19) apart from the GPU name passed with `--gpu`, which must be the name PyTorch reports
(`torch.cuda.get_device_name(0)`, with the MIG profile on a MIG slice). Regenerated chunks match by their arrays:
`test_outputs_index.json` lists each chunk's packed-array sha256 (np.savez writes the time into the file, so file
hashes differ). `test_outputs_BB72-3.npz` and `test_outputs_BB72-2.npz` hold every decoder's per-shot predictions,
the networks' margins and the true observables, so the statistics can be recomputed without decoding again.

## BB144: a registered test on the [[144,12,12]] gross code

The BB study registered a third cell, G-4: IBM's [[144,12,12]] bivariate bicycle code (the gross code), X-basis
memory over 12 rounds under uniform circuit noise at p = 0.004, with the circuits of relay-bp's repository at revision
d185194. It stopped at its probe gate before any test. A later extension
(`PLAN_gross2.md`, `PLAN_gross2_stage2.md`) trained a new ConvLines-BB network, g2_D12k113c10, picked it on 50,000
stored shots (the D-bank) and set its routing threshold τ on 100,000 calibration shots. The cell was then registered
again (`STAGE3_GROSS_PLAN.md`) with the same claim and α, and tested on 300,000 new shots. The decoder is a hybrid:
the network decides every shot whose margin is at least τ, and the rest (0.5 % of the calibration shots) go to sector
Relay-BP on the CPU. Everything for this cell is in [`bb/gross/`](gross/): the frozen code, the network, the registration
documents and every record of the test (`bb/gross/ising_compare/data/bb/stage3_gross/`).

The claim holds on the test set. The hybrid fails on 1,177 of 300,000 shots (0.392 %), fewer than sector Relay-BP
(1,456), BP-OSD on the CPU (7,494) and cudaq-qec's GPU BP-OSD (7,460). Its composite time, 8.41 ms per shot, is below
the fastest of the three, cudaq-qec's batch-1 call on the same H100 at 15.65 ms. τ sent 1,462 test shots (0.49 %) to
Relay-BP. The network alone fails on 2,037 shots, more often than sector Relay-BP, so the claim is about the hybrid
only. Full-DEM Relay-BP, reported beside on the first 10,000 shots, is level with the hybrid there (32 against 36
failures, McNemar p 0.68).

| cell | what | test shots | ours: failures (rate) | opponent | its failures | McNemar p | difference in failure rate, 98.33 % interval | verdict |
|---|---|---|---|---|---|---|---|---|
| G-4 | p = 0.004, hybrid with sector Relay-BP fallback (q = 0.5 %) | 300,000 | 1,177 (0.392 %) | sector Relay-BP | 1,456 (0.485 %) | 4.5e-13 | [-0.124, -0.062] pp | better |
| | | | | BP-OSD (CPU) | 7,494 (2.498 %) | < 1e-300 | [-2.173, -2.039] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 7,460 (2.487 %) | < 1e-300 | [-2.162, -2.028] pp | better |

| cell | composite (ms) | GPU term + handoff (ms) | CPU term (ms) | routed shots | bar (ms) | bar | speed | claim |
|---|---|---|---|---|---|---|---|---|
| G-4 | 8.407 | 3.452 | 4.955 | 1,462 (0.49 %) | 15.653 | cudaq-qec BP-OSD (GPU) | holds | holds |

- G-4 against full-DEM Relay-BP (first 10,000 shots; reported, not part of the claim): 36 against 32 failures of
  10,000, McNemar p 6.8e-01, level. Its time was not measured.
- The network alone (reported, not part of the claim): 2,037 failures of 300,000 (0.679 %).
- The CPU opponents' times (the fastest of 3 processes each, one Apple M4 core): sector Relay-BP 60.98 ms and BP-OSD
  79.42 ms per shot.

Speed is the hybrid's measured composite (GPU term on the H100 + handoff + the routed CPU part on one Apple M4 core),
the network alone's GPU term + handoff, and each opponent's time from the test session (sector Relay-BP and BP-OSD
(ldpc) on one M4 core, cudaq-qec on the H100). No time is converted to the Grace basis. Full-DEM Relay-BP decoded only
the first 10,000 test shots and its own time was not measured in this test.

### How the gross test was run

- The network (final EMA weights `weights/bb/g2_D12k113c10_final_ema.pt`, sha256 in `weights/MANIFEST.sha256`), its
  engine (fp32 CUDA graph at
  batch 1, torch 2.11.0+cu126, NVIDIA H100 80GB HBM3), τ = 0.56292 (q = 0.5 %), the opponents' configurations, the
  timing hosts and protocol, the test size and the analysis command were committed before the test set existed
  (freeze commit `c68c3fa4`, 2026-10-07 15:45:30 PDT, mirrored to GitHub by 15:46:15 PDT). Before the freeze the
  engine re-read all 100,000 calibration shots on the test pod, and every decision and every margin bit matched the
  stage-2 read.
- The test seeds come from drand mainnet round 6532759 (15:56:30 PDT), the first round at least 600 s after the freeze
  commit was public: seed = 2^62 + (the first 8 bytes of SHA-256(R ‖ TAG ‖ chunk index)) mod 2^62, with R the round's
  randomness (`beacon.json`, `pod/test_rows.csv`). Stim 1.16.0 sampled the 300,000 shots once on the pod's x86-64 CPU,
  after a calibration chunk and a D-bank chunk regenerated there matched the stored ones bit for bit. Every decoder
  read the test once, and the analysis ran once.
- Each opponent was paired with the hybrid on the same shots: sector Relay-BP (relay-bp 0.2.2 with the authors'
  gross-code parameters, X-detector sector), BP-OSD on the CPU (ldpc 2.4.1; min-sum, 80 iterations, OSD-CS order 60),
  NVIDIA cudaq-qec 0.8.0's GPU BP-OSD in the same configuration, and full-DEM Relay-BP (both detector sectors) on the
  first 10,000 shots. Sector Relay-BP's test read also gives the routed shots their predictions. The CPU reads ran on
  one Derecho node (AMD EPYC 7763), the network's and cudaq-qec's on the pod.
- Each comparison uses the exact two-sided McNemar test at α = 0.05/3 and the paired Newcombe interval (level 1 − α)
  of the difference in failure rate, which must lie entirely below 0. The claim needs all three comparisons and the
  speed condition. 0.05/3 is G-4's share of the three registered cells' family-wise 0.05, so the union bound over the
  study's claims stays at 0.1167.
- Speed is the composite: GPU term (the network through the CUDA-graph engine at batch 1, the slowest of 5 processes,
  3.430 ms) plus handoff (the slowest measured, 22.6 µs; at least 1.1 µs) plus CPU term (the test's routed share,
  0.487 %, times sector Relay-BP's mean time per routed shot on the first 1,000 routed test shots, about 1.02 s, the
  slowest of 3 processes: 4.955 ms). The GPU term and handoff were measured on the pod before the freeze. The bars
  come from the same hosts: cudaq-qec's batch-1 call on the first test shots on the H100, and sector Relay-BP and
  BP-OSD on the first 1,000 test shots on one M4 core, the fastest of 3 processes each. A GPU process started only
  after the GPU showed 0 % utilization and less than 1 GiB of memory in use, and during it there was one compute app
  and the SM clock stayed at 97 % of its maximum or more. CPU processes ran one at a time, single-threaded, on the
  otherwise idle M4, after an untimed warm-up over 100 shots. No time is converted between hosts, and the speed
  condition is a measured inequality with no statistical test.

### Disclosed changes and limits (gross)

- The claim and α are those of `PLAN.md` (version 3). `STAGE3_GROSS_PLAN.md` registered these changes before the test:
  one network from the gross2 extension in place of the two registered replicas; no dev split (the D-bank chose the
  network and the calibration split set τ, so the test is the only out-of-sample paired evidence); the test size from
  the power rule on the calibration counts, raised to the 300,000 cap; the CUDA-graph engine on a RunPod H100 in place
  of the fused engine on an RTX PRO 6000; CPU timing on the M4 in place of an Alpine node; sampling on the pod; the
  CPU opponents' reads on Derecho.
- τ was set on the calibration split, in sample. On the test it routed 0.487 % of the shots, against 0.5 % there.
- Before the freeze, 20 cudaq-qec processes on the pod stopped at import, before any timing, because the relay test
  module that builds the X-sector DEM and its dependencies were missing. They were installed at the versions of BB72's
  pod and the timings repeated. 16 of the 47 GPU idle checks found the GPU still busy just after the
  previous process; each check repeated until it passed, and every timed process began right after a passing check.
- The plan named an x86 host for the analysis; it ran on the M4 instead (Python 3.12.14, NumPy 2.2.6, SciPy 1.18.1).
  The statistics do not depend on the host.
- macOS cannot pin a core type. Every CPU row records a fixed integer loop, which took 0.049-0.051 s in all nine
  timed processes of the test session (an efficiency core takes about twice as long).
- The composite is a benchmark score: the input is already on the GPU and the CPU part is an average, so it does not
  measure end-to-end latency. Nothing is claimed for other codes, noise strengths, noise models or hardware.

### Reproduce the gross test

From `bb/gross/ising_compare`, with `PYTHONPATH=.:../qldpc_neural`, `QND_RELAY` pointing at relay-bp's repository at
d185194 (its `tests/testdata`), and `BB_SHOTS`, `BB_PREDS` set to working directories:

```bash
python -m bb.shots --code bb144 --p 0.004 --split test --chunks 0-29 --rows rows.csv   # x86-64; reads data/bb/stage3_gross/beacon.json
python -m bb.stage3_gross infer --split test --chunks 0-29 --ckpt ../../../weights/bb/g2_D12k113c10_final_ema.pt
python -m bb.opp --code bb144 --p 0.004 --split test --chunks 0-29 --dec relay,bposd --procs 6 --out opp.csv
python -m bb.opp --code bb144 --p 0.004 --split test --chunks 0 --dec relay_full --slice 10 --procs 6 --out opp.csv
python -m bb.gpu_bposd read --code bb144 --p 0.004 --split test --chunks 0-29 --out gpu.csv
python -m bb.stage3_gross analyze --out results/
```

`infer` checks the checkpoint's sha256 and the frozen engine signature (torch 2.11.0+cu126, CUDA 12.6, cuDNN 91002,
an NVIDIA H100 80GB HBM3). On other hardware, compare with the stored predictions instead: `test_outputs_G-4.npz`
holds every decoder's per-shot test predictions, the network's margins and the true observables, and
`cal_outputs_G-4.npz` the calibration reads behind the test size. `outputs_index_G-4.json` lists each chunk's seed and
packed-array sha256, so regenerated chunks match by their arrays.

## BB144: a registered test of two hybrids on the [[144,12,12]] gross code

The study continued on IBM's [[144,12,12]] gross code (BB144): X-basis memory over 12 rounds under uniform circuit
noise at p = 0.004, with the circuits of relay-bp's repository at revision d185194. One network, g2_D12k113c10, reads
every detector of a shot and returns the 12 logical bits and a confidence margin. Here it runs through fusedt, the
BB72 engine's Triton kernels with launch settings tuned for the H100 (0.257 ms per shot at batch 1). Two hybrids were
registered and tested on one fresh test set of 150,000 shots. G-4R sends the shots whose margin is below τ1, the 1 %
quantile of the network's margins on the calibration split, to sector Relay-BP. G-4F sends those below τ2, the 2 %
quantile, to full-DEM Relay-BP. Both thresholds were frozen before the test set existed, and its seeds come from the
first drand round at least 600 s after the frozen commit was public. Everything for this test is in
[`bb/gross_stage4/`](gross_stage4/): the frozen code, the network (the gross network,
`weights/bb/g2_D12k113c10_final_ema.pt`), the registration document
(`bb/gross_stage4/ising_compare/bb/STAGE4_GROSS_PLAN.md`), the three calibration-only studies that chose the two cells
(`GROSS_FULLDEM_CAL.md`, `GROSS_FAST_ENGINE.md`, `GROSS_BPOSD_FALLBACK.md` in the same folder) and every record of
the test (`bb/gross_stage4/ising_compare/data/bb/stage4_gross/`).

Both claims hold. G-4R fails 387 times in 150,000 shots, against 716 for sector Relay-BP, 3,865 for BP-OSD (ldpc) and
3,789 for cudaq-qec, and its composite of 9.36 ms per shot is below the fastest of them, cudaq-qec at 13.90 ms on the
H100. G-4F fails 229 times. Full-DEM Relay-BP alone fails 484 times; the two disagree on 269 shots, and on 7 of those
the hybrid is the one that fails. G-4F also beats the three gating opponents. Its composite, 232 ms per shot, is about
a twelfth of full-DEM Relay-BP's own 2.77 s on one M4 core, which is the weaker speed claim this cell registered (as
BB72-3F did). It is much slower than the gating opponents, so G-4F is an accuracy result. The network alone fails 970
times, and G-4's earlier setting (the 0.5 % least confident shots to sector Relay-BP) fails 560 times with the same
engine.

| cell | what | test shots | ours: failures (rate) | opponent | its failures | McNemar p | difference in failure rate, 98.33 % interval | verdict |
|---|---|---|---|---|---|---|---|---|
| G-4R | hybrid, the 1 % least confident shots to sector Relay-BP | 150,000 | 387 (0.258 %) | sector Relay-BP | 716 (0.477 %) | 4.1e-61 | [-0.254, -0.187] pp | better |
| | | | | BP-OSD (CPU) | 3,865 (2.577 %) | < 1e-300 | [-2.416, -2.225] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 3,789 (2.526 %) | < 1e-300 | [-2.364, -2.175] pp | better |
| G-4F | hybrid, the 2 % least confident shots to full-DEM Relay-BP (weaker speed claim) | 150,000 | 229 (0.153 %) | full-DEM Relay-BP | 484 (0.323 %) | 4.0e-68 | [-0.198, -0.145] pp | better |
| | | | | sector Relay-BP | 716 (0.477 %) | 3.9e-79 | [-0.369, -0.283] pp | better |
| | | | | BP-OSD (CPU) | 3,865 (2.577 %) | < 1e-300 | [-2.523, -2.328] pp | better |
| | | | | cudaq-qec BP-OSD (GPU) | 3,789 (2.526 %) | < 1e-300 | [-2.471, -2.279] pp | better |

| cell | composite (ms) | GPU term + handoff (ms) | CPU term (ms) | routed shots | bar (ms) | bar | speed | claim |
|---|---|---|---|---|---|---|---|---|
| G-4R | 9.358 | 0.279 | 9.080 | 1,447 (0.96 %) | 13.899 | cudaq-qec BP-OSD (GPU) | holds | holds |
| G-4F | 232.187 | 0.279 | 231.908 | 2,983 (1.99 %) | 2,774.707 | full-DEM Relay-BP's own mean | holds | holds |

- Reported, not part of a claim: the network alone 970 failures of 150,000; G-4's old q (0.5 %, 720 shots to sector Relay-BP) 560.
- G-4F's composite beside the fastest gating opponent (reported, not part of the claim): 232.2 ms against 13.899 ms.

McNemar p values shown as < 1e-300 are below the smallest double; the analysis records them as 0.

### How the test was run

- GPU side: one RunPod H100 80GB HBM3. The engine re-read the calibration split bit for bit as in the fast-engine
  study, and τ1 and τ2 were set from that read. The GPU term is the slowest of three accepted fresh processes
  (0.2569 ms), the handoff the slowest of the same three (0.0219 ms). cudaq-qec's batch-1 call on the first 300 test
  shots is the fastest of three (13.90 ms).
- CPU reads: sector Relay-BP, BP-OSD and full-DEM Relay-BP decoded every test shot on Derecho CPU nodes (full-DEM
  Relay-BP took about 13.4 s per shot on a loaded node, about 590 core-hours in all). Accuracy does not depend on the
  host.
- CPU timing: an idle Apple M4 Mac mini, one single-threaded process at a time, each after an idle check, three fresh
  processes per quantity. The cells' CPU terms count the slowest process (sector Relay-BP on the first 1,000 routed
  shots, full-DEM Relay-BP on the first 100), the opponents' times the fastest (the first 1,000 test shots, and the
  first 300 for full-DEM Relay-BP's own mean).
- Statistics: per comparison, exact two-sided McNemar p < 0.05 / 3 and the paired Newcombe interval at 1 - 0.05 / 3
  below 0; a cell needs all its comparisons and its speed condition. Each cell spends a new share of 0.05 / 3, which
  brings the study-wide union bound to 9 x 0.05 / 3 = 0.15.
- The analysis command was committed with the frozen method and ran once, from a commit mirrored to GitHub before the
  run. Run again from `bb/gross_stage4/`, it writes `results.json`, `RESULTS_stage4_gross.md` and the figure byte for
  byte. `test4_outputs_G-4RF.npz` holds every decoder's per-shot predictions, the network's margins and the true
  observables, so the statistics can be recomputed without decoding again.

### Reproduce

From `bb/gross_stage4/ising_compare`, with `PYTHONPATH=.:../qldpc_neural`, `QND_RELAY` pointing at relay-bp's repository at
d185194 (its `tests/testdata`), and `BB_SHOTS`, `BB_PREDS` set to working directories:

```bash
python -m bb.shots --code bb144 --p 0.004 --split test4 --chunks 0-14 --rows rows.csv   # x86-64; reads data/bb/stage4_gross/beacon.json
python -m bb.stage4_gross infer --split test4 --chunks 0-14 --ckpt ../../../weights/bb/g2_D12k113c10_final_ema.pt
python -m bb.opp --code bb144 --p 0.004 --split test4 --chunks 0-14 --dec relay,bposd --procs 64 --out opp.csv
python -m bb.stage4_gross fullread --split test4 --chunks 0-14 --procs 64 --out opp.csv
python -m bb.gpu_bposd read --code bb144 --p 0.004 --split test4 --chunks 0-14 --out gpu.csv
python -m bb.stage4_gross analyze --out results/
```

`infer` checks the frozen engine signature: an H100 80GB HBM3 and the environment in
`data/bb/stage4_gross/pod/env.lock` (torch 2.11.0+cu126, Triton 3.6.0). Regenerated chunks match by their arrays:
`outputs_index_G-4RF.json` lists each chunk's packed-array sha256 (np.savez writes the time into the file, so file
hashes differ); after that check, the rows of `rows.csv` take the place of the test4 rows in
`data/bb/shot_manifest.csv`, whose file hashes `bb.shots` checks on every read.
