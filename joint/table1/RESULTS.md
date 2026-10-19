# ARB4: the frozen joint decision on the test shots of Table I

2026-10-19. Table I (the Cascade table) counts failures of Cascade alone and of the
composition (the network, with its least confident shots routed to Cascade) on the re-test's reads. This adds a third
count on the same shots: the joint decision, frozen before this run. Its parameters were fitted on other shots (ARB's
and arb2's fresh reads, even chunks), so these shots are a held-out test.

## Result

| setting | path | shots | w | Cascade alone | composition | joint decision | repairs / breaks (vs composition) | change vs composition | McNemar p, joint vs composition | McNemar p, joint vs Cascade alone |
|---|---|---|---|---|---|---|---|---|---|---|
| BB72, p = 0.2 % | eager | 11,000,000 | 6.30957 | 8,063 | 7,981 | 6,580 | 1,433 / 32 | -17.6 % | < 1e-300 | < 1e-300 |
| BB72, p = 0.3 % | eager | 17,000,000 | 3.75837 | 75,957 | 75,700 | 65,166 | 11,348 / 814 | -13.9 % | < 1e-300 | < 1e-300 |
| d = 13, p = 0.6 % | single | 1,000,000 | 7.86574 | 4,936 | 4,917 | 4,886 | 249 / 218 | -0.6 % | 0.16 | 0.033 |
| d = 13, p = 0.6 % | batched | 1,000,000 | 7.86574 | 4,947 | 4,927 | 4,884 | 260 / 217 | -0.9 % | 0.054 | 0.0075 |
| d = 13, p = 0.35 % | single | 30,000,000 | 13.8423 | 2,354 | 2,325 | 2,306 | 90 / 71 | -0.8 % | 0.16 | 0.00061 |
| d = 13, p = 0.35 % | batched | 30,000,000 | 13.8423 | 2,352 | 2,322 | 2,308 | 86 / 72 | -0.6 % | 0.3 | 0.0016 |

- Failures are counts of shots whose logical observable(s) the decoder got wrong, out of the shots in the third
  column.
- Paths. BB72: the re-test's eager accuracy read (C14' run eagerly under bf16 autocast, S55'-deep on the fused
  engine). The compiled files hold C14''s compiled decisions and its smallest absolute logit (`bits1`/`conf1`,
  `bits4`/`conf4`, `bitsR`/`confR`) but none of its 12 logits, so the joint decision cannot be formed on the compiled
  paths and only the eager path is reported. d = 13: single = S1 and B1 (Cascade compiled at batch 1); batched = SB
  and BQ5 (Cascade compiled at batch 16 on the queue of routed shots); Cascade alone is B1 (single) and Bb (batched),
  as in the re-test.
- Repairs: shots only the composition fails; breaks: shots only the joint decision fails. The joint decision differs
  from the composition only on routed shots where the two decoders disagree, so these are all its changes.
- McNemar: exact two-sided test on the discordant shots (scipy `binomtest(b, b + c, 0.5)`; `bb.route.paired` for BB72,
  `sv_common.mcnemar_p` for d = 13, the same test). "< 1e-300" means the p-value underflows double precision.
- The Cascade-alone and composition counts equal Table I's in all six cells (`code/table1.py` refuses to run
  otherwise): BB72 eager 8,063 / 7,981 and 75,957 / 75,700; d = 13 single 4,936 / 4,917 and 2,354 / 2,325, batched
  4,947 / 4,927 and 2,352 / 2,322.

### What the table shows

- BB72: on the same shots the joint decision fails 17.6 % (0.2 %) and 13.9 % (0.3 %) less often than the composition,
  close to ARB's estimates on its own evaluation half (17.9 % and 14.2 %). It changes the answer on 3,185 of the 8,946
  routed shots where the two decoders disagree at 0.2 % (1,825 to S55'-deep's answer, 1,360 to a third class) and on
  28,869 of 94,413 at 0.3 % (16,568 and 12,301).
- d = 13: the joint decision takes the network's answer on 467 (single) and 477 (batched) of the 6,201 and 6,207
  routed disagreement shots at 0.6 %, and on 161 and 158 of 2,958 and 2,959 at 0.35 %. Net, it has 31 and 43 fewer
  failures than the composition at 0.6 % and 19 and 14 fewer at 0.35 % (0.6 % to 0.9 % of the composition's failures);
  none of these differences is significant (p = 0.054 to 0.30). Against Cascade alone it is better at both p and on
  both paths (p = 0.033 and 0.0075 at 0.6 %, 6.1e-4 and 1.6e-3 at 0.35 %; the composition's own p in Table I are
  0.020, 0.015, 3.7e-9 and 1.9e-9).

## The frozen joint decision, as applied

Both rules act only on routed shots whose two decisions differ; every other shot keeps the composition's answer.

- BB72, rule A3 of ARB (`~/work/arb` on the Linux workstation, branch `arb-exp` a354360c of the research repository): answer argmax over the 4,096 classes y of
  s(y) + w (y . z), with s the class scores of S55'-deep (the six-block network) from the fused engine and z C14''s 12
  logits, class bit j = observable j, ties to the lowest class index (`arb_analysis.a3_answers`, float64 on the GPU,
  first argmax). w = 6.309573444801933 at 0.2 % and 3.758374042884441 at 0.3 % (ARB's fit on its even chunks).
  Routing: S55'-deep's margin below D1's tau1, 7.3331888389587405 (0.2 %) and 7.60439167022705 (0.3 %).
- d = 13, rule J of arb2: answer 1 iff c + w z > 0 in float64, c the network's logit and z Cascade's (both float32 as
  saved); on a tie, Cascade's answer (arb2's `j_switch`). w = 7.86574441929562 at 0.6 % and 13.842311471679352 at 0.35
  %: rule J's fit per p on arb2's even chunks (its fresh shots: 4,000,000 at 0.6 %, 30,000,000 at 0.35 %), read from
  `~/work/arb2/out/analysis/summary.json` (`p.<p>.rules.J.params.w`; sha256 bd4e9f3d..., written when arb2's
  pod run ended, ALL_DONE at 02:40 PDT). arb2's own selection rule carried its rule P forward; J is used here because
  it is the joint decision frozen for this test. Routing: the lock's C5 tau1 on |c|, 5.242445659637451 (0.6 %) and
  8.111059150695802 (0.35 %), strictly below (`sv_common.routed`; release lock `cascade/d13/out/locked_d13.json`,
  sha256 455eb2a6...).

## How it ran

1. BB72 data. The re-test's eager reads (`<cluster-scratch>/retest/out/d1/<p>/read/p<p>.npz`, sha256
   69dcad1f... and 48d47b86..., equal to `retest/out/d1/confirm/npz_sha256.txt`) stay on Alpine. The link to Alpine
   (through the Windows workstation) ran at about 1.3 MB/s, so `code/alpine_extract_bb.py` (standard library only; Alpine's
   login node has no numpy) selected the rows there with arb_scores.py's rule (routed: `conf1` in float64 below tau1;
   differing: packed `pred0` != `pred1`) and streamed back every member of those rows plus `fail0`, `fail1` and
   `conf1` of every shot. Rows: 8,946 of 219,891 routed shots at 0.2 %, 94,413 of 1,699,201 at 0.3 %.
   `code/prep_bb.py` checked them (fail flags equal decisions against observables; the float16 logits' signs equal
   C14''s bits; min |z| equals the float16 of `conf0`; S55'-deep's class index equals its bits) and wrote reduced
   reads beside the reads' JSON.
2. GPU pod: one H100 80GB HBM3 (secure cloud, 3.99 USD/h; the RTX PRO 6000 was sold out), driver
   580.126.09, from 00:41 to 01:07 PDT (DONE written; about 2 USD). Setup is ARB's `pod_setup.sh` with D1's venv pins
   (`PINS_EQUAL`; Python 3.12.14, torch 2.11.0+cu128, triton 3.6.0, numpy 2.2.6). Code: the release at 94aaaa7f (ARB's
   bundle without the weights) and ARB's `arb_fused.py`, `arb_scores.py`, `arb_analysis.py` from `arb-exp` (sha256
   equal to `~/work/arb/code`). S55'-deep's checkpoint is the read's own file
   (`<cluster-scratch>/cascade_w3/ckpt/S55_dfm_deep/final_ema.pt`, sha256 051370c3..., as the read records).
3. Score export: `code/run_scores.py` runs `arb_scores.py` unchanged after setting pc_common's GPU guard to the H100
   (logged as `GPU_GUARD ... "lifted": true`). Every check passed bit for bit at both p: the read's own engine rerun
   on these rows gives the read's decisions and margins; the scoring engine gives the same decisions, margins and
   per-program top two; the exported scores give back each program's top two, the decision and the margin; the read's
   stored class index equals the bits. Mismatches: 0. On these rows the H100 gives the RTX read's decisions and
   margins (top score minus second score) bit for bit; the read never stored the other scores.
4. A3: `code/a3_pod.py` applies A3 with the frozen w through `arb_analysis.a3_answers`, with z the read's logits
   (`z0`). No exact ties occurred.
5. Float16 logits. The re-test's read keeps C14''s logits only in float16 (ARB fitted w on float32 logits), so A3 here
   uses the float16 values. `code/a3_bound.py` bounds the effect: any float32 logits that round to the stored values
   move each A3 score difference by at most w times the sum, over the bits where the two classes differ, of half the
   float16 spacing. At 0.2 % no answer can change, so 6,580 is exact. At 0.3 % 17 answers could change, which could
   repair at most 2 and break at most 5 shots: the joint decision's failures lie in [65,164, 65,171] whatever the
   float32 logits were.
6. d = 13 (no GPU). The per-chunk files `S1`, `SB`, `B1`, `Bb`, `BQ5`, `obs` of
   `<cluster-scratch>/retest/out/d13/conf/p0.0060` (chunks 0-99) and `p0.0035` (chunks 0-2999) were copied
   whole. `code/table1.py` loads them with the release's `sv_common.load_read` (every decoder's file must carry the
   observable file's chunk digests), builds C5 and C2 with `sv_common.assemble` and applies J from the saved float32
   logits.
7. Counts and tests: `code/table1.py` on this box (numpy and scipy in a throwaway venv; about 3 s).

## Commands

Run from `~/work/arb4` on the Linux workstation unless marked "pod". `<relay>` runs a command on the cluster's login
node through the internal relay.

```
# 1. BB72 rows, selected on Alpine's login node (the script travels on stdin)
cat code/alpine_extract_bb.py | <relay> 'python3 - <cluster-scratch>/retest/out/d1' \
  | tar xf - -C data/bb
# 2. the reads' JSON, the re-test's BB72 analysis and d = 13 report, the read's checkpoints (C14's was not used)
<relay> 'cd <cluster-scratch>/retest/out && tar cf - d1/confirm \
  d1/0.002/read/p0.00200.json d1/0.002/read/rows_0.002.json d1/0.002/read/seedcheck_0.002.json \
  d1/0.003/read/p0.00300.json d1/0.003/read/rows_0.003.json d1/0.003/read/seedcheck_0.003.json \
  d1/0.002/compiled/p0.00200.json d1/0.003/compiled/p0.00300.json d13/route \
  -C <cluster-scratch>/cascade_w3 ckpt/S55_dfm_deep/final_ema.pt ckpt/C14_dfm/final_ema.pt' | tar xf - -C data
# 3. d = 13 per-chunk outputs (BQ5 fetched in a second call)
<relay> 'cd <cluster-scratch>/retest/out/d13/conf && tar cf - \
  p0.0060/S1_* p0.0060/SB_* p0.0060/B1_* p0.0060/Bb_* p0.0060/obs_* p0.0060/read_*.json p0.0060/progress_* \
  p0.0035/S1_* p0.0035/SB_* p0.0035/B1_* p0.0035/Bb_* p0.0035/obs_* p0.0035/read_*.json p0.0035/progress_*' \
  | tar xf - -C data/d13
<relay> 'cd <cluster-scratch>/retest/out/d13/conf && tar cf - p0.0060/BQ5_* \
  p0.0035/BQ5_*' | tar xf - -C data/d13
# 4. checks and reduced reads
python3 code/prep_bb.py data/bb data/d1 data/stage/read
# 5. pod bundle: the release without weights, ARB's code from arb-exp, these scripts, the checkpoint, the reduced reads
cd data/stage && cat ~/work/arb/pod_release.tar | tar xf - --exclude='*.pt'
cat ~/work/arb/pod_extra.tar | tar xf - release/RELEASE_COMMIT relay_min
for f in arb_fused.py arb_scores.py arb_analysis.py pod_env.sh pod_setup.sh; do
  git -C <research-repo> show arb-exp:code/$f > code/$f; done
cp ~/work/arb4/code/{run_scores.py,a3_pod.py,pod_run.sh} code/; mkdir -p ckpt/S55_dfm_deep
cp ../ckpt/S55_dfm_deep/final_ema.pt ckpt/S55_dfm_deep/
tar cf - release code relay_min ckpt read | xz -T4 -6 > ../pod_bundle.tar.xz   # sha256 dd52bcac...
# (uploaded as 8 parts in parallel, joined and checked on the pod, unpacked under /workspace/arb4)
# 6. pod: the venv first (pod_setup.sh's install, started while the bundle was on its way), then the run
uv venv --python 3.12.14 /workspace/arb4/venv && uv pip install -p /workspace/arb4/venv/bin/python \
  --index-strategy unsafe-best-match --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple -r req.txt                        # req.txt = phase_c h100_req_cu128.txt
setsid nohup bash code/pod_run.sh > pod_run.log 2>&1 < /dev/null &            # setup, run_scores.py x 2, a3_pod.py
export W=/workspace/arb4; source code/pod_env.sh; $PY code/a3_bound.py $W/read
# copy-back (sha256 checked against the pod's in out/pod/pod_read.sha256), then DONE
ssh <pod> 'cd /workspace/arb4 && tar cf - pod_run.log out/env.json out/logs read/a3_summary.json read/a3_bound.json \
  read/*/*_scores.json read/*/*_a3.npz read/*/*_a3_open.npy read/*/sel_info.json' | tar xf - -C out/pod
# 7. local counts and tests
python3 -m venv data/venv && data/venv/bin/pip install numpy==2.2.6 scipy==1.18.1 stim==1.16.0
data/venv/bin/python code/table1.py --w13 0.006=7.86574441929562 0.0035=13.842311471679352 \
  --src "arb2 rule J, per p: ~/work/arb2/out/analysis/summary.json p.<p>.rules.J.params.w (arb2's fit on
  its own even chunks; status ALL_DONE 2026-10-19 02:40 PDT)"
```

## Files

- `out/table1.json`, `out/table1.md`: every count, the paired tests (with Newcombe intervals of the difference in
  failure rate), the routed and disagreement counts, and the Table I cross-check.
- `out/pod/`: the pod's logs (`setup.log`, `scores_<p>.log`, `a3.log`, `a3_bound.log`, `venv_freeze.txt`, `env.json`),
  the score-export records `read/<tag>/<tag>_scores.json` (the 4,096-score arrays themselves, 0.15 and 1.5 GB, were
  not copied back; their sha256 is in the records and in `pod_read.sha256`), and per disagreement row A3's answer with
  the three decoders' classes and the observable (`<tag>_a3.npz`) and the float16 bound flags (`<tag>_a3_open.npy`).
- `code/`: the scripts above. ARB's code that ran on the pod is on `arb-exp`, unchanged.

## Not done, and limits

- BB72 compiled paths: not computed (no logits saved there; see above).
- C14''s float32 logits were not recomputed; the float16 bound above covers them (0 at 0.2 %, -2 to +5 at 0.3 %).
- The pod was an H100, not the RTX PRO 6000 of the reads (pc_eval's GPU guard lifted by `code/run_scores.py`); every
  bitwise check against the read passed, so on these rows the decisions and margins are the read's.
- No timing: the cost of the joint decision (the 4,096 class scores and a 4,096 x 12 product per disagreement shot on
  BB72) is not measured.
- d = 13: arb2's w arrived at 02:43 PDT, so the fallback (fitting w on the re-test's even chunks) was not used and
  every chunk is reported. arb2 fitted w with Cascade's eager logits (the uncompiled module at batch 16); here it is
  applied to the compiled paths' logits (B1, BQ5), the paths Table I reports. Cascade's logits are not identical
  across paths: in the re-test B1's and Bb's decisions differ on 65 (0.6 %) and 30 (0.35 %) shots (its bitwise notes).
- No eager path at d = 13: the re-test has none (its Cascade computations are the compiled ones).
