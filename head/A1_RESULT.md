# A1 result: the readout ablation, read once on fresh test shots

12 October 2026, times PDT. Research branch `ising-rev-a1`, freeze G = `2e69738e74be71c5cfbcb6661ecfa28a54048877`.
The plan (`revision/A1_READ_PLAN.md`), the design (`revision/REVISION_PLAN.md` section 2.1) and the notes of the pod
run (`A1_POD_RUN.md`) are research documents that this repository does not include. Every number below comes from
`results/<code>/` beside this file, the outputs of the frozen `a1.analyze` (`ising_compare/a1/analyze.py`), which are
byte-identical to the copies on the cluster. Each calibration and test set was drawn once and read once.

## 0. Verdicts

The rule was frozen before any test shot existed. A comparison is better or worse when its Holm-adjusted p-value (over
14 surface tests, 10 BB72 tests) is below 0.05, which needs both seeds to reject in the same direction; otherwise it is
inconclusive. Better means X has fewer failures than Y. Each comparison got the same verdict at q = 0 and at the
registered routed fraction (1 % on the surface, 2 % on BB72).

| code | id | factor (held fixed) | X vs Y | verdict, q = 0 and registered q | in words |
|---|---|---|---|---|---|
| surface | R1 | readout (no line loss) | S-P vs S-V | worse, worse | the pooled head fails more often than the line vote |
| surface | R2 | readout (no line loss) | S-L vs S-V | worse, worse | the centre line alone fails more often than the vote |
| surface | R3 | readout (line loss) | S-PA vs S-VA | inconclusive, inconclusive | the two seeds point in opposite directions |
| surface | U1 | supervision (vote read-out) | S-V vs S-VA | better, better | the vote fails less often without the line loss |
| surface | U2 | supervision (pooled read-out) | S-PA vs S-P | inconclusive, inconclusive | opposite directions; seed 1's S-P never trained |
| surface | W1 | whole design | S-L vs S-VA | worse, worse | the deployed design beats the centre line alone |
| surface | W2 | whole design | S-P vs S-VA | inconclusive, inconclusive | opposite directions; seed 1's S-P never trained |
| BB72 | R1 | readout (no translate loss) | B-P vs B-J | worse, worse | pooled outputs fail about 5 times as often as the joint score |
| BB72 | R2 | readout (translate loss) | B-PA vs B-JT | worse, worse | about 4 times as often standalone, 2.6-3 times at q = 2 % |
| BB72 | U1 | supervision (joint read-out) | B-J vs B-JT | inconclusive, inconclusive | no measurable effect of the translate loss on the joint head |
| BB72 | U2 | supervision (pooled read-out) | B-PA vs B-P | better, better | the translate loss helps the pooled head |
| BB72 | W1 | whole design | B-P vs B-JT | worse, worse | the deployed design beats the plain pooled head |

In plain English:
- BB72: at matched supervision a pooled head of the same size fails 3.8 to 5.0 times as often as the joint translate
  score standalone, and 2.6 to 4.6 times as often at the registered 2 % routing, in both seeds
  (R1, R2). With and without the translate loss the joint head's failures differ by at most 246 per million in either
  seed (U1); the loss helps only the pooled head (U2).
- Surface, d = 9: trained without the line loss, the line vote beats a pooled head of matched size and the centre line
  alone, in both seeds (R1, R2). Under the deployed supervision, with the line loss, the vote's edge over the pooled
  head is not established: seed 1 favours the pooled head by 335 failures per million, seed 2 favours the vote by 1,744
  (R3). The line loss hurts the vote: S-VA fails 28 % (seed 2) to 43 % (seed 1) more often than S-V (U1). The best
  surface network of the study is S-V, the vote trained without the line loss, with 10,638 and 10,881 failures per
  million. S-VA has 15,252 and 13,975, the deployed reference r764 12,293 (it was trained differently and sits outside
  the comparison), and correlated PM alone 13,080.
- In seed 1, S-P never trained (478,846 failures per million, close to always guessing the same answer), and S-PA
  failed the not-trained check at 2e7 samples but trained later (14,917). Both stay in the main table, as the plan
  requires. Their ramp reruns trained, and on seed 1 the vote's ramp rerun beat both pooled ramp reruns (section 1.6,
  descriptive).

## 1. Surface code, d = 9, p = 0.6 %

Sets: calibration 200,000 shots (`surface cal`, chunks 0-19, digest `6313c8e9cb11e09b...`) and test 1,000,000 shots
(`surface test`, chunks 0-99, digest `95d4d8324bf82e8b...`). Fallback: correlated PyMatching on the verdict DEM, 13,080
failures alone on the test set; its batch answers equal its single-shot answers on every shot. Reads in fp32 with TF32
off, at batch 4,096, on H200 `2g.35gb` MIG slices. In every read the first 2,000 shots, repeated at batch one, gave the
same decisions.

### 1.1 Standalone failures (q = 0)

On the 1,000,000 test shots, failures per million equal the failure counts.

| arm | what it is | seed 1: failures per million | seed 2: failures per million | cal failures (s1, s2) | not-trained check at 2e7: ratio (s1, s2) |
|---|---|---|---|---|---|
| S-VA | line vote + line loss (deployed design) | 15,252 | 13,975 | 3,045, 2,812 | 0.0643, 0.0636 |
| S-V | line vote, no line loss | 10,638 | 10,881 | 2,118, 2,139 | 0.0602, 0.0614 |
| S-L | centre line alone, no line loss | 17,088 | 15,059 | 3,356, 3,009 | 0.152, 0.14 |
| S-P | pooled head, no line loss | 478,846 | 13,750 | 95,545, 2,737 | 1 (not trained), 0.255 |
| S-PA | pooled head + line loss (training-only line read-out) | 14,917 | 15,719 | 2,946, 3,103 | 1 (not trained), 0.154 |

Reference r764, the deployed d = 9 network (grown from a 2-block network, outside the controlled comparison): 12,293
failures per million. Correlated PM alone: 13,080.

### 1.2 Failures at each routed fraction, against the fallback alone

A shot whose confidence falls below the network's calibration quantile tau_q goes to correlated PM, and every arm takes
the same PM answers. Failures per million test shots of each network-plus-fallback pair:

| network (seed) | q = 0 % | q = 0.5 % | q = 1 % | q = 2 % | q = 4 % | q = 8 % | q = 16 % |
|---|---|---|---|---|---|---|---|
| S-VA s1 | 15,252 | 14,083 | 13,239 | 12,136 | 11,575 | 12,034 | 12,694 |
| S-VA s2 | 13,975 | 12,909 | 12,217 | 11,522 | 11,405 | 12,071 | 12,749 |
| S-V s1 | 10,638 | 10,094 | 9,942 | 10,203 | 11,146 | 12,150 | 12,841 |
| S-V s2 | 10,881 | 10,314 | 10,069 | 10,249 | 11,093 | 12,149 | 12,850 |
| S-L s1 | 17,088 | 15,733 | 14,628 | 13,196 | 12,075 | 12,175 | 12,726 |
| S-L s2 | 15,059 | 13,900 | 13,076 | 12,103 | 11,566 | 12,060 | 12,728 |
| S-P s1 | 478,846 | 476,451 | 474,111 | 469,398 | 459,991 | 441,379 | 403,144 |
| S-P s2 | 13,750 | 12,745 | 12,017 | 11,274 | 11,281 | 12,075 | 12,770 |
| S-PA s1 | 14,917 | 13,871 | 12,981 | 12,015 | 11,556 | 12,100 | 12,731 |
| S-PA s2 | 15,719 | 14,540 | 13,610 | 12,510 | 11,868 | 12,096 | 12,715 |
| reference r764 | 12,293 | 11,517 | 10,976 | 10,815 | 11,211 | 12,123 | 12,828 |
| fallback alone | 13,080 | 13,080 | 13,080 | 13,080 | 13,080 | 13,080 | 13,080 |

Shots the reference r764 routed to PM at each q: 5,046 at q = 0.5 %, 10,119 at q = 1 %, 20,022 at q = 2 %, 40,236 at q = 4 %, 80,185 at q = 8 %, 160,342 at q = 16 %.

The routed share of the test shots stayed within 0.1 percentage point of q for every network. S-V's pairs have the
fewest failures at every q up to 4 %: 9,942 and 10,069 per million at q = 1 %, 23 to 24 % below PM alone. S-VA's pairs
bottom out at q = 4 % (11,575 and 11,405). By q = 16 % every trained network's pair sits at 12,694-12,850, close to PM
alone. The figure `results/surface/pair_curves.png` is drawn on a log axis that S-P's untrained seed 1
stretches to 500,000; the table is easier to read for the trained networks.

### 1.3 Paired tests, per seed

The exact two-sided McNemar test on that seed's discordant shots ("only X": shots that X fails and Y does not), and the
difference X - Y in failures per million test shots with Newcombe's 95 % interval. Discordant counts are never summed
over seeds.

| id | X vs Y (held fixed) | q | seed | failures X, Y | only X / only Y | X - Y per million [95 %] | McNemar p |
|---|---|---|---|---|---|---|---|
| R1 | S-P vs S-V (no line loss) | 0 % | 1 | 478,846, 10,638 | 473,474 / 5,266 | +468,208 [+467,209, +469,206] | < 1e-300 |
| R1 | S-P vs S-V (no line loss) | 0 % | 2 | 13,750, 10,881 | 5,510 / 2,641 | +2,869 [+2,693, +3,046] | 3.2e-226 |
| R1 | S-P vs S-V (no line loss) | 1 % | 1 | 474,111, 9,942 | 469,027 / 4,858 | +464,169 [+463,172, +465,165] | < 1e-300 |
| R1 | S-P vs S-V (no line loss) | 1 % | 2 | 12,017, 10,069 | 3,846 / 1,898 | +1,948 [+1,800, +2,097] | 2e-148 |
| R2 | S-L vs S-V (no line loss) | 0 % | 1 | 17,088, 10,638 | 9,325 / 2,875 | +6,450 [+6,235, +6,667] | < 1e-300 |
| R2 | S-L vs S-V (no line loss) | 0 % | 2 | 15,059, 10,881 | 7,152 / 2,974 | +4,178 [+3,981, +4,376] | < 1e-300 |
| R2 | S-L vs S-V (no line loss) | 1 % | 1 | 14,628, 9,942 | 6,931 / 2,245 | +4,686 [+4,499, +4,874] | < 1e-300 |
| R2 | S-L vs S-V (no line loss) | 1 % | 2 | 13,076, 10,069 | 5,185 / 2,178 | +3,007 [+2,839, +3,176] | 8.4e-277 |
| R3 | S-PA vs S-VA (line loss) | 0 % | 1 | 14,917, 15,252 | 4,878 / 5,213 | -335 [-532, -138] | 0.00088 |
| R3 | S-PA vs S-VA (line loss) | 0 % | 2 | 15,719, 13,975 | 5,880 / 4,136 | +1,744 [+1,548, +1,940] | 2.9e-68 |
| R3 | S-PA vs S-VA (line loss) | 1 % | 1 | 12,981, 13,239 | 3,620 / 3,878 | -258 [-428, -88] | 0.003 |
| R3 | S-PA vs S-VA (line loss) | 1 % | 2 | 13,610, 12,217 | 4,414 / 3,021 | +1,393 [+1,224, +1,562] | 5.8e-59 |
| U1 | S-V vs S-VA (vote read-out) | 0 % | 1 | 10,638, 15,252 | 2,569 / 7,183 | -4,614 [-4,808, -4,421] | < 1e-300 |
| U1 | S-V vs S-VA (vote read-out) | 0 % | 2 | 10,881, 13,975 | 2,506 / 5,600 | -3,094 [-3,271, -2,918] | 3e-265 |
| U1 | S-V vs S-VA (vote read-out) | 1 % | 1 | 9,942, 13,239 | 2,006 / 5,303 | -3,297 [-3,465, -3,130] | < 1e-300 |
| U1 | S-V vs S-VA (vote read-out) | 1 % | 2 | 10,069, 12,217 | 1,942 / 4,090 | -2,148 [-2,301, -1,996] | 6.7e-172 |
| U2 | S-PA vs S-P (pooled read-out) | 0 % | 1 | 14,917, 478,846 | 7,402 / 471,331 | -463,929 [-464,935, -462,923] | < 1e-300 |
| U2 | S-PA vs S-P (pooled read-out) | 0 % | 2 | 15,719, 13,750 | 6,090 / 4,121 | +1,969 [+1,771, +2,167] | 5.4e-85 |
| U2 | S-PA vs S-P (pooled read-out) | 1 % | 1 | 12,981, 474,111 | 6,425 / 467,555 | -461,130 [-462,132, -460,128] | < 1e-300 |
| U2 | S-PA vs S-P (pooled read-out) | 1 % | 2 | 13,610, 12,017 | 4,595 / 3,002 | +1,593 [+1,422, +1,764] | 4.5e-75 |
| W1 | S-L vs S-VA (neither (credited to neither)) | 0 % | 1 | 17,088, 15,252 | 7,535 / 5,699 | +1,836 [+1,611, +2,062] | 1.9e-57 |
| W1 | S-L vs S-VA (neither (credited to neither)) | 0 % | 2 | 15,059, 13,975 | 5,802 / 4,718 | +1,084 [+883, +1,285] | 4.2e-26 |
| W1 | S-L vs S-VA (neither (credited to neither)) | 1 % | 1 | 14,628, 13,239 | 5,651 / 4,262 | +1,389 [+1,194, +1,584] | 2.6e-44 |
| W1 | S-L vs S-VA (neither (credited to neither)) | 1 % | 2 | 13,076, 12,217 | 4,305 / 3,446 | +859 [+687, +1,032] | 1.7e-22 |
| W2 | S-P vs S-VA (neither (credited to neither)) | 0 % | 1 | 478,846, 15,252 | 471,082 / 7,488 | +463,594 [+462,587, +464,600] | < 1e-300 |
| W2 | S-P vs S-VA (neither (credited to neither)) | 0 % | 2 | 13,750, 13,975 | 4,538 / 4,763 | -225 [-414, -36] | 0.020 |
| W2 | S-P vs S-VA (neither (credited to neither)) | 1 % | 1 | 474,111, 13,239 | 467,338 / 6,466 | +460,872 [+459,870, +461,874] | < 1e-300 |
| W2 | S-P vs S-VA (neither (credited to neither)) | 1 % | 2 | 12,017, 12,217 | 3,275 / 3,475 | -200 [-361, -39] | 0.015 |

### 1.4 Intersection-union and Holm verdicts

p_iut is the larger of the two seeds' p-values when both differences have the same sign, and 1 otherwise. Holm runs
over all 14 surface tests. The seed mean's interval treats the test shot as the unit (each shot's difference is
averaged over the seeds first), so it reflects shot noise only, not the spread between training runs.

| id | factor | X vs Y | q | p_iut | p_holm | seed mean X - Y per million [95 %, shot-clustered] | seed range | verdict |
|---|---|---|---|---|---|---|---|---|
| R1 | readout | S-P vs S-V | 0 % | 3.2e-226 | 3.5e-225 | +235,538 [+235,031, +236,046] | +2,869 to +468,208 | worse |
| R1 | readout | S-P vs S-V | 1 % | 2e-148 | 1.8e-147 | +233,058 [+232,554, +233,563] | +1,948 to +464,169 | worse |
| R2 | readout | S-L vs S-V | 0 % | < 1e-300 | < 1e-300 | +5,314 [+5,150, +5,478] | +4,178 to +6,450 | worse |
| R2 | readout | S-L vs S-V | 1 % | 8.4e-277 | 1.1e-275 | +3,846 [+3,704, +3,989] | +3,007 to +4,686 | worse |
| R3 | readout | S-PA vs S-VA | 0 % | 1 | 1 | +704 [+560, +849] | -335 to +1,744 | inconclusive |
| R3 | readout | S-PA vs S-VA | 1 % | 1 | 1 | +568 [+443, +692] | -258 to +1,393 | inconclusive |
| U1 | supervision | S-V vs S-VA | 0 % | 3e-265 | 3.6e-264 | -3,854 [-4,002, -3,706] | -4,614 to -3,094 | better |
| U1 | supervision | S-V vs S-VA | 1 % | 6.7e-172 | 6.7e-171 | -2,722 [-2,852, -2,593] | -3,297 to -2,148 | better |
| U2 | supervision | S-PA vs S-P | 0 % | 1 | 1 | -230,980 [-231,494, -230,466] | -463,929 to +1,969 | inconclusive |
| U2 | supervision | S-PA vs S-P | 1 % | 1 | 1 | -229,768 [-230,278, -229,259] | -461,130 to +1,593 | inconclusive |
| W1 | whole design | S-L vs S-VA | 0 % | 4.2e-26 | 3.3e-25 | +1,460 [+1,296, +1,624] | +1,084 to +1,836 | worse |
| W1 | whole design | S-L vs S-VA | 1 % | 1.7e-22 | 1.2e-21 | +1,124 [+983, +1,265] | +859 to +1,389 | worse |
| W2 | whole design | S-P vs S-VA | 0 % | 1 | 1 | +231,684 [+231,170, +232,199] | -225 to +463,594 | inconclusive |
| W2 | whole design | S-P vs S-VA | 1 % | 1 | 1 | +230,336 [+229,827, +230,845] | -200 to +460,872 | inconclusive |

### 1.5 The supervision-matched read-out contrasts

Only the read-out contrasts at matched supervision credit the read-out, and they are the ones the stage-1 rule watched:
R1 and R2 without the line loss, R3 with it.
- R1, S-P against S-V: worse. Seed 1's S-P never trained. Seed 2's S-P trained and still failed on 13,750 shots
  against S-V's 10,881: +2,869 [+2,693, +3,046] per million, p = 3.2e-226, and +1,948 [+1,800, +2,097] at q = 1 %.
- R2, S-L against S-V: worse in both seeds, +6,450 and +4,178 per million at q = 0 and +4,686 and +3,007 at q = 1 %.
- R3, S-PA against S-VA: inconclusive. Seed 1: -335 [-532, -138] per million, p = 0.00088, the pooled head ahead.
  Seed 2: +1,744 [+1,548, +1,940], p = 2.9e-68, the vote ahead. Each seed's difference lies well outside its own
  shot-noise interval, in opposite directions, so here the spread between training runs is larger than the effect.
  The seed mean, +704 [+560, +849], leans towards the vote, but its interval leaves out that spread, and the frozen
  rule gives no verdict.

The supervision contrasts at matched read-out: U1, S-V against S-VA, is better in both seeds (-4,614 and -3,094 per
million at q = 0; -3,297 and -2,148 at q = 1 %): the vote fails less often without the line loss. U2, S-PA against S-P, is
inconclusive: seed 1 favours the line loss only because its S-P never trained, and in seed 2 S-PA fails more often by
1,969 per million.

### 1.6 Ramp reruns and the not-trained checks (seed 1)

Seed 1's S-P and S-PA stayed at their constant-guess loss at 2e7 samples (ratio 1.000), so S-P, S-PA and the deployed
S-VA of seed 1 were each trained once more with the noise ramp (p rising from 0.1 % to 0.6 % over the first quarter of
the same 2e8 samples). All three ramp reruns passed the check. Their comparison with the ramp rerun of S-VA is
descriptive and outside the Holm family:

| seed | ramp rerun vs ramp rerun of the deployed arm | q | failures | only arm / only deployed | arm - deployed per million [95 %] | McNemar p |
|---|---|---|---|---|---|---|
| 1 | S-P vs S-VA (ramp) | 0 % | 16,339 vs 15,169 | 6,016 / 4,846 | +1,170 [+966, +1,374] | 3e-29 |
| 1 | S-P vs S-VA (ramp) | 1 % | 14,189 vs 13,162 | 4,615 / 3,588 | +1,027 [+850, +1,205] | 8e-30 |
| 1 | S-PA vs S-VA (ramp) | 0 % | 15,851 vs 15,169 | 5,184 / 4,502 | +682 [+489, +875] | 4.4e-12 |
| 1 | S-PA vs S-VA (ramp) | 1 % | 13,693 vs 13,162 | 3,864 / 3,333 | +531 [+365, +697] | 4.1e-10 |

ramp checks: a1ramp_S-VA_s1 0.00173, a1ramp_S-P_s1 0.0264, a1ramp_S-PA_s1 0.0128

With the ramp both pooled heads train, and on seed 1 both still fail more often than the vote with the line loss
(+1,170 and +682 per million at q = 0).

### 1.7 Sizes of the decision paths

| arm | parameters | read-out multiply-adds | vs deployed |
|---|---|---|---|
| S-VA | 389,395 | 16,394,841 | 1.000 |
| S-V | 389,395 | 16,394,841 | 1.000 |
| S-L | 388,235 | 12,307,409 | 0.751 |
| S-P | 507,115 | 16,388,361 | 1.000 |
| S-PA | 507,115 | 16,388,361 | 1.000 |

The pooled heads match the deployed read-out's multiply-adds to within 0.1 % and have 30 % more parameters. The
centre line uses 75 % of the read-out's multiply-adds. S-PA's training-only line read-out (18
tensors) is not in its decision path.

## 2. BB72, p = 0.3 %

Sets: calibration 200,000 shots (`bb72 a1cal`, chunks 0-19, digest `fd05bc8b40d69031...`) and test 500,000 shots
(`bb72 a1test`, chunks 0-49, digest `afbeca6530dc109b...`). Fallback: full-DEM Relay-BP under the policy locked for A1
(relay-bp 0.2.2, `qnd.teachers.CONFIGS['relay']`, a new decoder per shot seeded with the shot's index in its set):
2,691 failures alone, 5,382 per million, at 0.130 s a shot including a 0.013 s decoder build, on 64 cores of an Alpine
CPU node. Reads on RTX PRO 6000 `1g.24gb` MIG slices; the batch-one check gave the same decisions in all 18 reads. Every
BB72 run passed its not-trained check, so BB72 has no ramp rerun.

### 2.1 Standalone failures (q = 0)

Failures per million are twice the failure counts on the 500,000 test shots.

| arm | what it is | seed 1: failures per million | seed 2: failures per million | cal failures (s1, s2) | not-trained check at 2e7: ratio (s1, s2) |
|---|---|---|---|---|---|
| B-JT | joint score + translate loss (deployed design) | 7,044 | 6,864 | 1,333, 1,325 | 0.00967, 0.00949 |
| B-J | joint score, no translate loss | 7,046 | 7,110 | 1,360, 1,370 | 0.0114, 0.0109 |
| B-P | pooled independent outputs, no translate loss | 34,048 | 35,360 | 6,769, 6,923 | 0.0674, 0.0767 |
| B-PA | pooled independent outputs + translate loss | 28,146 | 26,216 | 5,692, 5,230 | 0.0528, 0.0509 |

Reference r12, the registered BB72 network: 7,126 failures per million. Relay-BP alone: 5,382.

### 2.2 Failures at each routed fraction, against the fallback alone

| network (seed) | q = 0 % | q = 0.25 % | q = 0.5 % | q = 1 % | q = 2 % | q = 3 % | q = 5 % |
|---|---|---|---|---|---|---|---|
| B-JT s1 | 7,044 | 5,804 | 5,048 | 4,442 | 4,268 | 4,432 | 4,750 |
| B-JT s2 | 6,864 | 5,688 | 4,950 | 4,374 | 4,306 | 4,496 | 4,780 |
| B-J s1 | 7,046 | 5,880 | 5,096 | 4,408 | 4,310 | 4,430 | 4,754 |
| B-J s2 | 7,110 | 6,008 | 5,144 | 4,370 | 4,262 | 4,476 | 4,766 |
| B-P s1 | 34,048 | 32,082 | 30,056 | 25,876 | 18,524 | 12,088 | 6,210 |
| B-P s2 | 35,360 | 33,162 | 31,086 | 27,130 | 19,644 | 13,154 | 6,196 |
| B-PA s1 | 28,146 | 26,058 | 24,014 | 20,256 | 12,816 | 7,488 | 4,736 |
| B-PA s2 | 26,216 | 24,268 | 22,234 | 18,418 | 11,046 | 6,378 | 4,742 |
| reference r12 | 7,126 | 6,042 | 5,186 | 4,404 | 4,336 | 4,504 | 4,774 |
| fallback alone | 5,382 | 5,382 | 5,382 | 5,382 | 5,382 | 5,382 | 5,382 |

Shots the reference r12 routed to Relay-BP at each q: 1,224 at q = 0.25 %, 2,488 at q = 0.5 %, 5,209 at q = 1 %, 10,247 at q = 2 %, 15,176 at q = 3 %, 25,245 at q = 5 %.

The joint heads' pairs bottom out at the registered q = 2 %: 4,262 to 4,310 failures per million, about 20 % below
Relay-BP alone. At q = 5 % the pooled heads come closer: B-PA reaches 4,736 and 4,742, while B-P stays above Relay-BP alone at
6,196 and 6,210.

### 2.3 Paired tests, per seed

| id | X vs Y (held fixed) | q | seed | failures X, Y | only X / only Y | X - Y per million [95 %] | McNemar p |
|---|---|---|---|---|---|---|---|
| R1 | B-P vs B-J (no translate loss) | 0 % | 1 | 17,024, 3,523 | 13,747 / 246 | +27,002 [+26,547, +27,463] | < 1e-300 |
| R1 | B-P vs B-J (no translate loss) | 0 % | 2 | 17,680, 3,555 | 14,330 / 205 | +28,250 [+27,787, +28,719] | < 1e-300 |
| R1 | B-P vs B-J (no translate loss) | 2 % | 1 | 9,262, 2,155 | 7,434 / 327 | +14,214 [+13,874, +14,560] | < 1e-300 |
| R1 | B-P vs B-J (no translate loss) | 2 % | 2 | 9,822, 2,131 | 7,978 / 287 | +15,382 [+15,031, +15,739] | < 1e-300 |
| R2 | B-PA vs B-JT (translate loss) | 0 % | 1 | 14,073, 3,522 | 10,762 / 211 | +21,102 [+20,698, +21,511] | < 1e-300 |
| R2 | B-PA vs B-JT (translate loss) | 0 % | 2 | 13,108, 3,432 | 9,868 / 192 | +19,352 [+18,965, +19,744] | < 1e-300 |
| R2 | B-PA vs B-JT (translate loss) | 2 % | 1 | 6,408, 2,134 | 4,590 / 316 | +8,548 [+8,277, +8,824] | < 1e-300 |
| R2 | B-PA vs B-JT (translate loss) | 2 % | 2 | 5,523, 2,153 | 3,740 / 370 | +6,740 [+6,492, +6,993] | < 1e-300 |
| U1 | B-J vs B-JT (joint read-out) | 0 % | 1 | 3,523, 3,522 | 1,019 / 1,018 | +2 [-175, +179] | 1 |
| U1 | B-J vs B-JT (joint read-out) | 0 % | 2 | 3,555, 3,432 | 1,055 / 932 | +246 [+71, +421] | 0.006 |
| U1 | B-J vs B-JT (joint read-out) | 2 % | 1 | 2,155, 2,134 | 266 / 245 | +42 [-47, +131] | 0.376 |
| U1 | B-J vs B-JT (joint read-out) | 2 % | 2 | 2,131, 2,153 | 242 / 264 | -44 [-133, +44] | 0.351 |
| U2 | B-PA vs B-P (pooled read-out) | 0 % | 1 | 14,073, 17,024 | 3,820 / 6,771 | -5,902 [-6,306, -5,499] | 4.3e-183 |
| U2 | B-PA vs B-P (pooled read-out) | 0 % | 2 | 13,108, 17,680 | 2,942 / 7,514 | -9,144 [-9,545, -8,745] | < 1e-300 |
| U2 | B-PA vs B-P (pooled read-out) | 2 % | 1 | 6,408, 9,262 | 3,188 / 6,042 | -5,708 [-6,085, -5,333] | 5.8e-197 |
| U2 | B-PA vs B-P (pooled read-out) | 2 % | 2 | 5,523, 9,822 | 2,360 / 6,659 | -8,598 [-8,971, -8,228] | < 1e-300 |
| W1 | B-P vs B-JT (neither (credited to neither)) | 0 % | 1 | 17,024, 3,522 | 13,719 / 217 | +27,004 [+26,550, +27,464] | < 1e-300 |
| W1 | B-P vs B-JT (neither (credited to neither)) | 0 % | 2 | 17,680, 3,432 | 14,415 / 167 | +28,496 [+28,032, +28,966] | < 1e-300 |
| W1 | B-P vs B-JT (neither (credited to neither)) | 2 % | 1 | 9,262, 2,134 | 7,442 / 314 | +14,256 [+13,916, +14,602] | < 1e-300 |
| W1 | B-P vs B-JT (neither (credited to neither)) | 2 % | 2 | 9,822, 2,153 | 7,965 / 296 | +15,338 [+14,987, +15,695] | < 1e-300 |

### 2.4 Intersection-union and Holm verdicts

Holm runs over all 10 BB72 tests.

| id | factor | X vs Y | q | p_iut | p_holm | seed mean X - Y per million [95 %, shot-clustered] | seed range | verdict |
|---|---|---|---|---|---|---|---|---|
| R1 | readout | B-P vs B-J | 0 % | < 1e-300 | < 1e-300 | +27,626 [+27,223, +28,029] | +27,002 to +28,250 | worse |
| R1 | readout | B-P vs B-J | 2 % | < 1e-300 | < 1e-300 | +14,798 [+14,518, +15,078] | +14,214 to +15,382 | worse |
| R2 | readout | B-PA vs B-JT | 0 % | < 1e-300 | < 1e-300 | +20,227 [+19,878, +20,576] | +19,352 to +21,102 | worse |
| R2 | readout | B-PA vs B-JT | 2 % | < 1e-300 | < 1e-300 | +7,644 [+7,436, +7,852] | +6,740 to +8,548 | worse |
| U1 | supervision | B-J vs B-JT | 0 % | 1 | 1 | +124 [-6, +254] | +2 to +246 | inconclusive |
| U1 | supervision | B-J vs B-JT | 2 % | 1 | 1 | -1 [-65, +63] | -44 to +42 | inconclusive |
| U2 | supervision | B-PA vs B-P | 0 % | 4.3e-183 | 1.3e-182 | -7,523 [-7,818, -7,228] | -9,144 to -5,902 | better |
| U2 | supervision | B-PA vs B-P | 2 % | 5.8e-197 | 2.3e-196 | -7,153 [-7,426, -6,880] | -8,598 to -5,708 | better |
| W1 | whole design | B-P vs B-JT | 0 % | < 1e-300 | < 1e-300 | +27,750 [+27,346, +28,154] | +27,004 to +28,496 | worse |
| W1 | whole design | B-P vs B-JT | 2 % | < 1e-300 | < 1e-300 | +14,797 [+14,517, +15,077] | +14,256 to +15,338 | worse |

### 2.5 The supervision-matched read-out contrasts

- R1, B-P against B-J (no translate loss): worse. +27,002 and +28,250 per million at q = 0 (the pooled head fails 4.8
  and 5.0 times as often), +14,214 and +15,382 at q = 2 %.
- R2, B-PA against B-JT (translate loss): worse. +21,102 and +19,352 per million at q = 0 (4.0 and 3.8 times), +8,548
  and +6,740 at q = 2 % (3.0 and 2.6 times).

The supervision contrasts at matched read-out: U1, B-J against B-JT, is inconclusive (+2 and +246 per million at q = 0
with p = 1 and 0.006; +42 and -44 at q = 2 %). U2, B-PA against B-P, is better in both seeds (-5,902 and -9,144 per
million at q = 0; -5,708 and -8,598 at q = 2 %).

### 2.6 Sizes of the decision paths

| arm | parameters | read-out multiply-adds | vs deployed |
|---|---|---|---|
| B-JT | 483,357 | 45,524,736 | 1.000 |
| B-J | 483,357 | 45,524,736 | 1.000 |
| B-P | 997,905 | 43,573,824 | 0.957 |
| B-PA | 997,905 | 43,573,824 | 0.957 |

The pooled heads use 96 % of the joint read-out's multiply-adds and have about twice its parameters. B-PA's
training-only translate read-out (17 tensors) is not in its decision path.

## 3. What this means for the paper

The plan fixed in advance how each outcome would be reported (`REVISION_PLAN.md` 2.1, risk R1 and decision D11):
1. The BB72 joint translate score supports the read-out claim. It beats a pooled head of matched size at both supervision
   settings, in both seeds, by large factors (section 2.5). This is the case R1 names as the basis of the contribution
   claim, and Sec. VI-G can state it with the intervals of section 2.
2. Surface, d = 9: the vote's advantage holds when the heads are trained without the line loss (R1, R2) and is not
   established under the deployed supervision (R3). Following R1, the paper gives the intervals and says that the
   vote's edge over a pooled head is not established for the deployed recipe at d = 9. It can add that without the
   auxiliary loss the vote beat both simpler heads in both seeds, and it has to say that seed 1's pooled head never
   trained, so seed 2 gives the size of R1's effect (+2,869 per million at q = 0). The time each head saves needs
   A3's timings.
3. The data do not support naming the line supervision as an active ingredient at d = 9: S-PA did not beat S-P (U2),
   and the line loss made the vote worse (U1: 28-43 % more failures). Sec. IV-B and IV-C, which
   describe the read-out with its ablation, report this.
4. D11: no d = 13 runs. They were tied to S-VA beating S-PA at d = 9 (R3 is inconclusive), and S-V's d = 13 runs to
   the line loss helping the vote (U1 shows the opposite).
5. r764 (12,293 per million) sits between S-V and S-VA. It was grown from a 2-block network rather than trained from
   scratch, so it stays a reference row. Whether the deployed surface networks would gain from training without the
   line loss is outside A1.
6. Two seeds bound the spread between training runs only by their range, as the plan says. R3 and W2 show that at d = 9
   this spread can be larger than the effect between heads.

## 4. Provenance

- Code: the freeze G `2e69738e`, whose deployed code is that of 65f6e75f (both have PINSUMS `0098f4b42c153b69...`).
  `a1/analyze.py` and `data/a1/analysis_{surface,bb72}.json` are unchanged since 673ed040, the commit stage 1 trained
  at; the configs' sha256 are `328e54c0b7a09154...` (surface) and `652dfe390b6e3ce5...` (BB72).
- Jobs on Alpine, each checked against the pinned deployment of G before it ran: a barrier (07:40); the draw (1 min
  21 s, 190 chunks, none kept or recovered); two surface read jobs (H200 `2g.35gb`, 20 min 52 s each, 14 reads each);
  the BB72 read job (RTX PRO 6000 `1g.24gb`, 9 min 35 s, 18 reads); correlated PM (50 s); Relay-BP (18 min 21 s); the
  analyses (surface, 34 s; BB72, 43 s). Submitted at 07:41:51, both analyses done by 08:06.
- Sets: the 190 chunks' rows, each with FREEZE = G, are in `ising_compare/data/a1/sets_manifest.csv`; the four set
  digests are given in sections 1 and 2.
- Results: `results/{surface,bb72}/` holds `summary.md`, `standalone.csv`, `pair_curves.csv`,
  `comparisons.csv`, `references.csv`, `ramp.csv` (surface), `train_checks.csv`, `sizes.csv`, `pair_curves.png`,
  `analysis.json`, `reads_meta.json` and `checks.json`, sha256-checked against Alpine.

## 5. Departures and what is still missing

- Devices: the surface reads ran on H200 `2g.35gb` slices and the BB72 reads on RTX PRO 6000 `1g.24gb` slices, by the
  plan's per-code rule at submission (`A1_READ_PLAN.md` section 9); each code was read on one model. The arms trained
  on H100 SXM pods (`A1_POD_RUN.md`).
- The BB72 fallback is A1's locked rendering of A6's policy (`A1_READ_PLAN.md` section 6). If A6 later fixes another
  policy, a comparison under it is a separate, labelled secondary analysis; these verdicts stand.
- The time axis: `a1.analyze --timing` needs A3's network times. Running `analysis_main.txt` with `TIMING` adds the
  time curves without changing any failure count or verdict.
- Two seeds per arm, the plan's minimal set.
