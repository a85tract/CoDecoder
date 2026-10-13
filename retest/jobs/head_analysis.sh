# The re-test's analysis of the output-head comparison (A1's analysis_main.txt through the shim), once per code, CODE =
# surface or bb72: the barrier finds no read pending, the not-trained checks are gathered (A1's checkpoints, read only),
# a1.analyze with A1's frozen config data/a1/analysis_$CODE.json on $RT_OUT/head/reads/$CODE (the original calibration reads
# linked in + the new test reads + the fallback), no --timing; then every read's and the fallback's meta beside the
# analysis, and the thresholds' check: pair_curves.csv's tau column must equal the original analysis's, digit for digit
# (head/taus.csv, the 18 arm/seed networks x the q grid; the original calibration reads make them so). The output is
# $RT_OUT/head/analysis_$CODE. RT_DRY: the same commands with the config's test set shrunk to chunk 0 (the dry run only).
# CPU only.
#   sbatch -p acpu --qos=cpu-normal -c 4 --mem=32G -t 00:30:00 -J head_an_s CMD=head_analysis.sh,CODE=surface
source $RT/retest/jobs/head_common.sh
set -x
R=$OUT/reads/$CODE; OUTA=$OUT/analysis_$CODE; CFG=data/a1/analysis_$CODE.json
CK=$([ $CODE = surface ] && echo $ICMP_CKPT || echo $BB_CKPT)
if [ -n "${RT_DRY:-}" ]; then
  CFG=$OUT/analysis_${CODE}_dry_config.json
  $PY -c "import json, sys; c = json.load(open(sys.argv[1])); c['test']['chunks'] = '0-0'; json.dump(c, open(sys.argv[2], 'w'), indent=1)" data/a1/analysis_$CODE.json $CFG
fi
[ ! -e $OUTA/analysis.json ] || { echo "$OUTA/analysis.json exists: the analysis runs once"; exit 1; }
cal_check
$PYS a1.preflight --code $CODE --decisions $A/rule --ckpt-root $CK --reads $R --out $R/reads_final.json
$PY -c "import json, sys; j = json.load(open('$R/reads_final.json')); print('READS', len(j), 'pending before the analysis'); sys.exit(len(j) != 0)"
arms=$([ $CODE = surface ] && echo "S-VA S-V S-L S-P S-PA" || echo "B-JT B-J B-P B-PA")
runs=$(for arm in $arms; do for s in 1 2; do echo a1_${arm}_s$s
       if [ -d $CK/a1ramp_${arm}_s$s ]; then echo a1ramp_${arm}_s$s; fi
       done; done)
$PYS a1.check checks --code $CODE --runs $runs --out $R/checks.json
$PYS a1.analyze --config $CFG --reads $R --checks $R/checks.json --out $OUTA
$PY -c "import glob, json, os, numpy as np; m = {os.path.basename(f): json.loads(str(np.load(f)['meta'])) for f in sorted(glob.glob('$R/*.npz')) if not f.endswith(('.failed.npz', '.tmp.npz'))}; json.dump(m, open('$OUTA/reads_meta.json', 'w'), indent=1); print('META', len(m), 'files')"
$PY $H/check_taus.py $CODE $OUTA/pair_curves.csv $H/taus.csv $OUTA/taus_check.json
