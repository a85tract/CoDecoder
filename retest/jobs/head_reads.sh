# The re-test's reads of the output-head comparison (A1's reads_main.txt through the shim): the networks of $CODE (surface
# or bb72) that A1's continuation decision requires, on the NEW test set only, through a1.read (decision path only, fp32,
# TF32 off, batch 4,096, the first 2,000 shots again at batch one; a differing decision fails the read). The calibration
# reads are A1's own, symlinked into $R (head_common.sh cal_links, sha256-checked here): a1.preflight keeps them (same
# checkpoint sha256, arm, seed, kind, chunks, passed batch-one check) and plans the 14 surface / 9 BB72 test reads only; the
# shim refuses a job list with any other kind. PARTS=k PART=i: this job reads the networks at positions i, i+k, ... of the
# code's sorted list (as in A1); the barrier runs again after the reads and must find nothing pending.
#   sbatch -p ah200 --qos=gpu-normal --gres=gpu:h200_2g.35gb:1 -c 8 --mem=48G -t 01:00:00 -J head_reads_s0 CMD=head_reads.sh,CODE=surface,PARTS=2,PART=0
#   sbatch -p artxpro6000 --qos=gpu-normal --gres=gpu:rtx_pro_6000_1g.24gb:1 -c 8 --mem=48G -t 00:45:00 -J head_reads_b CMD=head_reads.sh,CODE=bb72
source $RT/retest/jobs/head_common.sh
set -x
R=$OUT/reads/$CODE
cal_check
net_check
C=$([ $CODE = surface ] && echo $ICMP_CKPT || echo $BB_CKPT)
P=${PARTS:+_p$PART}
# part(FILE): keep in FILE only this part's networks (a network = an output name without _cal / _test); A1's function
part() { [ -z "${PARTS:-}" ] || $PY -c "import json, os, sys; f, i, k = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); rec = json.load(open(f + '.preflight.json')); key = lambda o: os.path.basename(o).rsplit('_', 1)[0]; nets = sorted({key(o) for o in rec['pending'] + rec['kept']}); mine = set(nets[i::k]); jobs = [j for j in json.load(open(f)) if key(j['out']) in mine]; json.dump(jobs, open(f, 'w'), indent=1); print('PART', i, 'of', k, sorted(mine), len(jobs), 'reads pending')" $1 $PART $PARTS; }
$PYS a1.preflight --code $CODE --decisions $A/rule --ckpt-root $C --reads $R --out $R/reads$P.json
part $R/reads$P.json
set +x; echo "+ $(date +%T) a1.read --code $CODE --many $R/reads$P.json"
$PYS a1.read --code $CODE --many $R/reads$P.json 2>&1 | while IFS= read -r l; do echo "$(date +%T) $l"; done
set -x
$PYS a1.preflight --code $CODE --decisions $A/rule --ckpt-root $C --reads $R --out $R/reads_again$P.json
part $R/reads_again$P.json
$PY -c "import json, sys; j = json.load(open('$R/reads_again$P.json')); print('READS', len(j), 'pending after the reads'); sys.exit(len(j) != 0)"
