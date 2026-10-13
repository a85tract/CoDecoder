# The re-test's test sets of the output-head comparison, drawn once after the beacon (RETEST.md section head): surface kind
# test (chunks 0-99, seeds of the set head-surface) and BB72 kind a1test (native chunks 0-49, seeds of head-bb72, beacon
# indices 20-69), 10,000 shots each, by A1's own `a1.sets draw` through the shim. No calibration shot is drawn: the
# original calibration chunks and reads are symlinked in (head_common.sh cal_links). $PIN is the freeze every row records
# (A1's draw_sets.txt requires FREEZE = PIN). The rows go to $OUT/sets/rows.csv (A1_SETS_ROWS). A chunk is published
# atomically and never redrawn; submitted again, the job keeps the chunks with rows (a1.sets). CPU only, about 1 minute.
#   sbatch -p acpu --qos=cpu-normal -c 4 --mem=16G -t 00:30:00 -J head_draw --export=ALL,... CMD=head_draw.sh jobs/head_job.sbatch
source $RT/retest/jobs/head_common.sh
set -x
cal_links
$PYS a1.sets draw --code surface --kind test --chunks $CH_surface --freeze $PIN --rows $OUT/sets/rows.csv
$PYS a1.sets draw --code bb72 --kind a1test --chunks $CH_bb72 --freeze $PIN --rows $OUT/sets/rows.csv
cal_check
