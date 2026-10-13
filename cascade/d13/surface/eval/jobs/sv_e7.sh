# Added in the job's command file, not in the pinned code: nvidia-smi samples this card's SM clock every 100 ms for
# the whole job (phase C's gpu_watch rule: a timing is accepted when its lowest sample is >= 97 % of clocks.max.sm and
# no other process is on the card), the active clock-event reasons, and the compute processes on the card every 2 s.
CK=$R/clocks; mkdir -p $CK; CKF=$CK/${SLURM_JOB_NAME}_${SLURM_JOB_ID}
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.max.sm,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 100 > ${CKF}_clk.csv 2>&1 & CW1=$!
nvidia-smi --query-gpu=timestamp,clocks_event_reasons.active --format=csv,noheader -lms 100 > ${CKF}_why.csv 2>&1 & CW2=$!
( while sleep 2; do echo "$(date '+%Y/%m/%d %H:%M:%S.%3N') $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr '\n' ' ')"; done ) > ${CKF}_apps.txt 2>&1 & CW3=$!
source $T/surface/eval/jobs/e7_pipes.sh
kill $CW1 $CW2 $CW3 2>/dev/null || true
