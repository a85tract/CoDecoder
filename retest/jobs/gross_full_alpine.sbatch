#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 64 --mem=128G -t 06:00:00 -J rt_g4full
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# Fallback for gross's full-DEM Relay-BP read (RETEST.md section 3.1) if Derecho cannot run it: the same command as
# gross_full_derecho.pbs on an Alpine CPU node, CHUNKS of the 15 (resumable: a resubmission keeps the decoded shots;
# never two jobs on one chunk range at once). With OPP=1 first sector Relay-BP and BP-OSD on all 15 chunks.
set -euo pipefail
source ${T:-<cluster-scratch>/retest/code/freeze}/retest/jobs/bb_env.sh
gross
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 RAYON_NUM_THREADS=1
if [ "${OPP:-0}" = 1 ] && [ ! -f $OUT/preds/bb144_p0.004_test40014_bposd.npz ]; then
  $PYC $S bb.opp --code bb144 --p 0.004 --split test4 --chunks 0-14 --dec relay,bposd --procs $SLURM_CPUS_PER_TASK \
    --out $OUT/rows/opp_gross_sector_bposd.csv | grep -v "^relay \|^bposd "
fi
$PYC $S bb.stage4_gross fullread --split test4 --chunks $CHUNKS --procs $SLURM_CPUS_PER_TASK --out $OUT/rows/opp_gross_full_$CHUNKS.csv
echo FULLREAD_END $CHUNKS
