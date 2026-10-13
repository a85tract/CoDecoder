#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 32 --mem=64G -t 03:00:00 -J rt_bbcpu
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# One CPU opponent read of the BB test sets (RETEST.md section 3.1): bb.opp (sector Relay-BP "relay", ldpc BP-OSD
# "bposd", full-DEM Relay-BP "relay_full"; relay-bp 0.2.2, ldpc 2.4.1, the frozen parameters of bb.opp / qnd.teachers),
# blind. Env: TREE=bb72|gross, CODE, P, SPLIT, CHUNKS, DEC (decoders joined by "+": sbatch --export splits on commas).
set -euo pipefail
source ${T:-<cluster-scratch>/retest/code/freeze}/retest/jobs/bb_env.sh
$TREE
$PYC $S bb.opp --code $CODE --p $P --split $SPLIT --chunks $CHUNKS --dec ${DEC//+/,} --procs $SLURM_CPUS_PER_TASK \
  --out $OUT/rows/opp_${CODE}_p${P}_${DEC//+/_}_${CHUNKS}.csv | grep -v "^relay\|^bposd" | cut -c1-200
echo "OPP_END $CODE $P $DEC $CHUNKS"
