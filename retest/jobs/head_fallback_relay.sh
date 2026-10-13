# The re-test's BB72 fallback answers (A1's fallback_relay.txt through the shim): full-DEM Relay-BP (relay-bp 0.2.2,
# RelayDecoderF32 with qnd.teachers.CONFIGS['relay']) on every shot of the new a1test set, a NEW decoder per shot seeded with
# the shot's index in its set (a1.sets.index: 10,000 c + r for native chunk c, row r: the frozen seeds, unchanged by the
# shim), tasks of 2,000 shots over one spawned worker per core. A1: 0.1300 s per shot, 65,000 core-s = 18.1 core-hours for
# 500,000 shots (64 workers: 18.35 min); at 32 cores about 35-50 min. The answers depend only on the shot, so the core
# count is free. bb's venv (relay-bp lives there, not in the torch venv). A finished output is kept, never replaced.
#   sbatch -p acpu --qos=cpu-normal -c 32 --mem=48G -t 01:30:00 -J head_relay CMD=head_fallback_relay.sh
source $RT/retest/jobs/head_common.sh
set -x
PYR="apptainer exec -B <cluster-scratch-root> $I/pyfull.sif <cluster-scratch>/bb/venv/bin/python"
R=$OUT/reads/bb72
if [ -e $R/fallback_test.npz ]; then echo "KEPT $R/fallback_test.npz"
else $PYR $SHIM a1.fallback relay --code bb72 --kind a1test --chunks $CH_bb72 --out $R/fallback_test.npz --workers ${SLURM_CPUS_PER_TASK:-32}; fi
