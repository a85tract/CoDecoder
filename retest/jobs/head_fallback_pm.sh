# The re-test's surface fallback answers (A1's fallback_pm.txt through the shim): correlated PyMatching on every test shot
# of the new surface test set, once, one worker per core (1,000,000 shots: about 20 s on 32 cores, 0.44 core-hours). A
# finished output is kept, never replaced. CPU only.
#   sbatch -p acpu --qos=cpu-normal -c 32 --mem=64G -t 00:30:00 -J head_pm CMD=head_fallback_pm.sh
source $RT/retest/jobs/head_common.sh
set -x
R=$OUT/reads/surface
if [ -e $R/fallback_test.npz ]; then echo "KEPT $R/fallback_test.npz"
else $PYS a1.fallback pm --kind test --chunks $CH_surface --out $R/fallback_test.npz --workers ${SLURM_CPUS_PER_TASK:-32}; fi
