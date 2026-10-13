# The dry run's reads: d1_read_rt.sh at p = 0.002 and 0.003, 1 chunk each (RT_DRY_CHUNKS=1), the dry seeds. STEP of d1_gpu_job.sh.
# The GPU job is the real one (d1_R1.sbatch, overridden: -J, -t, --export STEP / RT / RT_OUT / RT_BEACON / RT_SEEDS / RT_DRY_CHUNKS).
for P in 0.002 0.003; do
  ( export P; source $RT/retest/jobs/d1_read_rt.sh ) || echo "DRY READ at $P FAILED: exit $?"
done
