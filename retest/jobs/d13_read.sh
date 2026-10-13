# Sourced by d13_e5.sbatch and d13_e6.sbatch (READ=e5 | e6) inside the Slurm job: the evaluation's e5 / e6 command
# files (cascade_w5/code.20261012_060258/surface/eval/jobs/e5_conf_batched.sh, e6_conf_single.sh) with the same
# sv_read arguments, run through retest/shim/rt_surface.py and written to a new --out. e5 and e6 write the same
# directory, as the originals did ($E/conf): confirm needs all nine computations in one directory, and the observable
# files are published once and compared by the second reader. Variables as in the originals: PS (e5), P and CH (e6),
# RESUME=1. RT_DRY_CHUNKS=a-b (dry runs only) replaces every p's chunk range; it is refused unless the seeds in use
# are the dry ones, and dry seeds are refused without it. RT, RT_BEACON, RT_SEEDS, RT_OUT must be set.
source ${RT:?RT: the deployed freeze tree (contains retest/)}/retest/jobs/d13_env.sh
: "${RT_BEACON:?}" "${RT_SEEDS:?}" "${RT_OUT:?}" "${READ:?e5 or e6}"
PY="apptainer exec --nv -B <cluster-scratch-root> $I/pyfull.sif $V/bin/python"
if $PY $SHIM --check-dry > /dev/null 2>&1; then DRYSEEDS=1; else DRYSEEDS=0; fi
if [ -n "${RT_DRY_CHUNKS:-}" ] && [ $DRYSEEDS = 0 ]; then echo "RT_DRY_CHUNKS is refused: the seeds in use are not the dry ones"; exit 1; fi
if [ -z "${RT_DRY_CHUNKS:-}" ] && [ $DRYSEEDS = 1 ]; then echo "the dry seeds are for dry chunk ranges only (RT_DRY_CHUNKS)"; exit 1; fi
source $T/surface/eval/jobs/common.sh       # E=$R/eval, LOCK, GATE, the staged venv (PY), run, lockchunks
OUT=$RT_OUT/d13/conf
echo "sha256 (shim, seeds, tags, beacon, seeds file, lock, gate, receipt):"
sha256sum $SHIM $RTD/seeds.py $RTD/tags.json $RT_BEACON $RT_SEEDS $LOCK $GATE $E/lock_commit.json | awk '{print substr($1, 1, 16), $2}'
echo "read $READ -> $OUT (dry seeds: $DRYSEEDS)"
case $READ in
  e5)   # batched: SB, Bb, BQ1, BQ5, MB, MP, each p in one process (eval/jobs/e5_conf_batched.sh)
    for p in ${PS:-0.006 0.0035}; do
      run $PY $SHIM surface.sv_read --stage conf --p $p --chunks ${RT_DRY_CHUNKS:-$(lockchunks $p)} \
          --comp SB,Bb,BQ1,BQ5,MB,MP --lock $LOCK --gate $GATE ${RESUME:+--resume} --out $OUT
    done ;;
  e6)   # single: S1, B1, MS (eval/jobs/e6_conf_single.sh)
    if [ -n "${P:-}" ]; then LIST="$P:${RT_DRY_CHUNKS:-$CH}"
    else LIST="0.006:${RT_DRY_CHUNKS:-$(lockchunks 0.006)} 0.0035:${RT_DRY_CHUNKS:-$(lockchunks 0.0035)}"; fi
    for x in $LIST; do
      run $PY $SHIM surface.sv_read --stage conf --p ${x%%:*} --chunks ${x#*:} --comp S1,B1,MS --lock $LOCK --gate $GATE \
          ${RESUME:+--resume} --out $OUT
    done ;;
  *) echo "READ must be e5 or e6"; exit 1 ;;
esac
