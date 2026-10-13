# The one analysis (STAGE4_GROSS_PLAN.md 5), run once on the Mac mini from the records commit's tree.
W=~/work/gross_stage4; C=$W/code_an
export PYTHONPATH=$C/ising_compare:$C/qldpc_neural BB_SHOTS=$W/shots BB_PREDS=$W/preds QND_RELAY=~/work/relay
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd $C/ising_compare
echo "== analysis $(date '+%F %T %Z') on $(uname -sm), tree $(cat $C/COMMIT)"
~/work/.venv/bin/python -m bb.stage4_gross analyze --out data/bb/stage4_gross 2>&1
echo "== exit $? $(date +%T)"
