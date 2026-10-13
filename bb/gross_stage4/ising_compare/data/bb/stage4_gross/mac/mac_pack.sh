# Pack the published per-shot outputs (STAGE4_GROSS_PLAN.md 5) on the Mac mini from the analysis tree.
W=~/work/gross_stage4; C=$W/code_an
export BB_SHOTS=$W/shots BB_PREDS=$W/preds QND_RELAY=~/work/relay OMP_NUM_THREADS=1
rm -rf $W/pack && mkdir -p $W/pack && cd $W/pack
~/work/.venv/bin/python $C/ising_compare/bb/scripts/stage4_gross_pack.py $C $W/pack 2>&1 | tail -n 3
shasum -a 256 *
