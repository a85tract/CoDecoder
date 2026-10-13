#!/bin/bash
#SBATCH -p acpu --qos=cpu-normal --nice=1000 -N 1 -n 1 -c 4 --mem=8G -t 00:20:00 -J rt_bbseeds
#SBATCH -o <cluster-scratch>/retest/logs/%x_%j.out
# The BB shim's patch-on seed checks alone (bb_replay.sbatch's last step), with the seeds of SEEDS_DIR.
set -euo pipefail
T=${T:-<cluster-scratch>/retest/code/dry}; R=${R:-<cluster-scratch>/retest/replay/bb}
SEEDS_DIR=${SEEDS_DIR:-<cluster-scratch>/retest/dry}; PY=<cluster-scratch>/retest/bin/pycpu
export QND_RELAY=<cluster-scratch>/bb/relay PYTHONPATH=.:../qldpc_neural
export RT_BEACON=$SEEDS_DIR/beacon_dry.json RT_SEEDS=$SEEDS_DIR/seeds_dry.json
cd $T/bb/bb72/ising_compare && $PY $T/retest/replay/bb_seedcheck.py $R/seedcheck_bb72.json
cd $T/bb/gross_stage4/ising_compare && $PY $T/retest/replay/bb_seedcheck.py $R/seedcheck_gross.json
