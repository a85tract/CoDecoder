#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -J trel_run3
#SBATCH -o <cluster-scratch>/rel_train_smoke/logs/%x_%j.out
# Final record: the smoke test as documented, the released d7 / d9 calibration checks, both rules, growth, BB, frozen.
set -uo pipefail
B=<cluster-scratch>/rel_train_smoke; D=$B/run_${SLURM_JOB_ID}; mkdir -p $D
SIF=<cluster-scratch>/ising/pyfull.sif; V=<cluster-scratch>/release_r
PYR="apptainer exec -B <cluster-scratch> $SIF $V/venv/bin/python"; PYI="apptainer exec -B <cluster-scratch> $SIF $V/venv_ising/bin/python"
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ISING_DECODING_DIR=<cluster-scratch>/ising/ext/ising_nv QND_RELAY=<cluster-scratch>/bb/relay
cp -r $B/code $D/repo && cd $D/repo && rm -rf runs
grep -m1 "model name" /proc/cpuinfo; nproc; date -Is
$PYR -c "import sys, numpy, stim, pymatching, torch; print(sys.version.split()[0], numpy.__version__, stim.__version__, pymatching.__version__, torch.__version__)"
echo "== A smoke: bash training/smoke.sh (defaults)"; t=$(date +%s)
PY="$PYR" timeout 1500 bash training/smoke.sh; echo "rc $? $(( $(date +%s) - t )) s"
echo "== A2 the budget rule on the smoke front end: front end charged 1 us, budget 1.04 us"; t=$(date +%s)
timeout 300 $PYR -m convlines_train.tune --frontend runs/smoke/train/final_ema.pt --shots 100000 --seed 2 --strong convlines_train.strong:pymatching --max-fraction 0.2 --step 0.01 --front-us 1 --rule budget --budget-us 1.04 --out runs/smoke_budget/tune.json | grep -E "calibration shots|^q =  [01]\.|^rule"; echo "rc $? $(( $(date +%s) - t )) s"
echo "== B the released d7 network on study 2's calibration set (records: 4,260 alone, 3,204 at q = 4 %, tau 1.3934324979782104)"; t=$(date +%s)
timeout 600 $PYR -m convlines_train.tune --frontend d7_b1w64_r740 --chunks 60070000:20 --strong convlines_train.strong:pymatching_correlated --grid 0,0.01,0.02,0.04,0.06,0.08,0.1 --rule paper --target-strong 0.9 --out runs/released/tune_d7.json; echo "rc $? $(( $(date +%s) - t )) s"
echo "== C the released d9 network by the paper's rule, M = 2,500 (records: 2,484 alone, q = 1 %: 2,188, tau 0.6820775270462036)"; t=$(date +%s)
timeout 900 $PYR -m convlines_train.tune --frontend d9_b3_r764 --chunks 60090000:20 --strong convlines_train.strong:pymatching_correlated --grid 0,0.01,0.02,0.04,0.06,0.08,0.1 --rule paper --target 2500 --out runs/released/tune_d9.json; echo "rc $? $(( $(date +%s) - t )) s"
echo "== D growth (surface d3 -> d5, BB 1 -> 2 blocks) and BB with BP-OSD"; t=$(date +%s)
$PYR -c "import stim; stim.Circuit.generated('surface_code:rotated_memory_x', distance=3, rounds=3, after_clifford_depolarization=0.003, before_round_data_depolarization=0.003, before_measure_flip_probability=0.003, after_reset_flip_probability=0.003).to_file('runs/d3.stim')"
timeout 300 $PYR -m convlines_train.train --circuit runs/d3.stim --out runs/g3 --samples 3e4 --eval-shots 2000 --eval-every 1e4 --workers 1 > runs/g3.log 2>&1; echo "rc $?"; tail -1 runs/g3.log
timeout 300 $PYR -m convlines_train.train --circuit runs/smoke/d5_p003.stim --out runs/g5 --init-from runs/g3/final_ema.pt --samples 2e4 --eval-shots 2000 --workers 1 > runs/g5.log 2>&1; echo "rc $?"; tail -1 runs/g5.log; grep -o '"init": {[^}]*}' runs/g5/run.json | cut -c1-300
C=$QND_RELAY/tests/testdata/bicycle_bivariate/circuit=bicycle_bivariate_72_12_6_memory_X,distance=6,rounds=6,error_rate=0.003,noise_model=uniform_circuit,basis=CX,A=x^3+y+y^2,B=y^3+x+x^2.stim
timeout 600 $PYI -m convlines_train.train --circuit "$C" --bb 6,6,3,1,2,3,1,2 --out runs/bb --samples 2e4 --batch 256 --width 32 --bott 8 --blocks 1 --eval-shots 1000 --eval-every 1e4 --workers 1 > runs/bb.log 2>&1; echo "rc $?"; tail -1 runs/bb.log
timeout 600 $PYI -m convlines_train.train --circuit "$C" --bb 6,6,3,1,2,3,1,2 --out runs/bb2 --init-from runs/bb/final_ema.pt --samples 1e4 --batch 256 --width 32 --bott 8 --blocks 2 --eval-shots 1000 --workers 1 > runs/bb2.log 2>&1; echo "rc $?"; tail -1 runs/bb2.log; grep -o '"step0[^}]*}' runs/bb2/run.json | cut -c1-200
timeout 600 $PYI -m convlines_train.tune --frontend runs/bb/final_ema.pt --shots 3000 --seed 3 --strong convlines_train.strong:bposd --max-fraction 0.05 --step 0.01 --rule budget --budget-us 1e9 --out runs/bb_tune.json | tail -3; echo "rc $? $(( $(date +%s) - t )) s"
echo "== E check_frozen (surface 7-17, bb)"; t=$(date +%s)
timeout 1200 $PYI -m convlines_train.check_frozen --surface 7,9,11,13,15,17 --bb > runs/check_frozen.log 2>&1; echo "rc $? $(( $(date +%s) - t )) s"; grep -E 'frozen trees' runs/check_frozen.log
date -Is; echo JOBDONE
