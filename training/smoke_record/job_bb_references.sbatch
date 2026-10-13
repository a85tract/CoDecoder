#!/bin/bash
#SBATCH -p acpu
#SBATCH --qos=cpu-normal
#SBATCH -N 1
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 00:45:00
#SBATCH -J trel_run4
#SBATCH -o <cluster-scratch>/rel_train_smoke/logs/%x_%j.out
# BB references for training/TRAINING.md (released r11, r12 alone on fresh shots; gross loads) and run_stage.sh dry runs.
set -uo pipefail
B=<cluster-scratch>/rel_train_smoke; D=$B/run_${SLURM_JOB_ID}; mkdir -p $D
SIF=<cluster-scratch>/ising/pyfull.sif; V=<cluster-scratch>/release_r
PYI="apptainer exec -B <cluster-scratch> $SIF $V/venv_ising/bin/python"
export OMP_NUM_THREADS=16 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export ISING_DECODING_DIR=<cluster-scratch>/ising/ext/ising_nv QND_RELAY=<cluster-scratch>/bb/relay
cp -r $B/code $D/repo && cd $D/repo && rm -rf runs
grep -m1 "model name" /proc/cpuinfo; date -Is
echo "== F run_stage.sh dry runs (PY=echo)"
for s in d7b1w64_r740 g2_r710 g2_r711 g6_d9_b3t_r764 d13b3t_r732 d13b3t_r733 d11b3t_r735 g8_d11_b5t_r777d g7_d13_b5t_r774 g7_d13_b5t_r775 g9_d15_b5t_r778 g12_d15_b5t_r782 g14_d17_b5t_r787; do
  ICMP_CKPT=/CKPT ICMP_OUT=$D/out PY=echo bash training/surface/run_stage.sh $s 2>&1 | tail -1 | grep -oE -- "--run [^ ]+|--init(-transfer)? [^ ]+|--init-sha|--no-val" | tr '\n' ' '; echo
done
echo "== G released BB networks alone on fresh shots (bb.train checkpoints through convlines_train.frontend)"
for n in r11 r12; do t=$(date +%s)
  timeout 1200 $PYI -m convlines_train.tune --frontend bb72/weights/${n}_final_ema.pt --shots 100000 --seed 5 --strong convlines_train.strong:bposd --grid 0 --rule budget --budget-us 1e9 --out runs/released/tune_$n.json | head -1; echo "rc $? $(( $(date +%s) - t )) s"
done
t=$(date +%s)
timeout 900 $PYI -m convlines_train.tune --frontend gross/weights/g2_D12k113c10_final_ema.pt --shots 5000 --seed 5 --strong convlines_train.strong:bposd --grid 0 --rule budget --budget-us 1e9 --out runs/released/tune_g2.json | head -1; echo "rc $? $(( $(date +%s) - t )) s"
date -Is; echo JOBDONE
