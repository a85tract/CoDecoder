# Sourced by the A1 args files (a1/jobs/job.sbatch has set A, PY and the environment). The arms' flags, the deployed
# recipe and helpers. Surface (arch.train): the deployed recipe of r764 (AdamW (0.9, 0.99), wd 0.01, 1,000 warm-up
# steps, cosine to 2 %, peak 1e-3, batch 1,024, clip 1, EMA 0.999, bf16, 3 blocks of width 128), from scratch, 8
# sampler workers, val monitoring every 2e7. BB72 (bb.train): r11 / r12's recipe (p 0.003, design_bb72.json, warm-up
# 500, monitor chunk 0, eval every 1e7, 6 workers).
declare -A SFLAGS=([S-VA]="--head vote --line-weight 1" [S-V]="--head vote --line-weight 0" [S-L]="--head line --line-weight 0"
                   [S-P]="--head pooled --line-weight 0" [S-PA]="--head pooled --line-weight 1")
declare -A BFLAGS=([B-JT]="--head joint --translate-weight 1" [B-J]="--head joint --translate-weight 0"
                   [B-P]="--head pooled --translate-weight 0" [B-PA]="--head pooled --translate-weight 1")
declare -A SIDX=([S-VA]=0 [S-V]=1 [S-L]=2 [S-P]=3 [S-PA]=4)
declare -A BIDX=([B-JT]=0 [B-J]=1 [B-P]=2 [B-PA]=3)
SURF="--d 9 --phase truth --lr 1e-3 --warmup 1000 --wd 0.01 --batch 1024 --ema 0.999 --blocks 3 --compile"
BBR="--code bb72 --p 0.003 --design data/bb/design_bb72.json --monitor 0 --compile"
sha() { sha256sum "$1" | cut -c1-64; }
trunk() { [ -f $A/trunk/trunk_$1_s$2.pt ] || $PY -m a1.trunk $1 --seed $2 --out $A/trunk; }
# The registered trunk files' sha256 (data/a1/trunks.txt: code seed sha256), checked by every run's --init-sha.
regsha() { awk -v c=$1 -v s=$2 '$1 == c && $2 == s {print $3}' data/a1/trunks.txt; }
# Minutes a training segment may run before it pauses: until the job's end less 10 minutes (bb.train --max-minutes;
# arch.train --max-hours with --max-hours-at-pool).
left_min() { echo $(( ($(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s) - $(date +%s)) / 60 - 10 )); }
