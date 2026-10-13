#!/bin/bash
#SBATCH -t 00:30:00
#SBATCH -J r17_cpu
#SBATCH -o <cluster-scratch>/ising_r17/logs/%x_%j.out
# R17's CPU term (b) (PLAN_R17.md section 6), study 4's procedure (speed2/s19/alpine/s4_cpu.sbatch of the frozen tree
# f1c6a0e9) on the q = 8 % mask, before any T6 read, on ONE exclusive Alpine amem node with an AMD EPYC 7713:
#   sbatch -p amem --qos=mem-normal -N 1 --exclusive --mem=256G --constraint=epyc-7713 r17/alpine/r17_cpu.sbatch
# DRY=1 (the dry run): any allocation (e.g. -p acpu --qos=cpu-normal -n 1 -c 2 --mem=16G), three repetitions as in
# the run (r17.py analyze needs three), flagged dry; its numbers count for nothing.
# The node checks are s4_cpu.sbatch's (amem, one node, exclusive, 256G) plus the CPU model. First r17.py pins and the
# q = 8 % mask from the development record's c. Then three interleaved repetitions, each a fresh single-threaded
# process pinned to core CORE (default 8): the node's plain and correlated PM per block on V17b chunk 0 (the frozen
# speed2.s11.cpu12 ref, unchanged), then the term (r17.py cputerm: cpu12's loop on the routed V17b shots, every
# decision against the development record) and, beside it, the same term at the registered tau (reported, not
# counted). r17.py analyze takes max(term) x 714.05 / median(correlated reference).
set -uo pipefail
B=<cluster-scratch>/ising_r17; R=<cluster-scratch>/ising
DRY=${DRY:-0}; REPS=3
PY="apptainer exec -B <cluster-scratch> $R/pyfull.sif $R/venv_gpu/bin/python"
O=$B/out/cpu_$SLURM_JOB_ID; mkdir -p $O
export PYTHONPATH=.:../neural_emulation OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
       ICMP_NV=$R/ext/ising_nv/code ICMP_SHOTS=$R/shots
FIRST=$(awk '/Cpus_allowed_list/{print $2}' /proc/self/status | cut -d, -f1 | cut -d- -f1)
[ "$DRY" = 1 ] && CORE=${CORE:-$FIRST}; CORE=${CORE:-8}             # a dry run's allocation may lack core 8
if [ "$DRY" != 1 ]; then
  [ "${SLURM_JOB_PARTITION:-}" = amem ] && [ "${SLURM_JOB_NUM_NODES:-}" = 1 ] && [ "${SLURM_MEM_PER_NODE:-}" = 262144 ] &&
    scontrol show job $SLURM_JOB_ID | grep -q "OverSubscribe=NO" && grep -q "AMD EPYC 7713 64-Core" /proc/cpuinfo ||
    { echo "not one exclusive amem node with an EPYC 7713 and 256G: stop"; echo JOBDONE; exit 6; }
fi
cd $B/code/ising_compare || exit 1
{ date -Is; hostname; cat ../REVISION ising_compare/REVISION 2>/dev/null; cat $B/r17/REVISION; lscpu; free -g; uptime
  echo "core $CORE siblings $(cat /sys/devices/system/cpu/cpu$CORE/topology/thread_siblings_list)"
  echo "job $SLURM_JOB_ID nodes $SLURM_JOB_NODELIST cpus_on_node $SLURM_CPUS_ON_NODE dry $DRY"
  scontrol show job $SLURM_JOB_ID | grep -e OverSubscribe -e NumCPUs; ps -eo user --no-headers | sort | uniq -c | sort -rn | head; } \
  > $O/node.txt 2>&1
A="--freeze data/study4/freeze.json --record $B/r17/r17.json"
st() { local n=$1; shift; taskset -c $CORE "$@" > $O/$n.log 2>&1; local rc=$?; echo "== $n exit $rc $(date -Is)"; tail -1 $O/$n.log | cut -c1-300
       [ $rc = 0 ] || { echo "FAILED at $n (exit $rc)"; echo JOBDONE; exit $rc; }; }
st pins $PY $B/r17/r17.py pins $A --bundle $B/bundle --s4r3 $B/s4r3 --allow-missing --out $O/pins.json
st mask $PY $B/r17/r17.py mask $A --devrec $B/bundle/devrec/d17.npz --out $O/mask_q08.npz
for rep in $(seq 1 $REPS); do
  for kind in plain correlated; do
    st ref_d17_${kind}_rep$rep $PY -m speed2.s11.cpu12 ref --d 17 --kind $kind --rep $rep --core $CORE --out $O \
      --freeze data/study4/freeze.json
  done
  for w in q08 q10; do                 # q10: the registered routing, timed beside on the same node (reported only)
    st term_r17_${w}_rep$rep $PY $B/r17/r17.py cputerm $A --devrec $B/bundle/devrec/d17.npz --mask $O/mask_q08.npz \
      --which $w --rep $rep --core $CORE --out $O $([ "$DRY" = 1 ] && echo --dry)
  done
done
uptime >> $O/node.txt
$PY $B/r17/r17.py cpurecord --dir $O $([ "$DRY" = 1 ] && echo --dry)
sha256sum $O/cpu_term.json; date -Is; echo JOBDONE
