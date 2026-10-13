#!/bin/bash
# Timing stage: run only with an otherwise idle GPU.
cd /workspace/repo && . .venv/bin/activate
mkdir -p logs timing_out cpu_out cpu_out_pub out
# the session's core: on the GPU's NUMA node (node 1: 56-111 + siblings 168-223), outside core 0's physical core,
# the physical core (with its SMT sibling) least busy over 1 s
CORE=$(python - <<'PY'
import time
def busy():
    d={}
    for l in open('/proc/stat'):
        if l.startswith('cpu') and l[3].isdigit():
            f=l.split(); n=int(f[0][3:]); v=list(map(int,f[1:])); d[n]=(sum(v)-v[3]-v[4], sum(v))
    return d
a=busy(); time.sleep(1.0); b=busy()
best=None
for c in range(56,112):
    s=[c,c+112]
    fr=sum((b[x][0]-a[x][0]) for x in s)/max(1,sum((b[x][1]-a[x][1]) for x in s))
    if best is None or fr<best[0]: best=(fr,c)
print(best[1])
PY
)
echo "core $CORE" | tee logs/stage3_core.txt
# gate-like measurements, for the record: the pinned core's busy/steal over 5 s; plain PM per block (3 at d = 9, 2 at d = 13)
python - $CORE <<'PY' | tee out/gate_core.json
import sys,time,json
c=int(sys.argv[1])
def rd():
    for l in open('/proc/stat'):
        if l.startswith(f'cpu{c} '): v=list(map(int,l.split()[1:])); return v
a=rd(); time.sleep(5); b=rd(); d=[y-x for x,y in zip(a,b)]; tot=sum(d)
print(json.dumps(dict(core=c, busy=1-(d[3]+d[4])/tot, steal=d[7]/tot, seconds=5)))
PY
for r in 1 2 3; do python -m timing.cpu_term ref --plain --d 9 --val shots/val --rep $r --core $CORE --out cpu_out_plain; done > logs/gate_plain_d9.log 2>&1
for r in 1 2; do python -m timing.cpu_term ref --plain --d 13 --val shots/val --rep $r --core $CORE --out cpu_out_plain; done > logs/gate_plain_d13.log 2>&1
nvidia-smi --query-gpu=memory.used,clocks.sm,clocks.max.sm --format=csv > logs/gpu_before_timing.txt; nvidia-smi --query-compute-apps=pid,name --format=csv >> logs/gpu_before_timing.txt
# (a) the GPU term: five fresh processes per network, interleaved across networks
for p in 0 1 2 3 4; do for d in 7 9 11 13 15; do
  python -m timing.gpu_term --d $d --val shots/val --proc $p --core $CORE --deployed masks/val_d$d.npz --build build/d7 --out timing_out >> logs/gpu_term.log 2>&1
done; done
# (c) the handoff: d = 7 five processes; d = 9-15 one router process each
for p in 0 1 2 3 4; do python -m timing.gpu_term --d 7 --val shots/val --proc $p --core $CORE --handoff --build build/d7 --out timing_out >> logs/handoff.log 2>&1; done
for d in 9 11 13 15; do python -m timing.gpu_term --d $d --val shots/val --proc 0 --core $CORE --handoff --build build/d7 --deployed masks/val_d$d.npz --out timing_out >> logs/handoff.log 2>&1; done
# (b) the CPU term: three interleaved fresh-process repetitions of term and ref per network, then convert
for r in 1 2 3; do for d in 7 9 11 13 15; do
  python -m timing.cpu_term term --d $d --val shots/val --mask masks/val_d$d.npz --rep $r --core $CORE --out cpu_out >> logs/cpu_term.log 2>&1
  python -m timing.cpu_term ref --d $d --val shots/val --rep $r --core $CORE --out cpu_out >> logs/cpu_term.log 2>&1
done; done
for d in 7 9 11 13 15; do python -m timing.cpu_term convert --d $d --out cpu_out >> logs/cpu_convert.log 2>&1; done
# the studies' own d = 15 mask (the regenerated d = 15 shots are the studies')
python -m timing.cpu_term term --d 15 --val shots/val --mask routing/val_d15.npz --rep 1 --core $CORE --out cpu_out_pub >> logs/cpu_term_pub.log 2>&1
echo STAGE3 DONE
