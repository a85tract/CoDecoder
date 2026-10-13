"""The GPU check of a SURF-v2 timing session, at its start and at its end (README section 4, rule 3's last row, which
study 4's session and this one keep), as the studies' host gate ran it (the research repository's
ising_compare/icmp/host_gate.py: gpu_check, judge_alone and its child process). A child process takes a CUDA context
and 1 GiB on the GPU, then runs 3 s of 2,048 x 2,048 fp16 matrix products (20 between synchronizations) while
nvidia-smi samples that GPU's clocks every 100 ms, on CPUs other than the pinned core. It passes when the telemetry is
usable (the sampler ran until stopped, the child exited 0, every row reads, at least 20 rows after the first two), the
median SM clock after the first two samples is at least 97 % of the maximum, every throttle reason seen is none or idle,
and the GPU is alone: nothing on it before and under 64 MiB used, nothing but the child while it runs, and nothing after
it exited. The child is the process line with its pid or, where a container reports the host's pids, the one line that
appeared with it, vanished after it and holds its 1 GiB.

E0 (12 October) found the earlier version failing a healthy H100: its 8,192 x 8,192 products held the card at the 700 W
power cap (1,590 of 1,980 MHz, throttle 0x4), where the gate's 2,048 x 2,048 products ran at 1,980 MHz, and the pod
listed the check's own context under the host's pid, which it took for another process. At the end of E0's third run
the check failed on the throttle criterion alone (its samples included a software power cap, 0x4; median 1,980 MHz).
Five checks one after another on the idle pod passed twice, then failed three times the same way, and a burst sampled
row by row showed the cap in four rows early in the burst, at 1,965 MHz and 310-429 W. The criterion is the registered
one (README section 4, rule 3's table) and stays; the record keeps every sampler row (SM MHz, maximum, reasons, W).
Since that pod failed 4 of its 9 checks, every timing pod is pre-screened before anything is read or timed on it: this
check five times on the idle GPU (surf_v2/args/pod_prescreen.txt, --when prescreen), all five passing, else the pod is
replaced (surf_v2/REGISTRATION.md, Timing).

    python -m surf_v2.gpucheck --when start|end|prescreen --out SESSION/gpucheck_start.json [--core C]"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
import time

ALLOC_MIB = 1024         # the allocation that marks the child
IDLE_MIB = 64            # an idle H100 reports 0 MiB used; any CUDA context holds hundreds
MIN_SAMPLES = 20         # clock readings after the ramp: the 3 s burst sampled every 100 ms gives about 30
THROTTLE_OK = {"0x0000000000000000", "0x0000000000000001"}           # none, idle
CHILD = r"""
import os, sys, time, torch
mark = torch.empty(int(sys.argv[2]) << 20, dtype=torch.uint8, device="cuda")   # the allocation that marks this process
a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
torch.cuda.synchronize()
print("GPU-" + str(torch.cuda.get_device_properties(0).uuid), os.getpid(), flush=True)
sys.stdin.readline()                                                             # the parent has listed this context
t0, n = time.time(), 0
while time.time() - t0 < float(sys.argv[1]):
    for _ in range(20):
        a = (a @ a).clamp_(-1, 1)
    torch.cuda.synchronize()
    n += 20
print(n, flush=True)
"""


def smi(*args):
    r = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return [[x.strip() for x in line.split(",")] for line in r.stdout.splitlines() if line.strip()]


def _num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except ValueError:
        return None


def compute_apps():
    """[(GPU uuid or None, pid or None, used MiB or None)] for every process line nvidia-smi prints; a line it cannot
    read stays in the list with None fields, so an unknown process never drops out of a count."""
    return [(f[0] if f[0].startswith(("GPU-", "MIG-")) else None, int(f[1]) if len(f) > 1 and f[1].isdigit() else None,
             _num(f[2]) if len(f) > 2 else None)
            for f in smi("--query-compute-apps=gpu_uuid,pid,used_memory") if not f[0].lower().startswith("no running")]


def gpu_mem():
    """{GPU uuid: used MiB}, device-wide (it counts processes no list shows)."""
    return {f[0]: _num(f[1]) if len(f) > 1 else None for f in smi("--query-gpu=uuid,memory.used")}


def judge_alone(gpu, child, alloc_mib, s0, mem0, s1, s2):
    """The verdict from the lists before the child has a CUDA context (s0, and the used memory mem0), while it holds one
    (s1) and after it exited (s2). Only lines on the selected GPU count, and a line whose GPU is unknown counts as on
    it. The child is the s1 line with its pid, else the one s1 line with a pid in neither s0 nor s2 that holds at least
    its allocation; no other line is ever excluded."""
    on = lambda apps: [x for x in apps if x[0] in (gpu, None)]                         # noqa: E731
    b, d, e = on(s0), on(s1), on(s2)
    own, how = [x for x in d if x[1] == child][:1], "pid"
    if not own:
        seen = {x[1] for x in b + e}
        own = [x for x in d if x[1] is not None and x[1] not in seen and (x[2] or 0) >= alloc_mib]
        own, how = (own, "lifecycle and memory") if len(own) == 1 else ([], "not identified")
    others = [x for x in d if x not in own]
    mem = mem0.get(gpu)
    ok = bool(own) and not b and not others and not e and mem is not None and mem < IDLE_MIB
    return ok, dict(selected=gpu, child_pid=child, identified_by=how, own=own, before=b, during_others=others, after=e,
                    mem_before_mib=mem)


def clock_rows(text):
    """The sampler's rows as (SM MHz, max SM MHz, throttle reasons, power W); None for a row that does not read as two
    positive finite clocks and a hexadecimal reason word."""
    out = []
    for line in text.splitlines():
        f = [x.strip() for x in line.split(",")]
        try:
            sm, mx = float(f[0]), float(f[1])
            ok = len(f) == 4 and all(math.isfinite(x) and x > 0 for x in (sm, mx)) and re.fullmatch(r"0x[0-9a-fA-F]+",
                                                                                                    f[2])
        except (ValueError, IndexError):
            ok = False
        out.append((sm, mx, f[2], _num(f[3])) if ok else None)
    return out


def telemetry_error(rows, sampler_rc, child_rc):
    if sampler_rc is not None:
        return f"the clock sampler exited by itself (exit {sampler_rc})"
    if child_rc != 0:
        return f"the burst's child process exited {child_rc}"
    if any(r is None for r in rows):
        return f"{sum(r is None for r in rows)} of {len(rows)} sampler rows unreadable"
    if len(rows) - 2 < MIN_SAMPLES:
        return f"{len(rows)} sampler rows, fewer than {MIN_SAMPLES} after the ramp"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--when", required=True, choices=("start", "end", "prescreen"))
    ap.add_argument("--core", type=int, default=-1)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    allowed = os.sched_getaffinity(0)
    core = a.core if a.core >= 0 else max(allowed)
    others = (allowed - {core}) or allowed
    os.sched_setaffinity(0, {core})                        # the child runs on the pinned core, the sampler elsewhere
    s0, mem0 = compute_apps(), gpu_mem()                   # before the child has a CUDA context
    p = subprocess.Popen([sys.executable, "-c", CHILD, str(a.seconds), str(ALLOC_MIB)], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, text=True)
    smp, rec = None, {}
    try:
        gpu, child = p.stdout.readline().split()
        s1 = compute_apps()
        q = "clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,power.draw"
        smp = subprocess.Popen(["nvidia-smi", "-i", gpu, f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms",
                                "100"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                               preexec_fn=lambda: os.sched_setaffinity(0, others))
        p.stdin.write("\n")
        p.stdin.flush()
        matmuls = p.stdout.readline().strip()
        child_rc = p.wait(timeout=120)
        sampler_rc = smp.poll()                            # still running: it sampled until stopped
        smp.terminate()
        rows = clock_rows(smp.communicate(timeout=30)[0])
        for _ in range(20):                                # up to 10 s for the driver to let the child's context go
            s2 = compute_apps()
            if not [x for x in s2 if x[0] in (gpu, None)]:
                break
            time.sleep(0.5)
        alone_ok, alone = judge_alone(gpu, int(child), ALLOC_MIB, s0, mem0, s1, s2)
        good = [r for r in rows if r]
        sm = sorted(r[0] for r in good[2:])                # the first samples precede the ramp
        err = telemetry_error(rows, sampler_rc, child_rc)
        rec = dict(gpu_uuid=gpu, samples=len(rows), sm_median=sm[len(sm) // 2] if sm else None,
                   sm_max_clock=max((r[1] for r in good), default=None), sm_min=sm[0] if sm else None,
                   throttle_seen=sorted({r[2] for r in good}), power_max=max((r[3] or 0 for r in good), default=None),
                   matmuls=int(matmuls) if matmuls.isdigit() else None, child_rc=child_rc, sampler_rc=sampler_rc,
                   telemetry_error=err, alone=alone, rows=rows)
    except Exception as e:                                 # telemetry unavailable or no child: every criterion fails
        p.kill()
        if smp is not None:
            smp.kill()
        err, alone_ok = repr(e), False
        rec = dict(samples=0, sm_median=None, sm_max_clock=None, throttle_seen=[], telemetry_error=err)
    t = err is None
    chk = dict(telemetry=t, sm_median=bool(t and rec["sm_median"] >= 0.97 * rec["sm_max_clock"]),
               throttle=bool(t and set(rec["throttle_seen"]) <= THROTTLE_OK), alone=bool(t and alone_ok))
    res = dict(when=a.when, core=core, gpu=smi("--query-gpu=name")[0][0], held_mib=ALLOC_MIB, seconds=a.seconds,
               matrix=2048, clocks=rec, checks=chk, passed=all(chk.values()),
               load=open("/proc/loadavg").read().split()[:3], time=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               sm_median_frac=(rec["sm_median"] / rec["sm_max_clock"]) if rec.get("sm_median") else None,
               rule="research repository ising_compare/icmp/host_gate.py gpu_check (2,048 x 2,048 fp16, 20 a sync)")
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1, default=float)
    print(f"GPU check ({a.when}): median SM {rec.get('sm_median')} of {rec.get('sm_max_clock')} MHz, throttle "
          f"{rec.get('throttle_seen')}, alone {rec.get('alone', {}).get('identified_by')}, checks {chk}: "
          f"{'PASSED' if res['passed'] else 'FAILED'}", flush=True)
    raise SystemExit(0 if res["passed"] else 1)


if __name__ == "__main__":
    main()
