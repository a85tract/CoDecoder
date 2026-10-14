"""SPEED-RT BB stage: the pre-screen of a timing pod, stage 1's rule (SPEED_RT.md section 1.5) and nothing else. Pods time GPU work only
(CURRENT RULES, 13 October 15:45), so there is no host-CPU busy check and no PyMatching reference on the pod. On the idle pod:
  cpu    CPU steal at most 1 % over 5 s on the container's CPUs (/proc/stat), and the container's CPU quota at least 20 CPUs
  gpu    surf_v2.gpucheck's burst five times, 6 s apart, each judged as stage 1 judges it: the telemetry usable, the median SM clock at
         least 97 % of the maximum, every clock-event reason within GPU idle (0x1), applications clocks (0x2) and the SW power cap (0x4),
         the enforced power limit equal to the card's default, the GPU alone; gpucheck's own pass/fail is not used
All five must pass, else the pod is replaced (exit 4). The record also holds the card (name, driver, power limit, maximum SM clock), the
CPU model, visible CPUs, cpuset, quota, kernel, and the timed core: the process pins each timed process to one core, chosen mechanically
as the highest-numbered core of the container's cpuset on the GPU's NUMA node (the lowest cores carry the kernel's housekeeping work); it
is pinned for the launch latency of the GPU items and for CUDA-Q's host part, and nothing is judged on it.
    python -m ... prescreen_bb.py --out DIR      (run in an environment with torch and surf_v2 on PYTHONPATH: hcvenv, PYTHONPATH=surface_v2)
writes DIR/prescreen.json (passed, core, cpu, checks, host) and DIR/gpucheck_k.json; prints one summary line per step."""
import argparse
import json
import os
import subprocess
import sys
import time


def cpu_check():
    cpus = sorted(os.sched_getaffinity(0))

    def snap():
        d = {}
        for ln in open("/proc/stat"):
            if ln.startswith("cpu") and ln[3].isdigit():
                f = ln.split()
                d[int(f[0][3:])] = [int(x) for x in f[1:]]
        return d
    a = snap()
    time.sleep(5)
    b = snap()
    tot = steal = 0
    for c in cpus:
        d = [y - x for x, y in zip(a[c], b[c])]
        tot += sum(d[:8])
        steal += d[7]                                           # user nice system idle iowait irq softirq steal
    quota = None
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        quota = None if q == "max" else int(q) / int(per)
    except OSError:
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            quota = q / int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read()) if q > 0 else None
        except OSError:
            pass
    cpu_quota = quota if quota is not None else float(len(cpus))
    pct = 100.0 * steal / max(tot, 1)
    return dict(cpus_visible=len(cpus), steal_pct_5s=pct, quota_cpus=quota, quota_or_cpus=cpu_quota, steal_ok=pct <= 1.0,
                quota_ok=cpu_quota >= 20.0)


def timed_core():
    """The highest-numbered core of the cpuset on the GPU's NUMA node (the whole cpuset when the node is unknown)."""
    allowed = os.sched_getaffinity(0)
    try:
        bus = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True,
                             text=True).stdout.split()[0].lower()
        bus = bus[4:] if len(bus.split(":")[0]) == 8 else bus
        node = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
        if node >= 0:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from speed_rt import watch
            mine = watch.cpus(open(f"/sys/devices/system/node/node{node}/cpulist").read())
            if allowed & mine:
                return max(allowed & mine), node
    except (OSError, ValueError, IndexError, ImportError):
        pass
    return max(allowed), None


def judge(d, lim, default):
    c, ck = d["clocks"], d["checks"]
    reasons = sorted({r[2] for r in c.get("rows", []) if r})
    bad = [r for r in reasons if int(r, 16) & ~0x7]             # anything but GPU idle 0x1, applications clocks 0x2, SW power cap 0x4
    v = dict(telemetry=ck["telemetry"], sm_median_ge_97pct=ck["sm_median"], reasons_within_0x1_0x2_0x4=ck["telemetry"] and not bad,
             power_limit_is_default=lim == default, alone=ck["alone"], reasons_seen=reasons, sm_median=c.get("sm_median"),
             sm_max_clock=c.get("sm_max_clock"), sm_min=c.get("sm_min"), power_max_w=c.get("power_max"), enforced_w=lim, default_w=default,
             power_capped_rows=sum(1 for r in c.get("rows", []) if r and int(r[2], 16) & 4), rows=c.get("samples"))
    v["passed"] = all(v[k] for k in ("telemetry", "sm_median_ge_97pct", "reasons_within_0x1_0x2_0x4", "power_limit_is_default", "alone"))
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    q = lambda f: subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    host = dict(card=q("name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.limit,enforced.power.limit,power.max_limit"),
                cpu=next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo") if ln.startswith("model name")), None),
                cpuset=open("/proc/self/status").read().split("Cpus_allowed_list:")[1].split()[0],
                cpu_max=open("/sys/fs/cgroup/cpu.max").read().strip() if os.path.exists("/sys/fs/cgroup/cpu.max") else None,
                kernel=os.uname().release)
    print("pre-screen of", os.uname().nodename, "|", host["card"], "|", host["cpu"])
    cpu = cpu_check()
    print(f"CPU: {cpu['cpus_visible']} visible, steal {cpu['steal_pct_5s']:.3f} % over 5 s (limit 1 %), quota {cpu['quota_cpus']} CPUs (limit 20): "
          f"{'ok' if cpu['steal_ok'] and cpu['quota_ok'] else 'FAILED'}", flush=True)
    core, node = timed_core()
    checks = []
    if cpu["steal_ok"] and cpu["quota_ok"]:
        for k in range(1, 6):
            if k > 1:
                time.sleep(6)
            lim, default = (float(x) for x in q("enforced.power.limit,power.default_limit").split("\n")[0].replace(" ", "").split(","))
            f = f"{a.out}/gpucheck_{k}.json"
            r = subprocess.run([sys.executable, "-m", "surf_v2.gpucheck", "--when", "prescreen", "--out", f], capture_output=True, text=True)
            open(f"{a.out}/gpucheck_{k}.log", "w").write(r.stdout + r.stderr)
            if not os.path.exists(f):
                checks.append(dict(passed=False, error="gpucheck wrote no record"))
                print(f"GPU check {k}: no record: FAILED", flush=True)
                break
            v = judge(json.load(open(f)), lim, default)
            checks.append(v)
            print(f"GPU check {k}: median SM {v['sm_median']} of {v['sm_max_clock']} MHz (min {v['sm_min']}), reasons {v['reasons_seen']}, "
                  f"power-capped rows {v['power_capped_rows']} of {v['rows']}, limit {lim:g}/{default:g} W, alone {v['alone']}: "
                  f"{'PASSED' if v['passed'] else 'FAILED'}", flush=True)
            if not v["passed"]:
                break
    passed = bool(cpu["steal_ok"] and cpu["quota_ok"] and len(checks) == 5 and all(c["passed"] for c in checks))
    doc = dict(passed=passed, core=core, numa_node=node, cpu=cpu, checks=checks, host=host, rule="stage 1's pre-screen (SPEED_RT.md 1.5)")
    json.dump(doc, open(f"{a.out}/prescreen.json", "w"), indent=1, sort_keys=True)
    print(f"PRE-SCREEN {'PASSED' if passed else 'FAILED'}: core {core} (NUMA node {node}); {sum(c['passed'] for c in checks)} of 5 GPU checks; "
          f"steal {cpu['steal_pct_5s']:.3f} %, quota {cpu['quota_cpus']}; {host['cpu']}")
    sys.exit(0 if passed else 4)


if __name__ == "__main__":
    main()
