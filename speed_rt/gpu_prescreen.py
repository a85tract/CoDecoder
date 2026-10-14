"""The pre-screen of a surface-stage GPU pod: stage 1's rule (SPEED_RT.md section 1.5, as cascade/h100_pod.sh applies it), run on the
idle pod before anything is timed on it. The pods of this stage time GPU work only (CURRENT_RULES, 13 October 15:45), so there is
no host-CPU busy check here.

  five GPU checks   `surf_v2.gpucheck --when prescreen` (a separate process that holds 1 GiB and runs 3 s of fp16 matrix products),
                    6 s apart, each judged by this module: the median SM clock at least 97 % of the maximum; every sample's clock-event
                    reasons within GPU idle 0x1, applications clocks 0x2 and SW power cap 0x4; the enforced power limit equal to the
                    default; under 64 MiB in use before and no other process on the GPU. All five must pass.
  CPU steal         at most 1 % over 5 s on the container's CPUs (/proc/stat, summed over the cpuset).
  container quota   at least 20 CPUs (cgroup cpu.max).
  timed core        the core every timed process of the session is pinned to, chosen mechanically as the studies' sessions chose it
                    (hostcheck.pick_core: among the cores of the GPU's NUMA node outside core 0's physical core, the one whose physical
                    core was least busy over 10 s, ties to the highest index). It is a pin, not a gate: no busy check is made on it.
The record keeps the pod's CPU model, visible CPUs, cpuset, quota, driver, power limit, maximum SM clock and every check's samples.

    python -m speed_rt.gpu_prescreen --out DIR     -> DIR/prescreen.json (passed, core, host, steal, quota, checks); exit 4 when the pod
                                                       fails (it is released and replaced: SPEED_RT-style bound in the stage's section)
Shared file imported unchanged: speed_rt/hostcheck.py (pick_core, host_info)."""
import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "surface_v2"))

LIMITS = dict(steal_pct=1.0, steal_secs=5.0, quota_cpus=20.0, sm_median=0.97, checks=5, check_gap_s=6)


def steal_over(secs):
    cpus = sorted(os.sched_getaffinity(0))

    def snap():
        out = {}
        for ln in open("/proc/stat"):
            f = ln.split()
            if f[0].startswith("cpu") and f[0] != "cpu":
                v = list(map(int, f[1:9]))
                out[int(f[0][3:])] = (sum(v), v[7])             # user nice system idle iowait irq softirq steal
        return out
    a = snap()
    time.sleep(secs)
    b = snap()
    tot = sum(b[c][0] - a[c][0] for c in cpus if c in a and c in b)
    st = sum(b[c][1] - a[c][1] for c in cpus if c in a and c in b)
    return 100.0 * st / max(tot, 1), len(cpus)


def power_limits():
    r = subprocess.run(["nvidia-smi", "--query-gpu=enforced.power.limit,power.default_limit", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.splitlines()[0]
    lim, default = (float(x) for x in r.replace(" ", "").split(","))
    return lim, default


def judge(path, lim, default):
    """One GPU check's verdict by stage 1's rule (the measurement of surf_v2.gpucheck, judged here; its own exit code is not used)."""
    d = json.load(open(path))
    c, ck = d["clocks"], d["checks"]
    reasons = sorted({r[2] for r in c.get("rows", []) if r})
    bad = [r for r in reasons if int(r, 16) & ~0x7]               # anything but 0x1, 0x2, 0x4
    v = dict(telemetry=bool(ck["telemetry"]), sm_median_ge_97pct=bool(ck["sm_median"]),
             reasons_within_0x1_0x2_0x4=bool(ck["telemetry"]) and not bad, power_limit_is_default=lim == default,
             alone=bool(ck["alone"]), reasons_seen=reasons, sm_median=c.get("sm_median"), sm_max_clock=c.get("sm_max_clock"),
             sm_min=c.get("sm_min"), power_max_w=c.get("power_max"), enforced_w=lim, default_w=default,
             power_capped_rows=sum(1 for r in c.get("rows", []) if r and int(r[2], 16) & 4), rows=c.get("samples"))
    v["passed"] = all(v[k] for k in ("telemetry", "sm_median_ge_97pct", "reasons_within_0x1_0x2_0x4", "power_limit_is_default", "alone"))
    return v


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    from speed_rt import hostcheck
    core, pick = hostcheck.pick_core()
    host = hostcheck.host_info()
    card = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,mig.mode.current,memory.total,clocks.max.sm,power.limit,"
                           "enforced.power.limit,power.max_limit", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    steal, ncpu = steal_over(LIMITS["steal_secs"])
    quota = host["quota_cpus"] if host["quota_cpus"] is not None else float(ncpu)
    rec = dict(item="gpu_prescreen", core=core, pick=pick, host=host, card=card, steal_pct_5s=steal, cpus_in_cpuset=ncpu,
               quota_or_cpus=quota, steal_ok=steal <= LIMITS["steal_pct"], quota_ok=quota >= LIMITS["quota_cpus"], limits=LIMITS,
               start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checks=[])
    print(f"CPU: {ncpu} in the cpuset, steal {steal:.3f} % over {LIMITS['steal_secs']:.0f} s (limit {LIMITS['steal_pct']} %), quota {quota} CPUs "
          f"(limit {LIMITS['quota_cpus']:.0f}): {'ok' if rec['steal_ok'] and rec['quota_ok'] else 'FAILED'}; timed core {core}; {host['cpu_model']}", flush=True)
    for k in range(1, LIMITS["checks"] + 1):
        if k > 1:
            time.sleep(LIMITS["check_gap_s"])
        lim, default = power_limits()
        f = os.path.join(a.out, f"check_{k}.json")
        subprocess.run([sys.executable, "-m", "surf_v2.gpucheck", "--when", "prescreen", "--core", str(core), "--out", f],
                       stdout=open(os.path.join(a.out, f"check_{k}.log"), "w"), stderr=subprocess.STDOUT)
        try:
            v = judge(f, lim, default)
        except (OSError, KeyError, ValueError) as e:
            v = dict(passed=False, error=repr(e))
        rec["checks"].append(v)
        print(f"GPU check {k}: median SM {v.get('sm_median')} of {v.get('sm_max_clock')} MHz, reasons {v.get('reasons_seen')}, power-capped rows "
              f"{v.get('power_capped_rows')} of {v.get('rows')}, limit {lim:g}/{default:g} W, alone {v.get('alone')}: "
              f"{'PASSED' if v['passed'] else 'FAILED'}", flush=True)
    rec["passed"] = bool(rec["steal_ok"] and rec["quota_ok"] and len(rec["checks"]) == LIMITS["checks"] and all(v["passed"] for v in rec["checks"]))
    rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(a.out, "prescreen.json"), "x") as fh:
        json.dump(rec, fh, indent=1)
    print(f"GPU PRE-SCREEN {'PASSED' if rec['passed'] else 'FAILED'}: {sum(v['passed'] for v in rec['checks'])} of {LIMITS['checks']} GPU checks, "
          f"steal {rec['steal_ok']}, quota {rec['quota_ok']}; core {core}", flush=True)
    raise SystemExit(0 if rec["passed"] else 4)


if __name__ == "__main__":
    main()
