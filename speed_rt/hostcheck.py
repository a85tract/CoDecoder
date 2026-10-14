"""D6 (A): the pre-screen of a timing pod, before anything is timed on it, and the checks that repeat it during the session:
SURF-v2's GPU check five times and the studies' host gate (results/surface/SURFACE.md, "Qualifying a timing session", rule 3; the
research repository's ising_compare/icmp/host_gate.py), measured with the release's commands. Since 13 October (review finding 7)
the reference of the two timing checks is the session's OWN pod, measured at its pre-screen; the 8480+ host's values are kept in
the record as a diagnostic only and decide nothing.
  core   the timed core's busy and steal shares over 5 s (/proc/stat), nothing of ours on it: busy at most 5 %, steal at most 1 %
  pm     plain PyMatching per block on validation chunk 0 (`timing.cpu_term ref --plain`, pinned to the timed core), 3 repeats at
         d = 9 and 2 at d = 13; per d the largest repeat at most 1.03 x the smallest. The reference is the median of the repeats
         (this pod's own); every later check (mode check, recheck) is at most 1.10 x that reference.
  floor  the router's floor (`timing.gpu_term --d 9 --handoff --limit 20000`'s floor_mapped: the compiled loop around a CUDA graph
         without the engine, 200 warm-up and 20,000 timed launches): 2 repeats, the larger at most 1.03 x the smaller; the
         reference is their mean; later checks (mode recheck) at most 1.15 x it
  gpu    `surf_v2.gpucheck --when prescreen|start|end` five times, 6 s apart: all five pass
The timed core is chosen first, mechanically, as the studies' sessions chose it (SURFACE.md rule 2; changed on 13 October: a 10 s sample instead of 1 s, ties to the highest index): among the cores this process may
use on the GPU's NUMA node, outside core 0's physical core, the one whose physical core (with its SMT siblings) was least busy over
10 s (ties to the highest core index: the lowest cores of a NUMA node carry the kernel's interrupt and housekeeping work; on the
first pod, core 52, the first of its node, failed its 5 % idle-core check twice in a row). The session pins every timed process
to it. This process is pinned off the timed core while it measures it.
The record also names the host: CPU model, visible CPUs, the cpuset, the container's CPU quota (cgroup cpu.max), threads per core.

    python -m speed_rt.hostcheck --val VAL --build BUILD --out DIR              the pre-screen (the original command line)
        -> DIR/hostcheck.json (core, reference, checks, passed); exit 4 when the pod fails
    python -m speed_rt.hostcheck --mode check --ref PRE/hostcheck.json --val VAL --out DIR
        before every timed process: the core idle for 3 s (busy at most 5 %, steal at most 1 %), then PyMatching d = 9 and 13
        once each at most 1.10 x the reference -> DIR/hostcheck_check.json; exit 4 when it fails
    python -m speed_rt.hostcheck --mode recheck --when start|end --ref PRE/hostcheck.json --val VAL --build BUILD --out DIR
        the set again against the pod's own reference (PyMatching once per d, floor once, five GPU checks; the core's busy and steal over 5 s are
        recorded, not judged: every timed process has its own core check and window check)
        -> DIR/hostcheck_recheck.json; exit 4 when it fails
A failed check spends nothing: the pod is replaced (SPEED_RT.md, "Pods"). Changes of 13 October: the own-pod references, the floor's
repeats, the modes check and recheck, the host record (host_info), the process pinned off the timed core, the core pick (10 s, highest index on ties); the original command line
and the keys core / pick / checks.core / checks.pm / checks.floor / checks.gpu / passed of hostcheck.json are unchanged
(checks.pm[d].ref_us is now the pod's own reference, ref_old_us the 8480+ value)."""
import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time

REF = dict(pm_plain_us={9: 35.895, 13: 133.638}, floor_mapped_us=4.986, host="Intel Xeon Platinum 8480+, core 1")   # diagnostic only
LIMITS = dict(core_busy=0.05, core_steal=0.01, pm_rel=1.10, pm_spread=0.03, floor_rel=1.15, floor_spread=0.03, gpu_checks=5,
              pm_repeats={9: 3, 13: 2}, floor_repeats=2, core_secs_prescreen=5.0, core_secs_check=3.0, pick_secs=10.0)


def numa_cores():
    bus = subprocess.run(["nvidia-smi", "--query-gpu=pci.bus_id", "--format=csv,noheader"], capture_output=True,
                         text=True).stdout.split()[0].lower()
    bus = bus[4:] if len(bus.split(":")[0]) == 8 else bus             # 00000000:19:00.0 -> 0000:19:00.0
    node = int(open(f"/sys/bus/pci/devices/{bus}/numa_node").read())
    allowed = os.sched_getaffinity(0)
    if node < 0:
        return sorted(allowed), node
    from speed_rt import watch
    mine = watch.cpus(open(f"/sys/devices/system/node/node{node}/cpulist").read())
    return sorted(allowed & mine), node


def pick_core():
    from speed_rt import watch
    cores, node = numa_cores()
    zero = set(watch.siblings(0)) | {0}
    cand = [c for c in cores if c not in zero]
    s0 = watch.stat()
    time.sleep(LIMITS["pick_secs"])
    s1 = watch.stat()

    def busy(c):
        phys = [c] + watch.siblings(c)
        tot = sum(s1[x][2] - s0[x][2] for x in phys if x in s0)
        return sum(s1[x][0] - s0[x][0] for x in phys if x in s0) / tot if tot > 0 else 1.0
    best = min(cand, key=lambda c: (busy(c), -c))
    return best, dict(numa_node=node, candidates=len(cand), busy_1s=busy(best), pick_secs=LIMITS["pick_secs"],
                      busy_pick=busy(best))


def quota_cpus():
    """The container's CPU quota in CPUs (cgroup v2 cpu.max, else v1), None for no limit."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else round(int(q) / int(p), 2)
    except (OSError, ValueError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        return None if q < 0 else round(q / p, 2)
    except (OSError, ValueError):
        return None


def host_info():
    info = open("/proc/cpuinfo").read()
    ml = re.search(r"model name\s*:\s*(.*)", info)
    allowed = re.search(r"Cpus_allowed_list:\s*(\S+)", open("/proc/self/status").read()).group(1)
    tpc = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    tpc = re.search(r"Thread\(s\) per core:\s*(\d+)", tpc)
    return dict(cpu_model=ml.group(1).strip() if ml else "?", cpus_visible=os.cpu_count(), cpuset=allowed, quota_cpus=quota_cpus(),
                threads_per_core=int(tpc.group(1)) if tpc else None, kernel=os.uname().release, node=os.uname().nodename)


def pin_off(core):
    """This process off the timed core and its SMT siblings, so that it never adds to the core's busy time."""
    from speed_rt import watch
    off = watch.off_cores(core)
    if off:
        os.sched_setaffinity(0, watch.cpus(off))


def core_check(core, secs):
    from speed_rt import watch
    s0 = watch.stat()
    time.sleep(secs)
    hw = watch.host_window(s0, watch.stat(), core)
    return dict(secs=secs, busy=hw["core_busy"], steal=hw["core_steal"], siblings_busy=hw["sibling_busy"], others_busy=hw["others_busy"],
                loadavg=hw["loadavg"],
                ok=bool((hw["core_busy"] or 0) <= LIMITS["core_busy"] and (hw["core_steal"] or 0) <= LIMITS["core_steal"]))


def pm_runs(py, val, core, out, d, reps, first_rep=1):
    """Plain PyMatching per block (us) at d, reps repeats pinned to the timed core."""
    v = []
    for r in range(first_rep, first_rep + reps):
        subprocess.run([py, "-m", "timing.cpu_term", "ref", "--plain", "--d", str(d), "--val", val, "--rep", str(r), "--core", str(core),
                        "--out", os.path.join(out, "pm")], check=True, stdout=subprocess.DEVNULL)
    for r in range(first_rep, first_rep + reps):
        v.append(json.load(open(os.path.join(out, "pm", f"ref_plain_d{d}_rep{r}.json")))["per_block_us"])
    return v


def median(v):
    s = sorted(v)
    return s[len(s) // 2] if len(s) % 2 else 0.5 * (s[len(s) // 2 - 1] + s[len(s) // 2])


def floor_run(py, val, build, core, out, proc, masks):
    """One run of the router floor (floor_mapped, us) into OUT/floor<proc>."""
    d = os.path.join(out, f"floor{proc}")
    subprocess.run([py, "-m", "timing.gpu_term", "--d", "9", "--val", val, "--proc", str(proc), "--core", str(core), "--out", d,
                    "--deployed", masks, "--build", build, "--handoff", "--limit", "20000"], check=True, stdout=subprocess.DEVNULL)
    return json.load(open(glob.glob(os.path.join(d, "*_handoff_p*.json"))[0]))["floor_mapped"]


def make_masks(py, val, build, out):
    subprocess.run([py, "-m", "timing.deployed", "--d", "9", "--val", val, "--out", os.path.join(out, "masks"), "--build", build],
                   check=True, stdout=subprocess.DEVNULL)
    return os.path.join(out, "masks", "val_d9.npz")


def gpu_checks(py, core, out, when, n):
    g = []
    for k in range(1, n + 1):
        if k > 1:
            time.sleep(6)
        rc = subprocess.run([py, "-m", "surf_v2.gpucheck", "--when", when, "--core", str(core), "--out", os.path.join(out, f"gpucheck_{k}.json")],
                            stdout=subprocess.DEVNULL).returncode
        g.append(rc == 0)
    return dict(passed=g, ok=all(g))


def finish(rec, out, name, lines):
    rec["failed"] = sorted(k for k, v in rec["checks"].items() if not v["ok"])
    rec["passed"] = not rec["failed"]
    rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(out, name), "x") as fh:
        json.dump(rec, fh, indent=1)
    print(f"{rec['item'].upper()} {'PASSED' if rec['passed'] else 'FAILED ' + ', '.join(rec['failed'])}: core {rec['core']} {lines}", flush=True)
    raise SystemExit(0 if rec["passed"] else 4)


def prescreen(a):
    from speed_rt import watch
    os.makedirs(a.out, exist_ok=False)
    py = sys.executable
    core, pick = pick_core()
    pin_off(core)
    hi = host_info()
    rec = dict(item="pre-screen", ref=REF, limits=LIMITS, core=core, pick=pick, host=hi["cpu_model"], host_info=hi, cpus=os.cpu_count(),
               start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checks={})
    cc = core_check(core, LIMITS["core_secs_prescreen"])
    rec["checks"]["core"] = cc
    pm, ref_pm = {}, {}
    for d, reps in LIMITS["pm_repeats"].items():
        v = pm_runs(py, a.val, core, a.out, d, reps)
        ref_pm[d] = median(v)
        pm[str(d)] = dict(repeats_us=v, ref_us=ref_pm[d], ref_old_us=REF["pm_plain_us"][d], vs_old_host=max(v) / REF["pm_plain_us"][d],
                          ok=bool(len(v) == reps and max(v) <= LIMITS["pm_rel"] * ref_pm[d] and max(v) / min(v) - 1 <= LIMITS["pm_spread"]))
    rec["checks"]["pm"] = dict(pm, ok=all(x["ok"] for x in pm.values()))
    masks = make_masks(py, a.val, a.build, a.out)
    fl = [floor_run(py, a.val, a.build, core, a.out, k, masks) for k in range(LIMITS["floor_repeats"])]
    means = [f["mean"] for f in fl]
    ref_floor = sum(means) / len(means)
    rec["checks"]["floor"] = dict(floor_mapped=fl[0], floor_mapped_repeats=fl, repeats_us=means, ref_us=ref_floor, ref_old_us=REF["floor_mapped_us"],
                                  vs_old_host=max(means) / REF["floor_mapped_us"],
                                  ok=bool(max(means) / min(means) - 1 <= LIMITS["floor_spread"]))
    rec["checks"]["gpu"] = gpu_checks(py, core, a.out, "prescreen", LIMITS["gpu_checks"])
    rec["reference"] = dict(pm_plain_us={str(d): v for d, v in ref_pm.items()}, floor_mapped_us=ref_floor, masks=os.path.join(a.out, "masks", "val_d9.npz"),
                            what="this pod's own, at its pre-screen: the median of the PM repeats, the mean of the floor repeats")
    finish(rec, a.out, "hostcheck.json", f"(busy {cc['busy']}, steal {cc['steal']}); PM d9 {pm['9']['repeats_us']} d13 {pm['13']['repeats_us']}; "
           f"floor {means} us; GPU checks {sum(rec['checks']['gpu']['passed'])}/{LIMITS['gpu_checks']}; {hi['cpu_model']}, quota {hi['quota_cpus']} CPUs")


def later(a, full):
    """mode check (full False): the core and PM once per d; mode recheck (full True): also the floor and the five GPU checks."""
    ref = json.load(open(a.ref))
    assert ref.get("passed") and ref["item"] == "pre-screen", f"{a.ref} is not a passed pre-screen"
    core, r = ref["core"], ref["reference"]
    os.makedirs(a.out, exist_ok=False)
    pin_off(core)
    py = sys.executable
    rec = dict(item="recheck" if full else "check", when=a.when if full else None, ref_file=os.path.abspath(a.ref), limits=LIMITS, core=core,
               host_info=host_info(), start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checks={})
    cc = core_check(core, LIMITS["core_secs_prescreen" if full else "core_secs_check"])
    if full:       # the start and end check of a session: the core is recorded, not judged (every process has its own core check and window check)
        cc = dict(cc, ok_if_judged=cc["ok"], ok=True, judged=False)
    rec["checks"]["core"] = cc
    pm = {}
    for d in LIMITS["pm_repeats"]:
        v = pm_runs(py, a.val, core, a.out, d, 1)
        pm[str(d)] = dict(repeats_us=v, ref_us=r["pm_plain_us"][str(d)], vs_ref=v[0] / r["pm_plain_us"][str(d)],
                          ok=bool(v[0] <= LIMITS["pm_rel"] * r["pm_plain_us"][str(d)]))
    rec["checks"]["pm"] = dict(pm, ok=all(x["ok"] for x in pm.values()))
    tail = ""
    if full:
        fl = floor_run(py, a.val, a.build, core, a.out, 1, r["masks"])
        rec["checks"]["floor"] = dict(floor_mapped=fl, mean_us=fl["mean"], ref_us=r["floor_mapped_us"], vs_ref=fl["mean"] / r["floor_mapped_us"],
                                      ok=bool(fl["mean"] <= LIMITS["floor_rel"] * r["floor_mapped_us"]))
        rec["checks"]["gpu"] = gpu_checks(py, core, a.out, a.when, LIMITS["gpu_checks"])
        tail = f"; floor {fl['mean']:.3f} us ({fl['mean'] / r['floor_mapped_us']:.3f} x ref); GPU checks {sum(rec['checks']['gpu']['passed'])}/{LIMITS['gpu_checks']}"
    finish(rec, a.out, "hostcheck_recheck.json" if full else "hostcheck_check.json",
           f"(busy {cc['busy']}, steal {cc['steal']}); PM d9 {pm['9']['vs_ref']:.3f} x ref, d13 {pm['13']['vs_ref']:.3f} x ref{tail}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val", required=True, help="the studies' validation chunks (convlines.testset val)")
    ap.add_argument("--build", help="timing/d7/build.sh's output (the router library); needed by the pre-screen and by recheck")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=("prescreen", "check", "recheck"), default="prescreen")
    ap.add_argument("--ref", help="the pre-screen's hostcheck.json (modes check and recheck)")
    ap.add_argument("--when", choices=("start", "end"), default="start", help="mode recheck: the label of the GPU check")
    a = ap.parse_args()
    if a.mode == "prescreen":
        if not a.build:
            ap.error("--build is required")
        prescreen(a)
    else:
        if not a.ref or (a.mode == "recheck" and not a.build):
            ap.error("--ref (and --build for recheck) are required")
        later(a, a.mode == "recheck")


if __name__ == "__main__":
    main()
