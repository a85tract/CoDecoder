"""The host and clock checks inside every timed window on a pod (SPEED_RT.md, "Acceptance"; review findings 4, 5 and 7).

A window is the timed part of one process. Over it:
  host   /proc/stat at its start and end: the timed core's steal share and the busy share of the timed core by anything
         other than the timed process (the core's busy time minus the process's own CPU time, which is its user plus
         system time from the kernel's resource usage; the CPU time of the process's threads and of the children it
         waited for count as its own). The window holds when the steal is at most 1 % and that other busy share is at
         most 5 % of the window (the studies' gate limits for the idle core, applied to the window). /proc/stat is
         counted in ticks (1/CLK_TCK s, 10 ms); two ticks are allowed for that quantization, and the allowance is
         recorded (tick_allow_s). Recorded only, never gating: the SMT siblings' busy share, every other CPU's busy
         share, the load average. Finding 7: before 13 October the check tested the SMT siblings, not the timed core.
  clocks nvidia-smi every 100 ms on CPUs other than the timed core and its siblings (surface_v2/timing/gpu_term.py's
         Clocks, unchanged). Rule "floor" (a continuously loaded GPU: our engines, Ising's plans, trtexec): every
         sample at least 97 % of the maximum SM clock, throttle reasons none or idle. Rule "throttle" (a GPU loaded only
         now and then, whose idle gaps drop the clock: BP-OSD and CUDA-Q, whose OSD runs on the host): throttle reasons
         none or idle in every sample. Both rules also need no other compute process on the GPU before or after.
         Rule "host" (a CPU-only process: Relay-BP, ldpc BP-OSD, PyMatching): only the host check; the GPU is not
         sampled and nvidia-smi is not called, so it also runs where there is no GPU.
  power  a sample whose throttle reasons include the software power cap (0x4) while this process alone uses the GPU is
         the workload's own power capping (the H100's 700 W limit under Cascade's batched kernels), recorded apart
         from interference by others (another process on the GPU, thermal or hardware slowdown).

    python -m speed_rt.watch run --core C --rule floor|throttle|host --out REC.json [--marker TEXT] -- CMD ...
runs CMD pinned to core C (taskset) and writes REC.json: the command, its exit code, the window (from the start, or
from the first output line containing TEXT, to the end), the host and clock summaries (host: own_cpu_s, core_other_busy,
core_steal) and the verdict; CMD's output is passed through. The watch process itself is pinned off the timed core and
its SMT siblings, so that it never adds to the core's busy time. Library use: Window(core, rule) ... .stop() -> the
same summary; used inside the timed process itself, the process's own CPU time is taken from its resource usage
since the window's start (stop(own_cpu_s=SECONDS) overrides it).

Changes of 13 October (the command line and the library names are unchanged; "host" is a new rule; host_window and
Window.stop take an optional own_cpu_s; the host verdict no longer reads the SMT siblings; LIMITS has core_other_busy
and tick_allow in place of sibling_busy; the host record has core_other_busy, own_cpu_s, window_ticks, tick_allow_s)."""
import argparse
import json
import os
import resource
import subprocess
import sys
import time

THROTTLE_OK = {"0x0000000000000000", "0x0000000000000001"}
POWER_CAP = 0x4
LIMITS = dict(sm_floor=0.97, core_steal=0.01, core_other_busy=0.05, tick_allow=2)
CLK = os.sysconf("SC_CLK_TCK")


def cpus(text):
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def siblings(core):
    return sorted(cpus(open(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list").read()) - {core})


def off_cores(core):
    sib = set(siblings(core)) | {core}
    return ",".join(str(c) for c in sorted(cpus(open("/sys/devices/system/cpu/online").read()) - sib))


def stat():
    """{cpu index: (busy jiffies, steal jiffies, total jiffies)} from /proc/stat."""
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            v = [int(x) for x in f[1:]]
            user, nice, system, idle, iowait, irq, softirq, steal = (v + [0] * 8)[:8]
            busy = user + nice + system + irq + softirq
            out[int(f[0][3:])] = (busy, steal, busy + idle + iowait + steal)
    return out


def host_window(s0, s1, core, own_cpu_s=0.0):
    """The host record of one window from two stat() snapshots. own_cpu_s: CPU seconds of the timed process, taken off
    the core's busy time (0 for a core that should be idle, as the pre-screen's checks use it)."""
    def frac(c, k):
        d = s1[c][2] - s0[c][2]
        return (s1[c][k] - s0[c][k]) / d if d > 0 else None
    sib = siblings(core)
    others = [c for c in s1 if c != core and c not in sib and c in s0]
    tot = sum(s1[c][2] - s0[c][2] for c in others)
    busy_others = sum(s1[c][0] - s0[c][0] for c in others) / tot if tot > 0 else None
    ticks = s1[core][2] - s0[core][2]
    busy_s = (s1[core][0] - s0[core][0]) / CLK
    allow_s = LIMITS["tick_allow"] / CLK
    other = max(0.0, busy_s - own_cpu_s - allow_s) / (ticks / CLK) if ticks > 0 else None
    r = dict(core=core, window_ticks=ticks, core_busy=frac(core, 0), own_cpu_s=round(own_cpu_s, 4), tick_allow_s=allow_s,
             core_other_busy=other, core_steal=frac(core, 1), siblings=sib, sibling_busy=[frac(c, 0) for c in sib],
             others_busy=busy_others, loadavg=open("/proc/loadavg").read().split()[:3])
    r["ok"] = bool((r["core_steal"] or 0.0) <= LIMITS["core_steal"] and (r["core_other_busy"] or 0.0) <= LIMITS["core_other_busy"])
    return r


def proc_cpu(pid):
    """CPU seconds so far of a process (user + system, with its waited-for children), from /proc/PID/stat."""
    try:
        f = open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()
        return sum(int(x) for x in f[11:15]) / CLK
    except (OSError, IndexError, ValueError):
        return 0.0


def own_cpu():
    """CPU seconds of this process (all its threads) and of the children it has waited for."""
    a, b = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return a.ru_utime + a.ru_stime + b.ru_utime + b.ru_stime


def smi_apps():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,name", "--format=csv,noheader"], capture_output=True,
                       text=True)
    return [x for x in r.stdout.splitlines() if x.strip()]


class Clocks:
    """nvidia-smi every 100 ms on the CPUs other than the timed core and its siblings, each sample time-stamped on
    arrival so that a window can start after the process's own marker."""
    Q = "timestamp,clocks.sm,clocks.max.sm,clocks_throttle_reasons.active,power.draw,temperature.gpu"

    def __init__(self, core):
        cmd = ["taskset", "-c", off_cores(core), "nvidia-smi", f"--query-gpu={self.Q}", "--format=csv,noheader,nounits",
               "-lms", "100"]
        self.p = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.rows = []
        import threading
        self.t = threading.Thread(target=self._read, daemon=True)
        self.t.start()

    def _read(self):
        for line in self.p.stdout:
            f = [x.strip() for x in line.split(",")]
            try:
                self.rows.append(dict(t=time.time(), sm=float(f[1]), sm_max=float(f[2]), throttle=f[3],
                                      power=float(f[4]), temp=float(f[5])))
            except (ValueError, IndexError):
                pass

    def stop(self, t0=None, rule="floor", alone=True):
        self.p.terminate()
        try:
            self.p.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.t.join(timeout=5)
        rows = [r for r in self.rows if t0 is None or r["t"] >= t0]
        sm = [r["sm"] for r in rows]
        mx = max((r["sm_max"] for r in rows), default=None)
        thr = sorted({r["throttle"] for r in rows})
        own_cap = [r for r in rows if int(r["throttle"], 16) & POWER_CAP]
        res = dict(rule=rule, samples=len(rows), sm_min=min(sm, default=None), sm_max_clock=mx,
                   sm_median=sorted(sm)[len(sm) // 2] if sm else None,
                   sm_min_frac=(min(sm) / mx) if (sm and mx) else None, throttle_seen=thr,
                   power_cap_samples=len(own_cap), power_max=max((r["power"] for r in rows), default=None))
        floor_ok = res["sm_min_frac"] is not None and res["sm_min_frac"] >= LIMITS["sm_floor"]
        res["ok"] = bool(rows) and set(thr) <= THROTTLE_OK and (floor_ok or rule == "throttle")
        res["workload_power_cap"] = bool(own_cap) and alone and all(
            int(r["throttle"], 16) & ~(POWER_CAP | 0x1) == 0 for r in own_cap)
        return res


class Window:
    """Host and clock record of one window: Window(core, rule) at the start, .mark() where the timed part begins
    (optional), .stop(apps_before) at the end. Rule "host": no GPU sampling."""
    def __init__(self, core, rule="floor"):
        self.core, self.rule = core, rule
        self.gpu = rule != "host"
        self.apps0 = smi_apps() if self.gpu else []
        self.clocks = Clocks(core) if self.gpu else None
        self.t0 = time.time()
        self.s0 = stat()
        self.c0 = own_cpu()

    def mark(self):
        self.t0, self.s0, self.c0 = time.time(), stat(), own_cpu()

    def stop(self, own_pids=(), own_cpu_s=None):
        s1, t1 = stat(), time.time()
        own = (own_cpu() - self.c0) if own_cpu_s is None else own_cpu_s
        apps1 = smi_apps() if self.gpu else []
        mine = {str(p) for p in own_pids}
        alone = all(x.split(",")[0].strip() in mine for x in self.apps0 + apps1)
        host = host_window(self.s0, s1, self.core, own)
        if self.gpu:
            clk = self.clocks.stop(self.t0, self.rule, alone)
        else:
            clk = dict(rule="host", samples=0, ok=True, sm_min_frac=None, throttle_seen=[], workload_power_cap=False)
        return dict(window_s=round(t1 - self.t0, 3), host=host, clocks=clk, apps_before=self.apps0, apps_after=apps1,
                    gpu_alone=alone, accepted=bool(host["ok"] and clk["ok"] and alone))


def run(a, cmd):
    off = off_cores(a.core)
    if off:
        os.sched_setaffinity(0, cpus(off))                 # the watch itself never runs on the timed core
    w = Window(a.core, a.rule)
    t_start = time.time()
    p = subprocess.Popen(["taskset", "-c", str(a.core)] + cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, bufsize=1)
    marked = not a.marker
    c_mark = 0.0
    for line in p.stdout:
        sys.stdout.write(line)
        if not marked and a.marker in line:
            w.mark()
            c_mark = proc_cpu(p.pid)                       # the child's CPU time before the window's start
            marked = True
    # the child's own CPU time (its threads and the children it waited for) from its resource usage; Popen is told its code
    _, status, ru = os.wait4(p.pid, 0)
    p.returncode = rc = os.waitstatus_to_exitcode(status)
    own = max(0.0, ru.ru_utime + ru.ru_stime - c_mark)
    # the process is gone before the apps are read again; its own pid (and a container's host pid) cannot be listed
    rec = w.stop(own_pids=(), own_cpu_s=own)
    if w.gpu:
        rec["gpu_alone"] = not w.apps0 and not rec["apps_after"]
    else:
        rec["gpu_alone"] = True
    rec["accepted"] = bool(rec["host"]["ok"] and rec["clocks"]["ok"] and rec["gpu_alone"] and rc == 0 and marked)
    rec.update(cmd=cmd, exit=rc, marker=a.marker or None, marker_seen=marked, core=a.core, rule=a.rule, child_cpu_s=round(own, 4),
               start=time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t_start)),
               end=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    with open(a.out, "x") as fh:
        json.dump(rec, fh, indent=1)
    print(f"watch: exit {rc}, window {rec['window_s']} s, host ok {rec['host']['ok']} (other busy {rec['host']['core_other_busy']}, "
          f"steal {rec['host']['core_steal']}), clocks ok {rec['clocks']['ok']} (min {rec['clocks']['sm_min_frac']}), "
          f"alone {rec['gpu_alone']}: accepted {rec['accepted']}", flush=True)
    return rec


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: python -m speed_rt.watch run --core C --rule floor|throttle|host --out REC [--marker T] -- CMD")
    i = sys.argv.index("--")
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run",))
    ap.add_argument("--core", type=int, required=True)
    ap.add_argument("--rule", choices=("floor", "throttle", "host"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--marker", default="")
    a = ap.parse_args(sys.argv[1:i])
    rec = run(a, sys.argv[i + 1:])
    raise SystemExit(0 if rec["exit"] == 0 else rec["exit"])


if __name__ == "__main__":
    main()
