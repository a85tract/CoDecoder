"""The host checks of the Decoder Switching stage's timed processes (CPU only; the registered host checks), standard
library only. Every check reads /proc/stat and the process's own CPU time from inside the timed process, so nothing else
runs on the timed core or beside it, and the analysis re-derives every verdict from the raw counters kept in the record.

  idle    over 5 s just before the timed window: the timed core's busy share by others at most 5 % (plus two ticks: the
          counters are whole ticks of 10 ms) and its steal share at most 1 %
  window  the same two shares over the timed window itself
  probe   plain PyMatching per block (d = 9, the studies' validation chunk 0, 10,000 shots, three passes in the process)
          right before the window: every pass at most 1.10 x the session's reference and the largest pass over the
          smallest at most 1.06 (C1's segment limit); the session's reference is --probe-ref (on the Linux workstation the cpu_ws
          harness's session reference; on a pod the median of the pre-screen's probes)

"Busy share by others" of the timed core over an interval = (the core's busy ticks (user + nice + system + irq + softirq)
minus this process's own utime + stime ticks) / (the core's ticks of all states but guest). The process is pinned to the
core, so its own ticks are the core's. Steal share = steal ticks / the same total. The siblings' busy share and the load
average are recorded and decide nothing. selftest() proves that /proc/stat's CPU numbers are the pinned core's (a container
that renumbers or hides the host's CPUs fails it)."""
import os
import time

# probe_spread: C1's segment limit on the Linux workstation, largest over smallest at most 1.06 (STAGE2_gapsw 1.4; 1.03 was set on Alpine and the pods)
LIMITS = dict(core_busy=0.05, core_steal=0.01, probe_rel=1.10, probe_spread=0.06, idle_s=5.0, probe_passes=3, tick_slack=2)
CLK = os.sysconf("SC_CLK_TCK")
STATE = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")


def cpus(text):
    out = set()
    for part in text.strip().split(","):
        lo, _, hi = part.partition("-")
        out.update(range(int(lo), int(hi or lo) + 1))
    return out


def siblings(core):
    try:
        return sorted(cpus(open(f"/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list").read()) - {core})
    except OSError:
        return []


def stat_lines():
    """{cpu: the first eight /proc/stat counters (ticks)}."""
    out = {}
    for line in open("/proc/stat"):
        if line.startswith("cpu") and line[3].isdigit():
            f = line.split()
            out[int(f[0][3:])] = [int(x) for x in (f[1:] + ["0"] * 8)[:8]]
    return out


def own_ticks():
    """utime + stime of this process (all its threads), in ticks."""
    f = open("/proc/self/stat").read().rsplit(")", 1)[1].split()
    return int(f[11]) + int(f[12])


def snap(core):
    st = stat_lines()
    keep = [core] + siblings(core)
    return dict(t=time.time(), cpu={str(c): st[c] for c in keep if c in st}, own=own_ticks(), clk=CLK,
                load=open("/proc/loadavg").read().split()[:3])


def shares(s0, s1, core):
    """The verdict's numbers from two snapshots (also used by the analysis on the stored raw counters)."""
    c0, c1 = s0["cpu"][str(core)], s1["cpu"][str(core)]
    d = [b - a for a, b in zip(c0, c1)]
    total = sum(d)
    busy = d[0] + d[1] + d[2] + d[5] + d[6]
    own = s1["own"] - s0["own"]
    sib = {}
    for k in s1["cpu"]:
        if k != str(core) and k in s0["cpu"]:
            e = [b - a for a, b in zip(s0["cpu"][k], s1["cpu"][k])]
            sib[k] = (e[0] + e[1] + e[2] + e[5] + e[6]) / sum(e) if sum(e) else None
    if total <= 0:
        return dict(ticks=total, own_ticks=own, busy_others=None, steal=None, siblings_busy=sib, ok=False)
    others = max(0, busy - own)
    return dict(ticks=total, own_ticks=own, busy_others=others / total, steal=d[7] / total, siblings_busy=sib,
                ok=bool(others <= LIMITS["core_busy"] * total + LIMITS["tick_slack"] and d[7] / total <= LIMITS["core_steal"]))


def verdict(s0, s1, core):
    r = shares(s0, s1, core)
    return dict(start=s0, end=s1, **r)


def idle_check(core, secs=None):
    """Sleep and measure: the core's busy share by others and steal share over `secs` (default 5 s)."""
    s0 = snap(core)
    time.sleep(secs or LIMITS["idle_s"])
    return verdict(s0, snap(core), core)


class Window:
    """Window(core).start() ... .stop() -> the window's verdict (raw counters and shares)."""

    def __init__(self, core):
        self.core = core

    def start(self):
        self.s0 = snap(self.core)
        return self

    def stop(self):
        return verdict(self.s0, snap(self.core), self.core)


def probe_check(values, ref_us):
    """values: the probe's passes (us per shot) -> the verdict against the session's reference (None: no check, the
    pre-screen's own probes)."""
    spread = max(values) / min(values) - 1
    ok = spread <= LIMITS["probe_spread"]
    if ref_us is not None:
        ok = bool(ok and max(values) <= LIMITS["probe_rel"] * ref_us)
    return dict(values_us=values, ref_us=ref_us, spread=spread, ok=bool(ok))


def selftest(core, secs=3.0):
    """A pinned spin of `secs`: the core's busy ticks must rise by this process's own ticks (within 5 % of them plus 5
    ticks), so /proc/stat's CPU `core` is the CPU this process runs on."""
    os.sched_setaffinity(0, {core})
    s0 = snap(core)
    t = time.perf_counter()
    x = 0
    while time.perf_counter() - t < secs:
        x += 1
    s1 = snap(core)
    r = shares(s0, s1, core)
    c0, c1 = s0["cpu"][str(core)], s1["cpu"][str(core)]
    busy = sum(c1[i] - c0[i] for i in (0, 1, 2, 5, 6))
    own = s1["own"] - s0["own"]
    ok = own > 0.8 * secs * CLK and abs(busy - own) <= 0.05 * own + 5
    return dict(core=core, secs=secs, core_busy_ticks=busy, own_ticks=own, ok=bool(ok), **{k: r[k] for k in ("busy_others", "steal")})
