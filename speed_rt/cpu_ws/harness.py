"""SPEED-RT's CPU harness on the Windows CPU host (a WSL2 distro): timed windows of up to 8 concurrent single-threaded
processes, with the stage's CPU-host checks (tmpfs only, no disk read in a window, one physical core per process).

A window is a set of at most 8 timed processes started together, one per slot. A slot is one vCPU of one SMT sibling
group (a core) of the VM; the group's other vCPU stays idle, and so do the groups no slot uses. The harness itself, the
monitors and the fillers never run on a slot's group while a window is timed (they are pinned to the spare groups).
Per window, in this order:
  1. the block-device read counters (/proc/diskstats: reads and sectors of every whole disk) and /proc/vmstat's
     pgmajfault and pswpin;
  2. the Windows-side monitor starts (WSL interop, powershell.exe on the host: the processor's % Processor Performance
     and Processor Frequency, so the actual MHz, once a second; the hypervisor's per-logical-processor run time when the
     account may read it);
  3. the reference segment "before": the reference process (plain PM per block on validation chunk 0, d = 9, by
     `timing.cpu_term ref --plain`) on every slot of the window at once, then a 0.1 s clock probe on each slot;
  4. the window's timed processes, one per slot, all started at once; a slot whose process ends early runs a filler
     (an untimed plain-PM loop) until the last timed process ends, so every timed process runs at the same load level;
  5. the reference segment "after" (as 3);
  6. the monitor stops; the counters again.
The window passes when (a) no block-device read and no major fault or swap-in happened between 1 and 6, (b) every slot's
reference value before and after is at most 1.10 x the session reference, and the largest value of each segment is at
most 1.06 x its smallest, and (c) every timed process exited with code 0. The record keeps every number, so a reader can
apply other limits.
The session reference (`sessionref`) is the median over all slots of R reference segments at full concurrency (8 slots).

    python harness.py topo [--cfg C] [--out F.json] the topology, the slots, the host (the Windows side, or the Linux host)
    python harness.py disk                         the block-device and fault counters now
    python harness.py winpath '%USERPROFILE%\\X'    a Windows folder as a WSL path (through a live interop socket)
    python harness.py import --cfg C --units U1,U2 --log RUN/handoff [--wait S]    hand-off units into the tmpfs now
    python harness.py freq --secs S                the Windows-side monitor for S seconds (a test)
    python harness.py sessionref --cfg C --out DIR [--repeats 3]
    python harness.py run --cfg C --plan P --out RUN/win --sessionref RUN/sref/sessionref.json [--only w001,w002]
                          [--no-fill] [--handoff-from DIR] [--max-attempts N] [--prior OLD/win,...]
Each finished window (and topo.json, plan.json, sref/) is copied to the Windows folder save_to\\<session>\\ (the
run directory's name), where the fetcher (fetch.sh) reads it. Hand-off units (handoff_from\\<family>\\ with DONE and
SHA256SUMS, STAGE2_common C2) are copied into {handoff} and checked at window boundaries; a window starts once the
units its jobs need are in. A rejected window's processes and failed processes run again (at most 3 attempts).
The config (JSON): root, python, cwd and argv of the reference process (placeholders {py} {core} {out} {rep} {root}),
filler argv, slots ("auto" or a list of vCPUs), limits. The plan (JSON): {"windows": [{"id", "jobs": [{"id", "part",
"side", "argv", "cwd", "env"}]}]}; job k of a window runs on slot k, so the plan sets the interleaving. Placeholders in
a job's argv: {py} {core} {out} (the job's own output directory) {root}.
Every record is written once (mode "x"): a rerun cannot replace a window unseen.

The Linux host (config "host": "linux", cfg_box.json): no WSL and no Windows step. A slot is one CPU of each of the 8
physical cores of one CCD (the CPUs sharing an L3 cache, "timed_ccd"), read from /sys; each core's SMT sibling stays
idle, and the harness, its monitors and the spare clock probe run on the other CCD. The disk rule becomes the window's
own processes': the harness's read_bytes (/proc/self/io, which counts every child it has reaped and theirs) must not
grow between the window's first and last reading; each process's own block reads (wait4 rusage, its children
included) are kept beside it, and the box's /proc/diskstats (other users' I/O, which does not count) is recorded as
disk_box_delta. The monitor records, per window, the cpufreq governor, EPP and boost, each CPU's frequency every 2 s,
the load average, each CPU's busy share, and the CPU seconds of every process that is not the harness or one of its
descendants. Saving and the hand-off use local folders (save_to, handoff_from: plain paths, ~ expanded)."""
import argparse
import contextlib
import glob
import json
import os
import re
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time

LIMITS = dict(ref_rel=1.10, ref_spread=1.06, slots_max=8)
PS = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
DISK_RE = re.compile(r"^(sd[a-z]+|vd[a-z]+|xvd[a-z]+|nvme\d+n\d+|pmem\d+)$")


# ---------------------------------------------------------------------------------------------------------- the host
def cpus(text):
    out = set()
    for part in text.strip().split(","):
        if part:
            a, _, b = part.partition("-")
            out.update(range(int(a), int(b or a) + 1))
    return out


def groups():
    """The VM's SMT sibling groups (cores), each a sorted tuple of vCPUs, sorted by their lowest vCPU."""
    seen = {}
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/topology/thread_siblings_list"):
        seen[tuple(sorted(cpus(open(p).read())))] = 1
    return sorted(seen)


def pick_slots(n=8):
    """One vCPU per core: the lowest vCPU of n cores, skipping core 0's group (the kernel's housekeeping and the
    harness). The harness and monitors use the last group. Returns (slots, spare vCPUs, groups)."""
    g = groups()
    if len(g) < n + 2:
        raise SystemExit(f"only {len(g)} cores visible; need {n + 2}")
    use = g[1:n + 1]
    spare = sorted(set().union(*g[n + 1:]))
    return [c[0] for c in use], spare, g


HOST = dict(mode="wsl", ccd=0)          # from the config (set_host): "linux" = a Linux machine, no WSL, no Windows
HZ = os.sysconf("SC_CLK_TCK")


def set_host(cfg):
    HOST.update(mode=cfg.get("host", "wsl"), ccd=int(cfg.get("timed_ccd", 0)))


def ccds():
    """The CPUs that share an L3 cache (on AMD, one CCD), from /sys; sorted tuples, sorted by their lowest CPU."""
    seen = {}
    for p in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cache/index3/shared_cpu_list"):
        seen[tuple(sorted(cpus(open(p).read())))] = 1
    return sorted(seen)


def box_slots(n=8, ccd=0):
    """The Linux host: the lowest CPU of each of n physical cores of CCD `ccd`. The cores' SMT siblings (and the CCD's
    other cores) stay idle; the harness, its monitors and the clock samples use the other CCDs. Returns (slots, spare
    CPUs, groups)."""
    g, dom = groups(), set(ccds()[ccd])
    cores = [c for c in g if set(c) <= dom]
    if len(cores) < n:
        raise SystemExit(f"CCD {ccd} has {len(cores)} cores; need {n}")
    return [c[0] for c in cores[:n]], sorted(set().union(*g) - dom), g


def slots_for(n=8):
    return box_slots(n, HOST["ccd"]) if HOST["mode"] == "linux" else pick_slots(n)


def ranges(xs):
    """[0, 1, 2, 5] -> '0-2,5'."""
    out = []
    for x in sorted(set(xs)):
        if out and x == out[-1][1] + 1:
            out[-1][1] = x
        else:
            out.append([x, x])
    return ",".join(str(a) if a == b else f"{a}-{b}" for a, b in out)


def rd(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except OSError:
        return None


def online_cpus():
    """The CPUs online now (/sys/devices/system/cpu/online; an offline SMT sibling is left out)."""
    return sorted(cpus(rd("/sys/devices/system/cpu/online") or "")) or list(range(os.cpu_count()))


def io_of(pid):
    """read_bytes of /proc/<pid>/io: what the process, and every child it reaped, caused to be read from a block
    device."""
    for ln in (rd(f"/proc/{pid}/io") or "").splitlines():
        if ln.startswith("read_bytes:"):
            return int(ln.split()[1])
    return None


def own_io():
    """Our block reads so far: the harness's read_bytes (its threads, and every child it has reaped with theirs) and
    the reaped children's rusage (block input operations, major faults)."""
    s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return dict(t=time.time(), read_bytes=io_of(os.getpid()), inblock=s.ru_inblock + c.ru_inblock,
                majflt=s.ru_majflt + c.ru_majflt)


def own_delta(a, b):
    """The Linux host's disk rule: the window's own processes (the harness, the reference and timed processes, the
    fillers and the clock probes, all reaped before the second reading) read nothing from a block device, i.e.
    read_bytes did not grow. reads: block input operations (512-byte units); pswpin is not per process (a swap-in by
    one of ours is a read in read_bytes)."""
    rb = None if None in (a["read_bytes"], b["read_bytes"]) else b["read_bytes"] - a["read_bytes"]
    return dict(rule="own read_bytes", read_bytes=rb, reads=b["inblock"] - a["inblock"],
                sectors=None if rb is None else rb // 512, pgmajfault=b["majflt"] - a["majflt"], pswpin=None,
                passed=rb == 0)


def cpu_times():
    """Per CPU (busy, total) jiffies from /proc/stat (busy: user, nice, system, irq, softirq, steal)."""
    out = {}
    for ln in open("/proc/stat"):
        if ln.startswith("cpu") and ln[3].isdigit():
            f = ln.split()
            v = [int(x) for x in f[1:9]]
            out[int(f[0][3:])] = (v[0] + v[1] + v[2] + v[5] + v[6] + v[7], sum(v))
    return out


def freq_state():
    """The frequency settings now: per setting {value: CPUs} (cpufreq driver, governor, energy-performance preference,
    limits in kHz), the boost switch and the amd-pstate mode."""
    out = dict(online=rd("/sys/devices/system/cpu/online"), boost=rd("/sys/devices/system/cpu/cpufreq/boost"),
               amd_pstate=rd("/sys/devices/system/cpu/amd_pstate/status"))
    for k in ("scaling_driver", "scaling_governor", "energy_performance_preference", "scaling_min_freq",
              "scaling_max_freq"):
        by = {}
        for c in online_cpus():
            by.setdefault(str(rd(f"/sys/devices/system/cpu/cpu{c}/cpufreq/{k}")), []).append(c)
        out[k] = {v: ranges(c) for v, c in by.items()}
    return out


def proc_scan():
    """Every process now: pid -> (ppid, uid, comm, utime + stime in ticks, the CPU it last ran on, its start in ticks
    since boot)."""
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat") as fh:
                    s = fh.read()
                uid = os.stat(f"/proc/{d}").st_uid
            except OSError:
                continue
            i = s.rindex(")")
            f = s[i + 2:].split()
            out[int(d)] = (int(f[1]), uid, s[s.index("(") + 1:i], int(f[11]) + int(f[12]), int(f[36]), int(f[19]))
    return out


USERS = {}


def user_name(uid):
    if uid not in USERS:
        import pwd
        try:
            USERS[uid] = pwd.getpwuid(uid).pw_name
        except KeyError:
            USERS[uid] = str(uid)
    return USERS[uid]


def linux_facts():
    """The Linux host for topo.json: CPU, OS, kernel command line, frequency settings, CCDs, SMT, the load, the users
    logged in and the processes busy over 1 s (any user's)."""
    f = dict(source="linux", cpu=next((ln.split(":", 1)[1].strip() for ln in open("/proc/cpuinfo")
                                       if ln.startswith("model name")), None),
             os=next((ln.split("=", 1)[1].strip().strip('"') for ln in (rd("/etc/os-release") or "").splitlines()
                      if ln.startswith("PRETTY_NAME=")), None),
             kernel_cmdline=rd("/proc/cmdline"), cpufreq=freq_state(), ccds=[ranges(c) for c in ccds()],
             smt_active=rd("/sys/devices/system/cpu/smt/active"), loadavg=rd("/proc/loadavg"))
    try:
        who = subprocess.run(["who"], capture_output=True, text=True, timeout=10).stdout
        f["users_logged_in"] = sorted({ln.split()[0] for ln in who.splitlines() if ln.strip()})
    except (OSError, subprocess.SubprocessError):
        f["users_logged_in"] = None
    a = proc_scan()
    time.sleep(1.0)
    b = proc_scan()
    busy = [dict(pid=p, user=user_name(r[1]), comm=r[2], cpu_share=round((r[3] - a[p][3]) / HZ, 2), cpu=r[4])
            for p, r in b.items() if p in a and a[p][5] == r[5] and r[3] - a[p][3] >= 0.05 * HZ]
    f["busy_1s"] = sorted(busy, key=lambda x: -x["cpu_share"])
    f["processes"] = len(b)
    return f


class BoxMonitor:
    """The Linux host's monitor for the length of a window, in a thread of the harness (on the spare CPUs). Every 2 s:
    the clock probe on the last spare CPU, every CPU's current frequency (cpufreq scaling_cur_freq), the load average
    (a line per sample in the monitor file) and every process's CPU time. At the end: the frequency settings at both
    ends (governor, EPP, boost), each CPU's busy share over the window (/proc/stat), the slots' SMT siblings among them,
    and the CPU seconds of every process that is not ours (ours: the harness and its descendants), with the CPUs it
    was seen running on."""

    def __init__(self, path, cfg, slots):
        import threading
        self.fh = open(path, "x")
        self.spare, self.slots = sorted(cfg["spare"]), list(slots)
        self.sib = sorted(set().union(*[set(g) for g in groups() if set(g) & set(slots)]) - set(slots))
        self.me, self.ours, self.seen, self.clk = os.getpid(), set(), {}, []
        self.up0 = float(rd("/proc/uptime").split()[0]) * HZ
        self.gov0, self.cpu0, self.load = freq_state(), cpu_times(), []
        self.freq = {c: [] for c in online_cpus()}
        self._sample()
        self.stop_ev = threading.Event()
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def _sample(self):
        rows, kids = proc_scan(), {}
        for pid, r in rows.items():
            kids.setdefault(r[0], []).append(pid)
        todo = [self.me]
        while todo:                                                   # the harness's descendants are ours
            for k in kids.get(todo.pop(), []):
                if k not in self.ours:
                    self.ours.add(k)
                    todo.append(k)
        for pid, (_, uid, comm, ticks, cpu, start) in rows.items():
            if pid == self.me or pid in self.ours:
                continue
            e = self.seen.get((pid, start))
            if e is None:                                             # born in the window: all its time counts
                e = self.seen[(pid, start)] = dict(uid=uid, comm=comm, base=0 if start > self.up0 else ticks,
                                                   last=None, cpus=set())
            if ticks > (e["base"] if e["last"] is None else e["last"]):
                e["cpus"].add(cpu)
            e["last"] = ticks
        mhz = {}
        for c in self.freq:
            v = rd(f"/sys/devices/system/cpu/cpu{c}/cpufreq/scaling_cur_freq")
            if v:
                self.freq[c].append(int(v) / 1000)
                mhz[c] = int(v) // 1000
        la = rd("/proc/loadavg")
        self.load.append(la)
        self.fh.write(json.dumps(dict(t=round(time.time(), 2), loadavg=la, mhz=mhz)) + "\n")

    def _loop(self):
        while not self.stop_ev.wait(2.0):
            self.clk.append(clock_probe(self.spare[-1], 10_000_000))
            self._sample()

    def stop(self):
        self.stop_ev.set()
        self.t.join()
        self._sample()
        self.fh.close()
        c1 = cpu_times()
        busy = {c: round((c1[c][0] - self.cpu0[c][0]) / max(1, c1[c][1] - self.cpu0[c][1]), 3) for c in c1 if c in self.cpu0}
        others = sorted((dict(pid=p, user=user_name(e["uid"]), comm=e["comm"], cpu_s=round((e["last"] - e["base"]) / HZ, 2),
                              cpus=ranges(e["cpus"])) for (p, _), e in self.seen.items() if e["last"] > e["base"]),
                        key=lambda x: -x["cpu_s"])
        c = [x for x in self.clk if x]
        out = dict(host="linux", win_available=False, clk_samples=len(c), samples=len(self.load),
                   file=os.path.basename(self.fh.name))
        if c:
            out.update(clk_mhz_mean=round(statistics.fmean(c), 1), clk_mhz_min=min(c), clk_mhz_max=max(c))
        sl = [x for s in self.slots for x in self.freq[s]]
        if sl:                                                        # the slots' cpufreq MHz (APERF/MPERF based)
            out.update(mhz_mean=round(statistics.fmean(sl), 1), mhz_min=round(min(sl), 1), mhz_max=round(max(sl), 1))
        out["slot_mhz"] = {s: [round(statistics.fmean(self.freq[s]), 1), min(self.freq[s]), max(self.freq[s])]
                           for s in self.slots if self.freq[s]}
        one = [float(x.split()[0]) for x in self.load if x]
        out["loadavg"] = dict(start=self.load[0], end=self.load[-1], max_1min=max(one) if one else None)
        out["governor_start"], out["governor_end"] = self.gov0, freq_state()
        out["cpu_busy"] = busy
        out["sibling_cpus"] = ranges(self.sib)
        out["sibling_busy_max"] = max((busy[s] for s in self.sib if s in busy), default=None)
        out["not_ours_cpu_s"] = round(sum(x["cpu_s"] for x in others), 2)
        out["not_ours"] = others
        return out


def monitor_for(cfg, path, slots):
    if HOST["mode"] == "linux":
        return BoxMonitor(path, cfg, slots)
    return Monitor(path, set(cfg["spare"]), cfg.get("win_counters"))


def window_io(rec, start):
    """The disk readings at a window's (or a session reference's) two ends: the box's counters always; on the Linux
    host also our own, which decide (disk_delta), the box's being kept as disk_box_delta."""
    if start:
        rec["disk0"] = disk()
        if HOST["mode"] == "linux":
            rec["io0"] = own_io()
        return
    rec["disk1"] = disk()
    if HOST["mode"] == "linux":
        rec["io1"] = own_io()
        rec["disk_box_delta"] = disk_delta(rec["disk0"], rec["disk1"])
        rec["disk_delta"] = own_delta(rec["io0"], rec["io1"])
    else:
        rec["disk_delta"] = disk_delta(rec["disk0"], rec["disk1"])


def lscpu():
    try:
        return subprocess.run(["lscpu"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def ps_argv(script):
    """powershell.exe with the script as -EncodedCommand (UTF-16LE, base64): no quoting, non-ASCII names survive."""
    import base64
    return [PS, "-NoProfile", "-NonInteractive", "-EncodedCommand", base64.b64encode(script.encode("utf-16-le")).decode()]


def interop():
    """A working WSL interop socket. A detached run outlives the login session that started it; that session's socket
    (WSL_INTEROP) stays but no longer reaches Windows, while the wsl.exe that keeps the distro up holds a working one
    under /run/WSL. Each candidate is tried with `cmd.exe /c echo ok` (8 s at most)."""
    cmd = "/mnt/c/Windows/System32/cmd.exe"
    cands = [os.environ.get("WSL_INTEROP", "")] + sorted(glob.glob("/run/WSL/*_interop"), key=os.path.getmtime)
    for c in dict.fromkeys(x for x in cands if x):
        try:
            r = subprocess.run([cmd, "/c", "echo", "ok"], capture_output=True, timeout=8, cwd="/mnt/c",
                               env=dict(os.environ, WSL_INTEROP=c))
        except (OSError, subprocess.SubprocessError):
            continue
        if b"ok" in r.stdout:
            os.environ["WSL_INTEROP"] = c
            return c
    return None


def ps_run(script, timeout=60):
    """Runs a PowerShell script on the Windows host through WSL interop; returns stdout ('' when interop is off)."""
    if not os.path.exists(PS):
        return ""
    if not os.environ.get("SRT_INTEROP_OK"):
        os.environ["SRT_INTEROP_OK"] = interop() or "none"
    try:
        r = subprocess.run(ps_argv(script), capture_output=True, timeout=timeout)
        return r.stdout.decode("utf-8", "replace")
    except (OSError, subprocess.SubprocessError):
        return ""


CMD = "/mnt/c/Windows/System32/cmd.exe"


def cmd_line(line, timeout=60):
    """Runs a cmd.exe line on the Windows host through WSL interop; returns stdout ('' when interop is off). On this
    host powershell.exe started through interop exits with code 66 and no output, while cmd.exe works."""
    if not os.path.exists(CMD):
        return ""
    if not os.environ.get("SRT_INTEROP_OK"):
        os.environ["SRT_INTEROP_OK"] = interop() or "none"
    try:
        r = subprocess.run([CMD, "/c", line], capture_output=True, timeout=timeout, cwd="/mnt/c",
                           stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout.decode("gbk", "replace").replace("\r", "")


def win_facts_cmd():
    """The Windows facts through cmd.exe: the active power plan, the CPU, the OS version, .wslconfig, the VM and WSL
    processes on the host (tasklist; other users' VMs show as vmmem) and the logged-on users."""
    f = dict(source="cmd.exe")
    f["power_plan"] = cmd_line("powercfg /getactivescheme").strip()
    f["cpu"] = cmd_line('reg query "HKLM\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0" /v ProcessorNameString').strip()
    f["os"] = cmd_line("ver").strip()
    prof = win_dir("%USERPROFILE%")
    try:
        f["wslconfig"] = open(os.path.join(prof, ".wslconfig")).read() if prof else None
    except OSError:
        f["wslconfig"] = None
    tl = cmd_line("tasklist /fo csv /v", 120).splitlines()
    f["processes"] = len(tl)
    f["vm_processes"] = [x for x in tl if re.search(r"vmmem|vmwp|wsl|qemu|virtualbox", x, re.I)][:40]
    f["users"] = cmd_line("query user").strip()[-2000:]
    return f


def win_counters():
    """The first counter-path set Get-Counter accepts on this host (English, then the zh-CN names), or None."""
    for paths in WIN_COUNTER_SETS:
        q = ",".join("'" + c + "'" for c in paths)
        out = ps_run("[Console]::OutputEncoding = [Text.Encoding]::UTF8; "
                     f"(Get-Counter -Counter @({q}) -MaxSamples 1 -ErrorAction Stop).CounterSamples.Count", 30)
        if out.strip().splitlines()[-1:] == ["3"]:
            return paths
    return None


WIN_FACTS = r"""
$ErrorActionPreference = 'SilentlyContinue'
$o = [ordered]@{}
$cpu = Get-ItemProperty 'HKLM:\HARDWARE\DESCRIPTION\System\CentralProcessor\0'
$nt = Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion'
$o.cpu = $cpu.ProcessorNameString.Trim(); $o.nominal_mhz = $cpu.'~MHz'; $o.logical = [Environment]::ProcessorCount
$o.windows = $nt.ProductName; $o.build = "$($nt.CurrentBuild).$($nt.UBR)"
Add-Type -AssemblyName Microsoft.VisualBasic
$ci = New-Object Microsoft.VisualBasic.Devices.ComputerInfo
$o.ram_gb = [math]::Round($ci.TotalPhysicalMemory / 1GB, 1); $o.available_gb = [math]::Round($ci.AvailablePhysicalMemory / 1GB, 1)
$e = Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Hyper-V-Hypervisor'; Id=2} -MaxEvents 1
$o.hv_scheduler_event = if ($e) { $e.Message } else { 'not readable' }
$o.power_plan = (powercfg /getactivescheme) -join ' '
$o.wslconfig = if (Test-Path "$env:USERPROFILE\.wslconfig") { (Get-Content "$env:USERPROFILE\.wslconfig") -join ';' } else { $null }
$o | ConvertTo-Json -Compress
"""

# Windows counters through PDH (Get-Counter); a non-administrator login may not use WMI. English counter paths; the config's
# "win_counters" can name them; otherwise load_cfg takes the first set Get-Counter accepts (win_counters()).
WIN_COUNTER_SETS = [[r"\Processor Information(_Total)\% Processor Performance",
                     r"\Processor Information(_Total)\Processor Frequency",
                     r"\Processor Information(_Total)\% Processor Time"],
                    [r"\处理器信息(_Total)\% 处理器性能", r"\处理器信息(_Total)\处理器频率", r"\处理器信息(_Total)\% 处理器时间"]]
MONITOR = r"""
$ErrorActionPreference = 'SilentlyContinue'
Get-Counter -Counter @(PATHS) -SampleInterval 1 -Continuous | ForEach-Object {
  [Console]::Out.WriteLine((@{t = $_.Timestamp.ToString('o'); v = @($_.CounterSamples | ForEach-Object { $_.CookedValue })} | ConvertTo-Json -Compress))
  [Console]::Out.Flush()
}
"""


class Monitor:
    """For the length of a window: (a) the Windows counters (WSL interop, powershell.exe, on a spare vCPU): % Processor
    Performance x Processor Frequency = the package's actual MHz, and the host's busy share; (b) the clock probe on the
    last spare vCPU every 2 s (the effective MHz a core gets at that moment, from inside the VM)."""

    def __init__(self, path, spare, counters=None):
        import threading
        self.path, self.p, self.spare = path, None, sorted(spare)
        self.clk, self.stop_ev = [], threading.Event()
        if os.path.exists(PS) and counters:
            self.fh = open(path, "x")
            paths = ",".join("'" + c + "'" for c in counters)
            self.p = subprocess.Popen(ps_argv(MONITOR.replace("PATHS", paths)),
                                      stdout=self.fh, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                      preexec_fn=lambda: os.sched_setaffinity(0, set(self.spare[:-1])))
        self.t = threading.Thread(target=self._clk, daemon=True)
        self.t.start()

    def _clk(self):
        while not self.stop_ev.wait(2.0):
            self.clk.append(clock_probe(self.spare[-1], 10_000_000))

    def stop(self):
        self.stop_ev.set()
        self.t.join()
        c = [x for x in self.clk if x]
        out = dict(clk_samples=len(c))
        if c:
            out.update(clk_mhz_mean=round(statistics.fmean(c), 1), clk_mhz_min=min(c), clk_mhz_max=max(c))
        if self.p is None:
            return dict(out, win_available=False)
        self.p.terminate()
        try:
            self.p.wait(10)
        except subprocess.TimeoutExpired:
            self.p.kill()
        self.fh.close()
        rows = []
        for line in open(self.path):
            try:
                rows.append(json.loads(line)["v"])
            except (ValueError, KeyError):
                pass
        rows = [r for r in rows if isinstance(r, list) and len(r) >= 3]
        out.update(win_available=bool(rows), win_samples=len(rows), file=os.path.basename(self.path))
        if rows:
            mhz = [r[0] * r[1] / 100 for r in rows]
            out.update(mhz_mean=round(statistics.fmean(mhz), 1), mhz_min=round(min(mhz), 1),
                       mhz_max=round(max(mhz), 1), host_busy_mean=round(statistics.fmean(r[2] for r in rows), 2))
        return out


def clock_probe(core, n=20_000_000):
    """The effective clock of one vCPU: a chain of dependent adds (4 per iteration, one cycle each) for about 0.03 s,
    pinned to the vCPU. Needs the compiled probe (clk) beside this file; returns MHz or None."""
    exe = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clk")
    if not os.path.exists(exe):
        return None
    r = subprocess.run([exe, str(n)], capture_output=True, text=True,
                       preexec_fn=lambda: os.sched_setaffinity(0, {core}))
    try:
        return float(r.stdout.strip())
    except ValueError:
        return None


def disk():
    """Reads completed and sectors read, summed over the whole disks; major faults and swap-ins."""
    reads = sectors = 0
    per = {}
    for line in open("/proc/diskstats"):
        f = line.split()
        if DISK_RE.match(f[2]):
            reads += int(f[3])
            sectors += int(f[5])
            per[f[2]] = [int(f[3]), int(f[5])]
    vm = dict(line.split() for line in open("/proc/vmstat"))
    return dict(t=time.time(), reads=reads, sectors=sectors, per_disk=per, pgmajfault=int(vm.get("pgmajfault", 0)),
                pswpin=int(vm.get("pswpin", 0)))


def disk_delta(a, b):
    d = {k: b[k] - a[k] for k in ("reads", "sectors", "pgmajfault", "pswpin")}
    d["passed"] = all(v == 0 for v in d.values())
    return d


def host_record():
    mem = dict(line.split(":", 1) for line in open("/proc/meminfo"))
    slots, spare, g = slots_for()
    return dict(mode=HOST["mode"], kernel=os.uname().release, vcpus=os.cpu_count(), groups=[list(x) for x in g],
                ccds=[list(x) for x in ccds()],
                threads_per_core=max(len(x) for x in g), slots=slots, spare=spare,
                mem_total=mem["MemTotal"].strip(), swap_total=mem["SwapTotal"].strip(),
                shm=shutil.disk_usage("/dev/shm")._asdict(), lscpu=lscpu())


# -------------------------------------------------------------------------------------------------------- processes
def subst(cfg, x, **kw):
    """Fills the placeholders of a string, list or dict: {py} {root} {sessionref_us}, the config's vars ({code} {data}
    {relay}) and kw ({core} {out} {rep})."""
    v = dict(cfg.get("vars", {}), py=cfg["python"], root=cfg["root"], sessionref_us=cfg.get("sessionref_us", ""))
    v.update(kw)
    if isinstance(x, list):
        return [str(e).format(**v) for e in x]
    if isinstance(x, dict):
        return {k: str(e).format(**v) for k, e in x.items()}
    return str(x).format(**v)


def job_env(cfg, e=None):
    """The config's env for every process (the tmpfs C libraries, locale and zone: no file from the disk) and the job's."""
    return dict(subst(cfg, cfg.get("env", {})), **subst(cfg, e or {}))


def env_for(extra=None):
    e = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1",
             RAYON_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    for k in ("SRT_BEACON", "SRT_DRY", "SRT_DRY_SHOTS", "SRT_ORIGINAL_SEEDS"):   # the draw's switches (as cpu_item.sh)
        e.pop(k, None)
    e.update(extra or {})
    return e


def launch(argv, cwd, out, core, env=None):
    os.makedirs(out, exist_ok=True)
    fo = open(os.path.join(out, "stdout.txt"), "x")
    fe = open(os.path.join(out, "stderr.txt"), "x")
    p = subprocess.Popen(argv, cwd=cwd, stdout=fo, stderr=fe, stdin=subprocess.DEVNULL, env=env_for(env),
                         preexec_fn=lambda: os.sched_setaffinity(0, {core}), start_new_session=True)
    p._files = (fo, fe)
    p._t0 = time.time()
    return p


def exited(p, block=False):
    """Whether p has ended; it is reaped with wait4, whose rusage gives its block reads (ru_inblock, 512-byte units,
    the children it reaped included) and major faults."""
    if p.returncode is not None:
        return True
    try:
        pid, sts, ru = os.wait4(p.pid, 0 if block else os.WNOHANG)
    except ChildProcessError:
        return p.poll() is not None
    if pid == 0:
        return False
    p.returncode = os.waitstatus_to_exitcode(sts)
    p._io = dict(read_bytes=ru.ru_inblock * 512, majflt=ru.ru_majflt)
    return True


def finish(p):
    for f in p._files:
        f.close()
    return dict(rc=p.returncode, wall_s=round(time.time() - p._t0, 3), **getattr(p, "_io", {}))


def segment(cfg, slots, out, tag):
    """The reference process on every slot at once; returns per slot (vCPU, per_block_us, rc) plus the clock probes."""
    procs = []
    for k, core in enumerate(slots):
        d = os.path.join(out, f"{tag}_s{k}")
        r = cfg["ref"]
        procs.append((k, core, d, launch(subst(cfg, r["argv"], core=core, out=d, rep=1), subst(cfg, r["cwd"]), d, core,
                                         job_env(cfg, r.get("env")))))
    res = []
    for k, core, d, p in procs:
        exited(p, block=True)
        r = finish(p)
        v = None
        for f in glob.glob(os.path.join(d, "*.json")):
            v = json.load(open(f)).get("per_block_us", v)
        res.append(dict(slot=k, core=core, per_block_us=v, **r))
    clocks = [clock_probe(c) for c in slots]
    for r, c in zip(res, clocks):
        r["clock_mhz"] = c
    return res


def check_segment(seg, ref):
    v = [r["per_block_us"] for r in seg]
    if any(x is None for x in v) or any(r["rc"] != 0 for r in seg):
        return dict(passed=False, why="a reference process failed")
    worst = max(v) / ref
    spread = max(v) / min(v)
    return dict(passed=worst <= LIMITS["ref_rel"] and spread <= LIMITS["ref_spread"], worst_rel=round(worst, 4),
                spread=round(spread, 4), median_us=statistics.median(v))


def has_record(d):
    """A process left a record of its own: a file in its directory besides stdout.txt and stderr.txt."""
    return any(f not in ("stdout.txt", "stderr.txt") for f in os.listdir(d)) if os.path.isdir(d) else False


# ----------------------------------------------------------------------------- saving and the hand-off (STAGE2_common)
def win_dir(spec):
    """A Windows folder such as %USERPROFILE%\\speedrt_out as a WSL path (/mnt/c/...); '' when interop is off."""
    if not spec:
        return ""
    if "%USERPROFILE%" in spec:
        prof = cmd_line("echo %USERPROFILE%").strip()
        if not prof:
            return ""
        spec = spec.replace("%USERPROFILE%", prof)
    r = subprocess.run(["wslpath", "-u", spec], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def folder(spec):
    """A save or hand-off folder: a Windows one (%USERPROFILE%\\..., C:\\...) as its WSL path; else a local path."""
    return win_dir(spec) if ("%" in spec or "\\" in spec) else os.path.expanduser(spec)


def save(cfg, path):
    """Copies a finished record (a file or directory of the run directory) to the Windows folder
    <save_to>\\<session>\\, the only place the fetcher reads; called between windows, outside every check's span."""
    if not cfg.get("save_dir") or not os.path.exists(path):
        return
    rel = os.path.relpath(path, cfg["run_dir"])
    dst = os.path.join(cfg["save_dir"], cfg["session"], rel)
    try:
        if os.path.isdir(path):
            shutil.copytree(path, dst, dirs_exist_ok=True, copy_function=shutil.copyfile)
        else:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copyfile(path, dst)
    except OSError as e:                                             # costs safekeeping only, never the run
        print(f"save {rel}: {e}", flush=True)


def sha256_file(p):
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def handoff_ready(src):
    """The units whose marker exists in the hand-off folder: a folder U holding DONE is unit U (checksums
    U/SHA256SUMS); a file NAME_DONE in folder P makes unit P/name (checksums P/name/name_SHA256SUMS), e.g. bb/DRAW_DONE
    makes bb/draw. Returns {unit: checksum file relative to src}."""
    units = {}
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        for f in files:
            if f == "DONE" and rel != ".":
                units[rel] = os.path.join(rel, "SHA256SUMS")
            elif f.endswith("_DONE"):
                name = f[:-5].lower()
                u = rel if os.path.basename(rel) == name else os.path.normpath(os.path.join(rel, name))
                units[u] = os.path.join(u, f"{name}_SHA256SUMS")
    return units


def handoff_import(cfg, log, only=None):
    """At a window boundary: copies every ready unit not yet in the tmpfs from the Windows folder to {handoff}/<unit>/
    and checks each file its checksum file names (lines "sha256  ./path", relative to that file's folder). A unit is
    used only when every file matches; otherwise it is tried again at the next boundary. Each try is recorded in
    RUN/handoff/."""
    src, dst = cfg.get("handoff_dir"), cfg["vars"].get("handoff")
    if not src or not dst or not os.path.isdir(src):
        return
    for unit, sums in sorted(handoff_ready(src).items()):
        if unit in cfg["imported"] or (only and unit not in only):
            continue
        rec = dict(unit=unit, start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), checksums=sums, ok=False)
        try:
            shutil.copytree(os.path.join(src, unit), os.path.join(dst, unit), dirs_exist_ok=True,
                            copy_function=shutil.copyfile)
            base = os.path.dirname(os.path.join(dst, sums))
            lines = [ln.split(None, 1) for ln in open(os.path.join(dst, sums)) if ln.strip()]
            bad = [n.strip().lstrip("*") for h, n in lines
                   if not os.path.isfile(os.path.join(base, n.strip().lstrip("*")))
                   or sha256_file(os.path.join(base, n.strip().lstrip("*"))) != h]
            for n in bad:                                            # copied again at the next boundary
                with contextlib.suppress(OSError):
                    os.remove(os.path.join(base, n))
            rec.update(files=len(lines), bad=bad, ok=bool(lines) and not bad)
        except (OSError, ValueError) as e:
            rec["error"] = str(e)[:300]
        rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        os.makedirs(log, exist_ok=True)
        n = len(glob.glob(os.path.join(log, unit.replace("/", "_") + "_*.json"))) + 1
        p = os.path.join(log, f"{unit.replace('/', '_')}_{n}.json")
        with open(p, "x") as fh:
            json.dump(rec, fh, indent=1)
        save(cfg, p)
        print(f"hand-off {unit}: {'in the tmpfs' if rec['ok'] else 'FAILED'} ({rec.get('files')} files, "
              f"{len(rec.get('bad', []))} mismatched) {rec.get('error', '')}", flush=True)
        if rec["ok"]:
            cfg["imported"].add(unit)


# --------------------------------------------------------------------------------------------------------- commands
def cmd_topo(a):
    if a.cfg:
        set_host(json.load(open(a.cfg)))
    rec = dict(host=host_record())
    if HOST["mode"] == "linux":
        rec["linux"] = linux_facts()
    else:
        w = ps_run(WIN_FACTS)
        try:
            rec["windows"] = json.loads(w)
        except ValueError:
            rec["windows"] = win_facts_cmd()
    rec["disk"] = disk()
    text = json.dumps(rec, indent=1)
    if a.out:
        with open(a.out, "x") as fh:
            fh.write(text + "\n")
    h = rec["host"]
    print(json.dumps(dict(vcpus=h["vcpus"], threads_per_core=h["threads_per_core"], groups=h["groups"][:4],
                          slots=h["slots"], spare=h["spare"], mem=h["mem_total"], swap=h["swap_total"],
                          **({"linux": {k: rec["linux"][k] for k in ("cpu", "os", "ccds", "loadavg", "busy_1s")}}
                             if "linux" in rec else
                             {"windows": {k: v for k, v in rec["windows"].items() if k not in ("other_users_processes",)}}))))


def cmd_import(a):
    """Copies the named hand-off units into the tmpfs and checks them, as `run` does at a window boundary, waiting up
    to --wait seconds for their markers; for the shots, which go in before the session (prep.sh shots). It exits
    non-zero when a unit is still missing."""
    cfg = load_cfg(a.cfg)
    session_of(cfg, a.log)
    if a.handoff_from:
        cfg["handoff_dir"] = folder(a.handoff_from)
    want, t0 = set(a.units.split(",")), time.time()
    while True:
        handoff_import(cfg, a.log, want)
        if all(covered(u, cfg["imported"]) for u in want):
            return
        if time.time() - t0 > a.wait:
            sys.exit(f"hand-off units still missing after {a.wait} s: "
                     f"{sorted(u for u in want if not covered(u, cfg['imported']))}")
        time.sleep(30)


def cmd_winpath(a):
    print(win_dir(a.spec))


def cmd_disk(a):
    print(json.dumps(disk()))


def cmd_freq(a):
    _, spare, _ = pick_slots()
    c = win_counters()
    print("counters:", c)
    m = Monitor(f"/tmp/freq_test_{os.getpid()}.jsonl", set(spare), c)
    time.sleep(a.secs)
    print(json.dumps(m.stop()))


def load_cfg(path):
    cfg = json.load(open(path))
    set_host(cfg)
    LIMITS.update(cfg.get("limits", {}))
    if HOST["mode"] == "linux":
        cfg["win_counters"] = None
    else:
        cfg["win_counters"] = cfg.get("win_counters") or win_counters()     # None: the Windows counters are not readable
    if cfg.get("slots", "auto") == "auto":
        cfg["slots"], cfg["spare"], _ = slots_for(LIMITS["slots_max"])
    elif HOST["mode"] == "linux":                                    # the spare CPUs: the CCDs without a slot
        cfg["spare"] = sorted(set(online_cpus()) - set().union(*[d for d in ccds() if set(d) & set(cfg["slots"])]))
    else:
        cfg["spare"] = sorted(set(range(os.cpu_count())) - set().union(*[g for g in groups()
                                                                          if set(g) & set(cfg["slots"])]))
    os.sched_setaffinity(0, set(cfg["spare"]))                     # the harness never runs on a slot's core
    cfg["vars"] = cfg.get("vars", {})
    cfg["save_dir"] = folder(cfg.get("save_to", ""))
    cfg["handoff_dir"] = folder(cfg.get("handoff_from", ""))
    cfg["imported"] = set()
    return cfg


def session_of(cfg, out):
    """The run directory is the parent of a command's --out (RUN/sref, RUN/win); its name names the saved copy."""
    cfg["run_dir"] = os.path.dirname(os.path.abspath(out))
    cfg["session"] = os.path.basename(cfg["run_dir"])


def cmd_sessionref(a):
    cfg = load_cfg(a.cfg)
    session_of(cfg, a.out)
    os.makedirs(a.out, exist_ok=True)
    earlier = []                       # a reference that saw a disk read is taken again, at most --tries times in all
    for t in range(1, a.tries + 1):
        sfx = "" if t == 1 else f"_t{t}"
        rec = dict(what="session reference", slots=cfg["slots"], start=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   segments=[], attempt=t)
        window_io(rec, True)
        m = monitor_for(cfg, os.path.join(a.out, f"sessionref_monitor{sfx}.jsonl"), cfg["slots"])
        for r in range(a.repeats):
            rec["segments"].append(segment(cfg, cfg["slots"], a.out, f"sref{r}{sfx}"))
        rec["monitor"] = m.stop()
        window_io(rec, False)
        if rec["disk_delta"]["passed"] or t == a.tries:
            break
        earlier.append(dict(attempt=t, start=rec["start"], disk_delta=rec["disk_delta"], segments=rec["segments"]))
        print(f"session reference attempt {t}: disk {rec['disk_delta']}; taken again", flush=True)
    rec["earlier_attempts"] = earlier
    v = [x["per_block_us"] for s in rec["segments"] for x in s if x["per_block_us"] is not None]
    rec["reference_us"] = statistics.median(v)
    rec["spread"] = max(v) / min(v)
    rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(a.out, "sessionref.json"), "x") as fh:
        json.dump(rec, fh, indent=1)
    for f in ("topo.json", "plan.json", "sref"):
        save(cfg, os.path.join(cfg["run_dir"], f))
    print(f"session reference {rec['reference_us']:.3f} us per block over {len(v)} slot values, spread "
          f"{rec['spread']:.4f}; disk {rec['disk_delta']}; MHz win/clk {rec['monitor'].get('mhz_mean')}/{rec['monitor'].get('clk_mhz_mean')}")


def run_window(cfg, w, out, ref_us, use_fill=True):
    wd = os.path.join(out, w["id"])
    os.makedirs(wd)                                                  # a window is never run twice into one directory
    jobs = w["jobs"]
    assert len(jobs) <= min(LIMITS["slots_max"], len(cfg["slots"])), f"{w['id']}: too many jobs"
    slots = cfg["slots"][:len(jobs)]
    rec = dict(window=w["id"], rep=w.get("rep"), start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), slots=slots,
               needs=sorted(w.get("needs", [])), handoff_in=sorted(cfg.get("imported", ())))
    window_io(rec, True)
    mon = monitor_for(cfg, os.path.join(wd, "monitor.jsonl"), slots)
    rec["ref_before"] = segment(cfg, slots, wd, "ref_before")
    procs = []
    for k, (j, core) in enumerate(zip(jobs, slots)):
        jd = os.path.join(wd, f"s{k}_{j['id']}")
        procs.append((k, j, core, jd, launch(subst(cfg, j["argv"], core=core, out=jd), subst(cfg, j.get("cwd", "{root}")),
                                            jd, core, job_env(cfg, j.get("env")))))
    fillers, done, results = {}, set(), {}
    fl = cfg["filler"]
    while len(done) < len(procs):
        for k, j, core, jd, p in procs:
            if k not in done and exited(p):
                done.add(k)
                results[k] = dict(slot=k, core=core, job=j["id"], base=j.get("base", j["id"]), part=j.get("part"),
                                  side=j.get("side"), attempt=j.get("attempt", 1), argv=p.args, out=jd, **finish(p))
                results[k]["record"] = has_record(jd)
                if use_fill and len(done) < len(procs):
                    fillers[k] = subprocess.Popen(subst(cfg, fl["argv"], core=core), cwd=subst(cfg, fl["cwd"]),
                                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                                  stdin=subprocess.DEVNULL, env=env_for(job_env(cfg, fl.get("env"))),
                                                  start_new_session=True,
                                                  preexec_fn=lambda c=core: os.sched_setaffinity(0, {c}))
        time.sleep(0.05)
    for f in fillers.values():
        os.killpg(f.pid, signal.SIGKILL)
        f.wait()
    rec["jobs"] = [results[k] for k in sorted(results)]
    rec["fillers"] = sorted(fillers)
    rec["ref_after"] = segment(cfg, slots, wd, "ref_after")
    rec["monitor"] = mon.stop()
    window_io(rec, False)
    rec["check_before"] = check_segment(rec["ref_before"], ref_us)
    rec["check_after"] = check_segment(rec["ref_after"], ref_us)
    rec["jobs_ok"] = all(r["rc"] == 0 and r["record"] for r in rec["jobs"])
    rec["passed"] = bool(rec["disk_delta"]["passed"] and rec["check_before"]["passed"] and rec["check_after"]["passed"]
                         and rec["jobs_ok"])
    rec["end"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(os.path.join(wd, "window.json"), "x") as fh:
        json.dump(rec, fh, indent=1)
    return rec


def covered(need, imported):
    """A unit is in when it, or a unit that contains it (bb contains bb/draw), was imported."""
    return any(need == u or need.startswith(u + "/") for u in imported)


def prior_attempts(dirs):
    """Earlier sessions' windows (after a lost machine, STAGE2_common C5): per job, the attempts used and whether one
    was accepted (its window passed, exit 0, a record)."""
    used, accepted = {}, set()
    for d in dirs:
        for f in sorted(glob.glob(os.path.join(d, "*", "window.json"))):
            w = json.load(open(f))
            for j in w["jobs"]:
                b = j.get("base", j["job"])
                used[b] = used.get(b, 0) + 1
                if w["passed"] and j["rc"] == 0 and j.get("record", True):
                    accepted.add(b)
    return used, accepted


def cmd_run(a):
    """The plan's windows in order, each once every hand-off unit its jobs need is in the tmpfs (a window that waits
    lets the next ready one go first); before each window the ready units are copied in, after it the window is saved
    to the Windows folder. A process whose window is rejected, or which fails (non-zero exit, no record), runs again
    with the same arguments in a later window (ids r001, ...; plans.pack keeps the sides interleaved), at most
    max_attempts attempts in all (STAGE2_common C5). --prior: the windows of an earlier session of this plan on a lost
    machine; its accepted jobs are skipped and its attempts count."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from plans import pack
    cfg = load_cfg(a.cfg)
    session_of(cfg, a.out)
    if a.handoff_from:
        cfg["handoff_dir"] = folder(a.handoff_from)
    plan = json.load(open(a.plan))
    ref = json.load(open(a.sessionref))["reference_us"] if a.sessionref else None
    cfg["sessionref_us"] = f"{ref:.4f}" if ref else "0"          # gs_time reads 0 as "no reference"
    os.makedirs(a.out, exist_ok=True)
    for f in ("topo.json", "plan.json", "sref"):
        save(cfg, os.path.join(cfg["run_dir"], f))
    log = os.path.join(cfg["run_dir"], "handoff")
    attempts = a.max_attempts or cfg.get("max_attempts", 3)
    used, accepted = prior_attempts(a.prior.split(",") if a.prior else [])
    only = set(a.only.split(",")) if a.only else None
    pending = []
    for w in plan["windows"]:
        if only and w["id"] not in only:
            continue
        js = [dict(j, base=j["id"], attempt=used.get(j["id"], 0) + 1,
                   id=j["id"] if j["id"] not in used else f"{j['id']}_a{used[j['id']] + 1}")
              for j in w["jobs"] if j["id"] not in accepted]
        js = [j for j in js if j["attempt"] <= attempts]
        if js and len(js) < len(w["jobs"]):
            pending += [dict(id=f"{w['id']}p{k + 1}", rep=w.get("rep"), jobs=x) for k, x in enumerate(pack(js))]
        elif js:
            pending.append(dict(w, jobs=js))
    for w in pending:
        w["needs"] = sorted({n for j in w["jobs"] for n in j.get("needs", [])})
    retry, nr, waited = [], 0, 0.0
    while pending or retry:
        handoff_import(cfg, log)
        ready = [w for w in pending if all(covered(n, cfg["imported"]) for n in w["needs"])]
        if not ready and retry:                                       # replacements go while the hand-off is awaited
            for js in pack(retry, LIMITS["slots_max"]):
                nr += 1
                pending.append(dict(id=f"r{nr:03d}", rep=None, jobs=js,
                                    needs=sorted({n for j in js for n in j.get("needs", [])})))
            retry = []
            continue
        if not ready:
            if waited >= a.handoff_wait:
                print(f"hand-off wait over {a.handoff_wait} s: windows not run: {[w['id'] for w in pending]}", flush=True)
                break
            if waited % 600 < 30:
                print(f"waiting for the hand-off: "
                      f"{sorted({n for w in pending for n in w['needs'] if not covered(n, cfg['imported'])})}"
                      f" ({len(pending)} windows)", flush=True)
            time.sleep(30)
            waited += 30
            continue
        w = ready[0]
        pending.remove(w)
        t0 = time.time()
        r = run_window(cfg, w, a.out, ref or float("inf"), not a.no_fill)
        save(cfg, os.path.join(a.out, w["id"]))
        print(f"{w['id']}: {'PASS' if r['passed'] else 'FAIL'} {time.time() - t0:.1f} s; rc "
              f"{[j['rc'] for j in r['jobs']]}; ref {r['check_before'].get('median_us')}/"
              f"{r['check_after'].get('median_us')} us (spread {r['check_before'].get('spread')}/"
              f"{r['check_after'].get('spread')}); disk {r['disk_delta']['reads']} reads, "
              f"{r['disk_delta']['pgmajfault']} majflt; MHz win/clk {r['monitor'].get('mhz_mean')}/{r['monitor'].get('clk_mhz_mean')}", flush=True)
        for jr, j in zip(r["jobs"], w["jobs"]):
            ok = r["passed"] and jr["rc"] == 0 and jr["record"]
            if not ok and j["attempt"] < attempts:
                retry.append(dict(j, id=f"{j['base']}_a{j['attempt'] + 1}", attempt=j["attempt"] + 1))
        if not pending and retry:
            for js in pack(retry, LIMITS["slots_max"]):
                nr += 1
                pending.append(dict(id=f"r{nr:03d}", rep=None, jobs=js,
                                    needs=sorted({n for j in js for n in j.get("needs", [])})))
            retry = []
    print(f"=== run done {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    t = sub.add_parser("topo")
    t.add_argument("--out", default="")
    t.add_argument("--cfg", default="", help="the config; its \"host\" (linux) and \"timed_ccd\" set the slots")
    sub.add_parser("disk")
    im = sub.add_parser("import")
    im.add_argument("--cfg", required=True)
    im.add_argument("--units", required=True, help="e.g. surface_shots,gs")
    im.add_argument("--log", required=True, help="RUN/handoff: the import records")
    im.add_argument("--handoff-from", default="")
    im.add_argument("--wait", type=float, default=7200)
    wp = sub.add_parser("winpath")
    wp.add_argument("spec", help="a Windows folder such as %%USERPROFILE%%\\speedrt_in_dry")
    f = sub.add_parser("freq")
    f.add_argument("--secs", type=float, default=5)
    s = sub.add_parser("sessionref")
    s.add_argument("--cfg", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--repeats", type=int, default=3)
    s.add_argument("--tries", type=int, default=3, help="a reference that saw a disk read is taken again, at most this "
                   "many times in all")
    r = sub.add_parser("run")
    r.add_argument("--cfg", required=True)
    r.add_argument("--plan", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--sessionref", default="", help="sessionref.json; without it the reference check is not applied")
    r.add_argument("--only", default="")
    r.add_argument("--no-fill", action="store_true")
    r.add_argument("--handoff-from", default="", help="another hand-off folder (a dry run's), Windows or WSL path")
    r.add_argument("--handoff-wait", type=float, default=10800, help="seconds to wait for a missing hand-off unit")
    r.add_argument("--max-attempts", type=int, default=0, help="default: the config's max_attempts (3)")
    r.add_argument("--prior", default="", help="win directories of earlier sessions of this plan (a lost machine)")
    a = ap.parse_args()
    dict(topo=cmd_topo, disk=cmd_disk, freq=cmd_freq, sessionref=cmd_sessionref, run=cmd_run, winpath=cmd_winpath,
         **{"import": cmd_import})[a.what](a)


if __name__ == "__main__":
    main()
