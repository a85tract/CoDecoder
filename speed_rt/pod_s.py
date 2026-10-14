"""One pod session of the surface stage: a pod role's GPU items G1-G6, every process a fresh one and never two timed processes at once,
every attempt logged in SESSION/attempts.json (rewritten whole after each, so a lost pod leaves the attempts so far). The pods of this
stage time GPU work ONLY (CURRENT_RULES, 13 October 15:45): every CPU part (plain and correlated PyMatching, our CPU terms on the routed
shots, Ising's residual PyMatching, the Grace references) is timed on winserver by the CPU worker, from this session's untimed GPU pass
(the hand-off, below).

Roles: low = d 7, 9, 11 (pod S-low; G4, the d = 7 Triton engine, runs only here), high = d 13, 15, 17 (pod S-high; the d = 17 network at
study 4's tau, q = 10 %, is also prepared there). A role's comparisons use this pod only.

  prep      untimed, before the timed block: the routing masks of the role's networks on its timing set (G3; timing.deployed per network at
            its frozen tau), at S-high the d = 17 mask at study 4's tau (E8), at S-low the Triton engine's d = 7 reference (G4) and
            trtexec's input files for the role's distances. The plans are built and checked before, on the validation chunk 0
            (pod/run_s.sh build; ising_trt, ising_d5), from the frozen ONNX recipe; they read no timing shot.
  packm     hand-off part 1 (untimed): the masks -> HANDOFF/masks (+ HANDOFF_MASKS.json) for the CPU worker's C-term and C-base.
  timed     rounds r = 1-5, each in this order: G1 for the role's networks (timing.gpu_term, the order of surf_v2.session.networks); G4
            (speed_rt.triton7; S-low); G5's graph timing (speed_rt.ising_time, one process per d: r = 1 forward order, r = 2 reversed);
            G5's trtexec repetition r of every validated plan (speed_rt.ising_d5 trtexec); for r = 1-3, G6's repetition r per d and arm
            (speed_rt.bposd_time).
  handoff   G2: five handoff processes per d = 7 network, one per d = 9-17 network (timing.gpu_term --handoff).
  residuals the residual detectors of chunk 0 by each model's counted plan per d (speed_rt.ising_select; ising_d5 residuals), untimed.
  pack      hand-off part 2: the residuals -> HANDOFF/res (+ HANDOFF_RES.json) for the CPU worker's C-share, and SESSION/DONE.json, the
            record that the session reached its end.

Acceptance (the stage's section, "Acceptance"): the pod passed speed_rt/gpu_prescreen.py (stage 1's pre-screen: five GPU checks, CPU steal
at most 1 % over 5 s, quota at least 20 CPUs; its record is copied to SESSION/prescreen.json). Every timed process runs pinned to the
session's timed core, a GPU process inside a speed_rt.watch window whose verdict is its clock rule (rule "floor" for the continuous Ising
timings: every SM clock at least 97 % of the maximum; "throttle" for G1, G2, G4: throttle reasons none or idle; the modules apply the
97 % floor inside their own timed spans) and no other compute process on the card; the host-side shares (steal, busy) are recorded and
decide nothing. A process counts only when its window and its own module accept it (timing.gpu_term and triton7: outputs bit for bit,
the GPU alone; ising_time: each plan's window). A rejected or failed attempt is replaced at once, at most twice (a slot: one process of an
item in one round or repetition); a slot still rejected after that is not established, the session goes on, and the analysis withholds
every comparison that needs it. The session qualifies when its pre-screen record passed and DONE.json exists.
    python -m speed_rt.pod_s prep|packm|timed|handoff|residuals|pack --role low|high --view VIEW --val VAL --masks MASKS \\
        --onnx ONNX --work WORK --inputs INPUTS --res RES --out SESSION --build BUILD --prescreen PRE/prescreen.json \\
        --handoff HANDOFF --core C [--dry] [--limit N] [--rounds N] [--dry-inject ITEM:KEY[:proc|all],...]"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config_surf.json")))
ROLES = {"low": (7, 9, 11), "high": (13, 15, 17)}
MAX_REPL = 2
TAU_S4_D17 = 5.121974468231201          # study 4's registered tau at d = 17 (q = 10 %; results/r17/R17_RESULT.md)
Q10 = "d17_b5_r787_q10"                 # the d = 17 network at study 4's tau (E8); S-high only
Q08 = "d17_b5_r787_q08"
MODELS = ("M1", "M4")
PYCQ = os.environ.get("PYCQ", "")                    # the CUDA-Q venv's python (pod/run_s.sh sets it)


def role_ds(a):
    return ROLES[a.role]


def networks(a):
    """(d, name) of the role's networks, in the session's order (surf_v2.session.networks)."""
    from surf_v2 import session
    return [(d, n) for d, n in session.networks() if d in role_ds(a)]


def load_att(out):
    f = os.path.join(out, "attempts.json")
    return json.load(open(f)) if os.path.exists(f) else []


def save_att(out, att):
    f = os.path.join(out, "attempts.json")
    with open(f + ".tmp", "w") as fh:
        json.dump(att, fh, indent=1)
    os.replace(f + ".tmp", f)


def sh(cmd, log=None):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def watched(a, cmd, rule, rec):
    """CMD pinned to the timed core inside a watch window -> (exit code, the window's verdict). The verdict of a GPU process is its
    clock window and 'no other compute process on the card' (speed_rt/watch.py's samples); the host-side shares are recorded, not
    judged (the pods time GPU work only)."""
    rc = sh([sys.executable, "-m", "speed_rt.watch", "run", "--core", str(a.core), "--rule", rule, "--out", rec, "--"] + cmd)
    w = json.load(open(rec)) if os.path.exists(rec) else {}
    return rc, bool(w.get("clocks", {}).get("ok")) and bool(w.get("gpu_alone"))


def forced(a, item, key, r, k):
    """Dry runs only (--dry-inject ITEM:KEY[:MODE],...): an injected failure of round 1's slot ITEM:KEY, to exercise the failure
    rule on dry shots. MODE proc (the default): attempt 0's process is rejected (it does not run, exit 97); all: every attempt of the
    slot is rejected (the slot is not established). -> None or "proc"."""
    for spec in a.inject:
        it, ky, *m = spec.split(":")
        mode = m[0] if m else "proc"
        if (it, ky) == (item, key) and r == 1 and (k == 0 or mode == "all"):
            return "proc"
    return None


def slot(a, att, item, key, r, run):
    """One slot: run(k) -> (exit code, own record file or None, accepted); at most MAX_REPL replacements."""
    for k in range(MAX_REPL + 1):
        t0 = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        f = forced(a, item, key, r, k) if a.dry else None
        rc, rec, ok = run(k) if f != "proc" else (97, None, False)
        att.append(dict(item=item, key=key, round=r, attempt=k, start=t0, end=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        exit=rc, record=rec, accepted=bool(ok), **(dict(injected=f) if f else {})))
        save_att(a.out, att)
        if ok or (a.dry and not f):
            return ok
    print(f"{item} {key} round {r}: not established after {MAX_REPL} replacements", flush=True)
    return False


def g1(a, att, r, d, name):
    def run(k):
        proc = 10 * r + k
        w = os.path.join(a.out, "watch", f"g1_{name}_p{proc}.json")
        cmd = [sys.executable, "-m", "timing.gpu_term", "--d", str(d), "--net", name, "--val", a.view, "--proc",
               str(proc), "--out", a.out, "--deployed", os.path.join(a.masks, f"val_{name}.npz"), "--build", a.build,
               "--core", str(a.core)] + (["--limit", str(a.limit)] if a.limit else [])
        rc, wok = watched(a, cmd, "throttle", w)
        f = os.path.join(a.out, f"{name}_p{proc}.json")
        ok = rc == 0 and wok and os.path.exists(f) and bool(json.load(open(f)).get("accepted"))
        return rc, os.path.basename(f) if os.path.exists(f) else None, ok
    return slot(a, att, "G1", name, r, run)


def g4(a, att, r):
    def run(k):
        proc = 10 * r + k
        w = os.path.join(a.out, "watch", f"g4_p{proc}.json")
        cmd = [sys.executable, "-m", "speed_rt.triton7", "time", "--view", a.view, "--ref",
               os.path.join(a.masks, "triton7_d7.npz"), "--proc", str(proc), "--core", str(a.core), "--out", a.out] + \
              (["--limit", str(a.limit)] if a.limit else [])
        rc, wok = watched(a, cmd, "throttle", w)
        f = os.path.join(a.out, f"triton7_d7_p{proc}.json")
        ok = rc == 0 and wok and os.path.exists(f) and bool(json.load(open(f)).get("accepted"))
        return rc, os.path.basename(f) if os.path.exists(f) else None, ok
    return slot(a, att, "G4", "triton7_d7", r, run)


def g5_graph(a, att, r, d, order):
    def run(k):
        o = os.path.join(a.out, f"g5_{order}_k{k}")
        os.makedirs(o, exist_ok=True)
        rc = sh([sys.executable, "-m", "speed_rt.ising_time", "--d", str(d), "--order", order, "--view", a.view,
                 "--onnx", a.onnx, "--work", a.work, "--core", str(a.core), "--rep", str(r), "--out", o])
        f = os.path.join(o, f"ising_time_d{d}_{order}.json")
        if rc or not os.path.exists(f):
            return rc, None, False
        rec = json.load(open(f))
        timed = [x for x in rec["rows"] if x.get("timed")]
        ok = bool(timed) and all(x.get("accepted") for x in timed)
        return rc, os.path.relpath(f, a.out), ok
    return slot(a, att, "G5-graph", f"d{d}_{order}", r, run)


def validated_plans(a, d):
    from speed_rt.ising_time import PLANS, plan_files
    out = []
    for mdl, prec, kind in PLANS:
        f = plan_files(a.onnx, a.work, mdl, d, prec, kind)
        if os.path.exists(f["check"]):
            c = json.load(open(f["check"]))
            if c["test"] == "PASS" and (prec != "fp8" or (c.get("fp8") or {}).get("ok")):
                out.append(f["plan"])
    return out


def g5_trtexec(a, att, r, d, plan):
    def run(k):
        o = os.path.join(a.out, f"g5_trtexec_k{k}")
        os.makedirs(o, exist_ok=True)
        rc = sh([sys.executable, "-m", "speed_rt.ising_d5", "trtexec", "--d", str(d), "--plan", plan, "--input",
                 os.path.join(a.inputs, f"dets_d{d}.bin"), "--rep", str(r), "--core", str(a.core), "--out", o])
        f = os.path.join(o, f"trtexec_{os.path.basename(plan)[:-5]}_rep{r}.json")
        ok = rc == 0 and os.path.exists(f) and bool(json.load(open(f)).get("accepted"))
        return rc, os.path.relpath(f, a.out) if os.path.exists(f) else None, ok
    return slot(a, att, "G5-trtexec", os.path.basename(plan)[:-5], r, run)


def g6(a, att, r, d, arm):
    def run(k):
        o = os.path.join(a.out, f"g6_k{k}")
        os.makedirs(o, exist_ok=True)
        rc = sh([PYCQ, "-m", "speed_rt.bposd_time", "--d", str(d), "--arm", arm, "--rep", str(r), "--shots",
                 str(a.bposd_shots), "--view", a.view, "--core", str(a.core), "--out", o])
        f = os.path.join(o, f"bposd_d{d}_{arm}_rep{r}.json")
        ok = rc == 0 and os.path.exists(f) and bool(json.load(open(f)).get("accepted"))
        return rc, os.path.relpath(f, a.out) if os.path.exists(f) else None, ok
    return slot(a, att, "G6", f"d{d}_{arm}", r, run)


def meta(a):
    """The session's machine, recorded once (SESSION/META.json): the pod's CPU model and container quota, the timed core."""
    import platform
    q = ""
    for f in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        if os.path.exists(f):
            q = open(f).read().strip()
            break
    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit,clocks.max.sm", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.splitlines()
    cpu = open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t")
    sib = open(f"/sys/devices/system/cpu/cpu{a.core}/topology/thread_siblings_list").read().strip()
    return dict(role=a.role, ds=list(role_ds(a)), host=platform.node(), cpu=cpu, cgroup_cpu_max=q,
                cpus_allowed=len(os.sched_getaffinity(0)), core=a.core, core_siblings=sib, gpu=smi[0] if smi else None,
                deployed=open(os.path.join(os.path.dirname(HERE), "DEPLOYED")).read().strip() if os.path.exists(
                    os.path.join(os.path.dirname(HERE), "DEPLOYED")) else None,
                start=time.strftime("%Y-%m-%dT%H:%M:%S%z"), dry=a.dry)


def prep(a):
    import numpy as np
    os.makedirs(a.masks, exist_ok=True)
    assert not os.listdir(a.masks), f"{a.masks} is not empty"
    bad = []
    for d, name in networks(a):                                          # G3: surf_v2.session masks, per network
        if sh([sys.executable, "-m", "timing.deployed", "--d", str(d), "--net", name, "--val", a.view, "--out", a.masks,
               "--build", a.build]):
            bad.append(f"mask {name}")
    if a.role == "high":
        z = dict(np.load(os.path.join(a.masks, f"val_{Q08}.npz")))
        c = z["c"].astype(np.float64)
        z.update(routed=np.abs(c) < TAU_S4_D17, tau=np.float64(TAU_S4_D17), network=Q10)
        np.savez(os.path.join(a.masks, f"val_{Q10}.npz"), **z)
        print(f"d = 17 at study 4's tau: {int(z['routed'].sum())} routed", flush=True)
    if a.role == "low" and sh([sys.executable, "-m", "speed_rt.triton7", "ref", "--view", a.view, "--out", a.masks]):
        bad.append("triton7 ref")
    ds = [str(d) for d in role_ds(a)]
    if sh([sys.executable, "-m", "speed_rt.ising_d5", "input", "--d"] + ds + ["--view", a.view, "--out", a.inputs]):
        bad.append("trtexec inputs")
    subprocess.run(f"cd {a.masks} && sha256sum *.npz > MASKS.sha256", shell=True, check=True)
    print("PREP DONE" + (f"; failed: {bad}" if bad else ""), flush=True)
    raise SystemExit(1 if bad else 0)


def timed(a):
    os.makedirs(os.path.join(a.out, "watch"), exist_ok=True)
    att = load_att(a.out)
    assert not att, f"{a.out} holds a session already"
    with open(os.path.join(a.out, "META.json"), "x") as fh:
        json.dump(meta(a), fh, indent=1)
    assert a.dry or a.prescreen, "a registered session needs the pod's pre-screen record (--prescreen)"
    if a.prescreen:
        import shutil
        shutil.copyfile(a.prescreen, os.path.join(a.out, "prescreen.json"))
        assert a.dry or json.load(open(a.prescreen))["passed"], "the pod did not pass its pre-screen"
    bi = os.path.join(a.work, "BUILD_INPUTS.json")      # the plan builds' record (speed_rt/build_inputs.py), checked by the analysis
    assert a.dry or os.path.exists(bi), "a registered session needs the plan builds' record (WORK/BUILD_INPUTS.json)"
    if os.path.exists(bi):
        import shutil
        shutil.copyfile(bi, os.path.join(a.out, "BUILD_INPUTS.json"))
    nets = networks(a)
    ds = role_ds(a)
    plans = {d: validated_plans(a, d) for d in ds}
    for r in range(1, a.rounds + 1):
        print(f"=== round {r} {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
        for d, name in nets:
            g1(a, att, r, d, name)
        if a.role == "low":
            g4(a, att, r)
        if r in (1, 2):
            for d in ds:
                if plans[d]:
                    g5_graph(a, att, r, d, "fwd" if r == 1 else "rev")
        for d in ds:
            for plan in plans[d]:
                g5_trtexec(a, att, r, d, plan)
        if r <= CFG["sessions"]["S"]["bposd_rounds"]:
            for d in ds:
                for arm in ("minllr", "bp60"):
                    g6(a, att, r, d, arm)
    print("TIMED DONE " + time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)


def handoff(a):
    att = load_att(a.out)
    assert att, "the timed block first"
    for d, name in networks(a):
        for i in range(min(CFG["sessions"]["S"]["handoff_processes_d7"], a.rounds) if d == 7 else 1):
            def run(k, d=d, name=name, i=i):
                proc = 100 + 10 * i + k
                w = os.path.join(a.out, "watch", f"g2_{name}_p{proc}.json")
                cmd = [sys.executable, "-m", "timing.gpu_term", "--d", str(d), "--net", name, "--val", a.view, "--proc",
                       str(proc), "--out", a.out, "--deployed", os.path.join(a.masks, f"val_{name}.npz"), "--build", a.build,
                       "--core", str(a.core), "--handoff"] + (["--limit", str(max(a.limit, 20000))] if a.limit else [])
                rc, wok = watched(a, cmd, "throttle", w)
                f = os.path.join(a.out, f"{name}_handoff_p{proc}.json")
                ok = rc == 0 and wok and os.path.exists(f) and bool(json.load(open(f)).get("accepted"))
                return rc, os.path.basename(f) if os.path.exists(f) else None, ok
            slot(a, att, "G2", name, i + 1, run)
    print("HANDOFF DONE " + time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)


def residuals(a):
    from speed_rt import ising_select
    sel = ising_select.select(a.out, role_ds(a))
    os.makedirs(a.res, exist_ok=True)
    json.dump({f"{m} d{d}": v for (m, d), v in sel.items()}, open(os.path.join(a.res, "selected.json"), "x"), indent=1,
              sort_keys=True)
    bad = 0
    for (m, d), v in sel.items():
        plan = os.path.join(a.work, "plans", v["plan"])
        bad += bool(sh([sys.executable, "-m", "speed_rt.ising_d5", "residuals", "--d", str(d), "--model", m,
                        "--plan", plan, "--view", a.view, "--out", a.res]))
    subprocess.run(f"cd {a.res} && sha256sum res_*.npz > RES.sha256", shell=True, check=True)
    missing = [f"{m} d{d}" for m in MODELS for d in role_ds(a) if (m, d) not in sel]
    print(f"RESIDUALS DONE: {len(sel)} plans; no counted plan for {missing}" if missing else
          f"RESIDUALS DONE: {len(sel)} plans", flush=True)
    raise SystemExit(1 if bad else 0)


def sha256_file(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def handoff_manifest(a, kind, files, extra):
    man = dict(what=f"SPEED-RT surface stage: the pod's untimed GPU pass for the CPU worker, part '{kind}' (speed_rt/pod_s.py "
                    f"{'packm' if kind == 'masks' else 'pack'}; format in the stage's section 1.14)", role=a.role, ds=list(role_ds(a)),
               pod=__import__("platform").node(), deployed=open(os.path.join(os.path.dirname(HERE), "DEPLOYED")).read().strip()
               if os.path.exists(os.path.join(os.path.dirname(HERE), "DEPLOYED")) else None, dry=a.dry,
               created=time.strftime("%Y-%m-%dT%H:%M:%S%z"), files=files, **extra)
    out = os.path.join(a.handoff, kind)
    with open(os.path.join(out, f"HANDOFF_{kind.upper()}.json"), "x") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)
    print(f"HANDOFF {kind}: {len(files)} files -> {out}", flush=True)


def packm(a):
    """Hand-off part 1, after prep: the routing masks of the role's networks (G3, untimed) for the CPU worker's C-term (our CPU term b on
    the routed timing shots) and C-base: handoff/masks/val_<network>.npz (copies; arrays c, routed, tau, network, shots_sha256, ...) and
    HANDOFF_MASKS.json (per file: sha256, d, network, tau, the routed count, the digest of the timing set it was computed on)."""
    import numpy as np
    import shutil
    out = os.path.join(a.handoff, "masks")
    os.makedirs(out, exist_ok=False)
    files, digests = {}, {}
    for f in sorted(glob.glob(os.path.join(a.masks, "val_*.npz"))):
        z = np.load(f)
        name = os.path.basename(f)
        shutil.copyfile(f, os.path.join(out, name))
        d = int(z["d"]) if "d" in z.files else int(str(z["network"]).split("_")[0][1:])
        files[name] = dict(sha256=sha256_file(f), d=d, network=str(z["network"]), tau=float(z["tau"]), routed=int(z["routed"].sum()),
                           shots=int(len(z["routed"])), shots_sha256=str(z["shots_sha256"]))
        digests[str(d)] = str(z["shots_sha256"])
    handoff_manifest(a, "masks", files, dict(timing_set_shots_sha256=digests, mask_arrays="c (float32 logit per timing shot), routed "
                                             "(bool per timing shot, |c| < tau), tau, network, shots_sha256, d"))


def pack(a):
    """Hand-off part 2 and the end of the session, after the residuals: the residual detectors of each model's counted plan (untimed) for
    the CPU worker's C-share: handoff/res/res_d<d>_<M>_val000.npz (arrays pre_L, res, obs) with their .json records, selected.json and
    HANDOFF_RES.json; then SESSION/DONE.json (the record that the session reached its end)."""
    import shutil
    att = load_att(a.out)
    assert att and os.path.exists(os.path.join(a.res, "selected.json")), "the residuals first"
    out = os.path.join(a.handoff, "res")
    os.makedirs(out, exist_ok=False)
    files = {}
    for f in sorted(glob.glob(os.path.join(a.res, "*"))):
        if os.path.isfile(f):
            shutil.copyfile(f, os.path.join(out, os.path.basename(f)))
            files[os.path.basename(f)] = dict(sha256=sha256_file(f))
    sel = json.load(open(os.path.join(a.res, "selected.json")))
    handoff_manifest(a, "res", files, dict(counted_plans=sel))
    with open(os.path.join(a.out, "DONE.json"), "x") as fh:
        json.dump(dict(role=a.role, end=time.strftime("%Y-%m-%dT%H:%M:%S%z"), attempts=len(att), dry=a.dry), fh, indent=1)
    print("PACK DONE; SESSION DONE " + time.strftime("%Y-%m-%dT%H:%M:%S%z"), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("prep", "packm", "timed", "handoff", "residuals", "pack"))
    ap.add_argument("--role", required=True, choices=sorted(ROLES))
    for k in ("view", "val", "masks", "onnx", "work", "inputs", "res", "out", "build", "prescreen", "handoff"):
        ap.add_argument(f"--{k}", default="")
    ap.add_argument("--core", type=int, default=-1)
    ap.add_argument("--limit", type=int, default=0, help="dry runs only: the first N shots of G1, G2, G4")
    ap.add_argument("--bposd-shots", type=int, default=200)
    ap.add_argument("--rounds", type=int, default=CFG["sessions"]["S"]["rounds"], help="the GPU block's rounds (dry runs: fewer)")
    ap.add_argument("--dry", action="store_true", help="dry runs only: nothing replaced, a failed check does not stop")
    ap.add_argument("--dry-inject", default="", help="dry runs only: ITEM:KEY[:proc|all],... forces failures of round 1's slots "
                    "to exercise the failure rule (an injected slot is replaced as in a registered session)")
    a = ap.parse_args()
    a.inject = [x for x in a.dry_inject.split(",") if x]
    assert a.dry or not a.inject, "--dry-inject is for dry runs only"
    assert a.dry or (not a.limit and a.rounds == CFG["sessions"]["S"]["rounds"]), "--limit and --rounds are for dry runs only"
    {"prep": prep, "packm": packm, "timed": timed, "handoff": handoff, "residuals": residuals, "pack": pack}[a.what](a)


if __name__ == "__main__":
    main()
