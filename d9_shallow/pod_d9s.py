"""D9-SHALLOW's GPU session on one pre-screened H100 80GB HBM3 pod (PLAN.md section 4), with the speed re-test's code and
settings unchanged (speed_rt/pod_s.py's watched slot, speed_rt.watch rule throttle, timing.gpu_term, timing.deployed,
speed_rt.gpu_prescreen). Steps, in this order, each once:

  prescreen  speed_rt.gpu_prescreen on the idle pod -> OUT/pre/prescreen.json; exit 4 when it did not pass (the pod is
             released and replaced; nothing is timed on it)
  masks      after commit B is public: timing.deployed on the 200,000 timing shots (the view VIEW: chunk j of the timing
             set linked as validation chunk j), untimed, for d9_b1w64_sh (at its frozen tau) and d9_b3_sv2s1 ->
             OUT/masks/val_<net>.npz (c, routed = |double(c)| < tau, tau, shots_sha256)
  timed      5 rounds; in each, one fresh process of d9_b1w64_sh, then one of d9_b3_sv2s1: speed_rt.watch run --core C
             --rule throttle -- timing.gpu_term --d 9 --net NAME --deployed OUT/masks/val_NAME.npz ...; a process counts
             when it exits 0, its watch window is clean (clocks ok, the GPU alone) and gpu_term accepts it (no other
             compute process, clocks, outputs equal to the mask); at most 2 replacements a slot (3 attempts)
  handoff    the handoff c of d9_b1w64_sh at its frozen tau (timing.gpu_term --handoff: the compiled router, max(1.1 us,
             the router's estimate)), one slot, at most 3 attempts
  pack       the hand-off to the Linux workstation: OUT/handoff/surface/low/val_d9_b1w64_sh.npz (a copy of the counted
             session's mask), HANDOFF_MASKS.json (sha256, tau, routed count, shots digest), SHA256SUMS
Every attempt is in OUT/attempts.json; OUT/session.json is written last (done, the counts, the pre-screen's verdict).
    cd <tree>/surface_v2 && PYTHONPATH=<tree>/surface_v2:<tree> python ../d9_shallow/pod_d9s.py STEP --view VIEW --out OUT
        --build BUILD"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
B1, B3 = "d9_b1w64_sh", "d9_b3_sv2s1"
NETS = (B1, B3)
ROUNDS, MAX_REPL = 5, 2
FREEZE = json.load(open(os.path.join(HERE, "freeze_d9s.json")))


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sh(cmd):
    print("+ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd).returncode


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def load_att(out):
    f = os.path.join(out, "attempts.json")
    return json.load(open(f)) if os.path.exists(f) else []


def save_att(out, att):
    f = os.path.join(out, "attempts.json")
    with open(f + ".tmp", "w") as fh:
        json.dump(att, fh, indent=1)
    os.replace(f + ".tmp", f)


def core(out):
    p = json.load(open(os.path.join(out, "pre", "prescreen.json")))
    assert p.get("passed") is True, "the pre-screen did not pass: nothing is timed on this pod"
    return int(p["core"])


def watched(a, c, cmd, rec):
    """CMD pinned to the timed core inside a watch window -> (exit code, the window's verdict: clocks ok and the GPU alone)."""
    rc = sh([sys.executable, "-m", "speed_rt.watch", "run", "--core", str(c), "--rule", "throttle", "--out", rec, "--"] + cmd)
    w = json.load(open(rec)) if os.path.exists(rec) else {}
    return rc, bool(w.get("clocks", {}).get("ok")) and bool(w.get("gpu_alone"))


def slot(a, att, item, name, r, base):
    """One slot: at most MAX_REPL replacements; an attempt counts when the process exits 0, its watch window passes and
    its own record says accepted."""
    c = core(a.out)
    for k in range(MAX_REPL + 1):
        proc = base + k
        t0 = now()
        w = os.path.join(a.out, "watch", f"{item.lower()}_{name}_p{proc}.json")
        cmd = [sys.executable, "-m", "timing.gpu_term", "--d", "9", "--net", name, "--val", a.view, "--proc", str(proc),
               "--out", a.out, "--deployed", os.path.join(a.out, "masks", f"val_{name}.npz"), "--build", a.build,
               "--core", str(c)] + (["--handoff"] if item == "G2" else [])
        rc, wok = watched(a, c, cmd, w)
        f = os.path.join(a.out, f"{name}_{'handoff_' if item == 'G2' else ''}p{proc}.json")
        g = json.load(open(f)) if os.path.exists(f) else {}
        ok = rc == 0 and wok and bool(g.get("accepted"))
        att.append(dict(item=item, network=name, round=r, attempt=k, proc=proc, start=t0, end=now(), exit=rc, watch_ok=wok,
                        record=os.path.basename(f) if g else None, watch=os.path.relpath(w, a.out),
                        gpu_term_accepted=g.get("accepted"), value_us=g.get("handoff_us" if item == "G2" else "per_shot_mean"),
                        accepted=ok))
        save_att(a.out, att)
        print(json.dumps(att[-1]), flush=True)
        if ok:
            return True
    print(f"{item} {name} round {r}: not established after {MAX_REPL} replacements", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=("prescreen", "masks", "timed", "handoff", "pack"))
    ap.add_argument("--view", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", required=True)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "watch"), exist_ok=True)
    if a.step == "prescreen":
        assert not os.path.exists(os.path.join(a.out, "pre")), "one pre-screen per pod"
        rc = sh([sys.executable, "-m", "speed_rt.gpu_prescreen", "--out", os.path.join(a.out, "pre")])
        p = json.load(open(os.path.join(a.out, "pre", "prescreen.json")))
        print(f"PRESCREEN exit {rc}, passed {p.get('passed')}, core {p.get('core')}", flush=True)
        raise SystemExit(0 if rc == 0 and p.get("passed") is True else 4)
    if a.step == "masks":
        core(a.out)
        m = os.path.join(a.out, "masks")
        os.makedirs(m, exist_ok=False)
        for n in NETS:
            assert sh([sys.executable, "-m", "timing.deployed", "--d", "9", "--net", n, "--val", a.view, "--out", m,
                       "--build", a.build]) == 0, f"the mask of {n} failed"
        for n in NETS:
            import numpy as np
            z = np.load(os.path.join(m, f"val_{n}.npz"))
            want = FREEZE["networks"][n]["tau"]
            assert float(z["tau"]) == want, f"{n}: tau {float(z['tau'])} is not the frozen {want}"
            print(f"mask {n}: tau {float(z['tau'])!r}, routed {int(z['routed'].sum())}, shots {str(z['shots_sha256'])[:16]}",
                  flush=True)
        return
    att = load_att(a.out)
    if a.step == "timed":
        assert not [x for x in att if x["item"] == "G1"], "the timed block runs once"
        for r in range(1, ROUNDS + 1):
            for n in NETS:
                slot(a, att, "G1", n, r, 10 * r)
        acc = {n: [x["value_us"] for x in att if x["item"] == "G1" and x["network"] == n and x["accepted"]] for n in NETS}
        print("TIMED DONE " + json.dumps({n: dict(accepted=len(v), slowest=max(v) if v else None) for n, v in acc.items()}),
              flush=True)
    elif a.step == "handoff":
        assert [x for x in att if x["item"] == "G1"], "the timed block first"
        assert not [x for x in att if x["item"] == "G2"], "the handoff runs once"
        slot(a, att, "G2", B1, 1, 100)
    elif a.step == "pack":
        import numpy as np
        dst = os.path.join(a.out, "handoff", "surface", "low")
        os.makedirs(dst, exist_ok=False)
        f = os.path.join(a.out, "masks", f"val_{B1}.npz")
        shutil.copyfile(f, os.path.join(dst, f"val_{B1}.npz"))
        z = np.load(f)
        man = dict(what="D9-SHALLOW: the counted GPU session's routing mask of d9_b1w64_sh on the timing shots, for the "
                        "CPU term on the Linux workstation (never routed again there)", created=now(),
                   files={f"val_{B1}.npz": dict(sha256=sha256_file(f), d=9, network=str(z["network"]), tau=float(z["tau"]),
                                                routed=int(z["routed"].sum()), shots=int(len(z["routed"])),
                                                shots_sha256=str(z["shots_sha256"]))})
        with open(os.path.join(dst, "HANDOFF_MASKS.json"), "x") as fh:
            json.dump(man, fh, indent=1, sort_keys=True)
        subprocess.run(f"cd {dst} && sha256sum *.npz *.json > SHA256SUMS", shell=True, check=True)
        print(f"PACK: {man['files']}", flush=True)
    g1 = {n: [x for x in att if x["item"] == "G1" and x["network"] == n and x["accepted"]] for n in NETS}
    g2 = [x for x in att if x["item"] == "G2" and x["accepted"]]
    pre = json.load(open(os.path.join(a.out, "pre", "prescreen.json")))
    with open(os.path.join(a.out, "session.json.tmp"), "w") as fh:
        json.dump(dict(what="D9-SHALLOW GPU session", prescreen_passed=pre.get("passed"), core=pre.get("core"),
                       card=pre.get("card"), last_step=a.step, at=now(), rounds=ROUNDS,
                       g1_accepted={n: len(v) for n, v in g1.items()}, g2_accepted=len(g2),
                       done=a.step == "pack"), fh, indent=1)
    os.replace(os.path.join(a.out, "session.json.tmp"), os.path.join(a.out, "session.json"))


if __name__ == "__main__":
    main()
