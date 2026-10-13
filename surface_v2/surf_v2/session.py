"""SURF-v2's timing session on E1 (the research plan's section 4.6; config timing): every process a fresh one, in a
fixed interleaved order over the 12 networks (the six new and the six deployed; a new network under a
training-failure exception left out: SKIP, or the freeze record's exceptions once frozen), every attempt logged in
SESSION/attempts.json (rewritten whole after each, so a lost pod leaves the attempts so far).

  masks    timing.deployed for every network on its validation shots (VAL, surf_v2.valshots's): c, and the routing at
           the network's tau where it has one (after the freeze every network has) -> MASKS/val_<name>.npz
  gpu      the GPU check (start), then five rounds; in each, timing.gpu_term once per network (the GPU term), its
           outputs compared bit for bit with MASKS' c; a rejected process is replaced at once while the network has
           replacements left (two per network, GPU term and handoff together); a third rejection stops the session
           (not qualified)
  handoff  after the freeze (every network has its tau): the handoff processes, five at d = 7 (the kernel's own), one
           at d = 9-17 (the router's estimate), with the same replacement rule; then the GPU check (end)
    python -m surf_v2.session masks --val VAL --out MASKS [--build build/d7]
    python -m surf_v2.session gpu --val VAL --masks MASKS --out SESSION [--build build/d7] [--limit N]
    python -m surf_v2.session handoff --val VAL --masks MASKS --out SESSION [--build build/d7] [--limit N]
--limit N (dry passes only) times the first N validation shots; the handoff needs N >= 20,000, as d = 7's handoff reads
20,000 shots (E0 ran it at 2,000, and every d = 7 handoff process failed its assert). --dry (dry passes on a shared GPU
only, such as an Alpine slice): one process per slot, its acceptance recorded but never replaced, and a failed GPU
check reported without stopping, so every path runs; the analysis judges the session by the same rule (with nothing
replaced, it qualifies only if nothing was rejected)."""
import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.path.join(HERE, "config.json")))


def networks(freeze=os.path.join(HERE, "freeze.json")):
    """(d, name) in the session's order: per d, the new network, then the deployed one. A new network under a
    training-failure exception has no weights and is left out: SKIP names it (as in pod_e1_cal.txt) before the freeze,
    and once the tree holds the freeze record its exceptions decide, SKIP then naming the same networks or none."""
    skip = set(os.environ.get("SKIP", "").split())
    new = {CFG["networks"][str(d)]["new"] for d in CFG["distances"]}
    assert skip <= new, f"SKIP names networks that are not new ones: {sorted(skip - new)}"
    if os.path.exists(freeze):
        exc = {CFG["networks"][k]["new"] for k in json.load(open(freeze)).get("exceptions", {})}
        assert not skip or skip == exc, f"SKIP {sorted(skip)} is not the freeze's exceptions {sorted(exc)}"
        skip = exc
    return [(d, n) for d in CFG["distances"] for r in ("new", "deployed")
            if (n := CFG["networks"][str(d)].get(r)) is not None and n not in skip]


def run(args):
    print("+ " + " ".join(args), flush=True)
    return subprocess.run([sys.executable, "-m"] + args).returncode


def attempts(out):
    f = os.path.join(out, "attempts.json")
    return json.load(open(f)) if os.path.exists(f) else {}


def save(out, att):
    f = os.path.join(out, "attempts.json")
    with open(f + ".tmp", "w") as fh:
        json.dump(att, fh, indent=1)
    os.replace(f + ".tmp", f)


def one(a, att, d, name, kind):
    """One accepted process of `kind` for the network, replacing rejected ones within the budget; False: budget spent."""
    rows = att.setdefault(name, [])
    while True:
        rej = sum(1 for x in rows if not x["accepted"])
        if rej > CFG["timing"]["max_replacements"] and not a.dry:
            return False
        k = len(rows)
        tag = f"p{k}" if kind == "gpu" else f"handoff_p{k}"
        cmd = ["timing.gpu_term", "--d", str(d), "--net", name, "--val", a.val, "--proc", str(k), "--out", a.out,
               "--deployed", os.path.join(a.masks, f"val_{name}.npz"), "--build", a.build]
        if a.limit:
            cmd += ["--limit", str(a.limit)]
        if kind == "handoff":
            cmd.append("--handoff")
        rc = run(cmd)
        f = f"{name}_{tag}.json"
        p = os.path.join(a.out, f)
        ok = rc == 0 and os.path.exists(p) and bool(json.load(open(p)).get("accepted"))
        rows.append(dict(kind=kind, proc=k, file=f if os.path.exists(p) else None, exit=rc, accepted=ok))
        save(a.out, att)
        if ok or a.dry:
            return True


def masks(a):
    os.makedirs(a.out, exist_ok=True)
    bad = [name for d, name in networks()
           if run(["timing.deployed", "--d", str(d), "--net", name, "--val", a.val, "--out", a.out, "--build", a.build])]
    raise SystemExit(1 if bad else 0)


def gpu(a):
    os.makedirs(a.out, exist_ok=True)
    assert not attempts(a.out), f"{a.out} holds a session already"
    if run(["surf_v2.gpucheck", "--when", "start", "--out", os.path.join(a.out, "gpucheck_start.json")]):
        print("GPU check failed at the start: the session does not qualify" + ("; dry: going on" if a.dry else ""),
              flush=True)
        if not a.dry:
            raise SystemExit(1)
    att = {}
    for _ in range(CFG["timing"]["gpu_processes"]):
        for d, name in networks():
            if not one(a, att, d, name, "gpu"):
                print(f"{name}: a third rejected process: the session does not qualify", flush=True)
                raise SystemExit(1)
    print("GPU terms done: " + ", ".join(f"{n} {sum(x['accepted'] for x in r)}" for n, r in att.items()), flush=True)


def handoff(a):
    att = attempts(a.out)
    assert att, "the GPU terms first"
    for d, name in networks():
        for _ in range(CFG["timing"]["handoff_processes_d7"] if d == 7 else CFG["timing"]["handoff_processes_d9_17"]):
            if not one(a, att, d, name, "handoff"):
                print(f"{name}: a third rejected process: the session does not qualify", flush=True)
                raise SystemExit(1)
    rc = run(["surf_v2.gpucheck", "--when", "end", "--out", os.path.join(a.out, "gpucheck_end.json")])
    print("handoffs done; GPU check at the end " + ("passed" if rc == 0 else "FAILED: the session does not qualify"),
          flush=True)
    raise SystemExit(0 if a.dry else rc)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=("masks", "gpu", "handoff"))
    ap.add_argument("--val", required=True)
    ap.add_argument("--masks", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--build", default="build/d7")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry", action="store_true", help="dry passes on a shared GPU only: nothing replaced, never stops")
    a = ap.parse_args()
    assert a.what != "handoff" or not a.limit or a.limit >= 20000, "the d = 7 handoff reads 20,000 shots: --limit 20000"
    {"masks": masks, "gpu": gpu, "handoff": handoff}[a.what](a)


if __name__ == "__main__":
    main()
