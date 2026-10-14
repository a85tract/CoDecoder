"""The surface family's export (STAGE_surface section 1.14): a harness run -> OUT/records/cpu/, as speed_rt/analysis_s.cpu_reader reads it.
    cpu_session.json   cpu (the CPU model as the records name it), done (every planned window ran, no re-run owed), dry, machine
                       (topo.json), run, reference_us
    cpu_attempts.json  a list as the pods' attempts.json: item, key, round (= the repetition), attempt (0 = the slot's first), start,
                       end (the window's), exit, record (relative to cpu/), accepted; and window, job, slot, core, window_passed, wall_s
    cpu/<window>_s<slot>_<job>/<record>.json    the process's own record, copied byte for byte
Items: C-ref (part gref), C-base (surf_base), C-term (surf_term), C-share (surf_share). Item and key come from the plan job: its own
`item` / `key` when the planner records them, else the command's flags (the job id is the cross-check). A process of a rejected window,
with rc != 0, or without a readable record is an attempt with accepted false (the loader's flag: window passed and rc 0).
    python export_surf.py CPU_DIR [--dry]    (a test) every slot of an export through the analysis's own reader (PYTHONPATH=CODE/surface_v2:CODE)"""
import collections
import json
import os
import re
import shutil
import sys

ITEMS = dict(gref="C-ref", surf_base="C-base", surf_term="C-term", surf_share="C-share")
BY_ID = dict(gref=r"gref_((?:plain|corr)_d\d+_p\d)", surf_base=r"base_((?:plain|corr)_d\d+)", surf_term=r"term_(.+)",
             surf_share=r"share_((?:plain|corr)_M\d_d\d+)")
MAX_ATTEMPTS = 3                                     # STAGE2_common C5 (export.py --max-attempts for a dry run)
REC = re.compile(r"(ref|term|share)(_plain)?_.+_rep\d+\.json")


def base_id(jid):
    """A retried job's id without its _a<n> suffix (the plan's id)."""
    return re.sub(r"_a\d+$", "", jid)


def flags(argv):
    """{--flag: its value, or True for a flag without one} of a command line."""
    out, i = {}, 0
    while i < len(argv):
        more = argv[i].startswith("--") and i + 1 < len(argv) and not argv[i + 1].startswith("--")
        if argv[i].startswith("--"):
            out[argv[i]] = argv[i + 1] if more else True
        i += 1 + more
    return out


def item_key(j):
    """(item, key, rep, conflict) of a plan job; key or rep None when neither the plan, the flags nor the id give it."""
    part, fl = j["part"], flags(j["argv"])
    m = re.fullmatch(BY_ID[part] + r"_r(\d+)", base_id(j["id"]))
    dec = "plain" if fl.get("--plain") else "corr"
    try:
        d = int(fl["--d"])
        key = dict(gref=lambda: f"{fl['--dec']}_d{d}_p{round(float(fl['--p']) * 1000)}", surf_base=lambda: f"{dec}_d{d}",
                   surf_term=lambda: re.fullmatch(r"val_(.+)\.npz", os.path.basename(fl["--mask"])).group(1),
                   surf_share=lambda: f"{dec}_{fl['--model']}_d{d}")[part]()
    except (KeyError, ValueError, AttributeError, TypeError):
        key = None
    kid = m.group(1) if m else None
    cands = {x for x in (j.get("key"), key, kid) if x}
    rep = int(fl["--rep"]) if str(fl.get("--rep", "")).isdigit() else (int(m.group(2)) if m else None)
    return (j.get("item") or ITEMS[part], j.get("key") or key or kid, rep,
            f"{j['id']}: keys {sorted(cands)}" if len(cands) > 1 else None)


def record_file(a):
    """The name of the process's own record in its directory, or None (no record, or more than one)."""
    js = [f for f in a["files"] if f.endswith(".json")]
    js = [f for f in js if REC.fullmatch(f)] or js
    return js[0] if len(js) == 1 else None


def finished(R):
    """True when every planned job has an accepted attempt or has used all its attempts (STAGE2_common C5)."""
    used, acc = collections.Counter(), set()
    for x in R["attempts"]:
        used[base_id(x["job"])] += 1
        if x["accepted"]:
            acc.add(base_id(x["job"]))
    return all(b in acc or used[b] >= R.get("max_attempts", MAX_ATTEMPTS) for b, j in R["jobs"].items()
               if not j["part"].endswith("_warm"))


def export_surf(R, out, dry=False):
    """Write OUT/records/cpu/ (see the module's docstring) from the run R -> {dir, cpu, done, dry, attempts, accepted, records,
    slots: {item: [established, planned]}, missing: planned slots with no accepted attempt, conflicts, unplaced}."""
    cpu_dir = os.path.join(out, "records", "cpu")
    os.makedirs(os.path.join(cpu_dir, "cpu"))                           # never into an existing export
    wins = {w["window"]: w for w in R["windows"]}
    att, seq, got, conflicts, unplaced = [], collections.Counter(), set(), [], []
    cpus = dict(acc=collections.Counter(), all=collections.Counter())
    for a in R["attempts"]:
        j = a.get("plan_job") or R["jobs"].get(base_id(a["job"]))
        if a["part"] not in ITEMS or j is None:
            continue
        item, key, rep, why = item_key(j)
        if why:
            conflicts.append(why)
        if key is None or rep is None:
            unplaced.append(a["job"])
            continue
        rec, r, f = None, None, record_file(a)
        if f is None and any(x.endswith(".json") for x in a["files"]):
            conflicts.append(f"{a['dir']}: more than one record")
        if f:
            sd = f"{a['window']}_{os.path.basename(a['dir'])}"
            src = os.path.join(R["run"], a["dir"], f)
            os.makedirs(os.path.join(cpu_dir, "cpu", sd))
            shutil.copyfile(src, os.path.join(cpu_dir, "cpu", sd, f))
            rec = f"cpu/{sd}/{f}"
            try:
                with open(src) as fh:
                    r = json.load(fh)
            except ValueError:
                r = None
        ok = bool(a["accepted"]) and isinstance(r, dict)
        if isinstance(r, dict):
            cpus["all"][r.get("cpu")] += 1
            cpus["acc"][r.get("cpu")] += ok
            if r.get("rep") != rep:
                conflicts.append(f"{rec}: rep {r.get('rep')} in the record, {rep} in the plan")
        k = seq[(item, key, rep)]
        seq[(item, key, rep)] += 1
        w = wins.get(a["window"], {})
        att.append(dict(item=item, key=key, round=rep, attempt=k, start=w.get("start"), end=w.get("end"), exit=a["rc"], record=rec,
                        accepted=ok, window=a["window"], job=a["job"], slot=a["slot"], core=a["core"],
                        window_passed=a["window_passed"], wall_s=a["wall_s"]))
        if ok:
            got.add((item, key, rep))
    lscpu = (R["topo"].get("host") or {}).get("lscpu") or ""
    m = re.search(r"^Model name:\s*(.+?)\s*$", lscpu, re.M)
    cpu = next((c.most_common(1)[0][0] for c in (+cpus["acc"], cpus["all"]) if c), m and m.group(1))
    if len(cpus["all"]) > 1:
        conflicts.append(f"the records name several CPUs: {dict(cpus['all'])}")
    sess = dict(cpu=cpu, done=finished(R), dry=bool(dry or R["plan"].get("dry")), machine=R["topo"], run=R["name"],
                reference_us=(R.get("sessionref") or {}).get("reference_us"))
    for name, v in (("cpu_session.json", sess), ("cpu_attempts.json", att)):
        with open(os.path.join(cpu_dir, name), "x") as fh:
            json.dump(v, fh, indent=1)
            fh.write("\n")
    want = {x[:3] for j in R["jobs"].values() if j["part"] in ITEMS for x in [item_key(j)] if x[1] is not None and x[2] is not None}
    return dict(dir=cpu_dir, cpu=cpu, done=sess["done"], dry=sess["dry"], attempts=len(att), accepted=sum(x["accepted"] for x in att),
                records=sum(x["record"] is not None for x in att),
                slots={i: [sum(x[0] == i for x in got), sum(x[0] == i for x in want)] for i in ITEMS.values()},
                missing=[f"{i} {k} r{r}" for i, k, r in sorted(want - got)], conflicts=conflicts, unplaced=unplaced)


def check(cpu_dir, dry=False):
    """Every slot the analysis wants (analysis_s.cpu_quantities' item/key lists) through its own reader, speed_rt.analysis_s.cpu_reader
    (PYTHONPATH=CODE/surface_v2:CODE; the frozen code digest is not compared here, `analysis_surf fit` does) -> (qualifies_cpu's
    answer, {item: [read, wanted]}, {"item key": the reason a slot is not read})."""
    from speed_rt import analysis_s as AS, grace_fit
    with open(os.path.join(os.path.dirname(AS.__file__), "freeze_surf.json")) as fh:
        nets = sorted(json.load(fh)["surface"]["networks"]) + [AS.Q10]
    get, _, _ = AS.cpu_reader(cpu_dir, dict(surface=dict(code_digest=None)), dry)
    want = {"C-ref": [f"{dec}_d{d}_p{round(p * 1000)}" for dec, d, p in grace_fit.points()],
            "C-base": [f"{k}_d{d}" for k in ("corr", "plain") for d in AS.DS], "C-term": nets,
            "C-share": [f"{k}_{m}_d{d}" for k in ("corr", "plain") for m in AS.MODELS for d in AS.DS]}
    bad = {}
    for item, keys in want.items():
        for key in keys:
            try:
                get(item, key)
            except (AS.Missing, KeyError, OSError, ValueError) as e:
                bad[f"{item} {key}"] = str(e)
    return AS.qualifies_cpu(cpu_dir, dry), {i: [len(k) - sum(b.startswith(i + " ") for b in bad), len(k)] for i, k in want.items()}, bad


if __name__ == "__main__":
    ok, n, bad = check(sys.argv[1], "--dry" in sys.argv)
    print("qualifies_cpu:", ok, "; slots read / wanted:", n)
    for k, v in list(bad.items())[:12]:
        print("  ", k, "-", v)
    sys.exit(0 if ok[0] and not bad else 3)
