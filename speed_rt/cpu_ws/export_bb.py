"""SPEED-RT export, BB family: a CPU-session run directory -> the CPU input of speed_rt/bb/analysis_bb.py, as OUT/cpu/
    rows.csv        the rows (bb.timing opponent, bb.stage3 / bb.stage4_gross cputerm) of every BB process of every attempt, as the
                    processes wrote them, in the order of the attempts
    accepted.json   [{"item", "round", "process", ...}]: the process of each slot's accepted attempt (the reader takes the first three)
    session.json    {"qualified", "why_not_qualified", "reached_end", "reference_us", "not_established", ...}
    grace.json      the stage's fit_s2.json (analysis_surf fit on the C-ref records), a byte copy
    host.json       {"topo": topo.json, "sessionref": sessionref.json}, copied
    attempt_log.json  every attempt, accepted or not, with its outcome and reason (published, never read by analysis_bb)
It copies values and changes none. An item is "relay_p0.003", "cpu_BB72-3S", ... (speed_rt/bb/cpu_item.sh --list); its round is the
process's --proc. A slot is established by its first accepted attempt: the harness verdict (a process of a rejected window is a
rejected attempt), exit 0, and exactly one timing row that analysis_bb selects (role, name, code, p, note split). The fit comes from
R["fit_s2"] (a path; export.py --fit) or RUN/fit_s2.json, copied as it is; without one grace.json is not written and the stage's fit
goes in at the analysis. Plain Python 3.12, standard library only."""
import csv
import json
import os
import re
import shutil

ITEMS = ["relay_p0.003", "bposd_p0.003", "relay_p0.002", "bposd_p0.002", "relay_full_p0.003", "relay_full_p0.002",
         "cpu_BB72-3S", "cpu_BB72-2S", "cpu_BB72-3F", "cpu_BB72-2F", "relay_p0.004", "bposd_p0.004", "relay_full_p0.004",
         "cpu_G-4R", "cpu_G-4F"]                          # analysis_bb's items (cpu_item.sh --list), 3 rounds each
CELLS = {"BB72-3S": ("bb72", 0.003), "BB72-3F": ("bb72", 0.003), "BB72-2S": ("bb72", 0.002), "BB72-2F": ("bb72", 0.002),
         "G-4R": ("bb144", 0.004), "G-4F": ("bb144", 0.004)}   # analysis_bb.PAIRS: the code and p its cpu_term rows must carry
ROUNDS = 3
FIELDS = ["date", "host", "device", "role", "code", "p", "name", "proc", "n", "value_ms", "session", "process", "binding", "note"]   # bb.timing's


def _spec(argv):
    """(item, round, the row filter analysis_bb applies) of a BB timing command line; None when it is not one."""
    if not any(os.path.basename(x) == "srt_bb.py" for x in argv):
        return None
    arg = lambda k: argv[argv.index(k) + 1] if k in argv else None
    try:
        rnd = int(arg("--proc"))
    except (TypeError, ValueError):
        return None
    if "opponent" in argv:
        dec, p = arg("--dec"), arg("--p")
        return f"{dec}_p{p}", rnd, dict(role="opponent", name=dec, code=arg("--code"), p=float(p), split=arg("--split"))
    if "cputerm" in argv and arg("--cell") in CELLS:
        cell = arg("--cell")
        return f"cpu_{cell}", rnd, dict(role="cpu_term", name=cell, code=CELLS[cell][0], p=CELLS[cell][1], split=arg("--split"))
    return None


def _match(r, w):
    """analysis_bb's selection of a row: role, name, code, p (to 9 digits) and the note's split."""
    try:
        note = json.loads(r["note"]) if (r.get("note") or "").startswith("{") else {}
        return (r.get("role") == w["role"] and r.get("name") == w["name"] and r.get("code") == w["code"]
                and round(float(r.get("p")), 9) == round(w["p"], 9) and note.get("split") == w["split"])
    except (TypeError, ValueError):
        return False


def _read(path):
    return list(csv.DictReader(open(path, newline=""))) if os.path.exists(path) else []


def _dump(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, allow_nan=False)
        fh.write("\n")


def _relative(text):
    """A record names files relative, never by an absolute path (as speed_rt/bb/attempts.py): absolute directory prefixes are dropped."""
    return re.sub(r"(?<![\w./-])/(?:[\w.@+=-]+/)+", "", text or "")


def _fit(R):
    """(the path of the stage's fit_s2.json, None) or (None, why not): R["fit_s2"] (a path) or RUN/fit_s2.json. The
    reader (analysis_bb.grace_factor) takes the file as it is, so grace.json is a byte copy of it."""
    f = R.get("fit_s2")
    if f is None and os.path.exists(os.path.join(R["run"], "fit_s2.json")):
        f = os.path.join(R["run"], "fit_s2.json")
    if not f or not os.path.exists(f):
        return None, "no fit_s2.json (export.py --fit, or RUN/fit_s2.json)"
    return f, None


def export_bb(R, out, dry=False):
    """Writes OUT/cpu/ (module docstring) from the loaded run R; returns the counts and what is missing."""
    log, acc, tried, rows_out, fields, devices, labels, unknown = [], {}, {}, [], list(FIELDS), set(), set(), []
    planned = set()
    for j in R["jobs"].values():
        s = None if j["part"].endswith("_warm") else _spec(j["argv"])
        if s and s[0] in ITEMS:
            planned.add((s[0], s[1]))
    for a in R["attempts"]:
        s = None if a["part"].endswith("_warm") else _spec(a["argv"])
        if s is None:
            continue
        item, rnd, want = s
        if item not in ITEMS:
            unknown.append(a["job"])
            continue
        planned.add((item, rnd))
        rows = _read(os.path.join(R["run"], a["dir"], "rows.csv"))
        for r in rows:
            fields += [k for k in r if k not in fields]
            devices.add(r.get("device", ""))
            labels.add(r.get("session", ""))
        rows_out += rows
        mine = [r for r in rows if _match(r, want)]
        procs = sorted({r.get("process") or "" for r in mine})
        if a["rc"] != 0:
            res, why = "failed", f"exit {a['rc']}"
        elif not a["accepted"]:
            res, why = "rejected", f"window {a['window']} did not pass"
        elif len(mine) != 1 or len(procs) != 1 or not procs[0]:
            res, why = "failed", f"{len(mine)} timing rows of this item in the process's rows.csv (one expected)"
        elif (item, rnd) in acc:
            res, why = "surplus", "the slot already has an accepted attempt"
        else:
            res, why = "accepted", "accepted"
            acc[(item, rnd)] = dict(item=item, round=rnd, process=procs[0], attempt=a["attempt"], window=a["window"],
                                    job=a["job"], dir=a["dir"])
        tried[(item, rnd)] = tried.get((item, rnd), 0) + 1
        log.append(dict(item=item, round=rnd, attempt=a["attempt"], outcome=res, reason=why, processes=procs if mine else [],
                        window=a["window"], window_passed=a["window_passed"], rc=a["rc"], wall_s=a["wall_s"], slot=a["slot"],
                        core=a["core"], dir=a["dir"], cmd=_relative(" ".join(a["argv"]))))
    if not tried and not planned:
        return dict(skipped="no BB job in the run")
    missing = sorted(planned - set(acc), key=lambda k: (ITEMS.index(k[0]), k[1]))
    open_slots = [k for k in missing if tried.get(k, 0) < R.get("max_attempts", ROUNDS)]      # no attempt left unused: the session did not end
    sref, topo = R.get("sessionref") or {}, R.get("topo") or {}
    ref = sref.get("reference_us")
    ref_ok = isinstance(ref, (int, float)) and ref > 0 and (sref.get("disk_delta") or {}).get("passed", True) is True
    why = ([] if ref_ok else ["the session reference failed its check (sref/sessionref.json: reference_us > 0 and "
                              "disk_delta.passed; ref_ok false)" if isinstance(ref, (int, float)) and ref > 0 else
                              "no session reference (sref/sessionref.json: reference_us)"]) \
        + ([f"{len(open_slots)} slots neither established nor out of attempts: {', '.join(f'{i} r{r}' for i, r in open_slots[:6])}"]
           if open_slots else []) + ([] if planned else ["no BB slot in the plan"])
    model = re.search(r"Model name:\s*(.+)", (topo.get("host") or {}).get("lscpu") or "")
    wins = R.get("windows") or []
    d = os.path.join(out, "cpu")
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "rows.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fields, restval="")
        w.writeheader()
        w.writerows(rows_out)
    _dump(os.path.join(d, "accepted.json"), [acc[k] for k in sorted(acc, key=lambda k: (ITEMS.index(k[0]), k[1]))])
    _dump(os.path.join(d, "session.json"), dict(
        role="cpu", run=R["name"], dry=bool(dry), qualified=not why, why_not_qualified="; ".join(why) or None, reached_end=not open_slots,
        prescreen_passed=bool(ref_ok), reference_us=ref, reference_spread=sref.get("spread"),
        cpu_model=model.group(1).strip() if model else (sorted(x for x in devices if x) or [None])[0], row_sessions=sorted(labels),
        windows=len(wins), windows_passed=sum(bool(x.get("passed")) for x in wins), slots_planned=len(planned),
        slots_established=len(acc), not_established=[f"{i} r{r}" for i, r in missing]))
    _dump(os.path.join(d, "host.json"), dict(topo=topo, sessionref=sref))
    _dump(os.path.join(d, "attempt_log.json"), dict(format=2, what="every BB attempt of the CPU session (a pod's attempts.json, CPU side)",
                                                    attempts=log))
    g, no_fit = _fit(R)
    if g is not None:
        shutil.copyfile(g, os.path.join(d, "grace.json"))
    n = lambda o: sum(x["outcome"] == o for x in log)
    return dict(dir=d, rows=len(rows_out), attempts=len(log), accepted=len(acc), rejected=n("rejected"), failed=n("failed"),
                surplus=n("surplus"), slots_planned=len(planned), missing=[f"{i} r{r}" for i, r in missing], unknown_jobs=unknown,
                not_planned=[f"{i} r{r}" for i in ITEMS for r in range(1, ROUNDS + 1) if (i, r) not in planned] if not dry else [],
                qualified=not why, why_not_qualified=why, grace=g is not None, grace_missing=no_fit)
