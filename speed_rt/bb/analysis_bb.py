"""SPEED-RT BB stage's numbers: every number of rows B1-B10 of the inventory and every BB speed comparison, from the pod
sessions' records. A pure function, bb_numbers(sessions, cells, dry=False) -> dict (and table(numbers) -> the rows of the
speed-claims table, from_records(path, dry) -> the numbers of a collected records directory): no clock, no console output, no
random numbers, so the same records always give the same dict (json.dumps(..., sort_keys=True) is byte-stable). Standard library only.
Two record sets (CURRENT RULES of 13 October 15:45): the GPU pod's session (speed_rt/bb/pod_bb.sh, role "gpu": r11, r12, g2, graph32 and
CUDA-Q's BP-OSD, timed host to host on the pod, not converted) and the CPU worker's Linux workstation session (every CPU part: sector Relay-BP,
ldpc BP-OSD, full-DEM Relay-BP and our pairs' Relay-BP terms; speed_rt/bb/cpu_items.json, cpu_item.sh). Every CPU time is put on the Grace
basis: raw Linux workstation time x one factor, the one the fit of speed_rt/grace_fit.py gives (a common factor when plain and correlated
PyMatching agree within their spread, else the geometric mean of the two: Relay-BP, full-DEM Relay-BP and BP-OSD are neither). The raw times
are kept beside the converted ones, and the verdicts on the raw Linux workstation times are kept in the outputs as a sensitivity check
("sensitivity_raw_xeon"). Composite = GPU term + handoff (host-side allowance) + Grace-basis CPU term; a baseline and our CPU term are both Linux workstation times.
The statistics are the paper's (the registered bb.stage3 / bb.stage4_gross `_count`), over slots (one process of one item in one
round): an item is established only when every slot has an accepted attempt (pod_bb.sh, attempts.py); otherwise its value and every
comparison that uses it are withheld (None, with a warning), and no other process or earlier value stands in.
  ours      GPU term = the slowest of the accepted fresh processes (5 on BB72, 3 on gross, 5 for the fp32 graph); handoff = the frozen
            host-side allowance, 1.1 us (the measured pinned-memory round trip of one shot's copies is a transfer: neglected, as in NVIDIA's
            setting, and reported as context beside it); CPU term = the slowest of 3 repetitions of (routed fraction x mean time on the first
            N routed shots); composite = GPU term + handoff + CPU term (the fast mode and a network alone have no CPU term)
  baselines the fastest of 3 (sector Relay-BP = relay, ldpc BP-OSD = bposd, full-DEM Relay-BP = relay_full, CUDA-Q BP-OSD = gpu_bposd)
  bar       a fast or balanced mode and a network alone: the fastest of sector, ldpc and CUDA-Q; an accurate mode: its own bar,
            full-DEM Relay-BP
A pod whose session did not qualify (no session.json, it did not reach its end, its pre-screen, start check or end check failed) is
withheld whole. Inputs: gpu/ (rows.csv, attempts.json, session.json, prescreen/prescreen.json), cpu/ (rows.csv, accepted.json, session.json,
grace.json, host.json). Output keys: pods (CPU model, quota, core, pre-screen, slots, the windows' host numbers), grace, cpu_raw, pairs (every mode of the three code settings: parts with every process's
value and range, composite, CPU share, bar, ratio), baselines, B5 ... B10, warnings, complete (every number present).
    python speed_rt/bb/analysis_bb.py --records RECORDS --out numbers.json --table table.json [--dry]
        RECORDS = collect.sh's directory (gpu/, cpu/); run from the tree root; numbers.json and table.json are written
        once and are byte-identical on every run
    python analysis_bb.py --selftest ROOT [--graph32 stage3_gross/rows_gpu.csv]    the registered records give the registered numbers
        (the earlier studies' rows, counted in the legacy way: module acceptance only)"""
import argparse
import csv
import hashlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))              # the tree root (speed_rt/bb/ -> root): the cells' deployed copy
HANDOFF_MS = 1.1e-3                                        # the registered handoff floor
FIRST = {"BB72-3S": 1000, "BB72-2S": 1000, "BB72-3F": 1000, "BB72-2F": 5000, "G-4R": 300, "G-4F": 25}   # routed shots timed (gross cut, STAGE2_bb)
OPP_N = {("bb72", "relay"): 10000, ("bb72", "bposd"): 10000, ("bb72", "relay_full"): 2000,
         ("bb144", "relay"): 1000, ("bb144", "bposd"): 500, ("bb144", "relay_full"): 75}
GATING = ("sector", "ldpc", "cudaq")
BAR_FULL_DEM = ("BB72-3F", "G-4F")        # cells whose bar is full-DEM Relay-BP itself (cells.json speed "relay_full_own"; stage 4's
#                                           CELLS bar "relay_full"); every other mode's bar is the fastest of sector, ldpc and CUDA-Q
GPU_NAME, CPU_MARK = "NVIDIA H100 80GB HBM3", "Ryzen 9 9950X"      # the registered pod GPU; the CPU host of the CPU worker's rows (their "device" column)
BASE = {"sector": "relay", "ldpc": "bposd", "full_dem": "relay_full"}                 # paper name -> bb.timing opponent name
# pair name, code, p, split, section, [(mode, network, cell)]; the accurate cell's fallback is full-DEM, the others' sector
PAIRS = [("bb72_p0.003", "bb72", 0.003, "test", "bb72", [("fast", "r11", None), ("second_network_alone", "r12", None),
                                                          ("balanced", "r12", "BB72-3S"), ("accurate", "r12", "BB72-3F")]),
         ("bb72_p0.002", "bb72", 0.002, "test", "bb72", [("fast", "r12", None), ("balanced", "r12", "BB72-2S"),
                                                          ("accurate", "r12", "BB72-2F")]),
         ("gross_p0.004", "bb144", 0.004, "test4", "gross", [("network_alone", "g2", None), ("balanced", "g2", "G-4R"),
                                                              ("accurate", "g2", "G-4F")])]


def load_rows(src):
    """Rows of a CSV path, a list of paths or a list of rows, as dicts with value_ms, p, n as numbers and note as a dict."""
    if isinstance(src, str):
        src = [src]
    out = []
    for item in src:
        for r in (csv.DictReader(open(item)) if isinstance(item, str) else [item]):
            r = dict(r)
            for k, f in (("value_ms", float), ("p", float), ("n", int)):
                v = r.get(k)
                r[k] = f(v) if v not in (None, "") else None
            note = r.get("note")
            r["note"] = json.loads(note) if isinstance(note, str) and note.startswith("{") else (note if isinstance(note, dict) else {})
            out.append(r)
    return out


def _json(x):
    return json.load(open(x)) if isinstance(x, str) else x


def grace_factor(g):
    """The factor every BB CPU time takes (CURRENT RULES, Linux workstation timing): g = the result of speed_rt/grace_fit.py's fit (factors, spread,
    common, applied, points). When plain and correlated PyMatching agree within their spread, every CPU time takes the common factor;
    otherwise the matching decoders take their own and every other CPU decoder (Relay-BP, full-DEM Relay-BP, BP-OSD) the geometric mean of
    the two. g may also be the stage's fit_s2.json (speed_rt/analysis_s.fit_s2), which holds that result under "fit" (null: no fit -> None)."""
    if "fit" in g and "factors" not in g:
        g = g["fit"]
        if g is None:
            return None
    f = g["factors"]
    if "other" in g["applied"]:                                  # speed_rt/grace_fit.py's own factor for "other" CPU decoders
        bb = g["applied"]["other"]
        rule = ("common factor: plain and correlated PyMatching agree within their spread" if g["common"]
                else "geometric mean of the plain and the correlated factor: they differ by more than their spread")
    elif g["common"]:
        bb, rule = f["common"], "common factor: plain and correlated PyMatching agree within their spread"
    else:
        bb, rule = math.sqrt(f["plain"] * f["corr"]), "geometric mean of the plain and the correlated factor: they differ by more than their spread"
    return dict(bb_factor=bb, rule=rule, factors=f, spread=g["spread"], common=g["common"], applied=g["applied"], points=g.get("points"))


def load_cells(root):
    """The three cell records of a release tree."""
    s3 = os.path.join(root, "bb", "bb72", "ising_compare", "data", "bb", "stage3")
    s4 = os.path.join(root, "bb", "gross_stage4", "ising_compare", "data", "bb", "stage4_gross")
    return {"stage3": _json(os.path.join(s3, "cells.json")), "amend": _json(os.path.join(s3, "amendment_S.json")),
            "gross": _json(os.path.join(s4, "cell.json"))}


class _Pick:
    """Counting of rows into one value, collecting warnings."""

    def __init__(self, warnings):
        self.w = warnings
        self.devices = {"gpu": set(), "cudaq": set(), "cpu": set()}     # the "device" column of the rows that counted

    def __call__(self, rows, k, how, label, ok=lambda r: True, **match):
        sel = []
        for r in rows:
            if all((round(r[key], 9) if key == "p" else r.get(key)) == (round(v, 9) if key == "p" else v)
                   for key, v in match.items()) and ok(r):
                sel.append(r)
        if len({r["process"] for r in sel}) != len(sel):
            self.w.append(f"{label}: {len(sel)} rows from {len({r['process'] for r in sel})} processes")
            return dict(value_ms=None, n=len(sel), all_ms=[r["value_ms"] for r in sel], _rows=sel[:0], _sel=[])
        if len(sel) < k:
            self.w.append(f"{label}: {len(sel)} counted rows, need {k}")
            return dict(value_ms=None, n=len(sel), all_ms=[r["value_ms"] for r in sel], _rows=sel[:0], _sel=[])
        if len(sel) > k:
            self.w.append(f"{label}: {len(sel)} counted rows, the first {k} count")
            sel = sel[:k]
        sessions = sorted({r["session"] for r in sel})
        if len(sessions) > 1:
            self.w.append(f"{label}: rows of {len(sessions)} sessions {sessions}")
        v = [r["value_ms"] for r in sel]
        cat = "cudaq" if label.endswith("cudaq") else ("gpu" if label.endswith(("gpu_term", "handoff")) else "cpu")
        self.devices[cat].update(r.get("device", "") for r in sel)
        val = max(v) if how == "max" else min(v)
        return dict(value_ms=val, n=len(v), all_ms=v, min_ms=min(v), max_ms=max(v), spread=max(v) / min(v) - 1 if min(v) else None,
                    processes=[r["process"] for r in sel], _rows=[sel[v.index(val)]], _sel=sel)


def _clean(x):
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in sorted(x.items()) if not k.startswith("_")}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return x


def _sum(*parts):
    vs = [p["value_ms"] for p in parts]
    return None if any(v is None for v in vs) else sum(vs)


def _ratio(a, b):
    return None if a is None or b is None or not b else a / b


def _cmp(name, ours, base, base_name, relation="below"):
    """A comparison of the paper: ours strictly below (or, relation "above", strictly above) the baseline."""
    holds = None if ours is None or base is None else (ours < base if relation == "below" else ours > base)
    return dict(name=name, ours_ms=ours, baseline_ms=base, baseline=base_name, relation=relation, holds=holds,
                ours_over_baseline=_ratio(ours, base))


class _Pod:
    """One pod session's records: its rows, its attempts and its qualification. Registered format (attempts given): an item's
    value exists only when every slot (round 1 .. need) has an accepted attempt and the session qualified (it reached its end,
    its pre-screen, start check and end check passed); the rows of an item are those of its accepted attempts' processes.
    Legacy format (attempts None: the earlier studies' registered rows, the selftest): GPU rows count when the module accepted them."""

    def __init__(self, role, rows, attempts, session, prescreen, cells, dry, pick, warnings):
        self.role, self.rows, self.cells, self.dry, self.pick, self.w = role, rows, cells, dry, pick, warnings
        self.att = None if attempts is None else _json(attempts)["attempts"]
        self.session, self.pre = session, prescreen
        self.reason = None
        if self.att is not None:
            if not session:
                self.reason = "no session.json (the session did not reach its end or was not collected)"
            elif not session.get("qualified"):
                self.reason = (f"the session did not qualify (reached_end {session.get('reached_end')}, pre-screen passed "
                               f"{session.get('prescreen_passed')}, {session.get('why_not_qualified')})")
            elif role == "gpu" and not (prescreen or {}).get("passed"):
                self.reason = "no passed pre-screen"
        if self.reason:
            warnings.append(f"pod {role}: {self.reason}; every comparison that uses this pod is withheld")

    def need(self, item):
        if self.dry:
            return 1
        if item == "g2":
            return self.cells["gross"].get("gpu_processes", 3)
        return 5 if item in ("r11", "r12", "graph32") else 3

    def counted(self, item):
        """The processes of the item's established slots: a set, or None when a slot is not established (a warning says which)."""
        k = self.need(item)
        procs, missing = set(), []
        for r in range(1, k + 1):
            acc = [a for a in self.att if a["item"] == item and a["round"] == r and a["outcome"] == "accepted"]
            if len(acc) != 1:
                missing.append(r)
            else:
                procs.update(acc[0]["processes"])
        if missing:
            return None, f"{len(missing)} of {k} slots not established (rounds {missing})"
        return procs, ""

    def get(self, label, item, how, ok=lambda r: True, **match):
        """The counted value of an item: _Pick's result (value_ms None when withheld)."""
        k = self.need(item)
        none = dict(value_ms=None, n=0, all_ms=[], _rows=[], _sel=[])
        if self.reason:
            return none
        if self.att is None:                                    # legacy rows
            gpu = item in GPU_ITEMS
            return self.pick(self.rows, k, how, label, lambda r: ok(r) and (bool(r["note"].get("accepted")) if gpu else True), **match)
        procs, why = self.counted(item)
        if procs is None:
            self.w.append(f"{label}: {why} on pod {self.role}")
            return none
        res = self.pick(self.rows, k, how, label, lambda r: ok(r) and r["process"] in procs, **match)
        res["pod"] = self.role
        return res

    def summary(self):
        """What the analysis publishes about the pod: host, session, slots, replacements, the windows' host numbers."""
        pre, ses = self.pre or {}, self.session or {}
        att = self.att or []
        by = {}
        for a in att:
            by.setdefault(a["outcome"], []).append(f"{a['item']} r{a['round']} a{a['attempt']}")
        wins = [a["watch"] for a in att if a.get("watch")]
        oth = [w["host"]["core_other_busy"] for w in wins if (w.get("host") or {}).get("core_other_busy") is not None]
        stl = [w["host"]["core_steal"] for w in wins if (w.get("host") or {}).get("core_steal") is not None]
        hi = pre.get("host") or {}
        cpu = pre.get("cpu") or {}
        return dict(role=self.role, reason_withheld=self.reason, cpu_model=hi.get("cpu") or ses.get("cpu_model"), quota_cpus=cpu.get("quota_cpus"),
                    cpuset=hi.get("cpuset"), card=hi.get("card"), core=pre.get("core"), prescreen_passed=pre.get("passed"),
                    steal_pct_5s=cpu.get("steal_pct_5s"), session_qualified=ses.get("qualified"), not_established=ses.get("not_established"),
                    attempts={k: len(v) for k, v in sorted(by.items())}, not_accepted=sorted(x for k, v in by.items() if k != "accepted" for x in v),
                    windows=len(wins), window_other_busy_max=max(oth) if oth else None, window_steal_max=max(stl) if stl else None,
                    window_host_numbers="recorded, not judged (a pod times GPU work only)" if self.role == "gpu" else None)


GPU_ITEMS = ("r11", "r12", "g2", "graph32", "cq72p3", "cq72p2", "cqg")
CUDAQ_ITEM = {("bb72", 0.003): "cq72p3", ("bb72", 0.002): "cq72p2", ("bb144", 0.004): "cqg"}
ROLES = ("gpu", "cpu")


def pod_of_mode(pname, mode):
    return "gpu"                                  # every network term (GPU term, handoff) is the GPU pod's


def pod_of_base(pname, base):
    return "gpu" if base == "cudaq" else "cpu"    # CUDA-Q is timed on the GPU pod; the CPU baselines on the Linux workstation


def bb_numbers(sessions, cells, dry=False, raw_basis=False):
    """sessions: {"gpu": S, "cpu": S}, each S = dict(rows=path(s) or rows, attempts=path or dict or None, session=session.json (dict or
    path) or None, prescreen=prescreen.json (gpu; dict or path) or None); the cpu S also holds grace=grace.json (dict or path); a role that
    is absent is withheld. The GPU pod's records give the network terms, the handoffs, graph32 and CUDA-Q; the CPU worker's records
    give sector Relay-BP, ldpc BP-OSD, full-DEM Relay-BP and our CPU terms, converted to the Grace basis (factor from grace); with
    raw_basis=True the CPU times stay the raw Linux workstation times (the sensitivity check)."""
    warnings = []
    pick = _Pick(warnings)
    cells = load_cells(cells) if isinstance(cells, str) else {k: _json(v) for k, v in cells.items()}
    sha = {"r11": cells["stage3"]["cells"]["BB72-3"]["net_sha256"], "r12": cells["stage3"]["cells"]["BB72-3F"]["net_sha256"],
           "g2": cells["gross"]["net_sha256"]}
    assert sha["r12"] == cells["stage3"]["cells"]["BB72-2"]["net_sha256"] == cells["amend"]["cells"]["BB72-3S"]["net_sha256"]
    gname = {"r11": f"fused:{sha['r11'][:12]}", "r12": f"fused:{sha['r12'][:12]}", "g2": cells["gross"]["gpu_name"]}
    pods, grace, cpu_raw = {}, None, []
    for role in ROLES:
        s = sessions.get(role)
        if s is None:
            warnings.append(f"{role}: no records; every comparison that uses these records is withheld")
            pods[role] = _Pod(role, [], {"attempts": []}, None, None, cells, dry, pick, warnings)
            pods[role].reason = "no records"
            continue
        rows = load_rows(s["rows"])
        if role == "cpu":
            grace = grace_factor(_json(s["grace"])) if s.get("grace") else None
            if raw_basis:
                grace = dict(bb_factor=1.0, rule="raw Linux workstation times (sensitivity check)")
            if grace is None:
                warnings.append("cpu: no Grace fit (grace.json); the CPU times cannot be put on the Grace basis, every CPU time is withheld")
            for r in rows:                                          # the raw Linux workstation time stays in raw_ms; value_ms is the Grace-basis time
                r["raw_ms"] = r["value_ms"]
                r["value_ms"] = None if grace is None or r["raw_ms"] is None else r["raw_ms"] * grace["bb_factor"]
                if "mean_ms_routed" in r["note"] and grace is not None:
                    r["note"]["raw_mean_ms_routed"] = r["note"]["mean_ms_routed"]
                    r["note"]["mean_ms_routed"] = r["note"]["mean_ms_routed"] * grace["bb_factor"]
            if grace is None:
                rows = []
        pods[role] = _Pod(role, rows, s.get("attempts"), _json(s["session"]) if s.get("session") else None,
                          _json(s["prescreen"]) if s.get("prescreen") else None, cells, dry, pick, warnings)
        if role == "cpu":
            cpu_raw = [dict(item=r.get("name"), code=r.get("code"), p=r.get("p"), role=r.get("role"), proc=r.get("proc"), process=r.get("process"),
                            raw_ms=r["raw_ms"], grace_ms=r["value_ms"], factor=None if grace is None else grace["bb_factor"])
                       for r in rows if r.get("role") in ("opponent", "cpu_term")]
    net = {}
    legacy = any(p.att is None for p in pods.values())
    for n, roles in (("r11", ("gpu",)), ("r12", ("gpu",)), ("g2", ("gpu",))):
        for role in roles:
            g = pods[role].get(f"{n} gpu_term", n, "max", role="gpu_term", name=gname[n])
            tr = pods[role].get(f"{n} handoff", n, "max", role="handoff", name=gname[n])      # the measured transfers (px.hbb): context
            if legacy:                                                                      # the earlier registered rule: max(1.1 us, measured)
                h = tr if tr["value_ms"] is None else dict(tr, value_ms=max(HANDOFF_MS, tr["value_ms"]), min_ms=max(HANDOFF_MS, tr["min_ms"]),
                                                          max_ms=max(HANDOFF_MS, tr["max_ms"]), floor_ms=HANDOFF_MS)
            else:                                                                           # CURRENT RULES: transfers neglected, host-side work stays
                h = dict(value_ms=HANDOFF_MS, min_ms=HANDOFF_MS, max_ms=HANDOFF_MS, n=0, all_ms=[], floor_ms=HANDOFF_MS,
                         what="the frozen host-side allowance (DESIGN 5.3 (c)); the measured CPU-GPU transfer is context, see transfer")
            net[(n, role)] = {"gpu_term": g, "handoff": h, "transfer": tr}
    g32 = pods["gpu"].get("graph32 gpu_term", "graph32", "max", role="gpu_term", name=f"graph32:{sha['g2'][:12]}")
    g32h = pods["gpu"].get("graph32 handoff", "graph32", "max", role="handoff", name=f"graph32:{sha['g2'][:12]}")
    cell_first = dict(FIRST)
    pairs, base = {}, {}
    for pname, code, p, split, sec, modes in PAIRS:
        b = {}
        for pap, dec in BASE.items():
            pod = pods[pod_of_base(pname, pap)]
            b[pap] = pod.get(f"{pname} {pap}", f"{dec}_p{p}", "min", role="opponent", name=dec, code=code, p=p,
                             ok=lambda r, split=split: r["note"].get("split") == split)
            for r in ([] if dry else b[pap]["_sel"]):
                if r["n"] != OPP_N[(code, dec)]:
                    warnings.append(f"{pname} {pap}: a row timed {r['n']} shots, registered {OPP_N[(code, dec)]}")
                    break
        pod = pods[pod_of_base(pname, "cudaq")]
        b["cudaq"] = pod.get(f"{pname} cudaq", CUDAQ_ITEM[(code, p)], "min", role="opponent", name="gpu_bposd", code=code, p=p)
        fast_of = [b[x]["value_ms"] for x in GATING]
        b["fastest_gating"] = dict(value_ms=None if any(v is None for v in fast_of) else min(fast_of),
                                   name=None if any(v is None for v in fast_of) else min(GATING, key=lambda x: b[x]["value_ms"]))
        base[pname] = b
        out = {}
        for mode, n, cell in modes:
            mp = pod_of_mode(pname, mode)
            g, h = net[(n, mp)]["gpu_term"], net[(n, mp)]["handoff"]
            m = {"network": n, "pod": mp, "gpu_term": g, "handoff": h, "transfer": net[(n, mp)]["transfer"]}
            if cell:
                cpu_t = pods["cpu"].get(f"{cell} cpu_term", f"cpu_{cell}", "max", role="cpu_term", name=cell, code=code, p=p,
                                     ok=lambda r, split=split: r["note"].get("split") == split)
                m["cpu_term"] = cpu_t
                if cpu_t["_rows"]:
                    nt = dict(cpu_t["_rows"][0]["note"], **cpu_t["_rows"][0]["note"].get("population", {}))   # the gross cells keep routed, shots, timed in "population"
                    m["slowest_cpu_process"] = {k: nt.get(k) for k in ("routed", "shots", "timed", "routed_fraction", "mean_ms_routed")}
                    for r in ([] if dry else cpu_t["_sel"]):
                        nt2 = dict(r["note"], **r["note"].get("population", {}))
                        if nt2.get("timed") != min(cell_first[cell], nt2.get("routed", 0)):
                            warnings.append(f"{cell}: a process timed {nt2.get('timed')} routed shots, "
                                            f"expected min({cell_first[cell]}, routed {nt2.get('routed')})")
                            break
                m["fallback"] = "full_dem" if cell.endswith("F") else "sector"
            parts = [g, h] + ([m["cpu_term"]] if cell else [])
            m["composite_ms"] = _sum(*parts)
            m["composite_range_ms"] = None if m["composite_ms"] is None else [sum(x["min_ms"] for x in parts), sum(x["max_ms"] for x in parts)]
            m["cpu_share"] = _ratio(m["cpu_term"]["value_ms"], m["composite_ms"]) if cell else 0.0
            bar_name = "full_dem" if cell in BAR_FULL_DEM else "fastest_gating"
            bar = b["full_dem"]["value_ms"] if bar_name == "full_dem" else b["fastest_gating"]["value_ms"]
            m["bar"] = dict(name=bar_name if bar_name == "full_dem" else b["fastest_gating"]["name"], value_ms=bar,
                            gating_set=None if bar_name == "full_dem" else list(GATING),
                            pod="cpu" if bar_name == "full_dem" else "cpu (sector, ldpc), gpu (CUDA-Q)")
            m["below_bar"] = None if bar is None or m["composite_ms"] is None else m["composite_ms"] < bar
            m["over_bar"] = _ratio(m["composite_ms"], bar)
            m["vs_fastest_gating"] = _cmp(f"{pname} {mode} vs the fastest of sector, ldpc, CUDA-Q", m["composite_ms"],
                                          b["fastest_gating"]["value_ms"], b["fastest_gating"]["name"])
            out[mode] = m
        pairs[pname] = out
    p3, p2, gr = pairs["bb72_p0.003"], pairs["bb72_p0.002"], pairs["gross_p0.004"]
    b3, b2, bg = base["bb72_p0.003"], base["bb72_p0.002"], base["gross_p0.004"]
    cv = lambda m: m["composite_ms"]
    acc = {"bb72_p0.003": (p3["accurate"], b3), "bb72_p0.002": (p2["accurate"], b2), "gross_p0.004": (gr["accurate"], bg)}
    b5 = {k: dict(accurate_ms=cv(m), full_dem_ms=b["full_dem"]["value_ms"], full_dem_over_accurate=_ratio(b["full_dem"]["value_ms"], cv(m)),
                  accurate_below_full_dem=None if cv(m) is None or b["full_dem"]["value_ms"] is None else cv(m) < b["full_dem"]["value_ms"])
          for k, (m, b) in sorted(acc.items())}
    r5 = [v["full_dem_over_accurate"] for v in b5.values() if v["full_dem_over_accurate"] is not None]
    b5["range"] = dict(min=min(r5) if r5 else None, max=max(r5) if r5 else None, n=len(r5))
    cmp6 = [_cmp("fast p0.003 vs the fastest baseline (paper: a ninth of CUDA-Q)", cv(p3["fast"]), b3["fastest_gating"]["value_ms"],
                 b3["fastest_gating"]["name"]),
            _cmp("fast p0.002 vs the fastest baseline (paper: a fifth of ldpc)", cv(p2["fast"]), b2["fastest_gating"]["value_ms"],
                 b2["fastest_gating"]["name"]),
            _cmp("gross network alone vs the fastest baseline (paper: a fiftieth of CUDA-Q)", cv(gr["network_alone"]),
                 bg["fastest_gating"]["value_ms"], bg["fastest_gating"]["name"]),
            _cmp("accurate p0.003 slower than CUDA-Q (paper: slower)", cv(p3["accurate"]), b3["cudaq"]["value_ms"], "cudaq", "above"),
            _cmp("gross balanced vs sector Relay-BP", cv(gr["balanced"]), bg["sector"]["value_ms"], "sector"),
            _cmp("gross balanced vs the fastest baseline", cv(gr["balanced"]), bg["fastest_gating"]["value_ms"],
                 bg["fastest_gating"]["name"])]
    fd3, sec3 = b3["full_dem"]["value_ms"], b3["sector"]["value_ms"]
    b7 = dict(full_dem_p0003_ms=fd3, sector_p0003_ms=sec3, full_dem_over_sector=_ratio(fd3, sec3))
    fdg = bg["full_dem"]["value_ms"]
    b8 = dict(gross_full_dem_ms=fdg, seconds_per_shot=None if fdg is None else fdg >= 1000.0)
    b9 = {}
    for pname, mode, dec in (("bb72_p0.003", "balanced", "sector"), ("bb72_p0.002", "balanced", "sector"), ("gross_p0.004", "balanced", "sector"),
                             ("bb72_p0.003", "accurate", "full_dem"), ("bb72_p0.002", "accurate", "full_dem"),
                             ("gross_p0.004", "accurate", "full_dem")):
        m, bs = pairs[pname][mode], base[pname][dec]["value_ms"]
        mean = (m.get("slowest_cpu_process") or {}).get("mean_ms_routed")
        b9[f"{pname} {mode}"] = dict(mean_ms_routed=mean, baseline=dec, baseline_ms=bs, routed_over_baseline_mean=_ratio(mean, bs),
                                     cpu_share=m["cpu_share"], pod=m["pod"])
    shares = [v["cpu_share"] for v in b9.values() if v["cpu_share"] is not None]
    r_sec = [v["routed_over_baseline_mean"] for k, v in sorted(b9.items()) if v["baseline"] == "sector" and v["routed_over_baseline_mean"] is not None]
    r_full = [v["routed_over_baseline_mean"] for k, v in sorted(b9.items()) if v["baseline"] == "full_dem" and v["routed_over_baseline_mean"] is not None]
    b9["ranges"] = dict(sector=[min(r_sec), max(r_sec)] if len(r_sec) == 3 else None,
                        full_dem=[min(r_full), max(r_full)] if len(r_full) == 3 else None,
                        cpu_share=[min(shares), max(shares)] if len(shares) == 6 else None,
                        what="each routed-over-baseline ratio is computed inside the CPU worker's session (both on the Grace basis, one factor); the CPU share divides the Grace-basis CPU term by the composite (GPU pod + the Linux workstation)")
    g2g, g32v = net[("g2", "gpu")]["gpu_term"]["value_ms"], g32["value_ms"]
    b10 = dict(r11_gpu_term_us=None if net[("r11", "gpu")]["gpu_term"]["value_ms"] is None else 1e3 * net[("r11", "gpu")]["gpu_term"]["value_ms"],
               gross_gpu_term_ms=g2g, graph32_gpu_term_ms=g32v, graph32_handoff_ms=g32h["value_ms"],
               graph32=dict(gpu_term=g32, handoff=g32h), gross_below_graph32=None if g2g is None or g32v is None else g2g < g32v,
               graph32_over_gross=_ratio(g32v, g2g), descriptive=True, pod="gpu")
    dev = {k: sorted(x for x in v if x) for k, v in pick.devices.items()}
    hw_ok = bool(dev["gpu"] and dev["cudaq"] and dev["cpu"] and all(x == GPU_NAME for x in dev["gpu"] + dev["cudaq"])
                 and all(CPU_MARK in x for x in dev["cpu"]))
    if not hw_ok:
        warnings.append(f"hardware: rows from {dev}, expected GPU {GPU_NAME!r} and a CPU named {CPU_MARK}")
    res = dict(basis="accepted attempts of the qualified GPU pod session and accepted processes of the CPU worker's session (Grace basis)"
               if not any(p.att is None for p in pods.values()) else "legacy rows (module acceptance only)", dry=bool(dry),
               grace=grace, cpu_raw=cpu_raw,
               hardware=dict(devices=dev, expected_gpu=GPU_NAME, expected_cpu=CPU_MARK, ok=hw_ok),
               pods={r: pods[r].summary() for r in ROLES}, pairs=pairs, baselines=base, B5=b5, B6=cmp6, B7=b7, B8=b8, B9=b9, B10=b10,
               networks={n: dict(sha256=sha[n], gpu_name=gname[n]) for n in sha}, warnings=sorted(set(warnings)))
    res = _clean(res)
    res["complete"] = _all_present({k: v for k, v in res.items() if k != "pods"})
    return res


def from_records(path, dry=False):
    """The numbers of RECORDS (path), from the layout collect.sh writes, with the cells of this tree (ROOT):
        gpu/   a pod_bb.sh output: rows.csv, attempts.json, session.json, prescreen/prescreen.json
        cpu/   the CPU worker's Linux workstation session: rows.csv (the modules' rows of every CPU process), accepted.json (the processes its windows
               accepted: [{"item", "round", "process"}]), session.json ({"qualified": bool, "why_not_qualified": text, ...}), grace.json (the fit
               of speed_rt/grace_fit.py), host.json (the machine's record, published as it is)
    A part whose directory is absent is withheld. dry=True counts one process per item (dry runs only). The dict is bb_numbers'
    plus "inputs" (sha256 of every file read): JSON-serialisable and deterministic, so table(from_records(...)) gives the same
    rows every time."""
    sessions, inputs = {}, {}
    for role in ROLES:
        d = os.path.join(path, role)
        if not os.path.isdir(d):
            continue
        if role == "gpu":
            f = dict(rows=os.path.join(d, "rows.csv"), attempts=os.path.join(d, "attempts.json"), session=os.path.join(d, "session.json"),
                     prescreen=os.path.join(d, "prescreen", "prescreen.json"))
        else:
            f = dict(rows=os.path.join(d, "rows.csv"), accepted=os.path.join(d, "accepted.json"), session=os.path.join(d, "session.json"),
                     grace=os.path.join(d, "grace.json"), host=os.path.join(d, "host.json"))
        for k, v in f.items():
            if os.path.exists(v):
                inputs[os.path.relpath(v, path)] = hashlib.sha256(open(v, "rb").read()).hexdigest()
        if not os.path.exists(f["rows"]):
            continue
        if role == "cpu":
            if not os.path.exists(f["accepted"]):
                continue
            acc = _json(f["accepted"])
            f["attempts"] = {"attempts": [dict(item=a["item"], round=a["round"], attempt=1, outcome="accepted", processes=[a["process"]], watch=None)
                                          for a in acc]}
            del f["accepted"], f["host"]
        elif not os.path.exists(f["attempts"]):
            continue
        sessions[role] = {k: (v if not isinstance(v, str) or os.path.exists(v) else None) for k, v in f.items()}
    nb = bb_numbers(sessions, ROOT, dry)
    raw = bb_numbers(sessions, ROOT, dry, raw_basis=True)                # the same verdicts on the raw Linux workstation times (a sensitivity check)
    nb["sensitivity_raw_xeon"] = {k: raw[k] for k in ("pairs", "baselines", "B5", "B6", "B7", "B8", "B9")}
    nb["inputs"] = dict(sorted(inputs.items()))
    return json.loads(json.dumps(nb, sort_keys=True))


def load_rows_jsonl(src):
    out = []
    for item in ([src] if isinstance(src, str) else src):
        if isinstance(item, dict):
            out.append(item)
        else:
            out.extend(json.loads(ln) for ln in open(item) if ln.strip())
    return out


def _all_present(x):
    """No None among the numbers (an absent value is None)."""
    if isinstance(x, dict):
        return all(_all_present(v) for k, v in x.items() if k not in ("slowest_cpu_process", "floor_ms", "spread", "busy", "gating_set", "idle_flagged"))
    if isinstance(x, list):
        return all(_all_present(v) for v in x)
    return x is not None


def summary(res):
    m = lambda pair, mode: res["pairs"][pair][mode]["composite_ms"]
    f = lambda v: "n/a" if v is None else f"{v:.4g}"
    return (f"BB numbers ({res['basis']}): p0.3% fast {f(m('bb72_p0.003', 'fast'))} balanced {f(m('bb72_p0.003', 'balanced'))} "
            f"accurate {f(m('bb72_p0.003', 'accurate'))} | p0.2% fast {f(m('bb72_p0.002', 'fast'))} balanced "
            f"{f(m('bb72_p0.002', 'balanced'))} accurate {f(m('bb72_p0.002', 'accurate'))} | gross network "
            f"{f(m('gross_p0.004', 'network_alone'))} balanced {f(m('gross_p0.004', 'balanced'))} accurate "
            f"{f(m('gross_p0.004', 'accurate'))} ms; {sum(1 for c in res['B6'] if c['holds'])}/{len(res['B6'])} B6 comparisons hold; "
            f"complete {res['complete']}; {len(res['warnings'])} warnings")


# ---- the table of speed claims: the paper's printed numbers (old) against the re-measured ones (new) ----------------------------
TXT = {"bb72_p0.003": "BB72 p=0.3%", "bb72_p0.002": "BB72 p=0.2%", "gross_p0.004": "gross p=0.4%"}
MODE_TXT = {"network_alone": "network alone", "fast": "fast mode", "second_network_alone": "second network alone",
            "balanced": "balanced mode", "accurate": "accurate mode"}
OLD = {                                # inventory section 1.2, as printed: (value, decimals printed), ms
    "bb72_p0.003": dict(fast=(0.102, 3), second_network_alone=(0.102, 3), balanced=(0.260, 3), accurate=(3.43, 2), sector=(1.61, 2),
                        ldpc=(1.35, 2), cudaq=(0.957, 3), full_dem=(58.6, 1)),
    "bb72_p0.002": dict(fast=(0.102, 3), balanced=(0.115, 3), accurate=(0.324, 3), sector=(1.01, 2), ldpc=(0.501, 3),
                        cudaq=(0.517, 3), full_dem=(36.4, 1)),
    "gross_p0.004": dict(network_alone=(0.279, 3), balanced=(9.36, 2), accurate=(232, 0), sector=(59.0, 1), ldpc=(73.7, 1),
                         cudaq=(13.9, 1), full_dem=(2775, 0))}
OLD_CPU = {("bb72_p0.003", "balanced"): (0.1582, 4), ("bb72_p0.003", "accurate"): (3.3286, 4), ("bb72_p0.002", "balanced"): (0.0126, 4),
           ("bb72_p0.002", "accurate"): (0.2218, 4), ("gross_p0.004", "balanced"): (9.080, 3), ("gross_p0.004", "accurate"): (231.9, 1)}
OLD_SHARE = {("bb72_p0.003", "balanced"): 60.8, ("bb72_p0.003", "accurate"): 97.0, ("bb72_p0.002", "balanced"): 11.0,
             ("bb72_p0.002", "accurate"): 68.5, ("gross_p0.004", "balanced"): 97.0, ("gross_p0.004", "accurate"): 99.9}
_R3 = lambda x: float("%.3g" % x)


def _basis_ours(mode):
    return "H100 GPU term + handoff" + {"balanced": " + the Linux workstation's Ryzen, routed sector Relay-BP (one core), Grace basis",
                                        "accurate": " + the Linux workstation's Ryzen, routed full-DEM Relay-BP (one core), Grace basis"}.get(mode, "")


BASIS_THEIRS = {"sector": "sector Relay-BP, the Linux workstation's Ryzen, one core, Grace basis, fastest of 3 processes",
                "ldpc": "ldpc BP-OSD, the Linux workstation's Ryzen, one core, Grace basis, fastest of 3 processes",
                "cudaq": "CUDA-Q BP-OSD, batch 1 host to host, on the H100 pod, its host pinned to one Xeon core, fastest of 3 processes",
                "full_dem": "full-DEM Relay-BP, the Linux workstation's Ryzen, one core, Grace basis, fastest of 3 processes",
                "fastest": "the fastest of sector Relay-BP and ldpc (the Linux workstation's Ryzen, Grace basis) and CUDA-Q (the GPU pod, host to host, not converted)"}


def table(numbers):
    """The speed claims as one list of rows in inventory order (B1 ... B10; B11, the earlier gross test, is out of the re-test), from bb_numbers' dict. Number rows: row, item, kind
    "number", unit, old (as printed), old_decimals, new (None when missing), why, basis. Comparison rows: row, item, kind
    "comparison", unit, ours_old, theirs_old, ratio_old (theirs / ours; printed by the paper where ratio_old_printed is true,
    else computed from the printed values), ours, theirs, ratio (theirs / ours), holds (ours strictly below theirs; None when a
    side is missing; relation "above" marks the one sentence that says ours is slower, and then holds means strictly above),
    relation, headline (None; "H-relay" on the three B5 rows, "H5" on B7's, which the headline arithmetic reads), why, basis. Pure: the same numbers give the same rows."""
    P, B, W = numbers["pairs"], numbers["baselines"], numbers["warnings"]
    rows = []

    def why(missing, prefixes):
        if not missing:
            return None
        return "; ".join(w for w in W if any(w.startswith(x) for x in prefixes)) or "no counted rows for this item"

    def num(row, item, unit, old, dec, new, basis, pre=()):
        rows.append(dict(row=row, item=item, kind="number", unit=unit, old=old, old_decimals=dec, new=new, why=why(new is None, pre),
                         basis=basis))

    def cmp(row, item, ours_old, theirs_old, ratio_old, printed, ours, theirs, basis, pre=(), relation="below", headline=None):
        holds = None if ours is None or theirs is None else (ours < theirs if relation == "below" else ours > theirs)
        rows.append(dict(row=row, item=item, kind="comparison", unit="ms", ours_old=ours_old, theirs_old=theirs_old,
                         ratio_old=ratio_old if printed else _R3(theirs_old / ours_old), ratio_old_printed=printed, ours=ours,
                         theirs=theirs, ratio=_ratio(theirs, ours), holds=holds, relation=relation, headline=headline,
                         why=why(ours is None or theirs is None, pre), basis=basis))

    meta = {(pn, m): (n, cell) for pn, _, _, _, _, modes in PAIRS for m, n, cell in modes}

    def pre_mode(pn, m):
        n, cell = meta[(pn, m)]
        return [f"{n} gpu_term", f"{n} handoff"] + ([f"{cell} cpu_term"] if cell else [])

    def comp(pn, m):
        return P[pn][m]["composite_ms"]

    def base(pn, k):
        return B[pn][k]["value_ms"]

    def bar_old(pn, m):
        o = OLD[pn]
        return (o["full_dem"][0], "full_dem") if meta[(pn, m)][1] in BAR_FULL_DEM else (min(o[k][0] for k in GATING), "fastest")

    def bar_new(pn, m):
        return P[pn][m]["bar"]["value_ms"]

    def bar_pre(pn, m):
        return [f"{pn} full_dem"] if meta[(pn, m)][1] in BAR_FULL_DEM else [f"{pn} {k}" for k in GATING]

    names = {"sector": "sector Relay-BP", "ldpc": "ldpc BP-OSD", "cudaq": "CUDA-Q BP-OSD", "full_dem": "full-DEM Relay-BP"}
    for row, pn in (("B1", "bb72_p0.003"), ("B2", "bb72_p0.002"), ("B3", "gross_p0.004")):
        T, o = TXT[pn], OLD[pn]
        modes = [m for m in MODE_TXT if m in o]
        for m in modes:
            num(row, f"{T}: {MODE_TXT[m]} composite", "ms", o[m][0], o[m][1], comp(pn, m), "composite: " + _basis_ours(m),
                pre_mode(pn, m))
        for k in ("sector", "ldpc", "cudaq", "full_dem"):
            num(row, f"{T}: {names[k]}", "ms", o[k][0], o[k][1], base(pn, k), BASIS_THEIRS[k], [f"{pn} {k}"])
        for m in modes:
            old, kind = bar_old(pn, m)
            cmp(row, f"{T}: {MODE_TXT[m]} vs " + ("full-DEM Relay-BP" if kind == "full_dem" else "the fastest baseline"), o[m][0], old,
                None, False, comp(pn, m), bar_new(pn, m), f"ours: {_basis_ours(m)} composite; theirs: {BASIS_THEIRS[kind]}",
                pre_mode(pn, m) + bar_pre(pn, m))
    # B4: Table V
    HB = {"gpu_term": "", "handoff": "the frozen host-side allowance, 1.1 us; the CPU-GPU transfer is neglected (as in NVIDIA's setting)",
          "transfer": "context only, not in the composite: the measured pinned-memory round trip of one shot's copies"}
    NM = {"gpu_term": "network (GPU term)", "handoff": "handoff (host-side work)", "transfer": "CPU-GPU transfer (context)"}
    for key, old in (("gpu_term", 0.079), ("handoff", 0.023), ("transfer", 0.023)):
        num("B4", f"Table V BB72: {NM[key]} of the second network", "ms", old, 3, val_path(P, "bb72_p0.003", "second_network_alone", key),
            "H100 80GB HBM3, " + ("slowest of 5 fresh processes" if key == "gpu_term" else HB[key]),
            ["r12 gpu_term", "r12 handoff"])
    for key, old in (("gpu_term", 0.257), ("handoff", 0.022), ("transfer", 0.022)):
        num("B4", f"Table V gross: {NM[key]}", "ms", old, 3, val_path(P, "gross_p0.004", "network_alone", key),
            "H100 80GB HBM3, " + ("slowest of 3 fresh processes" if key == "gpu_term" else HB[key]),
            ["g2 gpu_term", "g2 handoff"])
    six = [("bb72_p0.003", "balanced"), ("bb72_p0.003", "accurate"), ("bb72_p0.002", "balanced"), ("bb72_p0.002", "accurate"),
           ("gross_p0.004", "balanced"), ("gross_p0.004", "accurate")]
    for pn, m in six:
        o, c = OLD_CPU[(pn, m)]
        num("B4", f"Table V {TXT[pn]}: {MODE_TXT[m]} routed fallback (CPU term)", "ms", o, c, val_path(P, pn, m, "cpu_term"),
            "routed fraction x mean over the first routed shots, " + BASIS_THEIRS["sector" if m == "balanced" else "full_dem"].split(",")[0]
            + ", slowest of 3 repetitions", pre_mode(pn, m)[2:])
    for pn, m in six:
        num("B4", f"Table V {TXT[pn]}: {MODE_TXT[m]} total", "ms", OLD[pn][m][0], OLD[pn][m][1], comp(pn, m), "composite: " + _basis_ours(m),
            pre_mode(pn, m))
    for pn, m in six:
        v = P[pn][m]["cpu_share"]
        num("B4", f"Table V {TXT[pn]}: {MODE_TXT[m]} CPU share of the total", "%", OLD_SHARE[(pn, m)], 1, None if v is None else 100 * v,
            "CPU term / composite", pre_mode(pn, m))
    # B5: accurate mode against full-DEM Relay-BP, "12 to 112 times less time"
    for pn, ro in (("bb72_p0.003", 17), ("bb72_p0.002", 112), ("gross_p0.004", 12)):
        o = OLD[pn]
        cmp("B5", f"{TXT[pn]}: accurate mode vs full-DEM Relay-BP", o["accurate"][0], o["full_dem"][0], ro, True, comp(pn, "accurate"),
            base(pn, "full_dem"), f"ours: {_basis_ours('accurate')} composite; theirs: {BASIS_THEIRS['full_dem']}",
            pre_mode(pn, "accurate") + [f"{pn} full_dem"], headline="H-relay")
    # B6: App. C
    f3, f2, gn, gb = ("bb72_p0.003", "fast"), ("bb72_p0.002", "fast"), ("gross_p0.004", "network_alone"), ("gross_p0.004", "balanced")
    fastest = lambda pn: B[pn]["fastest_gating"]["value_ms"]
    fo = lambda pn: min(OLD[pn][k][0] for k in GATING)
    cmp("B6", f"{TXT[f2[0]]}: fast mode vs the fastest baseline (a fifth)", 0.102, fo(f2[0]), 5, True, comp(*f2), fastest(f2[0]),
        f"ours: {_basis_ours('fast')} composite; theirs: {BASIS_THEIRS['fastest']}", pre_mode(*f2) + bar_pre(*f2))
    cmp("B6", f"{TXT[f3[0]]}: fast mode vs the fastest baseline (a ninth)", 0.102, fo(f3[0]), 9, True, comp(*f3), fastest(f3[0]),
        f"ours: {_basis_ours('fast')} composite; theirs: {BASIS_THEIRS['fastest']}", pre_mode(*f3) + bar_pre(*f3))
    cmp("B6", f"{TXT[gn[0]]}: network alone vs the fastest baseline (a fiftieth)", 0.279, fo(gn[0]), 50, True, comp(*gn), fastest(gn[0]),
        f"ours: {_basis_ours('fast')} composite; theirs: {BASIS_THEIRS['fastest']}", pre_mode(*gn) + bar_pre(*gn))
    cmp("B6", f"{TXT['bb72_p0.003']}: accurate mode slower than CUDA-Q BP-OSD", 3.43, 0.957, None, False, comp("bb72_p0.003", "accurate"),
        base("bb72_p0.003", "cudaq"), f"ours: {_basis_ours('accurate')} composite; theirs: {BASIS_THEIRS['cudaq']}",
        pre_mode("bb72_p0.003", "accurate") + ["bb72_p0.003 cudaq"], "above")
    cmp("B6", f"{TXT[gb[0]]}: balanced mode vs sector Relay-BP", 9.36, 59.0, None, False, comp(*gb), base(gb[0], "sector"),
        f"ours: {_basis_ours('balanced')} composite; theirs: {BASIS_THEIRS['sector']}", pre_mode(*gb) + [f"{gb[0]} sector"])
    cmp("B6", f"{TXT[gb[0]]}: balanced mode vs the fastest baseline", 9.36, fo(gb[0]), None, False, comp(*gb), fastest(gb[0]),
        f"ours: {_basis_ours('balanced')} composite; theirs: {BASIS_THEIRS['fastest']}", pre_mode(*gb) + bar_pre(*gb))
    # B7, B8
    cmp("B7", f"{TXT['bb72_p0.003']}: full-DEM Relay-BP vs sector Relay-BP (36 times as long)", 1.61, 58.6, 36, True,
        base("bb72_p0.003", "sector"), base("bb72_p0.003", "full_dem"),
        f"ours: {BASIS_THEIRS['sector']}; theirs: {BASIS_THEIRS['full_dem']}", ["bb72_p0.003 sector", "bb72_p0.003 full_dem"],
        headline="H5")
    num("B7", f"{TXT['bb72_p0.003']}: full-DEM Relay-BP per shot on one CPU core", "ms", 58.6, 1, base("bb72_p0.003", "full_dem"),
        BASIS_THEIRS["full_dem"], ["bb72_p0.003 full_dem"])
    num("B8", f"{TXT['gross_p0.004']}: full-DEM Relay-BP per shot (seconds)", "ms", 2775, 0, base("gross_p0.004", "full_dem"),
        BASIS_THEIRS["full_dem"], ["gross_p0.004 full_dem"])
    # B9: Sec. IX
    rg = numbers["B9"]["ranges"]
    sh = {k: v["cpu_share"] for k, v in numbers["B9"].items() if k != "ranges"}
    keep = [sh[f"{pn} {m}"] for pn, m in six if (pn, m) != ("bb72_p0.002", "balanced")]
    cs = None if any(v is None for v in keep) else [100 * min(keep), 100 * max(keep)]
    b9 = "routed-shot mean time of the cell's fallback / the same decoder's baseline mean, the Linux workstation's Ryzen, one core, both on the Grace basis"
    for nm, rng, i, old, dec in (("sector Relay-BP, low end of the range", rg["sector"], 0, 5, 0), ("sector Relay-BP, high end", rg["sector"], 1, 16, 0),
                                 ("full-DEM Relay-BP, low end", rg["full_dem"], 0, 2.5, 1), ("full-DEM Relay-BP, high end", rg["full_dem"], 1, 4.2, 1)):
        num("B9", f"routed shot vs mean shot, {nm}", "x", old, dec, None if rng is None else rng[i], b9,
            [f"{pn} {k}" for pn in OLD for k in ("sector", "full_dem")] + [f"{c} cpu_term" for c in ("BB72-3S", "BB72-2S", "BB72-3F", "BB72-2F", "G-4R", "G-4F")])
    for nm, i, old, dec in (("low end of the CPU-term share (all pairs but BB72 balanced p=0.2%)", 0, 61, 0), ("high end", 1, 99.9, 1)):
        num("B9", f"routed CPU term share of the composite, {nm}", "%", old, dec, None if cs is None else cs[i], "CPU term / composite",
            [f"{c} cpu_term" for c in ("BB72-3S", "BB72-3F", "BB72-2F", "G-4R", "G-4F")])
    num("B9", f"{TXT['bb72_p0.002']}: balanced mode CPU term share of the composite", "%", 11, 0, None if sh["bb72_p0.002 balanced"] is None
        else 100 * sh["bb72_p0.002 balanced"], "CPU term / composite", ["BB72-2S cpu_term"])
    # B10: Sec. V, App. B
    b10 = numbers["B10"]
    num("B10", "BB72 engine GPU term per shot", "us", 79.4, 1, b10["r11_gpu_term_us"], "H100 GPU term of the first network r11, slowest of 5 processes",
        ["r11 gpu_term"])
    num("B10", "gross engine GPU term per shot", "ms", 0.257, 3, b10["gross_gpu_term_ms"], "H100 GPU term of g2 (fusedt), slowest of 3 processes",
        ["g2 gpu_term"])
    num("B10", "gross fp32 PyTorch CUDA graph (graph32) GPU term per shot", "ms", 3.43, 2, b10["graph32_gpu_term_ms"],
        "H100 GPU term of graph32, slowest of 5 processes", ["graph32 gpu_term"])
    cmp("B10", "gross: engine GPU term vs the fp32 graph's", 0.257, 3.43, None, False, b10["gross_gpu_term_ms"], b10["graph32_gpu_term_ms"],
        "ours: H100 GPU term of g2 (fusedt); theirs: H100 GPU term of graph32", ["g2 gpu_term", "graph32 gpu_term"])
    return rows


def val_path(d, *path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    if isinstance(d, dict) and "value_ms" in d:
        return d["value_ms"]
    return d


REGISTERED = {                         # the registered Table I values (ms) and their sources; 5 significant digits
    ("bb72_p0.003", "fast"): 0.10187, ("bb72_p0.003", "second_network_alone"): 0.10204, ("bb72_p0.003", "balanced"): 0.26022,
    ("bb72_p0.003", "accurate"): 3.4306, ("bb72_p0.002", "fast"): 0.10204, ("bb72_p0.002", "balanced"): 0.11468,
    ("bb72_p0.002", "accurate"): 0.32379, ("gross_p0.004", "network_alone"): 0.27885, ("gross_p0.004", "balanced"): 9.3584,
    ("gross_p0.004", "accurate"): 232.1868}
REG_BASE = {("bb72_p0.003", "sector"): 1.6140, ("bb72_p0.003", "ldpc"): 1.3450, ("bb72_p0.003", "cudaq"): 0.9566,
            ("bb72_p0.003", "full_dem"): 58.5963, ("bb72_p0.002", "sector"): 1.0057, ("bb72_p0.002", "ldpc"): 0.50091,
            ("bb72_p0.002", "cudaq"): 0.51659, ("gross_p0.004", "sector"): 59.044, ("gross_p0.004", "ldpc"): 73.659,
            ("gross_p0.004", "cudaq"): 13.899, ("gross_p0.004", "full_dem"): 2774.707}


def selftest(root, graph32=None):
    """The registered records (release rows, their registered session labels) give the registered numbers. Returns (ok, lines).
    BB72's CUDA-Q rows are the 'post' session's, the networks' the 'pre' session's; full-DEM at p 0.002 has no registered row."""
    s3 = os.path.join(root, "bb", "bb72", "ising_compare", "data", "bb", "stage3")
    s4 = os.path.join(root, "bb", "gross_stage4", "ising_compare", "data", "bb", "stage4_gross")
    pre = [r for r in load_rows(os.path.join(s3, "rows_gpu.csv")) if r["session"] == "H100 session pre" and r["role"] != "opponent"]
    post = [r for r in load_rows(os.path.join(s3, "rows_gpu.csv")) if r["session"] == "H100 session post" and r["role"] == "opponent"]
    g4 = [r for r in load_rows(os.path.join(s4, "rows_gpu.csv")) if r["session"] == "H100 session s4" and r["role"] != "idle"]
    extra = [r for r in load_rows(graph32) if r["session"] == "h100-gpu-pod-pre" and r["name"].startswith("graph32")] if graph32 else []
    c72 = [r for r in load_rows([os.path.join(s3, "rows_cpu.csv"), os.path.join(s3, "rows_cpu_S.csv")])
           if r["session"] in ("M4 session B", "M4 session S")]
    cg = [r for r in load_rows(os.path.join(s4, "rows_cpu.csv")) if r["session"] == "M4 session s4"]
    legacy = {r: dict(rows=pre + post + g4 + extra + c72 + cg, attempts=None) for r in ROLES}
    res = bb_numbers(legacy, root)
    lines, ok = [], True
    for key, want in sorted(REGISTERED.items()):
        got = res["pairs"][key[0]][key[1]]["composite_ms"]
        good = got is not None and abs(got - want) <= max(6e-6, 6e-5 * want)
        ok &= good
        lines.append(f"{'ok  ' if good else 'FAIL'} {key[0]} {key[1]}: {got} vs registered {want}")
    for key, want in sorted(REG_BASE.items()):
        got = res["baselines"][key[0]][key[1]]["value_ms"]
        good = got is not None and abs(got - want) <= max(6e-6, 6e-5 * want)
        ok &= good
        lines.append(f"{'ok  ' if good else 'FAIL'} {key[0]} {key[1]}: {got} vs registered {want}")
    fd2 = res["baselines"]["bb72_p0.002"]["full_dem"]["value_ms"]
    ok &= fd2 is None
    lines.append(f"{'ok  ' if fd2 is None else 'FAIL'} bb72_p0.002 full_dem: {fd2} (the registered records hold no row)")
    one = json.dumps(res, sort_keys=True)
    again = json.dumps(bb_numbers(legacy, root), sort_keys=True)
    ok &= one == again
    lines.append(f"{'ok  ' if one == again else 'FAIL'} deterministic: two calls give identical JSON ({len(one)} bytes)")
    tok, tlines = check_table(table(res))
    ok &= tok
    lines += tlines
    lines.append(summary(res))
    return ok, lines


def check_table(rows):
    """Every table row against its printed old value: numbers within the printed rounding (half a unit of the last printed digit),
    comparison sides within 1 %, printed ratios within 0.5 and computed ones within 2 %, every comparison holding. Rows whose new
    value is missing are skipped and listed. Returns (no row failed, lines)."""
    fail, skip, good, lines = 0, 0, 0, []
    for r in rows:
        tag = f"{r['row']} {r['item']}"
        if r["kind"] == "number":
            if r["new"] is None:
                skip += 1
                lines.append(f"skip {tag}: {r['why']}")
            elif abs(r["new"] - r["old"]) <= 0.5 * 10 ** -r["old_decimals"] + 1e-9:
                good += 1
            else:
                fail += 1
                lines.append(f"FAIL {tag}: new {r['new']} old {r['old']} ({r['old_decimals']} decimals)")
        else:
            if r["ratio"] is None or r["holds"] is None:
                skip += 1
                lines.append(f"skip {tag}: {r['why']}")
                continue
            ro = r["ratio_old"]
            bad = [x for x, a, b in (("ours", r["ours"], r["ours_old"]), ("theirs", r["theirs"], r["theirs_old"])) if abs(a - b) > 0.01 * b]
            if abs(r["ratio"] - ro) > (0.5 if r["ratio_old_printed"] else 0.02 * ro):
                bad.append(f"ratio {r['ratio']:.4g} vs {ro}")
            if not r["holds"]:
                bad.append("does not hold")
            if bad:
                fail += 1
                lines.append(f"FAIL {tag}: {', '.join(str(x) for x in bad)}")
            else:
                good += 1
    lines.append(f"{'ok  ' if not fail else 'FAIL'} table: {len(rows)} rows, {good} equal their printed value, {skip} skipped (no registered record), {fail} differ")
    return fail == 0, lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records")
    ap.add_argument("--out")
    ap.add_argument("--table")
    ap.add_argument("--dry", action="store_true", help="dry records: one counted process per item")
    ap.add_argument("--selftest")
    ap.add_argument("--graph32")
    a = ap.parse_args()
    if a.selftest:
        ok, lines = selftest(a.selftest, a.graph32)
        print("\n".join(lines))
        raise SystemExit(0 if ok else 1)
    if not (a.records and a.out):
        raise SystemExit("--records and --out are required")
    res = from_records(a.records, a.dry)
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    if a.table:
        with open(a.table, "x") as fh:
            json.dump(table(res), fh, indent=1, sort_keys=True)
    print(summary(res))


if __name__ == "__main__":
    main()
