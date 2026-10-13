"""analysis_cas.py: SPEED-RT's Cascade speed numbers from the records of R1-R4 on two cards, as rows C1-C10 of the inventory
(the Cascade parts of H3 and H4 are headline flags on their rows). Every cell and every comparison is computed per card, from
that card's own records, and each card has its own verdict per claim. Pure and deterministic: analyze(records) -> numbers,
table(numbers) -> rows and verdicts(rows) -> verdicts depend on their arguments only (no clock, no console output);
load_round(), from_records() and main() do the file reading and writing.

    from speed_rt.cascade import analysis_cas
    numbers = analysis_cas.from_records(RECORDS_DIR, dry)      # RECORDS_DIR = records/cascade
    rows = analysis_cas.table(numbers); verdicts = analysis_cas.verdicts(rows)
    python analysis_cas.py --records RECORDS_DIR [--dry] --out numbers.json --table table.json --verdicts verdicts.json
    python analysis_cas.py --selftest PUBLIC_ROOT     the registered (pre-freeze) records through the same code

Layout of RECORDS_DIR (collect.sh CARD SRC... DEST copies each job's output directory into it): RECORDS_DIR/<CARD>/<ROUND>/
with CARD = RTX (Alpine's RTX PRO 6000) or H100 (the pod) and ROUND = R1 (D1-S, 36 slots), R2 (phase C pipelines, 36), R3 (M2
networks and pipelines, 33), R4 (d = 13, 48); every directory is one drive.py output directory, any of them may be absent and
its cells are then "no records".
    each: attempts.json                     every slot and attempt, the beacon, the seeds' and launchers' sha256, the gate
          slots/<slot id>/a<k>/{rows.jsonl, windows.jsonl, trace.jsonl, log.txt}   (R4 also pipes/*.npz)
dry=True: the records are dry ones (hash checks against the freeze are skipped); a record whose "dry" differs from the
argument is listed in numbers["checks"].

The statistic (the paper's, item by item): a cell's value is the slowest accepted process, on both sides of a comparison;
Cascade alone's slowest-against-slowest is the registered exception to "ours slowest, theirs fastest", and every cell also
carries its fastest accepted process, from which table() adds the fastest-Cascade sensitivity rows ("sensitivity, decides
nothing"). A process is accepted when drive.py's gate accepted it and the analysis re-derives the same verdict from the
attempt's own window records, by window_why() (the one place of the rule, with srtlib.Window's constants): no other compute
app, no sampler gap above 1 s, no clock-event reason other than GPU idle, applications clocks or SW power cap, the power limit
at the card's default, and no sample below 90 % of the maximum SM clock without the SW power cap. The workload's own power cap
and App. D's flag (a sample below 97 %) are published per card and round (numbers["cards"][CARD]["clock"]) and decide nothing.
A cell whose planned slot is not established (three rejected attempts, a failure, or a job that stopped) has no value: every
comparison that uses it is withheld with the reason. Rejected attempts stay in the numbers, as "rejected", with their
values. Every repetition's value and the range of each cell are in numbers["cards"][CARD]["cells"]; numbers["checks"] lists
every disagreement the analysis found (a row that is not its slot's, a hash that is not the freeze's, a chunk whose digest
differs between processes or cards, an accepted attempt whose windows do not hold, a burn in a registered record, a gate that
was off) and each card's "shots" the digest of every sampled chunk.
C5 (App. D's H100 paragraph) exists on the H100 card only, restated from that card's R3 networks and pipelines and R2 chain;
it is descriptive and decides nothing, and its one RTX value is context taken from the RTX card.

Units: times in microseconds; per round = per shot / 6 (BB72) or / 13 (d = 13); p99.9 of C7 in ms. holds = ours strictly
below theirs, where for the tail statements (C7, C8's and C10's 99.9th percentile) "ours" is Cascade alone, so that "holds"
says the paper's statement (the routed decoder's tail is higher) is true. A claim's verdict (verdicts()) is "holds" when every
deciding comparison of its row holds on that card, "fails" when one does not, "withheld" when one has no value."""
import argparse
import json
import os
import sys

try:                                                    # the constants of the rule are srtlib.Window's own
    from . import srtlib
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import srtlib

CARDS = ("RTX", "H100")
HW = {"RTX": "RTX PRO 6000, node EPYC 9555", "H100": "RunPod H100 80GB HBM3 SXM pod"}
R_BB, R_D13 = 6, 13
FREEZE = {                                              # both cards run the same checkpoints and locks; a d = 13 lock of
    "speed_sha256": "1ec6b2563060f0042f866f19841631e8e1499872e6c15ed77dae7a01c8c811f5",   # the H100 that differs from the
    "locks": {"r1": "3ab39941df356883be395ef7bbf39672fdb546729187bd89fa523be22714f96c",    # RTX's must be added here
              "r2": "398d02c60e08535a16a1d37a325e3aec9537f8a3608c794c985f471c0be1c41d",
              "r3": "398d02c60e08535a16a1d37a325e3aec9537f8a3608c794c985f471c0be1c41d"},
    "nets": {"S55": "87886c65fbb6cc4955b9a05725e837f38f0f7081c28c89fcf448cb1700943344",
             "S55p_deep": "051370c37d5aef08880d55a89490b3de2109a66c800635dc97583edcf4e281af",
             "C14": "0f44e6141e5a0758a3d2278e6f79ea2d81356fb984c3464e0a029ae46c13c63f",
             "C14p": "dd0c3b1488eed58d02a1a54a7ac3a77d5d3521162c9f7db7da49fb0085c2e80a"},
    "d13_lock_sha256": "5767020407ed11f3ec3a2ef985f2c41e7d0ac93a489cb7929e16a8159b5a98f1",
    "d13_lock_rest_sha256": "7a92a1e9c8e89cd241b61bfd08a00ac19d750c2a3723d7295a572db025e15b06",    # srtlib.lock_rest_sha256 of that lock
    "d13_nets": {"S": "cbd2b84a47dd6bf71c62eb0cae4d660e2811db5efb9b55b695dd9337d141f12a",
                 "B": "5351b27b449145ae6c461260b6320925cebba56d2b0449f5a23884d9f787a659"},
}
WHO = {-1: "alone", 0: "pair", 1: "pair", 2: "pair", 3: "chain", 7: "pair", 8: "chain"}
FAM = {"uniform": "uni", "depspam_fm": "dfm"}
UNIT = {"mean": "us", "p999": "us", "per_shot": "us per shot", "h2h": "us per shot"}


# ------------------------------------------------------------------------------------------------------- the records
def jl(path):
    return [json.loads(x) for x in open(path) if x.strip()] if os.path.exists(path) else []


def load_round(path):
    """A drive.py output directory -> its attempts.json with each attempt's rows and shot trace attached."""
    A = json.load(open(os.path.join(path, "attempts.json")))
    dl, ss = os.path.join(path, "derived_lock.json"), os.path.join(path, "session.json")
    if os.path.exists(dl):                                  # an H100 R4 record: the lock its rows name, and the session's record
        A["derived_lock"] = dict(sha256=srtlib.sha256_file(dl), lock=json.load(open(dl)))
        A["session_sha256"] = srtlib.sha256_file(ss) if os.path.exists(ss) else None
    for s in A["slots"]:
        for at in s["attempts"]:
            d = os.path.join(path, at["dir"])
            at["rows_data"], at["trace_data"] = jl(os.path.join(d, "rows.jsonl")), jl(os.path.join(d, "trace.jsonl"))
    return A


def keys_of(meta):
    """[(cell key, stat, getter of the stat from a row)] that a slot's process feeds."""
    rnd = meta["round"]
    if meta["role"] == "net":
        return [(f"{rnd}.net.{meta['net']}.{meta['engine']}.h2h", "h2h", lambda r: r["amortized"]["h2h"])]
    if "family" in meta:                                    # BB pipeline
        who = WHO[meta["config"]]
        eng = "" if who == "alone" or not meta.get("engine") else f".{meta['engine']}"
        base = f"{rnd}.{FAM[meta['family']]}.{who}.p{meta['p']:g}.{meta['mode']}"
    else:                                                   # d = 13 pipeline
        eng, base = "", f"{rnd}.{meta['config']}.p{meta['p']:g}.{meta['mode']}"
    if meta["mode"] == "single":
        return [(f"{base}.mean{eng}", "mean", lambda r: r["latency_us"]["mean"]),
                (f"{base}.p999{eng}", "p999", lambda r: r["latency_us"]["p999"])]
    return [(f"{base}.per_shot{eng}", "per_shot", lambda r: r["per_shot_us"])]


def window_why(w):
    """Why the rule rejects a window record: a list of strings, empty when it holds. The rule is srtlib.Window's, from the
    record alone (what Window.end writes: the SM clock histogram, the power-cap count, the OR of the clock-event reasons, the
    gap, the other compute apps). The two clauses that need per-sample facts use the record's own count when it has one
    (low_uncapped, power_default_w) and otherwise a bound that never rejects a window that holds: the samples below the floor
    beyond the power-capped ones; no check of the power limit."""
    if "error" in w:
        return [f"window error: {w['error']}"]
    if not w.get("samples"):
        return ["no clock sample in the window"]
    why = []
    low = w.get("low_uncapped")
    if low is None:
        floor = srtlib.SM_FLOOR * w["max_mhz"]
        low = max(0, sum(c for mhz, c in w["hist"].items() if int(mhz) < floor) - w["power_capped"])
    if low:
        why.append(f"{low} samples below {srtlib.SM_FLOOR:g} of the maximum SM clock without the power cap")
    if int(w["reasons"], 16) & ~srtlib.ALLOWED:
        why.append(f"clock-event reasons {int(w['reasons'], 16) & ~srtlib.ALLOWED:#x}")
    if w.get("power_default_w") is not None and w["power_limit_w"] < w["power_default_w"]:
        why.append(f"power limit {w['power_limit_w']:g} W below the default {w['power_default_w']:g} W")
    if (w.get("gap_max_s") or 0) > srtlib.MAX_GAP_S:
        why.append(f"sampler gap {w['gap_max_s']:.2f} s")
    if w.get("other_pids"):
        why.append(f"other compute apps {w['other_pids']}")
    return why


def windows_ok(ws):
    """The analysis's own verdict on an attempt's window records."""
    return bool(ws) and not any(window_why(w) for w in ws)


def clock_stats(wins):
    """App. D's statistics over window records (they decide nothing): windows, those with a sample below 97 % of the maximum,
    the lowest ratio, those with the SW power cap active in a sample, the lowest enforced power limit."""
    r = [w["clock_min_ratio"] for w in wins if w.get("clock_min_ratio") is not None]
    return {"windows": len(wins), "below_97": sum(x < srtlib.APPX_D for x in r), "min_ratio": min(r) if r else None,
            "power_capped": sum(bool(w.get("power_capped")) for w in wins),
            "min_power_limit_w": min((w["power_limit_w"] for w in wins if w.get("power_limit_w") is not None), default=None)}


def process_row(meta, rows):
    """The one row a slot's process wrote, or None."""
    if meta["role"] == "net":
        rs = [r for r in rows if r.get("metric") == "time" and r.get("net") == meta["net"]]
    else:
        rs = [r for r in rows if r.get("role") == "pipeline"]
    return rs[0] if len(rs) == 1 else None


def derived_lock(A, card, freeze):
    """The lock a d = 13 record of the H100 card carries (derived_lock.json, written by srt_d13.py h100-lock from the study's
    lock and the session's record) -> (its sha256, which the rows may name in place of the freeze's lock, or None; problems).
    It counts only on the H100 card, when it names the freeze's lock as its origin, lists exactly the fields a derivation
    replaces (the card's and the venv's identity and the cache digest), and everything else in it equals the freeze's lock
    (srtlib.lock_rest_sha256), and when the session record beside it is the one it was derived from."""
    d = A.get("derived_lock")
    if d is None:
        return None, []
    if card != "H100":
        return None, ["a derived lock belongs to the H100 card only"]
    L, why = d["lock"], []
    der = L.get("h100_derivation") or {}
    if der.get("lock_sha256") != freeze["d13_lock_sha256"]:
        why.append("it was not derived from the freeze's d = 13 lock")
    if sorted(der.get("replaced") or []) != sorted(srtlib.D13_REPLACED):
        why.append(f"it replaced {der.get('replaced')}, not {list(srtlib.D13_REPLACED)}")
    if srtlib.lock_rest_sha256(L) != freeze["d13_lock_rest_sha256"] or der.get("rest_sha256") != freeze["d13_lock_rest_sha256"]:
        why.append("a field other than the replaced ones differs from the freeze's lock")
    if L.get("identity", {}).get("gpu") != "NVIDIA H100 80GB HBM3":
        why.append("its card is not the NVIDIA H100 80GB HBM3")
    if der.get("session_sha256") != A.get("session_sha256"):
        why.append("the session record beside it is not the one it was derived from")
    return (None if why else d["sha256"]), why


def row_problems(meta, r, rnd, freeze, derived=None):
    """Why a row is not its slot's (strings). `derived`: the sha256 of the valid derived d = 13 lock of the record, if any."""
    bad = []
    if meta["role"] == "net":
        want = {"net": meta["net"], "engine": meta["engine"], "batch": meta["batch"]}
    else:
        want = {"config": meta["config"], "p": meta["p"], "mode": meta["mode"]}
        if "family" in meta:
            want["family"] = meta["family"]
            if meta["config"] != -1 and meta.get("engine") and rnd != "r2":
                want["engine"] = meta["engine"]
        if r.get("shots") not in (None, meta["n"]):
            bad.append(f"shots {r.get('shots')} not {meta['n']}")
    for k, v in want.items():
        if r.get(k) != v:
            bad.append(f"{k} {r.get(k)!r} not {v!r}")
    if r.get("kind_seed", 12) != 12:
        bad.append(f"seed kind {r.get('kind_seed')}")
    if freeze:
        if meta["role"] == "net":
            if r.get("sha256") != freeze["nets"].get(meta["net"]):
                bad.append("network sha256 is not the freeze's")
        elif "family" in meta:
            lock = freeze["locks"].get(rnd)
            if r.get("speed_sha256") != freeze["speed_sha256"] or (lock and r.get("locked_sha256") != lock):
                bad.append("lock or speed-components sha256 is not the freeze's")
        else:
            nets = r.get("nets") or {}
            if r.get("lock_sha256") not in (freeze["d13_lock_sha256"], derived) or \
                    any(freeze["d13_nets"][k] != v for k, v in nets.items()):
                bad.append("d = 13 lock or network sha256 is not the freeze's")
    return bad


def analyze_card(card, records, freeze, verify):
    """One card's records {"r1": load_round(...), ...} (any subset) -> (its numbers, its check lines)."""
    cells, checks, status, clock, shots = {}, [], {}, {}, {}
    used_w, all_w = [], []
    for rnd in sorted(records):
        A = records[rnd]
        if not A["gate"]["on"]:
            checks.append(f"{card} {rnd}: the gate was off (no process of this record is accepted)")
        for where, rec in [("the record", A)] + [(f"slot {s['id']}", s) for s in A["slots"]] + \
                [(f"{s['id']} a{at['k']}", at) for s in A["slots"] for at in s["attempts"]]:
            if rec.get("card") and str(rec["card"]).upper() != card:      # drive.py writes the card into each level
                checks.append(f"{card} {rnd}: {where} says it was made on card {rec['card']}")
        derived = None
        if freeze:
            derived, why = derived_lock(A, card, freeze)
            checks += [f"{card} {rnd}: derived lock: {w}" for w in why]
        st = status[rnd] = {"slots": len(A["slots"]), "established": 0, "not established": 0, "failed": 0, "pending": 0}
        n_att = n_rej = 0
        r_used, r_all = [], []
        for s in A["slots"]:
            meta, used, extra = s["meta"], None, []
            for at in s["attempts"]:
                n_att += 1
                r_all += at["windows"]
                for w in at["windows"]:
                    if not A.get("dry") and (w.get("burn_s") or w.get("forced")):
                        checks.append(f"{card} {rnd} {s['id']} a{at['k']}: a registered record has a burn or a forced window")
                for t in at.get("trace_data", []):
                    d = shots.setdefault(t["set"], {}).setdefault(str(t["chunk"]), [])
                    if t["sha256"] not in d:
                        d.append(t["sha256"])
                ok = at["accepted"] and A["gate"]["on"]
                if ok and verify and not windows_ok(at["windows"]):
                    checks.append(f"{card} {rnd} {s['id']} a{at['k']}: marked accepted but its windows do not hold "
                                  f"({'; '.join(w for x in at['windows'] for w in window_why(x)) or 'no window'})")
                    ok = False
                if not ok and not at["hard"]:
                    n_rej += 1
                row = process_row(meta, at.get("rows_data", [])) if not at["hard"] else None
                if row is not None:
                    bad = row_problems(meta, row, rnd, freeze, derived)
                    if bad:
                        checks.append(f"{card} {rnd} {s['id']} a{at['k']}: row is not the slot's ({'; '.join(bad)})")
                        row = None
                if ok and row is not None and used is None:
                    used = (at, row)
                    r_used += at["windows"]
                elif row is not None:
                    extra.append((at, row))
            if used:
                st["established"] += 1
            elif not s["attempts"]:
                st["pending"] += 1
            else:
                st["failed" if s["attempts"][-1]["hard"] else "not established"] += 1
            for key, stat, get in keys_of(meta):
                c = cells.setdefault(key, {"unit": UNIT[stat], "slots": [], "reps": [], "withheld": [], "rejected": []})
                c["slots"].append(s["id"])
                if used:
                    at, row = used
                    c["reps"].append({"slot": s["id"], "rep": meta["rep"], "attempt": at["k"], "value": get(row),
                                      "host": at.get("host"),
                                      "clock_min_ratio": (at["windows"][0].get("clock_min_ratio") if at["windows"] else None)})
                else:
                    c["withheld"].append(s["id"])
                for at, row in extra:
                    c["rejected"].append({"slot": s["id"], "attempt": at["k"], "value": get(row),
                                          "why": "; ".join(at["soft"] or at["hard"]) or "not accepted"})
        clock[rnd] = {"attempts": n_att, "rejected": n_rej, "used": clock_stats(r_used), "all": clock_stats(r_all)}
        used_w += r_used
        all_w += r_all
    clock["total"] = {"used": clock_stats(used_w), "all": clock_stats(all_w)}
    for key, c in cells.items():
        v = [r["value"] for r in c["reps"]]
        c["reps"].sort(key=lambda r: (r["rep"], r["slot"]))
        c["n_planned"], c["n_established"] = len(c["slots"]), len(v)
        c["value"] = max(v) if v and not c["withheld"] else None
        c["fastest"] = min(v) if v and not c["withheld"] else None
        c["range"] = [min(v), max(v)] if v else None
        c["why"] = None if c["value"] is not None else (
            f"withheld: slot(s) {', '.join(c['withheld'])} not established" if c["withheld"] else "no records")
        del c["slots"]
    for name, by in shots.items():
        for ch, d in by.items():
            if len(d) > 1:
                checks.append(f"{card} shots: chunk {ch} of {name} has {len(d)} different digests across processes")
    return {"cells": cells, "status": status, "clock": clock,
            "shots": {n: {c: d[0] for c, d in sorted(by.items(), key=lambda x: int(x[0]))} for n, by in sorted(shots.items())}}, checks


def analyze(records, freeze=FREEZE, verify=True):
    """records: {"RTX": {"r1": load_round(...), ...}, "H100": {...}} (any subset of cards and rounds) -> numbers dict.
    freeze=None skips the hash checks, verify=False takes drive.py's verdicts as they are (the self-test's synthetic
    attempts)."""
    cards, checks = {}, []
    for card in CARDS:
        cards[card], ch = analyze_card(card, records.get(card, {}), freeze, verify)
        checks += ch
    a, b = (cards[c]["shots"] for c in CARDS)
    for name in sorted(set(a) & set(b)):
        for ch in sorted(set(a[name]) & set(b[name]), key=int):
            if a[name][ch] != b[name][ch]:
                checks.append(f"shots: chunk {ch} of {name} has another digest on {CARDS[1]} than on {CARDS[0]}")
    return {"cards": cards, "checks": sorted(checks), "freeze_checked": freeze is not None, "windows_verified": verify}


LAYOUT = {"R1": "r1", "R2": "r2", "R3": "r3", "R4": "r4"}


def from_records(path, dry):
    """RECORDS_DIR (the layout of the module text), dry flag -> the numbers dict (JSON-serialisable, deterministic)."""
    recs = {c: {r: load_round(os.path.join(path, c, d)) for d, r in LAYOUT.items() if os.path.isdir(os.path.join(path, c, d))}
            for c in CARDS}
    numbers = analyze(recs, freeze=None if dry else FREEZE)
    for c, rr in sorted(recs.items()):
        for r, A in sorted(rr.items()):
            if bool(A.get("dry")) != bool(dry):
                numbers["checks"].append(f"{c} {r}: the record's dry flag is {bool(A.get('dry'))}, the analysis was asked for {bool(dry)}")
    numbers["checks"].sort()
    return numbers


# ------------------------------------------------------------------------------------------------------------ table
def _rows(numbers, card):
    """[(row dict, tolerance of its old value or None)] of one card in inventory order. Every row names its card; `old` is the
    paper's printed value, measured on the card named by `old_card` (the RTX PRO 6000, except C5's), so only the RTX card's
    rows and C5 are self-tested."""
    cells = numbers["cards"][card]["cells"]
    hw = HW[card]
    out = []

    def V(key, scale=1.0, how="value", cs=None):
        def f():
            c = (cells if cs is None else cs).get(key)
            if c is None:
                return None, f"no records of {key}"
            if c[how] is None:
                return None, c["why"]
            return c[how] * scale, None
        return f

    def op(fn, *xs):
        def f():
            vs = []
            for x in xs:
                v, w = x()
                if v is None:
                    return None, w
                vs.append(v)
            return fn(*vs), None
        return f

    def agg(fn, xs):
        return op(lambda *v: fn(v), *xs)

    def half(s):
        """Half a unit of the last printed digit of the string s (the tolerance of a printed number)."""
        return 0.5 * 10.0 ** -(len(s.split(".")[1]) if "." in s else 0)

    def num(row, item, unit, old, expr, basis, extra=None, tol=None):
        """old: the printed string (its last digit sets the self-test's tolerance, unless tol is given: "about 40") or a
        float (no tolerance)."""
        v, why = expr()
        d = dict(row=row, item=item, card=card, old_card="H100" if row == "C5" else "RTX", kind="number", unit=unit,
                 old=float(old), new=v, why=why, basis=basis)
        d.update(extra or {})
        out.append((d, tol if tol is not None else half(old) if isinstance(old, str) else None))

    def cmp(row, item, unit, ours, theirs, ours_old, theirs_old, basis, ratio_old=None, extra=None, decides=True):
        """ours_old, theirs_old, ratio_old: printed strings or None; the tolerance of the printed ratio (or of the ratio of
        the two printed values) goes to the self-test."""
        o, w1 = ours()
        t, w2 = theirs()
        ok = o is not None and t is not None
        if ratio_old is not None:
            ro, tol = float(ratio_old), half(ratio_old)
        elif ours_old is not None and theirs_old is not None:
            ro = float(theirs_old) / float(ours_old)
            tol = ro * (half(ours_old) / float(ours_old) + half(theirs_old) / float(theirs_old))
        else:
            ro, tol = None, None
        d = dict(row=row, item=item, card=card, old_card="H100" if row == "C5" else "RTX", kind="comparison", decides=decides,
                 unit=unit, ours_old=None if ours_old is None else float(ours_old),
                 theirs_old=None if theirs_old is None else float(theirs_old), ratio_old=ro, ours=o, theirs=t,
                 ratio=(t / o if ok else None), holds=((o < t) if ok and decides else None), why=w1 or w2, basis=basis)
        d.update(extra or {})
        out.append((d, tol))

    B1 = (f"end to end, host to host; BB72 D1 pairs (six-block S55'-deep -> C14', fitted noise) and Cascade alone; {hw}; "
          "slowest accepted process of 3, same job")
    B2 = (f"end to end, host to host; BB72 phase C pairs and chains (three-block S55' -> C14' [-> Relay-BP]) and Cascade alone; "
          f"{hw}; slowest accepted process of 3, same job")
    B3 = (f"network alone at batch 4,096, host to host, per shot; M2 and BB engines, Cascade networks compiled at 4,096; {hw}; "
          "slowest accepted process of 3")
    B4 = (f"end to end, host to host, batched 500,000 shots; phase C pair on the BB and multi-shot engines, Cascade alone; {hw}; "
          "slowest accepted process of 3")
    B5 = (f"App. D's H100 paragraph restated from this card's R3 networks and pipelines and R2 chain; {hw}; slowest accepted "
          "process of 3; descriptive, decides nothing")
    B8 = (f"end to end, host to host; d = 13 network d13_b5_r775 routing to CS13 (C5), Cascade alone (C2), chain (C1); {hw}; "
          "slowest accepted process of 3, same job")
    PS = (0.002, 0.003)
    P13 = (0.006, 0.0035)
    pc = lambda p: f"{100 * p:g} %"
    H3 = {"headline": "H3-cascade"}
    sens = "sensitivity, decides nothing: fastest accepted Cascade-alone process; "

    def alone(r, p, mode, stat, how="value", sc=1.0, fam="dfm"):
        return V(f"{r}.{fam}.alone.p{p:g}.{mode}.{stat}", sc, how)

    def pair(r, p, mode, stat, eng=None, sc=1.0, fam="dfm", who="pair"):
        return V(f"{r}.{fam}.{who}.p{p:g}.{mode}.{stat}" + (f".{eng}" if eng else ""), sc)

    # ---- C1 and C2: D1-S (R1)
    old_alone = {"single": ("1759.8", "1763.8"), "batched": ("12.00", "12.01")}
    old_base = {"single": ("163.2", "300.5"), "batched": ("1.355", "2.332")}
    old_new = {"single": ("138.8", "276.6"), "batched": ("0.734", "1.710")}
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            num("C1", f"BB72 p={pc(p)}: Cascade alone, {mode} mean" if mode == "single" else
                f"BB72 p={pc(p)}: Cascade alone, batched", unit, old_alone[mode][i], alone("r1", p, mode, stat, "value", sc), B1)
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            num("C1", f"BB72 p={pc(p)}: pair on the BB engine, {mode}", unit, old_base[mode][i],
                pair("r1", p, mode, stat, "base", sc), B1)
    cells_all = [(e, p) for e in ("base", "new") for p in PS]

    def speedup(mode, stat, how="value"):
        return [op(lambda a, b: a / b, alone("r1", p, mode, stat, how), pair("r1", p, mode, stat, e)) for e, p in cells_all]

    def excess():
        return [op(lambda a, b: 100 * (b / a - 1), alone("r1", p, "single", "p999"), pair("r1", p, "single", "p999", e))
                for e, p in cells_all]
    for name, lst, lo, hi, u in (("single-shot mean speed-up over Cascade alone", speedup("single", "mean"), "5.9", "12.7", "x"),
                                 ("batched speed-up over Cascade alone", speedup("batched", "per_shot"), "5.1", "16.4", "x"),
                                 ("pair's 99.9th percentile above Cascade alone's", excess(), "3.5", "7.7", "%")):
        for lab, fn, o in (("lowest", min, lo), ("highest", max, hi)):
            num("C1", f"BB72, two engines x two p: {name}, {lab}", u, o, agg(fn, lst), B1)
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            cmp("C1", f"BB72 p={pc(p)}: pair on the BB engine vs Cascade alone, {mode}", unit, pair("r1", p, mode, stat, "base", sc),
                alone("r1", p, mode, stat, "value", sc), old_base[mode][i], old_alone[mode][i], B1, extra=H3)
    # ---- C2
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            num("C2", f"BB72 p={pc(p)}: pair on the multi-shot engine, {mode}", unit, old_new[mode][i],
                pair("r1", p, mode, stat, "new", sc), B1)
    for i, p in enumerate(PS):
        num("C2", f"BB72 p={pc(p)}: batched speed-up of the multi-shot-engine pair over Cascade alone", "x", ("16.4", "7.0")[i],
            op(lambda a, b: a / b, alone("r1", p, "batched", "per_shot"), pair("r1", p, "batched", "per_shot", "new")), B1)
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            cmp("C2", f"BB72 p={pc(p)}: pair on the multi-shot engine vs Cascade alone, {mode}", unit,
                pair("r1", p, mode, stat, "new", sc), alone("r1", p, mode, stat, "value", sc), old_new[mode][i],
                old_alone[mode][i], B1, extra=H3)
    # ---- C3: M2 networks (R3)
    n = lambda net, eng, sc=1.0: V(f"r3.net.{net}.{eng}.h2h", sc)
    num("C3", "three-block network S55, multi-shot engine, batch 4,096, host to host, per shot", "us per shot", "1.883",
        n("S55", "new"), B3)
    num("C3", "three-block network S55, multi-shot engine, per round", "us per round", "0.314", n("S55", "new", 1 / R_BB), B3,
        extra={"headline": "H4-round"})
    num("C3", "BB engine over multi-shot engine, same batches (S55)", "x", "2.20",
        op(lambda a, b: a / b, n("S55", "base"), n("S55", "new")), B3)
    num("C3", "six-block network S55'-deep, multi-shot engine, per round", "us per round", "0.448", n("S55p_deep", "new", 1 / R_BB), B3)
    num("C3", "Cascade alone C14 at batch 4,096, per round", "us per round", "11.96", n("C14", "cascade", 1 / R_BB), B3)
    num("C3", "Cascade alone C14' at batch 4,096, per round", "us per round", "11.96", n("C14p", "cascade", 1 / R_BB), B3)
    cmp("C3", "three-block network, multi-shot engine vs BB engine, per shot at batch 4,096", "us per shot",
        n("S55", "new"), n("S55", "base"), "1.883", "4.143", B3, ratio_old="2.20")
    cmp("C3", "three-block network (multi-shot engine) vs Cascade alone C14, per round", "us per round", n("S55", "new", 1 / R_BB),
        n("C14", "cascade", 1 / R_BB), "0.314", "11.96", B3)
    cmp("C3", "six-block network (multi-shot engine) vs Cascade alone C14', per round", "us per round",
        n("S55p_deep", "new", 1 / R_BB), n("C14p", "cascade", 1 / R_BB), "0.448", "11.96", B3)
    # ---- C4: phase C pairs on M2 (R3), fitted setting
    cfg4 = {0.002: 2, 0.003: 7}
    p3 = lambda p, eng: V(f"r3.dfm.pair.p{p:g}.batched.per_shot.{eng}", 1 / R_BB)
    a3 = lambda p, how="value": V(f"r3.dfm.alone.p{p:g}.batched.per_shot", 1 / R_BB, how)
    for i, p in enumerate(PS):
        num("C4", f"BB72 p={pc(p)}: phase C pair on the multi-shot engine, batched", "us per round", ("0.510", "0.999")[i],
            p3(p, "new"), B4)
    for i, p in enumerate(PS):
        cmp("C4", f"BB72 p={pc(p)}: pair on the multi-shot engine vs on the BB engine, batched", "us per round", p3(p, "new"),
            p3(p, "base"), None, None, B4, ratio_old=("1.74", "1.38")[i])
        cmp("C4", f"BB72 p={pc(p)}: pair on the multi-shot engine vs Cascade alone, batched", "us per round", p3(p, "new"),
            a3(p), None, None, B4, ratio_old=("23.5", "12.0")[i])
    rr = op(lambda a, b: a / b, n("S55p_deep", "new"), n("C14p", "cascade"))
    num("C4", "six-block network over Cascade alone, per shot at batch 4,096 (C14')", "x", "0.04", rr, B3)
    for q, o in ((0.02, "17"), (0.10, "7")):
        num("C4", f"Eq. (3) estimate of the batched speed-up at {100 * q:g} % routed (descriptive)", "x", o,
            op(lambda r_, q=q: 1 / (round(r_, 2) + q), rr), B3)
    meas = {("0.002", "base"): (166.5, 8.2), ("0.002", "new"): (136.0, 4.6), ("0.003", "base"): (309.0, 13.9),
            ("0.003", "new"): (279.0, 10.4)}
    s_single, s_batch, q1 = {"base": 130.9, "new": 100.5}, {"base": 6.26, "new": 2.69}, {"0.002": 0.02, "0.003": 0.10}
    formula = {k: (s_single[k[1]] + q1[k[0]] * 1782, s_batch[k[1]] + q1[k[0]] * 71.6 + 0.5) for k in meas}
    cell_ms = [(k, j, V(f"r1.dfm.pair.p{float(k[0]):g}.single.mean.{k[1]}") if j == 0 else
                V(f"r1.dfm.pair.p{float(k[0]):g}.batched.per_shot.{k[1]}")) for k in meas for j in (0, 1)]
    for name, pred, o in (("printed predictions", meas, "4.3"), ("unrounded formula", formula, "4.8")):
        num("C4", f"D1 plan 4.3: largest deviation of D's measured times from the {name} (descriptive)", "%", o,
            agg(max, [op(lambda m, k=k, j=j: 100 * abs(m / pred[k][j] - 1), e) for k, j, e in cell_ms]), B1)
    # ---- C5: App. D's H100 paragraph, on the H100 card only, restated from its R3 and R2 values; descriptive
    if card == "H100":
        a5 = lambda p: V(f"r3.dfm.alone.p{p:g}.batched.per_shot")
        p5 = lambda p: V(f"r3.dfm.pair.p{p:g}.batched.per_shot.new")
        num("C5", "H100: Cascade alone, batched, per shot (R3 pipelines, slowest of both points)", "us per shot", "40",
            agg(max, [a5(p) for p in PS]), B5, tol=1.5)
        num("C5", "H100: Cascade networks C14 and C14' at batch 4,096, per shot (slowest)", "us per shot", "40",
            agg(max, [V("r3.net.C14.cascade.h2h"), V("r3.net.C14p.cascade.h2h")]), B5, tol=1.5)
        num("C5", "H100: three-block network S55, multi-shot engine, per shot", "us per shot", "2.82", V("r3.net.S55.new.h2h"), B5)
        num("C5", "context from the RTX card: three-block network S55, multi-shot engine, per shot (R3)", "us per shot", "1.88",
            V("r3.net.S55.new.h2h", cs=numbers["cards"]["RTX"]["cells"]), B3,
            extra={"old_card": "RTX", "new_card": "RTX"})
        sp5 = [op(lambda a, b: a / b, a5(p), p5(p)) for p in PS]
        for lab, fn, o in (("lowest", min, "7.6"), ("highest", max, 11.4)):       # 11.4 as a float: not self-tested
            num("C5", f"H100: batched speed-up of the phase C pair (multi-shot engine) over Cascade alone, {lab} of both points",
                "x", o, agg(fn, sp5), B5, extra={"note": "the paper's range 7.6-11.4 spans the uniform setting too, which R3 "
                                                         "does not time; the highest is not self-tested"})
        for p in PS:
            cmp("C5", f"H100 p={pc(p)}: pair (multi-shot engine) vs Cascade alone, batched", "us per shot", p5(p), a5(p), None,
                None, B5, decides=False)
        cmp("C5", "H100 p=0.3 %: Cascade alone vs the phase C chain, batched (R2)", "us per shot",
            alone("r2", 0.003, "batched", "per_shot"), pair("r2", 0.003, "batched", "per_shot", None, 1.0, "dfm", "chain"),
            None, None, B5, decides=False)
    # ---- C6 and C7: phase C (R2)
    o6 = {"alone": {"single": ("1696.5", "1700.8"), "batched": ("11.98", "11.98")},
          "pair": {"single": ("105.5", "178.2"), "batched": ("0.89", "1.38")}}
    cfg6 = {0.002: (2, 3), 0.003: (7, 8)}
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            num("C6", f"BB72 p={pc(p)}: Cascade alone, {mode}", unit, o6["alone"][mode][i], alone("r2", p, mode, stat, "value", sc), B2)
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            num("C6", f"BB72 p={pc(p)}: pair (S -> B), {mode}", unit, o6["pair"][mode][i], pair("r2", p, mode, stat, None, sc), B2)
    for mode, stat, lo, hi in (("single", "mean", "9.5", "16.1"), ("batched", "per_shot", "8.7", "13.5")):
        lst = [op(lambda a, b: a / b, alone("r2", p, mode, stat), pair("r2", p, mode, stat)) for p in PS]
        for lab, fn, o in (("lowest", min, lo), ("highest", max, hi)):
            num("C6", f"BB72: {mode} speed-up of the pair over Cascade alone, {lab}", "x", o, agg(fn, lst), B2)
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            cmp("C6", f"BB72 p={pc(p)}: pair vs Cascade alone, {mode}", unit, pair("r2", p, mode, stat, None, sc),
                alone("r2", p, mode, stat, "value", sc), o6["pair"][mode][i], o6["alone"][mode][i], B2)
    o7 = {"single": ("254.3", "882.4"), "batched": ("1.70", "10.66")}
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            num("C7", f"BB72 p={pc(p)}: Relay-BP chain, {mode}", unit, o7[mode][i], pair("r2", p, mode, stat, None, sc, who="chain"), B2)
    num("C7", "BB72 p=0.3 %: chain's 99.9th percentile, single", "ms", "129.6", pair("r2", 0.003, "single", "p999", None, 1e-3, who="chain"), B2)
    num("C7", "BB72 p=0.3 %: Cascade alone's 99.9th percentile, single", "ms", "1.72", alone("r2", 0.003, "single", "p999", "value", 1e-3), B2)
    cmp("C7", "BB72 p=0.3 %: 99.9th percentile single, Cascade alone vs the chain", "ms", alone("r2", 0.003, "single", "p999", "value", 1e-3),
        pair("r2", 0.003, "single", "p999", None, 1e-3, who="chain"), "1.72", "129.6", B2)
    # ---- C8, C9: d = 13 (R4)
    d = lambda c, p, mode, stat, sc=1.0, how="value": V(f"r4.{c}.p{p:g}.{mode}.{stat}", sc, how)
    o8 = {"alone": {"single": ("397.0", "397.1"), "batched": ("10.48", "10.48")},
          "pair": {"single": ("107.4", "69.4"), "batched": ("2.34", "1.37")}}
    for i, p in enumerate(P13):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_D13, "us per round")):
            num("C8", f"d=13 p={pc(p)}: Cascade alone, {mode}", unit, o8["alone"][mode][i], d("C2", p, mode, stat, sc), B8)
    for i, p in enumerate(P13):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_D13, "us per round")):
            num("C8", f"d=13 p={pc(p)}: pair, {mode}", unit, o8["pair"][mode][i], d("C5", p, mode, stat, sc), B8)
    for i, p in enumerate(P13):
        for mode, stat, o in (("single", "mean", ("3.70", "5.72")), ("batched", "per_shot", ("4.49", "7.67"))):
            num("C8", f"d=13 p={pc(p)}: {mode} speed-up of the pair over Cascade alone", "x", o[i],
                op(lambda a, b: a / b, d("C2", p, mode, stat), d("C5", p, mode, stat)), B8)
    for i, p in enumerate(P13):
        num("C8", f"d=13 p={pc(p)}: Cascade alone, batched, per shot (the repeat's value)", "us per shot", ("136.21", "136.20")[i],
            d("C2", p, "batched", "per_shot"), B8)
        num("C8", f"d=13 p={pc(p)}: pair's 99.9th percentile, single", "us", "490", d("C5", p, "single", "p999"), B8)
        num("C8", f"d=13 p={pc(p)}: Cascade alone's 99.9th percentile, single", "us", ("411", "417")[i], d("C2", p, "single", "p999"), B8)
    for i, p in enumerate(P13):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_D13, "us per round")):
            cmp("C8", f"d=13 p={pc(p)}: pair vs Cascade alone, {mode}", unit, d("C5", p, mode, stat, sc),
                d("C2", p, mode, stat, sc), o8["pair"][mode][i], o8["alone"][mode][i], B8, extra=H3)
        cmp("C8", f"d=13 p={pc(p)}: 99.9th percentile single, Cascade alone vs pair", "us", d("C2", p, "single", "p999"),
            d("C5", p, "single", "p999"), ("411", "417")[i], "490", B8)
    o9 = {"single": ("229.0", "312.7"), "batched": ("5.60", "7.75")}
    for i, p in enumerate(P13):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_D13, "us per round")):
            num("C9", f"d=13 p={pc(p)}: chain (network -> CS13 -> correlated PM), {mode}", unit, o9[mode][i], d("C1", p, mode, stat, sc), B8)
    # ---- C10: tails
    for name, lst, lo, hi in (("pair's 99.9th percentile above Cascade alone's (BB72, D1-S)", excess(), "3.5", "7.7"),):
        for lab, fn, o in (("lowest", min, lo), ("highest", max, hi)):
            num("C10", f"{name}, {lab}", "%", o, agg(fn, lst), B1)
    for e, p in cells_all:
        cmp("C10", f"BB72 p={pc(p)}, {e} engine: 99.9th percentile single, Cascade alone vs pair", "us",
            alone("r1", p, "single", "p999"), pair("r1", p, "single", "p999", e), None, None, B1)
    o10 = {0.006: {"C1": "870", "C5": "490"}, 0.0035: {"C1": "695", "C5": "490"}}
    for i, p in enumerate(P13):
        for c in ("C1", "C5"):
            cmp("C10", f"d=13 p={pc(p)}: 99.9th percentile single, Cascade alone vs {c}", "us", d("C2", p, "single", "p999"),
                d(c, p, "single", "p999"), ("411", "417")[i], o10[p][c], B8)
    # ---- the fastest-Cascade sensitivity rows (decide nothing): Cascade alone's fastest accepted process on the same cells
    for i, p in enumerate(PS):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_BB, "us per round")):
            num("C1", f"{sens}BB72 p={pc(p)}: Cascade alone, {mode}", unit, old_alone[mode][i],
                alone("r1", p, mode, stat, "fastest", sc), B1)
            num("C6", f"{sens}BB72 p={pc(p)} (phase C): Cascade alone, {mode}", unit, o6["alone"][mode][i],
                alone("r2", p, mode, stat, "fastest", sc), B2)
    for i, p in enumerate(P13):
        for mode, stat, sc, unit in (("single", "mean", 1, "us"), ("batched", "per_shot", 1 / R_D13, "us per round")):
            num("C8", f"{sens}d=13 p={pc(p)}: Cascade alone, {mode}", unit, o8["alone"][mode][i],
                d("C2", p, mode, stat, sc, "fastest"), B8)
    for net in ("C14", "C14p"):
        num("C3", f"{sens}Cascade alone {net} at batch 4,096, per round", "us per round", "11.96",
            V(f"r3.net.{net}.cascade.h2h", 1 / R_BB, "fastest"), B3)
    div = lambda a, b: a / b
    for i, p in enumerate(PS):                                  # speed-ups with the fastest Cascade (old: ratio of printed values)
        for m, s_ in (("single", "mean"), ("batched", "per_shot")):
            for e, oldp, row in (("base", old_base, "C1"), ("new", old_new, "C2")):
                num(row, f"{sens}BB72 p={pc(p)}: {m} speed-up of the {'BB' if e == 'base' else 'multi-shot'}-engine pair over Cascade alone",
                    "x", float(old_alone[m][i]) / float(oldp[m][i]),
                    op(div, alone("r1", p, m, s_, "fastest"), pair("r1", p, m, s_, e)), B1)
        num("C4", f"{sens}BB72 p={pc(p)}: multi-shot-engine pair vs Cascade alone, batched, speed-up", "x", float(("23.5", "12.0")[i]),
            op(div, a3(p, "fastest"), p3(p, "new")), B4)
    for i, p in enumerate(P13):
        for m, s_ in (("single", "mean"), ("batched", "per_shot")):
            num("C8", f"{sens}d=13 p={pc(p)}: {m} speed-up of the pair over Cascade alone", "x",
                float(o8["alone"][m][i]) / float(o8["pair"][m][i]), op(div, d("C2", p, m, s_, 1.0, "fastest"), d("C5", p, m, s_)), B8)
    order = {r: k for k, r in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"])}
    out.sort(key=lambda x: order[x[0]["row"]])                  # stable: the order of definition within a row
    return out


def table(numbers):
    """numbers (analyze's result) -> the list of row dicts, card by card (RTX, then H100), each in inventory order (C1 ... C10;
    C5 on the H100 card only). The comparisons that H3's Cascade range is taken over carry "headline": "H3-cascade" (D1-S's
    pair vs Cascade alone, both engines, and e7's C5 vs C2: single mean and batched, both points), C3's per-round number
    "headline": "H4-round"."""
    return [d for c in CARDS for d, _ in _rows(numbers, c)]


ROWS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"]


def verdicts(rows):
    """The verdict of every claim on every card, from table()'s rows: [{card, claim, verdict, comparisons, holds, fails,
    withheld, ...}]. Only deciding comparisons count (the sensitivity rows are numbers, C5's comparisons do not decide):
    "holds" = all of them hold, "fails" = one does not, "withheld" = one has no value, "numbers only" = the claim has no
    deciding comparison (C5, C9). H3-cascade adds the lowest and highest ratio of its comparisons, H4-round the value."""
    out = []
    for card in CARDS:
        rs = [r for r in rows if r["card"] == card]
        for claim in ROWS + ["H3-cascade", "H4-round"]:
            if not any(r["row"] == claim or r.get("headline") == claim for r in rs):
                continue                                            # C5 is not on the RTX card
            if claim == "H4-round":
                for r in rs:
                    if r.get("headline") == claim:
                        out.append(dict(card=card, claim=claim, verdict="value", value=r["new"], old=r["old"], why=r["why"]))
                continue
            sel = [r for r in rs if r["kind"] == "comparison" and r["decides"] and
                   (r["row"] == claim or r.get("headline") == claim)]
            holds, fails = sum(r["holds"] is True for r in sel), sum(r["holds"] is False for r in sel)
            v = dict(card=card, claim=claim, comparisons=len(sel), holds=holds, fails=fails, withheld=len(sel) - holds - fails)
            v["verdict"] = ("numbers only" if not sel else "fails" if fails else "withheld" if v["withheld"] else "holds")
            if claim == "H3-cascade" and sel:
                ratios = [r["ratio"] for r in sel]
                v["ratio_lowest"], v["ratio_highest"] = (None, None) if None in ratios else (min(ratios), max(ratios))
            out.append(v)
    return out


# ---------------------------------------------------------------------------------------------------------- self-test
def public_records(root):
    """The registered records in the public tree (pre-freeze, ungated) as records for analyze(): every process one slot,
    taken as accepted (the check is of the arithmetic, not of the gate). The RTX card gets R1-R4; the H100 card gets the
    old pod session's networks and phase C pipelines as its R3 and its chain and Cascade alone as its R2."""
    CC = os.path.join(root, "cascade", "bb72", "cascade_compare")
    D13 = os.path.join(root, "cascade", "d13", "out")
    slots = {c: {} for c in CARDS}

    def add(card, rnd, rows, meta_of):
        sl = slots[card].setdefault(rnd, [])
        for r in rows:
            m = meta_of(r)
            if m is not None:
                sl.append({"id": f"{rnd}.{len(sl)}", "meta": dict(m, round=rnd, rep=r.get("rep", 1)), "attempts": [{
                    "k": 0, "accepted": True, "hard": [], "soft": [], "rows_data": [r], "trace_data": [], "host": r.get("host"),
                    "windows": [{"clock_min_ratio": 1.0, "other_pids": [], "gap_max_s": 0.0}]}]})

    def pipes(configs, fams=("depspam_fm",), batched_tail0=False, base_only=False, engine=True):
        def f(r):
            if r.get("role") != "pipeline" or r["family"] not in fams or r["config"] not in configs:
                return None
            if batched_tail0 and (r.get("tail_b", 0) != 0 or r["mode"] != "batched"):
                return None
            if base_only and r.get("engine") != "base":
                return None
            return dict(role="pipeline", family=r["family"], config=r["config"], p=r["p"], mode=r["mode"],
                        engine=r.get("engine") if engine else None, n=r["shots"])
        return f

    def nets(names):
        def f(r):
            if r.get("role") != "net" or r.get("metric") != "time" or r["net"] not in names or r.get("batch") != 4096:
                return None
            return dict(role="net", net=r["net"], engine=r["engine"], batch=4096)
        return f
    add("RTX", "r1", jl(os.path.join(CC, "phase_d/out/speed/rows.jsonl")), pipes((-1, 0, 1)))
    add("RTX", "r2", jl(os.path.join(CC, "phase_c/out/speed/rows.jsonl")), pipes((-1, 2, 3, 7, 8)))
    m2 = os.path.join(CC, "batchopt/out_m2/rows")
    add("RTX", "r3", jl(os.path.join(m2, "m2_CJ57/nets.jsonl")), nets(("S55", "S55p_deep", "C14", "C14p")))
    add("RTX", "r3", jl(os.path.join(m2, "m2_CJ59/pipe.jsonl")), pipes((-1, 2, 7), batched_tail0=True))
    h = os.path.join(CC, "batchopt/out_h100/20261011_204455")
    add("H100", "r3", jl(os.path.join(h, "nets.jsonl")), nets(("S55", "C14", "C14p")))
    add("H100", "r3", jl(os.path.join(h, "pipe.jsonl")), pipes((-1, 2, 7), batched_tail0=True))
    add("H100", "r2", jl(os.path.join(h, "pipe.jsonl")), pipes((-1, 8), batched_tail0=True, base_only=True, engine=False))
    add("RTX", "r4", jl(os.path.join(D13, "pipes.jsonl")) + jl(os.path.join(D13, "pipes_e7r.jsonl")),
        lambda r: dict(role="pipeline", config=r["config"], p=r["p"], mode=r["mode"], n=r["shots"]))
    return {c: {r: {"gate": {"on": True}, "slots": v} for r, v in rr.items()} for c, rr in slots.items()}


def selftest(root):
    """analyze + table on the public records: each number row's new value against its old one within the printed rounding,
    each comparison's ratio against the printed ratio (or the ratio of the printed values) within theirs; the sensitivity rows
    are not checked (they differ by design), nor are the rows whose old value was measured on the other card than the one
    computed. -> (rows checked, [(card, row, item, old, new) that differ], numbers)"""
    numbers = analyze(public_records(root), freeze=None, verify=False)
    n, bad = 0, []
    for card in CARDS:
        for d, tol in _rows(numbers, card):
            if tol is None or d["item"].startswith("sensitivity") or d.get("new_card", card) != d["old_card"]:
                continue
            new, old = (d["new"], d["old"]) if d["kind"] == "number" else (d["ratio"], d["ratio_old"])
            if new is None:
                bad.append((card, d["row"], d["item"], old, None))
                continue
            n += 1
            if abs(new - old) > tol * (1 + 1e-9):
                bad.append((card, d["row"], d["item"], old, new))
    return n, bad, numbers


def write_json(path, obj):
    with open(path, "x") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", metavar="RECORDS_DIR")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--out")
    ap.add_argument("--table")
    ap.add_argument("--verdicts")
    ap.add_argument("--selftest", metavar="PUBLIC_ROOT")
    a = ap.parse_args()
    if a.selftest:
        n, bad, numbers = selftest(a.selftest)
        print(f"{n} rows checked against the paper's printed values; {len(bad)} differ")
        for b in bad:
            print("  differs:", b)
        print("checks:", numbers["checks"])
        return
    numbers = from_records(a.records, a.dry)
    if a.out:
        write_json(a.out, numbers)
    rows = table(numbers)
    if a.table:
        write_json(a.table, rows)
    vs = verdicts(rows)
    if a.verdicts:
        write_json(a.verdicts, vs)
    print(json.dumps({"checks": numbers["checks"], "status": {c: v["status"] for c, v in numbers["cards"].items()},
                      "verdicts": {c: {v["claim"]: v["verdict"] for v in vs if v["card"] == c} for c in CARDS}}, sort_keys=True))


if __name__ == "__main__":
    main()
