"""analysis_ctrt.py: CASCADE-TRT-REG's numbers and tables from the records of its R1, R2 and R4 jobs on two cards.

    python analysis_ctrt.py --records RECORDS --out OUTDIR      OUTDIR must not exist; it gets numbers.json and tables.md

RECORDS/<CARD>/<ROUND>/   CARD = RTX or H100, ROUND = R1 (D1 harness, deep pairs), R2 (phase C pairs) or R4 (d = 13); each one
                          ctrt_drive.py output directory (drive.py's layout; attempts.json format "ctrt-cas-attempts-1"); a
                          missing one gives "no records". Slot ids end in .cas-compiled or .cas-trt, the slot meta has
                          "cascade_engine".
RECORDS/<CARD>/check/check_bb.json, check_d13.json    trt_check.py --compare outputs, reported as they are
RECORDS/<CARD>/build/build.jsonl                      trt_build.py rows

Rules (RULES.md), computed per card:
* Acceptance as in analysis_cas.py, which is reused (its analyze_card runs once per round and Cascade engine): an attempt counts
  iff it is marked accepted, the gate is on and windows_ok() holds for its window records. A slot's value is the stat of its
  accepted attempt's row (single: latency_us.mean in us, p999 kept as info in the cells; batched: per_shot_us). A cell (round,
  who, p, mode, S engine, Cascade engine) needs the accepted values of reps 1, 2 and 3; otherwise it is not established and
  every comparison that uses it is withheld, with the reason.
* Cascade alone = the fastest of its 3 reps; a pair = the slowest of its 3.
* A comparison is (R1, p, mode, S engine base|new), (R2, p, mode) or (R4, p, mode). Its Cascade alone is the alone cell of the
  same round, p and mode (R1: config -1, shared by both S engines). The chosen engine is the Cascade engine with the lower
  alone value (tie: compiled); a comparison whose two alone cells are not both established has none and is withheld.
  Speed-up = alone / pair on the chosen engine. The other engine's pair and speed-up are reported and decide nothing.
* Units: us; batched values per round (per-shot time / 6 for BB72, / 13 for d = 13) as stage 1 reports them, single values in us
  per shot. The cells of numbers.json stay per shot: "value" = slowest rep, "fastest" = fastest rep (None unless complete).
* Stage 1's values beside each comparison: the published slowest-of-3 cells of ../../speed_rt/cascade/results/numbers.json for
  the same card, round, who, p, mode and S engine (alone, pair, alone / pair); "n/a" when no such cell has a value.
Deterministic: no clock, host, random value or absolute path in the outputs; keys sorted; two runs give identical bytes."""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STAGE1 = os.path.normpath(os.path.join(HERE, "..", "..", "speed_rt", "cascade"))
sys.path.insert(0, STAGE1)
import analysis_cas as ac  # noqa: E402  (it imports srtlib)

CARDS, ROUNDS, ENGINES = ac.CARDS, ("R1", "R2", "R4"), ("compiled", "trt")
PS = {"R1": (0.002, 0.003), "R2": (0.002, 0.003), "R4": (0.006, 0.0035)}
ROUNDS_PER_SHOT = {"R1": 6, "R2": 6, "R4": 13}          # BB72: 6 rounds per shot, d = 13: 13
NETS = (("bb", "BB72 (C14p)"), ("d13", "d = 13 (CS13)"))
BUILD_FIELDS = ("net", "batch", "prec", "conv_prec", "opt_level", "ws_gb", "onnx_sha256", "plan_sha256", "trt", "gpu")
NE = "not established"
LEGEND = "\n\n".join([
    "Times are in microseconds (us). Single: mean latency of one shot. Batched: time per round (the per-shot time divided by "
    "6 for BB72 and by 13 for d = 13), as stage 1 reports it.",
    "R1: BB72, D1 harness, deep pairs (S55p_deep -> C14p); S = the S network's engine (base = BB engine, new = multi-shot "
    "engine). R2: BB72, phase C pairs (S55p -> C14p). R4: d = 13 pair (ConvLines -> CS13).",
    "Cascade alone is the fastest of 3 fresh processes, a pair the slowest of 3. The chosen engine is the Cascade engine with "
    "the lower Cascade-alone time (a tie goes to compiled). Speed-up = Cascade alone / pair, both on the chosen engine. The "
    "other engine's pair and speed-up decide nothing.",
    "The stage 1 columns are the published stage-1 values (compiled path, slowest of 3 on both sides, speed-up = alone / "
    "pair); n/a = no matching published value."])

_row_problems = ac.row_problems


def row_problems(meta, r, rnd, freeze, derived=None):
    """analysis_cas.row_problems, plus: a row's own Cascade engine (its "ctrt" field, when it has one) is its slot's."""
    bad = _row_problems(meta, r, rnd, freeze, derived)
    e = (r.get("ctrt") or {}).get("engine")
    if e is not None and e != meta.get("cascade_engine"):
        bad.append("Cascade engine %r not %r" % (e, meta.get("cascade_engine")))
    return bad


ac.row_problems = row_problems                           # analyze_card looks it up in its module at call time


def clean(c):
    """An analysis_cas cell as numbers.json keeps it: no host, and "complete" = the accepted values of reps 1, 2 and 3."""
    reps = sorted(r["rep"] for r in c["reps"])
    done = not c["withheld"] and reps == [1, 2, 3]
    return dict(c, complete=done, why=None if done else (c["why"] or "reps %s are not 1, 2, 3" % reps),
                reps=[{k: v for k, v in r.items() if k != "host"} for r in c["reps"]])


def val(cells, key, how):
    """(the cell's fastest or slowest value, None), or (None, why) when the cell is not established."""
    c = cells.get(key)
    if c is None:
        return None, "no records"
    return (c[how], None) if c["complete"] else (None, c["why"])


def key(rnd, who, p, mode, s_eng=None):
    """Stage 1's cell key of one side of a comparison (analysis_cas.keys_of's: single mean, batched per shot)."""
    r = rnd.lower()
    head = "%s.%s" % (r, "C2" if who == "alone" else "C5") if r == "r4" else "%s.dfm.%s" % (r, who)
    return "%s.p%g.%s.%s" % (head, p, mode, "mean" if mode == "single" else "per_shot") + (".%s" % s_eng if s_eng else "")


def load_card(root, card, checks, shots, seeds):
    """One card's cells {engine: {cell key: cell}} and the status of each (round, engine) group; appends to the checks."""
    cells, status = {e: {} for e in ENGINES}, {}
    for rnd in ROUNDS:
        path = os.path.join(root, card, rnd)
        if not os.path.isfile(os.path.join(path, "attempts.json")):
            checks.append("%s %s: no records (no attempts.json)" % (card, rnd))
            continue
        A = ac.load_round(path)
        seeds["%s %s" % (card, rnd)] = A.get("seeds_sha256")
        if A.get("format") != "ctrt-cas-attempts-1" or str(A.get("round")).lower() != rnd.lower():
            checks.append("%s %s: format %r, round %r (expected 'ctrt-cas-attempts-1', %r)"
                          % (card, rnd, A.get("format"), A.get("round"), rnd.lower()))
        if A.get("dry"):
            checks.append("%s %s: a dry record" % (card, rnd))
        odd = [s["id"] for s in A["slots"] if str(s["meta"].get("round")).lower() != rnd.lower()
               or s["meta"].get("cascade_engine") not in ENGINES or not s["id"].endswith(".cas-" + s["meta"]["cascade_engine"])]
        if odd:
            checks.append("%s %s: slots whose id, round or cascade_engine do not fit the record: %s" % (card, rnd, ", ".join(odd)))
        for s in A["slots"]:                             # the disagreement analyze_card does not report: windows hold, gate said no
            for at in s["attempts"]:
                if not at["hard"] and not at["accepted"] and ac.windows_ok(at["windows"]):
                    checks.append("%s %s %s a%s: not accepted by the gate (%s) but its windows hold"
                                  % (card, rnd, s["id"], at["k"], "; ".join(at["soft"]) or "no reason"))
        for e in ENGINES:
            B = dict(A, slots=[s for s in A["slots"] if s["meta"].get("cascade_engine") == e])
            res, ch = ac.analyze_card(card, {rnd.lower(): B}, None, True)
            checks += ch
            cells[e].update({k: clean(c) for k, c in res["cells"].items()})
            status["%s.%s" % (rnd, e)] = res["status"][rnd.lower()]
            for name, by in res["shots"].items():
                for chunk, digest in by.items():
                    shots.setdefault((name, int(chunk)), set()).add(digest)
    for e in ENGINES:
        miss = sorted({sid for c in cells[e].values() for sid in c["withheld"]})
        if miss:
            checks.append("%s %s: slot(s) not established: %s" % (card, e, ", ".join(miss)))
    return cells, status


def compare(cells, s1, rnd, p, mode, s_eng):
    """One comparison of one card: Cascade alone and the pair on each engine, the chosen engine, stage 1's values."""
    ka, kp = key(rnd, "alone", p, mode), key(rnd, "pair", p, mode, s_eng)
    sc = 1.0 if mode == "single" else 1.0 / ROUNDS_PER_SHOT[rnd]
    c = dict(round=rnd, p=p, mode=mode, s_engine=s_eng, unit="us" if mode == "single" else "us per round",
             key_alone=ka, key_pair=kp, alone={}, pair={}, speedup={})
    why = {}
    for e in ENGINES:
        a, why["alone", e] = val(cells[e], ka, "fastest")
        q, why["pair", e] = val(cells[e], kp, "value")
        c["alone"][e] = None if a is None else a * sc
        c["pair"][e] = None if q is None else q * sc
        c["speedup"][e] = None if a is None or q is None else a / q
    alone = c["alone"]
    c["chosen"] = None if None in alone.values() else min(ENGINES, key=lambda e: alone[e])     # a tie: the first, compiled
    need = [("alone", e) for e in ENGINES] + ([("pair", c["chosen"])] if c["chosen"] else [])
    c["withheld"] = "; ".join("%s %s: %s" % (s, e, why[s, e]) for s, e in need if why[s, e]) or None
    sa, sp = (s1.get(k, {}).get("value") for k in (ka, kp))
    c["stage1"] = dict(alone=None if sa is None else sa * sc, pair=None if sp is None else sp * sc,
                       speedup=None if sa is None or sp is None else sa / sp)
    return c


def comparisons(cells, s1):
    return [compare(cells, s1, rnd, p, mode, s_eng) for rnd in ROUNDS for p in PS[rnd] for mode in ("single", "batched")
            for s_eng in (("base", "new") if rnd == "R1" else (None,))]


# ------------------------------------------------------------------------------------------------------------ tables.md
def fmt(v, spec, none="n/a"):
    return none if v is None else format(v, spec)


def mdt(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + " --- |" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


def label(c):
    return "%s p=%g %% %s%s (%s)" % (c["round"], 100 * c["p"], c["mode"], " S=" + c["s_engine"] if c["s_engine"] else "", c["unit"])


def comparison_rows(comps):
    rows = []
    for c in comps:
        ch, s1 = c["chosen"], c["stage1"]
        oth = None if ch is None else ENGINES[1 - ENGINES.index(ch)]
        held = c["withheld"] is not None
        rows.append([label(c), fmt(c["alone"]["compiled"], ".1f", NE), fmt(c["alone"]["trt"], ".1f", NE), ch or "withheld",
                     "withheld" if held else fmt(c["pair"][ch], ".1f"), "withheld" if held else fmt(c["speedup"][ch], ".2f"),
                     fmt(s1["alone"], ".1f"), fmt(s1["pair"], ".1f"), fmt(s1["speedup"], ".2f"),
                     "-" if oth is None else "%s: %s" % (oth, fmt(c["pair"][oth], ".1f", NE)),
                     "-" if oth is None else "%s: %s" % (oth, fmt(c["speedup"][oth], ".2f", NE))])
    return rows


def decision_rows(numbers):
    rows = []
    for card in CARDS:
        for net, name in NETS:
            res = numbers["cards"][card]["decision_checks"][net]
            if res is None:
                rows.append([card, name, "no records"] + ["-"] * 9)
                continue
            for path, pname in (("bB", "batch %s" % res["batch"]), ("b1", "batch 1")):
                parts = [(p, res["p"][p][path]) for p in sorted(res["p"], key=float)] + [("total", res.get("total_" + path))]
                for p, r in parts:
                    if r is not None:
                        rows.append([card, name, "total" if p == "total" else "%g %%" % (100 * float(p)), pname,
                                     fmt(r.get("shots"), "d"), fmt(r.get("decisions_differ"), "d"),
                                     fmt(r.get("agreement"), ".6f"), fmt(r.get("fail_compiled"), "d"), fmt(r.get("fail_trt"), "d"),
                                     fmt(r.get("only_compiled_fails"), "d"), fmt(r.get("only_trt_fails"), "d"),
                                     fmt(r.get("mcnemar_exact_p"), ".4g")])
    return rows


def build_rows(numbers):
    rows = []
    for card in CARDS:
        bs = numbers["cards"][card]["builds"]
        rows += [[card] + [fmt(b[k], "") for k in BUILD_FIELDS] for b in bs] or [[card, "no records"] + ["-"] * 9]
    return rows


def tables(numbers):
    L = ["# CASCADE-TRT-REG: Cascade on two engines", "", LEGEND, ""]
    for card in CARDS:
        comps = numbers["cards"][card]["comparisons"]
        head = ["Comparison", "Cascade alone, compiled (fastest of 3)", "Cascade alone, TensorRT (fastest of 3)", "Chosen engine",
                "Pair, chosen engine (slowest of 3)", "Speed-up", "Stage 1 alone", "Stage 1 pair", "Stage 1 speed-up",
                "Other engine's pair (decides nothing)", "Other engine's speed-up (decides nothing)"]
        L += ["## %s: Cascade alone, pairs and speed-ups" % card, "", mdt(head, comparison_rows(comps)), ""]
        held = ["- %s: %s" % (label(c), c["withheld"]) for c in comps if c["withheld"]]
        L += (["Withheld:"] + held + [""]) if held else []
    L += ["## Decision checks (TensorRT against the compiled path, fresh shots)", "",
          mdt(["Card", "Cascade", "p", "Path", "Shots", "Decisions differ", "Agreement", "Failures, compiled", "Failures, TensorRT",
               "Only compiled fails", "Only TensorRT fails", "Exact McNemar p"], decision_rows(numbers)), "",
          "## Engines (TensorRT builds)", "",
          mdt(["Card", "Net", "Batch", "Precision", "Convolutions held", "Optimisation level", "Workspace GB (0 = default)",
               "ONNX sha256", "Plan sha256", "TensorRT", "GPU"], build_rows(numbers)), "", "## Checks", ""]
    return "\n".join(L + (["- " + x for x in numbers["checks"]] or ["none"])) + "\n"


# ---------------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True, metavar="RECORDS")
    ap.add_argument("--out", required=True, metavar="OUTDIR")
    a = ap.parse_args()
    if os.path.exists(a.out):
        raise SystemExit("%s exists: the analysis writes into a fresh directory" % a.out)
    with open(os.path.join(STAGE1, "results", "numbers.json")) as fh:
        stage1 = json.load(fh)["cards"]
    checks, shots, seeds, cards = [], {}, {}, {}
    for card in CARDS:
        cells, status = load_card(a.records, card, checks, shots, seeds)
        dec, builds = {}, ac.jl(os.path.join(a.records, card, "build", "build.jsonl"))
        for net, _ in NETS:
            path = os.path.join(a.records, card, "check", "check_%s.json" % net)
            dec[net] = json.load(open(path)) if os.path.isfile(path) else None
            if dec[net] is None:
                checks.append("%s: no check_%s.json" % (card, net))
        if not builds:
            checks.append("%s: no build.jsonl rows" % card)
        cards[card] = dict(cells=cells, status=status, comparisons=comparisons(cells, stage1[card]["cells"]),
                           decision_checks=dec, builds=[{k: b.get(k) for k in BUILD_FIELDS} for b in builds])
    for name, chunk in sorted(shots):
        if len(shots[name, chunk]) > 1:
            checks.append("shots: chunk %s of %s has %d different digests across the records" % (chunk, name, len(shots[name, chunk])))
    if len(set(seeds.values())) > 1 or None in seeds.values():
        checks.append("seeds_sha256 differs between records or is missing: " +
                      ", ".join("%s=%s" % (k, "none" if v is None else v) for k, v in sorted(seeds.items())))
    numbers = dict(cards=cards, checks=sorted(set(checks)))
    os.makedirs(a.out)
    with open(os.path.join(a.out, "numbers.json"), "w") as fh:
        json.dump(numbers, fh, indent=1, sort_keys=True)
        fh.write("\n")
    with open(os.path.join(a.out, "tables.md"), "w") as fh:
        fh.write(tables(numbers))
    print("%d checks" % len(numbers["checks"]))
    for x in numbers["checks"]:
        print("  " + x)


if __name__ == "__main__":
    main()
