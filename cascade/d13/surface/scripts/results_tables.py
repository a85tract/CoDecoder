"""results_tables.py OUT: the tables of surface/RESULTS.md, printed as markdown from confirm_d13.json,
bitwise_c7.json (its per-process timings), locked_d13.json and clockflags_e7.jsonl in OUT (surface/out)."""
import json
import sys

D = sys.argv[1]
C = json.load(open(f"{D}/confirm_d13.json"))
B = json.load(open(f"{D}/bitwise_c7.json"))
L = json.load(open(f"{D}/locked_d13.json"))
F7 = [json.loads(x) for x in open(f"{D}/clockflags_e7.jsonl")]
PS = ("0.006", "0.0035")
PN = {"0.006": "0.6 %", "0.0035": "0.35 %"}
MODES = ("single", "batched")
lab = C["labels"]
hyp = {(f"{h['p']:g}", h["mode"], h["test"], h["x"], h["y"]): h for h in C["hypotheses"]}


def key(p):
    return f"{float(p):g}"


def n(x):
    return f"{x:,}"


def g(x, d=3):
    return f"{x:.{d}g}"


plain = {"better": "fails less often than Cascade alone (shown)", "worse": "fails more often than Cascade alone (shown)",
         "level": "no difference shown either way", "non-inferior": "at most 10 % more failures than Cascade alone (shown)"}

print("### Labels against Cascade alone (C2)\n")
print("| p | mode | C1 vs C2 | C3 vs C2 | C4 vs C2 | C5 vs C2 | plain meaning |")
print("|---|---|---|---|---|---|---|")
for p in PS:
    for m in MODES:
        ls = {x: lab[f"{p}|{m}|{x}-C2"] for x in ("C1", "C3", "C4", "C5")}
        words = {"better": ("fails", "less often"), "worse": ("fails", "more often"),
                 "level": ("shows", "no difference"), "non-inferior": ("stays", "within the 10 % margin")}
        pm = []
        for k in ("better", "non-inferior", "level", "worse"):
            xs = [x for x in ls if ls[x] == k]
            if xs:
                v, rest = words[k]
                pm.append(" and ".join(xs) + " " + (v if len(xs) == 1 else v[:-1]) + " " + rest)
        pm = "; ".join(pm)
        print(f"| {PN[p]} | {m} | " + " | ".join(ls.values()) + f" | {pm} |")

print("\n### Failures on the confirmation block\n")
print("| p | mode | shots | C1 | C2 | C3 | C4 | C5 | plain meaning |")
print("|---|---|---|---|---|---|---|---|---|")
for p in PS:
    for m in MODES:
        cf = C["p"][p]["modes"][m]["configs"]
        print(f"| {PN[p]} | {m} | {n(cf['C2']['shots'])} | " + " | ".join(n(cf[x]["fail"]) for x in ("C1", "C2", "C3", "C4", "C5"))
              + " | " + ", ".join(f"{x} {100 * (cf[x]['fail'] / cf['C2']['fail'] - 1):+.1f} %" for x in ("C1", "C3", "C4", "C5"))
              + " failures against C2 |")

print("\n### Each chain and baseline against Cascade alone (paired on the same shots)\n")
print("| p | mode | X | X fails, C2 not | C2 fails, X not | McNemar p | Holm level | P(X) - P(C2), Newcombe 95 % | "
      "non-inferiority p (margin delta_p) | label | plain meaning |")
print("|---|---|---|---|---|---|---|---|---|---|---|")
for p in PS:
    dp = L["p"][p]["delta"]
    for m in MODES:
        for x in ("C1", "C5", "C4", "C3"):
            h = hyp[(key(p), m, "S", x, "C2")]
            hn = hyp.get((key(p), m, "N", x, "C2"))
            lv = g(h["holm_level"]) if h.get("rejected") else "not reached"
            ni = "-"
            if hn:
                ni = f"{g(hn['pvalue'])} ({'rejected' if hn['rejected'] else 'not rejected'}; delta_p {g(dp, 2)})"
            lb = lab[f"{p}|{m}|{x}-C2"]
            print(f"| {PN[p]} | {m} | {x} | {n(h['only_x'])} | {n(h['only_y'])} | {g(h['pvalue'])} | {lv} | "
                  f"({g(h['lo95'])}, {g(h['hi95'])}) | {ni} | {lb} | {plain[lb]} |")

print("\n### The other comparisons of the family\n")
print("| p | mode | C1 vs C4 | C5 vs C4 | C1 vs C5 | C4 vs C3 | plain meaning |")
print("|---|---|---|---|---|---|---|")
for p in PS:
    for m in MODES:
        cs = ("C1-C4", "C5-C4", "C1-C5", "C4-C3")
        w2 = {"better": "fails less often than", "worse": "fails more often than", "level": "shows no difference from",
              "non-inferior": "stays within the margin of"}
        pm = "; ".join(f"{c[:2]} {w2[lab[f'{p}|{m}|{c}']]} {c[3:]}" for c in cs)
        print(f"| {PN[p]} | {m} | " + " | ".join(lab[f"{p}|{m}|{c}"] for c in cs) + f" | {pm} |")

print("\n### Speed: each chain against Cascade alone timed in the same job (e7)\n")
rows = B["rows"]
fl = {(r["config"], key(r["p"]), r["mode"], r["rep"]): r for r in F7}


def slow(cfg, p, m):
    rs = [r for r in rows if r["config"] == cfg and key(r["p"]) == key(p) and r["mode"] == m]
    assert len(rs) == 3, (cfg, p, m)
    v = max(r["timing"]["value"] for r in rs)
    t = max(r["timing"].get("p999") or 0.0 for r in rs)
    bad = [f"rep {r['rep']}" for r in sorted(rs, key=lambda r: r["rep"])
           if fl[(cfg, key(p), m, r["rep"])]["flag"] != "ok"]
    vrep = max(rs, key=lambda r: r["timing"]["value"])["rep"]
    return v, t, bad, vrep


print("| p | mode | chain | chain, us per shot | C2, us per shot | C2 / chain | p99.9 latency, chain / C2 (us) | "
      "clock flags, chain | clock flags, C2 | plain meaning |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in PS:
    for m in MODES:
        v2, t2, b2, r2 = slow("C2", p, m)
        for x in ("C1", "C5", "C4"):
            v, t, b, r = slow(x, p, m)
            tail = f"{t:.0f} / {t2:.0f}" if m == "single" else "-"
            fx = "all 3 ok" if not b else "below 97 %: " + ", ".join(b) + f" (value from rep {r})"
            f2 = "all 3 ok" if not b2 else "below 97 %: " + ", ".join(b2) + f" (value from rep {r2})"
            pm = f"{x} takes {v / v2:.2f} of C2's time per shot"
            if m == "single" and t > t2:
                pm += "; slower in the tail"
            print(f"| {PN[p]} | {m} | {x} | {v:.1f} | {v2:.1f} | {v2 / v:.2f} | {tail} | {fx} | {f2} | {pm} |")
