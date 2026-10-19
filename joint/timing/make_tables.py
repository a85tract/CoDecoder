"""make_tables.py NEW.json STAGE1.json [BENCH_DIR] > tables.md: TIMING.md's tables from summarize.py's outputs (this
timing and stage 1's records, the latter computed the same way, which reproduces Table I), and the joint step's
benchmark (bench_joint.py outputs BENCH_DIR/{RTX,H100}/{bb_P,d13_P}.json) when given."""
import json
import os
import sys

SETTINGS = [("bb72", 0.002, "BB72, p = 0.2 %"), ("bb72", 0.003, "BB72, p = 0.3 %"),
            ("d13", 0.006, "d = 13, p = 0.6 %"), ("d13", 0.0035, "d = 13, p = 0.35 %")]
CARDS = ("RTX", "H100")


def f(x, nd=1):
    return "-" if x is None else f"{x:,.{nd}f}"


def cells(path):
    out = {}
    for E in json.load(open(path))["cells"]:
        out[(E["card"], E["code"], E["p"], E["mode"])] = E
    return out


def main():
    new, old = cells(sys.argv[1]), cells(sys.argv[2])
    bench = sys.argv[3] if len(sys.argv) > 3 else ""
    L = []
    for code, p, title in SETTINGS:
        L.append(f"### {title}")
        L.append("")
        L.append("| | RTX: single mean | RTX: p99.9 | RTX: batched per round | H100: single mean | H100: p99.9 | "
                 "H100: batched per round |")
        L.append("|---|---|---|---|---|---|---|")

        def row(name, src, dec, key=None):
            r = [name]
            for card in CARDS:
                s, b = src.get((card, code, p, "single")), src.get((card, code, p, "batched"))
                if key is None:
                    vs = s["decoders"][dec]["value"] if s and s["decoders"].get(dec) else {}
                    vb = b["decoders"][dec]["value"] if b and b["decoders"].get(dec) else {}
                    r += [f(vs.get("mean")), f(vs.get("p999")), f(vb.get("per_round"), 3)]
                else:
                    ss = s["decoders"][dec].get("speedup", {}) if s else {}
                    sb = b["decoders"][dec].get("speedup", {}) if b else {}
                    r += [f(ss.get("mean"), 2), "", f(sb.get("per_round"), 2)]
            L.append("| " + " | ".join(r) + " |")
        row("Cascade alone (fastest of 3)", new, "alone")
        row("composition (slowest of 3)", new, "comp")
        row("composition + joint decision (slowest of 3)", new, "joint")
        row("speed-up of the composition", new, "comp", "speedup")
        row("speed-up of the composition + joint decision", new, "joint", "speedup")
        row("*Table I (stage 1): Cascade alone*", old, "alone")
        row("*Table I (stage 1): composition*", old, "comp")
        row("*Table I (stage 1): speed-up*", old, "comp", "speedup")
        L.append("")
        fr = []
        for card in CARDS:
            for mode in ("single", "batched"):
                E = new.get((card, code, p, mode))
                if E:
                    fr.append(f"{card} {mode} {E['decoders']['comp']['frac'] * 100:.2f} %")
        L.append("Routed share (mean of the three composition processes): " + "; ".join(fr) + ".")
        L.append("")
    # the joint step's added time
    L.append("### The joint decision's added time")
    L.append("")
    L.append("| setting | card | mode | routed | routed shots: added per routed shot, paired (range) | the same per "
             "incoming shot | end to end, paired: per incoming shot (per routed shot) | end to end, slowest - slowest | "
             "benchmark, per routed shot |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for code, p, title in SETTINGS:
        for card in CARDS:
            for mode in ("single", "batched"):
                E = new.get((card, code, p, mode))
                if not E:
                    continue
                a = E["joint_added"]
                if mode == "single":
                    pr = a.get("per_routed_via_b_paired") or []
                    per_r, per_i = a.get("per_routed_via_b_mean"), a.get("per_incoming_via_b")
                    e2e = f"{f(a.get('slowest_minus_slowest'), 2)} us per shot"
                else:
                    pr = a.get("per_routed_b_stage_paired") or []
                    per_r, per_i = a.get("per_routed_b_stage_mean"), a.get("per_incoming_b_stage")
                    d = a.get("per_round_slowest_minus_slowest")
                    e2e = f"{f(a.get('slowest_minus_slowest'), 4)} us per shot ({f(d, 4)} per round)"
                rng = f"({f(min(pr), 2)} to {f(max(pr), 2)})" if pr else ""
                bm = "-"
                if bench:
                    bf = os.path.join(bench, card, f"{'bb' if code == 'bb72' else 'd13'}_{p}.json")
                    if os.path.exists(bf):
                        B = json.load(open(bf))
                        if mode == "single":
                            g, sb = B["single"]["gpu_ms"], B["single"]
                            bm = (f"host {f(sb.get('added_agree_us'), 1)} / {f(sb.get('added_differ_us'), 1)} us "
                                  f"(agree / differ); GPU {f(1e3 * (g['joint'] - g['plain']), 1)} us per replay")
                        else:
                            ad = B["batched"].get("added_us_per_routed_shot")
                            if isinstance(ad, dict):
                                k = [x for x in ad if x not in ("joint_ms_frac0", "joint_ms_frac1")]
                                fr = float(k[0].replace("joint_ms_frac", "")) * 100 if k else None
                                bm = (f"{f(ad[k[0]], 3)} us ({f(fr, 1)} % of rows differ; {f(ad['joint_ms_frac0'], 3)} "
                                      f"at none, {f(ad['joint_ms_frac1'], 3)} at all)") if k else "-"
                            else:
                                bm = f"{f(ad, 3)} us (batch {B['batched'].get('batch')}, every row)"
                L.append(f"| {title} | {card} | {mode} | {E['decoders']['joint']['frac'] * 100:.2f} % | "
                         f"{f(per_r, 2 if mode == 'single' else 3)} us {rng} | "
                         f"{f(per_i, 3 if mode == 'single' else 4)} us | "
                         f"{f(a.get('paired_mean'), 2 if mode == 'single' else 4)} us"
                         f"{'' if mode == 'single' else ' (' + f(a.get('per_routed_from_incoming'), 2) + ')'} | "
                         f"{e2e} | {bm} |")
    L.append("")
    # attempts
    bad = json.load(open(sys.argv[1])).get("not_first_attempt", [])
    L.append("Slots not established at their first attempt: " +
             ("; ".join(f"{b['card']} {b['id']} ({b['status']} after {b['attempts']} attempts)" for b in bad) or "none")
             + ".")
    print("\n".join(L))


if __name__ == "__main__":
    main()
