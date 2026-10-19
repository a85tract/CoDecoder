"""Table I's extra column: failures of the frozen joint decision on the re-test's shots, beside Cascade alone and the
composition, with repairs and breaks against the composition and exact McNemar tests (scipy binomtest(b, b + c, 0.5)
on the discordant shots: bb.route.paired for BB72, sv_common.mcnemar_p for d = 13, the same test).

BB72 (eager path): every shot's fail0 / fail1 / conf1 from the read (alpine_extract_bb.py), the composition = C14' on
the routed shots (S55'-deep's margin below tau1), S55'-deep elsewhere; the joint decision = the composition except on
the routed shots whose decisions differ, where A3's answer (a3_pod.py on the pod) replaces it.
d = 13 (single and batched): sv_common.load_read over the confirmation chunks, the composition C5 and Cascade alone C2
by sv_common.assemble (single: S1, B1; batched: SB, BQ5 / Bb); the joint decision = C5 except on the routed shots
whose two decisions differ, where it answers 1 iff c + w z > 0 in float64 (on a tie, Cascade's answer).

  data/venv/bin/python code/table1.py --w13 0.006=W 0.0035=W [--src TEXT]  ->  out/table1.json, out/table1.md"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path[:0] = [f"{DATA}/stage/release/cascade/bb72/ising_compare", f"{DATA}/stage/release/cascade/bb72/qldpc_neural",
                f"{DATA}/release_d13/cascade/d13"]
from bb.route import paired                      # noqa: E402
from surface import sv_common as sc              # noqa: E402

TAU1_BB = {"p0.00200": 7.3331888389587405, "p0.00300": 7.60439167022705}
W_BB = {"p0.00200": 6.309573444801933, "p0.00300": 3.758374042884441}
# Table I (the re-test): BB72 eager path (D1 confirm run_1, d1_tables.md); d = 13 C2 / C5 (SURFACE_RESULT_retest.md)
REF = {("bb72", "p0.00200", "eager"): (8063, 7981), ("bb72", "p0.00300", "eager"): (75957, 75700),
       ("d13", 0.006, "single"): (4936, 4917), ("d13", 0.006, "batched"): (4947, 4927),
       ("d13", 0.0035, "single"): (2354, 2325), ("d13", 0.0035, "batched"): (2352, 2322)}
LOCK13 = f"{DATA}/release_d13/cascade/d13/out/locked_d13.json"


def row(casc, comp, joint, ref, extra):
    vc, vC = paired(joint, comp), paired(joint, casc)
    r = {"shots": int(len(comp)), "cascade_fail": int(casc.sum()), "composition_fail": int(comp.sum()),
         "joint_fail": int(joint.sum()), "repairs": int((comp & ~joint).sum()), "breaks": int((~comp & joint).sum()),
         "p_joint_vs_composition": float(vc["mcnemar_p"]), "p_joint_vs_cascade": float(vC["mcnemar_p"]),
         "p_composition_vs_cascade": float(paired(comp, casc)["mcnemar_p"]),
         "only_joint_vs_cascade": [vC["only_a"], vC["only_b"]], "diff_ler_vs_composition": [vc["diff_lo"], vc["diff_hi"]],
         "diff_ler_vs_cascade": [vC["diff_lo"], vC["diff_hi"]],
         "table_I": {"cascade": ref[0], "composition": ref[1],
                     "equal": [int(casc.sum()), int(comp.sum())] == list(ref)}, **extra}
    assert r["table_I"]["equal"], f"counts {r['cascade_fail']}, {r['composition_fail']} are not Table I's {ref}"
    return r


def bb72():
    out = {}
    bound = json.load(open(f"{ROOT}/out/pod/read/a3_bound.json"))
    for tag, tau1 in TAU1_BB.items():
        d = f"{DATA}/bb/{tag}"
        f0, f1, c1 = (np.load(f"{d}/all_{k}.npy") for k in ("fail0", "fail1", "conf1"))
        routed = c1.astype(np.float64) < tau1
        comp = np.where(routed, f0, f1)
        a = np.load(f"{ROOT}/out/pod/read/{tag}/{tag}_a3.npz")
        rows = a["rows"]
        assert np.array_equal(rows, np.load(f"{d}/sel_rows.npy")) and routed[rows].all()
        assert np.array_equal(comp[rows], a["yC"] != a["yO"]) and np.array_equal(f1[rows], a["yN"] != a["yO"])
        joint = comp.copy()
        joint[rows] = a["yA"] != a["yO"]
        b = bound[tag]
        lo, hi = int(joint.sum()) - b["may_repair"], int(joint.sum()) + b["may_break"]
        out[tag] = {"eager": row(f0, comp, joint, REF[("bb72", tag, "eager")], {
            "w": W_BB[tag], "tau1": tau1, "routed": int(routed.sum()), "disagreement_rows": int(len(rows)),
            "joint_takes_network": int((a["yA"] == a["yN"]).sum()), "joint_third_answer":
                int(((a["yA"] != a["yN"]) & (a["yA"] != a["yC"])).sum()),
            "float16_logits": {"answers_that_may_change": b["answer_may_change"], "joint_fail_range": [lo, hi]}})}
    return out


def d13(w13):
    L = json.load(open(LOCK13))
    out = {}
    for p in (0.006, 0.0035):
        q = L["p"][str(p)]
        ch = q["chosen"]
        o = sc.load_read(f"{DATA}/d13", p, ["S1", "SB", "B1", "Bb", "BQ5"], range(*q["conf_chunks"]))
        obs = o["obs"]
        out[str(p)] = {}
        for mode in ("single", "batched"):
            S, Bn = ("S1", "B1") if mode == "single" else ("SB", "BQ5")
            d5, st5 = sc.assemble("C5", mode, o, ch)
            d2, _ = sc.assemble("C2", mode, o, ch)
            c, z = o[S]["c"].astype(np.float64), o[Bn]["z"].astype(np.float64)
            sd, bd = o[S]["dec"], o[Bn]["dec"]
            routed = st5 == 1
            assert np.array_equal(routed, sc.routed(o[S]["c"], ch["C5"]["tau1"])) and np.isfinite(z[routed]).all()
            dis = routed & (sd != bd)
            s = c + w13[p] * z
            jd = d5.copy()
            jd[dis] = np.where(s[dis] > 0, 1, np.where(s[dis] < 0, 0, bd[dis]))
            extra = {"w": w13[p], "tau1": ch["C5"]["tau1"], "routed": int(routed.sum()), "disagreement_rows": int(dis.sum()),
                     "joint_takes_network": int((jd[dis] == sd[dis]).sum()), "ties": int((s[dis] == 0).sum()),
                     "dec_ne_sign": {"S": int((sd[dis] != (c[dis] > 0)).sum()), "B": int((bd[dis] != (z[dis] > 0)).sum())},
                     "cascade": "B1" if mode == "single" else "Bb", "composition": f"{S} + {Bn}"}
            out[str(p)][mode] = row(d2 != obs, d5 != obs, jd != obs, REF[("d13", p, mode)], extra)
    return out


def fmt_p(p):
    return "< 1e-300" if p == 0 else f"{p:.2g}"


def markdown(res):
    """The one table of RESULTS.md."""
    head = ("| setting | path | shots | w | Cascade alone | composition | joint decision | repairs / breaks (vs composition) "
            "| change vs composition | McNemar p, joint vs composition | McNemar p, joint vs Cascade alone |")
    lines = [head, "|" + "---|" * 11]
    names = {"p0.00200": "BB72, p = 0.2 %", "p0.00300": "BB72, p = 0.3 %", "0.006": "d = 13, p = 0.6 %",
             "0.0035": "d = 13, p = 0.35 %"}
    for code in ("bb72", "d13"):
        for p, modes in res[code].items():
            for mode, r in modes.items():
                rel = (r["joint_fail"] - r["composition_fail"]) / r["composition_fail"]
                lines.append(f"| {names[p]} | {mode} | {r['shots']:,} | {r['w']:.6g} | {r['cascade_fail']:,} | "
                             f"{r['composition_fail']:,} | {r['joint_fail']:,} | {r['repairs']:,} / {r['breaks']:,} | "
                             f"{100 * rel:+.1f} % | {fmt_p(r['p_joint_vs_composition'])} | "
                             f"{fmt_p(r['p_joint_vs_cascade'])} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--w13", nargs=2, required=True, metavar="P=W", help="rule J's w per p (0.006=..., 0.0035=...)")
    ap.add_argument("--src", default="", help="where the d = 13 w come from (recorded)")
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "table1.json"))
    a = ap.parse_args()
    w13 = {float(k): float(v) for k, v in (x.split("=") for x in a.w13)}
    res = {"bb72": bb72(), "d13": d13(w13), "w13_source": a.src}
    json.dump(res, open(a.out, "w"), indent=1)
    open(a.out[:-5] + ".md", "w").write(markdown(res))
    for code, per in (("bb72", res["bb72"]), ("d13", res["d13"])):
        for p, modes in per.items():
            for mode, r in modes.items():
                print(f"{code} {p} {mode}: shots {r['shots']:,} | Cascade {r['cascade_fail']:,} | composition "
                      f"{r['composition_fail']:,} | joint {r['joint_fail']:,} (repairs {r['repairs']}, breaks {r['breaks']}) "
                      f"| p vs composition {r['p_joint_vs_composition']:.3g} | p vs Cascade {r['p_joint_vs_cascade']:.3g}")


if __name__ == "__main__":
    main()
