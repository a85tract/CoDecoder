"""Recompute the registered speed scores from the recorded components (results/) and the Grace table
(timing/grace_basis.csv), and check them against the recorded composites and the 38 speed cells (30 of studies 2 and 3
at d = 7-15, 8 of study 4 at d = 17).

    composite = (a) + (b) + (c)    us per shot, batch 1
      (a) the network's GPU time per shot on the session H100, GPU-resident input (timing/gpu_term.py; slowest of five
          fresh processes)
      (b) b_node x ratio: the routed shots' correlated-PM time per shot on one exclusive CPU node (timing/cpu_term.py;
          slowest of three repetitions) times ratio = G(d) / (the same node's correlated-PM time per block, median of
          three), G(d) = the Grace-basis correlated-PM time per block (grace_basis.csv)
      (c) the handoff between the GPU and the host: at d = 7 measured for the d = 7 kernel (host round trip minus the
          device time, largest of five processes); for the FROZEN engine max(1.1 us, the router's estimate on the
          session host)
    opponents: Ising = its fastest validated TensorRT plan at batch 1 on the session H100 + its PM share on the Grace
    basis (study 4 also registered Ising + correlated PM: the same plan + the correlated-PM share of grace_basis.csv);
    plain and correlated PM = grace_basis.csv; BP-OSD = measured on the session host (GPU BP + host OSD), unconverted.
    A speed cell holds when ours is strictly below the opponent.

    The two post hoc controls Ising M1 / M4 + correlated PM (results/controls_speed.json; README, "The two post hoc
    controls"): the registered plan time + the node's correlated-PM share x factor, factor = Ising's Grace PM share
    after the model at d = 13 / the node's plain-PM share after it at d = 13 (beside it: G(13) / the node's correlated
    PM per block at d = 13), recomputed from the recorded medians and checked against the recorded values; then ours'
    composite against both, with and without the handoff (c), under both factors. On the M4 basis (decides nothing):
    the plan time + the slowest of the Apple M4 core's three shares, recomputed and checked likewise.

    python -m timing.composite"""
import csv
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEY = {7: "d7_b1w64_r740", 9: "d9_b3_r764", 11: "d11_b5_r777", 13: "d13_b5_r775"}
SHARE = {"Ising M1 + PM": "PM share after Ising M1", "Ising M4 + PM": "PM share after Ising M4",
         "Ising M1 + correlated PM": "correlated-PM share after Ising M1",
         "Ising M4 + correlated PM": "correlated-PM share after Ising M4"}


def grace():
    return {(int(r["d"]), r["decoder"]): float(r["grace_us_per_shot"])
            for r in csv.DictReader(open(os.path.join(ROOT, "timing", "grace_basis.csv")))}


def rows():
    """[(d, a, b_node, ratio, c, composite recorded, {opponent: us recorded}, {opponent: plan us or None})]."""
    s2 = json.load(open(os.path.join(ROOT, "results", "s2_composite.json")))
    out = []
    for d, k in KEY.items():
        c = s2["cands"][k]
        r = s2["ratio"][str(d)]
        out.append((d, c["a_gpu"], c["b_node"], r["numerator"] / r["node_median"], c["c_handoff"], c["composite_grace"],
                    {n: o["grace"] for n, o in c["opponents"].items()},
                    {n: o.get("plan_us") for n, o in c["opponents"].items()}, r["numerator"],
                    c.get("c_allowance"), c.get("c_measured")))
    s3 = json.load(open(os.path.join(ROOT, "results", "results_accuracy_speed_d15.json")))["composite_inputs"]
    out.append((15, s3["a_gpu"], s3["b_node"], s3["ratio"]["numerator"] / s3["ratio"]["node_median"], s3["c_handoff"],
                s3["composite_grace"], s3["opponents_grace"],
                {n: s3["ising_plans"].get(n, {}).get("plan_us") for n in s3["opponents_grace"]}, s3["ratio"]["numerator"],
                s3["c_allowance"], s3["c_measured"]))
    s4 = json.load(open(os.path.join(ROOT, "results", "results_accuracy_speed_d17.json")))["composite_inputs"]
    out.append((17, s4["a_gpu"], s4["b_node"], s4["ratio"]["numerator"] / s4["ratio"]["node_median"], s4["c_handoff"],
                s4["composite_grace"], s4["opponents_grace"],
                {n: s4["ising_plans"].get(n, {}).get("plan_us") for n in s4["opponents_grace"]}, s4["ratio"]["numerator"],
                s4["c_allowance"], s4["c_measured"]))
    return out


def controls(g, rows_):
    sp = json.load(open(os.path.join(ROOT, "results", "controls_speed.json")))
    fac, bes = sp["factors"]["primary"], sp["factors"]["beside"]
    ok = abs(bes["value"] - g[(13, "correlated PM")] / bes["node_corr_ref_d13_us"]) < 1e-12
    for m, f in fac.items():
        ok &= abs(f["grace_share_d13_us"] - g[(13, f"PM share after Ising {m}")]) < 1e-9
        ok &= f["node_plain_share_d13_us"] == sp["d"]["13"][m]["node_plain_share"]["median_us"]
        ok &= abs(f["value"] - f["grace_share_d13_us"] / f["node_plain_share_d13_us"]) < 1e-12
    print("\ncontrols  plan     node share  Grace share  composite (beside)  ours / without (c)  cells     M4 basis")
    cells = 0
    for d, _, _, _, c, rec, _, plan, *_ in rows_:
        if str(d) not in sp["d"]:                                             # d = 17: registered opponents, not controls
            continue
        for m, f in fac.items():
            x = sp["d"][str(d)][m]
            share = x["node_corr_share"]["median_us"]
            ok &= share == statistics.median(x["node_corr_share"]["reps_us"])
            ok &= x["plan_us"] == plan[f"Ising {m} + PM"]
            us, beside = x["plan_us"] + share * f["value"], x["plan_us"] + share * bes["value"]
            ok &= abs(us - x["grace_us"]) < 1e-9 and abs(beside - x["grace_beside_us"]) < 1e-9
            m4 = x["plan_us"] + max(x["m4_corr_share"]["reps_us"])
            ok &= x["m4_corr_share"]["slowest_us"] == max(x["m4_corr_share"]["reps_us"]) and abs(m4 - x["m4_us"]) < 1e-9
            held = sum((rec < us, rec < beside, rec - c < us, rec - c < beside))
            cells += held
            print(f"{d:<3d} {m}    {x['plan_us']:7.3f}  {share:9.3f}  {share * f['value']:11.3f}  {us:9.3f} "
                  f"({beside:7.3f})  {rec:7.3f} / {rec - c:7.3f}  {held}/4 hold  {m4:8.3f}")
    print(f"controls' composites reproduced: {ok}; ours below the controls in {cells} of 40 cells (5 d x 2 models x 2 "
          "boundaries x 2 factors)")
    return ok


def main():
    g = grace()
    ok = True
    print("d   (a) GPU  (b) node  ratio   (b) Grace  (c)    composite  fastest opponent      margin  cells")
    for d, a, b_node, ratio, c, rec, opp, plan, num, allow, meas in rows():
        ok &= abs(num - g[(d, "correlated PM")]) < 1e-9                       # the ratio's numerator is the table's
        if allow is not None:
            ok &= abs(c - max(allow, meas)) < 1e-12                           # FROZEN: max(1.1 us, the estimate)
        comp = a + b_node * ratio + c
        ok &= abs(comp - rec) < 1e-9
        cells = 0
        for n, us in opp.items():
            if n in SHARE:                                                     # Ising: plan + the Grace PM share
                ok &= abs(us - (plan[n] + g[(d, SHARE[n])])) < 0.006
            elif n in ("plain PM", "correlated PM"):
                ok &= abs(us - g[(d, n)]) < 1e-9
            cells += comp < us
        fast = min(opp, key=opp.get)
        print(f"{d:<3d} {a:8.3f}  {b_node:8.3f}  {ratio:.4f}  {b_node * ratio:9.3f}  {c:5.3f}  {comp:9.3f}  "
              f"{fast:20s} {opp[fast] - comp:7.3f}  {cells}/{len(opp)} hold")
    print("recorded composites, Grace numbers and opponents reproduced:", ok)
    ok &= controls(g, rows())
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
