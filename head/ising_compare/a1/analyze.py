"""A1's one analysis (revision/REVISION_PLAN.md 2.1, "Evaluation"), frozen with the arms before any test shot is drawn.
    python -m a1.analyze --config data/a1/analysis_surface.json --reads DIR --checks FILE --out DIR [--timing FILE]

The config (data/a1/analysis_{surface,bb72}.json; the dry runs use the *_dry.json twins, which differ in the sets
alone) names the code, the arms in their fixed order, the seeds, the cal and test sets, the q grid, the registered q,
the comparisons, the reference rows and the ramp table. Inputs in --reads, by role (cal, test): {arm}_s{seed}_{role}.npz
(a1.read) for every arm, seed and role; fallback_test.npz (a1.fallback); ref_{name}_{role}.npz for the reference rows;
ramp_{arm}_s{seed}_{role}.npz for the ramp runs that exist. --checks: the runs' not-trained checks (a JSON list).
Every read must carry the set's digest (a1.sets.load, sha256-checked); a missing arm, seed or set stops the analysis.

  1. Standalone failures on the test set (q = 0) per arm and seed, and the reference rows.
  2. Pair curves. tau_q = the q-quantile (numpy, linear) of the confidence on the cal set in float64, for q > 0 (q = 0
     routes nothing); a test shot goes to the fallback when its float32 confidence, compared in float64, is below tau_q.
     Pair failures = the fallback's answer on the routed shots and the network's on the others; every arm takes the
     same fallback answers.
  3. Comparisons, "X against Y", the difference = X's failures minus Y's per million test shots, at q = 0 and at the
     registered q. Per seed: the exact two-sided McNemar test on that seed's discordant shots and Newcombe's 95 %
     interval (icmp.stats2.newcombe8, the repository's registered method); discordant counts are never summed over
     seeds. The comparison's p-value is the largest of its seeds' p-values when their differences share one sign, else
     1 (intersection-union). Holm runs over the code's comparisons at both q as one family (7 x 2 on the surface, 5 x 2
     on BB72). Verdict: better (X fewer failures) or worse when the Holm-adjusted p is below alpha = 0.05, else
     inconclusive; never level. Seeds summarized: each seed's difference and interval, their mean and range, and the
     mean's 95 % interval with the test shot as unit (each shot's failure difference averaged over the seeds first;
     normal interval of the mean of those averages).
  4. Failures against time, when --timing gives the arms' network times (A3): t(q) = t_net + the realized routed
     fraction x the fallback's mean time on that arm's routed test shots + the handoff (q > 0). The fallback's times
     come from its own host (a1.fallback), not A3's, so t(q) is labelled a projection (time_basis, the axis label)
     until A3 supplies fallback timings aligned by shot (gate finding 7).
  5. The not-trained table (the checks), the ramp table (ramp runs: standalone and registered-q failures, each ramp
     arm against the deployed arm's ramp run of its seed, per seed, descriptive) and the size table (parameters and
     read-out multiply-adds of each arm's decision path, against the deployed arm's).
Outputs in --out: standalone.csv, pair_curves.csv, comparisons.csv, references.csv, ramp.csv, train_checks.csv,
sizes.csv, time_curves.csv (with timing), pair_curves.png, time_curves.png (with timing), summary.md, analysis.json
(every number, the config, the inputs' sha256)."""
import argparse
import csv
import hashlib
import json
import math
import os

import numpy as np

from a1 import sets

ALPHA = 0.05
Z95 = 1.959963984540054
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]   # dataviz slots 1-8


def sha_file(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_read(path, digest, arm=None, seed=None):
    """A read (a1.read), refused if it is on another set, failed its batch-one check, or is not of arm / seed."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    if meta["set_digest"] != digest:
        raise SystemExit(f"{path}: read on another set ({meta['set_digest'][:12]}, the set is {digest[:12]})")
    if meta["b1_check"]["decisions_differ"]:
        raise SystemExit(f"{path}: the read failed its batch-one check")
    if arm is not None and (meta.get("arm"), meta.get("seed")) != (arm, seed):
        raise SystemExit(f"{path}: a read of {meta.get('arm')} seed {meta.get('seed')}, not {arm} seed {seed}")
    return {"pred": z["pred"], "conf": z["conf"].astype(np.float32), "meta": meta, "sha256": sha_file(path)}


def fails(pred, obs):
    p = pred.reshape(len(pred), -1)
    return (p != obs[:, :p.shape[1]]).any(1)


def tau_of(conf_cal, q):
    return -math.inf if q == 0 else float(np.quantile(conf_cal.astype(np.float64), q))


def pair(read, fb_fail, obs, t):
    routed = read["conf"].astype(np.float64) < t
    return np.where(routed, fb_fail, fails(read["pred"], obs)), routed


def mcnemar(fx, fy):
    from scipy.stats import binomtest
    n10, n01 = int((fx & ~fy).sum()), int((~fx & fy).sum())
    return n10, n01, (binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)


def newcombe(fx, fy):
    from icmp.stats2 import newcombe8
    a = int((fx & fy).sum())
    b = int((fx & ~fy).sum())
    c = int((~fx & fy).sum())
    dt, lo, hi, _ = newcombe8(a, b, c, len(fx) - a - b - c, Z95)
    return 1e6 * dt, 1e6 * lo, 1e6 * hi


def holm(ps):
    ps = np.asarray(ps, float)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(np.argsort(ps, kind="stable")):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = run
    return adj


def shot_clustered(fx_by_seed, fy_by_seed):
    """Mean over the seeds of X - Y per million and its 95 % interval with the test shot as unit."""
    D = np.mean([fx.astype(np.float64) - fy for fx, fy in zip(fx_by_seed, fy_by_seed)], axis=0)
    m, se = D.mean(), D.std(ddof=1) / math.sqrt(len(D))
    return 1e6 * m, 1e6 * (m - Z95 * se), 1e6 * (m + Z95 * se)


def compare(comparisons, fail_at, seeds, qr):
    """The comparisons at q = 0 and at the registered q: per seed McNemar and Newcombe, the intersection-union p-value,
    Holm over all of them, the verdicts and the seeds' summary. fail_at[(arm, seed, q)] = bool failures per test shot."""
    comps = []
    for c in comparisons:
        for q in (0, qr):
            fx = [fail_at[(c["x"], s, q)] for s in seeds]
            fy = [fail_at[(c["y"], s, q)] for s in seeds]
            row = {"id": c["id"], "factor": c["factor"], "x": c["x"], "y": c["y"], "held": c.get("held", ""), "q": q}
            ps, diffs = [], []
            for s, a_, b_ in zip(seeds, fx, fy):
                n10, n01, p = mcnemar(a_, b_)
                dt, lo, hi = newcombe(a_, b_)
                row.update({f"s{s}_fail_x": int(a_.sum()), f"s{s}_fail_y": int(b_.sum()), f"s{s}_only_x": n10,
                            f"s{s}_only_y": n01, f"s{s}_p": p, f"s{s}_diff_pm": dt, f"s{s}_lo": lo, f"s{s}_hi": hi})
                ps.append(p)
                diffs.append(dt)
            same = all(v > 0 for v in diffs) or all(v < 0 for v in diffs)
            row["p_iut"] = max(ps) if same else 1.0
            m, lo, hi = shot_clustered(fx, fy)
            row.update({"mean_diff_pm": m, "mean_lo": lo, "mean_hi": hi, "range_lo": min(diffs), "range_hi": max(diffs)})
            comps.append(row)
    adj = holm([r["p_iut"] for r in comps])
    for r, p in zip(comps, adj):
        r["p_holm"] = float(p)
        r["verdict"] = ("better" if r["mean_diff_pm"] < 0 else "worse") if p < ALPHA else "inconclusive"
    return comps


def write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0])
    for r in rows[1:]:
        keys += [k for k in r if k not in keys]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)


def md_table(rows, keys):
    out = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    for r in rows:
        out.append("| " + " | ".join(fmt(r.get(k, "")) for k in keys) + " |")
    return "\n".join(out)


def fmt(v):
    if isinstance(v, float):
        return f"{v:.3g}" if (v != 0 and (abs(v) < 1e-2 or abs(v) >= 1e5)) else f"{v:.2f}"
    return str(v)


def log_axis(axis):
    """A log axis labelled with plain numbers at 1, 2 and 5 times a power of ten, minor ticks unlabelled."""
    from matplotlib import ticker
    axis.set_major_locator(ticker.LogLocator(subs=(1.0, 2.0, 5.0)))
    axis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 1 else f"{v:g}"))
    axis.set_minor_formatter(ticker.NullFormatter())


def plot_pairs(path, cfg, curves, refs, n_test):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for i, arm in enumerate(cfg["arms"]):
        for k, s in enumerate(cfg["seeds"]):
            rows = [r for r in curves if r["arm"] == arm and r["seed"] == s]
            ax.plot([100 * r["q"] for r in rows], [r["per_million"] for r in rows], color=COLORS[i], lw=1.0,
                    ls="-" if k == 0 else "--", marker="o" if k == 0 else "s", ms=4, mec="white", mew=0.6,
                    label=f"{arm} seed {s}")
    for name, rows in refs.items():
        ax.plot([100 * r["q"] for r in rows], [r["per_million"] for r in rows], color="#6b6b6b", lw=1.0, ls=":",
                marker="^", ms=4, mec="white", mew=0.6, label=f"{name} (reference)")
    ax.set_yscale("log")
    log_axis(ax.yaxis)
    ax.set_xlabel("routed fraction q (%)", color="#3d3d3a")
    ax.set_ylabel(f"pair failures per million test shots (n = {n_test:,})", color="#3d3d3a")
    ax.set_title(f"{cfg['code']}: failures against routed fraction, fallback {cfg['fallback']}", fontsize=9,
                 color="#1f1f1e")
    ax.grid(True, which="both", color="#e6e5df", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors="#3d3d3a", labelsize=8)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_times(path, cfg, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=150)
    for i, arm in enumerate(cfg["arms"]):
        for k, s in enumerate(cfg["seeds"]):
            rr = [r for r in rows if r["arm"] == arm and r["seed"] == s]
            ax.plot([r["time_us"] for r in rr], [r["per_million"] for r in rr], color=COLORS[i], lw=1.0,
                    ls="-" if k == 0 else "--", marker="o" if k == 0 else "s", ms=4, mec="white", mew=0.6,
                    label=f"{arm} seed {s}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    log_axis(ax.xaxis)
    log_axis(ax.yaxis)
    ax.set_xlabel("projected mean time per shot (us): A3 network + realized routed fraction x fallback (its own host)"
                  " + handoff", color="#3d3d3a")
    ax.set_ylabel("pair failures per million test shots", color="#3d3d3a")
    ax.set_title(f"{cfg['code']}: failures against time", fontsize=9, color="#1f1f1e")
    ax.grid(True, which="both", color="#e6e5df", lw=0.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors="#3d3d3a", labelsize=8)
    ax.legend(fontsize=6.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reads", required=True)
    ap.add_argument("--checks", default="", help="JSON list of the runs' not-trained checks")
    ap.add_argument("--timing", default="", help="JSON: network_us {arm: {seed: us}}, handoff_us")
    ap.add_argument("--seeds", default="", help="comma list: these seeds only (the rule's STOP: 1)")
    ap.add_argument("--arms", default="", help="comma list: these arms only; comparisons of others are left out")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    if a.seeds:                                  # the rule's STOP: seed 1 alone (revision/A1_POD_RUN.md)
        cfg["seeds"] = [int(x) for x in a.seeds.split(",")]
    if a.arms:                                   # ... and its stage-1 arms: comparisons of other arms are left out
        keep = a.arms.split(",")
        cfg["arms"] = [x for x in cfg["arms"] if x in keep]
        cfg["comparisons"] = [c for c in cfg["comparisons"] if c["x"] in keep and c["y"] in keep]
    code, arms, seeds, qs, qr = cfg["code"], cfg["arms"], cfg["seeds"], cfg["q"], cfg["q_registered"]
    assert qr in qs and qs[0] == 0, "the registered q must be on the grid, which starts at 0"
    os.makedirs(a.out, exist_ok=True)
    rows_files = sets.manifest()
    cal_det, cal_obs, cal_dig = sets.load(code, cfg["cal"]["kind"], sets.parse_range(cfg["cal"]["chunks"]), rows_files)
    _, obs, dig = sets.load(code, cfg["test"]["kind"], sets.parse_range(cfg["test"]["chunks"]), rows_files)
    del cal_det
    N = len(obs)
    inputs, ckpts = {}, {}

    def read(name, role, digest):
        p = os.path.join(a.reads, f"{name}_{role}.npz")
        if name.startswith("ref_"):
            arm, seed = name[4:], 0
        else:
            arm, _, s_ = name.removeprefix("ramp_").rpartition("_s")
            seed = int(s_)
        r = load_read(p, digest, arm, seed)
        inputs[os.path.basename(p)] = r["sha256"]
        ck = ckpts.setdefault(name, r["meta"]["ckpt_sha256"])     # the cal and test reads of one checkpoint
        if ck != r["meta"]["ckpt_sha256"]:
            raise SystemExit(f"{name}: the cal and test reads are of different checkpoints")
        return r

    fb_path = os.path.join(a.reads, "fallback_test.npz")
    fbz = np.load(fb_path)
    fb_meta = json.loads(str(fbz["meta"]))
    assert fb_meta["set_digest"] == dig, "the fallback answers are on another set"
    inputs[os.path.basename(fb_path)] = sha_file(fb_path)
    fb_fail, fb_secs = fails(fbz["pred"], obs), fbz["secs"].astype(np.float64)
    R = {(arm, s): (read(f"{arm}_s{s}", "cal", cal_dig), read(f"{arm}_s{s}", "test", dig))
         for arm in arms for s in seeds}
    curves, std, fail_at = [], [], {}
    for (arm, s), (rc, rt) in R.items():
        for q in qs:
            t = tau_of(rc["conf"], q)
            f, routed = pair(rt, fb_fail, obs, t)
            fail_at[(arm, s, q)] = f
            curves.append({"arm": arm, "seed": s, "q": q, "tau": t, "routed": int(routed.sum()),
                           "routed_frac": float(routed.mean()), "failures": int(f.sum()), "per_million": 1e6 * f.mean(),
                           "fallback_mean_us_routed": 1e6 * float(fb_secs[routed].mean()) if routed.any() else 0.0})
        std.append({"arm": arm, "seed": s, "failures": int(fail_at[(arm, s, 0)].sum()),
                    "per_million": 1e6 * float(fail_at[(arm, s, 0)].mean()), "test_shots": N,
                    "cal_failures": int(fails(rc["pred"], cal_obs).sum()), "params": rt["meta"]["params"],
                    "readout_macs": rt["meta"]["readout_macs"], "b1_differ": rt["meta"]["b1_check"]["decisions_differ"],
                    "device": rt["meta"]["device"]})
    comps = compare(cfg["comparisons"], fail_at, seeds, qr)
    # reference rows (each on the same test shots), ramp table, sizes, checks
    refs, ref_rows = {}, []
    for ref in cfg.get("references", []):
        try:
            rc = read(f"ref_{ref['name']}", "cal", cal_dig)
            rt = read(f"ref_{ref['name']}", "test", dig)
        except FileNotFoundError:
            if ref.get("required", True):
                raise
            continue
        refs[ref["name"]] = []
        for q in qs:
            f, routed = pair(rt, fb_fail, obs, tau_of(rc["conf"], q))
            refs[ref["name"]].append({"q": q, "per_million": 1e6 * float(f.mean())})
            ref_rows.append({"reference": ref["name"], "label": ref.get("label", ""), "q": q, "failures": int(f.sum()),
                             "per_million": 1e6 * float(f.mean()), "routed": int(routed.sum())})
    ramp = []
    dep = cfg["deployed"]
    for s in seeds:
        if not os.path.exists(os.path.join(a.reads, f"ramp_{dep}_s{s}_test.npz")):
            continue
        rd = (read(f"ramp_{dep}_s{s}", "cal", cal_dig), read(f"ramp_{dep}_s{s}", "test", dig))
        fd = {q: pair(rd[1], fb_fail, obs, tau_of(rd[0]["conf"], q))[0] for q in (0, qr)}
        for arm in arms:
            p_t = os.path.join(a.reads, f"ramp_{arm}_s{s}_test.npz")
            if arm == dep or not os.path.exists(p_t):
                continue
            rr = (read(f"ramp_{arm}_s{s}", "cal", cal_dig), read(f"ramp_{arm}_s{s}", "test", dig))
            for q in (0, qr):
                f = pair(rr[1], fb_fail, obs, tau_of(rr[0]["conf"], q))[0]
                n10, n01, p = mcnemar(f, fd[q])
                dt, lo, hi = newcombe(f, fd[q])
                ramp.append({"seed": s, "arm": arm, "against": f"{dep} (ramp)", "q": q, "fail": int(f.sum()),
                             "fail_deployed": int(fd[q].sum()), "only_arm": n10, "only_deployed": n01, "p": p,
                             "diff_pm": dt, "lo": lo, "hi": hi})
    sizes = []
    for arm in arms:
        m0 = R[(dep, seeds[0])][1]["meta"]
        m1 = R[(arm, seeds[0])][1]["meta"]
        sizes.append({"arm": arm, "params": m1["params"], "readout_macs": m1["readout_macs"],
                      "readout_macs_vs_deployed": m1["readout_macs"] / m0["readout_macs"], "dropped_at_export":
                      len(m1["dropped"]), "pooled_within_10pct": abs(m1["readout_macs"] / m0["readout_macs"] - 1) <= 0.10})
    checks = json.load(open(a.checks)) if a.checks else []
    times = []
    if a.timing:
        tm = json.load(open(a.timing))
        for r in curves:
            t_net = float(tm["network_us"][r["arm"]][str(r["seed"])])
            r_t = dict(r, time_us=t_net + (r["routed_frac"] * r["fallback_mean_us_routed"] + float(tm["handoff_us"])
                                           if r["q"] > 0 else 0.0), network_us=t_net,
                       time_basis=f"projection: A3 network time + the realized routed fraction x the fallback's mean "
                                  f"time on the routed shots measured on {fb_meta.get('host', 'the fallback host')} "
                                  f"(not A3's host) + A3's handoff")
            times.append(r_t)
        inputs[os.path.basename(a.timing)] = sha_file(a.timing)
    # outputs
    write_csv(os.path.join(a.out, "standalone.csv"), std)
    write_csv(os.path.join(a.out, "pair_curves.csv"), curves)
    write_csv(os.path.join(a.out, "comparisons.csv"), comps)
    write_csv(os.path.join(a.out, "references.csv"), ref_rows)
    write_csv(os.path.join(a.out, "ramp.csv"), ramp)
    write_csv(os.path.join(a.out, "train_checks.csv"), checks)
    write_csv(os.path.join(a.out, "sizes.csv"), sizes)
    plot_pairs(os.path.join(a.out, "pair_curves.png"), cfg, curves, refs, N)
    if times:
        write_csv(os.path.join(a.out, "time_curves.csv"), times)
        plot_times(os.path.join(a.out, "time_curves.png"), cfg, times)
    seed_keys = [k for s in seeds for k in (f"s{s}_diff_pm", f"s{s}_lo", f"s{s}_hi", f"s{s}_p")]
    md = [f"# A1 analysis: {code} ({cfg.get('label', '')})", "",
          f"Test set {cfg['test']['kind']} chunks {cfg['test']['chunks']} ({N:,} shots), cal set {cfg['cal']['kind']} "
          f"chunks {cfg['cal']['chunks']}; fallback {cfg['fallback']} ({fb_meta['decoder']}); registered q {qr}; "
          f"Holm over {len(comps)} tests (both q); alpha {ALPHA}.", "", "## Standalone failures (q = 0)", "",
          md_table(std, ["arm", "seed", "failures", "per_million", "cal_failures", "params", "readout_macs",
                         "b1_differ"]), "", "## Comparisons (X against Y, X - Y per million)", "",
          md_table(comps, ["id", "factor", "x", "y", "q", *seed_keys, "p_iut", "p_holm", "mean_diff_pm", "mean_lo",
                           "mean_hi", "verdict"]), "", "## Pair curves", "",
          md_table(curves, ["arm", "seed", "q", "tau", "routed", "failures", "per_million"]), ""]
    for title, rows in (("Reference rows", ref_rows), ("Ramp table", ramp), ("Sizes", sizes), ("Train checks", checks),
                        ("Time curves", times)):
        if rows:
            md += [f"## {title}", "", md_table(rows, list(rows[0])), ""]
    open(os.path.join(a.out, "summary.md"), "w").write("\n".join(md))
    res = {"config": cfg, "config_sha256": sha_file(a.config), "inputs_sha256": inputs, "test_digest": dig,
           "cal_digest": cal_dig, "fallback": fb_meta, "standalone": std, "comparisons": comps, "pair_curves": curves,
           "references": ref_rows, "ramp": ramp, "sizes": sizes, "train_checks": checks, "time_curves": times}
    json.dump(res, open(os.path.join(a.out, "analysis.json"), "w"), indent=1, default=float)
    for r in comps:
        print(f"{r['id']} {r['x']} vs {r['y']} q={r['q']}: mean {r['mean_diff_pm']:+.1f} pm "
              f"[{r['mean_lo']:+.1f}, {r['mean_hi']:+.1f}] p_iut {r['p_iut']:.3g} p_holm {r['p_holm']:.3g} {r['verdict']}",
              flush=True)
    print("ANALYSIS DONE", a.out, flush=True)


if __name__ == "__main__":
    main()
