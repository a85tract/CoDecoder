"""X2 figures (drafts at journal sizes): $X2_OUT/fig/*.pdf and *.png from the joins' outputs.

  python figures.py

Colour follows the entity in every figure: blue = shots only the network (or S55'-deep) fails, orange = shots only
the strong decoder (correlated PM, or C14') fails, aqua = both fail, grey = all shots. Line style separates d = 13
(solid) from d = 17 (dashed), and p = 0.2 % (solid) from 0.3 % (dashed). The three colours are chosen to stay distinct
under colour-vision deficiency; identity is also carried by legends and line style."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = os.environ.get("X2_OUT", "out")
FIG = os.path.join(OUT, "fig")
BLUE, ORANGE, AQUA, GREY, INK, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#0b0b0b", "#e1e0d9"
CELL = {1: ("only the network fails", BLUE), 2: ("only correlated PM fails", ORANGE), 3: ("both fail", AQUA)}
W2 = 7.0                                   # double-column width, inches
LS = {13: "-", 17: "--", 2: "-", 3: "--"}

plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5, "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 1.4, "axes.edgecolor": GREY,
    "axes.labelcolor": INK, "xtick.color": GREY, "ytick.color": GREY, "text.color": INK, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 300, "pdf.fonttype": 42, "font.family": "DejaVu Sans"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def surf(d):
    return np.load(os.path.join(OUT, "surface", f"fig_d{d}.npz"))


def ecdf(ax, x, color, ls="-", lw=1.4):
    x = np.sort(np.asarray(x, float))
    ax.step(x, 100.0 * np.arange(1, len(x) + 1) / len(x), where="post", color=color, ls=ls, lw=lw)


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def share_curve(a, b, tau, below=8, above=4):
    """Bins of the pooled percentiles of a (only the first decoder fails) and b (only the second fails), equal-count
    on each side of the routing threshold tau: bin median, share of a with its Wilson 95 % interval."""
    pool = np.sort(np.concatenate([a, b]))
    lo_side, hi_side = pool[pool < tau], pool[pool >= tau]
    edges = np.concatenate([np.quantile(lo_side, np.linspace(0, 1, below + 1))[:-1], [tau],
                            np.quantile(hi_side, np.linspace(0, 1, above + 1))[1:]])
    edges[-1] += 1e-9
    x, k, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        ka, kb = int(((a >= lo) & (a < hi)).sum()), int(((b >= lo) & (b < hi)).sum())
        x.append(np.median(pool[(pool >= lo) & (pool < hi)]))
        k.append(ka)
        n.append(ka + kb)
    p, lo, hi = wilson(k, n)
    return np.array(x), 100 * p, 100 * lo, 100 * hi


def legend_below(fig, handles, ncol, y=-0.02):
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol, handlelength=2.2,
               columnspacing=1.6)


def cell_handles(extra=()):
    h = [Line2D([], [], color=c, label=n) for n, c in CELL.values()]
    return h + list(extra)


def mark_tau(ax, tp, label=True, ls="-", y=22):
    ax.axvline(tp, color=INK, lw=0.8, ls=ls)
    if label:
        ax.annotate(f"routed\n{tp:.2f} %", (tp, y), xytext=(4, 0), textcoords="offset points", ha="left",
                    va="center", fontsize=7)


def fig_confidence(S):
    """ECDF of the network's |c| percentile per cell; the routing threshold marked."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5), sharey=True)
    for ax, d, tag in zip(axes, (13, 17), "ab"):
        z, s = surf(d), S[str(d)]
        for key, k in (("pct_nf", 1), ("pct_mf", 2), ("pct_bf", 3)):
            ecdf(ax, z[key], CELL[k][1])
        xs = np.logspace(-3, 2, 200)
        ax.plot(xs, xs, color=GREY, ls="--", lw=1.0)
        mark_tau(ax, float(z["tau_pct"]), y=30 if d == 13 else 50)
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("network confidence |c|, percentile of test shots (%)")
        c = s["cells"]
        ax.set_title(f"({tag}) d = {d}, {s['test_set']}: {c['only the network fails']:,} / "
                     f"{c['only correlated PM fails']:,} / {c['both fail']:,} shots", loc="left")
    axes[0].set_ylabel("share of the cell at or below (%)")
    legend_below(fig, cell_handles([Line2D([], [], color=GREY, ls="--", lw=1.0, label="all shots")]), 4)
    save(fig, "x2_confidence")


def fig_who(S, B):
    """Among the shots exactly one decoder fails: the share that the front end fails, against its confidence."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6), sharey=True)
    ax = axes[0]
    for d in (13, 17):
        z = surf(d)
        x, p, lo, hi = share_curve(z["pct_nf"], z["pct_mf"], float(z["tau_pct"]))
        ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0)
        ax.plot(x, p, color=BLUE, ls=LS[d], marker="o", ms=3.2, mfc="white", mew=0.9)
        ax.axvline(float(z["tau_pct"]), color=INK, lw=0.8, ls=LS[d])
    ax.set_xlabel("network confidence |c|, percentile of test shots (%)")
    ax.set_title("(a) surface code: the network or correlated PM", loc="left")
    ax.legend(handles=[Line2D([], [], color=BLUE, ls="-", marker="o", ms=3.2, mfc="white", label="d = 13"),
                       Line2D([], [], color=BLUE, ls="--", marker="o", ms=3.2, mfc="white", label="d = 17"),
                       Line2D([], [], color=INK, lw=0.8, label="routing threshold")], loc="lower left")
    ax = axes[1]
    if B is not None:
        zb = np.load(os.path.join(OUT, "bb", "fig.npz"))
        for p in (2, 3):
            x, s, lo, hi = share_curve(zb[f"p{p}_pS_Sonly"], zb[f"p{p}_pS_Conly"], float(zb[f"p{p}_tau_pct"]),
                                       above=3)
            ax.fill_between(x, lo, hi, color=BLUE, alpha=0.15, lw=0)
            ax.plot(x, s, color=BLUE, ls=LS[p], marker="o", ms=3.2, mfc="white", mew=0.9)
            ax.axvline(float(zb[f"p{p}_tau_pct"]), color=INK, lw=0.8, ls=LS[p])
        ax.legend(handles=[Line2D([], [], color=BLUE, ls="-", marker="o", ms=3.2, mfc="white", label="p = 0.2 %"),
                           Line2D([], [], color=BLUE, ls="--", marker="o", ms=3.2, mfc="white", label="p = 0.3 %")],
                  loc="lower left")
    ax.set_xlabel("S55'-deep margin, percentile of test shots (%)")
    ax.set_title("(b) BB72 (D1): S55'-deep or C14'", loc="left")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_ylim(0, 100)
        ax.axhline(50, color=GREY, lw=0.8)
    axes[0].set_xlim(1e-2, 100)
    axes[1].set_xlim(1e-3, 100)
    axes[0].set_ylabel("only the front end fails\n(% of shots exactly one decoder fails)")
    save(fig, "x2_who_fails")


def fig_gap(S):
    """Histograms of matching's class gap per cell: plain (exact), correlated (approximate)."""
    fig, axes = plt.subplots(2, 2, figsize=(W2, 4.4), sharex=True)
    bins = np.arange(-25, 50.5, 1.0)
    for col, d in enumerate((13, 17)):
        z = surf(d)
        cell, ref = z["cell"], z["ref_sel"]
        for row, key, what in ((0, "gap_sel", "plain matching (exact)"),
                               (1, "cgap_sel", "correlated matching (approximate)")):
            ax = axes[row, col]
            g = z[key]
            for k, m in ((1, cell == 1), (2, cell == 2), (3, cell == 3), (0, ref)):
                x = g[m]
                out = float(((x < bins[0]) | (x > bins[-1])).mean())
                print(f"gap d = {d} {key} group {k}: {100 * out:.2f} % outside [{bins[0]}, {bins[-1]}] nats")
                w = np.full(len(x), 1.0 / len(x))                 # density of the whole group, tails included
                ax.hist(x, bins=bins, weights=w, histtype="step", color=CELL[k][1] if k else GREY,
                        ls="-" if k else "--", lw=1.3 if k else 1.0)
            ax.axvline(0, color=INK, lw=0.8)
            ax.set_title(f"({'abcd'[2 * row + col]}) d = {d}, {what}", loc="left")
            if col == 0:
                ax.set_ylabel("share of the group per nat")
            if row == 1:
                ax.set_xlabel("class gap W(wrong class) − W(true class) (nats)")
            ax.set_xlim(bins[0], bins[-1])
    legend_below(fig, cell_handles([Line2D([], [], color=GREY, ls="--", lw=1.0, label="all shots (every 32nd)")]),
                 4, y=0.0)
    save(fig, "x2_gap")


def fig_capture(S):
    """(a) the share of each decoder's failures among the routed shots; (b) the pair's failures minus correlated PM's,
    per million shots, against the routed fraction (descriptive)."""
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.6))
    fig.subplots_adjust(wspace=0.42)
    for d in (13, 17):
        z, s = surf(d), S[str(d)]
        n, tot_net, tot_cpm, _ = z["totals"]
        f = 100.0 * z["curve_k"] / n
        axes[0].plot(f, 100 * z["curve_net_cap"], color=BLUE, ls=LS[d])
        axes[0].plot(f, 100 * z["curve_cpm_cap"], color=ORANGE, ls=LS[d])
        axes[1].plot(f, 1e6 * (z["curve_pair"] - tot_cpm) / n, color=INK, ls=LS[d], lw=1.2)
        tp = float(z["tau_pct"])
        reg = 1e6 * (s["pair_failures"] - s["correlated_pm_failures"]) / n
        axes[1].plot([tp], [reg], "o", ms=4.5, mfc="white", mec=INK, mew=1.0)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlim(1e-2, 40)
        ax.set_xlabel("routed fraction: least confident test shots (%)")
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("failures among the routed shots\n(% of the decoder's failures)")
    axes[0].set_title("(a)", loc="left")
    axes[1].axhline(0, color=GREY, lw=0.8)
    axes[1].set_ylabel("pair − correlated PM failures\n(per million shots)")
    axes[1].set_title("(b)", loc="left")
    axes[1].set_ylim(-1500, 6000)
    h = [Line2D([], [], color=BLUE, label="the network's failures"),
         Line2D([], [], color=ORANGE, label="correlated PM's failures"),
         Line2D([], [], color=INK, ls="-", label="d = 13"), Line2D([], [], color=INK, ls="--", label="d = 17"),
         Line2D([], [], color=INK, ls="none", marker="o", mfc="white", label="registered threshold")]
    legend_below(fig, h, 5)
    save(fig, "x2_capture")


def fig_bb(B):
    """BB72 (D1): ECDF of S55'-deep's margin percentile per cell (top); kept discordant shots in the plane of S's
    margin and C14''s confidence percentiles (bottom)."""
    z = np.load(os.path.join(OUT, "bb", "fig.npz"))
    fig, axes = plt.subplots(2, 2, figsize=(W2, 5.0))
    for col, p in enumerate((2, 3)):
        pre = f"p{p}_"
        s = B[f"0.00{p}"]
        ax = axes[0, col]
        for key, k in (("pS_Sonly", 1), ("pS_Conly", 2), ("pS_both", 3)):
            ecdf(ax, z[pre + key], CELL[k][1])
        xs = np.logspace(-4, 2, 200)
        ax.plot(xs, xs, color=GREY, ls="--", lw=1.0)
        mark_tau(ax, float(z[pre + "tau_pct"]), y=30 if p == 2 else 55)
        ax.set_xscale("log")
        ax.set_xlim(1e-3, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel("S55'-deep margin, percentile of test shots (%)")
        ax.set_title(f"({'ab'[col]}) p = 0.{p} %, {s['shots']:,} shots", loc="left")
        if col == 0:
            ax.set_ylabel("share of the cell at or below (%)")
        ax = axes[1, col]
        for key_s, key_c, colr, mk in (("pS_breaks", "pC_breaks", BLUE, "o"), ("pS_repairs", "pC_repairs", ORANGE, "s")):
            ax.scatter(z[pre + key_s], z[pre + key_c], s=11, marker=mk, facecolors="none", edgecolors=colr, lw=0.8)
        ax.axvline(float(z[pre + "tau_pct"]), color=INK, lw=0.8)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 100)
        ax.set_ylim(1e-3, 100)
        ax.set_xlabel("S55'-deep margin, percentile (%)")
        ax.set_title(f"({'cd'[col]}) kept shots exactly one fails, p = 0.{p} %: "
                     f"{s['cells']['S only fails, kept (breaks)']} / {s['cells']['C only fails, kept (repairs)']}",
                     loc="left")
        if col == 0:
            ax.set_ylabel("C14' confidence (min |logit|),\npercentile (%)")
    h = [Line2D([], [], color=BLUE, label="only S55'-deep fails"), Line2D([], [], color=ORANGE, label="only C14' fails"),
         Line2D([], [], color=AQUA, label="both fail"), Line2D([], [], color=GREY, ls="--", lw=1.0, label="all shots")]
    legend_below(fig, h, 4, y=0.0)
    fig.subplots_adjust(hspace=0.45)
    save(fig, "x2_bb72")


def main():
    S = json.load(open(os.path.join(OUT, "surface", "summary.json")))
    bpath = os.path.join(OUT, "bb", "summary.json")
    B = json.load(open(bpath)) if os.path.exists(bpath) else None
    fig_confidence(S)
    fig_who(S, B)
    fig_gap(S)
    fig_capture(S)
    if B is not None:
        fig_bb(B)
    print("figures in", FIG, sorted(os.listdir(FIG)))


if __name__ == "__main__":
    main()
