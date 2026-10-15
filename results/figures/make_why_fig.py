"""The figure of Section VI (why the pairs win), drawn at its printed size as a vector PDF.

    python results/figures/make_why_fig.py [DATA_DIR]   (writes why_pairs_win.pdf and why_class_gap.pdf beside it; about 5 s)

DATA_DIR (default results/discordant/data) holds the inputs of the exploratory analysis of the discordant shots
(results/discordant/, X2 in its notes), derived from the per-shot records of the test reads:

  surface/fig_d13.npz  sha256 df3846aff25c6dc100eef9c182cd78ae7bd143d2a2c26ca22d7b6f42ce894d95
  surface/fig_d17.npz  sha256 fe37940a8c02226abaeae3ec8e787efd2d4fd6ae9229fd8bbd7be7615ff68387
  bb/fig.npz           sha256 7fbe367b312155d0105828b80bbafb076113f4b03e2d13128a0be484078e2ec3 (the first Cascade read)
The BB panel of why_pairs_win.pdf reads RETEST's re-read in front of our Cascade, as the paper does:
  retest/run/d1/x2/bb/fig.npz  sha256 287a96dce98d5e808bab8eab9de027481a07295e7b00735f456c08fc5d2174ca

The panels redraw two figures of results/discordant/figures.py in the paper's style. why_pairs_win.pdf (Section VII) shows
the share of the discordant shots that the network fails, against its confidence (x2_who_fails), with the same bins
and intervals; that share is conditional on the shot being discordant. why_class_gap.pdf (Appendix H) shows the
class-gap histograms at d = 13 (x2_gap, panels a and c).
Colours follow make_figs.py: the network vermillion, the strong decoder (the fallback) blue.
"""
import os
import sys

import numpy as np
from matplotlib.lines import Line2D

import make_figs as mf  # the paper's rcParams, colours and helpers

plt = mf.plt
BF, ALL = "#4D4D4D", "#9A9A9A"            # both fail, all shots
CELLS = ((1, "only CoDecoder fails", mf.CL), (2, "only correlated PM fails", mf.FB), (3, "both fail", BF))


def wilson(k, n, z=1.96):
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, mid - half, mid + half


def share_curve(a, b, tau, below=8, above=4):
    """Equal-count bins of the pooled confidence percentiles of a (only the network fails) and b (only the strong
    decoder fails) on each side of the threshold tau: bin median and the share of a, with its Wilson 95 % interval."""
    pool = np.sort(np.concatenate([a, b]))
    lo_side, hi_side = pool[pool < tau], pool[pool >= tau]
    edges = np.concatenate([np.quantile(lo_side, np.linspace(0, 1, below + 1))[:-1], [tau],
                            np.quantile(hi_side, np.linspace(0, 1, above + 1))[1:]])
    edges[-1] += 1e-9
    x, k, n = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        x.append(np.median(pool[(pool >= lo) & (pool < hi)]))
        k.append(int(((a >= lo) & (a < hi)).sum()))
        n.append(k[-1] + int(((b >= lo) & (b < hi)).sum()))
    p, lo, hi = wilson(k, n)
    return np.array(x), 100 * p, 100 * lo, 100 * hi


def gap_figure(s13):
    """why_class_gap.pdf (Appendix H): matching's class gap at d = 13, plain (a) and correlated (b)."""
    fig, axs = plt.subplots(1, 2, figsize=(mf.COL, 2.25))
    fig.subplots_adjust(left=0.14, right=0.985, top=0.735, bottom=0.18, wspace=0.6)
    bins = np.arange(-25, 50.5, 1.0)
    cell, ref = s13["cell"], s13["ref_sel"]
    for ax, key, title in ((axs[0], "gap_sel", "(a) plain matching"),
                           (axs[1], "cgap_sel", "(b) correlated matching")):
        g = s13[key]
        for k, _, col in reversed(CELLS):           # the network's lone failures on top
            x = g[cell == k]
            ax.hist(x, bins=bins, weights=np.full(len(x), 1.0 / len(x)), histtype="step", color=col, lw=0.9)
            print(f"{key} cell {k}: {100 * ((x < bins[0]) | (x > bins[-1])).mean():.2f} % outside the axis")
        x = g[ref]
        ax.hist(x, bins=bins, weights=np.full(len(x), 1.0 / len(x)), histtype="step", color=ALL, lw=0.8,
                ls=(0, (2.5, 1.5)))
        ax.axvline(0, color="black", lw=0.6)
        ax.set_xlim(bins[0], bins[-1])
        ax.set_xticks([-20, 0, 20, 40])
        ax.yaxis.set_major_locator(mf.ticker.MaxNLocator(4))
        ax.set_title(title, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("class gap (nats)", labelpad=1)
        ax.text(0.97, 0.96, "$d$ = 13", transform=ax.transAxes, ha="right", va="top", fontsize=8)
        ax.set_ylabel("share of shots per nat", labelpad=2)
        mf.grid(ax)
    fig.legend(handles=[Line2D([], [], color=col, lw=0.9, label=lab) for _, lab, col in CELLS]
               + [Line2D([], [], color=ALL, lw=0.8, ls=(0, (2.5, 1.5)), label="all shots")],
               loc="upper center", bbox_to_anchor=(0.55, 1.0), ncol=2, frameon=False, handlelength=1.8,
               columnspacing=1.6, labelspacing=0.25, borderaxespad=0.1)
    out = os.path.join(mf.OUT, "why_class_gap.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def share_figure(s13, s17, bb):
    """why_pairs_win.pdf (Section VII): the network's share of the discordant shots against its confidence, in front
    of correlated PM on the surface code (a) and of our Cascade on BB72 (b)."""
    fig, axs = plt.subplots(1, 2, figsize=(mf.COL, 1.95), sharey=True)
    fig.subplots_adjust(left=0.14, right=0.96, top=0.885, bottom=0.215, wspace=0.2)
    panels = ((axs[0], [(s13["pct_nf"], s13["pct_mf"], float(s13["tau_pct"]), 4, "-", "$d$ = 13"),
                        (s17["pct_nf"], s17["pct_mf"], float(s17["tau_pct"]), 4, "--", "$d$ = 17")],
               "(a) surface code"),
              (axs[1], [(bb["p2_pS_Sonly"], bb["p2_pS_Conly"], float(bb["p2_tau_pct"]), 3, "-", "$p$ = 0.2 %"),
                        (bb["p3_pS_Sonly"], bb["p3_pS_Conly"], float(bb["p3_tau_pct"]), 3, "--", "$p$ = 0.3 %")],
               "(b) BB72"))
    xmin = []
    for ax, rows, title in panels:
        hs = []
        for a, b, tau, above, ls, lab in rows:
            x, p, lo, hi = share_curve(a, b, tau, above=above)
            xmin.append(x.min())
            print(title, lab, "x from", f"{x.min():.4g}", "to", f"{x.max():.4g}", "tau", f"{tau:.4g}")
            ax.fill_between(x, lo, hi, color=mf.CL, alpha=0.15, lw=0)
            ax.plot(x, p, color=mf.CL, ls=ls, marker="o", ms=2.6, mfc="white", mew=0.7, lw=0.9)
            ax.axvline(tau, color="black", lw=0.6, ls=ls)
            hs.append(Line2D([], [], color=mf.CL, ls=ls, marker="o", ms=2.6, mfc="white", mew=0.7, label=lab))
        ax.axhline(50, color=ALL, lw=0.6)
        ax.set_title(title, fontsize=8, pad=3, loc="left")
        ax.set_xlabel("confidence percentile (%)", labelpad=1)
        ax.legend(handles=hs, loc="lower left", frameon=False, handlelength=2.0, borderaxespad=0.2,
                  labelspacing=0.25)
        mf.grid(ax)
    lo = 1e-2 if min(xmin) >= 1e-2 else 5e-3      # one x range for both panels, the lowest bin (0.008 %) inside
    for ax in axs:
        ax.set_xscale("log")
        ax.set_xlim(lo, 100)
        mf.plain_log(ax, "x", [1e-2, 1, 100])
        ax.xaxis.set_minor_locator(mf.ticker.LogLocator(base=10, subs=(1.0,), numticks=12))
    axs[0].set_ylim(0, 100)
    axs[0].set_ylabel("CoDecoder's share of\ndiscordant shots (%)", labelpad=2, linespacing=1.0)
    out = os.path.join(mf.OUT, "why_pairs_win.pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main(data):
    s13 = np.load(os.path.join(data, "surface", "fig_d13.npz"))
    s17 = np.load(os.path.join(data, "surface", "fig_d17.npz"))
    bb = np.load(os.path.join(os.path.dirname(os.path.dirname(mf.OUT)), "retest", "run", "d1", "x2", "bb", "fig.npz"))
    share_figure(s13, s17, bb)
    gap_figure(s13)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(os.path.dirname(mf.OUT), "discordant", "data"))
