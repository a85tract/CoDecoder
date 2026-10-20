"""Stage 3 figures: Figs. 3 and 4 with the batched (throughput-mode) time axis.
    python figs3.py speed_rt/s3/results/numbers.json OUTDIR [CARD]   (matplotlib)
Failure rates are the registered tests' (the paper's figures/make_figs.py: SURF, NETALONE, BB); times are stage 3's
cost per round on the Grace basis (CPU terms x 1.5098; GPU terms unconverted). N = 8 filled, N = 1 hollow."""
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

TEXT = 7.16
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False})
CL, FB, PMC, GRID = "#D55E00", "#0072B2", "#8C8C8C", "#E6E6E6"
MODE = {"fast": "o", "balanced": "D", "accurate": "s"}

D = [7, 9, 11, 13, 15, 17]
NS = [2_000_000] + [1_000_000] * 5
OURS = [31629, 11978, 6887, 5318, 3012, 2180]
CPM = [35135, 12802, 9252, 6344, 4396, 2973]
PM = [60016, 27028, 24365, 21086, 18259, 15726]
NETALONE = [36156, 18305, 6887, 5318, 4030, 3296]
SYLIM = [(1.43, 3.19), (0.83, 3.1), (0.49, 2.75), (0.386, 2.46), (0.215, 2.15), (0.142, 1.85)]  # the paper's
BYLIM = [(0.3, 2.0), (0.04, 0.5), (0.12, 3.5)]                                                          # the paper's
BB = [("BB72, $p$ = 0.3 %", "bb72-p3", 200000, dict(alone=1398, balanced=1554, accurate=857, sector=2769, full=1107)),
      ("BB72, $p$ = 0.2 %", "bb72-p2", 300000, dict(alone=269, balanced=297, accurate=204, sector=767, full=288)),
      ("gross, $p$ = 0.4 %", "gross-p4", 150000, dict(alone=1024, balanced=407, accurate=234, sector=733, full=472))]


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def pt(ax, x, f, n, col, mk, filled, size=4.5, z=3):
    if x is None:
        return
    r = 100 * f / n
    lo, hi = wilson(f, n)
    ax.errorbar([x], [r], yerr=[[r - 100 * lo], [100 * hi - r]], fmt=mk, color=col, ms=size,
                mfc=col if filled else "white", mew=0.8, capsize=0, elinewidth=0.6, zorder=z)


def style(ax, xlabel=None, ylim=None):
    xs = [x for ln in ax.lines for x in ln.get_xdata()] + [1.0]
    ax.set_xlim(min(xs) / 1.8, max(xs) * 1.8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axvline(1.0, color="black", lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.grid(True, which="major", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(ticker.NullFormatter())
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(
        lambda v, _: f"{v:g}" if round(v / 10 ** math.floor(math.log10(v) + 1e-9), 3) in (2, 3, 5) else ""))
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=1)


def main():
    num = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    card = sys.argv[3] if len(sys.argv) > 3 else "h100"
    comp, base, alone = num["compositions"], num["baselines"], num["alone"]

    def cr(name, n):
        r = comp.get(f"{card}.{name}.N{n}")
        return None if r is None else r["grace"]["us_round"]

    def br(name, n):
        r = base.get(f"{name}.N{n}")
        return None if r is None else r["grace_us_round"]

    def ar(p):
        r = alone.get(f"{card}.{p}")
        return None if r is None else r["us_round"]

    fig, axs = plt.subplots(1, 6, figsize=(TEXT, 2.2))
    for i, (ax, d) in enumerate(zip(axs, D)):
        p = f"surf-d{d}"
        for n, filled in ((1, False), (8, True)):
            pt(ax, cr(p, n), OURS[i], NS[i], CL, MODE["balanced"], filled, 5.0, 5)
            pt(ax, br(f"{p}.cpm", n), CPM[i], NS[i], FB, "s", filled)
            pt(ax, br(f"{p}.pm", n), PM[i], NS[i], PMC, "o", filled)
        pt(ax, ar(p), NETALONE[i], NS[i], CL, "o", False, 5.5, 4)
        style(ax, ylim=SYLIM[i])
        ax.text(0.97, 0.04, f"$d$ = {d}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
    axs[0].set_ylabel("logical error rate (%)", labelpad=2)
    fig.text(0.5, 0.01, "cost per round (µs), batched; CPU terms on the Grace basis; dashed: 1 µs", ha="center",
             va="bottom", fontsize=8)
    hs = [Line2D([], [], color=CL, marker=MODE["balanced"], ls="none", mfc=CL, ms=5),
          Line2D([], [], color=FB, marker="s", ls="none", mfc=FB, ms=4),
          Line2D([], [], color=PMC, marker="o", ls="none", mfc=PMC, ms=4),
          Line2D([], [], color=CL, marker="o", ls="none", mfc="white", ms=5.5),
          Line2D([], [], color="black", marker="s", ls="none", mfc="black", ms=4),
          Line2D([], [], color="black", marker="s", ls="none", mfc="white", ms=4)]
    fig.legend(hs, ["CoDecoder + correlated PM", "correlated PM", "PM", "CoDecoder alone", "N = 8 CPU processes",
                    "N = 1"], loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.2, top=0.86, wspace=0.35)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig3_batched.{ext}"), dpi=200)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(TEXT, 2.6))
    for i, (ax, (title, p, n_test, f)) in enumerate(zip(axs, BB)):
        for n, filled in ((1, False), (8, True)):
            pt(ax, cr(f"{p}.balanced", n), f["balanced"], n_test, CL, MODE["balanced"], filled, 5.0, 5)
            pt(ax, cr(f"{p}.accurate", n), f["accurate"], n_test, CL, MODE["accurate"], filled, 5.0, 5)
            pt(ax, br(f"{p}.sector", n), f["sector"], n_test, FB, "^", filled)
            pt(ax, br(f"{p}.full", n), f["full"], n_test, FB, "v", filled)
        pt(ax, ar(p), f["alone"], n_test, CL, MODE["fast"], False, 5.0, 4)
        style(ax, "cost per round (µs), batched", BYLIM[i])
        ax.set_title(title, fontsize=8, pad=3)
    axs[0].set_ylabel("block failure rate (%)", labelpad=2)
    hs = [Line2D([], [], color=CL, marker=MODE["fast"], ls="none", mfc="white", ms=5),
          Line2D([], [], color=CL, marker=MODE["balanced"], ls="none", mfc=CL, ms=5),
          Line2D([], [], color=CL, marker=MODE["accurate"], ls="none", mfc=CL, ms=5),
          Line2D([], [], color=FB, marker="^", ls="none", mfc=FB, ms=4),
          Line2D([], [], color=FB, marker="v", ls="none", mfc=FB, ms=4)]
    fig.legend(hs, ["CoDecoder", "CoDecoder + sector Relay-BP", "CoDecoder + full-DEM Relay-BP", "sector Relay-BP",
                    "full-DEM Relay-BP"], loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.99, 0.01, "N = 8 filled, N = 1 hollow; dashed: 1 µs", ha="right", va="bottom", fontsize=7.5)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.17, top=0.78, wspace=0.3)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"fig4_batched.{ext}"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
