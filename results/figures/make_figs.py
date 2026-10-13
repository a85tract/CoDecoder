"""Data figures of the ConvLines paper, drawn at their printed size as vector PDFs.

    python results/figures/make_figs.py    (writes the four PDFs below next to this file; about 5 s)

Figs. 1 and 2 are TikZ sources (overview.tex, output_head.tex) that main.tex inputs. This script draws:

  failures_by_confidence.pdf  where the network's failures sit, routed and retained shots (one column, stacked)
  surface_results.pdf         surface code, failure rate against composite time at each distance (full width)
  bb_results.pdf              BB72 and the gross code, block failure rate against composite time (full width)
  routing_frontiers.pdf       routing frontiers on calibration shots (one column, 2 x 2)

make_why_fig.py draws why_pairs_win.pdf and why_class_gap.pdf from the inputs of the analysis of the
discordant shots.

One visual language: ConvLines vermillion with one marker per mode. Every fallback is blue wherever it appears,
alone or in a pair: correlated PM on the surface code, sector and full-DEM Relay-BP on BB codes (and Cascade in
make_why_fig.py). The other baselines are muted: PM grey, Ising purple, BP-OSD green. Text is serif STIX of at least
7.5 pt at the printed size, and labels placed away from their markers get a thin leader line.
Every number is copied from a record; the comment beside each block names the file. R is the earlier release, whose
bb72/, gross/, gross_stage4/ and results/ are bb/bb72/, bb/gross/, bb/gross_stage4/ and results/surface/ here; H is the
research repository, whose files used here are under results/surface/development/, results/surface/frontiers/,
results/r17/ and bb/bb72/ising_compare/data/bb/stage3/.
"""
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
COL, TEXT = 3.5, 7.16                     # IEEE column and text widths, inches
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False,
})

CL, FB = "#D55E00", "#0072B2"             # ConvLines, every fallback (correlated PM, both Relay-BP, Cascade)
ISP, BPG = "#CC79A7", "#009E73"           # Ising, BP-OSD
PMC, GRID = "#8C8C8C", "#E6E6E6"          # PM, grid
MODE = {"fast": "o", "balanced": "D", "accurate": "s"}   # ConvLines' mode markers in every figure
LEAD = dict(arrowstyle="-", lw=0.45, shrinkA=1.0, shrinkB=2.6)   # leader line from a label to its marker


def label(ax, text, xy, off, ha, va, col, lead=False, fs=7.5):
    """Label the marker at xy, offset by off points; a leader line joins labels set away from their marker."""
    ax.annotate(text, xy, xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=fs, color=col,
                linespacing=1.0, arrowprops=dict(LEAD, color=col) if lead else None, zorder=6)


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return mid - half, mid + half


def grid(ax, axis="both"):
    ax.grid(True, which="major", axis=axis, color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def plain_log(ax, axis, ticks):
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(ticker.FixedLocator(ticks))
    a.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(ticker.NullFormatter())


def panel(ax, letter, x=-0.02, y=1.0, ha="right"):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha=ha, va="bottom", fontsize=8, fontweight="bold")


# ---------------------------------------------------------------------------------------------------------------
# failures_by_confidence.pdf: where the network fails. Exploratory data read before any test.
# BB72 (network 2, 200,000 calibration shots each): R bb72/ising_compare/data/bb/fullroute/bb72_cal.json; rows are
# (q %, network failures among the q least confident shots, full-DEM Relay-BP failures on the same shots); totals:
# network 1,396 / 146, full-DEM 1,117 / 169, sector Relay-BP 2,734 / 523 at p = 0.3 / 0.2 %. Sector Relay-BP at
# q = 2 %, p = 0.3 %: hybrid 1,560 (R bb72/ising_compare/data/bb/policies/bb72_p0.003_rep2.json, frontier), so it
# fails 1,560 - (1,396 - 1,206) = 1,370 of the 4,000 routed shots.
# Gross (100,000 calibration shots): R gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md lines 15-19, 49-50, 64-71:
# network 717, full-DEM exact on the least confident 3 % (173 failures), 4 of 4,100 sampled shots elsewhere, so on the
# 98,000 shots kept at q = 2 %: (173 - 146) + 97,000 * 4/4,100 = 121.6 (Clopper-Pearson 95 %: 52.8-269.1).
# Surface (fresh development shots): H master:ising_compare/interp/data/a2_summary.json, REPORT.md lines 158-168.
CONC = [  # label, rows (q %, share of the network's failures %), marker, label position (x, y, ha)
    ("BB72, $p$ = 0.2 %", [(0.25, 100 * 114 / 146), (0.5, 100 * 130 / 146), (1, 100 * 138 / 146),
                           (2, 100 * 142 / 146), (3, 100 * 143 / 146), (5, 100 * 145 / 146)], "s", (0.23, 78, "right")),
    ("gross, $p$ = 0.4 %", [(0.25, 100 * 222 / 717), (0.5, 100 * 420 / 717), (1, 100 * 656 / 717),
                            (1.5, 100 * 695 / 717), (2, 100 * 707 / 717), (3, 100 * 711 / 717)], "^", (0.23, 35, "right")),
    ("BB72, $p$ = 0.3 %", [(0.25, 100 * 335 / 1396), (0.5, 100 * 612 / 1396), (1, 100 * 928 / 1396),
                           (2, 100 * 1206 / 1396), (3, 100 * 1290 / 1396), (5, 100 * 1348 / 1396)], "o", (0.23, 20, "right")),
]
CONC_DEV = [(0.96, 100 * 189 / 590, "surface $d$ = 9"), (6.2, 100 * 152 / 171, "surface $d$ = 15")]
ROUTED = [  # label, fallback, (network %, fallback %) on R, (network %, fallback %, fallback lo, hi) on K
    ("surface $d$ = 9", "$q$ = 0.96 %, correlated PM", (39.3, 26.2), (0.810, 0.915, None, None)),
    ("surface $d$ = 15", "$q$ = 6.2 %, correlated PM", (12.3, 4.6), (0.101, 0.149, None, None)),
    ("BB72, $p$ = 0.3 %", "$q$ = 2 %, full-DEM Relay-BP", (100 * 1206 / 4000, 100 * 710 / 4000),
     (100 * 190 / 196000, 100 * 407 / 196000, None, None)),
    ("BB72, $p$ = 0.2 %", "$q$ = 0.25 %, full-DEM Relay-BP", (100 * 114 / 500, 100 * 74 / 500),
     (100 * 32 / 199500, 100 * 95 / 199500, None, None)),
    ("gross, $p$ = 0.4 %", "$q$ = 2 %, full-DEM Relay-BP", (100 * 707 / 2000, 100 * 146 / 2000),
     (100 * 10 / 98000, 100 * 121.6 / 98000, 100 * 52.8 / 98000, 100 * 269.1 / 98000)),
    ("BB72, $p$ = 0.3 %", "$q$ = 2 %, sector Relay-BP", (100 * 1206 / 4000, 100 * 1370 / 4000),
     (100 * 190 / 196000, 100 * 1364 / 196000, None, None)),
]


def fig_motivation(path):
    fig = plt.figure(figsize=(COL, 3.75))
    a = fig.add_axes([0.135, 0.665, 0.845, 0.315])
    q = [0.08 * 1.05 ** i for i in range(105)]
    a.plot(q, q, ls=(0, (1.2, 1.6)), color=PMC, lw=0.8)
    a.text(4.4, 8.5, "random routing", color=PMC, ha="center", va="bottom", fontsize=7.5)
    for lab, rows, mk, (tx, ty, ha) in CONC:
        a.plot([r[0] for r in rows], [r[1] for r in rows], marker=mk, color=CL, ms=3.6, mfc=CL, mew=0.5, lw=0.9)
        a.text(tx, ty, lab, color=CL, ha=ha, va="center", fontsize=7.5)
    for x, y, lab in CONC_DEV:
        a.plot([x], [y], marker="D", color=CL, ms=3.8, mfc="white", mew=0.8, ls="none")
    a.text(0.96 * 1.17, 100 * 189 / 590, "surface $d$ = 9", color=CL, ha="left", va="center", fontsize=7.5)
    a.text(6.2, 100 * 152 / 171 - 8, "surface\n$d$ = 15", color=CL, ha="center", va="top", fontsize=7.5,
           linespacing=1.0)
    a.set_xscale("log")
    a.set_xlim(0.06, 9)
    a.set_ylim(0, 103)
    plain_log(a, "x", [0.25, 0.5, 1, 2, 4, 8])
    a.set_xlabel("routed fraction $q$ (%), least confident first", labelpad=1)
    a.set_ylabel("failures captured (%)", labelpad=2)
    grid(a)
    panel(a, "a", x=-0.155, y=0.97, ha="left")

    rows = list(reversed(ROUTED))
    y = list(range(len(rows)))
    axR = fig.add_axes([0.435, 0.085, 0.255, 0.445])
    axK = fig.add_axes([0.725, 0.085, 0.255, 0.445], sharey=axR)
    for ax, key, lim, ticks, title in ((axR, 2, (3, 60), [5, 10, 20, 40], "routed shots $R$"),
                                       (axK, 3, (0.006, 1.2), [0.01, 0.1, 1], "retained shots $K$")):
        for yi, r in zip(y, rows):
            v = r[key]
            ax.plot([v[0], v[1]], [yi, yi], color="#BBBBBB", lw=1.0, zorder=1)
            ax.plot([v[0]], [yi], "o", color=CL, ms=3.8, zorder=3)
            if key == 3 and v[2] is not None:
                ax.errorbar([v[1]], [yi], xerr=[[v[1] - v[2]], [v[3] - v[1]]], fmt="o", color=FB, ms=3.8,
                            mfc="white", mew=0.8, capsize=1.2, elinewidth=0.6, zorder=3)
            else:
                ax.plot([v[1]], [yi], "o", color=FB, ms=3.8, zorder=3)
        ax.set_xscale("log")
        ax.set_xlim(*lim)
        plain_log(ax, "x", ticks)
        ax.set_title(title, fontsize=8, pad=3)
        grid(ax, "x")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    axK.tick_params(axis="y", labelleft=False)
    axR.set_yticks(y)
    axR.set_yticklabels([f"{r[0]}\n{r[1]}" for r in rows], fontsize=7.5, linespacing=1.05)
    axR.set_ylim(-0.55, len(rows) - 0.45)
    fig.text(0.708, 0.004, "failure rate (%)", ha="center", va="bottom", fontsize=8)
    r2 = [i for i, r in enumerate(rows) if r[0].startswith("gross")][0]
    axR.annotate("network", (rows[r2][2][0], r2), xytext=(0, 5), textcoords="offset points", ha="center",
                 va="bottom", color=CL, fontsize=7.5)
    axR.annotate("fallback", (rows[r2][2][1], r2), xytext=(0, 5), textcoords="offset points", ha="center",
                 va="bottom", color=FB, fontsize=7.5)
    axK.annotate("estimated", (rows[r2][3][1], r2), xytext=(0, 5), textcoords="offset points", ha="center",
                 va="bottom", color=FB, fontsize=7.5)
    panel(axR, "b", x=-1.6, y=1.02, ha="left")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# surface_results.pdf: surface code. Registered results; R README "Results" tables, results/s2_test_eval.csv, s3_test_eval.csv,
# s2_composite.json and results_accuracy_speed_d15.json (d = 7-15). d = 17 is R17 on T6 (hub branch ising-r17,
# ising_compare/r17/run/final/r17_eval.csv and r17_summary.json; R17_RESULT.md sections 1.2 and 3): ConvLines at
# q = 8 % (2,708 failures, 135.019 us), every opponent re-read on T6, BP-OSD on T6's first 200,000 shots, and the
# opponents' times study 4's registered constants. Times are the registered composite times in us per shot on the
# Grace basis (our slowest qualifying process, each opponent's fastest repetition). The Ising entries pair each
# model's fewest failures over its numerics and plans with its fastest plan. BP-OSD read the first 200,000 test shots
# at d = 15 and 17.
# One panel per distance, failure rate against composite time, with dotted lines through ConvLines. Every time axis
# spans SPAN decades from 1.5 times below ConvLines' time, so horizontal distances (time ratios) compare across panels.
# BP-OSD (2.9-619 ms) lies beyond every axis and sits at its right edge with its time. The old panel (c), ConvLines'
# time over the fastest opponent's (0.64-0.94), is in the RQ1 text.
D = [7, 9, 11, 13, 15, 17]
N = [2_000_000] + [1_000_000] * 5
NB = [2_000_000] + [1_000_000] * 3 + [200_000] * 2
SURF = {  # key: (label, colour, marker, filled, failures per d, shots per d, us per shot per d)
    "ours": ("ConvLines", CL, MODE["balanced"], True, [33582, 10789, 7215, 5627, 3911, 2708], N,
             [12.924, 30.049, 43.925, 57.957, 86.080, 135.019]),
    "cpm": ("correlated PM", FB, "s", True, [35291, 12917, 9325, 6549, 4311, 2943], N,
            [35.710, 67.590, 155.610, 279.630, 464.950, 714.050]),
    "pm": ("PM", PMC, "o", True, [60138, 27317, 24358, 21501, 18442, 15658], N,
           [17.160, 36.990, 72.020, 129.610, 208.530, 325.230]),
    "m1": ("Ising-Fast + PM", ISP, "^", True, [48875, 22841, 20261, 17478, 15232, 12840], N,
           [48.756, 56.401, 68.633, 90.262, 122.846, 161.330]),
    "m4": ("Ising-Accurate + PM", ISP, "v", True, [38000, 17001, 14817, 12399, 10077, 8070], N,
           [61.410, 68.083, 78.666, 98.342, 129.631, 165.970]),
    "minllr": ("BP-OSD, min-LLR", BPG, "p", True, [34227, 16332, 15078, 13317, 2318, 2114], NB,
               [2864.199, 9353.560, 25202.852, 68177.064, 197370.382, 444502.444]),
    "bp60": ("BP-OSD, 60 iterations", BPG, "p", False, [41124, 19873, 18881, 17623, 3247, 3141], NB,
             [5668.339, 18211.254, 46430.191, 115302.017, 294550.813, 619208.996]),
}
SURF_D17 = {  # registered at d = 17 only (R17 on T6): Ising followed by correlated PM (failures, us)
    "m1c": (12405, 297.508, "^"), "m4c": (7877, 275.794, "v")}
SPAN = 1.4   # decades of time in every panel
SXT = [[10, 30, 100], [30, 100, 300], [30, 100, 300], [100, 300], [100, 300, 1000], [100, 300, 1000]]
SYLIM = [(1.43, 3.19), (0.83, 3.1), (0.49, 2.75), (0.386, 2.46), (0.253, 2.15), (0.164, 1.85)]
SYT = [[1.5, 2, 3], [1, 1.5, 2, 3], [0.5, 1, 1.5, 2], [0.5, 1, 1.5, 2], [0.3, 0.5, 1, 2], [0.2, 0.5, 1, 1.5]]


def mark(col, mk, filled=True, ms=3.6):
    return Line2D([], [], color=col, marker=mk, ms=ms, mfc=col if filled else "white", mew=0.8, ls="none")


def fig_surface(path):
    fig = plt.figure(figsize=(TEXT, 2.14))
    w, h = fig.get_size_inches()
    left, gap, base, top = 0.40, 0.25, 0.31, 0.44   # inches
    pw, ph = (w - left - 5 * gap - 0.04) / 6, h - base - top
    axs = [fig.add_axes([(left + c * (pw + gap)) / w, base / h, pw / w, ph / h]) for c in range(6)]
    for i, (ax, d) in enumerate(zip(axs, D)):
        xlim = (SURF["ours"][6][i] / 1.5, SURF["ours"][6][i] / 1.5 * 10 ** SPAN)
        pts = [(k, col, mk, filled, fails[i], shots[i], us[i])
               for k, (_, col, mk, filled, fails, shots, us) in SURF.items()]
        if d == 17:
            pts += [(k, ISP, mk, False, f, 1_000_000, us) for k, (f, us, mk) in SURF_D17.items()]
        for k, col, mk, filled, f, n, us in pts:
            rate, (lo, hi) = 100 * f / n, wilson(f, n)
            size, off = 5.0 if k == "ours" else 4.0, us > xlim[1]
            ax.errorbar([xlim[1] if off else us], [rate], yerr=[[rate - 100 * lo], [100 * hi - rate]], fmt=mk,
                        color=col, ms=size, mfc=col if filled else "white", mew=0.8, capsize=0, elinewidth=0.6,
                        zorder=5 if k == "ours" else 3, clip_on=not off)
            if off:   # BP-OSD: its time, with an arrow to its marker at the right edge
                t = us / 1000
                ax.annotate(f"{t:.1f} ms" if t < 10 else f"{t:.0f} ms", (xlim[1], rate), xytext=(-9, 0),
                            textcoords="offset points", ha="right", va="center", fontsize=7.5, color=col, zorder=4,
                            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none"), annotation_clip=False,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=col, lw=0.6,
                                            shrinkA=0.5, shrinkB=size / 2 + 0.3))
        xo, ro = SURF["ours"][6][i], 100 * SURF["ours"][4][i] / SURF["ours"][5][i]
        ax.axvline(xo, color=CL, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
        ax.axhline(ro, color=CL, lw=0.7, ls=(0, (1, 1.5)), zorder=2)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(*SYLIM[i])
        plain_log(ax, "x", SXT[i])
        plain_log(ax, "y", SYT[i])
        ax.tick_params(pad=1.5)
        ax.text(0.97, 0.04, f"$d$ = {d}", transform=ax.transAxes, ha="right", va="bottom", fontsize=8.5)
        grid(ax)
    hs = [mark(col, mk, filled, 5.0 if k == "ours" else 4.0) for k, (_, col, mk, filled, *_) in SURF.items()]
    labs = [v[0] for v in SURF.values()]
    hs.insert(5, (mark(ISP, "^", False, 4.0), mark(ISP, "v", False, 4.0)))
    labs.insert(5, "Ising + correlated PM ($d$ = 17)")
    fig.legend(hs, labs, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=4, frameon=False, borderaxespad=0.1,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)}, handlelength=1.5, handletextpad=0.4,
               columnspacing=1.4, labelspacing=0.3)
    fig.text((left + 3 * pw + 2.5 * gap) / w, 0.01, "composite time per shot (µs), Grace basis", ha="center",
             va="bottom", fontsize=8)
    fig.text(0.008, (base + ph / 2) / h, "logical error rate (%)", rotation=90, ha="left", va="center",
             fontsize=8)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# bb_results.pdf: BB codes. Registered results: R README "BB72" (cells BB72-3, -2, -2F, -3F, -2S, -3S),
# bb72/ising_compare/data/bb/stage3/results.json and results_amendment_S.json; R README "BB144: two hybrids",
# gross_stage4/ising_compare/data/bb/stage4_gross/results.json. Times in ms per shot: our composite (GPU term on the
# H100 + handoff + routed CPU work on one Apple M4 core) and each opponent's time in the same session. Network 1 (r11)
# is the fast mode at p = 0.3 %; network 2 (r12) serves every other BB72 entry. Its standalone 1,387 at p = 0.3 % is
# reported beside the registered cells, with the hybrids' measured GPU term + handoff (0.102 ms). The gross network
# alone (970) was reported beside the registered results. Full-DEM Relay-BP at p = 0.2 % was timed after the test
# (hub ising-bb72-supp-time 6dfb2e7f, SUPPLEMENT_fulldem_time_p0.002.md: 36.4087 ms, the fastest of three fresh
# processes on the registered M4 basis; its p = 0.3 % control gave 58.05 against the registered 58.60 ms), drawn
# hollow. Composite times are each cell's composite_ms in results.json / results_amendment_S.json.
BB = [  # title, shots, [(key, failures, ms, label, label offset in points, ha, va, leader line)]
    ("BB72, $p$ = 0.3 %", 200000, [
        ("fast", 1341, 0.10187, "fast, first network", (12, -12), "left", "top", True),
        ("net2", 1387, 0.10204, "second network alone", (10, 16), "left", "bottom", True),
        ("balanced", 1571, 0.26022, "balanced, second network", (6, 0), "left", "center", False),
        ("accurate", 877, 3.4306, "accurate, second network", (0, -7), "center", "top", False),
        ("rb", 2704, 1.6140, "sector Relay-BP", (6, 0), "left", "center", False),
        ("bposd", 3299, 1.3450, "BP-OSD, ldpc", (5, 4), "left", "bottom", False),
        ("gpu", 3324, 0.9566, "BP-OSD,\nCUDA-Q", (-6, 0), "right", "center", False),
        ("fd", 1092, 58.596, "full-DEM Relay-BP", (6, 6), "right", "bottom", False)]),
    ("BB72, $p$ = 0.2 %", 300000, [
        ("fast", 254, 0.10204, "fast, second network", (6, -4), "left", "top", False),
        ("balanced", 290, 0.11468, "balanced", (6, 3), "left", "bottom", False),
        ("accurate", 171, 0.32379, "accurate", (6, -2), "left", "top", False),
        ("rb", 812, 1.0057, "sector Relay-BP", (0, -7), "center", "top", False),
        ("bposd", 903, 0.50091, "BP-OSD, ldpc and CUDA-Q", (4, 6), "left", "bottom", False),
        ("gpu", 917, 0.51659, "", (0, 0), "left", "center", False),
        ("fdsupp", 274, 36.4087, "full-DEM Relay-BP,\ntimed after the test", (-4, 5), "right", "bottom", False)]),
    ("gross, $p$ = 0.4 %", 150000, [
        ("fastx", 970, 0.27886, "network alone,\nnot a registered claim", (6, 7), "left", "bottom", True),
        ("balanced", 387, 9.3584, "balanced", (6, 0), "left", "center", False),
        ("accurate", 229, 232.187, "accurate", (-6, 0), "right", "center", False),
        ("rb", 716, 59.044, "sector Relay-BP", (-6, 0), "right", "center", False),
        ("bposd", 3865, 73.659, "BP-OSD, ldpc", (6, 0), "left", "center", False),
        ("gpu", 3789, 13.899, "BP-OSD,\nCUDA-Q", (-6, 0), "right", "center", False),
        ("fd", 484, 2774.707, "full-DEM\nRelay-BP", (-6, 0), "right", "center", False)]),
]
BSTYLE = {  # key: (colour, marker, filled); both Relay-BP decoders are fallbacks, so blue
    "fast": (CL, MODE["fast"], True), "balanced": (CL, MODE["balanced"], True),
    "accurate": (CL, MODE["accurate"], True), "net2": (CL, MODE["fast"], False), "fastx": (CL, MODE["fast"], False),
    "rb": (FB, "^", True), "bposd": (BPG, "h", True), "gpu": (BPG, "p", True), "fd": (FB, "v", True),
    "fdsupp": (FB, "v", False),
}
BXLIM = [(0.06, 200), (0.06, 80), (0.15, 6000)]
BYLIM = [(0.3, 2.0), (0.04, 0.4), (0.12, 3.5)]
BYT = [[0.3, 0.5, 1, 2], [0.05, 0.1, 0.2, 0.4], [0.2, 0.5, 1, 2, 3]]
BXT = [[0.1, 1, 10, 100], [0.1, 1, 10], [1, 10, 100, 1000]]


def fig_bb(path):
    fig = plt.figure(figsize=(TEXT, 2.3))
    axs = [fig.add_axes([x0, 0.165, w, 0.74]) for x0, w in ((0.068, 0.262), (0.4, 0.262), (0.732, 0.262))]
    for i, (ax, (title, n, rows)) in enumerate(zip(axs, BB)):
        for key, f, ms, lab, off, ha, va, lead in rows:
            col, mk, filled = BSTYLE[key]
            rate = 100 * f / n
            lo, hi = wilson(f, n)
            ours = col == CL
            ax.errorbar([ms], [rate], yerr=[[rate - 100 * lo], [100 * hi - rate]], fmt=mk, color=col,
                        ms=5.0 if ours else 4.0, mfc=col if filled else "white", mew=0.8, capsize=0,
                        elinewidth=0.6, zorder=4 if ours else 3)
            if lab:
                label(ax, lab, (ms, rate), off, ha, va, col, lead)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*BXLIM[i])
        ax.set_ylim(*BYLIM[i])
        plain_log(ax, "y", BYT[i])
        plain_log(ax, "x", BXT[i])
        ax.set_title(title, fontsize=8, pad=3)
        ax.set_xlabel("composite time per shot (ms)", labelpad=1)
        grid(ax)
    axs[0].set_ylabel("block failure rate (%)", labelpad=2)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# routing_frontiers.pdf: routing frontiers on calibration shots (all read before any test). Hybrid failures against q.
# Surface (200,000 calibration shots): H ising-arch:ising_compare/data/arch/policy_d9b3_r764.json and
# policy_d17b5_r787.json ("table" rows); finer d = 9 rows from frontier_d9b3_r764.json. Projected composite times
# against the speed limit: the same policy files (d = 9: 33.6 us at q = 1 %, 34.6 us at 2 %, limit 33.99 us; d = 17:
# at most 313.4 us up to 30 %, limit 322.23 us). The d = 9 shade starts where the projection crosses the limit:
# 31.44 + 1.1 + routed cost (frontier rows: 1.051 us at 1 %, 1.570 us at 1.5 %) reaches 33.99 us at q = 1.38 %.
# BB72 p = 0.3 % (network 2, 200,000): full-DEM rows from R bb72/.../fullroute/bb72_cal.json, sector rows from
# R bb72/ising_compare/data/bb/policies/bb72_p0.003_rep2.json; alone: full-DEM 1,117, sector 2,734.
# Gross (100,000): full-DEM rows from R gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md lines 64-71 (alone: about
# 268, estimated); sector rows from R gross/ising_compare/data/bb/stage3_gross/frontierf_g2_D12k113c10.json (alone 490).
# Rings: the registered q (their routed CPU terms are in the paper's time table). At d = 17 the curve is
# study 4's calibration set C17b and the ring marks R17's q = 8 % (573 failures on C17b, PLAN_R17.md section 1.3),
# which the rule chose on deep17's calibration set K17d; R17's CPU term is 66.816 us (R17_RESULT.md section 3, hub
# branch ising-r17).
FRONT = [
    ("surface $d$ = 9", "correlated PM", [
        ("balanced", [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 14, 16],
         [2484, 2362, 2288, 2226, 2188, 2146, 2129, 2179, 2226, 2324, 2392, 2441, 2473, 2494, 2511], 1)],
     [("correlated PM alone", 2562)], 1.38, (0, 16)),
    ("surface $d$ = 17", "correlated PM", [
        ("balanced", [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
         [1719, 860, 657, 591, 573, 560, 567, 578, 582, 589, 591, 592, 597, 598, 598, 599], 8)],
     [("correlated PM alone", 615)], None, (0, 30)),
    ("BB72, $p$ = 0.3 %", None, [
        ("accurate", [0, 0.25, 0.5, 1, 2, 3, 5], [1396, 1200, 1050, 937, 900, 929, 989], 2),
        ("balanced", [0, 0.5, 1, 2, 3, 4], [1396, 1230, 1298, 1560, 1786, 1945], 2)],
     [("full-DEM\nRelay-BP alone", 1117), ("sector Relay-BP alone", 2734)], None, (0, 7)),
    ("gross, $p$ = 0.4 %", None, [
        ("accurate", [0, 0.25, 0.5, 1, 1.5, 2, 3], [717, 523, 355, 164, 157, 156, 179], 2),
        ("balanced", [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4],
         [717, 577, 521, 463, 409, 345, 307, 280, 262, 260, 282, 336, 367], 1)],
     [("full-DEM Relay-BP\nalone (estimated)", 268), ("sector Relay-BP alone", 490)], None, (0, 9)),
]
FLAB = {  # per panel: label placements (x, y, ha, va) for curves and references, each in a free region
    0: {"correlated PM alone": (15.6, 2572, "right", "bottom"), "shade": (9.6, 2150)},
    1: {"correlated PM alone": (29.5, 690, "right", "bottom")},
    2: {"accurate": (5.25, 989, "left", "center"), "balanced": (4.2, 1945, "left", "center"),
        "full-DEM\nRelay-BP alone": (6.95, 1150, "right", "bottom"), "sector Relay-BP alone": (6.95, 2700, "right", "top")},
    3: {"accurate": (1.0, 98, "center", "center"), "balanced": (4.2, 367, "left", "center"),
        "full-DEM Relay-BP\nalone (estimated)": (8.9, 252, "right", "top"),
        "sector Relay-BP alone": (8.9, 508, "right", "bottom")},
}
FYLIM = [(2000, 2650), (420, 1800), (600, 2900), (30, 780)]
FXT = [[0, 4, 8, 12, 16], [0, 6, 12, 18, 24, 30], list(range(8)), [0, 2, 4, 6, 8]]


def fig_frontier(path):
    fig, axs = plt.subplots(2, 2, figsize=(COL, 3.3))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.94, bottom=0.11, wspace=0.3, hspace=0.42)
    for i, (ax, (title, _, curves, refs, shade, xlim)) in enumerate(zip(axs.flat, FRONT)):
        L = FLAB[i]
        if shade is not None:
            ax.axvspan(shade, xlim[1], color="#000000", alpha=0.07, lw=0)
            ax.text(*L["shade"], "over the\nspeed limit", ha="center", va="center", fontsize=7.5, color="#555555",
                    linespacing=1.0)
        dashes = [(0, (4, 2)), (0, (1.5, 1.5))]
        for (lab, v), ls in zip(refs, dashes):
            ax.axhline(v, color=FB, lw=0.8, ls=ls)
            x, y, ha, va = L[lab]
            ax.text(x, y, lab, color=FB, fontsize=7.5, ha=ha, va=va, linespacing=1.0)
        for mode, q, f, qreg in curves:
            ax.plot(q, f, color=CL, marker=MODE[mode], ms=2.8, lw=0.9, mew=0, zorder=3)
            j = q.index(qreg)
            ax.plot([qreg], [f[j]], "o", ms=7.5, mfc="none", mec="black", mew=0.7, zorder=4)
            if len(curves) > 1:
                x, y, ha, va = L[mode]
                ax.text(x, y, mode, fontsize=7.5, ha=ha, va=va, color=CL)
        ax.set_xlim(*xlim)
        ax.set_ylim(*FYLIM[i])
        ax.set_title(title, fontsize=8, pad=3)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(4))
        ax.set_xticks(FXT[i])
        grid(ax)
    for ax in axs[1]:
        ax.set_xlabel("routed fraction $q$ (%)", labelpad=1)
    for ax in axs[:, 0]:
        ax.set_ylabel("failures", labelpad=1)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    fig_motivation(os.path.join(OUT, "failures_by_confidence.pdf"))
    fig_surface(os.path.join(OUT, "surface_results.pdf"))
    fig_bb(os.path.join(OUT, "bb_results.pdf"))
    fig_frontier(os.path.join(OUT, "routing_frontiers.pdf"))
    print("wrote failures_by_confidence.pdf, surface_results.pdf, bb_results.pdf, routing_frontiers.pdf")
