"""Data figures of the CoDecoder paper, drawn at their printed size as vector PDFs.

    python results/figures/make_figs.py    (writes the four PDFs below next to this file; about 5 s)

Figs. 1 and 2 are TikZ sources (overview.tex, output_head.tex) that main.tex inputs. This script draws:

  failures_by_confidence.pdf  where the network's failures sit, routed and retained shots (one column, stacked)
  surface_results.pdf         surface code, failure rate against composite time at each distance (full width)
  bb_results.pdf              BB72 and the gross code, block failure rate against composite time (full width)
  routing_frontiers.pdf       routing frontiers on calibration shots (one column, 2 x 2)

make_why_fig.py draws why_pairs_win.pdf and why_class_gap.pdf from the inputs of the analysis of the
discordant shots.

One visual language: CoDecoder vermillion with one marker per mode. Every fallback is blue wherever it appears,
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

CL, FB = "#D55E00", "#0072B2"             # CoDecoder, every fallback (correlated PM, both Relay-BP, Cascade)
ISP, BPG = "#CC79A7", "#009E73"           # Ising, BP-OSD
PMC, GRID = "#8C8C8C", "#E6E6E6"          # PM, grid
MODE = {"fast": "o", "balanced": "D", "accurate": "s"}   # CoDecoder's mode markers in every figure
LEAD = dict(arrowstyle="-", lw=0.45, shrinkA=1.0, shrinkB=2.6)   # leader line from a label to its marker


def label(ax, text, xy, off, ha, va, col, lead=False, fs=7.5, sb=LEAD["shrinkB"]):
    """Label the marker at xy, offset by off points; a leader line joins labels set away from their marker."""
    rel = ({"left": 0.0, "right": 1.0}.get(ha, 0.5), {"top": 1.0, "bottom": 0.0}.get(va, 0.5))   # nearest corner
    ax.annotate(text, xy, xytext=off, textcoords="offset points", ha=ha, va=va, fontsize=fs, color=col,
                linespacing=1.0, arrowprops=dict(LEAD, color=col, relpos=rel, shrinkB=sb) if lead else None, zorder=6)


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
    ax.text(x, y, f"({letter})", transform=ax.transAxes, ha=ha, va="bottom", fontsize=8)   # as the other panel titles


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
    axR.annotate("CoDecoder", (rows[r2][2][0], r2), xytext=(0, 5), textcoords="offset points", ha="center",
                 va="bottom", color=CL, fontsize=7.5)
    axR.annotate("fallback", (rows[r2][2][1], r2), xytext=(0, 5), textcoords="offset points", ha="center",
                 va="bottom", color=FB, fontsize=7.5)
    axK.annotate("estimated", (rows[r2][3][1], r2), xytext=(0, 5), textcoords="offset points", ha="center",
                 va="bottom", color=FB, fontsize=7.5)
    panel(axR, "b", x=-1.6, y=1.02, ha="left")
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# surface_results.pdf: surface code. Failures: the registered surface-code test of networks trained on the final answer
# (public repository, surface_v2/surf_v2/analysis/paper_rows.csv; BP-OSD read the first 200,000 test shots at d = 15
# and 17; Ising's failures from its official PyTorch numerics). Times: the registered speed re-test (public repository
# a85tract/Convlines_Public at e0a9377, speed_rt/s2/surface/results/analysis.json and rows.csv S1), composite times in
# us per shot on the Grace basis: our slowest process (GPU term of five, CPU term of three), each baseline's fastest of
# three, Ising's fastest validated TensorRT plan plus its PM share, BP-OSD unconverted. The network alone (q = 0) is
# the network term plus the handoff with nothing routed (descriptive). Our Decoder Switching
# (speed_rt/s2/gapsw/results/gs_times.csv): the cross is its frozen q with its composite, the curve its test failures at
# every q >= 1 % of the grid (surface_v2/surf_v2/gapsw/analysis) with the time front end + (q / frozen q) x routed term.
# One panel per distance, failure rate against composite time, with dotted lines through CoDecoder. Every time axis
# spans SPAN decades from 1.5 times below CoDecoder's time, so horizontal distances (time ratios) compare across panels.
# BP-OSD (2.2-648 ms) lies beyond every axis and sits at its right edge; the paper's surface table gives its times.
# d = 9: CoDecoder and the network alone come from the second registered test, of a one-block network (public
# repository a85tract/convlines at 68947ee, d9_shallow/results/analysis.json): 11,978 and 18,305 failures in 1,000,000
# other test shots, 21.234 us (q = 0.06) and 15.305 us (q = 0, network term plus the 1.1 us allowance). PM and
# correlated PM at d = 9 take that test's counts on the same shots (27,028 and 12,802); the other d = 9 baselines keep
# the main test's counts, and every baseline time is the speed re-test's.
D = [7, 9, 11, 13, 15, 17]
N = [2_000_000] + [1_000_000] * 5
NB = [2_000_000] + [1_000_000] * 3 + [200_000] * 2
SURF = {  # key: (label, colour, marker, filled, failures per d, shots per d, us per shot per d)
    "ours": ("CoDecoder + correlated PM", CL, MODE["balanced"], True, [31629, 11978, 6887, 5318, 3012, 2180], N,
             [11.942, 21.234, 41.482, 51.245, 59.552, 81.579]),   # d = 9: the second registered test (one-block network)
    "cpm": ("correlated PM", FB, "s", True, [35135, 12802, 9252, 6344, 4396, 2973], N,   # d = 9: second test's shots
            [27.598, 63.916, 137.889, 253.071, 562.655, 846.008]),
    "pm": ("PM", PMC, "o", True, [60016, 27028, 24365, 21086, 18259, 15726], N,   # d = 9: second test's shots
           [11.890, 26.297, 52.234, 91.704, 160.091, 242.527]),
    "m1": ("Ising-Fast + PM", ISP, "^", True, [48752, 22908, 20650, 17491, 15043, 13149], N,
           [47.144, 57.204, 69.350, 92.311, 123.821, 173.116]),
    "m4": ("Ising-Accurate + PM", ISP, "v", True, [38157, 17400, 15040, 12440, 9932, 8083], N,
           [58.995, 67.223, 78.592, 100.809, 128.178, 172.988]),
    "m1c": ("Ising-Fast + correlated PM", ISP, "^", False, [48873, 22856, 20405, 17101, 14733, 12764], N,
            [51.068, 69.840, 99.694, 153.944, 238.682, 354.430]),
    "m4c": ("Ising-Accurate + correlated PM", ISP, "v", False, [38204, 17379, 14977, 12298, 9769, 7918], N,
            [61.907, 77.294, 102.477, 148.970, 218.353, 321.349]),
    "minllr": ("BP-OSD, min-LLR", BPG, "p", True, [34385, 16409, 15250, 13405, 2361, 2171], NB,
               [2183.543, 8655.385, 31365.008, 87743.224, 193252.381, 478767.236]),
    "bp60": ("BP-OSD, 60 iterations", BPG, "p", False, [41186, 19959, 19033, 17791, 3368, 3046], NB,
             [5950.825, 19033.813, 52051.064, 130648.507, 287949.212, 647847.051]),
}
DSW = "#000000"   # our Decoder Switching: black, the one decoder outside the palette's families
GAPSW = {  # d: (failures at the frozen q, measured composite us), q = 16 % at d = 7-15 and 30 % at d = 17
    7: (35322, 57.414), 9: (13159, 138.984), 11: (9512, 313.339), 13: (6594, 639.744),
    15: (4609, 1217.160), 17: (2990, 2402.314)}
GAPSW_Q = {  # d: [(q in %, test failures, us)] for every q >= 1 % of the grid; us = front end + (q / frozen q) x routed term
    7: [(1, 54087, 51.85), (2, 49520, 52.22), (4, 43352, 52.96), (6, 39698, 53.70), (8, 37590, 54.45),
        (10, 36428, 55.19), (12, 35785, 55.93), (14, 35514, 56.67), (16, 35322, 57.41)],
    9: [(1, 24158, 126.68), (2, 21490, 127.50), (4, 17787, 129.14), (6, 15725, 130.78), (8, 14509, 132.42),
        (10, 13875, 134.06), (12, 13549, 135.70), (14, 13299, 137.34), (16, 13159, 138.98)],
    11: [(1, 20624, 289.24), (2, 17825, 290.85), (4, 14109, 294.06), (6, 12087, 297.27), (8, 10961, 300.49),
        (10, 10311, 303.70), (12, 9943, 306.91), (14, 9667, 310.13), (16, 9512, 313.34)],
    13: [(1, 17412, 598.70), (2, 14523, 601.44), (4, 10892, 606.91), (6, 8915, 612.38), (8, 7881, 617.85),
        (10, 7299, 623.33), (12, 6957, 628.80), (14, 6742, 634.27), (16, 6594, 639.74)],
    15: [(1, 14442, 1141.01), (2, 11604, 1146.09), (4, 8175, 1156.24), (6, 6524, 1166.39), (8, 5648, 1176.55),
        (10, 5200, 1186.70), (12, 4926, 1196.85), (14, 4707, 1207.01), (16, 4609, 1217.16)],
    17: [(1, 11939, 2181.47), (2, 9154, 2189.08), (4, 6221, 2204.31), (6, 4792, 2219.55), (8, 4063, 2234.78),
        (10, 3647, 2250.01), (12, 3409, 2265.24), (14, 3263, 2280.47), (16, 3186, 2295.70), (18, 3141, 2310.93),
        (20, 3096, 2326.16), (22, 3059, 2341.39), (24, 3046, 2356.62), (26, 3025, 2371.85), (28, 3009, 2387.08),
        (30, 2990, 2402.31)],
}
NETALONE = {  # d: (test failures of the network alone, q = 0; us per shot: network term a + handoff, nothing routed)
    7: (36156, 10.904), 9: (18305, 15.305), 11: (6887, 41.482), 13: (5318, 51.245),
    15: (4030, 52.337), 17: (3296, 69.608)}
SPAN = 2.0   # decades of time in every panel
SXT = [[10, 30, 100, 300], [30, 100, 300, 1000], [30, 100, 300, 1000], [100, 300, 1000], [100, 300, 1000],
       [100, 300, 1000]]
SYLIM = [(1.43, 3.19), (0.83, 3.1), (0.49, 2.75), (0.386, 2.46), (0.215, 2.15), (0.142, 1.85)]
SYT = [[1.5, 2, 3], [1, 1.5, 2, 3], [0.5, 1, 1.5, 2], [0.5, 1, 1.5, 2], [0.3, 0.5, 1, 2], [0.2, 0.5, 1, 1.5]]


def mark(col, mk, filled=True, ms=3.6):
    return Line2D([], [], color=col, marker=mk, ms=ms, mfc=col if filled else "white", mew=0.8, ls="none")


def fig_surface(path):
    fig = plt.figure(figsize=(TEXT, 2.14))
    w, h = fig.get_size_inches()
    left, gap, base, top = 0.40, 0.25, 0.31, 0.42   # inches
    pw, ph = (w - left - 5 * gap - 0.04) / 6, h - base - top
    axs = [fig.add_axes([(left + c * (pw + gap)) / w, base / h, pw / w, ph / h]) for c in range(6)]
    for i, (ax, d) in enumerate(zip(axs, D)):
        x0 = min(SURF["ours"][6][i] / 1.5, NETALONE[d][1] / 1.25)   # room for the network-alone diamond (d = 9)
        xlim = (x0, x0 * 10 ** SPAN)
        pts = [(k, col, mk, filled, fails[i], shots[i], us[i])
               for k, (_, col, mk, filled, fails, shots, us) in SURF.items()]
        for k, col, mk, filled, f, n, us in pts:
            rate, (lo, hi) = 100 * f / n, wilson(f, n)
            size, off = 5.0 if k == "ours" else 4.0, us > xlim[1]
            ax.errorbar([xlim[1] if off else us], [rate], yerr=[[rate - 100 * lo], [100 * hi - rate]], fmt=mk,
                        color=col, ms=size, mfc=col if filled else "white", mew=0.8, capsize=0, elinewidth=0.6,
                        zorder=5 if k == "ours" else 3, clip_on=not off)
            if off:   # BP-OSD: beyond the axis, a short arrow to its marker at the right edge
                ax.annotate("", (xlim[1], rate), xytext=(-8, 0), textcoords="offset points", annotation_clip=False,
                            arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", color=col, lw=0.6,
                                            shrinkA=0, shrinkB=size / 2 + 0.3), zorder=4)
        n = N[i]
        qs = GAPSW_Q[d]   # our Decoder Switching over q (projected times), then its frozen q (measured time)
        ax.plot([u for _, _, u in qs], [100 * f / n for _, f, _ in qs], color=DSW, lw=0.7, zorder=2.5)
        f, us = GAPSW[d]
        rate, (lo, hi) = 100 * f / n, wilson(f, n)
        ax.errorbar([us], [rate], yerr=[[rate - 100 * lo], [100 * hi - rate]], fmt="X", color=DSW, ms=4.4,
                    mfc=DSW, mew=0.5, capsize=0, elinewidth=0.6, zorder=4)
        q1, f1, u1 = qs[0]   # the curve's top end, q = 1 %
        left = d in (15, 17)   # a centred label would touch BP-OSD's marker at the right edge there
        ax.annotate("$q$ = 1%", (u1, 100 * f1 / n), xytext=(4, 2) if left else (0, 2), textcoords="offset points",
                    fontsize=7.5, ha="right" if left else "center", va="bottom", color=DSW, zorder=6)
        f, us = NETALONE[d]   # our network alone (q = 0): an open diamond, larger than CoDecoder's marker, which it
        # rings at d = 9-13, where the pair routes no shots (its Wilson interval is smaller than the marker)
        ax.plot([us], [100 * f / n], ls="none", marker=MODE["balanced"], ms=7.0, mfc="none", mec=CL, mew=0.8,
                zorder=6)
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
    order = ["ours", "cpm", "pm", "m1", "m4", "minllr", "bp60"]
    hs = [mark(SURF[k][1], SURF[k][2], SURF[k][3], 5.0 if k == "ours" else 4.0) for k in order]
    labs = [SURF[k][0] for k in order]
    hs.insert(0, mark(CL, MODE["balanced"], False, 6.0))
    labs.insert(0, "CoDecoder")
    hs.insert(6, (mark(ISP, "^", False, 4.0), mark(ISP, "v", False, 4.0)))
    labs.insert(6, "Ising + correlated PM")
    hs.append((Line2D([], [], color=DSW, lw=0.7), mark(DSW, "X", True, 4.4)))
    labs.append("our Decoder Switching")
    fig.legend(hs, labs, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=5, frameon=False, borderaxespad=0.1,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)}, handlelength=1.5, handletextpad=0.4,
               columnspacing=1.2, labelspacing=0.3)
    fig.text((left + 3 * pw + 2.5 * gap) / w, 0.01, "composite time per shot (µs), Grace basis", ha="center",
             va="bottom", fontsize=8)
    fig.text(0.008, (base + ph / 2) / h, "logical error rate (%)", rotation=90, ha="left", va="center",
             fontsize=8)
    fig.savefig(path)
    plt.close(fig)


# ---------------------------------------------------------------------------------------------------------------
# bb_results.pdf: BB codes. Failures are those of the registered tests (unchanged). Times in ms per shot are the speed
# re-test's (public repo a85tract/Convlines_Public at e0a9377, speed_rt/s2/bb/results/numbers.json, `pairs` and
# `baselines`), on the Grace basis: the composite of our pair is the GPU term on the H100 (the slowest of 5 fresh
# processes, 3 for the gross network) + the 1.1 us host handoff + the routed fraction times the fallback's mean time per
# routed shot (the slowest of 3, the Linux workstation's time x 1.5098); each CPU baseline (sector Relay-BP, ldpc BP-OSD,
# full-DEM Relay-BP) is the fastest of 3 on the same basis; CUDA-Q's BP-OSD is the fastest of 3 on the pod, not
# converted. Network 1 (r11) is the fast mode at p = 0.3 %; network 2 (r12) serves every other BB72 entry, its stand-alone
# 1,398 failures at p = 0.3 % being the second network alone (0.0894 ms). The gross network alone (1,024 failures) is the
# re-test's `network_alone` composite (0.2555 ms) and stays hollow because its failures are reported beside the
# registered results. Full-DEM Relay-BP at p = 0.2 % is timed by the re-test (52.78 ms, filled marker like the others).
BB = [  # title, shots, [(key, failures, ms, label, label offset in points, ha, va, leader line)]
    ("BB72, $p$ = 0.3 %", 200000, [
        ("fast", 1376, 0.090721, "CoDecoder, first network", (12, -12), "left", "top", True),
        ("net2", 1398, 0.089408, "CoDecoder, second network", (10, 16), "left", "bottom", True),
        ("balanced", 1554, 0.414777, "CoDecoder + sector Relay-BP", (6, 0), "left", "center", False),
        ("accurate", 857, 4.8640, "CoDecoder + full-DEM Relay-BP", (0, -7), "center", "top", False),
        ("rb", 2769, 3.3016, "sector Relay-BP", (6, 0), "left", "center", False),
        ("bposd", 3337, 1.8182, "BP-OSD, ldpc", (5, 4), "left", "bottom", False),
        ("gpu", 3342, 0.89085, "BP-OSD,\nCUDA-Q", (-6, 0), "right", "center", False),
        ("fd", 1107, 85.370, "full-DEM Relay-BP", (6, 6), "right", "bottom", False)]),
    ("BB72, $p$ = 0.2 %", 300000, [
        ("fast", 269, 0.089408, "CoDecoder", (-3, -30), "left", "top", True),
        ("balanced", 297, 0.11685, "CoDecoder +\nsector Relay-BP", (6, 3), "left", "bottom", False),
        ("accurate", 204, 0.42237, "CoDecoder +\nfull-DEM Relay-BP", (6, -2), "left", "top", False),
        ("rb", 767, 1.9866, "sector Relay-BP", (0, -7), "center", "top", False),
        ("bposd", 880, 0.69077, "BP-OSD, ldpc\nand CUDA-Q", (4, 3), "left", "bottom", False),
        ("gpu", 856, 0.53956, "", (0, 0), "left", "center", False),
        ("fd", 288, 52.784, "full-DEM Relay-BP", (-6, 0), "right", "center", False)]),
    ("gross, $p$ = 0.4 %", 150000, [
        ("fastx", 1024, 0.25548, "CoDecoder,\nnot a registered claim", (6, 7), "left", "bottom", True),
        ("balanced", 407, 20.352, "CoDecoder +\nsector Relay-BP", (-6, 0), "right", "center", False),
        ("accurate", 234, 427.54, "CoDecoder +\nfull-DEM Relay-BP", (-6, 0), "right", "center", False),
        ("rb", 733, 127.61, "sector Relay-BP", (-6, 0), "right", "center", False),
        ("bposd", 3646, 114.00, "BP-OSD, ldpc", (6, 0), "left", "center", False),
        ("gpu", 3754, 13.701, "BP-OSD,\nCUDA-Q", (-6, 0), "right", "center", False),
        ("fd", 472, 3811.2, "full-DEM\nRelay-BP", (-6, 0), "right", "center", False)]),
]
BSTYLE = {  # key: (colour, marker, filled); both Relay-BP decoders are fallbacks, so blue
    "fast": (CL, MODE["fast"], True), "balanced": (CL, MODE["balanced"], True),
    "accurate": (CL, MODE["accurate"], True), "net2": (CL, MODE["fast"], False), "fastx": (CL, MODE["fast"], False),
    "rb": (FB, "^", True), "bposd": (BPG, "h", True), "gpu": (BPG, "p", True), "fd": (FB, "v", True),
}
BXLIM = [(0.06, 200), (0.06, 80), (0.15, 6000)]
BYLIM = [(0.3, 2.0), (0.04, 0.5), (0.12, 3.5)]
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
            ring = key == "net2"   # 1 pt from the fast point: a larger open ring behind it, same data position
            ax.errorbar([ms], [rate], yerr=[[rate - 100 * lo], [100 * hi - rate]], fmt=mk, color=col,
                        ms=10.0 if ring else 5.0 if ours else 4.0, mfc=col if filled else "white", mew=0.8, capsize=0,
                        elinewidth=0.6, zorder=3.5 if ring else 4 if ours else 3)
            if lab:
                label(ax, lab, (ms, rate), off, ha, va, col, lead, sb=5.5 if ring else LEAD["shrinkB"])
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
# routing_frontiers.pdf: routing frontiers on calibration shots (all read before any test). Pair failures against q.
# Surface (200,000 calibration shots sv2cal): P surface_v2/surf_v2/policies/policy_d9_b3_sv2s1.json and
# policy_d17_b5_sv2.json ("table" rows, with correlated PM alone from "pm_corr_failures"). Projected composite times
# against the speed limit from the same files: d = 9: 31.44 us at q = 0, 33.566 at 1 %, 34.572 at 2 %, limit 33.99 us,
# so the shade starts where the linear projection crosses the limit, q = 1.42 %; d = 17: at most 311.1 us up to 30 %,
# limit 322.23 us. Rings: the registered q (q = 0 at d = 9, 1 % at d = 17; their routed CPU terms are in the paper's
# time table).
# BB72 p = 0.3 % (network 2, 200,000): full-DEM rows from R bb72/.../fullroute/bb72_cal.json, sector rows from
# R bb72/ising_compare/data/bb/policies/bb72_p0.003_rep2.json; alone: full-DEM 1,117, sector 2,734.
# Gross (100,000): full-DEM rows from R gross_stage4/ising_compare/bb/GROSS_FULLDEM_CAL.md lines 64-71 (alone: about
# 268, estimated); sector rows from R gross/ising_compare/data/bb/stage3_gross/frontierf_g2_D12k113c10.json (alone 490).
FRONT = [
    ("surface $d$ = 9", "correlated PM", [
        ("balanced", [0, 1, 2, 4, 6, 8, 10, 12, 14, 16],
         [2005, 1909, 1993, 2201, 2326, 2396, 2443, 2484, 2509, 2523], 0)],
     [("correlated PM alone", 2562)], 1.42, (0, 16)),
    ("surface $d$ = 17", "correlated PM", [
        ("balanced", [0, 1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30],
         [672, 464, 482, 521, 550, 575, 584, 590, 598, 601, 603, 606, 608, 609, 612, 613, 614], 1)],
     [("correlated PM alone", 619)], None, (0, 30)),
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
    0: {"correlated PM alone": (15.6, 2572, "right", "bottom"), "shade": (9.0, 2120),
        "balanced": (0.3, 2470, "left", "center")},
    1: {"correlated PM alone": (29.5, 626, "right", "bottom"), "balanced": (8.0, 480, "left", "center")},
    2: {"accurate": (4.75, 700, "center", "center"), "balanced": (0.15, 2080, "left", "bottom"),
        "full-DEM\nRelay-BP alone": (6.95, 1150, "right", "bottom"), "sector Relay-BP alone": (6.95, 2700, "right", "top")},
    3: {"accurate": (0.08, -35, "left", "bottom"), "balanced": (4.2, 367, "left", "center"),
        "full-DEM Relay-BP\nalone (estimated)": (8.9, 252, "right", "top"),
        "sector Relay-BP alone": (8.9, 508, "right", "bottom")},
}
FYLIM = [(1850, 2650), (440, 700), (450, 2900), (-40, 780)]
FXT = [[0, 4, 8, 12, 16], [0, 6, 12, 18, 24, 30], list(range(8)), [0, 2, 4, 6, 8]]


def fig_frontier(path):
    fig, axs = plt.subplots(2, 2, figsize=(COL, 3.3))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.94, bottom=0.11, wspace=0.3, hspace=0.42)
    for i, (ax, (title, fallback, curves, refs, shade, xlim)) in enumerate(zip(axs.flat, FRONT)):
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
            x, y, ha, va = L[mode]
            fb = fallback if fallback else ("full-DEM Relay-BP" if mode == "accurate" else "sector Relay-BP")
            ax.text(x, y, "CoDecoder +\n" + fb, fontsize=7.5, ha=ha, va=va, color=CL, linespacing=1.0)
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
