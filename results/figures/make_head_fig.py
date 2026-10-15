"""The figure of the output-head comparison (Appendix F, Fig. 7), drawn at its printed size as a vector PDF.

    python results/figures/make_head_fig.py   (writes head_ablation.pdf next to this file)

Failures per million test shots of each pair against the routed fraction q, for every head and seed of the
comparison: (a) the surface code at d = 9 and p = 0.6 % with correlated PM as the fallback, (b) BB72 at p = 0.3 % with
full-DEM Relay-BP. Colour and marker name the head, the line style its supervision (solid: the final answer
alone; dashed: with the line or translate supervision), and the marker fill names the seed (filled: seed 1, hollow:
seed 2). The visual language is make_figs.py's: STIX serif at 7.5-8 pt, CoDecoder's head vermillion, every fallback
blue.

Every number is the re-test's (RETEST_RESULTS.md section 4): retest/run/head/surface/pair_curves.csv and
retest/run/head/bb72/pair_curves.csv (column per_million), the fallbacks alone from the same analyses (12,989 and 5,228 per million, Table III of the paper).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import ticker  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
COL = 3.5                                 # IEEE column width, inches
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5, "axes.linewidth": 0.6,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
    "axes.spines.right": False,
})

CL, FB = "#D55E00", "#0072B2"             # CoDecoder's structure-aware head, every fallback
PO, CE = "#CC79A7", "#009E73"             # pooled head (independent outputs on BB72), centre line
GRID, INK = "#E6E6E6", "#333333"
# Okabe-Ito: the four colours pass the dataviz validator (lightness band, chroma floor, all-pairs CVD and the
# normal-vision floor); pink and green sit in the 6-8 CVD band, so every head also has its own marker.

# q in percent; per million test shots (1,000,000 surface shots, 500,000 BB72 shots), seeds 1 and 2.
QS = [0, 0.5, 1, 2, 4, 8, 16]
SURF = {  # arm: (head, supervised, [seed 1], [seed 2])
    "S-VA": ("vote", True, [15018, 13883, 13015, 11938, 11510, 11984, 12660],
             [13981, 12949, 12202, 11343, 11222, 12000, 12666]),
    "S-V": ("vote", False, [10661, 10099, 9921, 10121, 11035, 12155, 12777],
            [10822, 10188, 9919, 10135, 11050, 12145, 12770]),
    "S-L": ("line", False, [16798, 15530, 14470, 13002, 12007, 12071, 12647],
            [15088, 13967, 13056, 12019, 11563, 12060, 12666]),
    "S-P": ("pooled", False, [478245, 475802, 473494, 468839, 459429, 440627, 402532],   # seed 1 did not train
            [13684, 12670, 11931, 11257, 11223, 11969, 12687]),
    "S-PA": ("pooled", True, [14924, 13737, 12891, 11820, 11466, 11967, 12675],
             [15859, 14686, 13667, 12507, 11822, 12084, 12646]),
}
QB = [0, 0.25, 0.5, 1, 2, 3, 5]
BB = {
    "B-JT": ("vote", True, [6766, 5722, 5024, 4414, 4246, 4416, 4644], [6714, 5632, 4918, 4262, 4256, 4372, 4638]),
    "B-J": ("vote", False, [6974, 5868, 5068, 4382, 4210, 4402, 4656], [6954, 5910, 5104, 4360, 4254, 4406, 4658]),
    "B-P": ("pooled", False, [33748, 31738, 29732, 25698, 18280, 12020, 5992],
            [34702, 32566, 30536, 26678, 19416, 12716, 6116]),
    "B-PA": ("pooled", True, [27788, 25854, 23722, 19926, 12504, 7282, 4652],
             [25840, 23770, 21918, 18166, 10892, 6250, 4608]),
}
STYLE = {"vote": (CL, "o"), "line": (CE, "^"), "pooled": (PO, "s")}
ORDER = {"vote": 3, "line": 2, "pooled": 1}   # the structure-aware head on top


def sqrt_x(ax, qs, top):
    ax.set_xscale("function", functions=(lambda v: v ** 0.5, lambda v: v ** 2))
    ax.set_xlim(0, top)
    ax.xaxis.set_major_locator(ticker.FixedLocator(qs))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())


def grid(ax):
    ax.grid(True, which="major", color=GRID, lw=0.4, zorder=0)
    ax.set_axisbelow(True)


def curves(ax, arms, qs, skip=()):
    for name, (head, sup, s1, s2) in arms.items():
        col, mk = STYLE[head]
        for i, ys in enumerate((s1, s2)):
            if (name, i + 1) in skip:
                continue
            ax.plot(qs, ys, color=col, ls=(0, (3.2, 1.6)) if sup else "-", lw=0.85, marker=mk, ms=3.4,
                    mfc=col if i == 0 else "white", mec=col, mew=0.75, zorder=2 + ORDER[head])


def fallback(ax, y, text, x, ha):
    ax.axhline(y, color=FB, lw=0.9, zorder=2)
    ax.annotate(text, (x, y), xytext=(0, 1.5), textcoords="offset points", color=FB, fontsize=7.5, ha=ha,
                va="bottom")


def registered(ax, q, y, text="registered $q$"):
    ax.axvline(q, color="#9A9A9A", lw=0.5, zorder=1)
    ax.text(q, y, " " + text, fontsize=7.5, color="#6A6A6A", ha="left", va="top")


def legend(ax, items, loc, anchor=None, ncol=1):
    h = [Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1], ms=3.2, lw=0.85, mec=STYLE[k][0]) for k, _ in items]
    ax.legend(h, [t for _, t in items], loc=loc, bbox_to_anchor=anchor, frameon=False, handlelength=1.8,
              borderaxespad=0.2, labelspacing=0.25, handletextpad=0.5, ncol=ncol, columnspacing=1.2)


def panel(ax, letter, title):
    ax.set_title(f"({letter}) {title}", fontsize=8, pad=3, loc="left")


def main(path):
    fig, (a, b) = plt.subplots(2, 1, figsize=(COL, 3.8))
    fig.subplots_adjust(left=0.145, right=0.985, top=0.847, bottom=0.094, hspace=0.45)
    # Line style names the supervision and marker fill the seed (column-major: the two styles, then the two seeds).
    key = [Line2D([], [], color=INK, ls="-", lw=0.85),
           Line2D([], [], color=INK, ls=(0, (3.2, 1.6)), lw=0.85),
           Line2D([], [], color=INK, ls="none", marker="o", ms=3.4, mfc=INK, mec=INK, mew=0.75),
           Line2D([], [], color=INK, ls="none", marker="o", ms=3.4, mfc="white", mec=INK, mew=0.75)]
    fig.legend(key, ["final answer only", "with line or translate supervision", "seed 1", "seed 2"],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.56, 1.0), handlelength=2.4,
               columnspacing=1.4, handletextpad=0.5, labelspacing=0.3)

    # (a) surface code, d = 9: linear failures; seed 1 of the pooled head without line supervision lies far above.
    curves(a, SURF, QS, skip={("S-P", 1)})
    sqrt_x(a, QS, 16.5)
    a.set_ylim(8000, 18000)   # the empty band below every curve holds the head legend in one row
    a.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fallback(a, 12989, "correlated PM alone", 16.3, "right")
    registered(a, 1, 17900)
    a.text(16.3, 17900, "pooled head, final answer only,\nseed 1: did not train (402,532 to\n478,245 per million, off the axis)",
           fontsize=7.5, color=INK, ha="right", va="top", linespacing=1.0)
    legend(a, [("vote", "line vote"), ("line", "centre line"), ("pooled", "pooled head")], "lower right", ncol=3)
    panel(a, "a", "surface code, $d$ = 9, $p$ = 0.6 %")
    a.set_ylabel("failures per million", labelpad=2)
    grid(a)

    # (b) BB72: log failures, so that the joint heads' pairs and Relay-BP alone stay apart.
    curves(b, BB, QB)
    sqrt_x(b, QB, 5.15)
    b.set_yscale("log")
    b.set_ylim(3500, 40000)
    b.yaxis.set_major_locator(ticker.FixedLocator([4000, 6000, 10000, 20000, 40000]))
    b.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    b.yaxis.set_minor_formatter(ticker.NullFormatter())
    fallback(b, 5228, "full-DEM Relay-BP alone", 0.62, "left")
    registered(b, 2, 39000)
    legend(b, [("vote", "joint class score"), ("pooled", "independent outputs")], "center left", (0.0, 0.53))
    panel(b, "b", "BB72, $p$ = 0.3 %")
    b.set_ylabel("failures per million", labelpad=2)
    b.set_xlabel("routed fraction $q$ (%)", labelpad=1)
    grid(b)

    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main(os.path.join(OUT, "head_ablation.pdf"))
    print("wrote head_ablation.pdf")
