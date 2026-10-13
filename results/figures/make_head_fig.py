"""The figure of the output-head comparison (Section VI), drawn at its printed size as a vector PDF.

    python results/figures/make_head_fig.py   (writes head_ablation.pdf next to this file)

Failures per million test shots of each pair against the routed fraction q, for every head and seed of the
comparison: (a) the surface code at d = 9 and p = 0.6 % with correlated PM as the fallback, (b) BB72 at p = 0.3 % with
full-DEM Relay-BP. Colour and marker name the head, the line style its supervision (solid: the final answer
alone; dashed: with the line or translate supervision), and the marker fill names the seed (filled: seed 1, hollow:
seed 2). The visual language is make_figs.py's: STIX serif at 7.5-8 pt, ConvLines' head vermillion, every fallback
blue.

Every number is copied from head/results/surface/pair_curves.csv and head/results/bb72/pair_curves.csv
(column per_million),
the fallbacks alone from the same files' analyses (13,080 and 5,382 per million, A1_RESULT.md sections 1 and 2).
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

CL, FB = "#D55E00", "#0072B2"             # ConvLines' structure-aware head, every fallback
PO, CE = "#CC79A7", "#009E73"             # pooled head (independent outputs on BB72), centre line
GRID, INK = "#E6E6E6", "#333333"
# Okabe-Ito: the four colours pass the dataviz validator (lightness band, chroma floor, all-pairs CVD and the
# normal-vision floor); pink and green sit in the 6-8 CVD band, so every head also has its own marker.

# q in percent; per million test shots (1,000,000 surface shots, 500,000 BB72 shots), seeds 1 and 2.
QS = [0, 0.5, 1, 2, 4, 8, 16]
SURF = {  # arm: (head, supervised, [seed 1], [seed 2])
    "S-VA": ("vote", True, [15252, 14083, 13239, 12136, 11575, 12034, 12694],
             [13975, 12909, 12217, 11522, 11405, 12071, 12749]),
    "S-V": ("vote", False, [10638, 10094, 9942, 10203, 11146, 12150, 12841],
            [10881, 10314, 10069, 10249, 11093, 12149, 12850]),
    "S-L": ("line", False, [17088, 15733, 14628, 13196, 12075, 12175, 12726],
            [15059, 13900, 13076, 12103, 11566, 12060, 12728]),
    "S-P": ("pooled", False, [478846, 476451, 474111, 469398, 459991, 441379, 403144],   # seed 1 did not train
            [13750, 12745, 12017, 11274, 11281, 12075, 12770]),
    "S-PA": ("pooled", True, [14917, 13871, 12981, 12015, 11556, 12100, 12731],
             [15719, 14540, 13610, 12510, 11868, 12096, 12715]),
}
QB = [0, 0.25, 0.5, 1, 2, 3, 5]
BB = {
    "B-JT": ("vote", True, [7044, 5804, 5048, 4442, 4268, 4432, 4750], [6864, 5688, 4950, 4374, 4306, 4496, 4780]),
    "B-J": ("vote", False, [7046, 5880, 5096, 4408, 4310, 4430, 4754], [7110, 6008, 5144, 4370, 4262, 4476, 4766]),
    "B-P": ("pooled", False, [34048, 32082, 30056, 25876, 18524, 12088, 6210],
            [35360, 33162, 31086, 27130, 19644, 13154, 6196]),
    "B-PA": ("pooled", True, [28146, 26058, 24014, 20256, 12816, 7488, 4736],
             [26216, 24268, 22234, 18418, 11046, 6378, 4742]),
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
    ax.text(q, y, " " + text, fontsize=7, color="#6A6A6A", ha="left", va="top")


def legend(ax, items, loc, anchor=None):
    h = [Line2D([], [], color=STYLE[k][0], marker=STYLE[k][1], ms=3.2, lw=0.85, mec=STYLE[k][0]) for k, _ in items]
    ax.legend(h, [t for _, t in items], loc=loc, bbox_to_anchor=anchor, frameon=False, handlelength=1.8,
              borderaxespad=0.2, labelspacing=0.25, handletextpad=0.5)


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
    a.set_ylim(9000, 18000)
    a.yaxis.set_major_locator(ticker.MultipleLocator(2000))
    a.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fallback(a, 13080, "correlated PM alone", 16.3, "right")
    registered(a, 1, 17900)
    a.text(16.3, 17900, "pooled head, final answer only,\nseed 1: did not train (403,144 to\n478,846 per million, off the axis)",
           fontsize=7, color=INK, ha="right", va="top", linespacing=1.0)
    legend(a, [("vote", "line vote"), ("line", "centre line"), ("pooled", "pooled head")], "lower right")
    panel(a, "a", "Surface code, $d=9$, $p=0.6\\%$")
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
    fallback(b, 5382, "full-DEM Relay-BP alone", 0.62, "left")
    registered(b, 2, 39000)
    legend(b, [("vote", "joint class score"), ("pooled", "independent outputs")], "center left", (0.0, 0.53))
    panel(b, "b", "BB72, $p=0.3\\%$")
    b.set_ylabel("failures per million", labelpad=2)
    b.set_xlabel("routed fraction $q$ (%)", labelpad=1)
    grid(b)

    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main(os.path.join(OUT, "head_ablation.pdf"))
    print("wrote head_ablation.pdf")
