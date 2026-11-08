"""Simulation scenario figure (Fig. 2), written to figures/out/scenario.{pdf,png}.

(a) Lianyungang network from Tan et al. (lyg_network.jpg);
(b) three typical phase combinations, read from lyg_safe.net.xml and cc120_signals.json
    (protected 'G' solid, permitted 'g' dashed, U-turns omitted);
(c), (d) planned departure rates of the 20 historical (arrival seeds 1-20) and 10 test days of the seed-77 study
    (realnet/demand); written to figures/out/scenario.{pdf,png}
    (same data and style as paper_tables_v1/code/plot_demand_days.py).
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.path import Path as MPath

from net_movements import load

HERE = Path(__file__).resolve().parent
OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "out"
DEMAND = HERE.parent / "realnet" / "demand"
# One intersection per typical phase combination (numbers as in panel (a)).
SHOWN = [(13, "4 phases"), (9, "3 phases"), (7, "2 phases")]

plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.linewidth": 0.6, "lines.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
})
PROT, PERM, ROAD = "#1f4e9a", "#7a8aa6", "#d9d9d9"
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#B9B9B9"
W, H = 3.5, 3.05  # single column: (a), (b) on the first row; (c), (d) on the second


def _approach(heading):
    """Number of clockwise quarter turns from a northbound approach."""
    return int(((heading + 45) % 360) // 90)


def _rot(p, k):
    x, y = p
    for _ in range(k):
        x, y = y, -x
    return x, y


# Northbound templates in a box spanning [-1, 1]^2 (right-hand traffic).
TEMPLATE = {
    "s": [((0.18, -0.95), MPath.MOVETO), ((0.18, 0.95), MPath.LINETO)],
    "l": [((0.18, -0.95), MPath.MOVETO), ((0.18, 0.05), MPath.LINETO),
          ((0.18, 0.30), MPath.CURVE3), ((-0.95, 0.30), MPath.CURVE3)],
    "r": [((0.38, -0.95), MPath.MOVETO), ((0.38, -0.55), MPath.LINETO),
          ((0.38, -0.32), MPath.CURVE3), ((0.95, -0.32), MPath.CURVE3)],
}


def phase_moves(info, stage):
    """{(quarter_turns, dir): protected} for one stage, U-turns dropped."""
    moves = {}
    for i, ch in enumerate(stage["green_state"]):
        link = info["links"].get(i)
        if ch not in "Gg" or link is None or link["dir"] not in TEMPLATE:
            continue
        key = (_approach(link["heading"]), link["dir"])
        moves[key] = moves.get(key, False) or ch == "G"
    return moves


def draw_box(ax, x0, y0, size, moves):
    s = size / 2
    cx, cy = x0 + s, y0 + s
    ax.add_patch(Rectangle((x0, y0), size, size, fc="white", ec="0.35", lw=0.5))
    w = 0.62 * s
    ax.add_patch(Rectangle((x0, cy - w), size, 2 * w, fc=ROAD, ec="none", lw=0))
    ax.add_patch(Rectangle((cx - w, y0), 2 * w, size, fc=ROAD, ec="none", lw=0))
    for (k, d), prot in sorted(moves.items(), key=lambda m: m[1]):
        verts = [_rot(p, k) for p, _ in TEMPLATE[d]]
        codes = [c for _, c in TEMPLATE[d]]
        path = MPath([(cx + 0.9 * s * x, cy + 0.9 * s * y) for x, y in verts], codes)
        ax.add_patch(FancyArrowPatch(path=path, arrowstyle="-|>,head_length=1.8,head_width=1.0",
                                     color=PROT if prot else PERM, lw=0.75 if prot else 0.6,
                                     ls="-" if prot else (0, (2, 1.2)), shrinkA=0, shrinkB=0))


def axes_in(x, y, w, h):
    """Axes placed in inches from the lower-left corner of the figure."""
    return plt.gcf().add_axes([x / W, y / H, w / W, h / H])


FS = 11      # text size in (b)-(d), about 1.5x the earlier 7-8 pt
FT = 11      # panel tags
FD = 8       # text in (c), (d), close to the 10-pt body text after printing


def panel_network():
    ax = axes_in(0.0, 1.55, 1.9, 1.35)
    ax.imshow(plt.imread(HERE / "lyg_network.jpg"))
    ax.axis("off")
    ax.set_title("(a)", loc="left", fontsize=FT, pad=2)


def panel_phases():
    data = load()
    fig_id = json.loads((HERE / "tls_to_fig_id.json").read_text(encoding="utf-8"))
    tls_of = {v: k for k, v in fig_id.items()}
    ax = axes_in(1.98, 1.5, 1.52, 1.45)
    size, gap, row_h, x_lab = 1.0, 0.14, 1.75, 0.0
    for r, (num, label) in enumerate(SHOWN):
        info = data[tls_of[num]]
        y0 = -r * row_h
        ax.text(x_lab, y0 + size + 0.08, label, ha="left", va="bottom", fontsize=FS - 2)
        for k, stage in enumerate(info["stages"]):
            x0 = x_lab + k * (size + gap)
            draw_box(ax, x0, y0, size, phase_moves(info, stage))
    x_end = x_lab + 4 * (size + gap) - gap
    y_leg = -2 * row_h - 0.55
    ax.plot([0.0, 0.45], [y_leg] * 2, color=PROT, lw=0.9)
    ax.text(0.52, y_leg, "prot.", va="center", fontsize=FS - 2)
    ax.plot([2.2, 2.65], [y_leg] * 2, color=PERM, lw=0.75, ls=(0, (2, 1.2)))
    ax.text(2.72, y_leg, "perm.", va="center", fontsize=FS - 2)
    ax.set_xlim(-0.05, x_end + 0.05)
    ax.set_ylim(y_leg - 0.35, size + 0.55)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("(b)", loc="left", fontsize=FT, pad=2)


def panel_demand():
    edges = np.arange(0, 7200 + 300, 300)
    mid = (edges[:-1] + edges[1:]) / 7200
    days = []
    for seed in list(range(1, 21)) + [101, 102, 103, 104, 106, 107, 108, 109, 110, 111]:
        item = {"seed": seed, "role": "history" if seed <= 20 else "test"}
        root = ET.parse(DEMAND / f"day{seed}.rou.xml").getroot()
        departures = np.array([float(v.get("depart")) for v in root.iter("vehicle")])
        counts = np.histogram(departures, edges)[0]
        assert counts.sum() == len(departures)
        days.append({**item, "rate": counts * 12})
    groups = [([d for d in days if d["role"] == "history"], BLUE, "(c) Historical", 20),
              ([d for d in days if d["role"] != "history"], ORANGE, "(d) Test", 10)]
    assert len(groups[0][0]) == 20 and len(groups[1][0]) == 10
    ymax = np.max([d["rate"].max() for d in days]) / 1000 * 1.08
    for r, (group, color, title, n) in enumerate(groups):
        ax = axes_in(0.42 + r * 1.58, 0.42, 1.45, 0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(direction="out", length=2.5, width=0.7, pad=1.5, labelsize=FD)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E4E4E4", linewidth=0.4)
        for d in group:
            ax.plot(mid, d["rate"] / 1000, color=GRAY, linewidth=0.5, alpha=0.8)
        ax.plot(mid, np.mean([d["rate"] for d in group], axis=0) / 1000, color=color, linewidth=1.3)
        ax.set(xlim=(0, 2.0), ylim=(0, ymax))
        ax.set_xticks([0, 1, 2]); ax.set_yticks([0, 5, 10])
        if r == 1:
            ax.set_yticklabels([])
        ax.set_xlabel("Time (h)", labelpad=1, fontsize=FD)
        ax.legend([Line2D([], [], color=GRAY), Line2D([], [], color=color, linewidth=1.3)], [f"Day ({n})", "Mean"],
                  loc="lower right", bbox_to_anchor=(1.0, 0.02), ncol=1, frameon=False, fontsize=FD,
                  handlelength=1.0, handletextpad=0.3, labelspacing=0.1, borderaxespad=0.0, borderpad=0.1)
        ax.set_title(title, loc="left", fontsize=FD + 1, pad=2)
    plt.gcf().text(0.06 / W, 0.82 / H, "Demand\n($10^3$ veh/h)", rotation=90, ha="center", va="center",
                   fontsize=FD, linespacing=1.0)


def main():
    fig = plt.figure(figsize=(W, H))
    panel_network()
    panel_phases()
    panel_demand()
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"scenario.{ext}", dpi=350, bbox_inches="tight", pad_inches=0.02)


if __name__ == "__main__":
    main()
