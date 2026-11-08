"""Fig. 1: centralized, iterative distributed, and information-region MPC on a 3 x 4 grid.

Redraws paradigms.png. Writes paradigms.{png,pdf} into the folder given as the first argument (default: this folder).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)
NAVY, ROAD, ARROW, RED, HALO, SHADE = "#2E4B7A", "#D0D0D0", "#6E6E6E", "#B85540", "#C5D5EA", "#EFEFEF"
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 11})

ROWS, COLS, DX = 3, 4, 1.0
R_NODE, BOX = 0.17, 0.29                    # node radius, half width of a dashed scope box


def grid(ax, x0, y0):
    """Roads and the node centres of one panel; returns {(row, col): (x, y)}."""
    pos = {(r, c): (x0 + c * DX, y0 - r * DX) for r in range(ROWS) for c in range(COLS)}
    for r in range(ROWS):
        ax.plot([pos[r, 0][0], pos[r, COLS - 1][0]], [pos[r, 0][1]] * 2, color=ROAD, lw=2.2, zorder=1)
    for c in range(COLS):
        ax.plot([pos[0, c][0]] * 2, [pos[0, c][1], pos[ROWS - 1, c][1]], color=ROAD, lw=2.2, zorder=1)
    return pos


def node(ax, xy, halo=False, label=None):
    if halo:
        ax.add_patch(Circle(xy, R_NODE * 1.45, color=HALO, zorder=2))
    ax.add_patch(Circle(xy, R_NODE, color=NAVY, zorder=3))
    if label:
        ax.text(*xy, label, color="white", ha="center", va="center", fontsize=LABEL, style="italic", zorder=4)


def scope(ax, cx, cy, half_w, half_h, lw=1.2):
    ax.add_patch(FancyBboxPatch((cx - half_w, cy - half_h), 2 * half_w, 2 * half_h,
                                boxstyle="round,pad=0,rounding_size=0.08", fill=False, ec=NAVY, lw=lw,
                                ls=(0, (3, 2)), zorder=2))


def exchange(ax, a, b):
    """Double-headed data exchange between two neighbouring nodes."""
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="<|-|>,head_length=3.5,head_width=2", color=ARROW, lw=1.1,
                                 shrinkA=R_NODE * 72 * 1.12, shrinkB=R_NODE * 72 * 1.12, zorder=2.5))


TITLE, FORMULA, LEGEND, LABEL = 28, 24, 24, 24   # about 2x the earlier sizes, readable at column width
fig, ax = plt.subplots(figsize=(12.25, 5.3))
ax.set_xlim(0, 12.25)
ax.set_ylim(0, 5.3)
ax.set_aspect("equal")
ax.axis("off")
GX = (0.55, 4.65, 8.75)                     # left node column of the three panels
GY = 3.05                                    # top node row
TY, FY = 5.0, 4.15                           # panel titles and formulas

# (a) centralized MPC: one QP over all intersections
pos = grid(ax, GX[0], GY)
for xy in pos.values():
    node(ax, xy)
scope(ax, GX[0] + 1.5 * DX, GY - DX, 1.5 * DX + BOX + 0.04, DX + BOX + 0.04, lw=1.5)
ax.text(GX[0] + 1.5 * DX, TY, "(a) Centralized MPC", ha="center", fontsize=TITLE)
ax.text(GX[0] + 1.5 * DX, FY, r"$\min_z\, J(z;p)$" "\n" r"s.t. all constraints", ha="center", va="center",
        fontsize=FORMULA, linespacing=1.1)

# (b) iterative distributed MPC: one QP per intersection, exchanges with neighbours, K rounds
pos = grid(ax, GX[1], GY)
for (r, c), xy in pos.items():
    node(ax, xy)
    scope(ax, *xy, BOX, BOX)
    if c + 1 < COLS:
        exchange(ax, xy, pos[r, c + 1])
    if r + 1 < ROWS:
        exchange(ax, xy, pos[r + 1, c])
ax.text(GX[1] + 1.5 * DX, TY, "(b) Iterative distributed", ha="center", fontsize=TITLE)
ax.text(GX[1] + 1.5 * DX, FY, r"$\min_{z_m}\, J_m(z_m;p_m)+\psi_m(z_m)$" "\n" r"$K$ iterations", ha="center",
        va="center", fontsize=FORMULA, linespacing=1.1)

# (c) IR-MPC: one QP per intersection; m reads current data of its information region only
ax.add_patch(FancyBboxPatch((GX[2] - 0.38, GY - 2 * DX - 0.38), 3 * DX + 0.76, 2 * DX + 0.76,
                            boxstyle="round,pad=0,rounding_size=0.08", color=SHADE, zorder=0))
pos = grid(ax, GX[2], GY)
m = (1, 1)
region = {(0, 1), (1, 2), (2, 3)}           # intersections whose current data m reads
for key, xy in pos.items():
    if key != m:
        node(ax, xy, halo=key in region)
        scope(ax, *xy, BOX, BOX)
mx, my = pos[m]
for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):  # exchanged flows on the boundary of m
    ax.plot([mx, mx + 0.5 * dx], [my, my + 0.5 * dy], color=RED, lw=3.2, zorder=2.6, solid_capstyle="butt")
scope(ax, mx, my, 0.58, 0.58, lw=1.5)
node(ax, (mx, my), label="m")
for key, rad in (((0, 1), 0.15), ((1, 2), 0.15), ((2, 3), -0.25)):
    ax.add_patch(FancyArrowPatch(pos[key], (mx, my), arrowstyle="-|>,head_length=4,head_width=2.2", color=ARROW,
                                 lw=1.1, connectionstyle=f"arc3,rad={rad}", shrinkA=R_NODE * 72 * 1.5,
                                 shrinkB=R_NODE * 72 * 1.12, zorder=2.7))
ax.text(GX[2] + 1.5 * DX, TY, "(c) IR-MPC", ha="center", fontsize=TITLE)
ax.text(GX[2] + 1.5 * DX, FY, r"$\min_{z_m,\,s_m}\, J_m(z_m;p_m)+\widehat V_m(s_m;p_{\mathcal{R}_m})$" "\n" r"each $m$", ha="center",
        va="center", fontsize=FORMULA, linespacing=1.1)

# legend
handles = [Line2D([], [], marker="o", ls="", color=NAVY, markersize=18),
           Rectangle((0, 0), 1, 1, fill=False, ec=NAVY, ls=(0, (3, 2)), lw=1.2),
           Line2D([], [], color=ARROW, lw=1.1, marker=">", markersize=6)]
fig.legend(handles, ["decision variables", "QP scope", "information exchange"], loc="lower center", ncol=3,
           frameon=False, fontsize=LEGEND, handlelength=1.4, columnspacing=1.4, handletextpad=0.4,
           bbox_to_anchor=(0.5, -0.02))
fig.subplots_adjust(0, 0.02, 1, 1)
fig.savefig(OUT / "paradigms.png", dpi=200, bbox_inches="tight", pad_inches=0.03)
fig.savefig(OUT / "paradigms.pdf", bbox_inches="tight", pad_inches=0.03)
print("written", OUT / "paradigms.png")
