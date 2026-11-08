"""Coupling map of the x2.4 study (style of LYG_coupling_map_v1/code/plot_coupling_map_v2.py).

Single-column version, panels side by side: (a) high-demand period (2400-4800 s): arrows j -> m for period-mean W >= 0.5 s on the road network; node color = |C| at
eps_W = 1 s (design/ir/REGIONS.json). (b) mean W over all history samples versus road hop distance: pairs, mean,
bootstrap 95% CI, weighted exponential fit.
Inputs (read only): realnet/design/ir/W_SAMPLES.json, REGIONS.json, and the Lianyungang network.
Writes figures/out/coupling_row.{pdf,png} and coupling_map_stats.json (Fig. 3 of the paper).
"""
import sys
from pathlib import Path as _P
sys.path.insert(0, str(_P(__file__).resolve().parents[1] / "realnet" / "code"))
import cc_design as d
d.enforce_threads()

import json  # noqa: E402
from collections import deque  # noqa: E402
from pathlib import Path  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import BoundaryNorm, ListedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

import ir_common as ir  # noqa: E402

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent / "out"
BLUE, ORANGE, GRAY = "#0072B2", "#D55E00", "#B9B9B9"
plt.rcParams.update({
    "font.family": "STIXGeneral", "mathtext.fontset": "stix",
    "font.size": 8.6, "axes.labelsize": 8.6, "axes.titlesize": 8.6,
    "xtick.labelsize": 8.6, "ytick.labelsize": 8.6,
    "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
})

R = json.loads((d.ROOT / "design/ir/W_SAMPLES.json").read_text(encoding="utf-8"))
REG = json.loads((d.ROOT / "design/ir/REGIONS.json").read_text(encoding="utf-8"))["periods"]
nodes = sorted(R[0]["W"])
n = len(nodes)
idx = {m: i for i, m in enumerate(nodes)}


def matrix(rec):
    # Row m holds the couplings of intersection m at this sample. An intersection whose cycle does not start
    # at this sample has no row; it stays NaN and is left out of the mean, as in Eq. (19) and ir_offline.py.
    w = np.full((n, n), np.nan)
    for m, row in rec["W"].items():
        for j, v in row.items():
            w[idx[m], idx[j]] = v
    return w


W = np.array([matrix(r) for r in R])
per = np.array([r["period"] for r in R])
Wp = {p: np.nanmean(W[per == p], axis=0) for p in ("low", "high")}
Wall = np.nanmean(W, axis=0)
for _M in [Wall, *Wp.values()]:
    np.fill_diagonal(_M, 0.0)

st = ir.static()
adj = ir.intersection_graph(st)
hop = np.zeros((n, n), int)
for a, m in enumerate(nodes):
    dist, q = {m: 0}, deque([m])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    hop[a] = [dist.get(j, 99) for j in nodes]
xy = {}
for _, el in ET.iterparse(d.NETWORK):
    if el.tag == "junction" and el.get("id") in idx:
        xy[el.get("id")] = (float(el.get("x")), float(el.get("y")))
P = np.array([xy[m] for m in nodes])
roads = [np.array([[float(v) for v in p.split(",")] for p in r["attributes"]["shape"].split()])
         for r in st["network"]["roads"].values() if r["attributes"].get("shape")]

THR, EPS = 0.5, 1.0
KS = [1, 2, 3, 5, 8, 12, 28]
cmap = ListedColormap(plt.get_cmap("Blues")(np.linspace(0.06, 0.92, len(KS))))
norm = BoundaryNorm([0.5, 1.5, 2.5, 4, 6.5, 10, 20, 29], cmap.N)

fig = plt.figure(figsize=(3.45, 2.3))
pad = 120
x0, x1 = P[:, 0].min() - pad, P[:, 0].max() + pad
y0, y1 = P[:, 1].min() - pad, P[:, 1].max() + pad
p = "high"                       # one map: the high-demand period (the low-demand map shows the same pattern)
Wm = Wp[p]
pairs = sorted((Wm[a, b], a, b) for a in range(n) for b in range(n) if a != b and Wm[a, b] >= THR)
ax = fig.add_axes([0.01, 0.27, 0.50, 0.6])
for pts in roads:
    ax.plot(pts[:, 0], pts[:, 1], color="#D9D9D9", lw=0.55, zorder=1, solid_capstyle="round")
for w, a, b in pairs:
    ax.add_patch(FancyArrowPatch(P[b], P[a], arrowstyle="-|>,head_length=2.2,head_width=1.1",
                                 lw=0.35 + 0.28 * min(w, 4), color=BLUE if hop[a, b] == 1 else ORANGE,
                                 alpha=0.85, connectionstyle="arc3,rad=0.15", zorder=3, shrinkA=3.2, shrinkB=3.2))
k = np.array([len(REG[p]["regions"][m]) for m in nodes])
sc = ax.scatter(P[:, 0], P[:, 1], s=6, c=k, cmap=cmap, norm=norm, edgecolors="#333333", linewidths=0.45, zorder=4)
ax.set_xlim(x0, x1)
ax.set_ylim(y0, y1)
ax.set_aspect("equal")
ax.axis("off")
fig.text(0.005, 0.995, "(a)", fontsize=10.6, va="top")
leg_a = ax.legend(handles=[Line2D([], [], color=BLUE, lw=1.0), Line2D([], [], color=ORANGE, lw=1.0)],
          labels=["Adjacent", "Non-adjacent"], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
          frameon=False, fontsize=8.6, handlelength=1.2, borderaxespad=0.1, columnspacing=0.8)
cax = fig.add_axes([0.05, 0.2, 0.42, 0.03])
cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=KS)
cb.ax.tick_params(labelsize=8.6, length=1.5, width=0.5, pad=1.5)
cb.outline.set_linewidth(0.5)
cb.set_label(r"Region size $|\mathcal{R}_m|$ ($\varepsilon_W=1$ s)", fontsize=8.6, labelpad=1)

bx = fig.add_axes([0.65, 0.2, 0.34, 0.6])
off = ~np.eye(n, dtype=bool)
h, w = hop[off], Wall[off]
rng = np.random.default_rng(0)
FLOOR = 1e-3
bx.scatter(h + rng.uniform(-0.18, 0.18, h.size), np.maximum(w, FLOOR), s=1.2, color=GRAY, alpha=0.7, lw=0, zorder=1)
H = np.array(sorted(set(h)))
mean = np.array([w[h == k].mean() for k in H])
ci = np.array([np.percentile([rng.choice(w[h == k], (h == k).sum()).mean() for _ in range(2000)], [2.5, 97.5])
               for k in H])
cnt = np.array([(h == k).sum() for k in H])
ok = mean > 0
A = np.vstack([np.ones_like(H[ok], float), H[ok]]).T
coef, *_ = np.linalg.lstsq(A * np.sqrt(cnt[ok])[:, None], np.log(mean[ok]) * np.sqrt(cnt[ok]), rcond=None)
lam = -1 / coef[1]
bx.fill_between(H, np.maximum(ci[:, 0], FLOOR), np.maximum(ci[:, 1], FLOOR), color=BLUE, alpha=0.13, lw=0, zorder=2)
hh = np.linspace(1, H.max(), 50)
bx.plot(hh, np.exp(coef[0] + coef[1] * hh), color="#555555", ls=(0, (3, 2)), lw=0.8, zorder=3)
bx.plot(H, np.maximum(mean, FLOOR), color=BLUE, marker="o", markersize=2.4, lw=1.0, markeredgewidth=0.45,
        markeredgecolor="white", zorder=4)
bx.set_yscale("log")
bx.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
bx.set_ylim(8e-4, 6)
bx.set_xlim(0.5, H.max() + 0.5)
bx.set_xticks(range(1, H.max() + 1))
bx.spines[["top", "right"]].set_visible(False)
bx.tick_params(direction="out", length=1.8, width=0.5, pad=1.5)
bx.set_axisbelow(True)
bx.grid(axis="y", color="#E4E4E4", linewidth=0.4)
bx.set_xlabel(r"Hop distance $\eta$", labelpad=1.5)
bx.set_ylabel(r"$\bar W_{mj}$ (s)", labelpad=1)
fig.canvas.draw()
top_a = leg_a.get_window_extent().transformed(fig.transFigure.inverted()).y1
bx.legend(handles=[Line2D([], [], color=GRAY, marker="o", ls="", markersize=2.5),
                   Line2D([], [], color=BLUE, marker="o", markersize=3, lw=1.2, markeredgecolor="white",
                          markeredgewidth=0.4),
                   plt.Rectangle((0, 0), 1, 1, color=BLUE, alpha=0.13, lw=0),
                   Line2D([], [], color="#555555", ls=(0, (3, 2)), lw=0.8)],
          labels=["Pairs", "Mean", "95% CI", rf"$\propto e^{{-\eta/{lam:.1f}}}$"], loc="upper left", bbox_to_anchor=(bx.get_position().x0, top_a),
          bbox_transform=fig.transFigure, ncol=2, frameon=False, fontsize=8.6, handlelength=1.0, handletextpad=0.3,
          columnspacing=0.8, borderaxespad=0.0, borderpad=leg_a.borderpad, labelspacing=0.15)
fig.text(0.535, 0.995, "(b)", fontsize=10.6, va="top")

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "coupling_row.png", dpi=500)
fig.savefig(OUT / "coupling_row.pdf")
fitlog = coef[0] + coef[1] * H[ok]
ss = float(np.sum(cnt[ok] * (np.log(mean[ok]) - fitlog) ** 2))
sst = float(np.sum(cnt[ok] * (np.log(mean[ok]) - np.average(np.log(mean[ok]), weights=cnt[ok])) ** 2))

# text statistics
strong = [(a, b) for a in range(n) for b in range(n) if a != b and Wall[a, b] >= 0.25]
peak = Wp["high"]
best = None                      # strongest non-adjacent influence on one intersection versus its strongest neighbor
for a in range(n):
    far = [(peak[a, b], b) for b in range(n) if hop[a, b] > 1]
    near = [peak[a, b] for b in range(n) if hop[a, b] == 1]
    if far:
        v, b = max(far)
        if best is None or v > best[0]:
            best = (v, nodes[a], nodes[b], int(hop[a, b]), max(near) if near else 0.0)
stats = {"samples": len(R), "days": len({r["seed"] for r in R}), "pairs": int(off.sum()), "hops": H.tolist(),
         "pairs_per_hop": cnt.tolist(), "mean_W": mean.tolist(), "ci95": ci.tolist(), "lambda_hops": float(lam),
         "weighted_R2_log_mean": 1 - ss / sst if sst > 0 else None,
         "pairs_W_ge_0.25": len(strong), "nonadjacent_W_ge_0.25": sum(hop[a, b] > 1 for a, b in strong),
         "strong_pairs_W_ge_0.5": {p: int(sum(1 for a in range(n) for b in range(n) if a != b and Wp[p][a, b] >= THR))
                                   for p in ("low", "high")},
         "mean_C": {p: float(np.mean([len(REG[p]["regions"][m]) for m in nodes])) for p in REG},
         "peak_case": {"W_far": best[0], "m": best[1], "j": best[2], "hops": best[3], "W_best_neighbor": best[4]},
         "bootstrap": {"draws": 2000, "seed": 0}}
(OUT / "coupling_map_stats.json").write_text(json.dumps(stats, indent=1, default=float), encoding="utf-8")
print(json.dumps({k: v for k, v in stats.items() if k not in ("ci95",)}, default=float))
