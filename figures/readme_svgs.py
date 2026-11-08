"""SVG versions of Figs. 1 and 3 for the README, in a light and a dark variant.

GitHub shows the light SVG in light mode and the dark SVG in dark mode (<picture> with prefers-color-scheme). The
paper figures (PNG, PDF) are not changed: this script runs the figure scripts in memory with SVG output, and for the
dark variant with light text and lines on a transparent background.
Usage (from figures/): python readme_svgs.py  -> out/{paradigms,coupling_row}{,-dark}.svg
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "realnet" / "code"))
import cc_design  # noqa: E402  (sets the BLAS thread variables before NumPy is loaded)
cc_design.enforce_threads()

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "out"

# colors of the light figures and their dark-mode replacements (GitHub dark palette)
DARK = {
    '"#2E4B7A"': '"#5B8BD0"',   # NAVY: nodes and scope boxes
    '"#D0D0D0"': '"#3D444D"',   # ROAD
    '"#6E6E6E"': '"#9DA7B3"',   # ARROW
    '"#C5D5EA"': '"#2F4A6D"',   # HALO
    '"#EFEFEF"': '"#161B22"',   # SHADE
    '"#D9D9D9"': '"#3D444D"',   # roads of the network map
    '"#333333"': '"#C9D1D9"',   # node edges
    '"#555555"': '"#9DA7B3"',   # fitted curve
    '"#E4E4E4"': '"#30363D"',   # grid
    '"#B9B9B9"': '"#6E7681"',   # pair dots
    '"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white"':
        '"figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none"',
}
DARK_RC = {"text.color": "#E6EDF3", "axes.edgecolor": "#C9D1D9", "axes.labelcolor": "#E6EDF3",
           "xtick.color": "#C9D1D9", "ytick.color": "#C9D1D9", "legend.labelcolor": "#E6EDF3",
           "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
           "savefig.transparent": True}


def run(script, name, dark):
    src = (HERE / script).read_text(encoding="utf-8")
    if dark:
        for old, new in DARK.items():
            src = src.replace(old, new)
        # the scripts set their own rcParams first; apply the dark text colors after them
        src = src.replace("plt.rcParams.update(", "_rc_dark(); plt.rcParams.update(", 1)
        src = src.replace("\n})\n", "\n})\n_rc_dark()\n", 1)
    saved = []
    orig = Figure.savefig

    def capture(self, fname, *a, **k):
        if str(fname).endswith(".png") and not saved:
            saved.append(self)
    Figure.savefig = capture
    try:
        g = {"__name__": "__main__", "__file__": str(HERE / script), "_rc_dark": (lambda: plt.rcParams.update(DARK_RC))}
        sys.argv = [script, str(OUT / "_tmp")]
        cwd = os.getcwd()
        os.chdir(HERE)
        try:
            exec(compile(src, script, "exec"), g)
        finally:
            os.chdir(cwd)
    finally:
        Figure.savefig = orig
    fig = saved[0]
    if dark:
        plt.rcParams.update(DARK_RC)
        fig.patch.set_alpha(0)
        for ax in fig.axes:
            ax.patch.set_alpha(0)
    out = OUT / f"{name}{'-dark' if dark else ''}.svg"
    fig.savefig(out, format="svg", bbox_inches="tight", pad_inches=0.04, transparent=dark,
                facecolor="none" if dark else "white")
    if os.environ.get("README_SVG_PREVIEW"):        # optional PNG preview on the GitHub page colors
        bg = "#0d1117" if dark else "white"
        fig.savefig(OUT / f"_preview_{name}{'-dark' if dark else ''}.png", dpi=110, facecolor=bg, transparent=False,
                    bbox_inches="tight")
    plt.close("all")
    plt.rcdefaults()
    print("written", out.name)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    for dark in (False, True):
        run("fig_paradigms.py", "paradigms", dark)
        run("fig_coupling_s77_row.py", "coupling_row", dark)
