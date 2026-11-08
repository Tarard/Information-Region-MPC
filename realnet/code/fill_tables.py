"""Paper tables and text numbers, filled from the results in realnet/results_ir_v1.
The layout comes from table_template.tex, which is only read, never written: the three table environments are copied, their bodies and notes replaced, and
written as a standalone document results_ir_v1/paper_tables_x24.tex (compile with xelatex). The computed values go
to results_ir_v1/PAPER_TABLES.json.
Usage: fill_tables.py
"""

import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np
from scipy import stats

import cc_design as d

R = d.ROOT / "results_ir_v1"
TEX = Path(__file__).resolve().parent / "table_template.tex"
PART = 7            # the paper reports P-OSD on the partition with at most 7 intersections per region (4 regions)
OSD_RUNS = d.ROOT / "results_osd_v1" / "runs"
PAPER_X24 = d.ROOT / "paper_tables"   # seed-77 tables; never the x24 folder
OSD_NAMES = [f"{v}, K={k}" for v in ("OSD-1", "OSD-2", "P-OSD") for k in (1, 4)]


def load(name):
    return json.loads((R / name).read_text(encoding="utf-8"))


def ci(v):
    v = np.asarray(v, float)
    return v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def table_rows(samples, names):
    out = {}
    for n in names:
        e = np.concatenate([s["rows"][n]["errors"] for s in samples])
        g = [s["rows"][n]["gap"] for s in samples if s["rows"][n]["gap"] is not None]
        out[n] = {"solved": 100 * np.mean([s["rows"][n]["solved"] for s in samples]), "mean": e.mean(),
                  "le1": 100 * (e <= 1).mean(), "le2": 100 * (e <= 2).mean(), "gap": 1e5 * np.median(g)}
    return out


def trips(path):
    """Mean delay (timeLoss, s/veh) and TTS (veh h) of the completed trips (as ir_tables.trips)."""
    delay, tts = [], 0.0
    for t in ET.parse(path).getroot().iter("tripinfo"):
        if float(t.get("arrival", "-1")) >= 0:
            delay.append(float(t.get("timeLoss")))
            tts += float(t.get("duration")) + float(t.get("departDelay"))
    return float(np.mean(delay)), tts / 3600


def osd_label(name):
    kind, k = name.split(", K=")
    return f"{kind}, $K={k}$"


def replace_tabular(tex, label, spec, header):
    """Replace the column spec and the header row of the tabular after `label`."""
    i = tex.index(f"\\label{{{label}}}")
    a = tex.index("\\begin{tabular}", i)
    b = tex.index("\\midrule", a)
    return tex[:a] + f"\\begin{{tabular}}{{{spec}}}\n\\toprule\n{header} \\\\\n" + tex[b:]


def replace_body(tex, label, body, note):
    i = tex.index(f"\\label{{{label}}}")
    a = tex.index("\\midrule", i) + len("\\midrule")
    b = tex.index("\\bottomrule", a)
    tex = tex[:a] + "\n" + body + "\n" + tex[b:]
    m0 = tex.index("\\begin{minipage}{\\columnwidth}", i)
    m1 = tex.index("\\end{minipage}", m0)
    head = "\\begin{minipage}{\\columnwidth}\n\\footnotesize\n"
    return tex[:m0] + head + note + "\n" + tex[m1:]


def main():
    cl = load("TABLES_closed.json")
    # paper: all 50 samples timed in one run (`ir_partition_timing_onepass.py`)
    f = R / "TABLES_partition_timing_onepass50.json"
    T = load(f.name) if f.exists() else []
    med = {k: float(np.median([t["seconds"][k] for t in T])) for k in T[0]["seconds"]} if T else {}

    def sec(name):                     # "--" until the timing pass has finished
        return f"{med[name]:.2f}" if name in med else "--"
    days = sorted(cl["table1"])
    perday = {"central": {s: cl["table1"][s]["central"] for s in days}, "IR-MPC": {s: cl["table1"][s]["ir"] for s in days}}
    for name in OSD_NAMES:                       # closed-loop OSD runs (run_osd.sh); a day counts once it cleared
        perday[name] = {}
        for s in days:
            run = OSD_RUNS / f"{name.replace(', K=', '_K').replace('-', '')}_seed{s}"
            # a day counts once every trip completed (same rule as ir_common.cleared; a SUMO collision warning or a
            # teleport does not drop the day)
            if (run / "COMPLETE.json").exists() and json.loads((run / "SUMMARY.json").read_text())["gates"]["all_trips_completed"]:
                perday[name][s] = trips(run / "tripinfo.xml")
    rows1, table1 = [], {}
    for name, label1 in [("central", "Centralized MPC"), ("IR-MPC", "IR-MPC")] + [(n, osd_label(n)) for n in OSD_NAMES]:
        got = perday[name]
        if len(got) == len(days):
            dm, dh = ci([got[s][0] for s in days])
            tm, th = ci([got[s][1] for s in days])
            diff = ci([got[s][0] - perday["central"][s][0] for s in days])
            table1[name] = {"delay": [dm, dh], "tts": [tm, th], "delay_diff": list(diff)}
            cells = (f"${dm:.2f}\\pm{dh:.2f}$ & ${tm:.1f}\\pm{th:.1f}$ & "
                     + ("--" if name == "central" else f"${diff[0]:+.2f}\\pm{diff[1]:.2f}$"))
        else:
            table1[name] = {"days_done": len(got)}
            cells = f"\\multicolumn{{3}}{{c}}{{{len(got)}/{len(days)} days done}}"
        rows1.append(f"{label1} & {sec(name.replace('P-OSD', f'P-OSD-{PART}'))} & {cells} \\\\")
    t1 = "\n".join(rows1)
    note1 = (f"Comp. time: median parallel time over {len(T)} test samples, one process (IR-MPC: slowest local QP; OSD: "
             "slowest region, summed over iterations). Delay and TTS: mean $\\pm$ half-width of the 95\\% confidence "
             f"interval over {len(days)} test days; $\\Delta$ delay: paired difference to centralized MPC on the same "
             "days. OSD rows: closed-loop runs with the OSD variants of Table~\\ref{tab:distributed}.")
    t2rows, merged = [], {}
    raw = cl["table2"]
    # seed-77 segments: medium 0-2400 s, high 2400-4800 s, low 4800-7200 s, clearance after 7200 s
    for label, p in (("Medium", "medium"), ("High", "high"), ("Low", "low"), ("Clearance", "clear")):
        if p in raw:
            merged[label] = raw[p]
    for label in merged:
        r = merged[label]
        t2rows.append(f"{label} & {r['mean_error']:.3f} & {r['le1']:.1f} & {r['le2']:.1f} & {r['identical']:.1f} \\\\")
    o = raw["overall"]
    t2 = "\n".join(t2rows) + f"\n\\midrule\nOverall & {o['mean_error']:.3f} & {o['le1']:.1f} & {o['le2']:.1f} & {o['identical']:.1f} \\\\"
    note2 = ("Error: maximum absolute next-cycle green-time difference over the phases of one intersection at one sample, "
             "relative to centralized MPC. Identical executed greens: the whole-second green times match centralized MPC. "
             f"All statistics include {o['failed_local']} failed local problems ({100 * o['failed_local'] / o['n']:.2f}\\%) "
             "that apply the previous green times.")
    S = load("TABLES_samples.json")
    P = load("TABLES_partition.json")
    base = table_rows(S, ["IR-MPC", "OSD-1, K=1", "OSD-1, K=4", "OSD-2, K=1", "OSD-2, K=4"])
    part_names = [f"P-OSD-{PART}, K={k}" for k in (1, 4)]
    part = table_rows(P, part_names)
    regions = P[0]["regions"]
    label = {"IR-MPC": "IR-MPC", "OSD-1, K=1": "OSD-1, $K=1$", "OSD-1, K=4": "OSD-1, $K=4$",
             "OSD-2, K=1": "OSD-2, $K=1$", "OSD-2, K=4": "OSD-2, $K=4$"}
    for k in (1, 4):
        label[f"P-OSD-{PART}, K={k}"] = f"P-OSD, $K={k}$"
    rows3 = []
    for name, r in list(base.items()) + list(part.items()):
        rows3.append(f"{label[name]} & {r['solved']:.1f} & {r['mean']:.3f} & {r['le1']:.1f} & {r['le2']:.1f} & "
                     f"{r['gap']:.2f} & {sec(name)} \\\\")
    t3 = "\n".join(rows3)
    note3 = ("OSD-$h$: OSD with one region per intersection and an $h$-hop overlap; P-OSD: OSD on a partition "
             f"into {regions[str(PART)]} regions of {PART} intersections (recursive bisection weighted by the peak-period "
             "coupling $\\bar W$) with a one-hop overlap; "
             "$K$: iterations. Solved: percentage of successfully solved local problems (OSD: in iteration $K$); a failed "
             "IR-MPC problem applies the previous green times, a failed OSD region keeps its previous iterate, and "
             "the error counts. Error: maximum absolute next-cycle green-time difference over "
             "phases, relative to centralized MPC. Cost gap: median relative objective increase over the samples; 1.00 "
             f"denotes $1.00\\times10^{{-3}}$\\%. Comp. time: median parallel time over {len(T)} samples (slowest local "
             f"problem, summed over iterations); centralized MPC {sec('central')}~s.")
    tex = TEX.read_text(encoding="utf-8")
    tex = replace_tabular(tex, "tab:summary", "@{}lrrrr@{}",
                          "Method & \\shortstack{Comp.\\\\time (s)} & \\shortstack{Mean delay\\\\(s/veh)} & "
                          "\\shortstack{TTS\\\\(veh$\\cdot$h)} & \\shortstack{$\\Delta$ delay\\\\(s/veh)}")
    tex = replace_body(tex, "tab:summary", t1, note1)
    tex = tex.replace("\\caption{Closed-loop traffic performance and computation time of IR-MPC and centralized MPC.}",
                      "\\caption{Closed-loop traffic performance and computation time of centralized MPC, IR-MPC, "
                      "and OSD.}")
    tex = replace_body(tex, "tab:central", t2, note2)
    tex = replace_body(tex, "tab:distributed", t3, note3)
    tex = tex.replace("\\caption{IR-MPC and OSD compared with centralized MPC on 50 test samples.}",
                      f"\\caption{{IR-MPC and OSD compared with centralized MPC on {len(S)} test samples.}}")
    tables = []
    for label in ("tab:summary", "tab:central", "tab:distributed"):
        i = tex.index(f"\\label{{{label}}}")
        a = tex.rindex("\\begin{table}", 0, i)
        b = tex.index("\\end{table}", i) + len("\\end{table}")
        tables.append(tex[a:b].replace("[h!]", "[H]"))
    doc = ("\\documentclass[10pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{booktabs,float,amsmath}\n"
           "\\setlength{\\columnwidth}{8.6cm}\n\\begin{document}\n\\begin{center}\\textbf{Tables I--III, demand x2.4, "
           "mixed 60/120-s cycles (draft for review; main.tex unchanged)}\\end{center}\n\n" + "\n\n".join(tables)
           + "\n\\end{document}\n")
    (R / "paper_tables_x24.tex").write_text(doc, encoding="utf-8")
    (R / "PAPER_TABLES.json").write_text(json.dumps({
        "table1": {"methods": table1, "time_ir_serial": med.get("IR-MPC serial"), "days": days},
        "table2": merged | {"Overall": o}, "table3": base | part, "times": med,
        "trajectory": cl["trajectory"], "closed_loop": cl["closed_loop"]}, indent=1, default=float), encoding="utf-8")
    print("filled; table I:", t1.replace("\n", " "), "\ntable II:\n" + t2 + "\ntable III:\n" + t3)
    PAPER_X24.mkdir(exist_ok=True)               # standalone table files in realnet/paper_tables
    for name, env in zip(("tab_summary", "tab_central", "tab_distributed"), tables):
        env = env.replace("[H]", "[h!]")
        if name == "tab_summary":            # wide OSD confidence intervals: scale Table I to the column width
            env = env.replace("\\begin{tabular}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}", 1)
            env = env.replace("\\end{tabular}", "\\end{tabular}}", 1)
        (PAPER_X24 / f"{name}.tex").write_text(env + "\n", encoding="utf-8")
    bp = {name: {s: trips_by_period(R.parent / f"results_v1/runs/mix_seed{s}/tripinfo.xml" if name == "central"
                                    else R / f"runs/ir_seed{s}/tripinfo.xml") for s in days}
          for name in ("central", "IR-MPC")}
    by_period = {p: {"central": ci([bp["central"][s][p] for s in days]),
                     "ir_minus_central": ci([bp["IR-MPC"][s][p] - bp["central"][s][p] for s in days])}
                 for p in ("medium", "high", "low")}
    print("delay by period:", json.dumps(by_period, default=float))


def trips_by_period(path):
    """Mean delay per demand period of the scheduled departure (depart - departDelay); completed trips only."""
    out = {}
    for t in ET.parse(path).getroot().iter("tripinfo"):
        if float(t.get("arrival", "-1")) >= 0:
            out.setdefault(period(float(t.get("depart")) - float(t.get("departDelay"))), []).append(
                float(t.get("timeLoss")))
    return {k: float(np.mean(v)) for k, v in out.items()}


def period(t):
    """cc_design.period (seed-77 segments medium, high, low; clear after the departures end at 7200 s)."""
    return d.period(t)


if __name__ == "__main__":
    main()
