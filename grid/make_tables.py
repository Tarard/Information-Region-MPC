"""Tables of the IR-MPC trial (trial_flow/IR_TRIAL_N*.json) for the paper: network, computation time, accuracy and
per-day values. Writes results/TRIAL_TABLES.md and results/trial_*.csv.
Usage: make_tables.py [N ...]   (default: every IR_TRIAL_N*.json in trial_flow/)
"""
import csv
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
os.environ.setdefault("NS_ROOT", str(HERE / "runs"))   # same folder as run_all.sh
os.environ.setdefault("NS_DEMAND_FACTOR", "0.6")
sys.path.insert(0, str(HERE / "code"))
import ns_design as d  # noqa: E402

d.enforce_threads()
import pickle  # noqa: E402

import numpy as np  # noqa: E402

import grid_net  # noqa: E402
import ns_offline as no  # noqa: E402

ROOT = Path(os.environ["NS_ROOT"])
OUT = HERE / "results"


def qp_size(n):
    """Equality and inequality rows of the centralized QP at the reference state (t = 4200 s)."""
    net = grid_net.load(n)
    with open(ROOT / f"trial_payload_N{n}.pkl", "rb") as h:
        pbar = pickle.load(h)["pbar"]
    model = no.model_at(net, no.labels_of(net), pbar, 4200)
    return model["E"].shape[0], model["G"].shape[0]


def med(rows, key):
    return float(np.median([r[key] for r in rows]))


def main(sizes):
    tables = {"network": [], "time": [], "accuracy": [], "days": []}
    for n in sizes:
        r = json.loads((ROOT / f"IR_TRIAL_N{n}.json").read_text(encoding="utf-8"))
        s = json.loads((ROOT / f"work/N{n}/NET_STATS.json").read_text(encoding="utf-8"))
        ev, con, reg = r["evaluation"], r["construction"], r["regions"]
        neq, nin = qp_size(n)
        fixed = any(day.get("states_only") for day in r["days"].values())
        stop = max((day.get("stopped_at") or 0) for day in r["days"].values())
        rows, cols = d.SIZES[n]
        tables["network"].append({
            "Intersections": n, "Grid": f"{rows} x {cols}", "Links": s["links_all"],
            "Stages 4/3/2": "/".join(str(s["stages"][k]) for k in ("4", "3", "2")),
            "Cycles 60/120 s": "/".join(str(s["cycles"][k]) for k in ("60", "120")),
            "Y_m median/p90/max": f"{s['Y']['median']:.2f}/{s['Y']['p90']:.2f}/{s['Y']['max']:.2f}",
            "Peak entry flow (veh/s)": round(s["peak_entry_flow_veh_per_second"], 1),
            "States": "fixed-time plan" if fixed else
                      f"centralized MPC days (to {stop} s)" if stop else "centralized MPC days"})
        c, p = med(ev, "central_seconds"), med(ev, "ir_parallel_seconds")
        tables["time"].append({
            "Intersections": n, "Central variables": ev[0]["central_variables"],
            "Central equalities": neq, "Central inequalities": nin,
            "Central time median (s)": round(c, 3),
            "Central time min-max (s)": f"{min(e['central_seconds'] for e in ev):.2f}-"
                                         f"{max(e['central_seconds'] for e in ev):.2f}",
            "IR-MPC parallel (s)": round(p, 4), "Speed-up": round(c / p),
            "IR-MPC serial, all intersections (s)": round(med(ev, "ir_serial_seconds_estimate_all"), 2),
            "Local QP variables median": int(r["local_variables"]["median"]),
            "Region size mean/max": f"{reg['mean']:.2f}/{reg['max']}",
            "Controllers": con["constructed"],
            "Construction median (s)": round(con["seconds_median"], 1)})
        for kind in ("history", "test"):
            rows_k = [e for e in ev if e["kind"] == kind]
            tables["accuracy"].append({
                "Intersections": n, "Days": f"{kind} ({len(rows_k)})",
                "Local QPs solved": f"{100 * np.mean([e['solved'] for e in rows_k]):.1f}%",
                "Mean error (s)": round(float(np.mean([e["error_mean"] for e in rows_k])), 3),
                "Error <= 1 s": f"{100 * np.mean([e['error_le_1'] for e in rows_k]):.1f}%",
                "Error <= 2 s": f"{100 * np.mean([e['error_le_2'] for e in rows_k]):.1f}%",
                "Max error (s)": round(max(e["error_max"] for e in rows_k), 2)})
        for e in ev:
            tables["days"].append({
                "Intersections": n, "Day": f"{e['kind']} {e['seed']}",
                "Central (s)": round(e["central_seconds"], 3), "Central accepted": e["central_valid"],
                "IR-MPC parallel (s)": round(e["ir_parallel_seconds"], 4),
                "IR-MPC median local (s)": round(e["ir_median_seconds"], 4),
                "IR-MPC serial (s)": round(e["ir_serial_seconds_estimate_all"], 2),
                "Solved": f"{100 * e['solved']:.1f}%", "Mean error (s)": round(e["error_mean"], 3),
                "Error <= 1 s": f"{100 * e['error_le_1']:.1f}%", "Error <= 2 s": f"{100 * e['error_le_2']:.1f}%",
                "Max error (s)": round(e["error_max"], 2)})
    titles = {"network": "Table 1. Networks", "time": "Table 2. Computation time at t = 4200 s (median over 6 days)",
              "accuracy": "Table 3. First-decision accuracy of IR-MPC against centralized MPC at t = 4200 s",
              "days": "Table 4. Every day"}
    OUT.mkdir(exist_ok=True)
    lines = []
    for key, rows in tables.items():
        with open(OUT / f"trial_{key}.csv", "w", newline="", encoding="utf-8") as h:
            w = csv.DictWriter(h, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        lines += [f"## {titles[key]}", "", "| " + " | ".join(rows[0]) + " |", "|" + "---|" * len(rows[0])]
        lines += ["| " + " | ".join(str(v) for v in row.values()) + " |" for row in rows] + [""]
    (OUT / "TRIAL_TABLES.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    args = [int(a) for a in sys.argv[1:]]
    main(args or sorted(int(p.stem.split("_N")[1]) for p in ROOT.glob("IR_TRIAL_N*.json")))
