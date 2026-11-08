"""Closed-loop cost J_cl on the SUMO trajectories (Table II, column Delta J_cl).

Definition (same terms and weights as cc_model.build / direct_cost, evaluated on what happened in SUMO):
  J_state = sum over updates t = 60, 120, ... of
            sum_e 0.5 n_e(t)^2 / cap_e  +  sum_entries 0.5 w_e(t)^2 / cap_e        (measured n0, backlog in p(t))
  J_green = sum over executed cycles c of intersection m (start s before the window end) of 0.5 mu_g ||g_c - g_{c-1}||^2
  J_red   = sum over the same cycles of sum_movements 0.5 mu_R beta(s) (C_m - gamma^T g_c)^2, weight 1 per cycle
  J = J_state + J_green + J_red.
g_c = executed integer greens, read from previous_green in p at s + 60 (the running or just ended cycle started at s).
g_{-1} = previous_green at t = 0 (the same plan for every method). beta(s) = the model's red-delay weight at clock s
(historical arrival row of s), taken from cc_model.build at s (it depends on the clock only, not on the state).
Every run continues until the network is empty (last update 7860-11280 s). Two windows:
  full   : all updates of the run, all cycles that start before its last update (main result; the state term is 0
           once the network is empty, so the window is common in effect for J_state);
  common : updates t <= 7800 s and cycles that start before 7800 s (every run reaches 7800 s).
After the last update the network still holds a few vehicles for < 60 s; their number is reported.
"""

import json
import math
import sys
from collections import defaultdict

import cc_design as d
d.enforce_threads()

import numpy as np
from scipy import stats

import ir_common as ir

OUT = d.ROOT / "results_review"
METHODS = {"Centralized": ("results_v1/runs", "mix_seed{}"), "IR-MPC": ("results_ir_v1/runs", "ir_seed{}")}
for name in ("OSD1_K1", "OSD1_K4", "OSD2_K1", "OSD2_K4", "POSD_K1", "POSD_K4"):
    METHODS[name.replace("_", "-").replace("OSD1", "OSD-1").replace("OSD2", "OSD-2").replace("POSD", "P-OSD")] = \
        ("results_osd_v1/runs", name + "_seed{}")
DAYS = d.TEST_DAYS
T_COMMON = 7800


def run_dir(method, day):
    root, pattern = METHODS[method]
    return d.ROOT / root / pattern.format(day)


def clock_tables(st, times):
    """Per update clock: capacity, deciding signals and their cycle-0 red-delay rows (built once per clock)."""
    snap = ir.snapshot_of(DAYS[0], 0)
    tables, ref = {}, None
    for t in times:
        snap = dict(snap, time_seconds=int(t))
        m = ir.model_at(st, int(t), snap)
        if ref is None:
            ref = m
        elif m["param_labels"] != ref["param_labels"]:
            raise SystemExit(f"Parameter labels differ at t={t}.")
        deciding = set(m["metadata"]["deciding_signals"])
        rows = [r for r in m["red_delay_rows"] if r["cycle"] == 0 and r["signal"] in deciding]
        if any(abs(r["coverage_weight"] - 1) > 1e-12 for r in rows):
            raise SystemExit("A deciding cycle is not fully inside the horizon.")
        tables[int(t)] = {"deciding": sorted(deciding), "rows": rows}
    return ref, tables


def run_cost(method, day, ref, tables, labels, window):
    run = run_dir(method, day)
    if json.loads((run / "MODEL_LABELS.json").read_text(encoding="utf-8"))["param_labels"] != labels:
        raise SystemExit(f"{run}: parameter labels differ from the model.")
    z = np.load(run / "snapshots.npz")
    times = [int(t) for t in z["t"]]
    if times != list(range(0, times[-1] + 1, d.STEP)):
        raise SystemExit(f"{run}: update clocks are not every {d.STEP} s.")
    last = times[-1]
    if window == "common":
        times = [t for t in times if t <= T_COMMON]
    P = {t: p for t, p in zip(times, z["p"])}
    pi, cap = ref["parameter_indices"], ref["metadata"]["capacity"]
    n_idx = np.asarray([pi["n0"][e] for e in sorted(pi["n0"])])
    n_cap = np.asarray([cap[e] for e in sorted(pi["n0"])], dtype=float)
    w_idx = np.asarray([pi["backlog"][e] for e in sorted(pi["backlog"])])
    w_cap = np.asarray([cap[e] for e in sorted(pi["backlog"])], dtype=float)
    mu_g, mu_r = ref["metadata"]["green_change_weight"], ref["metadata"]["mu"]
    phases = PHASES

    state_link = state_entry = 0.0
    per_period = defaultdict(float)
    for t in times[1:]:
        a = 0.5 * float(np.sum(P[t][n_idx] ** 2 / n_cap))
        b = 0.5 * float(np.sum(P[t][w_idx] ** 2 / w_cap))
        state_link += a
        state_entry += b
        per_period[d.period(t)] += a + b

    green = red = 0.0
    prev = {m: np.asarray([P[0][pi["previous_green"][m, ph]] for ph in phases[m]]) for m in phases}
    cycles = 0
    for s in times[:-1]:                      # cycle starts 0 ... 7980
        for m in tables[s]["deciding"]:
            g = np.asarray([P[s + d.STEP][pi["previous_green"][m, ph]] for ph in phases[m]])
            green += 0.5 * mu_g * float(np.sum((g - prev[m]) ** 2))
            prev[m] = g
            cycles += 1
        for r in tables[s]["rows"]:
            g = np.asarray([P[s + d.STEP][pi["previous_green"][r["signal"], ph]] for ph in phases[r["signal"]]])
            rd = r["cycle_seconds"] - r["fixed_service_seconds"] - float(np.asarray(r["gamma"]) @ g)
            red += 0.5 * mu_r * r["beta"] * rd ** 2
    trace = [json.loads(line) for line in open(run / "trace.jsonl", encoding="utf-8")]
    at_end = next(r for r in trace if r["time_seconds"] == last)
    after = [r for r in trace if r["time_seconds"] > last]
    return {"state_link": state_link, "state_entry": state_entry, "state": state_link + state_entry,
            "green_change": green, "red_delay": red, "total": state_link + state_entry + green + red,
            "state_by_period": dict(per_period), "cycles": cycles,
            "last_update": last, "vehicles_at_last_update": at_end["vehicles"],
            "pending_at_last_update": at_end["pending"], "last_trace_second": trace[-1]["time_seconds"],
            "vehicle_seconds_after_last_update": int(sum(r["vehicles"] for r in after))}


def ci(values):
    v = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v))
    return {"mean": float(v.mean()), "ci95": float(half), "sd": float(v.std(ddof=1))}


def check_snapshot(method, day, ref, t=3000):
    """p(t) in the npz equals the measured state in snapshots/t<t>.json."""
    run = run_dir(method, day)
    z = np.load(run / "snapshots.npz")
    p = z["p"][list(z["t"]).index(t)]
    snap = json.loads((run / f"snapshots/t{t}.json").read_text(encoding="utf-8"))
    pi = ref["parameter_indices"]
    err = max(abs(p[i] - snap["edge_vehicle_count"][e]) for e, i in pi["n0"].items())
    err = max(err, max(abs(p[i] - snap["boundary_backlog_vehicles"][e]) for e, i in pi["backlog"].items()))
    err = max(err, max(abs(p[i] - snap["signals"][m]["previous_executed_green_seconds"][
        PHASES[m].index(ph)]) for (m, ph), i in pi["previous_green"].items()))
    return float(err)


def summarize(rows):
    keys = ("total", "state", "state_link", "state_entry", "green_change", "red_delay")
    summary = {}
    for method in METHODS:
        s = {k: ci([rows[method][day][k] for day in DAYS]) for k in keys}
        for k in keys:
            diff = [rows[method][day][k] - rows["Centralized"][day][k] for day in DAYS]
            rel = [100 * x / rows["Centralized"][day][k] for x, day in zip(diff, DAYS)]
            s[k + "_paired_diff"] = ci(diff)
            s[k + "_paired_diff_percent"] = ci(rel)
            s[k + "_paired_t_p"] = float(stats.ttest_1samp(diff, 0).pvalue) if np.ptp(diff) > 0 else None
        s["vehicles_at_last_update"] = ci([rows[method][day]["vehicles_at_last_update"] for day in DAYS])
        s["vehicles_at_last_update_max"] = max(rows[method][day]["vehicles_at_last_update"] for day in DAYS)
        s["last_update_max"] = max(rows[method][day]["last_update"] for day in DAYS)
        summary[method] = s
    return summary


def table(summary, title):
    lines = [f"### {title}", "",
             "| Method | J | J_state | J_green | J_red | (J - J_cen)/J_cen (%) | (J_state - J_cen,state)/J_cen,state (%) |",
             "|---|---|---|---|---|---|---|"]
    for method, s in summary.items():
        f = lambda k: f"{s[k]['mean']:.0f} ± {s[k]['ci95']:.0f}"
        g = lambda k: f"{s[k]['mean']:+.2f} ± {s[k]['ci95']:.2f}"
        lines.append(f"| {method} | {f('total')} | {f('state')} | {s['green_change']['mean']:.2f} | {f('red_delay')} | "
                     f"{g('total_paired_diff_percent')} | {g('state_paired_diff_percent')} |")
    return lines + [""]


def main():
    global PHASES
    st = ir.static()
    PHASES = {m: st["network"]["signals"][m]["green_native_indices"] for m in st["signals"]}
    last = max(int(np.load(run_dir(m, day) / "snapshots.npz")["t"][-1]) for m in METHODS for day in DAYS)
    ref, tables = clock_tables(st, range(0, last + 1, d.STEP))
    labels = json.loads(json.dumps(ref["param_labels"]))
    rows = {w: {m: {} for m in METHODS} for w in ("full", "common")}
    checks = {}
    for method in METHODS:
        for day in DAYS:
            for w in rows:
                rows[w][method][day] = run_cost(method, day, ref, tables, labels, w)
            print(method, day, {k: round(v, 1) for k, v in rows["full"][method][day].items()
                                 if isinstance(v, (int, float))}, flush=True)
        checks[method] = max(check_snapshot(method, day, ref) for day in DAYS)
    summary = {w: summarize(rows[w]) for w in rows}
    OUT.mkdir(exist_ok=True)
    (OUT / "closed_loop_cost.json").write_text(json.dumps(
        {"definition": __doc__, "days": DAYS, "per_day": rows, "summary": summary,
         "snapshot_check_max_abs_error_t3000": checks}, indent=1), encoding="utf-8")
    lines = ["Mean ± 95% CI over the 10 test days (t, 9 dof); the last two columns are paired differences.", ""]
    lines += table(summary["full"], "Full run (until the network is empty)")
    lines += table(summary["common"], f"Common window [0, {T_COMMON}] s")
    lines += ["| Method | last update max (s) | vehicles at last update (mean, max) |", "|---|---|---|"]
    for method, s in summary["full"].items():
        lines.append(f"| {method} | {s['last_update_max']} | {s['vehicles_at_last_update']['mean']:.1f}, "
                     f"{s['vehicles_at_last_update_max']} |")
    text = "\n".join(lines) + "\n"
    (OUT / "closed_loop_cost.md").write_text(text, encoding="utf-8")
    print(text)
    print("snapshot check", checks)


if __name__ == "__main__":
    main()
