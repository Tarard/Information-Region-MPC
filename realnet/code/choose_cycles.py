"""Cycle of each intersection (60 or 120 s) from the calibrated history demand of the peak window (cc_design.PEAK).

For intersection m with stages k and movements l (service map gamma_lk, saturation S_l, historical rate
lambda_l = road arrival rate x declared turn share), the critical flow ratio is the smallest total green share
    Y_m = min sum_k u_k  subject to  S_l sum_k gamma_lk u_k >= lambda_l,  u >= 0,
which equals Webster's sum of critical flow ratios when each movement is served by one stage. Webster's cycle is
C0_m = (1.5 L_m + 5) / (1 - Y_m), L_m = 3 s x stages. It ignores the 10-s minimum green.
With the green bounds, for C in (60, 120) the balanced split maximizes z subject to
S_l sum_k gamma_lk g_k >= z lambda_l C, sum_k g_k = C - L_m, 10 <= g_k <= 60; then x_l = lambda_l C / (S_l G_l).
Webster delay of movement l: C (1 - G_l/C)^2 / (2 (1 - min(x_l, 0.95) G_l/C)) + x^2 / (2 lambda_l (1 - x)),
x = min(x_l, 0.95). Rule: C_m is the cycle with the lower flow-weighted Webster delay at the high segment.
Writes design/CYCLE_CHOICE.json. Usage: choose_cycles.py
"""

import json

import numpy as np
from scipy.optimize import linprog

import cc_design as d

X_CAP = 0.95


def balanced(movements, lam, n, C):
    """Green split with the largest common capacity reserve z under the green bounds; returns (g, x per movement)."""
    L = d.lost_time(n)
    c = np.zeros(n + 1)
    c[-1] = -1.0                                             # maximize z
    A = [[-g["saturation_rate_veh_per_second"] * y for y in g["gamma"]] + [l * C] for g, l in zip(movements, lam)]
    res = linprog(c, A_ub=A, b_ub=np.zeros(len(A)), A_eq=[[1.0] * n + [0.0]], b_eq=[C - L],
                  bounds=[(d.G_MIN, d.G_MAX)] * n + [(0, None)], method="highs")
    if res.status != 0:
        return None, None
    g = res.x[:n]
    G = [sum(y * v for y, v in zip(m["gamma"], g)) for m in movements]
    return g, [l * C / (m["saturation_rate_veh_per_second"] * Gl) for m, l, Gl in zip(movements, lam, G)]


def webster_delay(movements, lam, n, C):
    """Flow-weighted mean Webster delay (s/veh) with the balanced split."""
    g, x = balanced(movements, lam, n, C)
    if g is None:
        return float("inf"), None
    total = flow = 0.0
    for m, l, xl in zip(movements, lam, x):
        if l <= 0:
            continue
        share = sum(y * v for y, v in zip(m["gamma"], g)) / C
        xc = min(xl, X_CAP)
        dl = C * (1 - share) ** 2 / (2 * (1 - xc * share)) + xc ** 2 / (2 * l * (1 - xc))
        total, flow = total + l * dl, flow + l
    return total / flow, max(x)


def main():
    groups = json.loads(d.SERVICE_GROUPS.read_text(encoding="utf-8"))["signals"]
    table = json.loads(d.ARRIVAL_TABLE.read_text(encoding="utf-8"))
    rows = [r["arrival_rate_veh_per_second"] for r in table["rows"] if d.PEAK[0] <= r["clock_seconds"] < d.PEAK[1]]
    rate = dict(zip(table["edge_ids"], np.mean(rows, axis=0)))
    out = {}
    for tls, sig in sorted(groups.items()):
        movements = [g for g in sig["movement_capacity_groups"] if sum(g["gamma"]) > 0]
        n = len(sig["movement_capacity_groups"][0]["gamma"])
        A = [[-g["saturation_rate_veh_per_second"] * x for x in g["gamma"]] for g in movements]
        b = [-g["declared_turn_share"] * rate[g["incoming_edge"]] for g in movements]
        res = linprog(np.ones(n), A_ub=A, b_ub=b, bounds=[(0, None)] * n, method="highs")
        Y = float(res.fun) if res.status == 0 else float("inf")
        L = d.lost_time(n)
        C0 = (1.5 * L + 5) / (1 - Y) if Y < 1 else float("inf")
        lam = [-v for v in b]
        d60, x60 = webster_delay(movements, lam, n, 60)
        d120, x120 = webster_delay(movements, lam, n, 120)
        out[tls] = {"stages": n, "Y": Y, "lost_time": L, "webster_cycle": C0, "delay_60": d60, "delay_120": d120,
                    "max_x_60": x60, "max_x_120": x120, "cycle": 60 if d60 <= d120 else 120}
    (d.ROOT / "design/CYCLE_CHOICE.json").write_text(json.dumps({"rule": __doc__, "signals": out}, indent=1),
                                                     encoding="utf-8")
    for tls, r in out.items():
        print(f"{tls:>6} n={r['stages']} Y={r['Y']:.2f} C0={r['webster_cycle']:4.0f} | 60 s: x={r['max_x_60'] or 9:.2f} "
              f"d={r['delay_60']:6.1f} | 120 s: x={r['max_x_120'] or 9:.2f} d={r['delay_120']:6.1f} -> {r['cycle']}")
    print("60 s:", sum(r["cycle"] == 60 for r in out.values()), "120 s:", sum(r["cycle"] == 120 for r in out.values()))


if __name__ == "__main__":
    main()
