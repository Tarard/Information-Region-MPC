"""Shared pieces: model at a clock, solver settings, state <-> parameter vector, green execution."""

import math

import numpy as np

import ns_design as d
import cc_model
import fair_solver
import strict_qp

fair_solver.install()                 # Clarabel strict settings, no OSQP, accept at 1e-5 (closed loop)


def ref_key(refs, t):
    """Reference used at update t: its clock (nearest stored clock) in "update" mode, its period in "period" mode."""
    if d.REFERENCE_MODE == "period":
        return d.period(t)
    clocks = [k for k in refs if isinstance(k, int)]
    return min(clocks, key=lambda c: (abs(c - t), c))


def arrival_rates(net, t):
    table = net["arrival_table"]
    row = min(table["rows"], key=lambda r: (abs(r["clock_seconds"] - t), r["clock_seconds"]))
    return dict(zip(table["edge_ids"], row["arrival_rate_veh_per_second"]))


def build(net, snapshot, t):
    return cc_model.build(net["network"], net["signals"], snapshot, arrival_rates(net, t), net["service_groups"],
                          net["edge_owner"])


def solve_full(model, p, warm=None):
    """Centralized optimum at parameter p from the standard form (matrices do not depend on p)."""
    return strict_qp.solve_qp(model["H"], model["h0"] + model["hP"] @ p, model["E"], model["e0"] + model["eP"] @ p,
                              model["G"], model["g0"] + model["gP"] @ p, warm)


def objective(model, p, x):
    x = np.asarray(x)
    return float(0.5 * x @ (model["H"] @ x) + (model["h0"] + model["hP"] @ p) @ x)


def first_decision(model, x):
    return {m: np.asarray(x)[idx] for m, idx in model["output_indices"].items()}


def snapshot_from_p(net, labels, p, t):
    """Inverse of the parameter vector of cc_model.build (used to rebuild stored states)."""
    snap = {"time_seconds": t, "edge_vehicle_count": {e: 0.0 for e in net["network"]["roads"]},
            "movement_vehicle_count": {},
            "boundary_backlog_vehicles": {}, "boundary_release_rate_veh_per_second": {},
            "signals": {a: {"previous_executed_green_seconds": [0.0] * s["stage_count"]} for a, s in net["signals"].items()}}
    for lab, v in zip(labels, p):
        kind = lab[0]
        if kind == "n0" and len(lab) == 3:
            snap["movement_vehicle_count"][lab[1], lab[2]] = float(v)
        elif kind == "n0":
            snap["edge_vehicle_count"][lab[1]] = float(v)
        elif kind == "backlog":
            snap["boundary_backlog_vehicles"][lab[1]] = float(v)
        elif kind == "release_rate":
            snap["boundary_release_rate_veh_per_second"][lab[1]] = float(v)
        elif kind == "previous_green":
            snap["signals"][lab[1]]["previous_executed_green_seconds"][int(lab[2]) // 2] = float(v)
    return snap


def legal_green(values, signal):
    """Repair numerical error of at most 0.01 s onto the legal green set (as in the LYG executor)."""
    budget, n = float(signal["green_budget_seconds"]), signal["stage_count"]
    values = [float(x) for x in values]
    if len(values) != n or not all(math.isfinite(x) for x in values):
        raise ValueError("The green vector does not match the stages.")
    if abs(sum(values) - budget) > 0.01 or any(x < d.G_MIN - 0.01 or x > d.G_MAX + 0.01 for x in values):
        raise ValueError("Action is too far from the legal green set.")
    g = [min(max(x, d.G_MIN), d.G_MAX) for x in values]
    delta = budget - sum(g)
    for _ in g:
        if abs(delta) < 1e-8:
            break
        slack = [(d.G_MAX - x) if delta > 0 else (x - d.G_MIN) for x in g]
        i = max(range(n), key=slack.__getitem__)
        change = math.copysign(min(abs(delta), slack[i]), delta)
        g[i] += change
        delta -= change
    return g


def integer_green(values, signal):
    """Whole seconds: floor, then largest remainders; keeps the budget and the bounds."""
    budget = int(signal["green_budget_seconds"])
    whole = [math.floor(x + 1e-8) for x in values]
    for i in sorted(range(len(values)), key=lambda i: (-(values[i] - whole[i]), i))[:budget - sum(whole)]:
        whole[i] += 1
    if sum(whole) != budget or any(x < d.G_MIN or x > d.G_MAX for x in whole):
        raise ValueError("Rounded greens violate the signal plan.")
    return whole


def deciders(signals, t):
    """Intersections whose cycle starts at clock t (they decide at t)."""
    return sorted(a for a, s in signals.items() if t % s["cycle_seconds"] == 0)


def phase_plan(signals, tls_order, greens, t):
    """Per-second native phase index for the step [t, t + T): each intersection runs the part of its current cycle
    (greens[a], G0 Y0 G1 Y1 ...) that falls in the step."""
    plan = np.zeros((d.CYCLE, len(tls_order)), int)
    for k, a in enumerate(tls_order):
        s, col = signals[a], []
        for st, g in zip(s["stages"], greens[a]):
            col += [st["green_native_index"]] * int(g) + [st["yellow_native_index"]] * d.YELLOW
        if len(col) != s["cycle_seconds"]:
            raise ValueError(f"{a}: the cycle is not {s['cycle_seconds']} s.")
        pos = t % s["cycle_seconds"]
        plan[:, k] = col[pos:pos + d.CYCLE]
    return plan
