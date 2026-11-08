"""Centralized store-and-forward QP with own-cycle updates (main.tex Sec. II).

Changes from v3 aligned_model.build + augmented_model.build, nothing else:
- step T = 60 s, Np = 8 steps; C_m = 60 or 120 s; every cycle has its own greens (cycle_layout). A cycle that starts
  at this update is decided now; a running 120-s cycle keeps its greens (parameters previous_green) for step 0;
- greens 10-60 s, sum of greens = C_m - 3 s x stages; yellow gives no service, so Eq. (service) is u <= S G / C_m;
- red-delay rows with C = C_m, f_l = 0, rho_c = share of the cycle inside the horizon;
- entry queue w(k+1) = w(k) + T (r - d(k)) with d, w >= 0 only (no row d(k) <= w(k)).
previous_green = greens of the most recently started cycle (the running one, or the one ending now).
No route or future-demand file is read here.
"""

from collections import defaultdict
import math

import numpy as np
from scipy import sparse

import cc_design as d
from frozen import v3_model

DT = d.STEP


def cycle_layout(cycle_seconds, now):
    """Cycles of one intersection on the step grid at update `now`.

    A cycle starting at `now` is decided now. A 120-s cycle that started 60 s ago is running: its greens are data and
    serve the first step. starts[c] = first step of new cycle c (all c with starts[c] < STEPS); serving[k] = the new
    cycle serving step k, or None for the running cycle; rho[c] = share of cycle c inside the horizon."""
    if cycle_seconds % DT:
        raise ValueError("Every cycle must be a multiple of the step.")
    span, elapsed = cycle_seconds // DT, now % cycle_seconds
    first_start = (cycle_seconds - elapsed) // DT if elapsed else 0
    starts = list(range(first_start, d.STEPS, span))
    serving = [None if k < first_start else (k - first_start) // span for k in range(d.STEPS)]
    rho = [min(span, d.STEPS - s) / span for s in starts]
    return {"decides": elapsed == 0, "span": span, "starts": starts, "serving": serving, "rho": rho}


def arrival_row(rows, now):
    """Historical arrival row of the nearest table clock (ties: the earlier clock)."""
    clock = min(rows, key=lambda c: (abs(c - now), c))
    return clock, rows[clock]


def build(network, signals, snapshot, arrival_rates, service_groups, edge_owner, *,
          mu=d.MU, green_weight=d.GREEN_WEIGHT):
    roads = network["roads"]
    edges = sorted(roads)
    entries, exits = sorted(network["entry_edges"]), set(network["exit_edges"])
    if edge_owner is None or set(edge_owner) != set(edges) or set(edge_owner.values()) - set(signals):
        raise ValueError("Supply exactly one declared signal owner for every physical road.")
    if set(signals) != set(network["signals"]) or set(signals) != set(service_groups["signals"]):
        raise ValueError("The signal sets differ.")
    capacity = {e: sum(math.floor(float(lane["length"]) / 7.5) + 1 for lane in roads[e]["lanes"]) for e in edges}
    parameter, labels, owners = [], [], []
    pi = {name: {} for name in ("n0", "backlog", "release_rate", "previous_green")}

    def add_parameter(kind, key, value, owner):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"The current {kind} parameter must be finite and nonnegative.")
        pi[kind][key] = len(parameter)
        parameter.append(value)
        labels.append((kind, *key) if isinstance(key, tuple) else (kind, key))
        owners.append(owner)

    stage_phases = {tls: network["signals"][tls]["green_native_indices"] for tls in signals}
    for tls in sorted(signals):
        for edge in edges:
            if edge_owner[edge] == tls:
                add_parameter("n0", edge, snapshot["edge_vehicle_count"][edge], tls)
        for edge in entries:
            if edge_owner[edge] == tls:
                add_parameter("backlog", edge, snapshot["boundary_backlog_vehicles"][edge], tls)
                add_parameter("release_rate", edge, snapshot["boundary_release_rate_veh_per_second"][edge], tls)
        previous = snapshot["signals"][tls]["previous_executed_green_seconds"]
        phases, signal = stage_phases[tls], signals[tls]
        if [s["green_native_index"] for s in signal["stages"]] != phases:
            raise ValueError(f"{tls}: the stage order differs from the native order.")
        if len(previous) != len(phases) or abs(sum(map(float, previous)) - signal["green_budget_seconds"]) > 1e-6 \
                or any(not d.G_MIN - 1e-6 <= float(v) <= d.G_MAX + 1e-6 for v in previous):
            raise ValueError(f"{tls}: the previous green vector is not legal.")
        for ordinal, phase in enumerate(phases):
            add_parameter("previous_green", (tls, phase), previous[ordinal], tls)

    p = np.asarray(parameter)
    problem = v3_model.Problem(p)
    inventory, outflow, backlog, admission, greens, first = {}, {}, {}, {}, {}, {}
    now = int(snapshot["time_seconds"])
    if now % DT:
        raise ValueError("An update must start at a step boundary.")
    layout = {tls: cycle_layout(signals[tls]["cycle_seconds"], now) for tls in signals}
    for tls in sorted(signals):
        budget = float(signals[tls]["green_budget_seconds"])
        for cycle in range(len(layout[tls]["starts"])):
            ids = []
            for phase in stage_phases[tls]:
                index = problem.variable(("green", tls, cycle, phase), d.G_MIN, d.G_MAX)
                greens[tls, cycle, phase] = index
                problem.difference_penalty(index, green_weight,
                    previous=greens[tls, cycle - 1, phase] if cycle else None,
                    parameter=pi["previous_green"][tls, phase] if not cycle else None)
                ids.append(index)
            problem.constraint(("cycle_budget", tls, cycle), [(i, 1) for i in ids], budget, budget)
            if cycle == 0 and layout[tls]["decides"]:     # only an intersection whose cycle starts now decides
                first[tls] = ids

    for edge in edges:
        initial = p[pi["n0"][edge]]
        for k in range(d.STEPS + 1):
            inventory[edge, k] = problem.variable(("inventory", edge, k),
                initial if k == 0 else 0, initial if k == 0 else capacity[edge],
                0 if k == 0 else 0.5 / capacity[edge], fixed_parameter=pi["n0"][edge] if k == 0 else None)
        for k in range(d.STEPS):
            outflow[edge, k] = problem.variable(("link_outflow", edge, k))
    for edge in entries:
        initial = p[pi["backlog"][edge]]
        for k in range(d.STEPS + 1):
            backlog[edge, k] = problem.variable(("external_wait", edge, k),
                initial if k == 0 else 0, initial if k == 0 else math.inf,
                0 if k == 0 else 0.5 / capacity[edge], fixed_parameter=pi["backlog"][edge] if k == 0 else None)
        for k in range(d.STEPS):
            admission[edge, k] = problem.variable(("admission", edge, k))
            release = DT * p[pi["release_rate"][edge]]
            derivative = {pi["release_rate"][edge]: DT}
            problem.constraint(("external_wait_conservation", edge, k),
                [(backlog[edge, k + 1], 1), (backlog[edge, k], -1), (admission[edge, k], 1)],
                release, release, lower_derivative=derivative, upper_derivative=derivative)
            # No row admission(k) <= backlog(k): with w(k+1) >= 0 the admission is already at most w(k) + T r. That row
            # held every new arrival back one step and, with w(k) = 0, made the active rows dependent (failed KKT checks).

    connections, incoming = defaultdict(list), defaultdict(list)
    for connection in network["connections"]:
        connections[connection["from"]].append(connection)
    for source, links in list(connections.items()):
        shares = network["turn_priors"][source]["shares"]
        if set(shares) != {c["to"] for c in links} or abs(sum(shares.values()) - 1) > 1e-8:
            raise ValueError(f"The fixed turn shares do not match connections at {source}.")
        if len({bool(c.get("tl")) for c in links}) != 1:
            raise ValueError(f"Signal and nonsignal outflows are mixed at {source}.")
        for target, share in shares.items():
            if share > 0:
                incoming[target].append((source, float(share)))
    if set(edges) - set(connections) != exits:
        raise ValueError("Some physical road exits have no declared classification.")
    for edge in edges:
        for k in range(d.STEPS):
            row = [(inventory[edge, k + 1], 1), (inventory[edge, k], -1), (outflow[edge, k], 1)]
            row.extend((outflow[source, k], -share) for source, share in incoming[edge])
            if edge in entries:
                row.append((admission[edge, k], -1))
            problem.constraint(("link_conservation", edge, k), row, 0, 0)

    no_green_service = []
    for edge in edges:
        links = connections[edge]
        if not links:
            limit = DT * len(roads[edge]["lanes"]) * float(network["nonsignal_service_veh_per_lane_second"])
            for k in range(d.STEPS):
                problem.constraint(("network_exit_capacity", edge, k), [(outflow[edge, k], 1)], upper=limit)
            continue
        signaled = bool(links[0].get("tl"))
        shares = network["turn_priors"][edge]["shares"]
        groups = defaultdict(lambda: defaultdict(dict))
        for connection in links:
            lane, rate = int(connection["fromLane"]), float(connection["service_veh_per_second"])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Every declared service rate must be positive and finite.")
            for phase in connection["served_native_phases"] if signaled else [None]:
                by_lane = groups[connection["to"]][phase]
                by_lane[lane] = max(rate, by_lane.get(lane, 0))
        all_lanes = defaultdict(dict)
        for phase_rows in groups.values():
            for phase, lanes in phase_rows.items():
                for lane, rate in lanes.items():
                    all_lanes[phase][lane] = max(rate, all_lanes[phase].get(lane, 0))
        if signaled:
            tls = links[0]["tl"]
            if any(c["tl"] != tls for c in links) or edge_owner[edge] != tls:
                raise ValueError(f"The signal capacity at {edge} is not local to its declared owner.")

            overlap = DT / signals[tls]["cycle_seconds"]      # o_{k,c} / C_m: the step covers this share of a cycle

            def add_capacity(label, k, phase_lanes, share):
                # Eq. (service): T u <= S (T / C_m) G for the cycle serving step k, i.e. u <= S G / C_m. A running
                # cycle serves with its fixed greens (parameters). Yellow phases give no service.
                cycle = layout[tls]["serving"][k]
                row, upper, derivative = [(outflow[edge, k], share)], 0.0, {}
                for phase, lanes in phase_lanes.items():
                    if phase in stage_phases[tls]:
                        rate = overlap * sum(lanes.values())
                        if cycle is None:
                            column = pi["previous_green"][tls, phase]
                            upper += rate * p[column]
                            derivative[column] = derivative.get(column, 0.0) + rate
                        else:
                            row.append((greens[tls, cycle, phase], -rate))
                problem.constraint(label, row, upper=upper, upper_derivative=derivative or None)
                return len(row) == 1 and not derivative

            for k in range(d.STEPS):
                add_capacity(("signal_link_capacity", edge, k), k, all_lanes, 1)
                for target, phase_lanes in groups.items():
                    share = float(shares[target])
                    if share > 0:
                        if add_capacity(("signal_turn_capacity", edge, target, k), k, phase_lanes, share) and k == 0:
                            no_green_service.append((edge, target))
        else:
            total_rate = sum(all_lanes[None].values())
            for k in range(d.STEPS):
                problem.constraint(("nonsignal_link_capacity", edge, k), [(outflow[edge, k], 1)], upper=DT * total_rate)
                for target, phase_lanes in groups.items():
                    share = float(shares[target])
                    if share > 0:
                        problem.constraint(("nonsignal_turn_capacity", edge, target, k),
                            [(outflow[edge, k], share)], upper=DT * sum(phase_lanes[None].values()))

    def owner(label):
        return label[1] if label[0] in ("green", "cycle_budget") else edge_owner[label[1]]

    variable_owner = np.asarray([owner(label) for label in problem.labels])
    row_owner = np.asarray([owner(label) for label in problem.row_labels])
    matrices, H = problem.matrices()
    standard = problem.standard_form(matrices, H, variable_owner, row_owner)
    constant = 0.5 * green_weight * sum(p[col] ** 2 for col in pi["previous_green"].values())

    # Red-delay term of Eq. (traffic-cost), as in v3 augmented_model with R = C - gamma g (f_l = 0), for every new cycle
    # with rho_c = its share inside the horizon. A running cycle has fixed greens and adds only a constant (omitted).
    rows = []
    for tls in sorted(signals):
        for group in service_groups["signals"][tls]["movement_capacity_groups"]:
            edge = group["incoming_edge"]
            arrival = float(arrival_rates[edge])
            share, saturation = float(group["declared_turn_share"]), float(group["saturation_rate_veh_per_second"])
            gamma = np.asarray(group["gamma"], dtype=float)
            if (not math.isfinite(arrival) or arrival < 0 or not math.isfinite(saturation) or saturation <= 0
                    or gamma.shape != (len(stage_phases[tls]),) or not np.isfinite(gamma).all() or np.any(gamma < 0)
                    or not 0 <= share <= 1):
                raise ValueError("A movement has an invalid rate or service map.")
            movement_arrival = share * arrival
            beta = movement_arrival / (1 - min(movement_arrival / saturation, d.CHI_BAR))
            for cycle, rho in enumerate(layout[tls]["rho"]):
                rows.append({"signal": tls, "incoming_edge": edge, "destination_edge": group["destination_edge"],
                             "cycle": cycle, "columns": [greens[tls, cycle, ph] for ph in stage_phases[tls]],
                             "gamma": gamma.tolist(), "cycle_seconds": float(signals[tls]["cycle_seconds"]),
                             "fixed_service_seconds": 0.0, "movement_arrival_veh_per_second": movement_arrival,
                             "beta": beta, "coverage_weight": float(rho),
                             "flow_ratio_clipped": bool(movement_arrival / saturation > d.CHI_BAR)})
    data, rr, cc, b = [], [], [], []
    for j, row in enumerate(rows):
        scale = math.sqrt(mu * row["coverage_weight"] * row["beta"])
        for column, coefficient in zip(row["columns"], row["gamma"]):
            if coefficient and scale:
                data.append(scale * coefficient)
                rr.append(j)
                cc.append(column)
        b.append(scale * (row["cycle_seconds"] - row["fixed_service_seconds"]))
    n = H.shape[0]
    Aadd = sparse.coo_matrix((data, (rr, cc)), shape=(len(rows), n)).tocsc()
    badd = np.asarray(b)
    H = (standard["H"] + Aadd.T @ Aadd).tocsc()
    h = np.asarray(standard["h"] - Aadd.T @ badd)
    standard.update(H=H, h=h, h0=h - standard["hP"] @ p)
    matrices = {**matrices, "P": sparse.triu(H, format="csc"), "q": h}
    return {**standard, "matrices": matrices, "output_indices": first, "p": p, "param_labels": labels,
            "param_owner": np.asarray(owners), "parameter_indices": pi, "variable_labels": problem.labels,
            "row_labels": problem.row_labels, "variable_owner": variable_owner, "row_owner": row_owner,
            "objective_constant": float(constant + 0.5 * badd @ badd), "red_delay_rows": rows,
            "indices": {"inventory": inventory, "outflow": outflow, "external_wait": backlog,
                        "admission": admission, "green": greens},
            "metadata": {"prediction_step_seconds": DT, "horizon_steps": d.STEPS, "time_seconds": now,
                         "deciding_signals": sorted(first), "cycle_layout": layout,
                         "cycle_seconds": {tls: signals[tls]["cycle_seconds"] for tls in sorted(signals)},
                         "horizon_seconds": d.HORIZON, "green_bounds_seconds": [d.G_MIN, d.G_MAX],
                         "mu": mu, "green_change_weight": green_weight, "chi_bar": d.CHI_BAR,
                         "capacity": capacity, "turns_without_green_service": no_green_service,
                         "variable_count": n, "constraint_count_including_boxes": matrices["A"].shape[0],
                         "parameter_count": len(p)}}


def direct_cost(instance, x):
    """Evaluate Eq. (traffic-cost) directly, including its parameter-only constant."""
    x, p, meta = np.asarray(x, dtype=float), instance["p"], instance["metadata"]
    total = sum(0.5 * x[i] ** 2 / meta["capacity"][edge]
                for kind in ("inventory", "external_wait") for (edge, k), i in instance["indices"][kind].items() if k)
    greens = instance["indices"]["green"]
    for (tls, cycle, phase), i in greens.items():
        before = x[greens[tls, cycle - 1, phase]] if cycle else p[instance["parameter_indices"]["previous_green"][tls, phase]]
        total += 0.5 * meta["green_change_weight"] * (x[i] - before) ** 2
    for row in instance["red_delay_rows"]:
        red = row["cycle_seconds"] - row["fixed_service_seconds"] - np.asarray(row["gamma"]) @ x[row["columns"]]
        total += 0.5 * meta["mu"] * row["coverage_weight"] * row["beta"] * red ** 2
    return float(total)
