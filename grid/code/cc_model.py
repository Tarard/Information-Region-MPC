"""Centralized store-and-forward QP with multi-rate updates (main.tex Sec. II, Eq. (4); v2 user spec 2026-10-30).

- step T = 60 s, Np = 8 steps (480 s). Every intersection runs its own cycle C_m = 60 or 120 s from t = 0 (offset 0).
  At clock t the cycles of m in the horizon follow its calendar: a cycle that is running (t mod C_m > 0) is fixed and
  its greens are the previous_green parameter; every later cycle has its own green vector (variables).
- m decides at t only when a cycle of m starts at t (t mod C_m = 0): its first decision is the greens of that cycle.
- greens 10-60 s, sum of greens = C_m - 3 s x stages; yellow gives no service. Eq. (4): T u(k) <= sum over cycles c of
  o(k, c) / C_m x S x G_c, with o(k, c) the overlap of step k with cycle c.
- red-delay rows per decided cycle with C = C_m, f_l = 0, coverage rho_c = share of the cycle inside the horizon.
- flow model (user 2026-10-30): links store any number of vehicles (no storage bound); entries as Eq. (2): queue
  w outside the network, w(k+1) = w(k) + T r - d(k), w >= 0, admission d >= 0 into the entry link (r = observed
  arrival rate, a parameter; no other admission rule; with unlimited storage the optimum keeps w = 0); exit links
  have no capacity row (vehicles leave freely). One queue per movement on every non-exit link
  (its inflow joins the movements by the fixed turn shares), so a movement on red does not hold the others back.
  Capacity as Eq. (4) of the paper: one row for the whole link (its movements together, every lane once) and one
  row per movement (all lanes it uses); together they handle shared lanes.
  The storage of a link (lanes x (floor(length / 7.5) + 1)) is kept only in the cost weight 0.5 / storage of each of
  its movement queues (and of the vehicles on an exit link).
No route or future-demand file is read here.
"""

from collections import defaultdict
import math

import numpy as np
from scipy import sparse

import ns_design as d
import aligned_model as v3_model

DT = d.CYCLE


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

    connections = defaultdict(list)
    for connection in network["connections"]:
        connections[connection["from"]].append(connection)
    if set(edges) - set(connections) != exits:
        raise ValueError("Some physical road exits have no declared classification.")
    movements = defaultdict(list)                       # edge -> targets with trips (one queue per movement)
    for source, links in connections.items():
        shares = network["turn_priors"][source]["shares"]
        if set(shares) != {c["to"] for c in links} or abs(sum(shares.values()) - 1) > 1e-8:
            raise ValueError(f"The fixed turn shares do not match connections at {source}.")
        if len({bool(c.get("tl")) for c in links}) != 1:
            raise ValueError(f"Signal and nonsignal outflows are mixed at {source}.")
        movements[source] = sorted(y for y, share in shares.items() if share > 0)

    stage_phases = {tls: network["signals"][tls]["green_native_indices"] for tls in signals}
    for tls in sorted(signals):
        for edge in edges:
            if edge_owner[edge] != tls:
                continue
            if edge in exits:
                add_parameter("n0", edge, snapshot["edge_vehicle_count"][edge], tls)
            for target in movements[edge]:
                add_parameter("n0", (edge, target), snapshot["movement_vehicle_count"][edge, target], tls)
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
    inventory, outflow, greens, first = {}, {}, {}, {}
    now, horizon = float(snapshot["time_seconds"]), float(DT * d.CYCLES)
    calendar = {}
    for tls in sorted(signals):
        budget = float(signals[tls]["green_budget_seconds"])
        cyc = float(signals[tls]["cycle_seconds"])
        pos = now % cyc
        cal = [(-pos, cyc - pos, None)] if pos else []          # running cycle: fixed, greens = previous_green
        start, cycle = (cyc - pos) if pos else 0.0, 0
        while start < horizon - 1e-9:
            ids = []
            for phase in stage_phases[tls]:
                index = problem.variable(("green", tls, cycle, phase), d.G_MIN, d.G_MAX)
                greens[tls, cycle, phase] = index
                problem.difference_penalty(index, green_weight,
                    previous=greens[tls, cycle - 1, phase] if cycle else None,
                    parameter=pi["previous_green"][tls, phase] if not cycle else None)
                ids.append(index)
            problem.constraint(("cycle_budget", tls, cycle), [(i, 1) for i in ids], budget, budget)
            if cycle == 0 and not pos:
                first[tls] = ids
            cal.append((start, start + cyc, cycle))
            start, cycle = start + cyc, cycle + 1
        calendar[tls] = (cyc, cal)

    # vehicles: one queue per movement (approach -> outgoing link) on every non-exit link, its initial value the
    # measured count n0; every movement queue (k >= 1) carries the cost weight 0.5 / storage of its link (user
    # 2026-10-31: a cost on the link total alone leaves the split between movements free, the optimum is not unique
    # and the KKT matrix of the boundary construction is singular); exit links keep one count with the same weight
    queue, move = {}, {}
    for edge in edges:
        if edge in exits:
            for k in range(1, d.CYCLES + 1):
                inventory[edge, k] = problem.variable(("inventory", edge, k), 0, math.inf, 0.5 / capacity[edge])
            col = pi["n0"][edge]
            inventory[edge, 0] = problem.variable(("inventory", edge, 0), p[col], p[col], fixed_parameter=col)
            for k in range(d.CYCLES):
                outflow[edge, k] = problem.variable(("link_outflow", edge, k))
            continue
        for y in movements[edge]:
            col = pi["n0"][edge, y]
            queue[edge, y, 0] = problem.variable(("movement_queue", edge, y, 0), p[col], p[col], fixed_parameter=col)
            for k in range(1, d.CYCLES + 1):
                queue[edge, y, k] = problem.variable(("movement_queue", edge, y, k), 0, math.inf,
                                                     0.5 / capacity[edge])
            for k in range(d.CYCLES):
                move[edge, y, k] = problem.variable(("movement_outflow", edge, y, k))
    # Eq. (2): entry queue outside the network, w(k+1) = w(k) + T r - d(k), w >= 0, admission d(k) >= 0 (vehicles
    # per step); w carries the cost weight 0.5 / n_max of its entry link as in Eq. (5). With unlimited storage the
    # optimum admits every arrival (w stays 0), but the equations are kept as in the paper.
    wait, admit = {}, {}
    for edge in entries:
        col = pi["backlog"][edge]
        wait[edge, 0] = problem.variable(("external_wait", edge, 0), p[col], p[col], fixed_parameter=col)
        for k in range(d.CYCLES):
            wait[edge, k + 1] = problem.variable(("external_wait", edge, k + 1), 0, math.inf, 0.5 / capacity[edge])
            admit[edge, k] = problem.variable(("admission", edge, k))
            arrivals, derivative = DT * p[pi["release_rate"][edge]], {pi["release_rate"][edge]: DT}
            problem.constraint(("external_wait_conservation", edge, k),
                               [(wait[edge, k + 1], 1), (wait[edge, k], -1), (admit[edge, k], 1)], arrivals, arrivals,
                               lower_derivative=derivative, upper_derivative=derivative)
    upstream = defaultdict(list)                        # link -> approaches whose movement feeds it
    for (edge, y, k) in move:
        if k == 0:
            upstream[y].append(edge)
    for edge in edges:
        shares = None if edge in exits else network["turn_priors"][edge]["shares"]
        for k in range(d.CYCLES):
            feed = [(move[source, edge, k], -1.0) for source in upstream[edge]]
            if edge in entries:                                  # Eq. (1): the admission d enters the entry link
                feed.append((admit[edge, k], -1.0))
            if edge in exits:
                row = [(inventory[edge, k + 1], 1), (inventory[edge, k], -1), (outflow[edge, k], 1)] + feed
                problem.constraint(("link_conservation", edge, k), row, 0, 0)
                continue
            # the link inflow of the step as one variable (keeps the KKT matrix sparse), then split by the turn shares
            inflow = problem.variable(("link_inflow", edge, k))
            problem.constraint(("link_inflow_sum", edge, k), [(inflow, 1)] + feed, 0, 0)
            for y in movements[edge]:
                row = [(queue[edge, y, k + 1], 1), (queue[edge, y, k], -1), (move[edge, y, k], 1),
                       (inflow, -float(shares[y]))]
                problem.constraint(("movement_conservation", edge, y, k), row, 0, 0)

    no_green_service = []
    for edge in edges:
        links = connections[edge]
        if not links:
            continue                                            # exit link: vehicles leave freely (as in the plant)
        signaled = bool(links[0].get("tl"))
        groups = defaultdict(lambda: defaultdict(dict))
        for connection in links:
            if connection["to"] not in movements[edge]:
                continue
            lane, rate = int(connection["fromLane"]), float(connection["service_veh_per_second"])
            if not math.isfinite(rate) or rate <= 0:
                raise ValueError("Every declared service rate must be positive and finite.")
            for phase in connection["served_native_phases"] if signaled else [None]:
                by_lane = groups[connection["to"]][phase]
                by_lane[lane] = max(rate, by_lane.get(lane, 0))
        all_lanes = defaultdict(dict)                            # the lanes of the link per phase (shared lanes once)
        for phase_rows in groups.values():
            for phase, lanes in phase_rows.items():
                for lane, rate in lanes.items():
                    all_lanes[phase][lane] = max(rate, all_lanes[phase].get(lane, 0))
        if signaled:
            tls = links[0]["tl"]
            if any(c["tl"] != tls for c in links) or edge_owner[edge] != tls:
                raise ValueError(f"The signal capacity at {edge} is not local to its declared owner.")

            cyc, cal = calendar[tls]

            def capacity_row(k, targets, phase_lanes):
                # Eq. (4): T u(k) <= sum_c o(k, c) / C_m S G_c, for the whole link (all its movements, every lane
                # once) and for each movement (all lanes it uses); a fixed running cycle enters the bound through the
                # previous_green parameter. Yellow phases give no service.
                row, bound, derivative = [(move[edge, y, k], 1) for y in targets], 0.0, {}
                for start, end, cycle in cal:
                    o = v3_model.overlap(k * DT, (k + 1) * DT, start, end)
                    if o <= 0:
                        continue
                    for phase, lanes in phase_lanes.items():
                        if phase in stage_phases[tls]:
                            rate = o / cyc * sum(lanes.values())
                            if cycle is None:
                                col = pi["previous_green"][tls, phase]
                                bound += rate * p[col]
                                derivative[col] = derivative.get(col, 0.0) + rate
                            else:
                                row.append((greens[tls, cycle, phase], -rate))
                return row, bound, derivative

            for k in range(d.CYCLES):
                row, bound, derivative = capacity_row(k, movements[edge], all_lanes)
                problem.constraint(("signal_link_capacity", edge, k), row, upper=bound, upper_derivative=derivative)
                for target, phase_lanes in groups.items():
                    row, bound, derivative = capacity_row(k, [target], phase_lanes)
                    if len(row) == 1 and not derivative and k == 0:
                        no_green_service.append((edge, target))
                    problem.constraint(("signal_turn_capacity", edge, target, k), row, upper=bound,
                                       upper_derivative=derivative)
        else:
            for k in range(d.CYCLES):
                problem.constraint(("nonsignal_link_capacity", edge, k), [(move[edge, y, k], 1) for y in movements[edge]],
                                   upper=DT * sum(all_lanes[None].values()))
                for target, phase_lanes in groups.items():
                    problem.constraint(("nonsignal_turn_capacity", edge, target, k),
                                       [(move[edge, target, k], 1)], upper=DT * sum(phase_lanes[None].values()))

    def owner(label):
        return label[1] if label[0] in ("green", "cycle_budget") else edge_owner[label[1]]

    variable_owner = np.asarray([owner(label) for label in problem.labels])
    row_owner = np.asarray([owner(label) for label in problem.row_labels])
    matrices, H = problem.matrices()
    standard = problem.standard_form(matrices, H, variable_owner, row_owner)
    constant = 0.5 * green_weight * sum(p[col] ** 2 for col in pi["previous_green"].values())

    # Red-delay term of Eq. (traffic-cost), as in v3 augmented_model with R = C - gamma g (f_l = 0, rho_c = 1).
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
            cyc, cal = calendar[tls]
            for start, end, cycle in cal:
                if cycle is None:                                   # fixed running cycle: constant, left out
                    continue
                rows.append({"signal": tls, "incoming_edge": edge, "destination_edge": group["destination_edge"],
                             "cycle": cycle, "columns": [greens[tls, cycle, ph] for ph in stage_phases[tls]],
                             "gamma": gamma.tolist(), "cycle_seconds": cyc,
                             "fixed_service_seconds": 0.0, "movement_arrival_veh_per_second": movement_arrival,
                             "beta": beta, "coverage_weight": v3_model.overlap(0.0, horizon, start, end) / cyc,
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
            "indices": {"inventory": inventory, "outflow": outflow, "movement_queue": queue,
                        "external_wait": wait, "admission": admit,
                        "movement_outflow": move, "green": greens},
            "metadata": {"prediction_step_seconds": DT, "horizon_cycles": d.CYCLES,
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
    total += sum(0.5 * x[i] ** 2 / meta["capacity"][edge]
                 for (edge, y, k), i in instance["indices"]["movement_queue"].items() if k)
    greens = instance["indices"]["green"]
    for (tls, cycle, phase), i in greens.items():
        before = x[greens[tls, cycle - 1, phase]] if cycle else p[instance["parameter_indices"]["previous_green"][tls, phase]]
        total += 0.5 * meta["green_change_weight"] * (x[i] - before) ** 2
    for row in instance["red_delay_rows"]:
        red = row["cycle_seconds"] - row["fixed_service_seconds"] - np.asarray(row["gamma"]) @ x[row["columns"]]
        total += 0.5 * meta["mu"] * row["coverage_weight"] * row["beta"] * red ** 2
    return float(total)
