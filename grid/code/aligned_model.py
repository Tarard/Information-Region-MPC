"""LYG store-and-forward QP with the v12 consecutive-green cost.

The traffic constraints follow classic_model.py from the saved LYG preflight.
Native cycles, external waiting, admission, and turn capacities remain extensions
of the paper's common-cycle model. No route or future-demand file is read here.
"""

from collections import defaultdict
import math

import numpy as np
from scipy import sparse


DT = 60
GREEN_CHANGE_WEIGHT = 1e-4


def overlap(a, b, c, d):
    return max(0.0, min(b, d) - max(a, c))


def calendar(signal, current, horizon):
    cycle = float(signal["cycle_seconds"])
    remaining = float(current["remaining_cycle_seconds"])
    if not 0 <= remaining <= cycle + 1e-8:
        raise ValueError("The current cycle remainder is outside its native cycle.")
    result = [(0.0, remaining, None)] if remaining else []
    free = []
    start = remaining
    while start + cycle <= horizon + 1e-8:
        if len(free) < 3:
            free.append(start)
        result.append((start, start + cycle, len(free) - 1))
        start += cycle
    if not free:
        raise ValueError("The horizon does not cover one complete free native cycle.")
    if start < horizon:
        result.append((start, float(horizon), len(free) - 1))
    return result, free


class Problem:
    """Sparse quadratic objective and affine physical constraint records."""

    def __init__(self, parameter):
        self.parameter = np.asarray(parameter, dtype=float)
        self.labels, self.lo, self.hi, self.P, self.q = [], [], [], [], []
        self.rows, self.lower, self.upper, self.row_labels = [], [], [], []
        self.lo_derivative, self.hi_derivative = [], []
        self.lower_derivative, self.upper_derivative = [], []
        self.q_derivative = []
        self.cross = defaultdict(float)

    def variable(self, label, lower=0.0, upper=math.inf, weight=0.0,
                 center=0.0, linear=0.0, *, fixed_parameter=None):
        index = len(self.labels)
        self.labels.append(label)
        self.lo.append(float(lower))
        self.hi.append(float(upper))
        self.P.append(2.0 * weight)
        self.q.append(linear - 2.0 * weight * center)
        derivative = {} if fixed_parameter is None else {fixed_parameter: 1.0}
        self.lo_derivative.append(derivative)
        self.hi_derivative.append(derivative)
        self.q_derivative.append({})
        return index

    def constraint(self, label, entries, lower=-math.inf, upper=math.inf,
                   *, lower_derivative=None, upper_derivative=None):
        row = defaultdict(float)
        for index, value in entries:
            row[index] += value
        self.rows.append({index: value for index, value in row.items() if value})
        self.lower.append(float(lower))
        self.upper.append(float(upper))
        self.row_labels.append(label)
        self.lower_derivative.append(lower_derivative or {})
        self.upper_derivative.append(upper_derivative or {})

    def difference_penalty(self, current, weight, *, previous=None, parameter=None):
        self.P[current] += weight
        if previous is None:
            self.q[current] -= weight * self.parameter[parameter]
            self.q_derivative[current][parameter] = -weight
        else:
            self.P[previous] += weight
            self.cross[min(current, previous), max(current, previous)] -= weight

    @staticmethod
    def _records_matrix(records, columns):
        data, rows, cols = [], [], []
        for row, entries in enumerate(records):
            for col, value in entries.items():
                rows.append(row)
                cols.append(col)
                data.append(value)
        return sparse.coo_matrix((data, (rows, cols)), shape=(len(records), columns)).tocsc()

    def matrices(self):
        n = len(self.labels)
        H = sparse.diags(self.P, format="csc")
        if self.cross:
            pairs = list(self.cross)
            cross = sparse.coo_matrix(
                (list(self.cross.values()), ([a for a, b in pairs], [b for a, b in pairs])),
                shape=(n, n),
            ).tocsc()
            H = H + cross + cross.T
        physical = self._records_matrix(self.rows, n)
        A = sparse.vstack((physical, sparse.eye(n, format="csc")), format="csc")
        return {"P": sparse.triu(H, format="csc"), "q": np.asarray(self.q),
                "A": A, "l": np.asarray(self.lower + self.lo),
                "u": np.asarray(self.upper + self.hi)}, H

    def standard_form(self, matrices, H, variable_owner, row_owner):
        A, lower, upper = matrices["A"], matrices["l"], matrices["u"]
        all_row_owner = np.r_[row_owner, variable_owner]
        lower_P = self._records_matrix(self.lower_derivative + self.lo_derivative, len(self.parameter))
        upper_P = self._records_matrix(self.upper_derivative + self.hi_derivative, len(self.parameter))
        equal = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
        if (lower_P[equal] - upper_P[equal]).nnz:
            raise ValueError("The two sides of an equality have different parameter maps.")
        upper_rows = np.flatnonzero(np.isfinite(upper) & ~equal)
        lower_rows = np.flatnonzero(np.isfinite(lower) & ~equal)
        E = A[equal].tocsc()
        e = upper[equal]
        eP = upper_P[equal].tocsc()
        G = sparse.vstack((A[upper_rows], -A[lower_rows]), format="csc")
        g = np.r_[upper[upper_rows], -lower[lower_rows]]
        gP = sparse.vstack((upper_P[upper_rows], -lower_P[lower_rows]), format="csc")
        h = matrices["q"]
        hP = self._records_matrix(self.q_derivative, len(self.parameter))
        return {"H": H, "h": h, "E": E, "e": e, "G": G, "g": g,
                "hP": hP, "eP": eP, "gP": gP,
                "h0": h - hP @ self.parameter,
                "e0": e - eP @ self.parameter,
                "g0": g - gP @ self.parameter,
                "eq_owner": all_row_owner[equal],
                "ineq_owner": np.r_[all_row_owner[upper_rows], all_row_owner[lower_rows]],
                "equality_osqp_rows": np.flatnonzero(equal),
                "inequality_osqp_rows": np.r_[upper_rows, lower_rows],
                "inequality_osqp_sign": np.r_[np.ones(len(upper_rows)), -np.ones(len(lower_rows))]}


def build(network, snapshot, horizon=480, *, green_change_weight=GREEN_CHANGE_WEIGHT,
          edge_owner=None):
    """Build one public-clock template. All traffic parameters remain affine."""
    if horizon % DT or horizon <= 0 or green_change_weight <= 0:
        raise ValueError("Use a positive multiple of 60 seconds and a positive green weight.")
    roads, signals = network["roads"], network["signals"]
    edges = sorted(roads)
    entries, exits = sorted(network["entry_edges"]), set(network["exit_edges"])
    if edge_owner is None or set(edge_owner) != set(edges):
        raise ValueError("Supply exactly one declared signal owner for every physical road.")
    if set(edge_owner.values()) - set(signals):
        raise ValueError("A road owner is absent from the native signal model.")
    steps = horizon // DT
    capacity = {e: sum(math.floor(float(lane["length"]) / 7.5) + 1
                       for lane in roads[e]["lanes"]) for e in edges}
    parameter, parameter_labels, parameter_owner = [], [], []
    pi = {name: {} for name in ("n0", "backlog", "release_rate", "previous_green", "committed_green")}

    def add_parameter(kind, key, value, owner):
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"The current {kind} parameter must be finite and nonnegative.")
        pi[kind][key] = len(parameter)
        parameter.append(value)
        parameter_labels.append((kind, *key) if isinstance(key, tuple) else (kind, key))
        parameter_owner.append(owner)

    for tls in sorted(signals):
        for edge in edges:
            if edge_owner[edge] == tls:
                add_parameter("n0", edge, snapshot["edge_vehicle_count"][edge], tls)
        for edge in entries:
            if edge_owner[edge] == tls:
                add_parameter("backlog", edge, snapshot["boundary_backlog_vehicles"][edge], tls)
                add_parameter("release_rate", edge, snapshot["boundary_release_rate_veh_per_second"][edge], tls)
        signal, current = signals[tls], snapshot["signals"][tls]
        phases = signal["green_native_indices"]
        previous = current["previous_executed_green_seconds"]
        committed = current["committed_green_seconds"]
        if len(previous) != len(phases) or len(committed) != len(phases):
            raise ValueError("The observed green vectors do not match the native phases.")
        budget = float(signal["available_green_seconds"])
        for values in (previous, committed):
            if abs(sum(map(float, values)) - budget) > 1e-6:
                raise ValueError("The observed green vector does not preserve the native budget.")
            for value, nominal in zip(values, signal["nominal_green_seconds"]):
                minimum = 6 if float(nominal) < 10 else 10
                if not minimum - 1e-6 <= float(value) <= 60 + 1e-6:
                    raise ValueError("The observed green vector violates the declared green bounds.")
        for ordinal, phase in enumerate(phases):
            add_parameter("previous_green", (tls, phase), previous[ordinal], tls)
            add_parameter("committed_green", (tls, phase), committed[ordinal], tls)

    p = np.asarray(parameter)
    problem = Problem(p)
    inventory, outflow, backlog, admission, greens, first = {}, {}, {}, {}, {}, {}
    calendars = {}
    for tls, signal in sorted(signals.items()):
        segments, starts = calendar(signal, snapshot["signals"][tls], horizon)
        if len(starts) != 3:
            raise ValueError(f"The horizon does not cover three complete native cycles at {tls}.")
        calendars[tls] = segments
        for cycle in range(3):
            ids = []
            for ordinal, phase in enumerate(signal["green_native_indices"]):
                nominal = float(signal["nominal_green_seconds"][ordinal])
                index = problem.variable(("green", tls, cycle, phase), 6 if nominal < 10 else 10, 60)
                greens[tls, cycle, phase] = index
                problem.difference_penalty(index, green_change_weight,
                    previous=greens[tls, cycle - 1, phase] if cycle else None,
                    parameter=pi["previous_green"][tls, phase] if not cycle else None)
                ids.append(index)
            budget = float(signal["available_green_seconds"])
            problem.constraint(("cycle_budget", tls, cycle), [(index, 1) for index in ids], budget, budget)
            if cycle == 0:
                first[tls] = ids

    for edge in edges:
        initial = p[pi["n0"][edge]]
        for k in range(steps + 1):
            inventory[edge, k] = problem.variable(("inventory", edge, k),
                initial if k == 0 else 0, initial if k == 0 else capacity[edge],
                0 if k == 0 else 0.5 / capacity[edge],
                fixed_parameter=pi["n0"][edge] if k == 0 else None)
        for k in range(steps):
            outflow[edge, k] = problem.variable(("link_outflow", edge, k))
    for edge in entries:
        initial = p[pi["backlog"][edge]]
        for k in range(steps + 1):
            backlog[edge, k] = problem.variable(("external_wait", edge, k),
                initial if k == 0 else 0, initial if k == 0 else math.inf,
                0 if k == 0 else 0.5 / capacity[edge],
                fixed_parameter=pi["backlog"][edge] if k == 0 else None)
        for k in range(steps):
            admission[edge, k] = problem.variable(("admission", edge, k))
            release = DT * p[pi["release_rate"][edge]]
            derivative = {pi["release_rate"][edge]: DT}
            problem.constraint(("external_wait_conservation", edge, k),
                [(backlog[edge, k + 1], 1), (backlog[edge, k], -1), (admission[edge, k], 1)],
                release, release, lower_derivative=derivative, upper_derivative=derivative)
            problem.constraint(("previously_released_admission", edge, k),
                [(admission[edge, k], 1), (backlog[edge, k], -1)], upper=0)

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
        for k in range(steps):
            row = [(inventory[edge, k + 1], 1), (inventory[edge, k], -1), (outflow[edge, k], 1)]
            row.extend((outflow[source, k], -share) for source, share in incoming[edge])
            if edge in entries:
                row.append((admission[edge, k], -1))
            problem.constraint(("link_conservation", edge, k), row, 0, 0)

    for edge in edges:
        links = connections[edge]
        if not links:
            limit = DT * len(roads[edge]["lanes"]) * float(network["nonsignal_service_veh_per_lane_second"])
            for k in range(steps):
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
            if any(connection["tl"] != tls for connection in links) or edge_owner[edge] != tls:
                raise ValueError(f"The signal capacity at {edge} is not local to its declared owner.")
            signal = signals[tls]
            cycle_seconds = float(signal["cycle_seconds"])

            def capacity_row(k, phase_lanes, share):
                row, constant, derivative = [(outflow[edge, k], share)], 0.0, defaultdict(float)
                for phase, lanes in phase_lanes.items():
                    rate = sum(lanes.values())
                    for begin, end, native_cycle in calendars[tls]:
                        fraction = overlap(k * DT, (k + 1) * DT, begin, end) / cycle_seconds
                        if not fraction:
                            continue
                        if phase not in signal["green_native_indices"]:
                            constant += rate * fraction * float(signal["phases"][phase]["duration_seconds"])
                        elif native_cycle is None:
                            col = pi["committed_green"][tls, phase]
                            constant += rate * fraction * p[col]
                            derivative[col] += rate * fraction
                        else:
                            row.append((greens[tls, native_cycle, phase], -rate * fraction))
                return row, constant, dict(derivative)

            for k in range(steps):
                row, constant, derivative = capacity_row(k, all_lanes, 1)
                problem.constraint(("signal_link_capacity", edge, k), row,
                    upper=constant, upper_derivative=derivative)
                for target, phase_lanes in groups.items():
                    share = float(shares[target])
                    if share > 0:
                        row, constant, derivative = capacity_row(k, phase_lanes, share)
                        problem.constraint(("signal_turn_capacity", edge, target, k), row,
                            upper=constant, upper_derivative=derivative)
        else:
            total_rate = sum(all_lanes[None].values())
            for k in range(steps):
                problem.constraint(("nonsignal_link_capacity", edge, k),
                    [(outflow[edge, k], 1)], upper=DT * total_rate)
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
    constant = 0.5 * green_change_weight * sum(p[col] ** 2 for col in pi["previous_green"].values())
    return {**standard, "problem": problem, "matrices": matrices, "output_indices": first,
            "p": p, "param_labels": parameter_labels, "param_owner": np.asarray(parameter_owner),
            "parameter_indices": pi, "variable_owner": variable_owner, "row_owner": row_owner,
            "objective_constant": constant,
            "indices": {"inventory": inventory, "outflow": outflow, "external_wait": backlog,
                        "admission": admission, "green": greens},
            "metadata": {"prediction_step_seconds": DT, "horizon_seconds": horizon,
                "road_count": len(edges), "signal_count": len(signals),
                "variable_count": len(problem.labels), "constraint_count_including_boxes": matrices["A"].shape[0],
                "parameter_count": len(p), "capacity": capacity, "calendar": calendars,
                "green_change_weight": green_change_weight, "first_input_curvature": green_change_weight,
                "model": "LYG link-level store-and-forward with native-cycle overlap",
                "paper_cost": "0.5 sum n^2/storage + 0.5 w sum squared consecutive green changes",
                "traffic_extensions": ["external waiting and controlled admission", "asynchronous native cycles",
                    "60-second prediction intervals", "per-turn and total road service constraints",
                    "three future native cycles, last vector held over later intervals"],
                "previous_green": "explicit current committed actual cycle vector supplied by executor",
                "clock_template": "H,E fixed. G and constant bounds use the public current-cycle clock."
                    " Affine h/e/g parameter maps apply only at this fixed clock.",
                "turning": "fixed calibrated shares, including six retained zero empirical shares",
                "cost_constant": "0.5 w sum squared previous greens, omitted from the optimized objective"}}


def direct_cost(instance, x):
    """Evaluate the stated cost, including its parameter-only constant."""
    x, p = np.asarray(x), instance["p"]
    capacity = instance["metadata"]["capacity"]
    total = 0.0
    for kind in ("inventory", "external_wait"):
        for (edge, k), index in instance["indices"][kind].items():
            if k:
                total += 0.5 * x[index] ** 2 / capacity[edge]
    greens = instance["indices"]["green"]
    w = instance["metadata"]["green_change_weight"]
    for (tls, cycle, phase), index in greens.items():
        previous = (x[greens[tls, cycle - 1, phase]] if cycle else
                    p[instance["parameter_indices"]["previous_green"][tls, phase]])
        total += 0.5 * w * (x[index] - previous) ** 2
    return float(total)


def snapshot_with_parameters(snapshot, instance, p):
    """Make a counterfactual snapshot at the same public signal clock."""
    from copy import deepcopy
    result = deepcopy(snapshot)
    for label, value in zip(instance["param_labels"], np.asarray(p)):
        kind, key = label[:2]
        if kind == "n0":
            result["edge_vehicle_count"][key] = float(value)
        elif kind == "backlog":
            result["boundary_backlog_vehicles"][key] = float(value)
        elif kind == "release_rate":
            result["boundary_release_rate_veh_per_second"][key] = float(value)
        else:
            field = ("previous_executed_green_seconds" if kind == "previous_green" else "committed_green_seconds")
            native_phase = label[2]
            phases = [name[2] for name in instance["param_labels"] if name[:2] == (kind, key)]
            result["signals"][key][field][phases.index(native_phase)] = float(value)
    return result
