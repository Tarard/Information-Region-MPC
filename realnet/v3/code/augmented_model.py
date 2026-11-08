"""Add a fixed historical movement-delay penalty to the saved LYG QP."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy import sparse


ROOT = Path(__file__).resolve().parents[3]
_OLD_PATH = ROOT / "outputs/LYG_dynamic_claude_v1/code/aligned_model.py"
_SPEC = importlib.util.spec_from_file_location("lyg_original_aligned_model", _OLD_PATH)
old_model = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(old_model)


def budget_basis(size):
    """Return an orthonormal basis for vectors whose entries sum to zero."""
    if size < 2:
        return np.empty((size, 0))
    return np.linalg.qr(np.eye(size)[:, :-1] - np.eye(size)[:, [-1]])[0]


def delay_rows(network, instance, historical_arrival_rates, service_groups):
    """Use movement service rates and the saved public native-cycle calendar."""
    signals = service_groups["signals"]
    result = []
    for tls, signal in sorted(network["signals"].items()):
        phases = signal["green_native_indices"]
        cycle_seconds = float(signal["cycle_seconds"])
        weights = np.zeros(3)
        for begin, end, cycle in instance["metadata"]["calendar"][tls]:
            if cycle is not None:
                weights[cycle] += (end - begin) / cycle_seconds
        for group in signals[tls]["movement_capacity_groups"]:
            edge = group["incoming_edge"]
            if edge not in historical_arrival_rates:
                raise ValueError(f"The frozen historical arrival rate is absent for {edge}.")
            arrival = float(historical_arrival_rates[edge])
            share = float(group["declared_turn_share"])
            saturation = float(group["saturation_rate_veh_per_second"])
            gamma = np.asarray(group["gamma"], dtype=float)
            fixed = float(group["fixed_phase_service_seconds_equivalent"])
            if (not np.isfinite(arrival) or arrival < 0 or not np.isfinite(saturation)
                    or saturation <= 0 or gamma.shape != (len(phases),)
                    or not np.isfinite(gamma).all() or np.any(gamma < 0)
                    or not np.isfinite(share) or not 0 <= share <= 1
                    or not np.isfinite(fixed) or not 0 <= fixed <= cycle_seconds):
                raise ValueError("A movement has an invalid rate, service map, or fixed service time.")
            movement_arrival = share * arrival
            beta = movement_arrival / (1 - min(movement_arrival / saturation, 0.9))
            for cycle, weight in enumerate(weights):
                columns = [instance["indices"]["green"][tls, cycle, phase] for phase in phases]
                result.append({
                    "signal": tls, "incoming_edge": edge,
                    "destination_edge": group["destination_edge"], "cycle": cycle,
                    "columns": columns, "gamma": gamma.tolist(),
                    "cycle_seconds": cycle_seconds, "fixed_service_seconds": fixed,
                    "historical_road_arrival_veh_per_second": arrival,
                    "turn_share": share, "movement_arrival_veh_per_second": movement_arrival,
                    "saturation_rate_veh_per_second": saturation,
                    "beta": beta, "coverage_weight": float(weight),
                    "flow_ratio_clipped": bool(movement_arrival / saturation > 0.9),
                })
    return result


def least_squares(instance, rows, mu):
    """Construct the objective as one half of a squared residual norm."""
    data, row_ids, col_ids, target = [], [], [], []

    def append(entries, value):
        row = len(target)
        for column, coefficient in entries:
            if coefficient:
                row_ids.append(row)
                col_ids.append(column)
                data.append(coefficient)
        target.append(value)

    capacity = instance["metadata"]["capacity"]
    for kind in ("inventory", "external_wait"):
        for (edge, step), column in instance["indices"][kind].items():
            if step:
                append([(column, 1 / np.sqrt(capacity[edge]))], 0.0)
    green = instance["indices"]["green"]
    scale = np.sqrt(instance["metadata"]["green_change_weight"])
    for (tls, cycle, phase), column in green.items():
        entries, value = [(column, scale)], 0.0
        if cycle:
            entries.append((green[tls, cycle - 1, phase], -scale))
        else:
            value = scale * instance["p"][instance["parameter_indices"]["previous_green"][tls, phase]]
        append(entries, value)
    for row in rows:
        scale = np.sqrt(mu * row["coverage_weight"] * row["beta"])
        if scale:
            append([(column, scale * value) for column, value in zip(row["columns"], row["gamma"])],
                   scale * (row["cycle_seconds"] - row["fixed_service_seconds"]))
    A = sparse.coo_matrix((data, (row_ids, col_ids)),
                         shape=(len(target), instance["H"].shape[0])).tocsc()
    return A, np.asarray(target)


def green_curvature(instance):
    """Eliminate later green vectors before computing first-input curvature."""
    results = {}
    H = instance["H"]
    for tls, first in instance["output_indices"].items():
        phases = [label[3] for label in instance["problem"].labels
                  if label[:3] == ("green", tls, 0)]
        z = budget_basis(len(first))
        d = z.shape[1]
        columns = [instance["indices"]["green"][tls, cycle, phase]
                   for cycle in range(3) for phase in phases]
        block = H[columns][:, columns].toarray()
        basis = sparse.block_diag([z] * 3).toarray()
        reduced = basis.T @ block @ basis
        if d:
            future = reduced[d:, d:]
            coupling = reduced[d:, :d]
            schur = reduced[:d, :d] - coupling.T @ np.linalg.solve(future, coupling)
            schur = (schur + schur.T) / 2
            eigenvalues, eigenvectors = np.linalg.eigh(schur)
            v0 = eigenvectors[:, 0]
            vf = -np.linalg.solve(future, coupling @ v0)
            witness = basis @ np.r_[v0, vf]
            kappa = float(eigenvalues[0])
            projected_inverse = z @ np.linalg.inv(schur) @ z.T
            component_constants = np.diag(projected_inverse)
            witness_full = np.zeros(H.shape[0])
            witness_full[columns] = witness
            eq_residual = float(np.max(np.abs(instance["E"] @ witness_full), initial=0.0))
            ratio = float(witness @ block @ witness / (witness[:len(first)] @ witness[:len(first)]))
        else:
            schur = np.empty((0, 0))
            eigenvalues = np.empty(0)
            kappa, eq_residual, ratio = float("inf"), 0.0, float("inf")
            component_constants = np.zeros(len(first))
        results[tls] = {"phase_count": len(first), "kappa": kappa,
                        "ratio_to_old_green_weight": kappa / instance["metadata"]["green_change_weight"],
                        "first_input_schur_eigenvalues": eigenvalues.tolist(),
                        "first_input_schur_matrix": schur.tolist(),
                        "orthonormal_budget_basis": z.tolist(),
                        "coordinate_inverse_constants": component_constants.tolist(),
                        "sharp_direction_equality_residual": eq_residual,
                        "sharp_direction_curvature_ratio": ratio}
    return results


def build(network, snapshot, historical_arrival_rates, mu, *, service_groups=None,
          edge_owner=None, horizon=480, green_change_weight=old_model.GREEN_CHANGE_WEIGHT):
    """Keep all original constraints and add a frozen historical quadratic cost."""
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("The added cost weight must be finite and nonnegative.")
    if service_groups is None:
        source = Path(__file__).resolve().parents[1] / "results/service_groups.json"
        service_groups = json.loads(source.read_text(encoding="utf-8-sig"))
    instance = old_model.build(network, snapshot, horizon,
                               green_change_weight=green_change_weight, edge_owner=edge_owner)
    rows = delay_rows(network, instance, historical_arrival_rates, service_groups)
    n = instance["H"].shape[0]
    data, rr, cc, b = [], [], [], []
    for row in rows:
        scale = np.sqrt(mu * row["coverage_weight"] * row["beta"])
        j = len(b)
        for column, coefficient in zip(row["columns"], row["gamma"]):
            if coefficient and scale:
                data.append(scale * coefficient)
                rr.append(j)
                cc.append(column)
        b.append(scale * (row["cycle_seconds"] - row["fixed_service_seconds"]))
    Aadd = sparse.coo_matrix((data, (rr, cc)), shape=(len(rows), n)).tocsc()
    badd = np.asarray(b)
    H = (instance["H"] + Aadd.T @ Aadd).tocsc()
    h = instance["h"] - Aadd.T @ badd
    instance.update(H=H, h=np.asarray(h), h0=np.asarray(h) - instance["hP"] @ instance["p"],
                    objective_constant=float(instance["objective_constant"] + 0.5 * badd @ badd),
                    red_delay_rows=rows, red_delay_mu=float(mu),
                    red_delay_A=Aadd, red_delay_b=badd)
    problem = instance["problem"]
    problem.P = H.diagonal().tolist()
    problem.q = instance["h"].tolist()
    upper_cross = sparse.triu(H, k=1).tocoo()
    problem.cross.clear()
    problem.cross.update({(int(i), int(j)): float(value)
                          for i, j, value in zip(upper_cross.row, upper_cross.col, upper_cross.data)})
    instance["matrices"] = {**instance["matrices"], "P": sparse.triu(H, format="csc"), "q": instance["h"]}
    instance["least_squares_A"], instance["least_squares_b"] = least_squares(instance, rows, mu)
    instance["green_curvature"] = green_curvature(instance)
    instance["metadata"] = {**instance["metadata"],
        "objective_extension": "fixed historical movement arrivals and effective-green quadratic penalty",
        "arrival_estimator": "caller supplies a frozen independently observed historical road arrival table",
        "first_input_curvature": {tls: value["kappa"] for tls, value in instance["green_curvature"].items()}}
    return instance


def direct_cost(instance, x):
    """Evaluate the old cost and the added movement cost, including constants."""
    x = np.asarray(x, dtype=float)
    value = old_model.direct_cost(instance, x)
    for row in instance["red_delay_rows"]:
        red = row["cycle_seconds"] - row["fixed_service_seconds"] - np.asarray(row["gamma"]) @ x[row["columns"]]
        value += 0.5 * instance["red_delay_mu"] * row["coverage_weight"] * row["beta"] * red**2
    return float(value)
