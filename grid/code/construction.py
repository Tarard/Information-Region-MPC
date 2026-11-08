"""Construct the original LYG v12 quadratic at one public signal clock.

Only the nominal KKT conditions are checked here. Historical boundary ranges
and information scores do not provide a uniform error guarantee.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse

from v12_boundary_v2 import anchored_basis, reduction


TOL = 1e-7


def historical_boundary_bounds(red, historical_x, xref, padding_fraction=0.1):
    """Contain the supplied historical boundaries and the nominal boundary."""
    historical_x = np.asarray(historical_x, dtype=float)
    xref = np.asarray(xref, dtype=float)
    if (historical_x.ndim != 2 or historical_x.shape[1:] != xref.shape
            or not len(historical_x) or not np.isfinite(historical_x).all()
            or not np.isfinite(xref).all() or padding_fraction < 0):
        raise ValueError("Supply finite historical solutions with the nominal dimension.")
    B, outside = red["boundary"], red["outside"]
    values = np.asarray(B @ np.vstack((historical_x, xref))[:, outside].T).T
    lo, hi = values.min(axis=0), values.max(axis=0)
    padding = float(padding_fraction) * (hi - lo)
    return lo - padding, hi + padding


def construct(model, target, pbar, xref, historical_p, historical_x,
              boundary_bounds=None):
    """Use history to freeze Q, M, and node scores at the current public clock."""
    started = time.perf_counter()
    pbar, xref = np.asarray(pbar, float), np.asarray(xref, float)
    history = np.asarray(historical_p, float)
    if (pbar.shape != np.asarray(model["p"]).shape or xref.shape != (model["H"].shape[0],)
            or history.ndim != 2 or history.shape[1:] != pbar.shape or not len(history)
            or not all(np.isfinite(value).all() for value in (pbar, xref, history))):
        raise ValueError("Historical parameters and the nominal solution must match the QP.")
    if target not in model["output_indices"]:
        raise ValueError("The target is absent from the traffic QP.")
    red = reduction(model, target)
    d = red["data"]
    s0 = np.asarray(red["boundary"] @ xref[red["outside"]]).ravel()
    if boundary_bounds is None:
        boundary_bounds = historical_boundary_bounds(red, historical_x, xref)
    lo, hi = (np.asarray(value, float).copy() for value in boundary_bounds)
    if (lo.shape != s0.shape or hi.shape != s0.shape or not np.isfinite(lo).all()
            or not np.isfinite(hi).all() or np.any(lo > hi)
            or np.any(s0 < lo - TOL) or np.any(s0 > hi + TOL)):
        raise ValueError("The finite boundary range must contain the nominal boundary.")
    anchor = {**d, "e0": d["e0"] - d["F"] @ s0}
    basis = anchored_basis(anchor, pbar, solution={"x": xref[red["outside"]]})
    if basis["status"] != "PASS":
        return {"status": "FAIL", "target": target, "reason": "NOMINAL_EXTERIOR_KKT_FAILED",
                "basis": {key: value for key, value in basis.items()
                          if key not in {"x", "nu", "lambda", "K", "factor"}},
                "seconds": time.perf_counter() - started}
    n, ne, ns = len(red["outside"]), d["E"].shape[0], len(s0)
    active = basis["active_rows"]
    Bs = sparse.vstack((sparse.csc_matrix((n, ns)), d["F"],
                        sparse.csc_matrix((len(active), ns))), format="csc")
    adjoint = basis["factor"].solve(Bs.toarray(), trans="T")
    Ds = -adjoint[:n]
    Q = Ds.T @ (d["H"] @ Ds)
    Q = (Q + Q.T) / 2
    Bp = sparse.vstack((-d["hP"], d["eP"], d["gP"][active]), format="csc")
    M = np.asarray(Bp.T @ adjoint).T
    vbar = np.asarray(d["F"].T @ basis["nu"]).ravel()
    identity = float(np.max(np.abs(Q + Bs.T @ adjoint), initial=0.))
    probes = np.random.default_rng(4271).normal(size=(basis["K"].shape[0], 3))
    inverse_residual = float(np.max(np.abs(
        basis["K"] @ basis["factor"].solve(probes) - probes), initial=0.))
    minimum_eigenvalue = float(np.linalg.eigvalsh(Q).min(initial=0.))
    checks = dict(basis["residuals"])
    checks.update(curvature_identity=identity, inverse_probe=inverse_residual,
                  negative_curvature=max(0., -minimum_eigenvalue))
    if not np.isfinite(list(checks.values())).all() or max(checks.values()) > TOL:
        return {"status": "FAIL", "target": target, "reason": "NOMINAL_KKT_MAP_FAILED",
                "checks": checks, "seconds": time.perf_counter() - started}
    owners = np.asarray(model["param_owner"])
    nodes = sorted(set(owners.tolist()))
    delta_s = float(np.linalg.norm(np.maximum(np.abs(lo - s0), np.abs(hi - s0))))
    b, norms, c = {}, {}, {}
    for node in nodes:
        columns = np.flatnonzero(owners == node)
        b[node] = float(np.max(np.linalg.norm(history[:, columns] - pbar[columns], axis=1)))
        norms[node] = float(np.linalg.norm(M[:, columns], 2)) if ns else 0.
        c[node] = 0. if node == target else delta_s * b[node] * norms[node]
    ranked = [target] + sorted((node for node in nodes if node != target),
                               key=lambda node: (-c[node], node))
    # Exterior matrices and KKT factors are not needed by the online local QP.
    light_red = {key: value for key, value in red.items() if key != "data"}
    record = {"status": "CONSTRUCTED", "target": target, "nominal_kkt_pass": True,
              "core_dimension": len(red["core"]), "boundary_dimension": ns,
              "exterior_dimension": n, "exterior_equalities": ne,
              "anchor_checks": checks, "selected_active_rows": len(active),
              "basis_seconds": float(basis["seconds"]), "delta_s": delta_s,
              "multiplier_lp_attempts": basis.get("multiplier_lp_attempts", []),
              "basis_recovery_attempts": [{key: attempt.get(key) for key in
                  ("reference", "candidate_active_tolerance", "status", "reason", "seconds", "multiplier_lp_attempts",
                   "multiplier_seed", "candidate_active_count", "support_count", "refinement_distance")}
                  for attempt in basis.get("recovery_attempts", [])],
              "boundary_scope": "Historical condition boundaries and nominal boundary, with 10% span padding.",
              "guarantee_scope": "Nominal KKT check only. No uniform-domain or 2-second certificate.",
              "seconds": time.perf_counter() - started}
    return {**record, "record": record, "reduction": light_red,
            "pbar": pbar.copy(), "xref": xref.copy(), "s0": s0, "Q": Q, "M": M,
            "vbar": vbar, "S_lo": lo, "S_hi": hi, "ranked": ranked,
            "c": c, "b": b, "Mnorm": norms, "param_owner": owners.copy(),
            "input_indices": np.asarray(model["output_indices"][target], dtype=int),
            "local_H": model["H"][red["core"]][:, red["core"]].tocsc(),
            "local_h0": model["h0"][red["core"]].copy(),
            "local_hP": model["hP"][red["core"]].tocsc()}
