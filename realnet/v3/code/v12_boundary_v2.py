"""Fixed-boundary v12 controllers for a sparse, affine, node-local traffic QP.

All coefficients use historical design data. Failed construction and domain
checks stay failed. A diagnostic solve does not acquire a theoretical guarantee.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.linalg import qr
from scipy.optimize import linprog
from scipy.sparse.linalg import splu

from strict_qp import solve_qp
from active_basis_support import extend_active_basis


TOL = 1e-7


def _max_abs(value):
    return float(np.max(np.abs(value), initial=0.))


def _sparse_max(matrix):
    return _max_abs(matrix.data)


def vectors(data, p):
    return tuple(np.asarray(data[key + "0"] + data[key + "P"] @ p).ravel()
                 for key in ("h", "e", "g"))


def solve_affine(data, p):
    h, e, g = vectors(data, p)
    return solve_qp(data["H"], h, data["E"], e, data["G"], g)


def reduction(model, target):
    """Retain one node and all equalities touching its decision variables."""
    owner = np.asarray(model["variable_owner"])
    core, outside = np.flatnonzero(owner == target), np.flatnonzero(owner != target)
    if not len(core) or not len(outside):
        raise ValueError("The target must have a nonempty core and exterior.")
    E, G = model["E"].tocsr(), model["G"].tocsr()
    retained = np.asarray(E[:, core].getnnz(axis=1)).ravel() > 0
    B_all = E[retained][:, outside].tocsc()
    support = np.flatnonzero(np.asarray(B_all.getnnz(axis=0)).ravel())
    compact = B_all[:, support].toarray()
    if len(support):
        _, triangular, pivots = qr(compact.T, mode="economic", pivoting=True)
        diagonal = abs(np.diag(triangular))
        rank = int(np.sum(diagonal > 1e-10 * max(1., diagonal.max(initial=0))))
        chosen = pivots[:rank]
        B = B_all[chosen].tocsc()
        Fm = np.linalg.lstsq(compact[chosen].T, compact.T, rcond=1e-10)[0].T
    else:
        rank = 0
        B = sparse.csc_matrix((0, len(outside)))
        Fm = np.zeros((int(retained.sum()), 0))
    if _sparse_max(sparse.csc_matrix(Fm) @ B - B_all) > 1e-9:
        raise ValueError("The independent boundary coordinates do not reproduce the retained rows.")
    local = np.asarray(G[:, core].getnnz(axis=1)).ravel() > 0
    if _sparse_max(G[local][:, outside]) > 1e-12:
        raise ValueError("An inequality couples the core and exterior. This is outside the v12 reduction.")
    if _sparse_max(model["H"][core][:, outside]) > 1e-12:
        raise ValueError("The objective has a cross-node quadratic term.")
    Eo = sparse.vstack((E[~retained][:, outside], B), format="csc")
    Fo = sparse.vstack((sparse.csc_matrix((int((~retained).sum()), rank)),
                        -sparse.eye(rank)), format="csc")
    data = {
        "H": model["H"][outside][:, outside].tocsc(),
        "h0": model["h0"][outside], "hP": model["hP"][outside].tocsc(),
        "E": Eo, "F": Fo,
        "e0": np.r_[model["e0"][~retained], np.zeros(rank)],
        "eP": sparse.vstack((model["eP"][~retained],
                               sparse.csc_matrix((rank, len(model["p"])))), format="csc"),
        "G": G[~local][:, outside].tocsc(),
        "g0": model["g0"][~local], "gP": model["gP"][~local].tocsc(),
    }
    mine = np.asarray(model["param_owner"]) == target
    if any(_sparse_max(data[key][:, mine]) > 1e-12 for key in ("hP", "eP", "gP")):
        raise ValueError("The exterior problem contains a target-owned parameter.")
    if max(_sparse_max(model["eP"][retained][:, ~mine]),
           _sparse_max(model["gP"][local][:, ~mine])) > 1e-12:
        raise ValueError("A retained right-hand side uses an exterior parameter.")
    return {"core": core, "outside": outside, "boundary": B, "Fm": sparse.csc_matrix(Fm),
            "Em": E[retained][:, core].tocsc(),
            "em0": model["e0"][retained], "emP": model["eP"][retained].tocsc(),
            "Gm": G[local][:, core].tocsc(),
            "gm0": model["g0"][local], "gmP": model["gP"][local].tocsc(),
            "data": data}


def _basis_once(data, p0, *, solution=None, tolerance=TOL, active_tolerance=1e-5):
    """Find a nonnegative independent multiplier support, then verify all rows.

    Unpenalized flow directions use a feasible vertex and zero-dual active rows
    when needed. No regularization or change to the original QP is permitted.
    """
    start = time.perf_counter()
    H, E, G = (data[key].tocsc() for key in ("H", "E", "G"))
    h, e, g = vectors(data, p0)
    n, ne = H.shape[0], E.shape[0]
    answer = solve_qp(H, h, E, e, G, g) if solution is None else solution
    base = {"status": "FAIL", "tolerance": tolerance, "n_variables": n, "n_equalities": ne}
    x = np.asarray(answer.get("x", []))
    if x.shape != (n,) or not np.all(np.isfinite(x)):
        return {**base, "reason": "NO_FINITE_REFERENCE_SOLUTION"}
    input_residual = max(_max_abs(E @ x-e), float(np.max(G @ x-g, initial=0)))
    if input_residual > tolerance:
        return {**base, "reason": "REFERENCE_PRIMAL_RESIDUAL", "input_residual": input_residual}
    K0 = sparse.bmat([[H, E.T], [E, None]], format="csc")
    extensions = []
    try:
        probe = splu(K0)
        del probe
    except RuntimeError:
        extension = extend_active_basis(H, E, G, h, e, g, x, [])
        extensions.append({key: value for key, value in extension.items()
                           if key not in ("x", "active_rows", "added_rows", "K", "factor")})
        if extension["status"] != "PASS":
            return {**base, "reason": "ZERO_CURVATURE_VERTEX_FAILED", "extensions": extensions,
                    "seconds": time.perf_counter()-start}
        x = extension["x"]
    slack = g-G @ x
    active = np.flatnonzero(slack <= active_tolerance)
    normals = sparse.hstack((E.T, G[active].T), format="csc")
    gradient = H @ x+h
    lp_start = time.perf_counter()
    lp_attempts = []
    base["multiplier_lp_attempts"] = lp_attempts
    # Both positive equation scalings have the same exact feasible set.
    for row_scale in (100.0, 1.0):
        lp_tick = time.perf_counter()
        lp = linprog(np.r_[np.zeros(ne), np.ones(len(active))],
                     A_eq=row_scale * normals, b_eq=-row_scale * gradient,
                     bounds=[(None, None)]*ne+[(0., None)]*len(active), method="highs",
                     options={"presolve": False, "dual_feasibility_tolerance": 1e-9,
                              "primal_feasibility_tolerance": tolerance})
        lp_attempts.append({"stationarity_row_scale": row_scale, "success": bool(lp.success),
                            "status": int(lp.status), "message": lp.message,
                            "seconds": time.perf_counter()-lp_tick})
        if lp.success:
            break
    lp_seconds = time.perf_counter()-lp_start
    if not lp.success:
        return {**base, "reason": "NONNEGATIVE_MULTIPLIER_LP_FAILED", "lp_status": int(lp.status),
                "lp_message": lp.message, "lp_seconds": lp_seconds, "seconds": time.perf_counter()-start}
    selected = active[lp.x[ne:] > 1e-10]
    factor_seconds, refinements, visited = 0., [], set()
    converged = False
    for _ in range(300):
        signature = tuple(sorted(selected.tolist()))
        if signature in visited:
            break
        visited.add(signature)
        GB = G[selected]
        K = sparse.bmat([[H, E.T, GB.T], [E, None, None], [GB, None, None]], format="csc")
        factor_start = time.perf_counter()
        try:
            factor = splu(K)
        except RuntimeError:
            extension = extend_active_basis(H, E, G, h, e, g, x, selected)
            extensions.append({key: value for key, value in extension.items()
                               if key not in ("x", "active_rows", "added_rows", "K", "factor")})
            if extension["status"] != "PASS":
                return {**base, "reason": "SELECTED_KKT_EXTENSION_FAILED", "extensions": extensions,
                        "active_count": len(active), "support_count": len(selected),
                        "lp_seconds": lp_seconds, "seconds": time.perf_counter()-start}
            selected, K, factor = extension["active_rows"], extension["K"], extension["factor"]
            GB = G[selected]
        factor_seconds += time.perf_counter()-factor_start
        refined = factor.solve(np.r_[-h, e, g[selected]])
        xr, nu, lam = refined[:n], refined[n:n+ne], refined[n+ne:]
        full_lam = np.zeros(G.shape[0])
        full_lam[selected] = lam
        inequality = G @ xr-g
        residuals = {"equality": _max_abs(E @ xr-e),
                     "inequality": float(np.max(inequality, initial=0)),
                     "dual": float(np.max(-lam, initial=0)),
                     "stationarity": _max_abs(H @ xr+h+E.T @ nu+GB.T @ lam),
                     "complementarity": _max_abs(full_lam*inequality)}
        if max(residuals.values()) <= tolerance and residuals["dual"] <= 1e-10:
            converged = True
            break
        if residuals["dual"] > 1e-10:
            drop = lam < -1e-10
            refinements.append({"drop_negative_multiplier_rows": selected[drop].tolist(),
                                "minimum_multiplier": float(lam.min())})
            selected = selected[~drop]
        elif residuals["inequality"] > tolerance:
            row = int(np.argmax(inequality))
            if row in selected:
                break
            refinements.append({"add_violated_row": row, "residual": float(inequality[row])})
            selected = np.r_[selected, row]
        else:
            break
    if not converged or max(residuals.values()) > tolerance:
        return {**base, "reason": "REFINED_ORIGINAL_KKT_FAILED", "residuals": residuals,
                "active_set_refinements": refinements, "extensions": extensions,
                "seconds": time.perf_counter()-start}
    return {**base, "status": "PASS", "x": xr, "nu": nu, "lambda": lam,
            "active_rows": selected, "candidate_active_count": len(active),
            "support_count": len(selected), "factor": factor, "K": K,
            "factor_nonzeros": int(factor.L.nnz+factor.U.nnz),
            "residuals": residuals, "refinement_distance": _max_abs(xr-x),
            "active_set_refinements": refinements, "extensions": extensions,
            "lp_seconds": lp_seconds, "factor_seconds": factor_seconds,
            "seconds": time.perf_counter()-start}


def anchored_basis(data, p0, *, solution=None, tolerance=TOL):
    """Recover a nominal KKT basis using only the same historical exterior QP.

    Candidate tolerances choose rows. Acceptance always checks every original
    KKT condition at the fixed physical tolerance, with no QP regularization.
    """
    start = time.perf_counter()
    attempts = []
    external_record = None

    def record(result, active_tolerance, reference):
        attempts.append({"reference": reference, "candidate_active_tolerance": active_tolerance,
                         **{key: value for key, value in result.items()
                            if key not in ("x", "nu", "lambda", "factor", "K", "active_rows")}})

    def finish(result):
        return {**result, "recovery_attempts": attempts,
                "recovery_exterior_solve": external_record,
                "basis_method": "historical_exterior_KKT_recovery_v2",
                "seconds": time.perf_counter()-start}

    result = _basis_once(data, p0, solution=solution, tolerance=tolerance)
    record(result, 1e-5, "SUPPLIED_NOMINAL_REFERENCE")
    if result["status"] == "PASS":
        return finish(result)
    external = solve_affine(data, p0)
    external_record = {key: external.get(key) for key in
                       ("status", "valid", "seconds", "eq_violation", "ineq_violation",
                        "stationarity", "dual_violation", "complementarity")}
    if not external["valid"]:
        return finish({**result, "recovery_exhausted": True,
                       "recovery_failure": "HISTORICAL_EXTERIOR_QP_FAILED"})
    for active_tolerance in (1e-5, 1e-4, 1e-3):
        result = _basis_once(data, p0, solution=external, tolerance=tolerance,
                             active_tolerance=active_tolerance)
        record(result, active_tolerance, "TRUE_HISTORICAL_EXTERIOR_OPTIMUM")
        if result["status"] == "PASS":
            return finish(result)
    return finish({**result, "recovery_exhausted": True})


def affine_reference(data, p0, lower, upper, basis, *, tolerance=TOL):
    """Check the fixed KKT continuation on the entire declared parameter box."""
    if basis["status"] != "PASS":
        return {"status": "FAIL", "reason": "NO_VALID_ANCHOR_BASIS"}
    p0, lower, upper = (np.asarray(a, float) for a in (p0, lower, upper))
    if np.any(lower > upper) or np.any(p0 < lower-1e-12) or np.any(p0 > upper+1e-12):
        raise ValueError("The parameter box must contain the frozen nominal parameter.")
    columns = np.flatnonzero(upper > lower)
    center = (lower+upper)/2
    radius = (upper[columns]-lower[columns])/2
    n, ne = data["H"].shape[0], data["E"].shape[0]
    active = basis["active_rows"]
    rhs = sparse.vstack((-data["hP"][:, columns], data["eP"][:, columns],
                          data["gP"][active][:, columns]), format="csc")
    start = time.perf_counter()
    response = basis["factor"].solve(rhs.toarray()) if len(columns) else np.zeros((basis["K"].shape[0], 0))
    xp, nup, lamp = response[:n], response[n:n+ne], response[n+ne:]
    xc = basis["x"]+xp @ (center[columns]-p0[columns])
    nuc = basis["nu"]+nup @ (center[columns]-p0[columns])
    lamc = basis["lambda"]+lamp @ (center[columns]-p0[columns])
    h, e, g = vectors(data, center)
    eqd = data["E"] @ xp-data["eP"][:, columns].toarray()
    gd = data["G"] @ xp-data["gP"][:, columns].toarray()
    sd = (data["H"] @ xp+data["hP"][:, columns].toarray()+data["E"].T @ nup+
          data["G"][active].T @ lamp)
    maxima = {
        "equality": _max_abs(abs(data["E"] @ xc-e)+abs(eqd) @ radius),
        "inequality": float(np.max(data["G"] @ xc-g+abs(gd) @ radius, initial=0)),
        "dual": float(np.max(-lamc+abs(lamp) @ radius, initial=0)),
        "stationarity": _max_abs(abs(data["H"] @ xc+h+data["E"].T @ nuc+
                                           data["G"][active].T @ lamc)+abs(sd) @ radius),
    }
    return {"status": "PASS" if max(maxima.values()) <= tolerance else "FAIL",
            "tolerance": tolerance, "residuals": maxima, "varying_columns": columns,
            "x": basis["x"], "xP": xp, "lambdaP": lamp,
            "center_x": xc, "parameter_radius": radius,
            "parameter_center": center, "seconds": time.perf_counter()-start}


def freeze_reference(model, p0, lower, upper, nominal_full):
    basis = anchored_basis(model, p0, solution=nominal_full)
    check = affine_reference(model, p0, lower, upper, basis)
    return {"basis": basis, "check": check}


def construct_target(model, target, p0, lower, upper, nominal_full, design_full,
                     *, full_reference=None, tolerance_seconds=2., boundary_padding=0.):
    """Freeze a target's v12 coefficients before any evaluation state is read."""
    start = time.perf_counter()
    p0, lower, upper = (np.asarray(a, float) for a in (p0, lower, upper))
    base = {"status": "FAIL", "target": target, "numeric_conditions_pass": False}
    try:
        red = reduction(model, target)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {**base, "reason": str(exc), "seconds": time.perf_counter()-start}
    d = red["data"]
    s0 = np.asarray(red["boundary"] @ nominal_full["x"][red["outside"]]).ravel()
    anchor = {**d, "e0": d["e0"]-d["F"] @ s0}
    # A centralized optimum is an exterior optimum at its own boundary.
    supplied = {"x": nominal_full["x"][red["outside"]]}
    basis = anchored_basis(anchor, p0, solution=supplied)
    if basis["status"] != "PASS":
        return {**base, "reason": "EXTERIOR_BASIS_FAILED", "basis": basis,
                "seconds": time.perf_counter()-start}
    ref = affine_reference(anchor, p0, lower, upper, basis)
    if full_reference and full_reference["check"]["status"] == "PASS":
        full = full_reference["check"]
        sc = red["boundary"] @ full["center_x"][red["outside"]]
        sp = red["boundary"] @ full["xP"][red["outside"]]
        sr = abs(sp) @ full["parameter_radius"]
        slo, shi = sc-sr, sc+sr
        coverage = "NUMERIC_FULL_KKT_BOX_PASS"
    else:
        points = [s0]
        points.extend(np.asarray(red["boundary"] @ row["x"][red["outside"]]).ravel()
                      for row in design_full if row.get("valid", False))
        slo, shi = np.min(points, axis=0), np.max(points, axis=0)
        coverage = "HISTORICAL_DESIGN_BOUNDARIES_ONLY"
    slo = np.minimum(slo, s0)-boundary_padding
    shi = np.maximum(shi, s0)+boundary_padding
    n, ne, ns = len(red["outside"]), d["E"].shape[0], len(s0)
    active = basis["active_rows"]
    Bs = sparse.vstack((sparse.csc_matrix((n, ns)), d["F"],
                        sparse.csc_matrix((len(active), ns))), format="csc")
    adjoint = basis["factor"].solve(Bs.toarray(), trans="T") if ns else np.zeros((basis["K"].shape[0], 0))
    Ds = -adjoint[:n]
    Q = Ds.T @ (d["H"] @ Ds)
    Q = (Q+Q.T)/2
    Bp = sparse.vstack((-d["hP"], d["eP"], d["gP"][active]), format="csc")
    M = np.asarray(Bp.T @ adjoint).T
    v0 = np.asarray(d["F"].T @ basis["nu"]).ravel()
    curvature_identity_error = _max_abs(Q+Bs.T @ adjoint)
    if curvature_identity_error > 1e-7 or np.linalg.eigvalsh(Q).min(initial=0) < -1e-9:
        return {**base, "reason": "BOUNDARY_CURVATURE_IDENTITY_FAILED",
                "identity_error": curvature_identity_error, "seconds": time.perf_counter()-start}
    center_s, radius_s_box = (slo+shi)/2, (shi-slo)/2
    dg = d["G"] @ Ds
    de = d["E"] @ Ds+d["F"].toarray()
    columns = ref["varying_columns"]
    pcenter = ref["parameter_center"]
    pradius = ref["parameter_radius"]
    _, ep, gp = vectors(d, pcenter)
    xc = ref["center_x"]+Ds @ (center_s-s0)
    gdp = d["G"] @ ref["xP"]-d["gP"][:, columns].toarray()
    edp = d["E"] @ ref["xP"]-d["eP"][:, columns].toarray()
    plan_residuals = {
        "equality": _max_abs(abs(d["E"] @ xc+d["F"] @ center_s-ep)+
                                   abs(edp) @ pradius+abs(de) @ radius_s_box),
        "inequality": float(np.max(d["G"] @ xc-gp+abs(gdp) @ pradius+
                                              abs(dg) @ radius_s_box, initial=0)),
    }
    plan = {"status": "PASS" if max(plan_residuals.values()) <= TOL else "FAIL",
            "tolerance": TOL, "residuals": plan_residuals}
    delta_s = float(np.linalg.norm(np.maximum(abs(slo-s0), abs(shi-s0))))
    delta_full = .5*(float(np.linalg.norm(Q, 2)) if ns else 0.)*delta_s**2
    owners = np.asarray(model["param_owner"])
    nodes = sorted(set(owners.tolist()))
    contribution = {}
    for node in nodes:
        cols = np.flatnonzero(owners == node)
        radius_p = np.linalg.norm(np.maximum(abs(np.asarray(lower)[cols]-p0[cols]),
                                            abs(np.asarray(upper)[cols]-p0[cols])))
        contribution[node] = (delta_s*radius_p*np.linalg.norm(M[:, cols], 2)
                              if ns and node != target else 0.)
    ranked = [target]+sorted((node for node in nodes if node != target),
                            key=lambda node: (-contribution[node], node))
    kappa = float(model["metadata"]["first_input_curvature"])
    numeric_pass = coverage == "NUMERIC_FULL_KKT_BOX_PASS" and ref["status"] == plan["status"] == "PASS"
    prefix, accepted = [target], None
    for node in [None]+ranked[1:]:
        if node is not None:
            prefix.append(node)
        error = delta_full+sum(contribution[j] for j in nodes if j not in prefix)
        if numeric_pass and error <= kappa*tolerance_seconds**2/4:
            accepted = prefix.copy()
            break
    report = {"status": "CONSTRUCTED", "target": target, "numeric_conditions_pass": numeric_pass,
              "coverage": coverage, "reference": {key: ref[key] for key in ("status", "residuals", "tolerance", "seconds")},
              "plan": plan, "s0": s0, "slo": slo, "shi": shi, "Q": Q, "M": M, "v0": v0,
              "delta_s": delta_s, "delta_full": float(delta_full), "contribution": contribution,
              "ranked": ranked, "bound_C": accepted, "full_C": nodes, "kappa": kappa,
              "boundary_dimension": ns, "core_dimension": len(red["core"]),
              "basis": {key: basis[key] for key in ("status", "residuals", "support_count", "candidate_active_count",
                        "factor_nonzeros", "lp_seconds", "factor_seconds", "seconds", "refinement_distance",
                        "active_rows", "active_set_refinements", "extensions", "recovery_attempts",
                        "recovery_exterior_solve", "basis_method")},
              "curvature_identity_error": curvature_identity_error,
              "seconds": time.perf_counter()-start,
              "domain_scope": "Floating point affine-box checks at one frozen public clock, not an interval proof."}
    return {**report, "record": report, "reduction": red, "p0": np.asarray(p0).copy(),
            "p_lower": np.asarray(lower).copy(), "p_upper": np.asarray(upper).copy(),
            "param_owner": owners.copy(), "input_indices": np.asarray(model["output_indices"][target]),
            "local_H": model["H"][red["core"]][:, red["core"]].tocsc(),
            "local_h0": model["h0"][red["core"]], "local_hP": model["hP"][red["core"]].tocsc()}


def delta(controller, information):
    return controller["delta_full"]+sum(value for node, value in controller["contribution"].items()
                                        if node not in information)


def local_action(controller, p, information, *, allow_unverified=False):
    """Evaluate frozen coefficients. No coefficient uses an omitted current parameter."""
    if controller["status"] != "CONSTRUCTED":
        return {"valid": False, "status": "CONSTRUCTION_FAILED"}
    if not controller["numeric_conditions_pass"] and not allow_unverified:
        return {"valid": False, "status": "UNVERIFIED_CONSTRUCTION_REJECTED"}
    if controller["target"] not in information or not set(information) <= set(controller["full_C"]):
        raise ValueError("The information set must contain the target and only declared nodes.")
    current = controller["p0"].copy()
    selected = np.isin(controller["param_owner"], information)
    current[selected] = np.asarray(p)[selected]
    red = controller["reduction"]
    n, ns = len(red["core"]), len(controller["s0"])
    v = controller["v0"]+controller["M"] @ (current-controller["p0"])
    H = sparse.block_diag((controller["local_H"], sparse.csc_matrix(controller["Q"])), format="csc")
    h = np.r_[controller["local_h0"]+controller["local_hP"] @ current,
              v-controller["Q"] @ controller["s0"]]
    E = sparse.hstack((red["Em"], red["Fm"]), format="csc")
    G = sparse.vstack((sparse.hstack((red["Gm"], sparse.csc_matrix((red["Gm"].shape[0], ns)))),
                       sparse.hstack((sparse.csc_matrix((ns, n)), sparse.eye(ns))),
                       sparse.hstack((sparse.csc_matrix((ns, n)), -sparse.eye(ns)))), format="csc")
    result = solve_qp(H, h, E, red["em0"]+red["emP"] @ current, G,
                      np.r_[red["gm0"]+red["gmP"] @ current, controller["shi"], -controller["slo"]])
    in_box = bool(np.all(np.asarray(p) >= controller["p_lower"]-1e-9) and
                  np.all(np.asarray(p) <= controller["p_upper"]+1e-9))
    result.update(numeric_conditions_pass=controller["numeric_conditions_pass"],
                  diagnostic_unverified=not controller["numeric_conditions_pass"],
                  in_parameter_box=in_box,
                  theoretical_bound_applicable=controller["numeric_conditions_pass"] and in_box,
                  delta=delta(controller, information),
                  action_bound=2*np.sqrt(delta(controller, information)/controller["kappa"]))
    if result["valid"]:
        result["action"] = result["x"][np.searchsorted(red["core"], controller["input_indices"])]
        result["local_x"], result["s"] = result["x"][:n], result["x"][n:]
    return result


def true_exterior(controller, p, s):
    d = controller["reduction"]["data"]
    h, e, g = vectors(d, p)
    return solve_qp(d["H"], h, d["E"], e-d["F"] @ s, d["G"], g)
