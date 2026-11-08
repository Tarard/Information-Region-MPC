"""Solve sparse convex QPs and check absolute residuals in original units."""

from __future__ import annotations

import time

import clarabel
import numpy as np
import osqp
from scipy import sparse

TOLERANCE = 1e-7


def _maximum(values):
    return float(np.max(values, initial=0.0))


def symmetric_hessian(matrix):
    matrix = sparse.csc_matrix(matrix, dtype=float)
    lower = sparse.tril(matrix, -1)
    upper = sparse.triu(matrix, 1)
    if lower.nnz and upper.nnz:
        if _maximum(np.abs((matrix - matrix.T).data)) > 1e-12:
            raise ValueError("The Hessian must be symmetric or triangular.")
        return matrix
    return (matrix + matrix.T - sparse.diags(matrix.diagonal())).tocsc()


def residuals(matrices, x, y):
    P = symmetric_hessian(matrices["P"])
    A, q, lo, hi = (matrices[k] for k in ("A", "q", "l", "u"))
    Ax = np.asarray(A @ x)
    equality = np.isfinite(lo) & np.isfinite(hi) & (lo == hi)
    upper = np.isfinite(hi) & ~equality
    lower = np.isfinite(lo) & ~equality
    yp, yn = np.maximum(y, 0), np.maximum(-y, 0)
    complementarity = max(_maximum(np.abs(yp[upper] * (hi[upper] - Ax[upper]))),
                          _maximum(np.abs(yn[lower] * (Ax[lower] - lo[lower]))))
    return {
        "eq_violation": _maximum(np.abs(Ax[equality] - hi[equality])),
        "ineq_violation": max(_maximum(Ax[upper] - hi[upper]), _maximum(lo[lower] - Ax[lower])),
        "stationarity": _maximum(np.abs(P @ x + q + A.T @ y)),
        "dual_violation": max(_maximum(yn[~np.isfinite(lo)]), _maximum(yp[~np.isfinite(hi)])),
        "complementarity": complementarity,
    }


def _clarabel(matrices):
    A, lo, hi = (matrices[k] for k in ("A", "l", "u"))
    equality = np.flatnonzero(np.isfinite(lo) & np.isfinite(hi) & (lo == hi))
    upper = np.flatnonzero(np.isfinite(hi) & ~np.isin(np.arange(len(hi)), equality))
    lower = np.flatnonzero(np.isfinite(lo) & ~np.isin(np.arange(len(lo)), equality))
    blocks = [A[equality], A[upper], -A[lower]]
    bounds = np.r_[hi[equality], hi[upper], -lo[lower]]
    cones = [clarabel.ZeroConeT(len(equality)), clarabel.NonnegativeConeT(len(upper) + len(lower))]
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    settings.direct_solve_method = "qdldl"
    settings.max_iter = 300
    settings.tol_gap_abs = 1e-10
    settings.tol_gap_rel = 0.0
    settings.tol_feas = 1e-11
    settings.tol_ktratio = 1e-10
    settings.iterative_refinement_abstol = 1e-13
    settings.iterative_refinement_reltol = 1e-13
    solver = clarabel.DefaultSolver(sparse.triu(symmetric_hessian(matrices["P"]), format="csc"),
                                    matrices["q"], sparse.vstack(blocks, format="csc"), bounds, cones, settings)
    answer = solver.solve()
    z = np.asarray(answer.z)
    y = np.zeros(A.shape[0])
    a, b = len(equality), len(equality) + len(upper)
    y[equality] = z[:a]
    y[upper] += z[a:b]
    y[lower] -= z[b:]
    return np.asarray(answer.x), y, str(answer.status), int(answer.iterations)


def _osqp(matrices, warm):
    solver = osqp.OSQP()
    solver.setup(P=sparse.triu(symmetric_hessian(matrices["P"]), format="csc"),
                 q=matrices["q"], A=matrices["A"], l=matrices["l"], u=matrices["u"],
                 verbose=False, eps_abs=1e-9, eps_rel=0.0, max_iter=100000,
                 polishing=True, scaled_termination=False)
    if warm is not None and np.shape(warm[0]) == np.shape(matrices["q"]) and np.shape(warm[1]) == np.shape(matrices["l"]):
        solver.warm_start(x=warm[0], y=warm[1])
    answer = solver.solve(raise_error=False)
    return answer.x, answer.y, str(answer.info.status), int(answer.info.iter)


def solve_matrices(matrices, warm=None, *, primary="clarabel"):
    """Accept a solution only when five absolute KKT residuals pass."""
    data = {k: sparse.csc_matrix(matrices[k], dtype=float) if k in {"P", "A"}
            else np.asarray(matrices[k], dtype=float) for k in ("P", "q", "A", "l", "u")}
    n, m = len(data["q"]), len(data["l"])
    if data["P"].shape != (n, n) or data["A"].shape != (m, n) or data["u"].shape != (m,):
        raise ValueError("QP matrix dimensions differ.")
    if any(np.any(~np.isfinite(data[k].data)) for k in ("P", "A")) or not np.all(np.isfinite(data["q"])):
        raise ValueError("The QP contains nonfinite coefficients.")
    if np.any(np.isnan(data["l"])) or np.any(np.isnan(data["u"])) or np.any(data["l"] > data["u"]):
        raise ValueError("The QP bounds are invalid.")
    if primary not in {"clarabel", "osqp"}:
        raise ValueError("Choose clarabel or osqp.")
    started = time.perf_counter()
    attempts, result = [], {"valid": False, "x": None, "y": None, "obj": None, "status": "NO_SOLUTION"}
    for name in (primary, "osqp" if primary == "clarabel" else "clarabel"):
        tick = time.perf_counter()
        try:
            x, y, status, iterations = _clarabel(data) if name == "clarabel" else _osqp(data, warm)
            finite = x is not None and y is not None and np.all(np.isfinite(x)) and np.all(np.isfinite(y))
            checks = residuals(data, x, y) if finite else {}
            # The five residual checks decide acceptance. A Clarabel "AlmostSolved" label is accepted only when
            # all of them pass at TOLERANCE (approved 2026-10-24); the OSQP rule is unchanged.
            solved = status in {"Solved", "solved"} or (name == "clarabel" and status == "AlmostSolved")
            valid = finite and solved and max(checks.values(), default=float("inf")) <= TOLERANCE
            attempt = {"solver": name, "status": status, "iterations": iterations,
                       "seconds": time.perf_counter() - tick, "valid": bool(valid), **checks}
            attempts.append(attempt)
            result = {"valid": bool(valid), "x": x, "y": y, "status": status, "solver": name,
                      "obj": float(0.5 * x @ (symmetric_hessian(data["P"]) @ x) + data["q"] @ x) if finite else None,
                      **checks}
            if valid:
                break
        except Exception as exc:
            attempts.append({"solver": name, "status": "EXCEPTION", "exception": f"{type(exc).__name__}: {exc}",
                             "seconds": time.perf_counter() - tick, "valid": False})
    return {**result, "attempts": attempts, "seconds": time.perf_counter() - started,
            "absolute_tolerance": TOLERANCE}


def solve_qp(H, h, E, e, G, g, warm=None):
    E, G = sparse.csc_matrix(E), sparse.csc_matrix(G)
    e, g = np.asarray(e), np.asarray(g)
    answer = solve_matrices({"P": H, "q": h, "A": sparse.vstack((E, G), format="csc"),
                             "l": np.r_[e, np.full(len(g), -np.inf)], "u": np.r_[e, g]}, warm)
    answer["nu"] = answer["y"][:len(e)] if answer["y"] is not None else None
    answer["lambda"] = answer["y"][len(e):] if answer["y"] is not None else None
    return answer


def _self_check():
    H = np.array([[2.0, 0.2, 0], [0.2, 2.0, 0], [0, 0, 2.0]])
    h = np.array([-5.0, 4.0, 0])
    E, e = np.array([[0.0, 0, 1.0]]), np.array([0.5])
    G, g = np.array([[1.0, 0, 0], [0, -1.0, 0]]), np.array([1.0, 0.0])
    result = solve_qp(H, h, E, e, G, g)
    assert result["valid"], result
    assert np.max(np.abs(result["x"] - [1.0, 0.0, 0.5])) < 1e-7
    assert np.max(np.abs(result["lambda"] - [3.0, 4.2])) < 1e-7
    assert abs(result["nu"][0] + 1.0) < 1e-7
    broken = residuals({"P": H, "q": h, "A": sparse.vstack((E, G)),
                        "l": np.r_[e, [-np.inf] * len(g)], "u": np.r_[e, g]}, result["x"] + 0.1, result["y"])
    assert broken["eq_violation"] > 0.09 and broken["ineq_violation"] > 0.09
    print("strict_qp self-check: PASS")


if __name__ == "__main__":
    _self_check()
