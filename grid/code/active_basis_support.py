"""Extend a traffic-QP active basis along unpenalized flow directions.

The Hessian must be positive semidefinite, with a positive-definite principal
block on its nonzero columns. Zero columns then span its complete null space.
The original Hessian, objective, equalities, and inequalities are unchanged.
"""

from __future__ import annotations

import time

import numpy as np
from scipy import sparse
from scipy.linalg import null_space, qr
from scipy.optimize import linprog
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


def _maximum(value):
    return float(np.max(np.abs(value), initial=0.0))


def _column_kernel(matrix, *, component_limit=1024, rank_tolerance=1e-10):
    """Find a sparse null basis through separate column-connected blocks."""
    matrix = matrix.tocsc()
    incidence = abs(matrix)
    incidence.data[:] = 1.0
    count, labels = connected_components(incidence.T @ incidence, directed=False)
    rows, cols, values, width = [], [], [], 0
    largest = 0
    for component in range(count):
        selected = np.flatnonzero(labels == component)
        largest = max(largest, len(selected))
        if len(selected) > component_limit:
            raise ValueError("ZERO_CURVATURE_COMPONENT_EXCEEDS_DENSE_LIMIT")
        block = matrix[:, selected]
        used = np.flatnonzero(np.asarray(block.getnnz(axis=1)).ravel())
        kernel = (null_space(block[used].toarray(), rcond=rank_tolerance)
                  if len(used) else np.eye(len(selected)))
        rr, cc = np.nonzero(kernel)
        rows.extend(selected[rr].tolist())
        cols.extend((cc + width).tolist())
        values.extend(kernel[rr, cc].tolist())
        width += kernel.shape[1]
    result = sparse.coo_matrix((values, (rows, cols)),
                              shape=(matrix.shape[1], width)).tocsc()
    return result, largest


def extend_active_basis(H, E, G, h, e, g, x, positive_rows, *,
                        tolerance=1e-7, component_limit=1024):
    """Return positive-dual rows plus independent zero-dual active rows.

    The caller must refine the returned KKT system and verify original primal,
    dual, stationarity, and complementarity residuals. This helper does not
    replace those checks or claim that a floating-point rank test is a proof.
    """
    start = time.perf_counter()
    H, E, G = (sparse.csc_matrix(item).copy() for item in (H, E, G))
    H.eliminate_zeros()
    h, e, g, x = (np.asarray(item, dtype=float).ravel() for item in (h, e, g, x))
    n, ne = H.shape[0], E.shape[0]
    selected = np.asarray(positive_rows, dtype=int).ravel()
    if (H.shape != (n, n) or E.shape[1] != n or G.shape[1] != n or
            h.shape != (n,) or e.shape != (ne,) or g.shape != (G.shape[0],) or
            x.shape != (n,) or len(set(selected.tolist())) != len(selected) or
            np.any(selected < 0) or np.any(selected >= G.shape[0]) or
            not all(np.all(np.isfinite(item)) for item in (h, e, g, x))):
        raise ValueError("Invalid QP dimensions, data, or positive-row indices.")
    diagnostics = {"positive_rows": int(len(selected))}

    def failed(reason, **extra):
        return {"status": "FAIL", "reason": reason, **diagnostics, **extra,
                "seconds": time.perf_counter() - start}

    zero = np.flatnonzero(np.asarray(H.getnnz(axis=0)).ravel() == 0)
    curved = np.flatnonzero(np.asarray(H.getnnz(axis=0)).ravel() != 0)
    diagnostics["zero_hessian_columns"] = int(len(zero))
    if _maximum(h[zero]) > 1e-14:
        return failed("UNCURVED_LINEAR_COST_REQUIRES_A_GENERAL_OPTIMAL_FACE_METHOD")
    original_residual = max(_maximum(E @ x - e),
                            float(np.max(G @ x - g, initial=0.0)),
                            _maximum(G[selected] @ x - g[selected]))
    if original_residual > tolerance:
        return failed("REFERENCE_OR_POSITIVE_ROW_RESIDUAL", residual=original_residual)
    W = sparse.vstack((E, G[selected]), format="csc")
    try:
        if len(curved):
            splu(H[curved][:, curved])
        if W.shape[0]:
            row_factor = splu((W @ W.T).tocsc())
            pivots = abs(row_factor.U.diagonal())
            if pivots.min(initial=np.inf) <= 1e-11 * pivots.max(initial=1.0):
                return failed("EQUALITY_OR_POSITIVE_ROWS_DEPENDENT")
        N, largest = _column_kernel(W[:, zero], component_limit=component_limit)
    except RuntimeError:
        return failed("CURVED_HESSIAN_OR_EXISTING_ROWS_SINGULAR")
    except ValueError as exc:
        return failed(str(exc))
    diagnostics.update(null_dimension=int(N.shape[1]), largest_component=int(largest),
                       null_residual=_maximum((W[:, zero] @ N).data))
    if diagnostics["null_residual"] > 1e-9:
        return failed("NULL_BASIS_RESIDUAL")

    vertex = x.copy()
    added = np.empty(0, dtype=int)
    if N.shape[1]:
        fixed = np.asarray(W @ x).ravel()
        fixed[:ne] = e
        fixed[ne:] = g[selected]
        vertex_lp = linprog(np.linspace(1.0, 2.0, len(zero)),
            A_ub=G[:, zero], b_ub=g - G[:, curved] @ x[curved],
            A_eq=W[:, zero], b_eq=fixed - W[:, curved] @ x[curved],
            bounds=[(None, None)] * len(zero), method="highs-ds",
            options={"presolve": False, "primal_feasibility_tolerance": tolerance,
                     "dual_feasibility_tolerance": tolerance})
        if not vertex_lp.success:
            return failed("ZERO_CURVATURE_VERTEX_LP_FAILED",
                          lp_status=int(vertex_lp.status), lp_message=vertex_lp.message)
        vertex[zero] = vertex_lp.x
        residual = max(_maximum(E @ vertex - e),
                       float(np.max(G @ vertex - g, initial=0.0)),
                       _maximum(G[selected] @ vertex - g[selected]))
        if residual > tolerance:
            return failed("VERTEX_ORIGINAL_PRIMAL_RESIDUAL", residual=residual)
        active = np.flatnonzero(g - G @ vertex <= tolerance)
        candidates = np.setdiff1d(active, selected)
        response = np.asarray((G[candidates][:, zero] @ N).toarray())
        _, triangular, pivots = qr(response.T, mode="economic", pivoting=True)
        diagonal = abs(np.diag(triangular))
        rank = int(np.sum(diagonal > 1e-10 * diagonal.max(initial=1.0)))
        if rank != N.shape[1]:
            return failed("ACTIVE_ROWS_DO_NOT_REMOVE_ZERO_CURVATURE", response_rank=rank,
                          active_count=int(len(active)))
        added = candidates[pivots[:rank]]
        selected = np.r_[selected, added]

    GB = G[selected]
    K = sparse.bmat([[H, E.T, GB.T], [E, None, None], [GB, None, None]], format="csc")
    try:
        factor = splu(K)
    except RuntimeError:
        return failed("EXTENDED_KKT_SINGULAR", added_rows=added)
    objective_difference = (.5 * vertex @ (H @ vertex) + h @ vertex -
                            .5 * x @ (H @ x) - h @ x)
    return {"status": "PASS", "x": vertex, "active_rows": selected,
            "added_rows": added, "K": K, "factor": factor, **diagnostics,
            "objective_difference": float(objective_difference),
            "vertex_distance": _maximum(vertex - x),
            "seconds": time.perf_counter() - start}
