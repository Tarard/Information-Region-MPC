"""Evaluate fixed-boundary KKT and neighborhood MPC with masked current data."""

from __future__ import annotations

import numpy as np
from scipy import sparse

from strict_qp import solve_qp


def observed_parameter(p, pbar, owners, target, nodes):
    """Replace only the parameter blocks supplied by the declared node set."""
    p, pbar = np.asarray(p, float), np.asarray(pbar, float)
    nodes = tuple(nodes)
    if (p.shape != pbar.shape or np.shape(owners) != pbar.shape or not np.isfinite(p).all()
            or len(nodes) != len(set(nodes)) or target not in nodes
            or not set(nodes) <= set(owners.tolist())):
        raise ValueError("Supply finite parameters and unique known nodes including the target.")
    observed = np.isin(owners, nodes)
    used = pbar.copy()
    used[observed] = p[observed]
    return used


def compile_local(controller):
    if "P" in controller:
        return
    red = controller["reduction"]
    n, ns = len(red["core"]), len(controller["s0"])
    controller["P"] = sparse.block_diag((controller["local_H"], controller["Q"]), format="csc")
    controller["Aeq"] = sparse.hstack((red["Em"], red["Fm"]), format="csc")
    controller["Ain"] = sparse.vstack((
        sparse.hstack((red["Gm"], sparse.csc_matrix((red["Gm"].shape[0], ns)))),
        sparse.hstack((sparse.csc_matrix((ns, n)), sparse.eye(ns))),
        sparse.hstack((sparse.csc_matrix((ns, n)), -sparse.eye(ns)))), format="csc")
    controller["first"] = np.searchsorted(red["core"], controller["input_indices"])
    assert np.array_equal(red["core"][controller["first"]], controller["input_indices"])


def local_data(controller, p, C):
    used = observed_parameter(p, controller["pbar"], controller["param_owner"],
                              controller["target"], C)
    red = controller["reduction"]
    v = controller["vbar"] + controller["M"] @ (used - controller["pbar"])
    h = np.r_[controller["local_h0"] + controller["local_hP"] @ used,
              v - controller["Q"] @ controller["s0"]]
    e = np.asarray(red["em0"] + red["emP"] @ used).ravel()
    g = np.r_[red["gm0"] + red["gmP"] @ used, controller["S_hi"], -controller["S_lo"]]
    return h, e, g, v


def solve_local(controller, p, C):
    if controller.get("status") != "CONSTRUCTED":
        return {"valid": False, "status": "CONSTRUCTION_FAILED", "action": None}
    compile_local(controller)
    h, e, g, v = local_data(controller, p, C)
    result = solve_qp(controller["P"], h, controller["Aeq"], e, controller["Ain"], g)
    result.update(information_nodes=list(C), linear_coefficient=v,
                  uniform_error_certified=False, action=None)
    if result["valid"]:
        n = len(controller["reduction"]["core"])
        result.update(local_x=result["x"][:n], s=result["x"][n:],
                      action=result["x"][controller["first"]])
    return result


def compile_neighbor(model, target, C, pbar, xref):
    """Retain owned rows and variables, with frozen exterior predictions."""
    nodes = tuple(sorted(C))
    pbar, xref = np.asarray(pbar, float), np.asarray(xref, float)
    if pbar.shape != np.asarray(model["p"]).shape:
        raise ValueError("The historical parameters must match the full QP.")
    observed_parameter(pbar, pbar, model["param_owner"], target, nodes)
    if xref.shape != (model["H"].shape[0],) or not np.isfinite(xref).all():
        raise ValueError("The historical solution must match the full QP.")
    own = np.flatnonzero(np.isin(model["variable_owner"], nodes))
    outside = np.flatnonzero(~np.isin(model["variable_owner"], nodes))
    eq = np.flatnonzero(np.isin(model["eq_owner"], nodes))
    ine = np.flatnonzero(np.isin(model["ineq_owner"], nodes))
    assert model["G"][ine][:, outside].nnz == 0
    first = np.searchsorted(own, model["output_indices"][target])
    assert np.array_equal(own[first], model["output_indices"][target])
    return {"target": target, "nodes": nodes, "own": own, "outside": outside,
            "P": model["H"][own][:, own].tocsc(),
            "Aeq": model["E"][eq][:, own].tocsc(),
            "Ain": model["G"][ine][:, own].tocsc(),
            "h0": np.asarray(model["h0"][own] + model["H"][own][:, outside] @ xref[outside]).ravel(),
            "hP": model["hP"][own].tocsc(),
            "e0": np.asarray(model["e0"][eq] - model["E"][eq][:, outside] @ xref[outside]).ravel(),
            "eP": model["eP"][eq].tocsc(), "g0": model["g0"][ine].copy(),
            "gP": model["gP"][ine].tocsc(), "pbar": pbar.copy(),
            "param_owner": model["param_owner"].copy(), "first": first}


def neighbor_data(region, p):
    used = observed_parameter(p, region["pbar"], region["param_owner"],
                              region["target"], region["nodes"])
    return tuple(np.asarray(region[key + "0"] + region[key + "P"] @ used).ravel()
                 for key in ("h", "e", "g"))


def solve_neighbor(region, p):
    h, e, g = neighbor_data(region, p)
    result = solve_qp(region["P"], h, region["Aeq"], e, region["Ain"], g)
    result.update(information_nodes=list(region["nodes"]), action=None)
    if result["valid"]:
        result["action"] = result["x"][region["first"]]
    return result


def self_check():
    """Check exact full information and invariance to unread parameters."""
    from construction import construct
    from v12_boundary_v2 import solve_affine

    owners = np.array(["a", "a", "b", "b"])
    model = {"H": sparse.eye(4, format="csc"), "h0": np.zeros(4),
             "hP": sparse.csc_matrix([[0., 0.], [-1., 0.], [0., -1.], [0., 0.]]),
             "E": sparse.csc_matrix([[1., 0., -1., 0.]]), "e0": np.zeros(1),
             "eP": sparse.csc_matrix((1, 2)),
             "G": sparse.vstack((sparse.eye(4), -sparse.eye(4)), format="csc"),
             "g0": np.r_[np.full(4, 10.), np.zeros(4)], "gP": sparse.csc_matrix((8, 2)),
             "p": np.array([2., 2.]), "param_owner": np.array(["a", "b"]),
             "variable_owner": owners, "eq_owner": np.array(["a"]),
             "ineq_owner": np.r_[owners, owners], "output_indices": {"a": [0, 1], "b": [2, 3]}}
    nominal = solve_affine(model, model["p"])
    assert nominal["valid"]
    ps = np.array([[2., 2.], [2.5, 3.]])
    xs = np.asarray([solve_affine(model, p)["x"] for p in ps])
    entry = construct(model, "a", model["p"], nominal["x"], ps, xs,
                      boundary_bounds=(np.array([-10.]), np.array([0.])))
    assert entry["status"] == "CONSTRUCTED", entry
    p = ps[1]
    direct = solve_affine(model, p)
    local = solve_local(entry, p, ["a", "b"])
    assert local["valid"] and np.max(np.abs(local["action"] - direct["x"][:2])) < 1e-7
    changed = p.copy()
    changed[1] += 999.
    d1, d2 = local_data(entry, p, ["a"]), local_data(entry, changed, ["a"])
    assert all(np.array_equal(a, b) for a, b in zip(d1, d2))
    full = compile_neighbor(model, "a", ["a", "b"], model["p"], nominal["x"])
    assert (full["P"] != model["H"]).nnz == 0
    assert (full["Aeq"] != model["E"]).nnz == 0
    assert (full["Ain"] != model["G"]).nnz == 0
    neighbor = solve_neighbor(full, p)
    assert neighbor["valid"] and np.max(np.abs(neighbor["x"] - direct["x"])) < 1e-7
    partial = compile_neighbor(model, "a", ["a"], model["p"], nominal["x"])
    assert all(np.array_equal(a, b) for a, b in zip(neighbor_data(partial, p),
                                                  neighbor_data(partial, changed)))
    return {"passed": True, "full_information_first_input_error": float(
        np.max(np.abs(local["action"] - direct["x"][:2]))),
            "full_neighbor_solution_error": float(np.max(np.abs(neighbor["x"] - direct["x"]))),
            "unread_parameter_invariance": True}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2))
