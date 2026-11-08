"""Overlapping Schwarz decomposition (OSD) on the same QP, as in LYG_neighbor_exchange_v1/code/schwarz.py.

Region of j: j plus the intersections within h hops (h = 1 or 2) of the road graph. Each solve fixes the exterior
primal values x^l and the multipliers nu^l of exterior-owned rows that touch the region:
    min 1/2 x'H x + (h + E_out' nu^l)'x   s.t.  E_in x = e_in - E_in,ext x^l_ext,  G x <= g.
j keeps its own variables and the multipliers of its own rows; all intersections update simultaneously. A region
whose QP is not accepted keeps its previous iterate; the decision after k solves is the first step of iterate k. The start
point is the centralized primal-dual solution at the historical reference p-bar(t). Region data: current data of the
region's intersections, p-bar for the rest.
"""

import time

import numpy as np

from local_controller import observed_parameter
from strict_qp import solve_qp


def hop_region(adj, j, hops):
    nodes, frontier = {j}, {j}
    for _ in range(hops):
        frontier = {v for u in frontier for v in adj[u]} - nodes
        nodes |= frontier
    return [j] + sorted(nodes - {j})


class Region:
    def __init__(self, model, j, nodes):
        self.j, self.nodes = j, nodes
        vo, eo, io = (np.asarray(model[k]) for k in ("variable_owner", "eq_owner", "ineq_owner"))
        E, G = model["E"].tocsr(), model["G"].tocsr()
        own = np.isin(vo, self.nodes)
        self.own, self.ext = np.flatnonzero(own), np.flatnonzero(~own)
        eq_in = np.isin(eo, self.nodes)
        eq_out = ~eq_in & (np.asarray(E[:, self.own].getnnz(axis=1)).ravel() > 0)
        self.eq_in, self.eq_out = np.flatnonzero(eq_in), np.flatnonzero(eq_out)
        self.ine = np.flatnonzero(np.isin(io, self.nodes))
        assert G[self.ine][:, self.ext].nnz == 0
        self.P = model["H"][self.own][:, self.own].tocsc()
        self.Aeq = E[self.eq_in][:, self.own].tocsc()
        self.Cext = E[self.eq_in][:, self.ext].tocsc()
        self.Eout = E[self.eq_out][:, self.own].tocsc()
        self.Ain = G[self.ine][:, self.own].tocsc()
        mine = np.flatnonzero(vo == j)
        self.mine, self.mine_local = mine, np.searchsorted(self.own, mine)
        rows_j = np.flatnonzero(eo == j)
        self.rows_j, self.rows_j_local = rows_j, np.searchsorted(self.eq_in, rows_j)

    def solve(self, model, p, pbar, x, nu):
        used = observed_parameter(p, pbar, np.asarray(model["param_owner"]), self.j, self.nodes)
        h = np.asarray(model["h0"] + model["hP"] @ used).ravel()[self.own] + self.Eout.T @ nu[self.eq_out]
        e = np.asarray(model["e0"] + model["eP"] @ used).ravel()[self.eq_in] - self.Cext @ x[self.ext]
        g = np.asarray(model["g0"] + model["gP"] @ used).ravel()[self.ine]
        return solve_qp(self.P, h, self.Aeq, e, self.Ain, g)


def run(model, p, pbar, xref, nu_ref, adj, hops, solves=4):
    # every intersection keeps a region (its variables iterate); only intersections whose cycle starts now decide
    regions = {j: Region(model, j, hop_region(adj, j, hops)) for j in sorted(model["metadata"]["cycle_seconds"])}
    x, nu, history = xref.copy(), nu_ref.copy(), []
    for k in range(1, solves + 1):
        nx, nn = x.copy(), nu.copy()
        step = {"solve": k, "actions": {}, "failed": [], "seconds": {}}
        for j, reg in regions.items():
            tick = time.perf_counter()
            r = reg.solve(model, p, pbar, x, nu)
            step["seconds"][j] = time.perf_counter() - tick
            if r["valid"]:
                nx[reg.mine] = r["x"][reg.mine_local]
                nn[reg.rows_j] = r["nu"][reg.rows_j_local]
            else:                      # a failed region keeps its previous iterate (standard Schwarz update)
                if j in model["output_indices"]:      # failures count for deciding intersections
                    step["failed"].append(j)
        x, nu = nx, nn
        for j, idx in model["output_indices"].items():   # decision after k solves = first step of iterate k
            step["actions"][j] = x[idx]
        history.append(step)
    return history, {j: len(r.nodes) for j, r in regions.items()}
