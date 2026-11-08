"""Partitioned overlapping Schwarz decomposition (OSD; Shin et al. 2020) on the same QP (TASK.md section 5).

The intersections are split into P regions R_1..R_P (partition below). Region R_i solves over R_i plus every
intersection within `overlap` hops (its overlap ring). Each solve fixes the exterior primal values x^l and the
multipliers nu^l of exterior-owned rows that touch the region:
    min 1/2 x'H x + (h + E_out' nu^l)'x   s.t.  E_in x = e_in - E_in,ext x^l_ext,  G x <= g,
and the region keeps only the variables and the row multipliers of its own intersections R_i. All regions update
together (Jacobi). A region whose QP is not accepted keeps its previous iterate. The decision after k iterations is
the first step of iterate k. Start: the centralized primal-dual solution at the historical reference of the period.
Region data: current data of the region and its ring; the reference p-bar elsewhere.
With one intersection per region this is the per-intersection OSD of the LYG tables.

Partition: recursive Kernighan-Lin bisection of the intersection graph, edge weight = W-bar_mj + W-bar_jm + 1e-3
(historical mean coupling of the peak period, so strongly coupled intersections stay together), until every
region has at most `size` intersections. Each bisection keeps the smallest cut over 20 fixed starts; the graph is
built in sorted node order, so the partition is the same in every process.
"""

import time

import networkx as nx
import numpy as np

from local_controller import observed_parameter
from strict_qp import solve_qp


def _graph(adjacency, coupling, nodes):
    """Graph on `nodes` built in sorted order, so the bisection does not depend on the string hash order."""
    g, keep = nx.Graph(), set(nodes)
    g.add_nodes_from(sorted(keep))
    for a in sorted(keep):
        for b in sorted(adjacency[a]):
            if b in keep:
                w = coupling.get(a, {}).get(b, 0.0) + coupling.get(b, {}).get(a, 0.0) + 1e-3
                g.add_edge(a, b, weight=w)
    return g


def partition(adjacency, coupling, size, starts=20):
    """Each bisection keeps the smallest cut weight over `starts` fixed Kernighan-Lin starts (deterministic)."""
    parts, todo = [], [sorted(adjacency)]
    while todo:
        part = todo.pop()
        if len(part) <= size:
            parts.append(sorted(part))
            continue
        g = _graph(adjacency, coupling, part)
        best = None
        for seed in range(starts):
            a, b = nx.algorithms.community.kernighan_lin_bisection(g, weight="weight", seed=seed)
            cut = nx.cut_size(g, a, b, weight="weight")
            if best is None or cut < best[0] - 1e-12:
                best = (cut, sorted(a), sorted(b))
        todo += [best[1], best[2]]
    return sorted(parts)


def ring(adjacency, members, hops):
    nodes, frontier = set(members), set(members)
    for _ in range(hops):
        frontier = {v for u in frontier for v in adjacency[u]} - nodes
        nodes |= frontier
    return sorted(nodes - set(members))


class Region:
    def __init__(self, model, members, overlap_nodes):
        self.members = list(members)
        self.nodes = self.members + list(overlap_nodes)
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
        mine = np.flatnonzero(np.isin(vo, self.members))
        self.mine, self.mine_local = mine, np.searchsorted(self.own, mine)
        rows = np.flatnonzero(np.isin(eo, self.members))
        self.rows, self.rows_local = rows, np.searchsorted(self.eq_in, rows)
        self.owners = np.asarray(model["param_owner"])

    def solve(self, model, p, pbar, x, nu):
        used = observed_parameter(p, pbar, self.owners, self.members[0], self.nodes)
        h = np.asarray(model["h0"] + model["hP"] @ used).ravel()[self.own] + self.Eout.T @ nu[self.eq_out]
        e = np.asarray(model["e0"] + model["eP"] @ used).ravel()[self.eq_in] - self.Cext @ x[self.ext]
        g = np.asarray(model["g0"] + model["gP"] @ used).ravel()[self.ine]
        return solve_qp(self.P, h, self.Aeq, e, self.Ain, g)


def compile_regions(model, parts, adjacency, overlap=1):
    return [Region(model, part, ring(adjacency, part, overlap)) for part in parts]


def run(model, regions, p, pbar, xref, nu_ref, iterations, converge=None, min_iterations=4):
    """Jacobi OSD. Returns the actions after every iteration, failures, per-iteration slowest region time.
    With `converge`, stop once the first decisions change by less than it (never before min_iterations)."""
    x, nu = xref.copy(), nu_ref.copy()
    history, previous = [], None
    for k in range(1, iterations + 1):
        nx_, nn = x.copy(), nu.copy()
        step = {"iteration": k, "actions": {}, "failed": [], "max_seconds": 0.0}
        for reg in regions:
            tick = time.perf_counter()
            r = reg.solve(model, p, pbar, x, nu)
            step["max_seconds"] = max(step["max_seconds"], time.perf_counter() - tick)
            if r["valid"]:
                nx_[reg.mine] = r["x"][reg.mine_local]
                nn[reg.rows] = r["nu"][reg.rows_local]
            else:                      # a failed region keeps its previous iterate (standard Schwarz update)
                step["failed"] += [m for m in reg.members if m in model["output_indices"]]   # deciding ones
        x, nu = nx_, nn
        for m, idx in model["output_indices"].items():   # decision after k iterations = first step of iterate k
            step["actions"][m] = x[idx]
        history.append(step)
        if converge is not None and k >= min_iterations and previous is not None and not step["failed"]:
            change = max(float(np.max(np.abs(step["actions"][m] - previous[m]))) for m in step["actions"])
            if change < converge:
                break
        previous = step["actions"] if not step["failed"] else None
    return history
