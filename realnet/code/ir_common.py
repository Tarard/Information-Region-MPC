"""Shared pieces of the IR-MPC trial on the mixed-cycle model (main.tex Secs. III and V).

History days give, per update clock t: the reference p-bar(t) (mean parameter), the centralized solutions (boundary
range S(t)), and the coupling samples. The QP at clock t is the cc_model QP built with that clock's arrival row; its
matrices do not depend on p, so the centralized optimum at any p is solved from the standard form (h, e, g affine in p).
"""

import json
from pathlib import Path

import numpy as np

import cc_design as d
import cc_model
import cc_scenario
from frozen import strict_qp

RUNS = d.ROOT / "results_v1/runs"


def cleared(seed):
    """Day kept only if centralized MPC cleared the network (screening rule agreed with the user)."""
    path = RUNS / f"mix_seed{seed}/SUMMARY.json"
    return path.exists() and json.loads(path.read_text(encoding="utf-8"))["gates"]["all_trips_completed"]


HISTORY = tuple(s for s in d.HISTORY_DAYS if cleared(s))
TEST = tuple(s for s in d.TEST_DAYS if cleared(s))
EPS_W = 1.0


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def static():
    inputs = cc_scenario.load_inputs(HISTORY[0])
    return {k: inputs[k] for k in ("network", "signals", "edge_owner", "arrival_rows", "service_groups")}


def history(seed):
    z = np.load(RUNS / f"mix_seed{seed}/snapshots.npz")
    return {int(t): (p, x) for t, p, x in zip(z["t"], z["p"], z["xcentral"])}


def model_at(st, t, snapshot):
    _, rates = cc_model.arrival_row(st["arrival_rows"], t)
    return cc_model.build(st["network"], st["signals"], snapshot, rates, st["service_groups"], st["edge_owner"])


def snapshot_of(seed, t):
    return load(RUNS / f"mix_seed{seed}/snapshots/t{t}.json")


def solve_full(model, p):
    """Centralized optimum at parameter p from the standard form."""
    return strict_qp.solve_qp(model["H"], model["h0"] + model["hP"] @ p, model["E"], model["e0"] + model["eP"] @ p,
                              model["G"], model["g0"] + model["gP"] @ p)


def first_decision(model, x):
    return {m: np.asarray(x)[idx] for m, idx in model["output_indices"].items()}


def intersection_graph(st):
    """Undirected adjacency of intersections joined by a road connection."""
    owner, adj = st["edge_owner"], {m: set() for m in st["signals"]}
    for c in st["network"]["connections"]:
        a, b = owner.get(c["from"]), owner.get(c["to"])
        if a and b and a != b:
            adj[a].add(b)
            adj[b].add(a)
    return adj


def distance_order(adj, m):
    """m first, then the other intersections by hop distance, ties by index."""
    dist, frontier = {m: 0}, [m]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return sorted(adj, key=lambda j: (dist.get(j, 10 ** 6), j))
