"""Online OSD for the closed-loop runs (Table I), the same algorithms as Table III (osd.py, ns_osd.py).

At update t: start from the centralized primal-dual solution at the historical reference p-bar(t) of design/ir/bank
(same start as Table III; this reference solve is not counted as controller time), run K Jacobi iterations with the
current data p, and execute the first step of iterate K. A failed region keeps its previous iterate. A clock without a
bank entry, or a failed reference solve, applies the previous greens of every intersection (as IR-MPC).
Variants: OSD-1 / OSD-2 (one region per intersection, 1- or 2-hop overlap), P-OSD (partition with at most 7
intersections per region, one-hop overlap). Closed-loop acceptance: strict_qp.TOLERANCE as set by frozen.py (1e-5),
Clarabel only (no OSQP fall-back, frozen.py).
"""

import gzip
import json
import pickle
import time

import numpy as np

import cc_design as d
import ir_common as ir
import ns_osd
import osd
from frozen import strict_qp

PART_SIZE = 7


class OSDController:
    def __init__(self, variant, iterations, root=d.ROOT / "design/ir"):
        if variant not in ("OSD-1", "OSD-2", "P-OSD") or iterations < 1:
            raise ValueError("Use OSD-1, OSD-2 or P-OSD with K >= 1.")
        self.variant, self.K, self.root = variant, int(iterations), root
        self.name = f"{variant}, K={self.K}"
        self.adj = ir.intersection_graph(ir.static())
        self.parts = None
        if variant == "P-OSD":
            wbar = json.loads((root / "REGIONS.json").read_text(encoding="utf-8"))["periods"]["high"]["W_bar"]
            self.parts = ns_osd.partition(self.adj, wbar, PART_SIZE)
        self.stats = {"updates": 0, "failed_regions": 0, "no_bank_updates": 0, "reference_failures": 0,
                      "parallel_seconds": []}

    def pbar(self, t):
        path = self.root / f"bank/t{t}.pkl.gz"
        if not path.exists():
            return None
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)["pbar"]

    def decide(self, now, p, previous, model=None):
        p = np.asarray(p, float)
        pbar = self.pbar(now)
        ref = ir.solve_full(model, pbar) if pbar is not None else None
        self.stats["updates"] += 1
        if ref is None or not ref["valid"]:
            reason = "no_bank" if pbar is None else "reference_failed"
            self.stats["no_bank_updates" if pbar is None else "reference_failures"] += 1
            decide = sorted(model["output_indices"])
            return {"action": {m: list(previous[m]) for m in decide}, "failed": [[m, reason] for m in decide],
                    "period": d.period(now), "parallel_seconds": None}
        if self.variant == "P-OSD":
            regions = ns_osd.compile_regions(model, self.parts, self.adj)
            hist = ns_osd.run(model, regions, p, pbar, ref["x"], ref["nu"], self.K)
            seconds = sum(h["max_seconds"] for h in hist)
        else:
            hist, _ = osd.run(model, p, pbar, ref["x"], ref["nu"], self.adj, int(self.variant[-1]), solves=self.K)
            seconds = sum(max(h["seconds"].values()) for h in hist)
        last = hist[-1]
        failed = sorted(set(last["failed"]))
        self.stats["failed_regions"] += len(failed)
        self.stats["parallel_seconds"].append(seconds)
        return {"action": {m: [float(v) for v in last["actions"][m]] for m in model["output_indices"]},
                "failed": [[m, "region_not_accepted"] for m in failed], "period": d.period(now),
                "parallel_seconds": seconds}

    def summary(self):
        s = self.stats
        return {"variant": self.variant, "iterations": self.K, "updates": s["updates"],
                "failed_intersections_last_iteration": s["failed_regions"], "no_bank_updates": s["no_bank_updates"],
                "reference_failures": s["reference_failures"],
                "parallel_seconds_median": float(np.median(s["parallel_seconds"])) if s["parallel_seconds"] else None,
                "partition": self.parts}
