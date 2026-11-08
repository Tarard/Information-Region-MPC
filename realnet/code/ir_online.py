"""Online IR-MPC (main.tex Eq. (local)): at update t each intersection m reads the current data of its information
region C (fixed per demand period, design/ir/REGIONS.json), updates the boundary slope v_C(t) and solves one local QP
with its boundary function from design/ir/bank/t{t}.pkl.gz. A failed local QP, or a clock without a bank entry,
applies the previous greens of m (main.tex Sec. VI)."""

import gzip
import json
import pickle
import time

import numpy as np

import cc_design as d
import frozen  # noqa: F401  adds the v3 path and the sparse-Fm compile_local before local_controller is used
import local_controller


class IRController:
    def __init__(self, root=d.ROOT / "design/ir"):
        self.root = root
        self.regions = json.loads((root / "REGIONS.json").read_text(encoding="utf-8"))["periods"]
        self.stats = {"updates": 0, "local_failures": 0, "no_bank_updates": 0, "max_seconds": [], "sum_seconds": []}

    def bank(self, t):
        path = self.root / f"bank/t{t}.pkl.gz"
        if not path.exists():
            return None
        with gzip.open(path, "rb") as handle:
            return pickle.load(handle)["controllers"]

    def decide(self, now, p, previous, model):         # model: the QP of this update (deciding intersections)
        period = d.period(now)
        regions = self.regions[period]["regions"]
        bank = self.bank(now)
        action, failed, seconds = {}, [], {}
        for m in sorted(model["output_indices"]):      # intersections whose cycle starts now
            ctrl = None if bank is None else bank.get(m)
            if ctrl is None:
                action[m] = list(previous[m])
                failed.append([m, "no_bank" if bank is None else "construction_failed"])
                continue
            tick = time.perf_counter()
            res = local_controller.solve_local(ctrl, np.asarray(p, float), regions[m])
            seconds[m] = time.perf_counter() - tick
            if res["valid"]:
                action[m] = [float(v) for v in res["action"]]
            else:
                action[m] = list(previous[m])
                failed.append([m, str(res.get("status"))])
        self.stats["updates"] += 1
        self.stats["local_failures"] += len(failed)
        self.stats["no_bank_updates"] += bank is None
        if seconds:
            self.stats["max_seconds"].append(max(seconds.values()))
            self.stats["sum_seconds"].append(sum(seconds.values()))
        return {"action": action, "failed": failed, "period": period, "local_seconds": seconds,
                "region_sizes": {m: len(regions[m]) for m in action}}

    def summary(self):
        s = self.stats
        med = lambda v: float(np.median(v)) if v else None
        return {"updates": s["updates"], "local_failures": s["local_failures"], "no_bank_updates": s["no_bank_updates"],
                "local_max_seconds_median": med(s["max_seconds"]),
                "local_max_seconds_max": max(s["max_seconds"]) if s["max_seconds"] else None,
                "local_sum_seconds_median": med(s["sum_seconds"])}
