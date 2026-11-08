"""IR-MPC of the numerical experiment (main.tex Secs. III-V; TASK.md section 4).

Offline, per reference (ns_design.REFERENCE_MODE: every update clock t as in the paper, or one per demand period):
p-bar = mean history parameter, x-ref = centralized optimum at p-bar, and for every intersection m the KKT boundary
function (construction.construct; boundary range from the history solutions of the reference, widened by 10% on
each side). Only the columns of M that belong to m's information region C are kept,
because the online slope is v_C = v-bar + sum over j in C of M_j (p_j - p-bar_j) (Eq. (slope)).
Online, at update t: each intersection reads the current data of C, forms v_C and solves its local QP (Eq. (local)).
A failed local QP applies the previous greens of m and counts as a failure.
"""

import time

import numpy as np
from scipy import sparse

from strict_qp import solve_qp

KEEP_RED = ("core", "Em", "Fm", "em0", "emP", "Gm", "gm0", "gmP")


def compact(c, cols):
    """Keep what the online local QP needs; M only on the parameter columns of the information region."""
    red = {k: c["reduction"][k] for k in KEEP_RED}
    return {"status": c["status"], "target": c["target"], "reduction": red, "s0": c["s0"], "Q": c["Q"],
            "M": np.asarray(c["M"])[:, cols], "cols": np.asarray(cols), "vbar": c["vbar"], "S_lo": c["S_lo"],
            "S_hi": c["S_hi"], "input_indices": c["input_indices"], "local_H": c["local_H"],
            "local_h0": c["local_h0"], "local_hP": c["local_hP"], "boundary_dimension": len(c["s0"])}


def compile_local(c):
    if "P" in c:
        return
    red, n, ns = c["reduction"], len(c["reduction"]["core"]), len(c["s0"])
    c["P"] = sparse.block_diag((c["local_H"], c["Q"]), format="csc")
    c["Aeq"] = sparse.hstack((red["Em"], red["Fm"]), format="csc")
    c["Ain"] = sparse.vstack((sparse.hstack((red["Gm"], sparse.csc_matrix((red["Gm"].shape[0], ns)))),
                              sparse.hstack((sparse.csc_matrix((ns, n)), sparse.eye(ns))),
                              sparse.hstack((sparse.csc_matrix((ns, n)), -sparse.eye(ns)))), format="csc")
    c["first"] = np.searchsorted(red["core"], c["input_indices"])


def solve_local(c, p, pbar):
    """Local QP of Eq. (local) with the data of the information region (columns c['cols']) and p-bar elsewhere."""
    compile_local(c)
    used = pbar.copy()
    used[c["cols"]] = p[c["cols"]]
    red = c["reduction"]
    v = c["vbar"] + c["M"] @ (p[c["cols"]] - pbar[c["cols"]])
    h = np.r_[c["local_h0"] + c["local_hP"] @ used, v - c["Q"] @ c["s0"]]
    e = np.asarray(red["em0"] + red["emP"] @ used).ravel()
    g = np.r_[red["gm0"] + red["gmP"] @ used, c["S_hi"], -c["S_lo"]]
    r = solve_qp(c["P"], h, c["Aeq"], e, c["Ain"], g)
    r["action"] = r["x"][c["first"]] if r["valid"] else None
    if r["valid"]:
        n = len(red["core"])
        r["local_x"] = r["x"][:n]
    return r


def load_bank(folder, key):
    import gzip
    import pickle
    with gzip.open(folder / f"bank_{key}.pkl.gz", "rb") as h:
        return pickle.load(h)


class IRController:
    def __init__(self, folder):
        """Banks are read from folder/bank_{key}.pkl.gz on first use; one bank is kept in memory."""
        self.folder, self.key, self.bank = folder, None, None

    def decide(self, key, p, previous):
        if key != self.key:
            self.key, self.bank = key, load_bank(self.folder, key)
        bank = self.bank
        action, failed, seconds = {}, [], {}
        for m in sorted(previous):
            c = bank["controllers"].get(m)
            if c is None:
                action[m] = list(previous[m])
                failed.append([m, "construction_failed"])
                continue
            tick = time.perf_counter()
            r = solve_local(c, np.asarray(p, float), bank["pbar"])
            seconds[m] = time.perf_counter() - tick
            if r["valid"]:
                action[m] = [float(v) for v in r["action"]]
            else:
                action[m] = list(previous[m])
                failed.append([m, str(r.get("status"))])
        return {"action": action, "failed": failed, "max_seconds": max(seconds.values()) if seconds else None,
                "sum_seconds": sum(seconds.values())}
