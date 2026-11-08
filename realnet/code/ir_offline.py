"""Offline IR-MPC design from the history days (cc_design.HISTORY_DAYS) (main.tex Secs. III-V).

coupling: for every history sample r (day, update clock) and every pair (m, j), W_mj = max over sigma = +-1 of the
          largest first-decision change at m when j's road counts, entry backlogs and release rates move by sigma
          times their history standard deviation at that clock (negative values clipped to 0), by re-solving the
          centralized QP (Eq. (coupling)). W-bar is the mean over the samples of each demand period. Information
          region: distance order, k_m = min{k : sum of W-bar over unread intersections <= eps_W = 1 s} (Eq. (read-size)).
          Writes design/ir/W_SAMPLES.json and design/ir/REGIONS.json.
bank:     for every update clock t with at least BANK_MIN_DAYS history days, p-bar(t) = mean parameter of those days, x-ref = centralized
          optimum at p-bar(t), and for every intersection the KKT boundary function (construction.construct, boundary
          range from the history solutions at clock t and x-ref, widened by 10% of the span on each side).
          Writes design/ir/bank/t{t}.pkl.gz.
Usage: ir_offline.py coupling | bank
"""

import cc_design as d

d.enforce_threads()

from collections import defaultdict  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import ir_common as ir  # noqa: E402  (imports frozen, which adds the v3 code folder to the path)
import construction  # noqa: E402

ir.strict_qp.TOLERANCE = d.OPEN_LOOP_TOLERANCE      # offline solves are open loop (main.tex Sec. VI)

OUT = d.ROOT / "design/ir"
WORKERS = 14
RANGE_WINDOW = 0      # boundary range from the history solutions at the same clock (main.tex Sec. VI)
BANK_MIN_DAYS = 3     # a bank clock needs at least 3 history days that still run at that clock
PERTURBED = ("n0", "backlog", "release_rate")
_ST, _H = None, None


def _init():
    global _ST, _H
    _ST = ir.static()
    _H = {s: ir.history(s) for s in ir.HISTORY}


def clocks():
    h = {s: ir.history(s) for s in ir.HISTORY}
    return sorted(set.intersection(*(set(v) for v in h.values())))


def coupling_sample(job):
    seed, t = job
    p, x = _H[seed][t]
    model = ir.model_at(_ST, t, ir.snapshot_of(seed, t))
    sigma = np.std([_H[s][t][0] for s in ir.HISTORY], axis=0, ddof=1)
    kinds = np.asarray([lab[0] for lab in model["param_labels"]])
    owners = np.asarray(model["param_owner"])
    base = ir.first_decision(model, x)
    W, failed = defaultdict(dict), []
    for j in sorted(_ST["signals"]):
        cols = np.flatnonzero((owners == j) & np.isin(kinds, PERTURBED))
        change = defaultdict(float)
        for sign in (-1, 1):
            q = p.copy()
            q[cols] = np.maximum(0.0, p[cols] + sign * sigma[cols])
            r = ir.solve_full(model, q)
            if not r["valid"]:
                failed.append([j, sign])
                continue
            for m, a in ir.first_decision(model, r["x"]).items():
                change[m] = max(change[m], float(np.max(np.abs(a - base[m]))))
        for m in base:                     # only intersections whose cycle starts at t have a first decision
            if m != j:
                W[m][j] = change[m]
    return {"seed": seed, "t": t, "period": d.period(t), "W": W, "failed": failed}


def coupling():
    # sigma needs every history day at t; a sample needs the centralized solution of its day at t
    H = {s: ir.history(s) for s in ir.HISTORY}
    jobs = [(s, t) for s in ir.HISTORY for t in clocks() if np.all(np.isfinite(H[s][t][1]))]
    started = time.perf_counter()
    with Pool(WORKERS, initializer=_init) as pool:
        samples = pool.map(coupling_sample, jobs, chunksize=2)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "W_SAMPLES.json").write_text(json.dumps(samples), encoding="utf-8")
    st = ir.static()
    adj = ir.intersection_graph(st)
    out = {"rule": __doc__, "eps_W": ir.EPS_W, "history": ir.HISTORY, "periods": {}}
    for period in d.PERIODS:
        rows = [s["W"] for s in samples if s["period"] == period]
        # W-bar of m: mean over the samples of the period at which m decides (120-s intersections: every 120 s)
        Wbar = {m: {j: float(np.mean([r[m][j] for r in rows if m in r])) for j in st["signals"] if j != m}
                for m in st["signals"]}
        regions, sizes = {}, {}
        for m in sorted(st["signals"]):
            order = ir.distance_order(adj, m)
            k = next(k for k in range(1, len(order) + 1) if sum(Wbar[m][j] for j in order[k:]) <= ir.EPS_W)
            regions[m], sizes[m] = order[:k], k
        out["periods"][period] = {"samples": len(rows), "regions": regions, "sizes": sizes, "W_bar": Wbar}
        print(period, "samples", len(rows), "region sizes", sorted(sizes.values()), flush=True)
    (OUT / "REGIONS.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("coupling seconds", round(time.perf_counter() - started), "failed solves",
          sum(len(s["failed"]) for s in samples))


def bank_clocks():
    """Clocks where at least BANK_MIN_DAYS history days still have an update (after the earliest history day clears,
    p-bar(t) averages the days that still run), so the bank also covers the clearance of the test days."""
    h = {s: ir.history(s) for s in ir.HISTORY}
    count = defaultdict(int)
    for s in h:
        for t in h[s]:
            count[t] += 1
    return sorted(t for t, n in count.items() if n >= BANK_MIN_DAYS)


def bank_clock(t):
    days = [s for s in ir.HISTORY if t in _H[s]]                       # history days that still run at clock t
    ps = np.asarray([_H[s][t][0] for s in days])
    # Boundary range: history solutions at clock t (RANGE_WINDOW = 0), widened by 10% in construction.construct.
    xs = np.asarray([_H[s][u][1] for s in days for u in _H[s] if abs(u - t) <= RANGE_WINDOW
                     and np.all(np.isfinite(_H[s][u][1]))])            # skip a failed centralized QP
    pbar = ps.mean(axis=0)
    model = ir.model_at(_ST, t, ir.snapshot_of(days[0], t))
    ref = ir.solve_full(model, pbar)
    if not ref["valid"]:
        return {"t": t, "status": "REFERENCE_FAILED"}
    entry, status = {}, {}
    for m in sorted(model["output_indices"]):     # intersections that decide at clock t
        c = construction.construct(model, m, pbar, ref["x"], ps, np.vstack([xs, ref["x"][None]]))
        status[m] = c["status"]
        if c["status"] == "CONSTRUCTED":
            entry[m] = {k: c[k] for k in ("status", "target", "reduction", "pbar", "s0", "Q", "M", "vbar", "S_lo",
                                          "S_hi", "param_owner", "input_indices", "local_H", "local_h0", "local_hP")}
    with gzip.open(OUT / f"bank/t{t}.pkl.gz", "wb") as handle:
        pickle.dump({"t": t, "controllers": entry, "pbar": pbar, "xref": ref["x"], "history_days": days}, handle)
    return {"t": t, "status": "OK", "history_days": len(days), "failed_targets": [m for m, s in status.items() if s != "CONSTRUCTED"]}


def bank():
    (OUT / "bank").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with Pool(WORKERS, initializer=_init) as pool:
        rows = pool.map(bank_clock, bank_clocks(), chunksize=1)
    (OUT / "BANK_INDEX.json").write_text(json.dumps({"rule": __doc__, "clocks": rows}, indent=1), encoding="utf-8")
    print("clocks", len(rows), "reference failures", sum(r["status"] != "OK" for r in rows),
          "failed constructions", sum(len(r.get("failed_targets", [])) for r in rows),
          "seconds", round(time.perf_counter() - started))


if __name__ == "__main__":
    {"coupling": coupling, "bank": bank}[sys.argv[1]]()
