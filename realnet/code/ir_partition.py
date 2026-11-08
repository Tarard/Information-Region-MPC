"""Partitioned OSD on the 50 Table III samples (full OSD with regions of several intersections; ns_osd.py).

Partition: recursive Kernighan-Lin bisection of the intersection graph, edge weight = peak-period W-bar in both
directions + 0.001 (design/ir/REGIONS.json), until every region has at most SIZE intersections: SIZE = 14, 7, 4
(2, 4 and 8 regions; deterministic, see ns_osd.partition). One-hop overlap. Start: the centralized primal-dual solution at p-bar(t). Iterate
K = 1, 2, 4. Same samples,
error, failure rule and cost gap as ir_tables.py samples; open-loop acceptance 1e-7.
accuracy: 12 processes -> results_ir_v1/TABLES_partition.json (partitions: results_ir_v1/PARTITIONS.json)
timing  : ir_partition_timing_onepass.py (all 50 samples, one process, nothing else running)
Usage: ir_partition.py accuracy
"""

import cc_design as d

d.enforce_threads()

import hashlib  # noqa: E402
import json  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import ir_common as ir  # noqa: E402
import ir_tables as T  # noqa: E402
import ns_osd  # noqa: E402
from frozen import strict_qp  # noqa: E402

strict_qp.TOLERANCE = d.OPEN_LOOP_TOLERANCE

# No OSQP fall-back (frozen.py): a QP not accepted from Clarabel is a failure.
SIZES = (14, 7, 4)
MAX_K, CONVERGED = 4, None     # K = 1, 2, 4 (the table uses 1 and 4); convergence runs were too slow
_P = {}


def _init():
    T._init()
    wbar = json.loads((d.ROOT / "design/ir/REGIONS.json").read_text(encoding="utf-8"))["periods"]["high"]["W_bar"]
    _P["parts"] = {s: ns_osd.partition(T._ADJ, wbar, s) for s in SIZES}


def one(seed, t, timing=False):
    snap = ir.snapshot_of(seed, t)
    model = ir.model_at(T._ST, t, snap)
    z = np.load(ir.RUNS / f"mix_seed{seed}/vectors/t{t}.npz")
    p = z["p"]
    central = ir.solve_full(model, p)
    star = ir.first_decision(model, central["x"])
    prev = {m: snap["signals"][m]["previous_executed_green_seconds"] for m in star}
    b = T.bank(t)
    ref = ir.solve_full(model, b["pbar"])
    acts, secs = {}, {}
    for size in SIZES:
        regions = ns_osd.compile_regions(model, _P["parts"][size], T._ADJ)
        hist = ns_osd.run(model, regions, p, b["pbar"], ref["x"], ref["nu"], MAX_K, converge=CONVERGED)
        for k in (1, 2, 4):
            name = f"P-OSD-{size}, K={k}"
            acts[name] = (hist[k - 1]["actions"], sorted(set(hist[k - 1]["failed"])), k)
            secs[name] = sum(h["max_seconds"] for h in hist[:k])
    if timing:                         # per-intersection OSD timed the same way, for one comparable time column
        for hops in (1, 2):
            hist, _ = T.osd.run(model, p, b["pbar"], ref["x"], ref["nu"], T._ADJ, hops, solves=4)
            for k in (1, 4):
                secs[f"OSD-{hops}, K={k}"] = sum(max(h["seconds"].values()) for h in hist[:k])
        local = []
        for m, c in b["controllers"].items():
            tick = time.perf_counter()
            T.local_controller.solve_local(c, p, T._REG[d.period(t)]["regions"][m])
            local.append(time.perf_counter() - tick)
        secs["IR-MPC"], secs["IR-MPC serial"] = max(local), sum(local)
        tick = time.perf_counter()
        ir.solve_full(model, p)
        secs["central"] = time.perf_counter() - tick
        return {"seed": seed, "t": t, "seconds": secs}
    J, const = T.objective(model, p, central["x"]), model["objective_constant"]
    rows = {}
    for name, (a, failed, k) in acts.items():
        full = {m: (np.asarray(v) if v is not None else np.asarray(prev[m], float)) for m, v in a.items()}
        e = [float(np.max(np.abs(full[m] - star[m]))) for m in star]
        Ja = T.fixed_cost(model, p, full)
        rows[name] = {"solved": 1 - len(failed) / len(star), "errors": e, "iterations": k,
                      "gap": None if Ja is None else (Ja - J) / (J + const)}
    return {"seed": seed, "t": t, "rows": rows, "seconds": secs,
            "regions": {s: len(_P["parts"][s]) for s in SIZES},
            "partition_md5": hashlib.md5(json.dumps(_P["parts"], sort_keys=True).encode()).hexdigest()}


def accuracy():
    with Pool(12, initializer=_init) as pool:
        res = pool.starmap(one, T.samples_list(), chunksize=1)
    (T.OUT / "TABLES_partition.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    _init()
    (T.OUT / "PARTITIONS.json").write_text(json.dumps({str(k): v for k, v in _P["parts"].items()}, indent=1),
                                          encoding="utf-8")
    print("partition accuracy done", len(res))


if __name__ == "__main__":
    {"accuracy": accuracy}[sys.argv[1]]()
