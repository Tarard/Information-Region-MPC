"""IR-MPC trial at one size (user, 2026-10-30: try our method at 100, 200 and 500 intersections; in the spirit of
PLAN section 9). Needs, under NS_ROOT, work/N{N}/net.json and the centralized days runs/central_{1..4, 101, 102}.

1. Information regions: sensitivity coupling (v1 method) on up to 40 peak states of the history days 1-4 (sigma = the
   spread of the 4 days at the same clock); W-bar = mean over the states at which m decides; distance order, eps_W 1 s.
2. Reference at the peak clock TSTAR = 4200 s (every intersection decides): p-bar = mean of the 4 history days,
   x-ref = centralized optimum at p-bar (accepted at 1e-7). Boundary functions (construction.construct, boundary range
   from the 4 history days, ns_ir.compact) for every intersection, or for SAMPLE intersections stratified by stage
   count at 500 (fixed seed). Constructions run in worker processes with a memory cap; a worker that dies is retried
   alone with a larger cap.
3. Evaluation at TSTAR on the 4 history days (in sample) and the test days 101-102 (out of sample): centralized time
   and first decision (accepted at 1e-5, as in closed loop), IR-MPC local QP time, solved share and first-decision
   error per intersection (a failed local QP applies the previous greens and counts).
Usage: ir_trial.py N [workers] [memory_gb]   -> NS_ROOT/IR_TRIAL_N{N}.json
"""
import gzip
import json
import os
from pathlib import Path
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
import ns_design as d  # noqa: E402

d.enforce_threads()
import numpy as np  # noqa: E402

import construction  # noqa: E402
import grid_net  # noqa: E402
import ns_common as nc  # noqa: E402
import ns_ir  # noqa: E402
import ns_offline as no  # noqa: E402
import strict_qp  # noqa: E402

HIST, TEST, TSTAR, SAMPLE, COUPLING_STATES = (1, 2, 3, 4), (101, 102), 4200, 50, 40
_W = {}


def cap_memory(gigabytes):
    """Windows job object with a per-process committed-memory cap (as bank_runner.py)."""
    import ctypes
    from ctypes import wintypes

    class IO(ctypes.Structure):
        _fields_ = [(k, ctypes.c_ulonglong) for k in ("r", "w", "o", "rb", "wb", "ob")]

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXT(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", BASIC), ("IoInfo", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    job = k32.CreateJobObjectW(None, None)
    info = EXT()
    info.BasicLimitInformation.LimitFlags = 0x100
    info.ProcessMemoryLimit = int(gigabytes * 2 ** 30)
    k32.SetInformationJobObject(wintypes.HANDLE(job), 9, ctypes.byref(info), ctypes.sizeof(info))
    k32.AssignProcessToJobObject(wintypes.HANDLE(job), wintypes.HANDLE(k32.GetCurrentProcess()))
    _W["job"] = job


def states(w, seeds):
    out = {}
    for s in seeds:
        z = np.load(w / f"runs/central_{s}/states.npz")
        out[s] = {int(t): (p, x) for t, p, x in zip(z["t"], z["p"], z["x"])}
    return out


# ---------------------------------------------------------------- coupling

def _coupling_init(n, gb):
    cap_memory(gb)
    _W["net"] = grid_net.load(n)
    _W["labels"] = no.labels_of(_W["net"])


def _coupling_job(args):
    t, p, sigma = args
    net, labels = _W["net"], _W["labels"]
    model = no.model_at(net, labels, p, t)
    kinds = np.asarray([lab[0] for lab in model["param_labels"]])
    owners = np.asarray(model["param_owner"])
    cols = {j: np.flatnonzero((owners == j) & np.isin(kinds, no.PERTURBED)) for j in net["signals"]}
    tick = time.perf_counter()
    W = no.sensitivity_W(model, p, {j: (c, sigma[c]) for j, c in cols.items()}, owners)
    return t, (None if W is None else {m: dict(v) for m, v in W.items()}), time.perf_counter() - tick


# ---------------------------------------------------------------- constructions

def _bank_init(n, gb, payload):
    cap_memory(gb)
    _W["net"] = grid_net.load(n)
    _W["labels"] = no.labels_of(_W["net"])
    with open(payload, "rb") as h:
        _W.update(pickle.load(h))
    _W["model"] = no.model_at(_W["net"], _W["labels"], _W["pbar"], TSTAR)


def _bank_job(m):
    model = _W["model"]
    tick = time.perf_counter()
    try:
        c = construction.construct(model, m, _W["pbar"], _W["xref"], _W["hp"], _W["hx"])
    except MemoryError:
        return m, None, "OUT_OF_MEMORY", time.perf_counter() - tick
    seconds = time.perf_counter() - tick
    if c["status"] != "CONSTRUCTED":
        return m, None, c.get("reason", c["status"]), seconds
    cols = np.flatnonzero(np.isin(np.asarray(model["param_owner"]), _W["regions"][m]))
    return m, ns_ir.compact(c, cols), "CONSTRUCTED", seconds


def construct_all(n, targets, workers, gb, payload):
    done = {}
    todo = list(targets)
    try:
        with ProcessPoolExecutor(workers, initializer=_bank_init, initargs=(n, gb, payload)) as pool:
            for m, c, status, s in pool.map(_bank_job, todo):
                done[m] = (c, status, s)
                print(f"N={n} construct {m}: {status} {s:.1f} s", flush=True)
    except BrokenProcessPool:
        print(f"N={n} a construction worker died; the rest run alone with {2 * gb} GB", flush=True)
    for m in [m for m in todo if m not in done]:
        try:
            with ProcessPoolExecutor(1, initializer=_bank_init, initargs=(n, 2 * gb, payload)) as pool:
                done[m] = pool.submit(_bank_job, m).result()[1:]
        except BrokenProcessPool:
            done[m] = (None, "WORKER_DIED", None)
        print(f"N={n} construct {m} (alone): {done[m][1]}", flush=True)
    return done


def evaluate(n, net, labels, hist, test, ctrl, pbar, tls, out):
    """Centralized QP time and local QP time and first-decision error at TSTAR on every day."""
    evals = []
    for kind, data in (("history", hist), ("test", test)):
        for s, rows_s in data.items():
            if TSTAR not in rows_s:
                out.setdefault("days_without_state_at_tstar", []).append(s)   # centralized QP failed there
                continue
            p = rows_s[TSTAR][0]
            model = no.model_at(net, labels, p, TSTAR)
            strict_qp.TOLERANCE = d.TOLERANCE                        # centralized reference as in closed loop
            tick = time.perf_counter()
            central = nc.solve_full(model, p)
            central_s = time.perf_counter() - tick
            strict_qp.TOLERANCE = d.OFFLINE_TOLERANCE                # local QPs as in the open-loop samples
            star = nc.first_decision(model, central["x"]) if central["valid"] else None
            prev = {m: p[[i for i, lab in enumerate(model["param_labels"]) if lab[0] == "previous_green" and lab[1] == m]]
                    for m in ctrl}
            local = {}
            for m, c in ctrl.items():
                tick = time.perf_counter()
                r = ns_ir.solve_local(c, p, pbar)
                sec = time.perf_counter() - tick
                a = np.asarray(r["action"]) if r["valid"] else prev[m]
                local[m] = {"seconds": sec, "valid": bool(r["valid"]),
                            "error": float(np.max(np.abs(a - star[m]))) if star else None}
            secs = [v["seconds"] for v in local.values()]
            errs = np.array([v["error"] for v in local.values() if v["error"] is not None])
            evals.append({"kind": kind, "seed": s, "central_seconds": central_s, "central_valid": bool(central["valid"]),
                          "central_variables": int(model["H"].shape[0]),
                          "ir_parallel_seconds": float(max(secs)), "ir_median_seconds": float(np.median(secs)),
                          "ir_serial_seconds_estimate_all": float(np.mean(secs) * len(tls)),
                          "solved": float(np.mean([v["valid"] for v in local.values()])),
                          "error_mean": float(errs.mean()) if len(errs) else None,
                          "error_le_1": float((errs <= 1).mean()) if len(errs) else None,
                          "error_le_2": float((errs <= 2).mean()) if len(errs) else None,
                          "error_max": float(errs.max()) if len(errs) else None})
            print(f"N={n} eval {kind} {s}: " + json.dumps({k: v for k, v in evals[-1].items() if k not in ("kind", "seed")}),
                  flush=True)
    return evals


# ---------------------------------------------------------------- main

def main(n, workers, gb):
    w = d.size_dir(n)
    net = grid_net.load(n)
    labels = no.labels_of(net)
    hist, test = states(w, HIST), states(w, TEST)
    out = {"n": n, "tstar": TSTAR, "history_days": list(HIST), "test_days": list(TEST)}
    for s in HIST + TEST:
        out.setdefault("days", {})[s] = json.loads((w / f"runs/central_{s}/SUMMARY.json").read_text(encoding="utf-8"))
    # 1. regions from the sensitivity coupling on peak states
    rng = np.random.default_rng(n)
    peak = sorted({t for s in hist for t in hist[s] if d.period(t) == "peak" and all(t in hist[q] for q in hist)})
    picks = [(s, t) for s in hist for t in peak]
    picks = [picks[i] for i in sorted(rng.choice(len(picks), min(COUPLING_STATES, len(picks)), replace=False))]
    jobs = [(t, hist[s][t][0], np.std([hist[q][t][0] for q in hist], axis=0, ddof=1)) for s, t in picks]
    tick = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=_coupling_init, initargs=(n, gb)) as pool:
        res = list(pool.map(_coupling_job, jobs))
    out["coupling_seconds_wall"] = time.perf_counter() - tick
    out["coupling_seconds_per_state_median"] = float(np.median([r[2] for r in res]))
    tls = sorted(net["signals"])
    Wbar = {}
    for m in tls:
        rows = [W[m] for _, W, _ in res if W is not None and m in W]
        Wbar[m] = {j: float(np.mean([r[j] for r in rows])) for j in tls if j != m} if rows else {}
    regions = no.regions_from(Wbar, net["adjacency"], net["coords"])
    sizes = np.array([len(regions[m]) for m in tls])
    stages = {m: net["signals"][m]["stage_count"] for m in tls}
    out["regions"] = {"coupling_states": len(res), "failed_states": sum(W is None for _, W, _ in res),
                      "mean": float(sizes.mean()), "median": float(np.median(sizes)), "p90": float(np.percentile(sizes, 90)),
                      "max": int(sizes.max()), "one_share": float((sizes == 1).mean()),
                      "mean_by_stages": {k: float(np.mean([len(regions[m]) for m in tls if stages[m] == k]))
                                         for k in sorted(set(stages.values()))}}
    print(f"N={n} regions: {json.dumps(out['regions'])}", flush=True)
    # 2. reference and boundary functions at TSTAR
    used = [s for s in HIST if TSTAR in hist[s]]            # a day whose centralized QP failed at TSTAR has no state
    out["reference_days"] = used
    rows = [hist[s][TSTAR] for s in used]
    pbar = np.mean([p for p, _ in rows], axis=0)
    model_ref = no.model_at(net, labels, pbar, TSTAR)
    ref = nc.solve_full(model_ref, pbar)
    out["reference_valid_1e-7"] = bool(ref["valid"])
    if not ref["valid"]:
        out["stop"] = "reference QP not accepted at 1e-7"
        return out
    xref = ref["x"]
    # the stored x of the days may come from an earlier objective (CHANGES.md section 12): the history solutions at
    # TSTAR are solved again with the current model at the stored states (offline acceptance 1e-7)
    fresh = []
    for p_s, _ in rows:
        r_s = nc.solve_full(no.model_at(net, labels, p_s, TSTAR), p_s)
        if r_s["valid"]:
            fresh.append((p_s, r_s["x"]))
    out["history_solutions_resolved"] = len(fresh)
    rows = fresh
    hp = np.asarray([p for p, _ in rows])
    hx = np.vstack([np.asarray([x for _, x in rows]), xref[None]])
    if n <= 200:
        targets = sorted(model_ref["output_indices"])
    else:
        r2 = np.random.default_rng(0)
        targets = []
        for k in sorted(set(stages.values())):
            group = [m for m in tls if stages[m] == k]
            take = max(1, round(SAMPLE * len(group) / len(tls)))
            targets += sorted(r2.choice(group, min(take, len(group)), replace=False).tolist())
    payload = d.ROOT / f"trial_payload_N{n}.pkl"
    with open(payload, "wb") as h:
        pickle.dump({"pbar": pbar, "xref": xref, "hp": hp, "hx": hx, "regions": regions}, h)
    tick = time.perf_counter()
    built = construct_all(n, targets, workers, gb, payload)
    out["construction"] = {"targets": len(targets), "constructed": sum(c is not None for c, _, _ in built.values()),
                           "failed": {m: st for m, (c, st, _) in built.items() if c is None},
                           "seconds_median": float(np.median([s for _, _, s in built.values() if s])),
                           "seconds_max": float(np.max([s for _, _, s in built.values() if s])),
                           "wall_seconds": time.perf_counter() - tick, "workers": workers, "memory_cap_gb": gb}
    ctrl = {m: c for m, (c, _, _) in built.items() if c is not None}
    out["local_variables"] = {"median": float(np.median([len(c["reduction"]["core"]) + c["boundary_dimension"]
                                                          for c in ctrl.values()])) if ctrl else None}
    with gzip.open(d.ROOT / f"trial_bank_N{n}.pkl.gz", "wb") as h:
        pickle.dump({"pbar": pbar, "controllers": ctrl}, h)
    # 3. evaluation at TSTAR
    out["evaluation"] = evaluate(n, net, labels, hist, test, ctrl, pbar, tls, out)
    return out


if __name__ == "__main__":
    n = int(sys.argv[1])
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else d.WORKERS
    gb = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    res = main(n, workers, gb)
    (d.ROOT / f"IR_TRIAL_N{n}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"wrote IR_TRIAL_N{n}.json", flush=True)
