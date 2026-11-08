"""Offline design of one grid size from its history days (TASK.md section 4).

refs     : REFERENCE_MODE "update" (default, the paper's method): per update clock t, p-bar(t) = mean history
           parameter at t; "period": per demand period, the mean over its updates at the period's middle clock.
           x-ref and multipliers = centralized optimum at p-bar. -> offline/references.pkl.gz
coupling : W_mj per sample = largest first-decision change at m when j's road counts, entry backlogs and release
           rates move by one history standard deviation (clipped at zero). v2 (PLAN section 7): COUPLING = "resolve"
           re-solves the whole centralized QP for every j and sign (main.tex Eq. (24); a perturbed QP that fails is
           left out, and a pair with both signs failed is left out of the mean); "sensitivity" uses the sensitivity of
           the centralized solution at a fixed active set (v1). W-bar = mean over the period's samples (at most 40).
           Information region: distance order (hops, ties by grid position), k_m = min{k : sum of W-bar over unread
           intersections <= eps_W = 1 s} (Eq. (25)). -> offline/REGIONS.json
checkw   : the sensitivity coupling on the same samples as the coupling stage, compared with REGIONS.json (W-bar
           differences and regions per period). -> offline/REGIONS_SENSITIVITY.json, offline/CHECK_W.json
parts    : OSD partitions (ns_osd.partition) for OSD_REGION_SIZES, weighted by the peak-period W-bar.
           -> offline/PARTITIONS.json
bank     : per reference and intersection, construction.construct at (p-bar, x-ref), boundary range from the
           history solutions of that reference (same clock, or the period), widened by 10% on each side, compacted
           to the information region of the clock's period. -> offline/bank_{key}.pkl.gz
Usage: ns_offline.py N refs|coupling|checkw|parts|bank
"""

import ns_design as d

d.enforce_threads()

from collections import defaultdict  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402
from scipy.sparse.linalg import splu  # noqa: E402

import construction  # noqa: E402
import grid_net  # noqa: E402
import ns_common as nc  # noqa: E402
import fair_solver  # noqa: E402
import ns_ir  # noqa: E402
import ns_osd  # noqa: E402
from plant import Plant  # noqa: E402

PERTURBED = ("n0", "backlog", "release_rate")
COUPLING_SAMPLES_PER_PERIOD = 40
COUPLING = "resolve"
fair_solver.accept_offline()          # offline: accept at 1e-7 (PLAN v2 section 7)
_G = {}


def folder(n):
    f = d.size_dir(n) / "offline"
    f.mkdir(parents=True, exist_ok=True)
    return f


def history(n):
    """{seed: {t: (p, x)}} from the centralized history runs that cleared."""
    out = {}
    for seed in d.HISTORY_SEEDS:
        run = d.size_dir(n) / f"runs/central_{seed}"
        s = json.loads((run / "SUMMARY.json").read_text(encoding="utf-8"))
        if not s["cleared"]:
            print(f"history day {seed} did not clear; excluded", flush=True)
            continue
        z = np.load(run / "states.npz")
        out[seed] = {int(t): (p, x) for t, p, x in zip(z["t"], z["p"], z["x"])}
    return out


def labels_of(net):
    prev = {a: list(s["reference_green_seconds_at_t0"]) for a, s in net["signals"].items()}
    model = nc.build(net, Plant(net, 0).observe(prev), 0)
    return model["param_labels"]


def deciders_at(net, ref):
    """Intersections that decide at the reference clock (a cycle of theirs starts then)."""
    return nc.deciders(net["signals"], ref["clock"])


def model_at(net, labels, p, t):
    return nc.build(net, nc.snapshot_from_p(net, labels, p, t), t)


def samples_by_period(hist):
    out = defaultdict(list)
    for seed, rows in hist.items():
        for t, (p, x) in sorted(rows.items()):
            out[d.period(t)].append((seed, t, p, x))
    return out


def reference_groups(hist):
    """{key: (clock, [(p, x), ...])}: one group per update clock present in every history day ("update" mode, the
    paper's p-bar(t)), or one per demand period with its middle clock ("period" mode)."""
    groups = {}
    if d.REFERENCE_MODE == "update":
        common = sorted(set.intersection(*(set(rows) for rows in hist.values())))
        for t in common:
            groups[t] = (t, [hist[s][t] for s in hist])
    else:
        for period, rows in samples_by_period(hist).items():
            clocks = sorted({t for _, t, _, _ in rows})
            groups[period] = (clocks[len(clocks) // 2], [(p, x) for _, _, p, x in rows])
    return groups


def refs(n):
    net, hist = grid_net.load(n), history(n)
    labels = labels_of(net)
    out = {}
    for key, (clock, rows) in reference_groups(hist).items():
        pbar = np.mean([p for p, _ in rows], axis=0)
        model = model_at(net, labels, pbar, clock)
        r = nc.solve_full(model, pbar)
        if not r["valid"]:
            raise SystemExit(f"Reference QP {key} failed.")
        out[key] = {"clock": clock, "pbar": pbar, "xref": r["x"], "nu_ref": r["nu"], "samples": len(rows)}
    print(f"N={n} references ({d.REFERENCE_MODE} mode): {len(out)}", flush=True)
    with gzip.open(folder(n) / "references.pkl.gz", "wb") as h:
        pickle.dump(out, h)


def _init(n):
    _G["n"], _G["net"] = n, grid_net.load(n)
    _G["labels"] = labels_of(_G["net"])
    _G["hist"] = history(n)


def sensitivity_W(model, p, sigma_cols, owners):
    """W^S_mj for all m, j at the centralized optimum of p (fixed active set)."""
    r = nc.solve_full(model, p)
    if not r["valid"]:
        return None
    x, lam = r["x"], r["lambda"]
    G = model["G"].tocsr()
    slack = (model["g0"] + model["gP"] @ p) - G @ x
    active = np.flatnonzero((lam > 1e-7) | (slack < 1e-7))
    H, E, GA = model["H"].tocsc(), model["E"].tocsc(), G[active].tocsc()
    nx_, ne, na = H.shape[0], E.shape[0], GA.shape[0]
    reg = 1e-9
    K = sparse.bmat([[H + reg * sparse.eye(nx_), E.T, GA.T],
                     [E, -reg * sparse.eye(ne), None],
                     [GA, None, -reg * sparse.eye(na)]], format="csc")
    lu = splu(K)
    out_idx = model["output_indices"]
    W = defaultdict(dict)
    for j, (cols, sig) in sigma_cols.items():
        dp = np.zeros(len(p))
        dp[cols] = sig
        rhs = np.r_[-(model["hP"] @ dp), model["eP"] @ dp, (model["gP"] @ dp)[active]]
        dx = lu.solve(rhs)[:nx_]
        for m, idx in out_idx.items():
            if m != j:
                W[m][j] = float(np.max(np.abs(dx[idx])))
    return W


def resolve_W(model, p, sigma_cols):
    """W_mj for all m, j by re-solving the centralized QP for every j and sign (Eq. (24))."""
    r = nc.solve_full(model, p)
    if not r["valid"]:
        return None, 1, 1
    base = nc.first_decision(model, r["x"])
    W, solves, failed = defaultdict(dict), 1, 0
    for j, (cols, sig) in sigma_cols.items():
        change, ok = defaultdict(float), False
        for sign in (-1, 1):
            q = p.copy()
            q[cols] = np.maximum(0.0, p[cols] + sign * sig)
            rr = nc.solve_full(model, q)
            solves += 1
            if not rr["valid"]:
                failed += 1
                continue
            ok = True
            for m, a in nc.first_decision(model, rr["x"]).items():
                change[m] = max(change[m], float(np.max(np.abs(a - base[m]))))
        for m in base:
            if m != j:
                W[m][j] = change[m] if ok else None
    return W, solves, failed


def _coupling_job(job, method=None):
    seed, t = job
    net, labels, hist = _G["net"], _G["labels"], _G["hist"]
    p = hist[seed][t][0]
    model = model_at(net, labels, p, t)
    kinds = np.asarray([lab[0] for lab in model["param_labels"]])
    owners = np.asarray(model["param_owner"])
    same = [hist[s][t][0] for s in hist if t in hist[s]]
    if len(same) >= 2:
        sigma = np.std(same, axis=0, ddof=1)
    else:                                         # fewer than two days at this clock: the period's spread
        sigma = np.std([q for s in hist for tt, (q, _) in hist[s].items() if d.period(tt) == d.period(t)], axis=0, ddof=1)
    cols = {j: np.flatnonzero((owners == j) & np.isin(kinds, PERTURBED)) for j in net["signals"]}
    sigma_cols = {j: (c, sigma[c]) for j, c in cols.items()}
    tick = time.perf_counter()
    if (method or COUPLING) == "resolve":
        W, solves, failed = resolve_W(model, p, sigma_cols)
    else:
        W = sensitivity_W(model, p, sigma_cols, owners)
        solves, failed = 1, int(W is None)
    return {"seed": seed, "t": t, "period": d.period(t), "W": W, "seconds": time.perf_counter() - tick,
            "solves": solves, "failed_solves": failed}


def _sensitivity_job(job):
    return _coupling_job(job, "sensitivity")


def distance_order(adjacency, coords, m):
    dist, frontier = {m: 0}, [m]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adjacency[u]:
                if v not in dist:
                    dist[v] = dist[u] + 1
                    nxt.append(v)
        frontier = nxt
    return sorted(adjacency, key=lambda j: (dist.get(j, 10 ** 6), tuple(coords[j])))


def regions_from(Wbar, adjacency, coords):
    regions = {}
    for m in sorted(adjacency):
        order = distance_order(adjacency, coords, m)
        k = next(k for k in range(1, len(order) + 1) if sum(Wbar[m].get(j, 0.0) for j in order[k:]) <= d.EPS_W)
        regions[m] = order[:k]
    return regions


def coupling_jobs(n):
    hist = history(n)
    rng = np.random.default_rng(n)
    jobs = []
    for period, rows in samples_by_period(hist).items():
        rows = [(s, t) for s, t, _, _ in rows]
        if len(rows) > COUPLING_SAMPLES_PER_PERIOD:
            rows = [rows[i] for i in sorted(rng.choice(len(rows), COUPLING_SAMPLES_PER_PERIOD, replace=False))]
        jobs += rows
    return jobs


def regions_out(n, res, method):
    net = grid_net.load(n)
    out = {"eps_W": d.EPS_W, "method": method, "rule": __doc__, "periods": {}}
    for period in d.PERIODS:
        rows = [r["W"] for r in res if r["period"] == period and r["W"] is not None]
        if not rows:
            continue
        Wbar, missing = {}, 0
        for m in net["signals"]:
            Wbar[m] = {}
            for j in net["signals"]:
                if j != m:
                    have = [r[m][j] for r in rows if m in r]           # samples at which m decides
                    v = [w for w in have if w is not None]
                    missing += len(have) - len(v)
                    Wbar[m][j] = float(np.mean(v)) if v else 0.0
        regions = regions_from(Wbar, net["adjacency"], net["coords"])
        sizes = sorted(len(v) for v in regions.values())
        sel = [r for r in res if r["period"] == period]
        info = {"samples": len(rows), "failed_samples": sum(r["W"] is None for r in sel),
                "solves": sum(r["solves"] for r in sel), "failed_solves": sum(r["failed_solves"] for r in sel),
                "pairs_without_valid_sign": missing, "regions": regions, "W_bar": Wbar,
                "size_median": sizes[len(sizes) // 2], "size_max": sizes[-1]}
        out["periods"][period] = info
        print(f"N={n} {method} {period}: {len(rows)} samples, region size median {info['size_median']} max "
              f"{info['size_max']}, failed solves {info['failed_solves']} of {info['solves']}", flush=True)
    out["seconds_per_sample_median"] = float(np.median([r["seconds"] for r in res]))
    out["samples"] = [{k: r[k] for k in ("seed", "t", "period", "seconds", "solves", "failed_solves")} for r in res]
    return out


def coupling(n):
    with Pool(d.WORKERS, initializer=_init, initargs=(n,)) as pool:
        res = pool.map(_coupling_job, coupling_jobs(n), chunksize=1)
    (folder(n) / "REGIONS.json").write_text(json.dumps(regions_out(n, res, COUPLING)), encoding="utf-8")


def checkw(n):
    """Sensitivity coupling (v1) on the coupling samples, compared with the re-solve coupling of REGIONS.json."""
    with Pool(d.WORKERS, initializer=_init, initargs=(n,)) as pool:
        res = pool.map(_sensitivity_job, coupling_jobs(n), chunksize=1)
    sens = regions_out(n, res, "sensitivity")
    (folder(n) / "REGIONS_SENSITIVITY.json").write_text(json.dumps(sens), encoding="utf-8")
    ref = json.loads((folder(n) / "REGIONS.json").read_text(encoding="utf-8"))
    out = {"resolve_method": ref["method"], "periods": {}}
    for period, rp in ref["periods"].items():
        sp = sens["periods"].get(period)
        if not sp:
            continue
        a = np.array([rp["W_bar"][m][j] for m in rp["W_bar"] for j in rp["W_bar"][m]])
        b = np.array([sp["W_bar"][m][j] for m in rp["W_bar"] for j in rp["W_bar"][m]])
        rr, ss = rp["regions"], sp["regions"]
        out["periods"][period] = {
            "pairs": int(len(a)), "mean_abs_difference_s": float(np.mean(np.abs(a - b))),
            "max_abs_difference_s": float(np.max(np.abs(a - b))),
            "correlation": float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else None,
            "region_size_mean_resolve": float(np.mean([len(v) for v in rr.values()])),
            "region_size_mean_sensitivity": float(np.mean([len(v) for v in ss.values()])),
            "same_region": sum(rr[m] == ss[m] for m in rr), "intersections": len(rr)}
    (folder(n) / "CHECK_W.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out), flush=True)


def parts(n):
    net = grid_net.load(n)
    reg = json.loads((folder(n) / "REGIONS.json").read_text(encoding="utf-8"))["periods"]
    Wbar = (reg.get("peak") or next(iter(reg.values())))["W_bar"]
    out = {str(size): ns_osd.partition(net["adjacency"], Wbar, size) for size in sorted(set(d.OSD_REGION_SIZES + (d.OSD_CLOSED_LOOP_REGION,)))}
    (folder(n) / "PARTITIONS.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"N={n} partitions: " + ", ".join(f"size {k}: {len(v)} regions" for k, v in out.items()), flush=True)


def _bank_init(n):
    _init(n)
    with gzip.open(folder(n) / "references.pkl.gz", "rb") as h:
        _G["refs"] = pickle.load(h)
    _G["regions"] = json.loads((folder(n) / "REGIONS.json").read_text(encoding="utf-8"))["periods"]
    _G["models"] = {}


def _bank_job(job):
    key, m = job
    net, ref = _G["net"], _G["refs"][key]
    if key not in _G["models"]:
        _G["models"] = {key: model_at(net, _G["labels"], ref["pbar"], ref["clock"])}     # keep one model in memory
        _G["rows"] = reference_groups(_G["hist"])[key][1]
    model, rows = _G["models"][key], _G["rows"]
    hp = np.asarray([p for p, _ in rows])
    hx = np.vstack([np.asarray([x for _, x in rows]), ref["xref"][None]])
    tick = time.perf_counter()
    try:
        c = construction.construct(model, m, ref["pbar"], ref["xref"], hp, hx)
    except MemoryError:                     # retried alone after the pool (bank below)
        return key, m, "MEMORY", time.perf_counter() - tick
    seconds = time.perf_counter() - tick
    if c["status"] != "CONSTRUCTED":
        return key, m, None, seconds
    owners = np.asarray(model["param_owner"])
    period = d.period(ref["clock"])
    region = (_G["regions"].get(period) or _G["regions"]["peak"])["regions"][m]
    cols = np.flatnonzero(np.isin(owners, region))
    return key, m, ns_ir.compact(c, cols), seconds


def bank(n):
    with gzip.open(folder(n) / "references.pkl.gz", "rb") as h:
        refs_ = pickle.load(h)
    net = grid_net.load(n)
    dec = {key: deciders_at(net, refs_[key]) for key in refs_}
    jobs = [(key, m) for key in refs_ for m in dec[key]]                          # grouped by reference
    started = time.perf_counter()
    by, secs, failed, retry = defaultdict(dict), [], [], []

    def write(key):
        with gzip.open(folder(n) / f"bank_{key}.pkl.gz", "wb") as h:
            pickle.dump({"pbar": refs_[key]["pbar"], "controllers": by.pop(key)}, h)

    with Pool(d.WORKERS, initializer=_bank_init, initargs=(n,)) as pool:
        for key, m, c, s in pool.imap(_bank_job, jobs, chunksize=max(1, len(net["signals"]) // d.WORKERS)):
            if isinstance(c, str):
                retry.append([key, m])
                continue
            secs.append(s)
            if c is None:
                failed.append([key, m])
            else:
                by[key][m] = c
            done = len(by[key]) + sum(1 for k, _ in failed if k == key)
            if done == len(dec[key]):                                              # write each reference when complete
                write(key)
    # A construction that ran out of memory next to the other workers is retried alone; a second failure counts as
    # a failed construction.
    memory = []
    if retry:
        _bank_init(n)
        for key, m in retry:
            _, _, c, s = _bank_job((key, m))
            secs.append(s)
            memory.append([key, m, "retried" if not isinstance(c, str) else "out_of_memory"])
            if c is None or isinstance(c, str):
                failed.append([key, m])
            else:
                by[key][m] = c
        for key in [k for k in refs_ if k in by]:
            write(key)
    info = {"mode": d.REFERENCE_MODE, "references": len(refs_), "constructions": len(secs), "failed": failed,
            "memory_retries": memory,
            "seconds_median": float(np.median(secs)),
            "seconds_max": float(np.max(secs)), "wall_seconds": time.perf_counter() - started}
    (folder(n) / "BANK_INFO.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"N={n} bank: {json.dumps(info)}", flush=True)


if __name__ == "__main__":
    n, stage = int(sys.argv[1]), sys.argv[2]
    {"refs": refs, "coupling": coupling, "checkw": checkw, "parts": parts, "bank": bank}[stage](n)
