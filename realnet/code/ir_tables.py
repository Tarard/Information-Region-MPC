"""Paper-style tables for the IR-MPC trial (history 7001, 7002; test 7003, 77). main.tex Tables I-III and Sec. VI text.

closed : Table I traffic (delay, TTS per test day), Table II next-cycle errors by demand period from the IR-MPC
         closed-loop runs (error = max over phases of |IR - centralized| at the same state; identical executed greens =
         the whole-second greens equal those centralized MPC would execute at that state; failed local QPs apply the
         previous greens and count), predicted-trajectory differences of the solved local QPs (green per step, vehicle
         counts on the intersection's own roads per step), closed-loop vehicle counts per hour and clearance time.
samples: Table III on 50 samples (25 evenly spaced updates before 7200 s from each centralized test run): IR-MPC,
         OSD-1 and OSD-2 with K = 1 and 4. Solved, errors, cost gap = (J(a) - J*) / (J* + constant) with all
         next-cycle greens fixed to the method's actions and the rest re-optimized (median over samples).
timing : 19 of the 50 samples, one process, nothing else running: centralized solve, IR-MPC (slowest local QP =
         parallel time), OSD (sum over iterations of the slowest local QP).
Usage: ir_tables.py closed | samples | timing | report
"""

import cc_design as d

d.enforce_threads()

from collections import defaultdict  # noqa: E402
import gzip  # noqa: E402
import json  # noqa: E402
from multiprocessing import Pool  # noqa: E402
import pickle  # noqa: E402
import statistics as stats  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import xml.etree.ElementTree as ET  # noqa: E402

import numpy as np  # noqa: E402
from scipy import sparse  # noqa: E402

import cc_executor as ex  # noqa: E402
import ir_common as ir  # noqa: E402
import local_controller  # noqa: E402
import osd  # noqa: E402
from frozen import strict_qp  # noqa: E402

if len(sys.argv) > 1 and sys.argv[1] in ("samples", "timing"):
    strict_qp.TOLERANCE = d.OPEN_LOOP_TOLERANCE      # Table III is open loop; "closed" keeps the closed-loop value

OUT = d.ROOT / "results_ir_v1"
IR_RUNS = OUT / "runs"
PERIODS = d.PERIODS
_ST = _REG = _ADJ = None


def _init():
    global _ST, _REG, _ADJ
    _ST = ir.static()
    _REG = ir.load(d.ROOT / "design/ir/REGIONS.json")["periods"]
    _ADJ = ir.intersection_graph(_ST)


def bank(t):
    with gzip.open(d.ROOT / f"design/ir/bank/t{t}.pkl.gz", "rb") as handle:
        return pickle.load(handle)


def executed_integer(values, tls):
    legal, _ = ex.legal_green(values, _ST["signals"][tls])
    return ex.integer_green(legal, _ST["signals"][tls])


def trips(path):
    delay, tts = [], 0.0
    for t in ET.parse(path).getroot().iter("tripinfo"):
        if float(t.get("arrival", "-1")) >= 0:
            delay.append(float(t.get("timeLoss")))
            tts += float(t.get("duration")) + float(t.get("departDelay"))
    return float(np.mean(delay)), tts / 3600


def trajectory(job):
    """Re-solve the local QPs of one IR-run update; differences to the centralized prediction at the same state."""
    seed, t = job
    run = IR_RUNS / f"ir_seed{seed}"
    z = np.load(run / f"vectors/t{t}.npz")
    p, xc = z["p"], z["x"]
    model = ir.model_at(_ST, t, ir.load(run / f"snapshots/t{t}.json"))
    ctrls = bank(t)["controllers"]
    G, N = model["indices"]["green"], model["indices"]["inventory"]
    own_roads = defaultdict(list)
    for e, m in _ST["edge_owner"].items():
        own_roads[m].append(e)
    out = {"green": [[] for _ in range(d.STEPS)], "count": [[] for _ in range(d.STEPS)]}
    layout = model["metadata"]["cycle_layout"]
    for m, c in ctrls.items():
        r = local_controller.solve_local(c, p, _REG[d.period(t)]["regions"][m])
        if not r["valid"]:
            continue
        core = c["reduction"]["core"]
        x = np.full(len(xc), np.nan)
        x[core] = r["local_x"]
        phases = _ST["network"]["signals"][m]["green_native_indices"]
        for k in range(d.STEPS):               # green of the cycle serving step k (m decides now: no running cycle)
            cyc = layout[m]["serving"][k]
            out["green"][k].append(float(np.mean([abs(x[G[m, cyc, ph]] - xc[G[m, cyc, ph]]) for ph in phases])))
            idx = [N[e, k + 1] for e in own_roads[m]]
            if idx and np.isfinite(x[idx]).all():
                out["count"][k].append(float(np.mean(np.abs(x[idx] - xc[idx]))))
    return out


def closed():
    _init()
    rows = {"table1": {}, "table2": {}, "trajectory": {}, "closed_loop": {}}
    err = defaultdict(list)
    same = defaultdict(list)
    failures = defaultdict(int)
    jobs = []
    for seed in ir.TEST:
        crun, irun = ir.RUNS / f"mix_seed{seed}", IR_RUNS / f"ir_seed{seed}"
        rows["table1"][seed] = {"central": trips(crun / "tripinfo.xml"), "ir": trips(irun / "tripinfo.xml")}
        for line in open(irun / "decisions.jsonl", encoding="utf-8"):
            r = json.loads(line)
            per = d.period(r["time_seconds"])
            failures[per] += len(r["ir"]["failed"])
            if (d.ROOT / f"design/ir/bank/t{r['time_seconds']}.pkl.gz").exists():   # no bank after the history ends
                jobs.append((seed, r["time_seconds"]))
            for m, a in r["mpc_first_cycle_green_seconds"].items():
                c = r["central_first_green_seconds"][m]
                if c is None:
                    continue
                e = max(abs(x - y) for x, y in zip(a, c))
                err[per].append(e)
                same[per].append(executed_integer(c, m) == r["executed_green_seconds"][m])
        ct = [json.loads(x) for x in open(crun / "trace.jsonl", encoding="utf-8")]
        it = [json.loads(x) for x in open(irun / "trace.jsonl", encoding="utf-8")]
        hourly = {}
        for h in range(int(max(len(ct), len(it)) // 3600) + 1):
            a = [r["vehicles"] for r in ct[h * 3600:(h + 1) * 3600]]
            b = [r["vehicles"] for r in it[h * 3600:(h + 1) * 3600]]
            if a and b and np.mean(a) > 0:
                hourly[h] = 100 * (np.mean(b) - np.mean(a)) / np.mean(a)
        rows["closed_loop"][seed] = {"vehicles_diff_percent_by_hour": hourly,
                                     "clear_time_central": ct[-1]["time_seconds"], "clear_time_ir": it[-1]["time_seconds"]}
    for per in PERIODS + ("overall",):
        e = err[per] if per != "overall" else [v for p in PERIODS for v in err[p]]
        s = same[per] if per != "overall" else [v for p in PERIODS for v in same[p]]
        f = failures[per] if per != "overall" else sum(failures.values())
        if e:
            rows["table2"][per] = {"n": len(e), "mean_error": float(np.mean(e)), "le1": 100 * np.mean(np.asarray(e) <= 1),
                                   "le2": 100 * np.mean(np.asarray(e) <= 2), "identical": 100 * np.mean(s),
                                   "p95": float(np.quantile(e, .95)), "failed_local": f}
    with Pool(12, initializer=_init) as pool:
        parts = pool.map(trajectory, jobs, chunksize=4)
    rows["trajectory"] = {key: [float(np.mean([v for part in parts for v in part[key][k]])) for k in range(d.STEPS)]
                          for key in ("green", "count")}
    rows["trajectory"]["solved_local_problems"] = sum(len(part["green"][0]) for part in parts)
    (OUT / "TABLES_closed.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    print("closed done")


def samples_list(total=50):
    """total samples spread over the test days (as equal as possible), evenly spaced updates before DEMAND_END."""
    out, days = [], [s for s in ir.TEST if s in d.TABLE3_DAYS]     # Table III keeps the first six test days
    for i, seed in enumerate(days):
        n = total // len(days) + (i < total % len(days))
        ts = sorted(t for t in ir.history(seed) if t < d.DEMAND_END and (d.ROOT / f"design/ir/bank/t{t}.pkl.gz").exists())
        out += [(seed, ts[int(j)]) for j in np.linspace(0, len(ts) - 1, n)]
    return out


def objective(model, p, x):
    x = np.asarray(x)
    return float(0.5 * x @ (model["H"] @ x) + (model["h0"] + model["hP"] @ p) @ x)


def fixed_cost(model, p, actions):
    S, vals = [], []
    for m, idx in model["output_indices"].items():
        for i, v in zip(idx, actions[m]):
            S.append(i)
            vals.append(v)
    sel = sparse.csr_matrix((np.ones(len(S)), (range(len(S)), S)), shape=(len(S), model["H"].shape[0]))
    r = strict_qp.solve_qp(model["H"], model["h0"] + model["hP"] @ p, sparse.vstack([model["E"], sel]).tocsc(),
                           np.r_[model["e0"] + model["eP"] @ p, vals], model["G"], model["g0"] + model["gP"] @ p)
    return objective(model, p, r["x"]) if r["valid"] else None


def methods(seed, t, timing=False):
    snap = ir.snapshot_of(seed, t)
    model = ir.model_at(_ST, t, snap)
    z = np.load(ir.RUNS / f"mix_seed{seed}/vectors/t{t}.npz")
    p, xc = z["p"], z["x"]
    tick = time.perf_counter()
    central = ir.solve_full(model, p)
    central_seconds = time.perf_counter() - tick
    star = ir.first_decision(model, central["x"])
    prev = {m: snap["signals"][m]["previous_executed_green_seconds"] for m in star}
    b = bank(t)
    ref = ir.solve_full(model, b["pbar"])
    out, secs = {}, {}
    act, fail, local = {}, [], {}
    for m, c in b["controllers"].items():
        tick = time.perf_counter()
        r = local_controller.solve_local(c, p, _REG[d.period(t)]["regions"][m])
        local[m] = time.perf_counter() - tick
        act[m] = r["action"] if r["valid"] else None
        if not r["valid"]:
            fail.append(m)
    out["IR-MPC"] = (act, fail)
    secs["IR-MPC"] = max(local.values())
    for hops in (1, 2):
        hist, _ = osd.run(model, p, b["pbar"], ref["x"], ref["nu"], _ADJ, hops, solves=4)
        for k in (1, 4):
            out[f"OSD-{hops}, K={k}"] = (hist[k - 1]["actions"], hist[k - 1]["failed"])
            secs[f"OSD-{hops}, K={k}"] = sum(max(h["seconds"].values()) for h in hist[:k])
    if timing:
        return {"central": central_seconds, **secs}
    J = objective(model, p, central["x"])
    const = model["objective_constant"]
    rows = {}
    for name, (a, failed) in out.items():
        a = {m: (np.asarray(v) if v is not None else np.asarray(prev[m], float)) for m, v in a.items()}
        e = [float(np.max(np.abs(a[m] - star[m]))) for m in star]
        Ja = fixed_cost(model, p, a)
        rows[name] = {"solved": 1 - len(failed) / len(star), "errors": e,
                      "gap": None if Ja is None else (Ja - J) / (J + const)}
    return {"seed": seed, "t": t, "central_matches_run": float(np.max(np.abs(central["x"] - xc))), "rows": rows,
            "seconds": {"central": central_seconds, **secs}}


def samples():
    with Pool(12, initializer=_init) as pool:
        res = pool.starmap(methods, samples_list(), chunksize=1)
    (OUT / "TABLES_samples.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    print("samples done", len(res))


def timing():
    _init()
    allp = samples_list()
    picks = [allp[int(j)] for j in np.linspace(0, len(allp) - 1, 19)]     # 19 samples, as in main.tex Table I
    methods(*picks[0], timing=True)                                  # warm-up
    res = [methods(s, t, timing=True) for s, t in picks]
    (OUT / "TABLES_timing.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    print("timing done", len(res))


if __name__ == "__main__":
    {"closed": closed, "samples": samples, "timing": timing}[sys.argv[1]]()
