"""One closed-loop day on a grid: plant (plant.py) + controller, an update every T = 60 s; at each update only the
intersections whose cycle starts then decide (60-s ones every update, 120-s ones every other), the others keep
running their current cycle (v2 user spec 2026-10-30).

Controllers: central (centralized MPC), ir (IR-MPC, needs the offline bank), osd (partitioned OSD with
OSD_CLOSED_LOOP_REGION intersections per region, OSD_CLOSED_LOOP_K iterations, needs the offline references).
At every update the plant state is observed, the controller decides the greens of the step, and the step is
executed in the plant (T / C_m equal cycles, greens then 3-s yellow). A failed solve keeps the previous greens of
the affected intersections and is counted. Optionally the centralized QP is also solved at the same state, only to
measure the first-decision difference (--compare; used up to 200 intersections).
Outputs in <out>: decisions.jsonl, trace.jsonl (every update), states.npz (p and centralized x at every update, central
runs only), SUMMARY.json.
Usage: ns_scenario.py N SEED CONTROLLER OUT [--compare]
"""

import ns_design as d

d.enforce_threads()

import json  # noqa: E402
from pathlib import Path  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import grid_net  # noqa: E402
import ns_common as nc  # noqa: E402
import ns_osd  # noqa: E402
from plant import Plant  # noqa: E402
import strict_qp  # noqa: E402


def load_offline(n):
    import pickle
    import gzip
    folder = d.size_dir(n) / "offline"
    with gzip.open(folder / "references.pkl.gz", "rb") as h:
        refs = pickle.load(h)
    return folder, refs


def run_day(n, seed, controller, out, compare=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    net = grid_net.load(n)
    signals = net["signals"]
    tls = sorted(signals)
    plant = Plant(net, seed)
    previous = {a: list(signals[a]["reference_green_seconds_at_t0"]) for a in tls}
    ir = osd_regions = refs = None
    if controller in ("ir", "osd"):
        folder, refs = load_offline(n)
    if controller == "ir":
        from ns_ir import IRController
        ir = IRController(folder)
    if controller == "osd":
        parts = json.loads((folder / "PARTITIONS.json").read_text(encoding="utf-8"))[str(d.OSD_CLOSED_LOOP_REGION)]
    rows, times, P, X, failures = [], [], [], [], 0
    warm = None
    started = time.perf_counter()
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as dec, (out / "trace.jsonl").open("w", encoding="utf-8") as tr:
        while plant.t < d.MAX_TIME:
            t = plant.t
            if t >= d.DEMAND_END and plant.in_system() < 0.5:
                break
            if d.DAY_STOP and t >= d.DAY_STOP:                  # states only up to this clock (CHANGES.md)
                break
            snap = plant.observe(previous)
            tick = time.perf_counter()
            model = nc.build(net, snap, t)
            build_s = time.perf_counter() - tick
            p = model["p"]
            rec = {"time_seconds": t, "period": d.period(t), "build_seconds": build_s}
            central = None
            if controller == "central" or compare:
                central = strict_qp.solve_matrices(model["matrices"], warm)
                rec["central_seconds"] = central["seconds"]
                rec["central_valid"] = central["valid"]
                if central["valid"]:
                    warm = central["x"].copy(), central["y"].copy()
                    rec["central_first"] = {m: central["x"][idx].tolist() for m, idx in model["output_indices"].items()}
            deciding = sorted(model["output_indices"])
            rec["deciding"] = len(deciding)
            if controller == "central":
                if central["valid"]:
                    action = {m: central["x"][idx] for m, idx in model["output_indices"].items()}
                    P.append(p.copy())
                    X.append(central["x"].copy())
                    times.append(t)
                else:
                    failures += len(deciding)
                    action = {m: previous[m] for m in deciding}
                rec["controller_seconds"] = central["seconds"]
            elif controller == "ir":
                res = ir.decide(nc.ref_key(refs, t), p, {m: previous[m] for m in deciding})
                action, rec["failed"] = res["action"], res["failed"]
                failures += len(res["failed"])
                rec["controller_seconds"] = res["max_seconds"]
                rec["local_sum_seconds"] = res["sum_seconds"]
            else:
                ref = refs[nc.ref_key(refs, t)]
                # region blocks are cut from this clock's QP (the red-delay weights change with the clock);
                # cutting them is setup, not counted as controller time
                osd_regions = ns_osd.compile_regions(model, parts, net["adjacency"])
                hist = ns_osd.run(model, osd_regions, p, ref["pbar"], ref["xref"], ref["nu_ref"], d.OSD_CLOSED_LOOP_K)
                last = hist[-1]
                action = {m: (last["actions"][m] if last["actions"].get(m) is not None else previous[m]) for m in deciding}
                rec["failed"] = sorted(set(last["failed"]))
                failures += len(rec["failed"])
                rec["controller_seconds"] = sum(h["max_seconds"] for h in hist)
            executed = {m: list(previous[m]) for m in tls if m not in action}      # running cycles continue
            for m in action:
                try:
                    executed[m] = nc.integer_green(nc.legal_green(action[m], signals[m]), signals[m])
                except ValueError:
                    executed[m] = nc.integer_green(nc.legal_green(previous[m], signals[m]), signals[m])
                    failures += 1
            rec["executed"] = executed
            if central is not None and central["valid"] and controller != "central":
                rec["error_vs_central"] = {m: float(np.max(np.abs(np.asarray(action[m]) - central["x"][idx])))
                                           for m, idx in model["output_indices"].items()}
            dec.write(json.dumps(rec) + "\n")
            dec.flush()
            plant.run(nc.phase_plan(signals, tls, executed, t))       # seconds x intersections (sorted order)
            previous = {m: list(executed[m]) for m in tls}
            s = plant.summary()
            tr.write(json.dumps({"time_seconds": plant.t, "in_system": s["in_system"],
                                 "waiting": float(plant.q.sum())}) + "\n")
    s = plant.summary()
    cleared = plant.t >= d.DEMAND_END and plant.in_system() < 0.5
    summary = {"n": n, "seed": seed, "controller": controller, "cleared": bool(cleared), **s,
               "updates": len(open(out / "decisions.jsonl", encoding="utf-8").read().splitlines()),
               "failures": failures, "runtime_seconds": time.perf_counter() - started,
               "stopped_at": d.DAY_STOP or None}
    if times:
        np.savez_compressed(out / "states.npz", t=np.asarray(times), p=np.asarray(P), x=np.asarray(X))
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("n", "seed", "controller", "cleared", "mean_delay_seconds_per_vehicle",
                                               "completed", "generated", "updates", "failures")}), flush=True)
    return summary


if __name__ == "__main__":
    run_day(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4], "--compare" in sys.argv)
