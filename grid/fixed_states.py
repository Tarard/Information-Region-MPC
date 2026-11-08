"""Traffic states of a size without closed-loop centralized MPC (user, 2026-10-31, 500 intersections): the plant runs
each day (seeds 1-4 and 101-102) under the fixed-time plan of the design split (balanced greens of the peak demand,
`design_greens` in net.json, as central_time_test.py) up to the end of the peak, and the parameter vector p of the
centralized QP is stored at every update clock of the peak period. ir_trial.py reads them like the states of a
centralized day; the stored x are placeholders (NaN): ir_trial.py solves the current QP at these states itself.
Usage: fixed_states.py N   -> NS_ROOT/work/N{N}/runs/central_{seed}/states.npz and SUMMARY.json
"""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
import ns_design as d  # noqa: E402

d.enforce_threads()
import numpy as np  # noqa: E402

import grid_net  # noqa: E402
import ns_common as nc  # noqa: E402
from plant import Plant  # noqa: E402

SEEDS = (1, 2, 3, 4, 101, 102)


def one_day(net, n, seed):
    signals, tls = net["signals"], sorted(net["signals"])
    greens = {a: nc.integer_green(nc.legal_green(net["cycle_choice"][a]["design_greens"], signals[a]), signals[a])
              for a in tls}
    plant = Plant(net, seed)
    times, P = [], []
    while plant.t < d.PEAK[1]:
        if plant.t >= d.PEAK[0]:
            times.append(plant.t)
            P.append(nc.build(net, plant.observe(greens), plant.t)["p"].copy())
        plant.run(nc.phase_plan(signals, tls, greens, plant.t))
    out = d.size_dir(n) / f"runs/central_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    P = np.asarray(P)
    np.savez_compressed(out / "states.npz", t=np.asarray(times), p=P, x=np.full((len(P), 1), np.nan))
    summary = {"n": n, "seed": seed, "controller": "fixed_time_design_split", "states_only": True,
               "note": "plant under the fixed-time design split up to the end of the peak; no closed-loop MPC",
               **plant.summary()}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"N={n} seed {seed}: {len(times)} states, in network {plant.in_system():.0f}", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1])
    path = d.size_dir(n) / "net.json"
    net = grid_net.load(n) if path.exists() else grid_net.prepare(n)
    for seed in SEEDS:
        one_day(net, n, seed)
