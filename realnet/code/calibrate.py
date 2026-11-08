"""Design inputs from the history days (cc_design.HISTORY_DAYS; no test day is read).

Turn shares : for every road u with declared connections, count consecutive road pairs (u, v) over all vehicles
              of the history route files (routes are fixed, no rerouting, so this equals junction turning counts).
              Share_uv = count_uv / sum_v count_uv when sum_v count_uv >= MIN_COUNT; otherwise the v3 shares stay.
Arrival rate: vehicles entering each road per second, history mean. Entry time = departure + free-flow time of the
              roads before it on the route (lane length / lane speed). Centered 900-s window, rows every 60 s.
Writes inputs/network_model.json (turn_priors), design/service_groups.json (declared_turn_share),
design/arrival_table.json and design/TURN_CALIBRATION.json. Usage: calibrate.py
"""

from collections import Counter, defaultdict
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

import cc_design as d

MIN_COUNT, WINDOW, ROW_STEP = 30, 900, 60


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def main():
    net = read(d.V3 / "inputs/network_model.json")
    groups = read(d.V3 / "design/service_groups.json")
    roads = net["roads"]
    edges = sorted(roads)
    index = {e: i for i, e in enumerate(edges)}
    free = {e: float(roads[e]["lanes"][0]["length"]) / float(roads[e]["lanes"][0]["speed"]) for e in edges}
    counts = defaultdict(Counter)
    bins = int(d.MAX_TIME // 60) + 1
    entered = np.zeros((len(edges), bins))
    vehicles = {}
    for seed in d.HISTORY_DAYS:
        n = 0
        for v in ET.parse(d.route_path(seed)).getroot().iter("vehicle"):
            route = v.find("route").get("edges").split()
            t = float(v.get("depart"))
            for k, e in enumerate(route):
                if k:
                    counts[route[k - 1]][e] += 1
                entered[index[e], min(int(t // 60), bins - 1)] += 1
                t += free[e]
            n += 1
        vehicles[seed] = n

    targets = defaultdict(set)
    for c in net["connections"]:
        targets[c["from"]].add(c["to"])
    changed, kept, report = 0, [], {}
    for road, prior in net["turn_priors"].items():
        assert set(prior["shares"]) == targets[road]
        total = sum(counts[road][t] for t in targets[road])
        extra = sum(v for t, v in counts[road].items() if t not in targets[road])
        assert extra == 0, (road, extra)
        if total >= MIN_COUNT:
            new = {t: counts[road][t] / total for t in sorted(targets[road])}
            report[road] = {"count": total, "old": prior["shares"], "new": new}
            prior.update(shares=new, counts={t: counts[road][t] for t in sorted(targets[road])},
                         source=f"route turning counts, history days {d.HISTORY_DAYS}")
            changed += 1
        else:
            kept.append({"road": road, "count": total})
    net["turn_share_calibration"] = {"rule": __doc__, "history_vehicles": vehicles, "changed_roads": changed,
                                     "kept_roads": kept}
    (d.ROOT / "inputs").mkdir(exist_ok=True)
    (d.ROOT / "design").mkdir(exist_ok=True)
    d.NETWORK_MODEL.write_text(json.dumps(net, indent=1), encoding="utf-8")
    for sig in groups["signals"].values():
        for g in sig["movement_capacity_groups"]:
            g["declared_turn_share"] = net["turn_priors"][g["incoming_edge"]]["shares"][g["destination_edge"]]
    groups["source"] = {"network_model": str(d.NETWORK_MODEL), "base_groups": str(d.V3 / "design/service_groups.json")}
    d.SERVICE_GROUPS.write_text(json.dumps(groups, indent=2), encoding="utf-8")

    rows, half, days = [], WINDOW // 2, len(d.HISTORY_DAYS)
    for clock in range(0, int(d.MAX_TIME) + 1, ROW_STEP):
        lo, hi = max(0, (clock - half) // 60), min(bins, (clock + half) // 60)
        seconds = (hi - lo) * 60 * days
        rows.append({"clock_seconds": clock,
                     "arrival_rate_veh_per_second": (entered[:, lo:hi].sum(axis=1) / seconds).tolist()})
    table = {"history_route_seeds": list(d.HISTORY_DAYS), "measurement": "route entry times at free-flow speed",
             "measurement_period_seconds": 60, "window_seconds": WINDOW, "edge_ids": edges, "rows": rows}
    d.ARRIVAL_TABLE.write_text(json.dumps(table), encoding="utf-8")
    (d.ROOT / "design/TURN_CALIBRATION.json").write_text(
        json.dumps({"changed": changed, "kept": kept, "roads": report}, indent=1), encoding="utf-8")
    print("history vehicles", sum(vehicles.values()), "roads recalibrated", changed, "kept", len(kept))


if __name__ == "__main__":
    main()
