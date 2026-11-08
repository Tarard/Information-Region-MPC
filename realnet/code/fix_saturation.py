"""Saturation flows of the signalized lane connections.

The full_v2 inputs used, for 454 of 472 signalized connections, the 85% quantile of observed movement flows (median
0.21 veh/s per lane), which is a traffic volume, not a saturation flow. Rule here: a connection with a SUMO-measured
protected queued discharge rate keeps it (18 connections, 0.47-0.61 veh/s); every other signalized connection gets
the standard urban value 1800 veh/h per lane = 0.5 veh/s, the value already used on the nonsignalized connections.
Then design/service_groups.json is rebuilt from the connections: lane rate of a movement group = largest rate of the
connections from that lane to the group's destination road, group rate = sum over its lanes.
Usage: fix_saturation.py check | write
"""

import json
from pathlib import Path
import sys

import cc_design as d

MEASURED = "native_protected_queued_green_stop_lane_rate"
STANDARD = 0.5


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def group_rates(network, groups):
    rate = {}
    for c in network["connections"]:
        if c.get("tl"):
            key = (c["from"], c["to"], str(c["fromLane"]))
            rate[key] = max(rate.get(key, 0.0), float(c["service_veh_per_second"]))
    out = {}
    for tls, spec in groups["signals"].items():
        for i, g in enumerate(spec["movement_capacity_groups"]):
            lanes = {lane: rate[g["incoming_edge"], g["destination_edge"], lane]
                     for lane in g["lane_saturation_rates_veh_per_second"]}
            out[tls, i] = lanes
    return out


def main(mode):
    network, groups = read(d.NETWORK_MODEL), read(d.SERVICE_GROUPS)
    before = group_rates(network, groups)
    stored = {(tls, i): {k: float(v) for k, v in g["lane_saturation_rates_veh_per_second"].items()}
              for tls, spec in groups["signals"].items() for i, g in enumerate(spec["movement_capacity_groups"])}
    differ = [k for k in stored if any(abs(stored[k][lane] - before[k][lane]) > 1e-12 for lane in stored[k])]
    print("groups", len(stored), "whose stored lane rates differ from the connections:", len(differ))
    if mode != "write":
        return
    changed = 0
    for c in network["connections"]:
        if c.get("tl") and c.get("service_source") != MEASURED:
            c["service_veh_per_second"] = STANDARD
            c["service_source"] = "standard_urban_saturation_1800_veh_per_hour_lane"
            changed += 1
    after = group_rates(network, groups)
    for tls, spec in groups["signals"].items():
        for i, g in enumerate(spec["movement_capacity_groups"]):
            g["lane_saturation_rates_veh_per_second"] = after[tls, i]
            g["saturation_rate_veh_per_second"] = sum(after[tls, i].values())
    network["saturation_rule"] = __doc__
    groups["saturation_rule"] = __doc__
    Path(d.NETWORK_MODEL).write_text(json.dumps(network, indent=1), encoding="utf-8")
    Path(d.SERVICE_GROUPS).write_text(json.dumps(groups, indent=1), encoding="utf-8")
    print("connections set to", STANDARD, ":", changed)


if __name__ == "__main__":
    main(sys.argv[1])
