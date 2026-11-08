from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import random
import heapq
import tempfile
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "experiments" / "lyg-d100-dynamic-v6-internal-congestion-scale1p2-1p5-1p0"
LYG = ROOT / "network" / "mp_two_networks" / "networks" / "lyg"
NETWORK = ROOT / "network" / "lyg_drmp" / "lyg_safe.net.xml"
SEGMENTS = (
    (0.0, 2400.0, 1.2, 1.2, "medium"),
    (2400.0, 4800.0, 1.5, 1.5, "high"),
    (4800.0, 7200.0, 1.0, 1.0, "low"),
)

# Background-route policy.  The first signalized boundary ring keeps modest
# turning demand, while interior intersections receive a larger left-turn weight
# so congestion is encouraged to form inside the network rather than at the
# perimeter.  A tiny U-turn weight is retained only as a topology-safe fallback.
# The four semantic weights sum to one; SUMO uppercase L/R direction codes are
# normalized to the same left/right classes before weighting.
INTERNAL_DIRECTION_WEIGHT = {"s": 0.74, "r": 0.05, "l": 0.20, "t": 0.01}
PERIPHERAL_DIRECTION_WEIGHT = {"s": 0.84, "r": 0.05, "l": 0.10, "t": 0.01}


def _normalized_direction(direction: str) -> str:
    value = str(direction)
    if value in {"l", "L"}:
        return "l"
    if value in {"r", "R"}:
        return "r"
    if value == "t":
        return "t"
    return "s" if value == "s" else value


def _seed(seed: int, flow_id: str, label: str) -> int:
    payload = f"{seed}|{flow_id}|{label}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _load_design() -> tuple[list[ET.Element], list[dict[str, str]], dict[tuple[str, str], str]]:
    common: list[ET.Element] = []
    flows: list[dict[str, str]] = []
    routes: dict[tuple[str, str], str] = {}
    for name in ("flow.rou.xml", "artery.rou.xml"):
        root = ET.parse(LYG / name).getroot()
        for element in root:
            if element.tag in {"vType", "route"}:
                common.append(ET.fromstring(ET.tostring(element)))
                if element.tag == "route":
                    edges = element.attrib["edges"].split()
                    routes[(edges[0], edges[-1])] = element.attrib["id"]
            elif element.tag == "flow":
                flows.append(dict(element.attrib))
    if len(flows) != 36:
        raise ValueError(f"expected 36 original LYG flows, found {len(flows)}")
    return common, flows, routes



def _network_graph() -> tuple[dict[str, list[tuple[str, float]]], dict[str, list[tuple[str, str]]], set[str], set[str]]:
    root = ET.parse(NETWORK).getroot()
    edge_elements = {
        edge.attrib["id"]: edge
        for edge in root.findall("edge")
        if not edge.attrib["id"].startswith(":") and edge.attrib.get("function") != "internal"
    }
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    incoming: dict[str, set[str]] = defaultdict(set)
    outgoing: dict[str, set[str]] = defaultdict(set)
    lengths = {
        edge_id: sum(float(lane.attrib["length"]) for lane in edge.findall("lane"))
        / max(len(edge.findall("lane")), 1)
        for edge_id, edge in edge_elements.items()
    }
    for connection in root.findall("connection"):
        source = connection.attrib.get("from")
        target = connection.attrib.get("to")
        if source not in edge_elements or target not in edge_elements:
            continue
        outgoing[source].add(target)
        incoming[target].add(source)
    directions: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for connection in root.findall("connection"):
        source = connection.attrib.get("from")
        target = connection.attrib.get("to")
        if source in edge_elements and target in edge_elements:
            directions[source].append((target, connection.attrib.get("dir", "s")))
    for source, targets in outgoing.items():
        adjacency[source] = sorted((target, lengths[target]) for target in targets)
    entries = {edge_id for edge_id in edge_elements if not incoming[edge_id]}
    exits = {edge_id for edge_id in edge_elements if not outgoing[edge_id]}
    return dict(adjacency), dict(directions), entries, exits


def _peripheral_signal_approach_edges(
    adjacency: dict[str, list[tuple[str, float]]], entries: set[str]
) -> tuple[set[str], set[str]]:
    """Return the first signalized boundary ring and all approaches into it.

    A junction belongs to the peripheral ring when it is the nearest
    traffic-light junction reachable from at least one true network-entry edge.
    The routing bias is then applied to every approach entering those junctions,
    so the rule is topology-derived rather than tied to hard-coded junction IDs.
    """
    root = ET.parse(NETWORK).getroot()
    edge_to_junction = {
        edge.attrib["id"]: edge.attrib.get("to", "")
        for edge in root.findall("edge")
        if not edge.attrib["id"].startswith(":") and edge.attrib.get("function") != "internal"
    }
    junction_type = {
        junction.attrib["id"]: junction.attrib.get("type", "")
        for junction in root.findall("junction")
    }
    peripheral_tls: set[str] = set()
    for source in entries:
        queue: list[tuple[str, int]] = [(source, 0)]
        seen = {source}
        nearest_depth: int | None = None
        cursor = 0
        while cursor < len(queue):
            edge_id, depth = queue[cursor]
            cursor += 1
            junction_id = edge_to_junction.get(edge_id, "")
            if junction_type.get(junction_id, "").startswith("traffic_light"):
                if nearest_depth is None:
                    nearest_depth = depth
                if depth == nearest_depth:
                    peripheral_tls.add(junction_id)
                continue
            if nearest_depth is not None and depth >= nearest_depth:
                continue
            for target, _ in adjacency.get(edge_id, ()):
                if target not in seen:
                    seen.add(target)
                    queue.append((target, depth + 1))
    approaches = {
        edge_id for edge_id, junction_id in edge_to_junction.items()
        if junction_id in peripheral_tls
    }
    return approaches, peripheral_tls


def _shortest_edge_path(
    adjacency: dict[str, list[tuple[str, float]]], source: str, target: str
) -> tuple[str, ...]:
    if source == target:
        return (source,)
    queue: list[tuple[float, str]] = [(0.0, source)]
    distance = {source: 0.0}
    predecessor: dict[str, str] = {}
    while queue:
        cost, edge = heapq.heappop(queue)
        if cost != distance.get(edge):
            continue
        if edge == target:
            break
        for next_edge, weight in adjacency.get(edge, ()):
            next_cost = cost + weight
            if next_cost < distance.get(next_edge, float("inf")):
                distance[next_edge] = next_cost
                predecessor[next_edge] = edge
                heapq.heappush(queue, (next_cost, next_edge))
    if target not in distance:
        raise ValueError(f"no passenger edge path from {source} to {target}")
    path = [target]
    while path[-1] != source:
        path.append(predecessor[path[-1]])
    path.reverse()
    return tuple(path)


def _random_exit_path(
    source: str,
    adjacency: dict[str, list[tuple[str, float]]],
    directions: dict[str, list[tuple[str, str]]],
    exits: set[str],
    rng: random.Random,
    peripheral_approaches: set[str],
) -> tuple[str, ...]:
    current = source
    path = [source]
    visits = Counter([source])
    for _ in range(250):
        if current in exits:
            return tuple(path)
        direction_weight = (
            PERIPHERAL_DIRECTION_WEIGHT
            if current in peripheral_approaches
            else INTERNAL_DIRECTION_WEIGHT
        )
        options_by_edge: dict[str, float] = {}
        for target, direction in directions.get(current, ()):
            penalty = 0.05 ** visits[target] if visits[target] else 1.0
            weight = direction_weight.get(_normalized_direction(direction), 0.01) * penalty
            options_by_edge[target] = max(options_by_edge.get(target, 0.0), weight)
        if not options_by_edge:
            break
        targets = list(options_by_edge)
        weights = [options_by_edge[target] for target in targets]
        current = rng.choices(targets, weights=weights, k=1)[0]
        path.append(current)
        visits[current] += 1
        if visits[current] > 2:
            break

    # Deterministic safety fallback: connect the current edge to the nearest
    # reachable network exit.  This guarantees every generated vehicle has a
    # physical network-exit destination.
    candidates = []
    for exit_edge in exits:
        try:
            tail = _shortest_edge_path(adjacency, current, exit_edge)
        except ValueError:
            continue
        cost = sum(
            next(weight for target, weight in adjacency.get(edge, ()) if target == nxt)
            for edge, nxt in zip(tail, tail[1:])
        )
        candidates.append((cost, exit_edge, tail))
    if not candidates:
        raise ValueError(f"background route from {source} cannot reach a network exit")
    _, _, tail = min(candidates)
    return tuple(path + list(tail[1:]))

def _bytes(seed: int) -> tuple[bytes, bytes]:
    common, flows, routes = _load_design()
    adjacency, directions, network_entries, network_exits = _network_graph()
    peripheral_approaches, peripheral_tls = _peripheral_signal_approach_edges(
        adjacency, network_entries
    )
    vehicles: list[tuple[float, str, dict[str, str], tuple[str, ...] | None, str]] = []
    segment_counts: Counter[str] = Counter()
    flow_counts: Counter[str] = Counter()
    for flow in flows:
        rate = float(flow["vehsPerHour"])
        route_id = routes.get((flow["from"], flow["to"]))
        if flow["from"] not in network_entries:
            raise ValueError(f"flow origin is not a network entry: {flow['id']} -> {flow['from']}")
        if flow["from"] != flow["to"] and flow["to"] not in network_exits:
            raise ValueError(f"fixed OD destination is not a network exit: {flow['id']} -> {flow['to']}")
        fixed_edges = (
            None
            if route_id is not None or flow["from"] == flow["to"]
            else _shortest_edge_path(adjacency, flow["from"], flow["to"])
        )
        for begin, end, _, scale, label in SEGMENTS:
            rng = random.Random(_seed(seed, flow["id"], label))
            time_s = begin
            while True:
                time_s += rng.expovariate(rate * scale / 3600.0)
                if time_s >= end:
                    break
                index = flow_counts[flow["id"]]
                vehicle_id = f"{flow['id']}_{index:06d}"
                attrs = {
                    "id": vehicle_id,
                    "type": flow["type"],
                    "depart": f"{time_s:.6f}",
                    "departLane": flow.get("departLane", "best"),
                }
                if route_id is not None:
                    attrs["route"] = route_id
                    explicit_edges = None
                elif fixed_edges is not None:
                    explicit_edges = fixed_edges
                else:
                    route_rng = random.Random(_seed(seed, vehicle_id, "route"))
                    explicit_edges = _random_exit_path(
                        flow["from"], adjacency, directions, network_exits, route_rng,
                        peripheral_approaches,
                    )
                vehicles.append((time_s, vehicle_id, attrs, explicit_edges, flow["from"]))
                segment_counts[label] += 1
                flow_counts[flow["id"]] += 1

    root = ET.Element("routes")
    for element in common:
        root.append(element)
    for _, _, attrs, explicit_edges, source_edge in sorted(vehicles):
        vehicle = ET.SubElement(root, "vehicle", attrs)
        if explicit_edges is not None:
            ET.SubElement(vehicle, "route", {"edges": " ".join(explicit_edges)})
        elif "route" not in attrs:
            raise RuntimeError(f"vehicle has no complete OD route: {attrs['id']}")
    ET.indent(root, space="    ")
    output = io.BytesIO()
    ET.ElementTree(root).write(output, encoding="utf-8", xml_declaration=True)
    route_bytes = output.getvalue() + b"\n"
    manifest = {
        "schema": "lyg-d100-dynamic-poisson/v2-clean-od",
        "arrival_seed": seed,
        "sumo_seed": 0,
        "horizon_s": 7200,
        "flow_count": len(flows),
        "vehicle_count": len(vehicles),
        "network_entry_edge_count": len(network_entries),
        "network_exit_edge_count": len(network_exits),
        "fixed_od_flow_count": sum(flow["from"] != flow["to"] for flow in flows),
        "background_random_exit_flow_count": sum(flow["from"] == flow["to"] for flow in flows),
        "background_route_policy": "peripheral first-ring weights through/left/right/U-turn=0.84/0.10/0.05/0.01; interior=0.74/0.20/0.05/0.01; SUMO L/R normalized",
        "peripheral_direction_weights": dict(PERIPHERAL_DIRECTION_WEIGHT),
        "interior_direction_weights": dict(INTERNAL_DIRECTION_WEIGHT),
        "peripheral_signal_junction_count": len(peripheral_tls),
        "peripheral_signal_junction_ids": sorted(peripheral_tls),
        "segments": [
            {
                "begin_s": int(begin),
                "end_s": int(end),
                "set_scale": set_scale,
                "effective_scale": scale,
                "label": label,
                "vehicle_count": segment_counts[label],
            }
            for begin, end, set_scale, scale, label in SEGMENTS
        ],
        "flow_vehicle_counts": dict(sorted(flow_counts.items())),
        "route_sha256": hashlib.sha256(route_bytes).hexdigest(),
        "source_sha256": {
            name: hashlib.sha256((LYG / name).read_bytes()).hexdigest()
            for name in ("flow.rou.xml", "artery.rou.xml")
        },
    }
    return route_bytes, (json.dumps(manifest, indent=2) + "\n").encode()


def _freeze(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise FileExistsError(f"refusing to overwrite different content: {path}")
        return
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare(seed: int) -> tuple[Path, Path]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    route_bytes, manifest_bytes = _bytes(seed)
    route = EXPERIMENT / "routes" / f"arrival_seed{seed:03d}.rou.xml"
    manifest = EXPERIMENT / "routes" / f"arrival_seed{seed:03d}.manifest.json"
    _freeze(route, route_bytes)
    _freeze(manifest, manifest_bytes)
    return route, manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, action="append", required=True)
    args = parser.parse_args()
    for seed in args.seed:
        route, manifest = prepare(seed)
        print(json.dumps({"seed": seed, "route": str(route), "manifest": str(manifest)}))


if __name__ == "__main__":
    main()
