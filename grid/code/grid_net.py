"""Grid networks and demand of the numerical scaling experiment v2 (PLAN_numeric_scale_v2.md sections 3-4), in the
model format of cc_model.build.

Network (seed NET_SEED + N, so a size always gives the same network):
- rows x columns of signalized intersections J{r}_{c}; odd rows and columns (index 1, 3, 5, ...) are arterials.
  Every row spacing and every column spacing is drawn from SPACINGS; a link has the length of its spacing.
- DELETE_FRACTION of the links on minor roads (both directions) are deleted, keeping the grid connected and at
  least 3 approaches at every intersection; an intersection left with 3 approaches is a T intersection.
- approach lanes (lane 0 is the rightmost): arterial 4 (0 straight+right, 1-2 straight, 3 left), minor 2
  (0 straight+right, 1 left+straight); at a T the lanes of a missing movement serve the others. 0.5 veh/s per lane;
  storage lanes x (floor(length / 7.5) + 1) is only the cost weight, links have no storage limit (user 2026-10-30).
  Speed 16.67 m/s on arterials, 11.11 m/s on minor roads.
- fringe: one entry and one exit link (BOUNDARY_LENGTH, lanes of the road) on every open side; every intersection
  has a local source src_J (1 lane, served in stage 0) and a local sink snk_J (reached from the right lane).
- stages: arterial x arterial 4 (NS straight+right, NS left, EW straight+right, EW left); arterial x minor 4 with
  probability FOUR_STAGE_SHARE_ARTERIAL_MINOR, else 3 (arterial straight+right, arterial left, minor all);
  minor x minor 2 (NS all, EW all); a T on an arterial 3, else 2.
Demand (peak, before scaling): GROUP_SHARES of the trips: through (fringe entry -> fringe exit, arterial ends weigh
2, minor 1, not at the same intersection), central (local sources and fringe entries -> local sinks of the centre
zone, the CENTRE_SHARE intersections nearest the geometric centre), local (local source -> local sinks 2-6
intersections away). Routes: the ROUTES fastest routes at free speed (LEFT_PENALTY per left turn), logit with
LOGIT_THETA. The routes give the design turning shares of every approach and the entry rates of every group.
Capacity groups (as in cc_model.build): every movement (saturation 0.5 veh/s times its lanes) and, per approach
and stage, all movements of the approach served in the stage (0.5 veh/s times the lanes they use together).
Demand level: Y_m = sum over stages of the largest flow ratio of a capacity group served in the stage, i.e. the
smallest total green share that serves every group; Y_m is proportional to the demand, so lambda is
set once so that the median of Y_m at the peak equals Y_MEDIAN_TARGET (times DEMAND_FACTOR). Then every
intersection takes the cycle (60 or 120 s) with the lower flow-weighted Webster delay (LYG rule, balanced split
under the green bounds), once.
Historical arrival rates of the red-delay term: design link flows times the day curve r(t), one row per 60 s.
Usage: grid_net.py N  (writes work/N{N}/net.json)
"""

import itertools
import json
import math
from multiprocessing import Pool
import sys

import networkx as nx
import numpy as np
from scipy.optimize import linprog

import ns_design as d

SIDES = ("N", "E", "S", "W")
OPP = {"N": "S", "S": "N", "E": "W", "W": "E"}
STEP = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
_G = {}


def jid(r, c):
    return f"J{r}_{c}"


def geometry(rows, cols, rng):
    """Spacings, arterials, kept neighbor links (after the deletions)."""
    gap_r, gap_c = rng.choice(d.SPACINGS, rows - 1), rng.choice(d.SPACINGS, cols - 1)
    ypos, xpos = np.r_[0, np.cumsum(gap_r)], np.r_[0, np.cumsum(gap_c)]
    major_row = {r: r % 2 == 1 for r in range(rows)}
    major_col = {c: c % 2 == 1 for c in range(cols)}
    segments = {}                                    # frozenset(a, b) -> (length, arterial)
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                segments[frozenset((jid(r, c), jid(r, c + 1)))] = (float(gap_c[c]), major_row[r])
            if r + 1 < rows:
                segments[frozenset((jid(r, c), jid(r + 1, c)))] = (float(gap_r[r]), major_col[c])
    nodes = [jid(r, c) for r in range(rows) for c in range(cols)]
    coords = {jid(r, c): (r, c) for r in range(rows) for c in range(cols)}

    def approaches(a, kept):
        r, c = coords[a]
        count = 0
        for dr, dc in STEP.values():
            rr, cc = r + dr, c + dc
            if not (0 <= rr < rows and 0 <= cc < cols) or frozenset((a, jid(rr, cc))) in kept:
                count += 1
        return count

    kept = set(segments)
    minor = sorted((s for s, (_, art) in segments.items() if not art), key=lambda s: sorted(s))
    target = round(d.DELETE_FRACTION * len(minor))
    deleted = []
    for i in rng.permutation(len(minor)):
        if len(deleted) == target:
            break
        s = minor[i]
        trial = kept - {s}
        if any(approaches(a, trial) < 3 for a in s):
            continue
        g = nx.Graph()
        g.add_nodes_from(nodes)
        g.add_edges_from(tuple(x) for x in trial)
        if not nx.is_connected(g):
            continue
        kept, deleted = trial, deleted + [s]
    return {"rows": rows, "cols": cols, "x": xpos.tolist(), "y": ypos.tolist(), "major_row": major_row,
            "major_col": major_col, "segments": segments, "kept": kept, "deleted": deleted, "coords": coords,
            "nodes": nodes, "minor_links": len(minor)}


def lanes_for(turns, count):
    """Lanes of each turn of an approach with `count` lanes (lane 0 rightmost); 'k' (local sink) uses lane 0."""
    has = set(turns)
    if count == d.ARTERIAL_LANES:
        if {"s", "l"} <= has:
            out = {"r": [0], "s": [0, 1, 2], "l": [3]}
        elif "s" in has:
            out = {"r": [0], "s": [0, 1, 2, 3]}
        else:
            out = {"r": [0, 1], "l": [2, 3]}
    else:
        if {"s", "l"} <= has:
            out = {"r": [0], "s": [0, 1], "l": [1]}
        elif "s" in has:
            out = {"r": [0], "s": [0, 1]}
        else:
            out = {"r": [0], "l": [1]}
    out["k"] = [0]
    return {t: out[t] for t in turns}


def layout(geo, rng):
    rows, cols, coords, nodes = geo["rows"], geo["cols"], geo["coords"], geo["nodes"]

    def neighbor(a, side):
        r, c = coords[a]
        dr, dc = STEP[side]
        rr, cc = r + dr, c + dc
        if not (0 <= rr < rows and 0 <= cc < cols):
            return "fringe"
        b = jid(rr, cc)
        return b if frozenset((a, b)) in geo["kept"] else None

    def arterial_side(a, side):                     # class of the road that leaves a on this side
        r, c = coords[a]
        return geo["major_col"][c] if side in ("N", "S") else geo["major_row"][r]

    roads, owner, incoming, outgoing, entries, exits, cls = {}, {}, {}, {}, [], [], {}

    def road(z, length, arterial, lanes=None):
        lanes = lanes or (d.ARTERIAL_LANES if arterial else d.MINOR_LANES)
        speed = d.ARTERIAL_SPEED if arterial else d.MINOR_SPEED
        roads[z] = {"lanes": [{"length": float(length), "speed": speed} for _ in range(lanes)]}
        cls[z] = "arterial" if arterial else "minor"

    for a in nodes:
        for side in SIDES:
            nb = neighbor(a, side)
            if nb is None:
                continue
            art = arterial_side(a, side)
            if nb == "fringe":
                zin, zout = f"in_{a}_{side}", f"out_{a}_{side}"
                road(zin, d.BOUNDARY_LENGTH, art)
                road(zout, d.BOUNDARY_LENGTH, art)
                entries.append(zin)
                exits.append(zout)
                owner[zout] = a
            else:
                zin, zout = f"{nb}>{a}", f"{a}>{nb}"
                length = geo["segments"][frozenset((a, nb))][0]
                road(zin, length, art)
            incoming[a, side], outgoing[a, side] = zin, zout
            owner[zin] = a
    for a in nodes:
        src, snk = f"src_{a}", f"snk_{a}"
        road(src, d.BOUNDARY_LENGTH, False, 1)
        road(snk, d.BOUNDARY_LENGTH, False, 1)
        owner[src] = owner[snk] = a
        entries.append(src)
        exits.append(snk)
    sides_of = {a: [s for s in SIDES if (a, s) in incoming] for a in nodes}
    stages, kind, major_dir = {}, {}, {}
    for a in nodes:
        r, c = coords[a]
        ar, ac = geo["major_row"][r], geo["major_col"][c]
        tee = len(sides_of[a]) == 3
        if ar and ac:
            n = 4
        elif ar or ac:
            n = 3 if tee else (4 if rng.random() < d.FOUR_STAGE_SHARE_ARTERIAL_MINOR else 3)
        else:
            n = 2
        stages[a] = n
        kind[a] = ("T " if tee else "") + ("arterial x arterial" if ar and ac else "arterial x minor" if ar or ac
                                           else "minor x minor")
        major_dir[a] = "EW" if ar else "NS"            # the arterial direction of a 3-stage intersection

    def stage_of(a, side, turn):
        n, ns = stages[a], side in ("N", "S")
        if n == 4:
            return (0 if ns else 2) + (1 if turn == "l" else 0)
        if n == 3:
            if ns == (major_dir[a] == "NS"):
                return 1 if turn == "l" else 0
            return 2
        return 0 if ns else 1

    connections, movements = [], []                 # movements: (a, from, to, turn, stage, lanes)
    for a in nodes:
        for side in sides_of[a]:
            z = incoming[a, side]
            i = SIDES.index(side)
            want = {"r": SIDES[(i + 3) % 4], "s": SIDES[(i + 2) % 4], "l": SIDES[(i + 1) % 4]}
            turns = {t: outgoing[a, s] for t, s in want.items() if (a, s) in outgoing}
            turns["k"] = f"snk_{a}"
            lanes = lanes_for(list(turns), len(roads[z]["lanes"]))
            for t, y in turns.items():
                movements.append((a, z, y, t, stage_of(a, side, t), lanes[t]))
        for side in sides_of[a]:                     # local source: stage 0, one lane
            movements.append((a, f"src_{a}", outgoing[a, side], "src", 0, [0]))
    for a, z, y, t, st, lanes in movements:
        for lane in lanes:
            connections.append({"from": z, "to": y, "fromLane": lane, "tl": a, "turn": t,
                                "served_native_phases": [2 * st], "service_veh_per_second": d.LANE_RATE})
    adjacency = {a: sorted(nb for nb in (neighbor(a, s) for s in SIDES) if nb not in (None, "fringe")) for a in nodes}
    network = {"roads": roads, "connections": connections,
               "signals": {a: {"green_native_indices": [2 * k for k in range(stages[a])]} for a in nodes},
               "entry_edges": entries, "exit_edges": exits, "nonsignal_service_veh_per_lane_second": d.LANE_RATE}
    return network, movements, owner, adjacency, stages, kind, major_dir, cls, sides_of


# ---------------------------------------------------------------- demand

def od_table(geo, network, adjacency, cls):
    """{(origin link, destination link): (group, peak trips before scaling)}; group totals as GROUP_SHARES."""
    nodes, coords = geo["nodes"], geo["coords"]
    weight = {e: (2.0 if cls[e] == "arterial" else 1.0) for e in network["entry_edges"] + network["exit_edges"]}
    fringe_in = [e for e in network["entry_edges"] if e.startswith("in_")]
    fringe_out = [e for e in network["exit_edges"] if e.startswith("out_")]
    node_of = lambda e: e.split("_", 1)[1].rsplit("_", 1)[0]                  # in_J0_1_N -> J0_1
    pos = {a: (geo["x"][coords[a][1]], geo["y"][coords[a][0]]) for a in nodes}
    cx, cy = (geo["x"][0] + geo["x"][-1]) / 2, (geo["y"][0] + geo["y"][-1]) / 2
    k = max(1, round(d.CENTRE_SHARE * len(nodes)))
    centre = sorted(nodes, key=lambda a: (math.hypot(pos[a][0] - cx, pos[a][1] - cy), coords[a]))[:k]
    hops = dict(nx.all_pairs_shortest_path_length(nx.Graph([(a, b) for a in adjacency for b in adjacency[a]])))
    od = {}

    def spread(group, origins, dest_of):
        total = d.GROUP_SHARES[group]
        ow = {o: (weight[o] if o.startswith("in_") else 1.0) for o in origins}
        osum = sum(ow.values())
        for o in origins:
            dests = dest_of(o)
            dsum = sum(dests.values())
            for dd, w in dests.items():
                od[o, dd] = (group, od.get((o, dd), (group, 0.0))[1] + total * ow[o] / osum * w / dsum)

    spread("through", fringe_in, lambda o: {e: weight[e] for e in fringe_out if node_of(e) != node_of(o)})
    spread("central", [f"src_{a}" for a in nodes] + fringe_in,
           lambda o: {f"snk_{c}": 1.0 for c in centre if c != (o[4:] if o.startswith("src_") else None)})
    lo, hi = d.LOCAL_HOPS
    spread("local", [f"src_{a}" for a in nodes if any(lo <= h <= hi for h in hops[a].values())],
           lambda o: {f"snk_{b}": 1.0 for b, h in hops[o[4:]].items() if lo <= h <= hi})
    return od, centre


def link_graph(network, roads_tt, movements):
    g = nx.DiGraph()
    for a, z, y, t, st, lanes in movements:
        g.add_edge(z, y, weight=roads_tt[y] + (d.LEFT_PENALTY if t == "l" else 0.0))
    return g


def _routes_init(edges):
    g = nx.DiGraph()
    g.add_weighted_edges_from(edges)
    _G["g"] = g


def _routes_job(pairs):
    """For each (o, d): up to ROUTES fastest routes and their logit shares -> list of (o, d, [(path, share)])."""
    g, out = _G["g"], []
    for o, dd in pairs:
        paths = list(itertools.islice(nx.shortest_simple_paths(g, o, dd, weight="weight"), d.ROUTES))
        cost = np.array([sum(g[u][v]["weight"] for u, v in zip(p, p[1:])) for p in paths])
        w = np.exp(-d.LOGIT_THETA * (cost - cost.min()))
        out.append((o, dd, [(p, float(s)) for p, s in zip(paths, w / w.sum())]))
    return out


def assign(od, g, workers):
    pairs = sorted(od)
    chunks = [pairs[i::max(1, workers * 8)] for i in range(max(1, workers * 8))]
    edges = [(u, v, dd["weight"]) for u, v, dd in g.edges(data=True)]
    if workers > 1 and len(pairs) > 200:
        with Pool(workers, initializer=_routes_init, initargs=(edges,)) as pool:
            res = [r for part in pool.map(_routes_job, chunks) for r in part]
    else:
        _routes_init(edges)
        res = _routes_job(pairs)
    move = {}                                         # (from, to) -> peak flow per group
    origin = {}                                       # origin link -> peak flow per group
    groups = list(d.GROUP_SHARES)
    routes_used = []
    for o, dd, routes in res:
        grp, trips = od[o, dd]
        gi = groups.index(grp)
        origin.setdefault(o, np.zeros(len(groups)))[gi] += trips
        routes_used.append(len(routes))
        for path, share in routes:
            for u, v in zip(path, path[1:]):
                move.setdefault((u, v), np.zeros(len(groups)))[gi] += trips * share
    return move, origin, routes_used


# ---------------------------------------------------------------- signals and demand level

def balanced(groups, lam_mov, cycle):
    """Green split with the largest common capacity reserve under the green bounds; returns (greens, x per movement)."""
    n = len(groups[0]["gamma"])
    L = d.lost_time(n)
    c = np.zeros(n + 1)
    c[-1] = -1.0
    A = [[-g["saturation_rate_veh_per_second"] * y for y in g["gamma"]] + [lam * cycle] for g, lam in zip(groups, lam_mov)]
    res = linprog(c, A_ub=A, b_ub=np.zeros(len(A)), A_eq=[[1.0] * n + [0.0]], b_eq=[cycle - L],
                  bounds=[(d.G_MIN, d.G_MAX)] * n + [(0, None)], method="highs")
    g = res.x[:n]
    x = [lam * cycle / (m["saturation_rate_veh_per_second"] * float(np.dot(m["gamma"], g))) for m, lam in zip(groups, lam_mov)]
    return g, x


def webster(groups, lam_mov, cycle, movements=None):
    """Flow-weighted Webster delay over the first `movements` groups (the movements; the rest are approach groups
    that only constrain the split) and the largest degree of saturation over all groups."""
    g, x = balanced(groups, lam_mov, cycle)
    total = flow = 0.0
    k = len(groups) if movements is None else movements
    for m, lam, xl in zip(groups[:k], lam_mov[:k], x[:k]):
        if lam <= 0:
            continue
        share = float(np.dot(m["gamma"], g)) / cycle
        xc = min(xl, 0.95)
        total += lam * (cycle * (1 - share) ** 2 / (2 * (1 - xc * share)) + xc ** 2 / (2 * lam * (1 - xc)))
        flow += lam
    return total / flow, max(x)


def critical_ratio(groups, lam_mov):
    """Y_m: sum over stages of the largest flow ratio of a capacity group served in the stage."""
    n = len(groups[0]["gamma"])
    worst = np.zeros(n)
    for m, lam in zip(groups, lam_mov):
        st = int(np.argmax(m["gamma"]))
        worst[st] = max(worst[st], lam / m["saturation_rate_veh_per_second"])
    return float(worst.sum())


def signals_of(choice, stages_name):
    sig = {}
    for a, ch in sorted(choice.items()):
        c, n = ch["cycle"], ch["stages"]
        budget = d.green_budget(n, c)
        sig[a] = {"stage_count": n, "cycle_seconds": c, "steps_per_cycle": c // d.CYCLE, "offset_seconds": 0,
                  "yellow_seconds": d.YELLOW, "lost_time_seconds": d.lost_time(n), "green_budget_seconds": budget,
                  "green_min_seconds": d.G_MIN, "green_max_seconds": d.G_MAX,
                  "reference_green_seconds_at_t0": [budget / n] * n,
                  "stages": [{"name": stages_name[a][k], "green_native_index": 2 * k,
                              "yellow_native_index": 2 * k + 1} for k in range(n)]}
    return sig


def stage_names(stages, major_dir):
    out = {}
    for a, n in stages.items():
        if n == 4:
            out[a] = ("NS_sr", "NS_l", "EW_sr", "EW_l")
        elif n == 3:
            m = major_dir[a]
            out[a] = (f"{m}_sr", f"{m}_l", ("EW" if m == "NS" else "NS") + "_all")
        else:
            out[a] = ("NS_all", "EW_all")
    return out


def percentiles(v):
    v = np.asarray(v, float)
    return {"median": float(np.median(v)), "p75": float(np.percentile(v, 75)), "p90": float(np.percentile(v, 90)),
            "max": float(v.max()), "min": float(v.min())}


def prepare(n):
    rows, cols = d.SIZES[n]
    rng = np.random.default_rng(d.NET_SEED + n)
    geo = geometry(rows, cols, rng)
    network, movements, owner, adjacency, stages, kind, major_dir, cls, sides_of = layout(geo, rng)
    tt = {z: max(1.0, float(r["lanes"][0]["length"]) / float(r["lanes"][0]["speed"])) for z, r in network["roads"].items()}
    od, centre = od_table(geo, network, adjacency, cls)
    move, origin, routes_used = assign(od, link_graph(network, tt, movements), d.WORKERS)
    base = {(z, y): float(move.get((z, y), np.zeros(3)).sum()) for a, z, y, t, st, lanes in movements}
    # design turning shares (an approach without trips keeps equal shares, so the model stays defined)
    out_of = {}
    for a, z, y, t, st, lanes in movements:
        out_of.setdefault(z, []).append(y)
    shares = {}
    for z, ys in out_of.items():
        tot = sum(base[z, y] for y in ys)
        shares[z] = {y: (base[z, y] / tot if tot > 0 else 1.0 / len(ys)) for y in ys}
    network["turn_priors"] = {z: {"shares": s} for z, s in shares.items()}
    groups, cap_groups = {a: [] for a in geo["nodes"]}, {a: [] for a in geo["nodes"]}
    approach_stage = {}
    for a, z, y, t, st, lanes in movements:
        groups[a].append({"incoming_edge": z, "destination_edge": y, "declared_turn_share": shares[z][y],
                          "saturation_rate_veh_per_second": d.LANE_RATE * len(lanes), "turn": t,
                          "gamma": [1.0 if s == st else 0.0 for s in range(stages[a])],
                          "fixed_phase_service_seconds_equivalent": 0})
        cap_groups[a].append((groups[a][-1], [(z, y)]))
        g = approach_stage.setdefault((a, z, st), [set(), []])
        g[0].update(lanes)
        g[1].append((z, y))
    for (a, z, st), (lanes, members) in approach_stage.items():
        if len(members) > 1:
            cap_groups[a].append(({"saturation_rate_veh_per_second": d.LANE_RATE * len(lanes),
                                   "gamma": [1.0 if s == st else 0.0 for s in range(stages[a])]}, members))

    def group_flows(a, f):
        return [sum(f[k] for k in members) for _, members in cap_groups[a]]

    Y1 = {a: critical_ratio([g for g, _ in cap_groups[a]], group_flows(a, base)) for a in groups}
    lam = d.Y_MEDIAN_TARGET / float(np.median(list(Y1.values()))) * d.DEMAND_FACTOR
    flows = {k: v * lam for k, v in base.items()}
    choice = {}
    for a, g in groups.items():
        cg, lam_cg = [c for c, _ in cap_groups[a]], group_flows(a, flows)
        d60, x60 = webster(cg, lam_cg, 60, len(g))
        d120, x120 = webster(cg, lam_cg, 120, len(g))
        cyc = d.COMMON_CYCLE or (60 if d60 <= d120 else 120)
        choice[a] = {"cycle": cyc, "stages": stages[a], "delay_60": d60, "delay_120": d120,
                     "max_x": x60 if cyc == 60 else x120, "Y": Y1[a] * lam, "kind": kind[a],
                     "design_greens": balanced(cg, lam_cg, cyc)[0].tolist()}
    link = {z: 0.0 for z in network["roads"]}
    for (z, y), f in flows.items():
        link[z] += f
    edges = sorted(network["roads"])
    table = {"edge_ids": edges, "rows": [{"clock_seconds": t, "arrival_rate_veh_per_second":
                                          [link[e] * d.r(t) for e in edges]} for t in range(0, d.MAX_TIME + 1, 60)]}
    group_rates = {e: (origin.get(e, np.zeros(3)) * lam).tolist() for e in network["entry_edges"]}
    lengths = [float(network["roads"][z]["lanes"][0]["length"]) for z in network["roads"] if ">" in z]
    lanes_hist = {}
    for z in network["roads"]:
        if not z.startswith(("src_", "snk_")):
            k = len(network["roads"][z]["lanes"])
            lanes_hist[k] = lanes_hist.get(k, 0) + 1
    stage_hist = {k: sum(1 for v in stages.values() if v == k) for k in (2, 3, 4)}
    cyc_hist = {c: sum(1 for v in choice.values() if v["cycle"] == c) for c in d.CYCLE_CHOICES}
    stats = {"links_internal": sum(1 for z in network["roads"] if ">" in z), "links_all": len(network["roads"]),
             "minor_links_before_deletion": geo["minor_links"], "deleted_segments": len(geo["deleted"]),
             "tee_intersections": sum(1 for a in geo["nodes"] if len(sides_of[a]) == 3),
             "lanes_per_link": lanes_hist, "internal_link_length_m": percentiles(lengths),
             "stages": stage_hist, "cycles": cyc_hist, "lambda": lam,
             "Y": percentiles([c["Y"] for c in choice.values()]),
             "busiest_movement_x": percentiles([c["max_x"] for c in choice.values()]),
             "centre": centre, "od_pairs": len(od), "routes_per_pair_mean": float(np.mean(routes_used)),
             "peak_entry_flow_veh_per_second": float(sum(sum(v) for v in group_rates.values())),
             "group_peak_flow_veh_per_second": {g: float(sum(v[i] for v in group_rates.values()))
                                                for i, g in enumerate(d.GROUP_SHARES)},
             "lyg": d.LYG}
    out = {"n": n, "rows": rows, "cols": cols, "network": network,
           "service_groups": {"signals": {a: {"movement_capacity_groups": g} for a, g in groups.items()}},
           "edge_owner": owner, "coords": geo["coords"], "adjacency": adjacency,
           "position_m": {a: (geo["x"][c], geo["y"][r]) for a, (r, c) in geo["coords"].items()},
           "deleted_segments": [sorted(s) for s in geo["deleted"]],
           "signals": signals_of(choice, stage_names(stages, major_dir)), "cycle_choice": choice,
           "arrival_table": table, "entry_group_rates_veh_per_second": group_rates, "groups": list(d.GROUP_SHARES),
           "entry_peak_rate_veh_per_second": stats["peak_entry_flow_veh_per_second"], "stats": stats,
           "rule": __doc__}
    folder = d.size_dir(n)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "net.json").write_text(json.dumps(out), encoding="utf-8")
    (folder / "NET_STATS.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    print(f"N={n} ({rows}x{cols}) links={len(edges)} deleted={len(geo['deleted'])} T={stats['tee_intersections']} "
          f"stages 4/3/2={stage_hist[4]}/{stage_hist[3]}/{stage_hist[2]} cycles 60/120={cyc_hist[60]}/{cyc_hist[120]} "
          f"lambda={lam:.4f} Y median/p75/p90/max={stats['Y']['median']:.2f}/{stats['Y']['p75']:.2f}/"
          f"{stats['Y']['p90']:.2f}/{stats['Y']['max']:.2f} X median={stats['busiest_movement_x']['median']:.2f}",
          flush=True)
    return out


def load(n):
    return json.loads((d.size_dir(n) / "net.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    prepare(int(sys.argv[1]))
