"""Run every LYG signal with its native cycle, phase order, and clearance."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import argparse
import inspect
import json
from pathlib import Path
import sys
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import traci
import traci.constants as tc

import aligned_model
from causal_observation import EntryObserver
from native_executor import expected_state, integer_green, legal_green, schedule
from strict_qp import solve_matrices

ROOT = Path(__file__).resolve().parents[1]
FIRST_UPDATE, UPDATE_SECONDS, DISPATCH_RESERVE = 15, 45, 3.0
HORIZON, GREEN_WEIGHT, DEMAND_END, MAX_TIME = 480, 1e-4, 7200, 10800
SUMO = Path(__import__("os").environ.get("SUMO_BIN", "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"))
SOURCE_NAMES = ("scenario.py", "aligned_model.py", "strict_qp.py", "native_executor.py", "causal_observation.py")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def plain(value):
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(v) for v in value]
    return value


def save(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def condition(now):
    return "medium" if now < 2400 else "high" if now < 4800 else "low" if now < DEMAND_END else "clearance"


def observe(conn, network, signals, observer, active, now):
    roads = network["roads"]
    counts = {edge: conn.edge.getLastStepVehicleNumber(edge) for edge in roads}
    halting = {edge: conn.edge.getLastStepHaltingNumber(edge) for edge in roads}
    bins = defaultdict(Counter)
    for vehicle in conn.vehicle.getIDList():
        edge = conn.vehicle.getRoadID(vehicle)
        if edge not in roads:
            continue
        geometry = roads[edge]["lanes"][conn.vehicle.getLaneIndex(vehicle)]
        remaining = max(0.0, float(geometry["length"]) - conn.vehicle.getLanePosition(vehicle)) / float(geometry["speed"])
        bins[edge][str(int(remaining // 15))] += 1
    if any(sum(bins[edge].values()) != counts[edge] for edge in roads):
        raise ValueError("Road counts differ from vehicle position counts.")
    current = {}
    for tls, spec in signals.items():
        if tls in active:
            plan = active[tls]
            phase = next(p for p in plan["phases"] if p["start"] <= now < p["end"])
            current[tls] = {"phase": phase["phase"], "spent_seconds": now - phase["start"],
                            "remaining_cycle_seconds": plan["end"] - now,
                            "committed_green_seconds": list(plan["executed_green_seconds"])}
        else:
            if conn.trafficlight.getProgram(tls) != spec["program_attributes"]["programID"]:
                raise ValueError(f"An untracked manual signal state exists at {tls}.")
            phase = conn.trafficlight.getPhase(tls)
            remaining = conn.trafficlight.getNextSwitch(tls) - now + sum(
                float(p["duration_seconds"]) for p in spec["phases"][phase + 1:])
            current[tls] = {"phase": phase, "spent_seconds": conn.trafficlight.getSpentDuration(tls),
                            "remaining_cycle_seconds": remaining,
                            "committed_green_seconds": list(spec["nominal_green_seconds"])}
        current[tls]["previous_executed_green_seconds"] = list(current[tls]["committed_green_seconds"])
    pending, rates = observer.snapshot(conn, now)
    return {"time_seconds": now, "demand_condition": condition(now), "clearance_after_demand": now >= DEMAND_END,
            "edge_vehicle_count": counts, "edge_halting_count": halting,
            "remaining_freeflow_time_bins_15s": {edge: dict(row) for edge, row in bins.items() if row},
            "boundary_backlog_vehicles": pending, "boundary_release_rate_veh_per_second": rates,
            "signals": current,
            "previous_executed_green_seconds": {tls: item["previous_executed_green_seconds"] for tls, item in current.items()},
            "previous_green_meaning": "Current committed actual cycle, before the first free optimized cycle. Unexecuted offers are excluded."}


def physical_exit_audit(trips, network, planned):
    lane_to_edge = {lane["id"]: edge for edge, road in network["roads"].items() for lane in road["lanes"]}
    completed, bad = [], []
    for trip in trips:
        if float(trip.get("arrival", "-1")) < 0:
            continue
        vehicle = trip.get("id")
        completed.append(vehicle)
        edge = lane_to_edge.get(trip.get("arrivalLane", ""))
        if edge not in network["exit_edges"]:
            bad.append({"vehicle": vehicle, "arrival_lane": trip.get("arrivalLane"), "arrival_edge": edge})
    duplicate = [vehicle for vehicle, count in Counter(completed).items() if count > 1]
    return {"all_completed_trips_end_at_physical_exits": not bad, "bad_completed_trips": bad,
            "duplicate_completed_ids": duplicate, "missing_completed_ids": sorted(planned - set(completed)),
            "extra_completed_ids": sorted(set(completed) - planned), "completed": len(completed)}


def fixed_route_audit(route_file, output, trips, network, departed_routes):
    expected = {row.get("id"): tuple(row.find("route").get("edges").split())
                for row in ET.parse(route_file).getroot().findall("vehicle")}
    path = Path(output) / "vehroute.xml"
    parse_error = None
    try:
        final_rows = ET.parse(path).getroot().findall("vehicle")
    except (OSError, ET.ParseError) as exc:
        final_rows, parse_error = [], f"{type(exc).__name__}: {exc}"
    actual, duplicates, malformed = {}, [], []
    for row in final_rows:
        vehicle = row.get("id")
        routes = row.findall("route")
        if vehicle in actual:
            duplicates.append(vehicle)
        if len(routes) != 1 or row.find("routeDistribution") is not None:
            malformed.append(vehicle)
            continue
        actual[vehicle] = tuple(routes[0].get("edges", "").split())
    initial_bad = [{"vehicle": vehicle, "planned": expected.get(vehicle), "actual": edges}
                   for vehicle, edges in departed_routes.items() if edges != expected.get(vehicle)]
    final_bad = [{"vehicle": vehicle, "planned": expected.get(vehicle), "actual": edges}
                 for vehicle, edges in actual.items() if edges != expected.get(vehicle)]
    missing_final = sorted(set(departed_routes) - set(actual))
    trip_ids = [row.get("id") for row in trips]
    missing_trip = sorted(set(departed_routes) - set(trip_ids))
    extra_trip = sorted(set(trip_ids) - set(departed_routes))
    duplicate_trip = [vehicle for vehicle, count in Counter(trip_ids).items() if count > 1]
    rerouting_bad = [{"vehicle": row.get("id"), "rerouteNo": row.get("rerouteNo")}
                     for row in trips if row.get("rerouteNo") != "0"]
    lane_to_edge = {lane["id"]: edge for edge, road in network["roads"].items() for lane in road["lanes"]}
    wrong_exit = [{"vehicle": row.get("id"), "planned_exit": expected.get(row.get("id"), (None,))[-1],
                   "actual_exit": lane_to_edge.get(row.get("arrivalLane", ""))}
                  for row in trips if float(row.get("arrival", "-1")) >= 0
                  and lane_to_edge.get(row.get("arrivalLane", "")) != expected.get(row.get("id"), (None,))[-1]]
    passed = bool(parse_error is None and not duplicates and not malformed and not initial_bad and not final_bad
                  and not missing_final and not rerouting_bad and not wrong_exit
                  and not missing_trip and not extra_trip and not duplicate_trip)
    return {"passed": passed, "planned_vehicles": len(expected), "initial_route_records": len(departed_routes),
            "final_or_unfinished_route_records": len(actual), "completed_trip_records": sum(float(row.get("arrival", "-1")) >= 0 for row in trips),
            "trip_records_with_zero_rerouteNo": len(trips) - len(rerouting_bad),
            "initial_full_route_mismatches": initial_bad, "final_full_route_mismatches": final_bad,
            "missing_departed_final_route_records": missing_final, "nonzero_or_missing_rerouteNo": rerouting_bad,
            "completed_exit_mismatches": wrong_exit, "duplicate_vehicle_route_records": duplicates,
            "malformed_vehicle_route_records": malformed,
            "vehroute_parse_error": parse_error, "missing_departed_trip_records": missing_trip,
            "extra_trip_records": extra_trip, "duplicate_trip_records": duplicate_trip,
            "unfinished_vehicles": sorted(set(departed_routes) - {row.get("id") for row in trips if float(row.get("arrival", "-1")) >= 0}),
            "not_departed_vehicles": sorted(set(expected) - set(departed_routes))}


def source_binding(root, seed, controller=None):
    root = Path(root)
    manifest = read(root / "inputs/MANIFEST.json")
    inputs = [Path(item["path"]) for item in manifest["files"].values()]
    route = next(item for item in manifest["routes"] if item["seed"] == seed)
    files = inputs + [Path(route["path"]), root / "inputs/MANIFEST.json", SUMO]
    files += [Path(__file__).parent / name for name in SOURCE_NAMES]
    if controller is not None:
        source = inspect.getsourcefile(controller)
        if source:
            files.append(Path(source))
    return {str(path.resolve()): sha(path) for path in sorted(set(files), key=str)}


def load_red_delay(path):
    """Configuration {"mu", "arrival_table", "service_groups", "*_sha256"} for augmented_model; None keeps the
    original QP. The arrival row of a decision time is the row of the same public clock, or the last row
    (10770 s) for later times."""
    if path is None:
        return None
    cfg = read(path)
    for key in ("arrival_table", "service_groups"):
        if sha(cfg[key]) != cfg[key + "_sha256"]:
            raise ValueError(f"The red-delay {key} changed.")
    table = read(cfg["arrival_table"])
    edges = table["edge_ids"]
    rows = {int(r["clock_seconds"]): dict(zip(edges, map(float, r["arrival_rate_veh_per_second"]))) for r in table["rows"]}
    import augmented_model
    return {"mu": float(cfg["mu"]), "rows": rows, "groups": read(cfg["service_groups"]), "model": augmented_model,
            "files": [Path(path), Path(cfg["arrival_table"]), Path(cfg["service_groups"]),
                      Path(augmented_model.__file__), Path(augmented_model._OLD_PATH)]}


def build_instance(network, snapshot, edge_owner, red_delay, now):
    if red_delay is None:
        return aligned_model.build(network, snapshot, HORIZON, green_change_weight=GREEN_WEIGHT, edge_owner=edge_owner)
    rows = red_delay["rows"]
    clock = now if now in rows else max(rows)
    return red_delay["model"].build(network, snapshot, rows[clock], red_delay["mu"], service_groups=red_delay["groups"],
                                    edge_owner=edge_owner, horizon=HORIZON, green_change_weight=GREEN_WEIGHT)


def run_day(root, seed, tag, controller=None, history=False, max_time=MAX_TIME, red_delay_path=None):
    """Call controller(instance, snapshot, central_result). It must return all 28 actions.
    red_delay_path: see load_red_delay; None keeps the original QP."""
    root, seed = Path(root).resolve(), int(seed)
    red_delay = load_red_delay(red_delay_path)
    output = root / "runs" / tag
    if output.exists():
        raise FileExistsError(output)
    if max_time <= 0 or int(max_time) != max_time:
        raise ValueError("Use a positive integer simulation limit.")
    manifest = read(root / "inputs/MANIFEST.json")
    for item in manifest["files"].values():
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("A prepared LYG model input changed.")
    route_row = next(item for item in manifest["routes"] if item["seed"] == seed)
    route = Path(route_row["path"])
    if sha(route) != route_row["sha256"]:
        raise ValueError("A prepared departure input changed.")
    network = read(root / "inputs/network_model.json")
    signals = read(root / "inputs/native_signals.json")["signals"]
    edge_owner = read(root / "inputs/information_rule.json")["edge_owner"]
    if set(signals) != set(network["signals"]) or set(edge_owner) != set(network["roads"]):
        raise ValueError("The LYG signal or road ownership sets differ.")
    output.mkdir(parents=True)
    (output / "snapshots").mkdir()
    (output / "vectors").mkdir()
    binding = source_binding(root, seed, controller)
    for path in (red_delay["files"] if red_delay else []):
        binding[str(path.resolve())] = sha(path)
    save(output / "SOURCE_FREEZE.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": binding})
    command = [str(SUMO), "-n", str(root / "inputs/lyg_safe.net.xml"), "-r", str(route),
               "--device.rerouting.probability", "0", "--step-length", "1", "--seed", "0",
               "--time-to-teleport", "-1", "--collision.action", "warn", "--max-depart-delay", "-1",
               "--no-step-log", "true", "--duration-log.disable", "true", "--tripinfo-output", str(output / "tripinfo.xml"),
               "--tripinfo-output.write-unfinished", "true", "--vehroute-output", str(output / "vehroute.xml"),
               "--vehroute-output.last-route", "true", "--vehroute-output.write-unfinished", "true",
               "--collision-output", str(output / "collisions.xml"), "--error-log", str(output / "sumo.errors.log"),
               "--log", str(output / "sumo.log")]
    save(output / "COMMAND.json", {"command": command, "seed": seed, "tag": tag, "history": history,
                                   "first_update": FIRST_UPDATE, "update_seconds": UPDATE_SECONDS,
                                   "horizon_seconds": HORIZON, "prediction_step_seconds": 60,
                                   "green_change_weight": GREEN_WEIGHT, "dispatch_reserve_seconds": DISPATCH_RESERVE,
                                   "maximum_simulation_seconds": max_time, "demand_end_seconds": DEMAND_END,
                                   "all_28_signals_use_controller": True, "centralized_fallback_allowed": False,
                                   "forecast": manifest["forecast"], "sumo_pauses_during_computation": True,
                                   "online_rerouting": False, "rerouter_additional_file_loaded": False,
                                   "real_time_feasibility_established": False,
                                   "red_delay_config": None if red_delay_path is None else str(Path(red_delay_path).resolve()),
                                   "red_delay_mu": None if red_delay is None else red_delay["mu"]})
    observer, active, pending, last_state = EntryObserver(network), {}, {}, {}
    loaded, departed, arrived, collision_ids, teleport_ids = set(), set(), set(), set(), set()
    departed_routes = {}
    controls, error_count, integer_errors = 0, 0, 0
    action_error_sum = maximum_error = maximum_integer_error = 0.0
    checked = mismatch = vehicle_count_mismatch = late = replaced = compute_misses = 0
    vehicle_seconds = halting_seconds = pending_seconds = 0
    executed_by_signal, numeric_max, solver_counts, fallback_counts = Counter(), Counter(), Counter(), Counter()
    times, params, central_x, central_y, native_errors = [], [], [], [], []
    now, conn, warm, error = 0, None, None, None
    started = time.perf_counter()
    all_edges = []
    final_onroad = final_pending = 0
    try:
        traci.start(command, label=tag, numRetries=3)
        conn = traci.getConnection(tag)
        save(output / "ENVIRONMENT_ACTUAL.json", {"traci_version": conn.getVersion(), "python": sys.version,
                                                  "traci_file": traci.__file__, "sumo_sha256": sha(SUMO)})
        loaded.update(conn.simulation.getLoadedIDList())
        all_edges = list(conn.edge.getIDList())
        for edge in all_edges:
            conn.edge.subscribe(edge, (tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_HALTING_NUMBER))
        for tls, spec in signals.items():
            native = next(p for p in conn.trafficlight.getAllProgramLogics(tls)
                          if p.programID == spec["program_attributes"]["programID"])
            if [p.state for p in native.phases] != [p["state"] for p in spec["phases"]]:
                raise ValueError("The live native signal states changed.")
            if [float(p.duration) for p in native.phases] != [float(p["duration_seconds"]) for p in spec["phases"]]:
                raise ValueError("The live native signal durations changed.")
            conn.trafficlight.subscribe(tls, (tc.TL_RED_YELLOW_GREEN_STATE,))
        with (output / "decisions.jsonl").open("w", encoding="utf-8") as decisions, \
                (output / "trace.jsonl").open("w", encoding="utf-8") as trace, \
                (output / "signal_events.jsonl").open("w", encoding="utf-8") as events:
            while now < max_time:
                for tls, plan in list(active.items()):
                    if now == plan["end"]:
                        conn.trafficlight.setProgram(tls, signals[tls]["program_attributes"]["programID"])
                        conn.trafficlight.setPhase(tls, 0)
                        active.pop(tls)
                        last_state.pop(tls, None)
                for tls, plan in list(pending.items()):
                    if now == plan["start"]:
                        if tls in active:
                            raise ValueError("Two committed signal cycles overlap.")
                        active[tls] = pending.pop(tls)
                        executed_by_signal[tls] += 1
                        events.write(json.dumps(plain({"time_seconds": now, "tls": tls, "event": "cycle_started", **plan})) + "\n")
                if now >= FIRST_UPDATE and (now - FIRST_UPDATE) % UPDATE_SECONDS == 0 and (
                        history or conn.simulation.getMinExpectedNumber() > 0):
                    tick = time.perf_counter()
                    snapshot = observe(conn, network, signals, observer, active, now)
                    save(output / "snapshots" / f"t{now}.json", snapshot)
                    instance = build_instance(network, snapshot, edge_owner, red_delay, now)
                    build_seconds = time.perf_counter() - tick
                    central = solve_matrices(instance["matrices"], warm, primary="clarabel")
                    if not central["valid"]:
                        save(output / "FAILED_QP.json", {"time_seconds": now, **{k: v for k, v in central.items() if k not in ("x", "y")}})
                        raise ValueError(f"The centralized QP failed its original-unit KKT check at {now}.")
                    warm = central["x"].copy(), central["y"].copy()
                    solver_counts[central["solver"]] += 1
                    for name in ("eq_violation", "ineq_violation", "stationarity", "dual_violation", "complementarity"):
                        numeric_max[name] = max(numeric_max[name], central[name])
                    central_actions = {tls: legal_green([float(central["x"][i]) for i in indices], signals[tls])[0]
                                       for tls, indices in instance["output_indices"].items()}
                    actions, extra, controller_seconds = central_actions, {}, 0.0
                    if controller is not None:
                        controller_tick = time.perf_counter()
                        actions, extra = controller(instance, snapshot, central)
                        controller_seconds = time.perf_counter() - controller_tick
                    if set(actions) != set(signals):
                        raise ValueError("The controller did not return all 28 intersections.")
                    rows, executed_actions = [], {}
                    for tls in signals:
                        # A failed local solve keeps its own committed legal action. It never uses Full.
                        if actions[tls] is None:
                            actions[tls] = snapshot["signals"][tls]["committed_green_seconds"]
                            fallback_counts[tls] += 1
                        action, correction = legal_green(list(map(float, actions[tls])), signals[tls])
                        if correction > 1e-6:
                            raise ValueError("A controller action needs more than numerical green repair.")
                        executed_actions[tls] = action
                        continuous_error = float(np.max(np.abs(np.asarray(action) - central_actions[tls]), initial=0.0))
                        int_error = max(abs(a - b) for a, b in zip(integer_green(action, signals[tls]), integer_green(central_actions[tls], signals[tls])))
                        action_error_sum += continuous_error
                        maximum_error = max(maximum_error, continuous_error)
                        maximum_integer_error = max(maximum_integer_error, int_error)
                        error_count += continuous_error > 2.0
                        integer_errors += int_error > 2
                        rows.append({"target": tls, "continuous_error_seconds": continuous_error, "integer_error_seconds": int_error})
                    elapsed = time.perf_counter() - tick
                    offers, missed = {}, []
                    for tls, action in executed_actions.items():
                        start_value = now + float(snapshot["signals"][tls]["remaining_cycle_seconds"])
                        start = round(start_value)
                        if abs(start - start_value) > 1e-6:
                            raise ValueError("A native cycle boundary is not an integer second.")
                        compute_misses += start <= now + elapsed
                        if start <= now + DISPATCH_RESERVE:
                            late += 1
                            missed.append(tls)
                            continue
                        plan = schedule(signals[tls], action, start)
                        plan["decision_time_seconds"] = now
                        plan["controller_role"] = "FULL" if controller is None else tag
                        replaced += tls in pending
                        pending[tls] = plan
                        offers[tls] = start
                    np.savez_compressed(output / "vectors" / f"t{now}.npz", p=instance["p"], x=central["x"], y=central["y"])
                    times.append(now)
                    params.append(instance["p"])
                    central_x.append(central["x"])
                    central_y.append(central["y"])
                    controls += 1
                    record = {"time_seconds": now, "demand_condition": condition(now), "build_seconds": build_seconds,
                              "central_numeric": {k: v for k, v in central.items() if k not in ("x", "y")},
                              "controller_seconds": controller_seconds, "end_to_end_seconds": elapsed,
                              "central_first_green_seconds": central_actions, "requested_green_seconds": executed_actions,
                              "first_green_comparisons": rows, "accepted_start_seconds": offers, "late_intersections": missed,
                              "controller_extra": extra, "local_fallback_count": sum(fallback_counts.values())}
                    decisions.write(json.dumps(plain(record), ensure_ascii=False, allow_nan=False) + "\n")
                    decisions.flush()
                    print(f"CONTROL tag={tag} t={now} full={central['seconds']:.3f}s local={controller_seconds:.3f}s", flush=True)
                for tls, plan in active.items():
                    state = expected_state(plan, now)
                    if last_state.get(tls) != state:
                        conn.trafficlight.setRedYellowGreenState(tls, state)
                        last_state[tls] = state
                conn.simulationStep()
                now = int(conn.simulation.getTime())
                loaded.update(conn.simulation.getLoadedIDList())
                newly_departed = conn.simulation.getDepartedIDList()
                departed.update(newly_departed)
                for vehicle in newly_departed:
                    departed_routes[vehicle] = tuple(conn.vehicle.getRoute(vehicle))
                arrived.update(conn.simulation.getArrivedIDList())
                observer.record_departures(conn)
                collision_ids.update(conn.simulation.getCollidingVehiclesIDList())
                teleport_ids.update(conn.simulation.getStartingTeleportIDList())
                current_signals = conn.trafficlight.getAllSubscriptionResults()
                for tls, plan in active.items():
                    expected = expected_state(plan, now - 1)
                    actual = current_signals[tls][tc.TL_RED_YELLOW_GREEN_STATE]
                    checked += 1
                    if actual != expected:
                        mismatch += 1
                        native_errors.append({"t": now, "tls": tls, "actual": actual, "expected": expected})
                edges = conn.edge.getAllSubscriptionResults()
                onroad = len(conn.vehicle.getIDList())
                edge_total = sum(row[tc.LAST_STEP_VEHICLE_NUMBER] for row in edges.values())
                vehicle_count_mismatch += edge_total != onroad
                queued = sum(row[tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for row in edges.values())
                pending_count = len(conn.simulation.getPendingVehicles())
                vehicle_seconds += onroad + pending_count
                pending_seconds += pending_count
                halting_seconds += queued + pending_count
                trace.write(json.dumps({"time_seconds": now, "vehicles": onroad, "pending": pending_count,
                                        "halting": queued, "all_edge_vehicle_sum": edge_total}) + "\n")
                final_onroad, final_pending = onroad, pending_count
                if not history and now >= DEMAND_END and conn.simulation.getMinExpectedNumber() == 0:
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()
    route_vehicles = ET.parse(route).getroot().findall("vehicle")
    planned = {v.get("id") for v in route_vehicles}
    if len(planned) != len(route_vehicles):
        raise ValueError("The post-run planned inventory contains duplicate vehicle IDs.")
    output_errors = []
    try:
        trips = ET.parse(output / "tripinfo.xml").getroot().findall("tripinfo")
    except (OSError, ET.ParseError) as exc:
        trips = []
        output_errors.append(f"tripinfo.xml: {type(exc).__name__}: {exc}")
    complete = [row for row in trips if float(row.get("arrival", "-1")) >= 0]
    completed = {row.get("id") for row in complete}
    exit_audit = physical_exit_audit(trips, network, planned)
    route_audit = fixed_route_audit(route, output, trips, network, departed_routes)
    try:
        collision_records = ET.parse(output / "collisions.xml").getroot().findall("collision")
    except (OSError, ET.ParseError) as exc:
        collision_records = []
        output_errors.append(f"collisions.xml: {type(exc).__name__}: {exc}")
    unchanged = all(sha(path) == digest for path, digest in binding.items())
    ledger_valid = completed == arrived and departed == set(observer.departed) and not departed - planned
    valid = bool(error is None and not output_errors and unchanged and not collision_ids and not collision_records and not teleport_ids and not mismatch
                 and not vehicle_count_mismatch and ledger_valid and not exit_audit["bad_completed_trips"]
                 and not exit_audit["duplicate_completed_ids"] and not exit_audit["extra_completed_ids"]
                 and route_audit["passed"])
    summary = {"valid": valid, "all_trips_completed": completed == planned, "seed": seed, "tag": tag,
               "status": "VALID" if valid else "INVALID", "error": error, "history": history,
               "planned": len(planned), "loaded": len(loaded), "inserted": len(departed), "completed": len(completed),
               "unfinished": len(planned - completed), "on_road_remaining": final_onroad, "pending": final_pending,
               "final_time_seconds": now, "limit_reached": now >= max_time, "controls": controls,
               "comparisons": controls * len(signals), "mean_continuous_error_seconds": action_error_sum / (controls * len(signals)) if controls else None,
               "maximum_continuous_error_seconds": maximum_error, "continuous_errors_exceeding_2s": error_count,
               "maximum_integer_error_seconds": maximum_integer_error, "integer_errors_exceeding_2s": integer_errors,
               "local_fallback_calls": sum(fallback_counts.values()), "local_fallback_by_signal": fallback_counts,
               "checked_signal_seconds": checked, "signal_state_mismatches": mismatch,
               "all_edge_vehicle_count_mismatches": vehicle_count_mismatch, "collision_vehicle_ids": sorted(collision_ids),
               "teleport_vehicle_ids": sorted(teleport_ids), "source_hashes_unchanged": unchanged,
               "collision_output_events": len(collision_records),
               "required_output_errors": output_errors,
               "fixed_route_audit_passed": route_audit["passed"], "online_rerouting": False,
               "executed_by_signal": executed_by_signal, "late_plan_offers": late, "plans_replaced": replaced,
               "measured_compute_deadline_miss_offers": compute_misses, "maximum_absolute_residuals": numeric_max,
               "solver_counts": solver_counts, "vehicle_seconds_including_pending": vehicle_seconds,
               "halting_seconds_including_pending": halting_seconds, "pending_seconds": pending_seconds,
               "runtime_seconds": time.perf_counter() - started, "all_28_signals_controlled_in_clearance": True,
               "centralized_fallback_used": False, "real_time_feasibility_established": False,
               "total_time_loss_completed_seconds": sum(float(row.get("timeLoss", 0)) for row in complete),
               "total_depart_delay_inserted_seconds": sum(float(row.get("departDelay", 0)) for row in trips),
               "mean_time_loss_completed_seconds": float(np.mean([float(row.get("timeLoss", 0)) for row in complete])) if complete else None,
               "mean_trip_duration_completed_seconds": float(np.mean([float(row.get("duration", 0)) for row in complete])) if complete else None}
    np.savez_compressed(output / "snapshots.npz", t=np.asarray(times), p=np.asarray(params),
                        xcentral=np.asarray(central_x), ycentral=np.asarray(central_y))
    save(output / "SUMMARY.json", summary)
    save(output / "EXIT_AUDIT.json", exit_audit)
    save(output / "FIXED_ROUTE_AUDIT.json", route_audit)
    save(output / "DEPARTED_FULL_ROUTES.json", departed_routes)
    save(output / "FINAL_LEDGER.json", {"planned_ids": sorted(planned), "loaded_ids": sorted(loaded),
                                       "departed_ids": sorted(departed), "arrived_ids": sorted(arrived), "completed_ids": sorted(completed)})
    save(output / "ENTRY_OBSERVER_LEDGER.json", {"departed_origins": observer.departed,
                                               "admitted_by_entry": observer.admitted, "last_pending_by_edge": observer.pending_by_edge})
    save(output / "SIGNAL_MISMATCHES.json", native_errors)
    if error:
        raise RuntimeError(f"The run failed. See {output / 'SUMMARY.json'}: {error}")
    return {"summary": plain(summary), "run_dir": str(output), "t": np.asarray(times), "p": np.asarray(params),
            "xcentral": np.asarray(central_x), "ycentral": np.asarray(central_y)}


def self_check():
    assert FIRST_UPDATE == 15 and UPDATE_SECONDS == 45 and HORIZON == 480 and GREEN_WEIGHT == 1e-4
    assert [condition(t) for t in (0, 2399, 2400, 4799, 4800, 7199, 7200)] == ["medium", "medium", "high", "high", "low", "low", "clearance"]
    signals = read(ROOT / "inputs/native_signals.json")["signals"]
    for signal in signals.values():
        plan = schedule(signal, list(map(float, signal["nominal_green_seconds"])), 123)
        assert plan["end"] - plan["start"] == int(signal["cycle_seconds"])
        assert [row["state"] for row in plan["phases"]] == [row["state"] for row in signal["phases"]]
    assert len(signals) == 28
    with tempfile.TemporaryDirectory(prefix="lyg_fixed_route_check_") as directory:
        folder = Path(directory)
        (folder / "input.xml").write_text('<routes><vehicle id="a"><route edges="start end"/></vehicle></routes>', encoding="utf-8")
        (folder / "vehroute.xml").write_text('<routes><vehicle id="a"><route edges="start end"/></vehicle></routes>', encoding="utf-8")
        trips = [ET.fromstring('<tripinfo id="a" arrival="5" arrivalLane="end_0" rerouteNo="0"/>')]
        fixture = {"roads": {"end": {"lanes": [{"id": "end_0"}]}}}
        assert fixed_route_audit(folder / "input.xml", folder, trips, fixture, {"a": ("start", "end")})["passed"]
        assert not fixed_route_audit(folder / "input.xml", folder, trips, fixture, {"a": ("wrong", "end")})["passed"]
        assert not fixed_route_audit(folder / "input.xml", folder, [], fixture, {"a": ("start", "end")})["passed"]
        trips[0].set("rerouteNo", "1")
        assert not fixed_route_audit(folder / "input.xml", folder, trips, fixture, {"a": ("start", "end")})["passed"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3101)
    parser.add_argument("--tag", default="history_seed3101")
    parser.add_argument("--history", action="store_true")
    parser.add_argument("--max-time", type=int, default=MAX_TIME)
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    self_check()
    if not args.self_check:
        print(json.dumps(run_day(ROOT, args.seed, args.tag, history=args.history, max_time=args.max_time)["summary"], ensure_ascii=False, indent=2))
