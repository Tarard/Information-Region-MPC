"""One closed-loop day: centralized MPC decides every 120-s common cycle at its boundary (DESIGN.md).

Adapted from v3 scenario.run_day. Changes: common cycle clock, decision at every boundary from t = 0, every
decision executed, no committed/native cycle, cycle audit from the SUMO signal states, completion marker.
Usage: cc_scenario.py --seed 7001 --out <run dir> [--max-time 10800] [--freeze results_v1/FREEZE.json]
"""

import cc_design as d

d.enforce_threads()

from collections import Counter, defaultdict  # noqa: E402
import os  # noqa: E402
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import traci
import traci.constants as tc

import cc_env
import cc_executor as ex
import cc_model
from frozen import FILES as FROZEN_FILES, causal_observation, strict_qp, v3_scenario
import validate_run

read, save, sha, plain = v3_scenario.read, v3_scenario.save, v3_scenario.sha, v3_scenario.plain
HERE = Path(__file__).resolve().parent
CODE_FILES = ("cc_design.py", "cc_env.py", "cc_executor.py", "cc_model.py", "cc_scenario.py", "frozen.py",
              "validate_run.py")


def load_inputs(seed):
    route = d.route_path(seed)
    if float(read(d.RED_DELAY_CONFIG)["mu"]) != d.MU:
        raise ValueError("The red-delay mu differs from the design.")
    table = read(d.ARRIVAL_TABLE)
    rows = {int(r["clock_seconds"]): dict(zip(table["edge_ids"], map(float, r["arrival_rate_veh_per_second"])))
            for r in table["rows"]}
    signals = read(d.SIGNALS)
    # File hashes are recorded only; they do not stop a run.
    files = [d.NETWORK, d.NETWORK_MODEL, d.NATIVE_SIGNALS, d.INFORMATION_RULE, route, d.SUMO, d.RED_DELAY_CONFIG,
             d.ARRIVAL_TABLE, d.SERVICE_GROUPS, d.SIGNALS, *FROZEN_FILES, *(HERE / name for name in CODE_FILES)]
    binding = {str(path.resolve()): sha(path) for path in sorted(set(files), key=str)}
    return {"route": route, "network": read(d.NETWORK_MODEL), "signals": signals["signals"],
            "edge_owner": read(d.INFORMATION_RULE)["edge_owner"], "arrival_rows": rows,
            "service_groups": read(d.SERVICE_GROUPS), "binding": binding}


def observe(conn, network, observer, previous, now):
    """Causal state at a cycle boundary (v3 observe without the native-program part)."""
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
    pending, rates = observer.snapshot(conn, now)
    return {"time_seconds": now, "period": d.period(now), "clearance_after_demand": now >= d.DEMAND_END,
            "edge_vehicle_count": counts, "edge_halting_count": halting,
            "remaining_freeflow_time_bins_15s": {edge: dict(row) for edge, row in bins.items() if row},
            "boundary_backlog_vehicles": pending, "boundary_release_rate_veh_per_second": rates,
            "signals": {tls: {"previous_executed_green_seconds": list(v)} for tls, v in previous.items()},
            "previous_green_meaning": ("Equal-split reference; no cycle was executed before t = 0." if now == 0 else
                                       "Integer greens of the most recently started cycle (ending now, or still running).")}


def ledger_check(planned, loaded, departed, arrived, observer_departed, trips):
    """Every planned ID is in exactly one state at the end; holds for complete and truncated runs."""
    finished = {r.get("id") for r in trips if float(r.get("arrival", "-1")) >= 0}
    unfinished = {r.get("id") for r in trips if float(r.get("arrival", "-1")) < 0}
    ids = [r.get("id") for r in trips]
    not_departed = planned - departed
    return {"closes": bool(loaded <= planned and departed <= loaded and departed == set(observer_departed)
                           and finished == arrived and unfinished == departed - arrived
                           and len(ids) == len(set(ids)) and set(ids) == departed
                           and len(arrived) + len(unfinished) + len(not_departed) == len(planned)),
            "planned": len(planned), "loaded": len(loaded), "departed": len(departed), "arrived": len(arrived),
            "unfinished_on_road": len(unfinished), "not_departed": len(not_departed)}


def run_day(seed, output, max_time=d.MAX_TIME, environment=None, ir=None, gui=False):
    """ir: an ir_online.IRController; then its local actions are executed and the centralized QP is only compared."""
    """environment: the frozen {"identity": ...}; a mismatch stops the run before SUMO starts."""
    seed, output = int(seed), Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if max_time <= 0 or int(max_time) != max_time:
        raise ValueError("Use a positive integer simulation limit.")
    identity = cc_env.identity()
    if environment is not None and cc_env.differences(environment["identity"], identity):
        raise SystemExit(f"Environment differs from the freeze: {cc_env.differences(environment['identity'], identity)}")
    inputs = load_inputs(seed)
    network, signals, edge_owner, route = inputs["network"], inputs["signals"], inputs["edge_owner"], inputs["route"]
    if set(signals) != set(network["signals"]) or set(edge_owner) != set(network["roads"]):
        raise ValueError("The LYG signal or road ownership sets differ.")
    output.mkdir(parents=True)
    (output / "snapshots").mkdir()
    (output / "vectors").mkdir()
    binding = inputs["binding"]
    save(output / "SOURCE_FREEZE.json", {"created_utc": datetime.now(timezone.utc).isoformat(), "source_sha256": binding})
    command = [str(d.SUMO), "-n", str(d.NETWORK), "-r", str(route),
               "--device.rerouting.probability", "0", "--step-length", "1", "--seed", "0",
               "--time-to-teleport", "1000", "--collision.action", "warn", "--max-depart-delay", "-1",
               "--no-step-log", "true", "--duration-log.disable", "true", "--tripinfo-output", str(output / "tripinfo.xml"),
               "--tripinfo-output.write-unfinished", "true", "--vehroute-output", str(output / "vehroute.xml"),
               "--vehroute-output.last-route", "true", "--vehroute-output.write-unfinished", "true",
               "--collision-output", str(output / "collisions.xml"), "--error-log", str(output / "sumo.errors.log"),
               "--log", str(output / "sumo.log")]
    if gui:   # same simulation in the SUMO GUI: starts at once, no delay, closes at the end
        command = [str(d.SUMO_GUI)] + command[1:] + ["--start", "true", "--quit-on-end", "true", "--delay", "0"]
    save(output / "COMMAND.json", {"command": command, "python_argv": sys.argv, "seed": seed,
                                   "cycle_seconds": {tls: s["cycle_seconds"] for tls, s in sorted(signals.items())},
                                   "offset_seconds": 0, "update_interval_seconds": d.STEP,
                                   "prediction_step_seconds": d.STEP, "horizon_steps": d.STEPS,
                                   "update_rule": "each intersection decides at the start of its own cycle",
                                   "horizon_seconds": d.HORIZON, "first_update_seconds": 0,
                                   "green_bounds_seconds": [d.G_MIN, d.G_MAX], "yellow_seconds": d.YELLOW,
                                   "mu": d.MU, "green_change_weight": d.GREEN_WEIGHT, "chi_bar": d.CHI_BAR,
                                   "maximum_simulation_seconds": max_time, "demand_end_seconds": d.DEMAND_END,
                                   "arrival_row_rule": "nearest table clock, ties to the earlier clock",
                                   "sumo_pauses_during_computation": True, "every_decision_executed": True,
                                   "online_rerouting": False, "rerouter_additional_file_loaded": False,
                                   "thread_environment": {k: os.environ.get(k) for k in d.THREADS},
                                   "environment_checked_against_freeze": environment is not None})
    observer = causal_observation.EntryObserver(network)
    previous = {tls: list(s["reference_green_seconds_at_t0"]) for tls, s in signals.items()}
    plans, last_set, last_actual, actual_changes = {}, {}, {}, defaultdict(list)
    executed = defaultdict(list)
    loaded, departed, arrived, collision_ids, teleport_ids, departed_routes = set(), set(), set(), set(), set(), {}
    checked = mismatch = vehicle_count_mismatch = controls = 0
    max_round = max_repair = 0.0
    numeric_max, solver_counts, native_errors = Counter(), Counter(), []
    times, params, central_x, central_y, solve_seconds = [], [], [], [], []
    now, conn, warm, error, qp_failures = 0, None, None, None, 0
    final_onroad = final_pending = 0
    started = time.perf_counter()
    try:
        traci.start(command, label=f"cc120_{seed}_{os.getpid()}", numRetries=3)
        conn = traci.getConnection(f"cc120_{seed}_{os.getpid()}")
        server = list(conn.getVersion())
        save(output / "ENVIRONMENT_ACTUAL.json", {"traci_version": server, "python": sys.version,
                                                  "traci_file": traci.__file__, "sumo_sha256": sha(d.SUMO),
                                                  "identity": identity})
        if server != d.TRACI_SERVER_VERSION:
            raise ValueError(f"The TraCI server version {server} differs from {d.TRACI_SERVER_VERSION}.")
        loaded.update(conn.simulation.getLoadedIDList())
        for edge in conn.edge.getIDList():
            conn.edge.subscribe(edge, (tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_HALTING_NUMBER))
        for tls in signals:
            conn.trafficlight.subscribe(tls, (tc.TL_RED_YELLOW_GREEN_STATE,))
        with (output / "decisions.jsonl").open("w", encoding="utf-8") as decisions, \
                (output / "trace.jsonl").open("w", encoding="utf-8") as trace, \
                (output / "signal_events.jsonl").open("w", encoding="utf-8") as events:
            while now < max_time:
                if now % d.STEP == 0:
                    # previous = greens of the most recently started cycle: the one ending now, or the running one
                    for tls, plan in plans.items():
                        if plan["end"] < now or (plan["end"] == now) != (now % signals[tls]["cycle_seconds"] == 0):
                            raise ValueError(f"{tls}: a cycle does not end at its own cycle boundary.")
                        previous[tls] = list(plan["executed_green_seconds"])
                    tick = time.perf_counter()
                    snapshot = observe(conn, network, observer, previous, now)
                    save(output / "snapshots" / f"t{now}.json", snapshot)
                    clock, rates = cc_model.arrival_row(inputs["arrival_rows"], now)
                    instance = cc_model.build(network, signals, snapshot, rates, inputs["service_groups"], edge_owner)
                    build_seconds = time.perf_counter() - tick
                    deciding = {tls for tls in signals if tls not in plans or plans[tls]["end"] == now}
                    if set(instance["output_indices"]) != deciding:
                        raise ValueError("The deciding intersections of the QP differ from the cycles ending now.")
                    central = strict_qp.solve_matrices(instance["matrices"], warm, primary="clarabel")
                    solve_seconds.append(central["seconds"])
                    if not central["valid"]:
                        # Not accepted at the closed-loop KKT tolerance: every deciding intersection keeps its
                        # previous greens for its next cycle; the failure is recorded (main.tex Sec. VI failure rule).
                        qp_failures += 1
                        with (output / "FAILED_QP.jsonl").open("a", encoding="utf-8") as failed:
                            failed.write(json.dumps(plain({"time_seconds": now, **{k: v for k, v in central.items()
                                                                                    if k not in ("x", "y")}})) + "\n")
                        # NaN rows of the full length mark the failed update in the stored arrays
                        central["x"] = np.full(len(instance["matrices"]["q"]), np.nan)
                        central["y"] = np.full(instance["matrices"]["A"].shape[0], np.nan)
                    else:
                        warm = central["x"].copy(), central["y"].copy()
                    solver_counts[central["solver"]] += 1
                    for name in ("eq_violation", "ineq_violation", "stationarity", "dual_violation", "complementarity"):
                        numeric_max[name] = max(numeric_max[name], central[name])
                    mpc, integer = {}, {}
                    central_first = {tls: [float(central["x"][i]) for i in idx] if central["valid"] else None
                                     for tls, idx in instance["output_indices"].items()}
                    ir_record = ir.decide(now, instance["p"], previous, instance) if ir is not None else None
                    for tls, indices in instance["output_indices"].items():
                        mpc[tls] = (central_first[tls] if central_first[tls] is not None else list(previous[tls])) \
                            if ir is None else ir_record["action"][tls]
                        try:
                            legal, repair = ex.legal_green(mpc[tls], signals[tls])
                        except ValueError:
                            if ir is None:
                                raise
                            ir_record["failed"].append([tls, "illegal_action"])
                            mpc[tls] = list(previous[tls])
                            legal, repair = ex.legal_green(mpc[tls], signals[tls])
                        integer[tls] = ex.integer_green(legal, signals[tls])
                        max_repair = max(max_repair, repair)
                        max_round = max(max_round, max(abs(a - b) for a, b in zip(integer[tls], mpc[tls])))
                        plans[tls] = ex.schedule(signals[tls], integer[tls], now)
                        executed[tls].append(integer[tls])                   # one decision per cycle
                        events.write(json.dumps({"time_seconds": now, "tls": tls, "event": "cycle_started", **plans[tls]}) + "\n")
                    np.savez_compressed(output / "vectors" / f"t{now}.npz", p=instance["p"], x=central["x"], y=central["y"])
                    times.append(now)
                    params.append(instance["p"])
                    central_x.append(central["x"])
                    central_y.append(central["y"])
                    if controls == 0:
                        save(output / "MODEL_LABELS.json", {"param_labels": instance["param_labels"],
                                                            "variable_labels": instance["variable_labels"],
                                                            "row_labels": instance["row_labels"],
                                                            "output_indices": instance["output_indices"],
                                                            "metadata": instance["metadata"]})
                    controls += 1
                    record = {"time_seconds": now, "period": d.period(now), "arrival_table_clock_seconds": clock,
                              "build_seconds": build_seconds, "solve_seconds": central["seconds"],
                              "end_to_end_seconds": time.perf_counter() - tick,
                              "central_numeric": {k: v for k, v in central.items() if k not in ("x", "y")},
                              "objective_with_constant": float(central["obj"] + instance["objective_constant"]),
                              "mpc_first_cycle_green_seconds": mpc, "executed_green_seconds": integer,
                              "central_first_green_seconds": central_first,
                              "deciding_signals": sorted(instance["output_indices"]),
                              "cycle_end_seconds": {tls: plans[tls]["end"] for tls in sorted(instance["output_indices"])}}
                    if ir_record is not None:
                        record["ir"] = {k: v for k, v in ir_record.items() if k != "action"}
                    decisions.write(json.dumps(plain(record), ensure_ascii=False, allow_nan=False) + "\n")
                    decisions.flush()
                    print(f"CONTROL seed={seed} t={now} solve={central['seconds']:.3f}s", flush=True)
                for tls, plan in plans.items():
                    state = ex.expected_state(plan, now)
                    if last_set.get(tls) != state:
                        conn.trafficlight.setRedYellowGreenState(tls, state)
                        last_set[tls] = state
                conn.simulationStep()
                now = int(conn.simulation.getTime())
                loaded.update(conn.simulation.getLoadedIDList())
                newly = conn.simulation.getDepartedIDList()
                departed.update(newly)
                for vehicle in newly:
                    departed_routes[vehicle] = tuple(conn.vehicle.getRoute(vehicle))
                arrived.update(conn.simulation.getArrivedIDList())
                observer.record_departures(conn)
                collision_ids.update(conn.simulation.getCollidingVehiclesIDList())
                teleport_ids.update(conn.simulation.getStartingTeleportIDList())
                current = conn.trafficlight.getAllSubscriptionResults()
                for tls, plan in plans.items():
                    actual = current[tls][tc.TL_RED_YELLOW_GREEN_STATE]
                    if last_actual.get(tls) != actual:
                        actual_changes[tls].append((now - 1, actual))
                        last_actual[tls] = actual
                    checked += 1
                    if actual != ex.expected_state(plan, now - 1):
                        mismatch += 1
                        native_errors.append({"t": now, "tls": tls, "actual": actual})
                edges = conn.edge.getAllSubscriptionResults()
                onroad = len(conn.vehicle.getIDList())
                edge_total = sum(row[tc.LAST_STEP_VEHICLE_NUMBER] for row in edges.values())
                vehicle_count_mismatch += edge_total != onroad
                pending_count = len(conn.simulation.getPendingVehicles())
                trace.write(json.dumps({"time_seconds": now, "vehicles": onroad, "pending": pending_count,
                                        "halting": sum(r[tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for r in edges.values()),
                                        "all_edge_vehicle_sum": edge_total}) + "\n")
                final_onroad, final_pending = onroad, pending_count
                if now >= d.DEMAND_END and conn.simulation.getMinExpectedNumber() == 0:
                    break
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        if conn is not None:
            conn.close()
    route_vehicles = ET.parse(route).getroot().findall("vehicle")
    planned = {v.get("id") for v in route_vehicles}
    if len(planned) != len(route_vehicles):
        raise ValueError("The planned inventory contains duplicate vehicle IDs.")
    output_errors = []
    try:
        trips = ET.parse(output / "tripinfo.xml").getroot().findall("tripinfo")
    except (OSError, ET.ParseError) as exc:
        trips = []
        output_errors.append(f"tripinfo.xml: {type(exc).__name__}: {exc}")
    complete = [row for row in trips if float(row.get("arrival", "-1")) >= 0]
    completed = {row.get("id") for row in complete}
    exit_audit = v3_scenario.physical_exit_audit(trips, network, planned)
    route_audit = v3_scenario.fixed_route_audit(route, output, trips, network, departed_routes)
    try:
        collision_records = ET.parse(output / "collisions.xml").getroot().findall("collision")
    except (OSError, ET.ParseError) as exc:
        collision_records = []
        output_errors.append(f"collisions.xml: {type(exc).__name__}: {exc}")
    ledger = ledger_check(planned, loaded, departed, arrived, observer.departed, trips)
    cycle_audit = ex.audit_cycles(signals, actual_changes, executed, now)
    gates = {"no_process_error": error is None, "required_outputs_parse": not output_errors,
             "no_collision": not collision_ids and not collision_records,
             "signal_states_match_plan": mismatch == 0,   # teleports allowed after 1000 s (as in Tan et al.)
             "edge_vehicle_counts_match": vehicle_count_mismatch == 0, "ledger_closes": ledger["closes"],
             "exit_audit": not exit_audit["bad_completed_trips"] and not exit_audit["duplicate_completed_ids"]
                           and not exit_audit["extra_completed_ids"],
             "fixed_route_audit": route_audit["passed"], "all_trips_completed": completed == planned,
             "limit_not_reached": now < max_time,
             "cycle_audit": cycle_audit["passed"], "rounding_below_1s": max_round < 1.0,
             "solver_repair_below_0.01s": max_repair <= 0.01}
    valid = all(gates.values())
    delays = [float(r.get("timeLoss")) for r in complete]   # delay = SUMO timeLoss (as in Tan et al.)
    summary = {"valid": valid, "status": "VALID" if valid else "INVALID", "seed": seed, "error": error,
               "failed_gates": [k for k, v in gates.items() if not v], "gates": gates,
               "planned": len(planned), "loaded": len(loaded), "inserted": len(departed), "completed": len(completed),
               "unfinished": len(planned - completed), "on_road_remaining": final_onroad, "pending": final_pending,
               "final_time_seconds": now, "limit_reached": now >= max_time, "updates": controls,
               "qp_failures": qp_failures, "solver_counts": solver_counts, "maximum_absolute_residuals": numeric_max,
               "solve_seconds_median": float(np.median(solve_seconds)) if solve_seconds else None,
               "solve_seconds_max": float(np.max(solve_seconds)) if solve_seconds else None,
               "maximum_rounding_seconds": max_round, "maximum_solver_repair_seconds": max_repair,
               "checked_signal_seconds": checked, "signal_state_mismatches": mismatch,
               "all_edge_vehicle_count_mismatches": vehicle_count_mismatch,
               "collision_vehicle_ids": sorted(collision_ids), "teleport_vehicle_ids": sorted(teleport_ids), "teleports": len(teleport_ids),
               "collision_output_events": len(collision_records), "required_output_errors": output_errors,
               "ledger": ledger, "cycle_audit": cycle_audit,
               "mean_complete_delay_seconds": float(np.mean(delays)) if delays else None,
               "runtime_seconds": time.perf_counter() - started,
               "controller": "central" if ir is None else getattr(ir, "name", "ir"),
               "ir": None if ir is None else ir.summary()}
    np.savez_compressed(output / "snapshots.npz", t=np.asarray(times), p=np.asarray(params),
                        xcentral=np.asarray(central_x), ycentral=np.asarray(central_y))
    save(output / "SIGNAL_ACTUAL_CHANGES.json", actual_changes)
    save(output / "SIGNAL_MISMATCHES.json", native_errors)
    save(output / "EXIT_AUDIT.json", exit_audit)
    save(output / "FIXED_ROUTE_AUDIT.json", route_audit)
    save(output / "DEPARTED_FULL_ROUTES.json", departed_routes)
    save(output / "FINAL_LEDGER.json", {"planned_ids": sorted(planned), "loaded_ids": sorted(loaded),
                                       "departed_ids": sorted(departed), "arrived_ids": sorted(arrived),
                                       "completed_ids": sorted(completed)})
    save(output / "ENTRY_OBSERVER_LEDGER.json", {"departed_origins": observer.departed,
                                               "admitted_by_entry": observer.admitted,
                                               "last_pending_by_edge": observer.pending_by_edge})
    save(output / "SUMMARY.json", summary)
    # Completion marker: written last; binds the run to its source hashes and to every raw artifact.
    save(output / "COMPLETE.json", {"completed_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
                                    "valid": valid, "source_sha256": binding,
                                    "summary_sha256": sha(output / "SUMMARY.json"),
                                    "artifact_sha256": validate_run.artifact_hashes(output)})
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-time", type=int, default=d.MAX_TIME)
    parser.add_argument("--freeze")
    parser.add_argument("--controller", choices=("central", "ir", "osd"), default="central")
    parser.add_argument("--osd", choices=("OSD-1", "OSD-2", "P-OSD"), help="OSD variant (with --controller osd)")
    parser.add_argument("--k", type=int, help="OSD iterations K (with --controller osd)")
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    controller = None
    if args.controller == "ir":
        import ir_online
        controller = ir_online.IRController()
    elif args.controller == "osd":
        import osd_online
        controller = osd_online.OSDController(args.osd, args.k)
    result = run_day(args.seed, args.out, args.max_time, read(args.freeze)["environment"] if args.freeze else None,
                     controller, args.gui)
    print(json.dumps({k: result[k] for k in ("status", "failed_gates", "updates", "completed", "planned",
                                             "final_time_seconds", "mean_complete_delay_seconds")}))
    sys.exit(0 if result["error"] is None else 1)
