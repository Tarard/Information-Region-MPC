"""Validate one run directory from its raw files (DESIGN.md Sec. 5). Reads only; never changes a run.

integrity(): the evidence is complete and unchanged: marker, exact source binding, environment, artifact hashes.
validate(): integrity + all run gates + stored states + trip records. Every check is guarded: a missing or
truncated file gives a reason, never an exception.
"""

from hashlib import sha256
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import cc_design as d

d.enforce_threads()

import numpy as np  # noqa: E402

import cc_executor as ex  # noqa: E402

REQUIRED = ("COMMAND.json", "SOURCE_FREEZE.json", "ENVIRONMENT_ACTUAL.json", "sumo.log", "sumo.errors.log",
            "tripinfo.xml", "vehroute.xml", "collisions.xml", "decisions.jsonl", "trace.jsonl", "signal_events.jsonl",
            "SIGNAL_ACTUAL_CHANGES.json", "SIGNAL_MISMATCHES.json", "MODEL_LABELS.json", "snapshots.npz",
            "EXIT_AUDIT.json", "FIXED_ROUTE_AUDIT.json", "DEPARTED_FULL_ROUTES.json", "FINAL_LEDGER.json",
            "ENTRY_OBSERVER_LEDGER.json", "SUMMARY.json")
MARKER = "COMPLETE.json"


def sha(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def artifact_hashes(run):
    """Every required file plus every stored state; the marker binds this map."""
    run = Path(run)
    files = [run / name for name in REQUIRED if (run / name).exists()]
    files += sorted((run / "snapshots").glob("t*.json")) + sorted((run / "vectors").glob("t*.npz"))
    return {p.relative_to(run).as_posix(): sha(p) for p in files}


def integrity(run, unit_binding, environment, check_current=True):
    """unit_binding: the exact source map of this day; environment: {"identity": ...} expected for the run."""
    run, reasons = Path(run), []
    if not run.is_dir():
        return ["no_run_directory"]
    missing = [n for n in REQUIRED + (MARKER,) if not (run / n).is_file()]
    missing += [n for n in ("snapshots", "vectors") if not (run / n).is_dir()]
    reasons += [f"missing:{n}" for n in missing]
    if MARKER in missing:
        return reasons + ["no_completion_marker"]
    try:
        marker = read(run / MARKER)
        if marker.get("source_sha256") != unit_binding:
            reasons.append("marker_source_binding_differs")
        if "SOURCE_FREEZE.json" not in missing and read(run / "SOURCE_FREEZE.json").get("source_sha256") != unit_binding:
            reasons.append("source_freeze_binding_differs")
        recorded, now = marker.get("artifact_sha256"), artifact_hashes(run)
        if not isinstance(recorded, dict) or not recorded:
            reasons.append("marker_has_no_artifact_hashes")
        else:
            if set(recorded) != set(now):
                reasons.append("artifact_set_differs:" + ",".join(sorted(set(recorded) ^ set(now))[:10]))
            changed = sorted(k for k in set(recorded) & set(now) if recorded[k] != now[k])
            if changed:
                reasons.append("artifact_changed:" + ",".join(changed[:10]))
        if "SUMMARY.json" not in missing and marker.get("summary_sha256") != sha(run / "SUMMARY.json"):
            reasons.append("summary_hash_differs")
    except (OSError, ValueError) as exc:
        reasons.append(f"marker_unreadable:{type(exc).__name__}")
    if check_current:
        changed = [k for k, v in unit_binding.items() if not Path(k).is_file() or sha(k) != v]
        if changed:
            reasons.append("bound_file_changed_since_run:" + ",".join(Path(k).name for k in changed[:10]))
    try:
        actual = read(run / "ENVIRONMENT_ACTUAL.json")
        diff = sorted(k for k in environment["identity"] if actual.get("identity", {}).get(k) != environment["identity"][k])
        if diff:
            reasons.append("environment_differs:" + ",".join(diff))
        if actual.get("traci_version") != d.TRACI_SERVER_VERSION:
            reasons.append("traci_server_version_differs")
    except (OSError, ValueError, KeyError, AttributeError) as exc:
        reasons.append(f"environment_unreadable:{type(exc).__name__}")
    return reasons


def snapshot_parameters(snapshot, labels, signals):
    values = []
    for label in labels:
        kind = label[0]
        if kind == "n0":
            values.append(snapshot["edge_vehicle_count"][label[1]])
        elif kind == "backlog":
            values.append(snapshot["boundary_backlog_vehicles"][label[1]])
        elif kind == "release_rate":
            values.append(snapshot["boundary_release_rate_veh_per_second"][label[1]])
        elif kind == "previous_green":
            order = [s["green_native_index"] for s in signals[label[1]]["stages"]]
            values.append(snapshot["signals"][label[1]]["previous_executed_green_seconds"][order.index(label[2])])
        else:
            raise ValueError(f"Unknown parameter kind {kind}.")
    return np.asarray(values, dtype=float)


def stored_states(run, updates):
    """Shapes, finiteness, times, packed versus single records, snapshots versus p, decisions versus x."""
    run, reasons = Path(run), []
    try:
        labels = read(run / "MODEL_LABELS.json")
        signals = read(d.SIGNALS)["signals"]
        P, N = len(labels["param_labels"]), len(labels["variable_labels"])
        M = len(labels["row_labels"]) + N
        decisions = [json.loads(x) for x in (run / "decisions.jsonl").read_text(encoding="utf-8").splitlines()]
        times = [r["time_seconds"] for r in decisions]
        if times != [d.STEP * i for i in range(len(times))] or len(times) != updates:
            reasons.append("decision_times_not_every_boundary")
        with np.load(run / "snapshots.npz") as z:
            packed = {k: z[k] for k in ("t", "p", "xcentral", "ycentral") if k in z.files}
        if len(packed) != 4:
            return reasons + ["packed_array_missing:" + ",".join(sorted({"t", "p", "xcentral", "ycentral"} - set(packed)))]
        n = len(times)
        shapes = {"t": (n,), "p": (n, P), "xcentral": (n, N), "ycentral": (n, M)}
        bad = [k for k, s in shapes.items() if packed[k].shape != s]
        if bad:
            return reasons + ["packed_shape_differs:" + ",".join(bad)]
        log = run / "FAILED_QP.jsonl"
        failed = {json.loads(x)["time_seconds"] for x in log.read_text(encoding="utf-8").splitlines()} if log.exists() else set()
        ok_rows = np.asarray([t not in failed for t in times])     # a failed centralized QP stores NaN rows
        if not (np.all(np.isfinite(packed["p"])) and all(np.all(np.isfinite(packed[k][ok_rows])) for k in ("xcentral", "ycentral"))):
            reasons.append("packed_not_finite")
        if packed["t"].tolist() != times:
            reasons.append("packed_times_differ")
        names = {"snapshots": {f"t{t}.json" for t in times}, "vectors": {f"t{t}.npz" for t in times}}
        for folder, want in names.items():
            if {p.name for p in (run / folder).glob("t*")} != want:
                reasons.append(f"{folder}_files_differ_from_decisions")
        previous = None
        for i, row in enumerate(decisions):
            t = row["time_seconds"]
            try:
                with np.load(run / "vectors" / f"t{t}.npz") as z:
                    single = {k: z[k] for k in ("p", "x", "y")}
            except Exception as exc:  # noqa: BLE001  truncated or missing archive
                reasons.append(f"vector_unreadable:t{t}:{type(exc).__name__}")
                continue
            if not (np.array_equal(single["p"], packed["p"][i]) and np.array_equal(single["x"], packed["xcentral"][i])
                    and np.array_equal(single["y"], packed["ycentral"][i])):
                reasons.append(f"vector_differs_from_packed:t{t}")
            snap = read(run / "snapshots" / f"t{t}.json")
            if snap.get("time_seconds") != t:
                reasons.append(f"snapshot_time_differs:t{t}")
            if not np.array_equal(snapshot_parameters(snap, labels["param_labels"], signals), packed["p"][i]):
                reasons.append(f"snapshot_parameters_differ:t{t}")
            if i == 0:          # previous = greens of the most recently started cycle of every intersection
                previous = {k: s["reference_green_seconds_at_t0"] for k, s in signals.items()}
            want_previous = [previous[k] for k in sorted(signals)]
            got_previous = [snap["signals"][k]["previous_executed_green_seconds"] for k in sorted(signals)]
            if got_previous != want_previous:
                reasons.append(f"previous_greens_differ:t{t}")
            x = packed["xcentral"][i]
            for tls in row["deciding_signals"]:
                idx = labels["output_indices"][tls]     # t = 0 labels: the first cycle of every intersection
                mpc = row["mpc_first_cycle_green_seconds"][tls]
                if t not in failed and mpc != [float(x[j]) for j in idx]:
                    reasons.append(f"decision_differs_from_x:t{t}:{tls}")
                    break
                if row["executed_green_seconds"][tls] != ex.integer_green(ex.legal_green(mpc, signals[tls])[0], signals[tls]):
                    reasons.append(f"executed_not_rounded_mpc:t{t}:{tls}")
                    break
            previous.update(row["executed_green_seconds"])
    except Exception as exc:  # noqa: BLE001  any unreadable record makes the stored states invalid
        reasons.append(f"stored_states_unreadable:{type(exc).__name__}:{exc}"[:200])
    return reasons


def trip_check(run, summary):
    try:
        trips = ET.parse(Path(run) / "tripinfo.xml").getroot().findall("tripinfo")
        done = [t for t in trips if float(t.get("arrival", "-1")) >= 0]
        delay = [float(t.get("timeLoss")) for t in done]
        mean = float(np.mean(delay)) if delay else None
        reasons = []
        if len(done) != summary.get("completed"):
            reasons.append("tripinfo_completed_differs_from_summary")
        if (mean is None) != (summary.get("mean_complete_delay_seconds") is None) or (
                mean is not None and not math.isclose(mean, summary["mean_complete_delay_seconds"], rel_tol=0, abs_tol=1e-9)):
            reasons.append("tripinfo_delay_differs_from_summary")
        return reasons
    except Exception as exc:  # noqa: BLE001
        return [f"tripinfo_unreadable:{type(exc).__name__}"]


def validate(run, unit_binding, environment, check_current=True):
    reasons = integrity(run, unit_binding, environment, check_current)
    summary = None
    if Path(run).is_dir():
        try:
            summary = read(Path(run) / "SUMMARY.json")
            reasons += [f"gate:{k}" for k, v in summary["gates"].items() if not v]
            if not summary["valid"] and not any(r.startswith("gate:") for r in reasons):
                reasons.append("summary_invalid")
            if summary.get("error"):
                reasons.append("run_error:" + str(summary["error"])[:150])
            reasons += stored_states(run, summary["updates"])
            reasons += trip_check(run, summary)
        except Exception as exc:  # noqa: BLE001
            reasons.append(f"summary_unreadable:{type(exc).__name__}")
    return {"valid": not reasons, "reasons": reasons, "summary": summary}
