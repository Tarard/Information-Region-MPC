"""Execute one MPC cycle decision as G0 Y0 G1 Y1 ... in native stage order, C_m seconds in total."""

import math

import cc_design as d


def legal_green(values, signal):
    """Repair numerical solver error of at most 0.01 s onto the legal green set."""
    budget, n = float(signal["green_budget_seconds"]), signal["stage_count"]
    if len(values) != n or not all(math.isfinite(x) for x in values):
        raise ValueError("The MPC green vector does not match the stages.")
    if abs(sum(values) - budget) > 0.01 or any(x < d.G_MIN - 0.01 or x > d.G_MAX + 0.01 for x in values):
        raise ValueError("Solver action is too far from the legal green set.")
    corrected = [min(max(x, d.G_MIN), d.G_MAX) for x in values]
    delta = budget - sum(corrected)
    for _ in corrected:
        if abs(delta) < 1e-8:
            break
        slack = [(d.G_MAX - x) if delta > 0 else (x - d.G_MIN) for x in corrected]
        i = max(range(n), key=slack.__getitem__)
        change = math.copysign(min(abs(delta), slack[i]), delta)
        corrected[i] += change
        delta -= change
    if abs(sum(corrected) - budget) > 1e-8:
        raise ValueError("Legal green repair did not meet the cycle budget.")
    return corrected, max(abs(x - y) for x, y in zip(values, corrected))


def integer_green(values, signal):
    """Whole seconds for the 1-s SUMO step: floor, then largest remainders. Keeps budget and bounds."""
    budget = int(signal["green_budget_seconds"])
    whole = [math.floor(x + 1e-8) for x in values]
    remaining = budget - sum(whole)
    if not 0 <= remaining <= len(values):
        raise ValueError("Cannot round the MPC greens within the cycle.")
    for i in sorted(range(len(values)), key=lambda i: (-(values[i] - whole[i]), i))[:remaining]:
        whole[i] += 1
    if sum(whole) != budget or any(x < d.G_MIN or x > d.G_MAX for x in whole):
        raise ValueError("Rounded greens violate the signal plan.")
    return whole


def schedule(signal, greens, start):
    """One cycle of C_m seconds, G0 Y0 G1 Y1 ..., starting at a cycle boundary of this intersection."""
    if start % signal["cycle_seconds"]:
        raise ValueError("A cycle must start at a cycle boundary of its intersection.")
    phases, now = [], start
    for ordinal, (stage, green) in enumerate(zip(signal["stages"], greens)):
        for kind, length, state, native in (("green", green, stage["green_state"], stage["green_native_index"]),
                                            ("yellow", d.YELLOW, stage["yellow_state"], stage["yellow_native_index"])):
            phases.append({"stage": ordinal, "native_index": native, "kind": kind,
                           "start": now, "end": now + length, "state": state})
            now += length
    if now - start != signal["cycle_seconds"]:
        raise ValueError(f"The executed cycle is not {signal['cycle_seconds']} s.")
    return {"start": start, "end": now, "executed_green_seconds": list(greens), "phases": phases}


def expected_state(plan, now):
    return next(p["state"] for p in plan["phases"] if p["start"] <= now < p["end"])


def audit_cycles(signals, actual_changes, executed, final_time):
    """Rebuild every cycle from the SUMO signal states alone.

    actual_changes: {tls: [(t, state), ...]}, state in force from second t; executed: {tls: [greens of cycle 0,
    cycle 1, ...]} from the decision log; final_time: end of the last simulated second."""
    problems, complete_cycles = [], 0
    if final_time <= 0:
        return {"passed": False, "problems": ["No simulated second."], "complete_cycles_checked": 0}
    for tls, signal in sorted(signals.items()):
        cycle_seconds = signal["cycle_seconds"]
        expected_cycles = math.ceil(final_time / cycle_seconds)
        changes = actual_changes.get(tls, [])
        if not changes or changes[0][0] != 0:
            problems.append(f"{tls}: no signal state at t = 0.")
            continue
        segments = [(t, (changes[i + 1][0] if i + 1 < len(changes) else final_time) - t, state)
                    for i, (t, state) in enumerate(changes)]
        # A cycle decided at the last update may start exactly when the run ends: it was never due.
        if not 0 <= len(executed.get(tls, [])) - expected_cycles <= 1:
            problems.append(f"{tls}: {len(executed.get(tls, []))} decisions for {expected_cycles} started cycles.")
            continue
        size = 2 * signal["stage_count"]
        for c in range(expected_cycles):
            start = c * cycle_seconds
            group = segments[c * size:(c + 1) * size]
            want = [(state, length) for stage, green in zip(signal["stages"], executed[tls][c])
                    for state, length in ((stage["green_state"], green), (stage["yellow_state"], d.YELLOW))]
            last = start + cycle_seconds > final_time
            if not group or group[0][0] != start:
                problems.append(f"{tls}: cycle {c} does not start at {start} s.")
                break
            if not last:
                ok = [(s, n) for _, n, s in group] == want and sum(n for _, n, _ in group) == cycle_seconds
                complete_cycles += ok
            else:   # the run ended inside this cycle: the states must be a prefix of the plan
                got = [(s, n) for _, n, s in group]
                ok = (len(got) <= len(want) and got[:-1] == want[:len(got) - 1]
                      and got[-1][0] == want[len(got) - 1][0] and got[-1][1] <= want[len(got) - 1][1])
            if not ok:
                problems.append(f"{tls}: cycle {c} states or durations differ from the decision.")
                break
        if len(segments) > expected_cycles * size:
            problems.append(f"{tls}: extra signal states after the last cycle.")
    return {"passed": not problems, "problems": problems[:50], "problem_count": len(problems),
            "complete_cycles_checked": complete_cycles}
