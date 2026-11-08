"""Write inputs/mix_signals.json: native stage order and state strings, cycle 60 or 120 s per intersection from
design/CYCLE_CHOICE.json, offset 0, 3-s yellow. Usage: make_signals.py (refuses to overwrite)."""

from datetime import datetime, timezone
from hashlib import sha256
import json

import cc_design as d


def build(native, cycles):
    signals = {}
    for tls, spec in sorted(native["signals"].items()):
        greens, transitions, phases = spec["green_native_indices"], spec["fixed_transition_indices"], spec["phases"]
        # Every native program alternates green and transition phases: G0 T0 G1 T1 ...
        if greens != list(range(0, len(phases), 2)) or transitions != list(range(1, len(phases), 2)):
            raise ValueError(f"{tls}: the native program is not an alternating green/transition sequence.")
        if any("y" not in phases[i]["state"] for i in transitions):
            raise ValueError(f"{tls}: a transition phase has no yellow.")
        n = len(greens)
        stages = [{"green_native_index": g, "green_state": phases[g]["state"],
                   "yellow_native_index": y, "yellow_state": phases[y]["state"],
                   "native_green_seconds": int(phases[g]["duration_seconds"]),
                   "native_transition_seconds": int(phases[y]["duration_seconds"])}
                  for g, y in zip(greens, transitions)]
        cycle = int(cycles[tls]["cycle"])
        if cycle not in d.CYCLE_CHOICES or d.CYCLE % cycle or not n * d.G_MIN <= d.green_budget(n, cycle) <= n * d.G_MAX:
            raise ValueError(f"{tls}: cycle {cycle} s is not allowed or not feasible.")
        budget = d.green_budget(n, cycle)
        signals[tls] = {"stage_count": n, "cycle_seconds": cycle, "cycles_per_step": d.CYCLE // cycle, "offset_seconds": 0,
                        "yellow_seconds": d.YELLOW, "lost_time_seconds": d.lost_time(n),
                        "green_budget_seconds": budget, "green_min_seconds": d.G_MIN, "green_max_seconds": d.G_MAX,
                        "reference_green_seconds_at_t0": [budget / n] * n,
                        "native_cycle_seconds": int(spec["cycle_seconds"]), "stages": stages}
    return signals


if __name__ == "__main__":
    out = d.SIGNALS
    if out.exists():
        raise FileExistsError(out)
    raw = d.NATIVE_SIGNALS.read_bytes()
    value = {"created_utc": datetime.now(timezone.utc).isoformat(),
             "source": str(d.NATIVE_SIGNALS), "source_sha256": sha256(raw).hexdigest(),
             "rule": "Native stage order and state strings; cycle 60 or 120 s per intersection (CYCLE_CHOICE.json), "
                     "offset 0; every green 10-60 s; 3-s yellow after every green (native transition durations "
                     "replaced, state strings kept).",
             "signals": build(json.loads(raw.decode("utf-8-sig")),
                              json.loads((d.ROOT / "design/CYCLE_CHOICE.json").read_text(encoding="utf-8"))["signals"])}
    out.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(out)
