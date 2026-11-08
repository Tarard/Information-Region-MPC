"""Design constants of the Lianyungang experiment (seed-77 design).

Constants of the mixed-cycle MPC design with own-cycle updates: each intersection runs C_m = 60 or 120 s (offset
0). The prediction step and the update interval are T = 60 s (the shortest cycle), horizon 8 steps (480 s). At every
update the intersections whose cycle ends decide their next cycle (60-s ones every 60 s, 120-s ones every 120 s); a
120-s cycle that is running keeps its greens, which enter the QP as data. Every cycle has its own greens."""

import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
V3 = ROOT / "v3"  # copy of the v3 code, inputs and design (read-only)
SUMO = Path(os.environ.get("SUMO_BIN", "C:/Program Files (x86)/Eclipse/Sumo/bin/sumo.exe"))
SUMO_GUI = Path("C:/Program Files (x86)/Eclipse/Sumo/bin/sumo-gui.exe")   # same version; only for watching a run

STEP = 60                # prediction step T = update interval, seconds (the shortest cycle)
STEPS = 8                # Np: horizon steps of T seconds
HORIZON = STEP * STEPS   # 480 s
CYCLE_CHOICES = (60, 120)  # C_m of each intersection, set in the signal input; every C_m is a multiple of T
YELLOW = 3               # yellow after every green, seconds (execution only)
G_MIN, G_MAX = 10, 60    # green bounds of every stage, seconds
MU, GREEN_WEIGHT, CHI_BAR = 1e-2, 1e-4, 0.9
CLOSED_LOOP_TOLERANCE, OPEN_LOOP_TOLERANCE = 1e-5, 1e-7   # KKT acceptance, main.tex Sec. VI
DEMAND_END, MAX_TIME = 7200, 14400   # seed-77 departures end at 7200 s; the rest clears the network
PEAK = (2400, 4800)                  # seed-77 high segment (scale 1.5)
PERIODS = ("medium", "high", "low", "clear")

SCALE = float((ROOT / "SCALE").read_text().strip())   # 1.0: seed-77 days are not scaled
HISTORY_DAYS = tuple(range(1, 21))        # LYG generator (seed-77 design), arrival seeds 1-20
TEST_DAYS = (101, 102, 103, 104, 106, 107, 108, 109, 110, 111)   # seeds 77 and 105 dropped (deadlock inside junction 1674)
TABLE3_DAYS = (101, 102, 103, 104, 106, 107)    # the 50 open-loop samples of Table III come from these days
SCREEN_DAYS = ()
TEMPLATE_DAY = None
FORBIDDEN_DAYS = frozenset()
DEMAND = ROOT / "demand"

NETWORK = V3 / "inputs/lyg_safe.net.xml"
NETWORK_MODEL = ROOT / "inputs/network_model.json"   # turn shares calibrated on the history days of this folder
NATIVE_SIGNALS = V3 / "inputs/native_signals.json"
INFORMATION_RULE = V3 / "inputs/information_rule.json"
MANIFEST = V3 / "inputs/MANIFEST.json"
RED_DELAY_CONFIG = V3 / "design/red_delay_mu1e-2.json"
ARRIVAL_TABLE = ROOT / "design/arrival_table.json"
SERVICE_GROUPS = ROOT / "design/service_groups.json"
SIGNALS = ROOT / "inputs/mix_signals.json"

# Execution environment (DESIGN.md Sec. 7)
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
TRACI_SERVER_VERSION = [22, "SUMO 1.26.0"]       # as in the v3 ENVIRONMENT_ACTUAL.json
MIN_FREE_KB = 1 * 1024 ** 2      # reserve: below it no work of this task runs
LAUNCH_FREE_KB = 3 * 1024 ** 2 // 2   # a SUMO or solver process starts only above it (1.5 GB)
_threads_enforced = False


def enforce_threads():
    """Set the thread variables before NumPy/SciPy load; reject conflicting inherited values."""
    global _threads_enforced
    bad = {k: os.environ[k] for k, v in THREADS.items() if os.environ.get(k, v) != v}
    if bad:
        raise SystemExit(f"Thread settings conflict with the design: {bad}")
    if not _threads_enforced and any(m in sys.modules for m in ("numpy", "scipy")):
        raise SystemExit("NumPy or SciPy was imported before the thread settings.")
    os.environ.update(THREADS)
    _threads_enforced = True


def child_environment():
    """Explicit environment for every child process of this task."""
    env = dict(os.environ)
    bad = {k: env[k] for k, v in THREADS.items() if env.get(k, v) != v}
    if bad:
        raise SystemExit(f"Thread settings conflict with the design: {bad}")
    env.update(THREADS)
    return env


def lost_time(stage_count):
    return YELLOW * stage_count


def green_budget(stage_count, cycle):
    return cycle - lost_time(stage_count)


def period(t):
    """Seed-77 segments: medium 0-2400 (scale 1.2), high 2400-4800 (1.5), low 4800-7200 (1.0); clearance after."""
    if t >= DEMAND_END:
        return "clear"
    return "medium" if t < 2400 else "high" if t < 4800 else "low"


def route_path(seed):
    """Route file of a generator day (inline routes, made by resolve_routes.py from the generator output)."""
    if int(seed) not in HISTORY_DAYS + TEST_DAYS + SCREEN_DAYS:
        raise ValueError(f"Day {seed} is not a day of this task.")
    return DEMAND / f"day{int(seed)}.rou.xml"
