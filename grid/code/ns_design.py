"""Constants of the numerical scaling experiment v2 (PLAN_numeric_scale_v2.md). Model settings are those of the LYG
study (main.tex Sec. II and the mixed-cycle design), with multi-rate updates (CHANGES.md section 7): prediction step
T = 60 s, horizon 8 steps (480 s), cycle 60 or 120 s per intersection (offset 0), an intersection decides when its
cycle starts, greens 10-60 s, 3-s yellow after each green, mu = 1e-2, w_g = 1e-4, chi-bar = 0.9. Flow model (no
storage limit, no lanes beyond the saturation flow of a movement, no queue outside the network; CHANGES.md section 10). v2: LYG-like grid (uneven spacing, deleted minor links, 2/3/4-stage plans, 4/2 lanes), OD demand in
three groups with logit routes, demand level from the median critical flow ratio Y_m, day-to-day variation."""

import os
from pathlib import Path
import sys

ROOT = Path(os.environ.get("NS_ROOT") or Path(__file__).resolve().parents[1])   # NS_ROOT: work folder of a test
TEST_MODE = os.environ.get("NS_TEST") == "1"          # tiny run for checking the code (TASK.md step 0)

# Signals and MPC
CYCLE = int(os.environ.get("NS_STEP", "60"))    # prediction step T (s) = shortest update interval; divides every cycle
CYCLE_CHOICES = (60, 120)
COMMON_CYCLE = int(os.environ.get("NS_COMMON_CYCLE", "0"))   # test switch: one cycle for every intersection (0: LYG rule)
YELLOW = 3
G_MIN, G_MAX = 10, 60
CYCLES = int(os.environ.get("NS_STEPS", "8"))   # horizon steps (8 x 60 s = 480 s)
HORIZON = CYCLE * CYCLES
MU, GREEN_WEIGHT, CHI_BAR = 1e-2, 1e-4, 0.9
TOLERANCE = 1e-5             # KKT acceptance in closed loop (history and test days), as in LYG
OFFLINE_TOLERANCE = 1e-7     # KKT acceptance offline (references, coupling, bank) and in the open-loop samples

# Grid networks: rows x columns of signalized intersections
SIZES = {49: (7, 7), 50: (5, 10), 100: (10, 10), 200: (10, 20), 300: (15, 20), 500: (20, 25), 625: (25, 25), 1000: (25, 40)}
if TEST_MODE:
    SIZES = {12: (3, 4)}
NET_SEED = 2026              # network and demand of size N use seed NET_SEED + N (same network in every run)
SPACINGS = tuple(range(150, 501, 50))                     # row and column spacings (m), equally likely
DELETE_FRACTION = 0.10       # of the links on minor roads (both directions), keeping the grid connected
ARTERIAL_SPEED, MINOR_SPEED = 16.67, 11.11                # m/s (60 and 40 km/h)
ARTERIAL_LANES, MINOR_LANES = 4, 2
BOUNDARY_LENGTH = 300.0      # entry and exit links at the fringe, local source and sink links (m)
LANE_RATE = 0.5              # saturation flow per lane (veh/s)
FOUR_STAGE_SHARE_ARTERIAL_MINOR = 0.64                    # else 3 stages (LYG: 16 / 4 / 8 of 4 / 3 / 2 stages)
LEFT_PENALTY, LOGIT_THETA, ROUTES = 10.0, 0.05, 3         # route choice: 3 fastest routes, logit
GROUP_SHARES = {"through": 0.40, "central": 0.30, "local": 0.30}
CENTRE_SHARE = 0.09          # share of intersections nearest the geometric centre that form the centre zone
LOCAL_HOPS = (2, 6)          # local trips go 2-6 intersections away
Y_MEDIAN_TARGET = 0.50       # peak median critical flow ratio Y_m (LYG 0.50)
DEMAND_FACTOR = float(os.environ.get("NS_DEMAND_FACTOR", "1.0"))   # x 0.95 per spill-back retry (CHANGES.md)
DAY_SIGMA, DIRICHLET_CONCENTRATION = 0.05, 200.0          # day-to-day variation of group demand and turning shares
LYG = {"stages": {4: 16, 3: 4, 2: 8}, "cycles": {60: 21, 120: 7}, "Y_median": 0.50, "Y_p75": 0.62, "Y_p90": 0.81,
       "Y_max": 1.29, "X_median": 0.69}

# Demand: same day curve as the LYG days
DEMAND_END, MAX_TIME = (1800, 3600) if TEST_MODE else (9000, 14400)
DAY_STOP = int(os.environ.get("NS_DAY_STOP", "0"))   # test switch: end a closed-loop day at this clock (0: run to the end)
LOW = 0.35
PEAK = (3000, 5400)
PERIODS = ("low", "rise", "peak", "fall", "clear")
HISTORY_SEEDS = (1, 2) if TEST_MODE else tuple(range(1, 21))     # 20 history days, as in the paper
TEST_SEEDS = (101,) if TEST_MODE else tuple(range(101, 107))     # 6 test days, as in the paper
OBS_WINDOW = 300             # causal release-rate window at entries (s)

# IR-MPC and OSD
REFERENCE_MODE = os.environ.get("NS_REFERENCE_MODE", "update")   # "update": p-bar(t) and boundary functions at
                             # every update clock (the paper's method); "period": one reference per demand period
EPS_W = 1.0                  # read-size rule threshold (s), distance order (main.tex Eq. (read-size))
OSD_REGION_SIZES = (10, 25)  # target intersections per partition region (open-loop table)
OSD_CLOSED_LOOP_REGION = 25  # region size of the partitioned OSD run in closed loop
OSD_CLOSED_LOOP_K = 4
OSD_MAX_K, OSD_CONVERGED = 30, 0.01   # iterate to convergence: first-decision change below 0.01 s
SAMPLES_PER_SIZE = 30        # open-loop samples per size (from the centralized test runs)

WORKERS = int(os.environ.get("NS_WORKERS", "24"))
THREADS = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}


def enforce_threads():
    """Single-threaded BLAS. Must take effect before NumPy loads; a later call is fine if the settings already hold."""
    if all(os.environ.get(k) == v for k, v in THREADS.items()):
        return
    if any(m in sys.modules for m in ("numpy", "scipy")):
        raise SystemExit("Import ns_design and call enforce_threads() before NumPy or SciPy.")
    os.environ.update(THREADS)


def lost_time(stage_count):
    return YELLOW * stage_count


def green_budget(stage_count, cycle):
    return cycle - lost_time(stage_count)


def r(t):
    """Day curve of the LYG demand (0.35 low; ramps 1800-3000 and 5400-6600 s; 1.0 peak; ends at DEMAND_END)."""
    if TEST_MODE:
        return 1.0 if t < DEMAND_END else 0.0
    if t >= DEMAND_END:
        return 0.0
    if t < 1800:
        return LOW
    if t < 3000:
        return LOW + (1 - LOW) * (t - 1800) / 1200
    if t < 5400:
        return 1.0
    if t < 6600:
        return 1 - (1 - LOW) * (t - 5400) / 1200
    return LOW


def period(t):
    if t >= DEMAND_END:
        return "clear"
    if TEST_MODE:
        return "peak"
    return "rise" if 1800 <= t < 3000 else "peak" if 3000 <= t < 5400 else "fall" if 5400 <= t < 6600 else "low"


def size_dir(n):
    return ROOT / "work" / f"N{n}"
