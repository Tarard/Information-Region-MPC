# IR-MPC numerical scaling trial

Code of the numerical experiment that compares the computation time and first-decision accuracy of
information-region MPC (IR-MPC) with centralized MPC on synthetic grid networks of 49, 100, 200 and 500 signalized
intersections. There is no microscopic simulator: a 1-s flow model plays the role of the traffic network.

## Model

| Part | Setting |
|---|---|
| Network | rows x columns grid (49 = 7x7, 100 = 10x10, 200 = 10x20, 500 = 20x25), spacings 150-500 m, arterials on odd rows and columns (4 lanes, 60 km/h), minor roads 2 lanes (40 km/h), 10% of minor links deleted, local source and sink at every intersection |
| Signals | 4, 3 or 2 stages; cycle 60 or 120 s by the lower Webster delay; greens 10-60 s, 3-s yellow, lost time 3 s x stages; offsets 0 |
| Demand | OD trips (through 40%, to the centre 30%, local 30%), 3 fastest routes with logit choice; level set by the median critical flow ratio Y_m = 0.50, times 0.6; LYG day curve; day-to-day variation of group demand and turning shares |
| MPC | step T = 60 s, 8 steps; an intersection decides at the start of each of its cycles, the running cycle of a 120-s intersection is fixed |
| QP constraints | link storage unlimited; one queue per movement, the inflow of a link split by fixed design turning shares; entry queue Eq. (2) of the paper; capacity Eq. (4): one row for the whole link and one per movement; green bounds and cycle sum Eq. (3) |
| QP objective | 0.5 q^2 / n_max for every movement queue q and entry queue (n_max = storage of the link, 7.5 m per vehicle and lane), green-change weight 1e-4, red-delay term with mu = 0.01, chi-bar = 0.9 |
| Solver | Clarabel (strict settings, one thread, no fallback), accepted by absolute KKT residuals: 1e-5 in closed loop, 1e-7 offline |
| Plant | 1-s flow model with link travel times; per lane 0.5 veh/s while green, a shared lane split between its green movements by their queues; no storage limit |

## Pipeline

1. `code/grid_net.py N` builds the network and demand (`runs/work/N{N}/net.json`).
2. `code/ns_scenario.py N SEED central OUT` runs one closed-loop day under centralized MPC and stores the state
   (parameter vector p) at every update. Seeds 1-4 are history days, 101-102 test days.
   At 500 intersections the days run only to the end of the peak (`NS_DAY_STOP=5400`), which is all the trial
   uses. (`fixed_states.py` stores states under the fixed-time design split instead; it is not used for the
   reported numbers.)
3. `ir_trial.py N WORKERS GB`:
   - information regions from the coupling of 40 peak states (sensitivity of the centralized first decision,
     eps_W = 1 s, distance order);
   - reference at t = 4200 s (mean of the history states), boundary function of every intersection (50 sampled by
     stage count at 500);
   - evaluation at t = 4200 s on the 6 days: centralized QP time, local QP time, first-decision difference.
4. `resample_trial.py N` replaces a failed sampled construction by the next not-sampled intersection with the
   same stage count and repeats the evaluation (not needed for the reported numbers: every construction was built).
5. `make_tables.py` writes `results/TRIAL_TABLES.md` and `results/trial_*.csv`.

`run_all.sh` runs everything (set `PY` to the interpreter). Python 3.13 with numpy 2.4.4, scipy 1.17.1,
clarabel 0.11.1, osqp 1.1.1, networkx 3.6.1.

## Archived data

`archive/` holds the `NS_ROOT` folder of the reported trial: `work/N{N}/net.json`, `NET_STATS.json`, the states and summaries of the six centralized days, `IR_TRIAL_N{N}.json` and `results/` (the tables behind Table III of the paper). `archive/VERSION.md` records the code, the machine and the software of every run. The state files of N = 200 and 500 are larger than 100 MB; `python archive/get_large_states.py` downloads them from the release `grid-states-v1`. Then `NS_ROOT=archive python ir_trial.py N` repeats the trial at the archived states.

## How the reported numbers were produced

- The closed-loop days at 100 and 200 were run with the vehicle cost on the link total (0.5 n^2 / n_max). The
  boundary-function construction then failed (with the cost on the link total only, flows can move between the
  movements of a link at no cost, so the optimum is not unique), and the cost was moved to every movement queue.
  The stored traffic states were kept; `ir_trial.py` solves the current QP at them (reference, history solutions,
  coupling and evaluation), so all reported times and errors come from the current model. The days at 49 were run
  with the current model.
- 49 = 7 x 7 replaces 50 = 5 x 10: the 5-row grid put the through traffic on two arterials, and 11 of its 50
  intersections were above saturation at the cycle chosen by the Webster rule (3 of 49 on the 7 x 7 grid, 5 of 100
  on 10 x 10). The rules (cycle choice, demand level) are the same for every size.
- Timings are on one PC (Intel i5-12400, 6 cores, 16 GB), one process during the evaluation. IR-MPC parallel time
  is the slowest local QP; serial time is the mean local time x number of intersections.
