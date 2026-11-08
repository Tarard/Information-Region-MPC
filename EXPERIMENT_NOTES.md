# Experiment notes

This file records details of the experiments that the 8-page paper does not report, and explains why. All numbers come from the files in this repository.

## 1. Simulation events (SUMO collisions and teleports)

SUMO runs with `--collision.action warn` and `--time-to-teleport 1000` (`realnet/code/cc_scenario.py`). With `warn`, a collision is logged and both vehicles continue. A teleport removes a vehicle that has waited 1000 s at the same position and inserts it further along its route.

The runs of Table II with at least one event (all other runs have none):

| Run | Teleports | Collision events (vehicles) |
|---|---:|---:|
| IR-MPC, day 102 | 0 | 1 (2) |
| OSD-1, K=1, day 110 | 9 | 0 |
| OSD-1, K=4, day 104 | 92 | 0 |
| OSD-1, K=4, day 110 | 4 | 0 |
| OSD-2, K=1, day 101 | 5 | 0 |
| OSD-2, K=1, day 104 | 124 | 0 |
| OSD-2, K=1, day 110 | 24 | 0 |
| P-OSD, K=1, day 103 | 0 | 1 (2) |
| P-OSD, K=1, day 104 | 25 | 0 |
| P-OSD, K=4, day 102 | 12 | 0 |

Centralized MPC has no event on the ten test days. On the history days, day 16 has 90 teleports and one collision event, and day 18 has 10 teleports.

Why the paper keeps these runs and does not list the events:

- The two collision events (one IR-MPC day, one P-OSD day) each involve two vehicles in a run of about four hours. SUMO reports such events inside junctions when vehicles on conflicting connections overlap; the controller only sets green times and cannot move vehicles. With `warn`, both vehicles continue, so the effect on delay and TTS is negligible. The same event type also occurs once under centralized MPC (history day 16).
- Teleports occur only under OSD variants, on the days on which some of their control inputs deviate considerably from centralized MPC (Sec. V-D). They indicate long queues, which is the effect the paper reports. A teleported vehicle skips part of its queue, so these runs underestimate the delay of the OSD variants. Removing the runs would remove exactly the days that show this effect.
- Days 77 and 105 were dropped from the test set before any controller was compared, because a vehicle deadlock inside junction 1674 stopped the centralized run (`realnet/code/cc_design.py`, `TEST_DAYS`).

## 2. Computation time

- Tables I and III report the median time of the slowest simultaneously solved local QP or subnetwork. For OSD, the time is summed over the K iterations. The time assumes one processor per local problem and excludes communication time, so it is favorable to OSD, which communicates in every iteration.
- Real network (Table I): the times are the medians over all 50 test samples, timed in one run on 7 Nov 2026 with the laptop otherwise idle: one process, one warm-up solve that is not counted, then every sample in turn (`realnet/code/ir_partition_timing_onepass.py`, `results_ir_v1/TABLES_partition_timing_onepass50.json`). Laptop: AMD Ryzen 7 5825U, 16 GB RAM.
- Medians in seconds: centralized MPC 0.968, IR-MPC 0.110, OSD-1 0.177 / 0.705 (K = 1 / 4), OSD-2 0.398 / 1.608, P-OSD 0.380 / 1.528. IR-MPC is 8.8 times faster than centralized MPC and 14.6 times faster than OSD-2 with K = 4 (the paper rounds to 9 and 15); with K = 4, OSD needs 6.4 to 14.6 times the time of IR-MPC.
- Grids (Table III): the times are taken at t = 4200 s, median over six days (four history days and two test days), on one PC (Intel i5-12400, 6 cores, 16 GB). See `grid/archive/results/TRIAL_TABLES.md`.

## 3. Definitions behind Tables I and II

- Error: maximum absolute next-cycle green-time difference over the phases, relative to centralized MPC at the same traffic state. Table I pools all next-cycle decisions of the 50 samples (1,232 decisions), including failed local problems: a failed IR-MPC problem keeps the previous green times, and a failed OSD region keeps its previous iterate.
- Solved: mean over the 50 samples of the share of local problems that were solved (OSD: in iteration K). It is a per-sample average, unlike the pooled error columns.
- The 50 samples: test days 101, 102, 103, 104, 106 and 107, with 9, 9, 8, 8, 8 and 8 evenly spaced updates before 7200 s (`samples_list` in `realnet/code/ir_tables.py`).
- Closed-loop agreement (Sec. V-D): 85.1% of the executed whole-second green times equal those of centralized MPC; 99.3% of the continuous green-time differences are at most 2 s.
- Whole seconds: the executed green times are the QP green times rounded down, with the remaining seconds given to the phases with the largest remainders. This keeps the cycle length and the green-time bounds (`realnet/code/cc_executor.py`).
- Number of vehicles (Sec. V-D, "0.24%"): for each test day, the mean number of vehicles in the network between 3600 and 7200 s under IR-MPC is compared with that under centralized MPC. The 0.24% is the mean of the absolute relative differences over the ten days (largest day: 0.75%). Between 0 and 3600 s the mean is 0.09%. After 7200 s no new vehicles depart and the network clears; the relative differences are larger there (mean 3.7%) because the number of remaining vehicles is small.
- ± in Table II: half-width of the 95% confidence interval over the ten test days (t distribution, 9 degrees of freedom).
- Closed-loop cost J_cl: the stage cost, green-time term and delay term of objective (1a), with the controller weights, evaluated on the SUMO trajectory of each day (`realnet/code/closed_loop_cost.py`).

## 4. Information regions

- Candidate sizes: every integer from 1 to 28. Intersections with the same hop distance are ordered by their identifier (`realnet/code/ir_offline.py`).
- The tolerance ε_W = 1 s is applied to the historical mean coupling, Eq. (20). The theorem in Sec. IV uses worst-case couplings and Assumptions 1-2 on the whole data domain; these assumptions are not verified in the experiments. Thus ε_W selects the regions empirically and is not a certified bound on the total error of the first control input. The accuracy in Tables I and II is measured, not guaranteed.

## 5. Service constraint

The paper writes the service constraint (1f) per link. The code also bounds the flow of every movement by the green time of the phases that serve it (`realnet/code/cc_model.py`, `signal_turn_capacity`). These rows have the same affine form, so the parametric QP (2) and the analysis are unchanged. The paper shows the link form only, for space.

## 6. Synthetic grids (Table III)

- A 1-s flow model replaces SUMO. Every movement has its own queue, and links have no storage limit. The days at 500 intersections run to the end of the peak (5400 s).
- Accuracy: at t = 4200 s on two test days (and four history days, in `grid/archive/results/TRIAL_TABLES.md`). At 500 intersections, the timing and the accuracy are evaluated on 50 sampled local controllers, not on all 500 (`grid/ir_trial.py`, `SAMPLE = 50`); the information regions and their sizes are computed for all 500.
- At 100 and 200 intersections the traffic states come from centralized days that used an earlier vehicle cost (the cost on the link total instead of every movement queue); the current model is solved at these states. The archived states are in `grid/archive/` (N = 49 and 100) and in the release `grid-states-v1` (N = 200 and 500, see `grid/archive/get_large_states.py`); `grid/archive/VERSION.md` gives the code change as a diff. With these states, `ir_trial.py` repeats the Table III trial; `grid/run_all.sh` instead generates new states with the current model.

## 7. Correction of Fig. 3

The first version of `figures/fig_coupling_s77_row.py` set the couplings of an intersection to zero at the samples at which its cycle does not start, and then averaged. This halved the plotted couplings of the seven intersections with 120-s cycles. The script now averages only over the samples at which the cycle starts, as in Eq. (19) and in `realnet/code/ir_offline.py`. The information regions and all controller results did not use the figure script and are not affected. The corrected statistics are in `figures/out/coupling_map_stats.json` (weighted R² of the exponential fit 0.89; 44 of the 71 pairs with mean coupling of at least 0.25 s are non-adjacent).
