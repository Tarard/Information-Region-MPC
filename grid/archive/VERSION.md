# Grid archive: code version, machine and software

Data of the grid experiment (Table III of the paper), laid out as the `NS_ROOT` folder that `grid/ir_trial.py`
used: `work/N{N}/net.json`, `work/N{N}/NET_STATS.json`, `work/N{N}/runs/central_{seed}/states.npz` and
`SUMMARY.json` (seeds 1-4 history, 101-102 test), `IR_TRIAL_N{N}.json`, and `results/` written by `make_tables.py`.

## Code that produced the closed-loop days

The grid code was not under version control when the experiment ran. The trial code (`ir_trial.py`,
`resample_trial.py`, `make_tables.py`, `code/`) is the one in this repository at commit `1c4c5b0` (the `grid/`
files have been unchanged since `2423d0c`, 2026-11-07). The closed-loop days were run with the following
versions (Windows local time, CET):

| N | Days run | Code of the days |
|---|---|---|
| 49 (7 x 7) | 2026-10-31 13:16-13:23 | as `grid/code` (vehicle cost on every movement queue); `ns_design.py` without the sizes 625 and `DAY_STOP`, `ns_scenario.py` without the `DAY_STOP` check (no effect on these days) |
| 100 (10 x 10) | 2026-10-30 23:00 - 2026-10-31 00:12 | as `grid/code` except `cc_model.py` (vehicle cost 0.5 n^2 / n_max on the link total, diff below); `ns_design.py` without the sizes 49 and 625 and `DAY_STOP`, `ns_scenario.py` without the `DAY_STOP` check (no effect) |
| 200 (10 x 20) | 2026-10-31 00:17-04:05 | same as N = 100 |
| 500 (20 x 25) | 2026-11-01 15:14-16:46 | as `grid/code`, run with `NS_DAY_STOP=5400` (each day ends at the end of the peak; `SUMMARY.json` has `"stopped_at": 5400`) |

Trials (`ir_trial.py`, current model at the stored states): N = 100, 200 on 2026-10-31 07:50-11:54, N = 49 on
2026-10-31 13:26-13:30, N = 500 on 2026-11-01 16:46 - 2026-11-02 02:40.

### N = 100 and 200: change between the day runs and the trial

Only `code/cc_model.py` differs in a way that affects the results. At the day runs the cost 0.5 n^2 / n_max was on
the link total n (sum of the movement queues of a link, one extra variable and equality row per link and step);
for the trial it is on every movement queue q (0.5 q^2 / n_max of its link). With the cost on the link total only,
flows can move between the movements of a link at no cost, the optimum is not unique and the boundary-function
construction failed. The stored states p were kept; `ir_trial.py` solves the current QP at them (reference,
history solutions at t = 4200 s, coupling states and evaluation). The stored `x` at N = 100 and 200 are solutions
of the earlier model (45,040 and 90,030 variables, against 41,184 and 82,318 now) and are not used by the trial.

```diff
--- cc_model.py (closed-loop days at N = 100 and 200)
+++ cc_model.py (trial, and days at N = 49 and 500; grid/code/cc_model.py)
@@ vehicles
     queue, move = {}, {}
     for edge in edges:
-        for k in range(1, d.CYCLES + 1):
-            inventory[edge, k] = problem.variable(("inventory", edge, k), 0, math.inf, 0.5 / capacity[edge])
         if edge in exits:
+            for k in range(1, d.CYCLES + 1):
+                inventory[edge, k] = problem.variable(("inventory", edge, k), 0, math.inf, 0.5 / capacity[edge])
             col = pi["n0"][edge]
@@
             for k in range(1, d.CYCLES + 1):
-                queue[edge, y, k] = problem.variable(("movement_queue", edge, y, k))
+                queue[edge, y, k] = problem.variable(("movement_queue", edge, y, k), 0, math.inf,
+                                                     0.5 / capacity[edge])
             for k in range(d.CYCLES):
                 move[edge, y, k] = problem.variable(("movement_outflow", edge, y, k))
-        for k in range(1, d.CYCLES + 1):
-            problem.constraint(("link_total", edge, k), [(inventory[edge, k], 1)]
-                               + [(queue[edge, y, k], -1) for y in movements[edge]], 0, 0)
@@ direct_cost
     total = sum(0.5 * x[i] ** 2 / meta["capacity"][edge]
                 for kind in ("inventory", "external_wait") for (edge, k), i in instance["indices"][kind].items() if k)
+    total += sum(0.5 * x[i] ** 2 / meta["capacity"][edge]
+                 for (edge, y, k), i in instance["indices"]["movement_queue"].items() if k)
```

(Comment lines of the docstring and the vehicle block also changed; they are left out above.)

## Files not in this archive

GitHub rejects files above 100 MB. These state files are larger and are not committed. They are assets of the release
`grid-states-v1`; `python get_large_states.py` downloads them into this folder and checks their SHA-256:

| File | Size (bytes) |
|---|---|
| work/N200/runs/central_1/states.npz | 126,776,030 |
| work/N200/runs/central_2/states.npz | 125,403,152 |
| work/N200/runs/central_3/states.npz | 126,738,389 |
| work/N200/runs/central_4/states.npz | 125,471,509 |
| work/N200/runs/central_101/states.npz | 128,826,526 |
| work/N200/runs/central_102/states.npz | 136,166,462 |
| work/N500/runs/central_1/states.npz | 139,734,635 |
| work/N500/runs/central_2/states.npz | 139,717,254 |
| work/N500/runs/central_3/states.npz | 139,762,342 |
| work/N500/runs/central_4/states.npz | 139,783,058 |
| work/N500/runs/central_101/states.npz | 139,761,269 |
| work/N500/runs/central_102/states.npz | 139,836,062 |

Also not included, as requested: the boundary-function banks and payloads (`trial_bank_N*.pkl.gz`,
`trial_payload_N*.pkl`), `decisions.jsonl`, `trace.jsonl` and logs. `make_tables.py` reads
`trial_payload_N{N}.pkl` for the equality and inequality counts of Table II, so it does not run on this archive
alone; `ir_trial.py` runs on it for N = 49 and 100.

## Machine and software

- Intel Core i5-12400 (6 cores, 12 threads), 16 GB RAM (15.7 GB usable), Windows 11 Pro 10.0.26200.
- Python 3.13.14; numpy 2.4.4, scipy 1.17.1, clarabel 0.11.1, osqp 1.1.1, networkx 3.6.1.
- Single-threaded BLAS (`OPENBLAS_NUM_THREADS`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS` = 1); Clarabel with one thread.
- Settings of the runs: `NS_DEMAND_FACTOR=0.6`, `NS_WORKERS=4`; trials with 2 construction workers and a 6-GB memory
  cap per worker.
