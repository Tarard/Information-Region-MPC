"""Traffic plant of the numerical experiment v2 (replaces SUMO): flow model, 1-s step (user 2026-10-30).

Each link has a travel pipeline of round(length / speed) seconds and, at its stop line, one queue per movement
(approach -> outgoing link). Flow reaching the stop line joins the movement queues by the turning shares of the
day: every day draws, per approach, shares from a Dirichlet distribution with mean = design shares and concentration
DIRICHLET_CONCENTRATION (the MPC keeps the design shares). While its stage is green a movement discharges up to
0.5 veh/s on each lane it uses; a lane used by several green movements is split between them in proportion to their
queues per lane (the MPC model caps the link and each movement, Eq. (4)); yellow and red give no service and a
movement on red takes no part of a shared lane (no blocking). No storage limit: links hold any number of vehicles,
so there is no spill-back. Recorded only: the largest link occupancy (vehicles / lanes x (floor(length / 7.5) + 1)).
Observed state for the MPC: per movement its stop-line queue plus the design share of the vehicles still moving on
the link; per exit link its vehicles.
An exit link releases its flow when it reaches its end. Entries: a flow per second (rate = sum over the demand groups
of the day's group factor x the group's design rate x r(t); one factor per group and day, drawn from
N(1, DAY_SIGMA^2)) enters the entry link at once (no queue outside the network).
Measures: delay = vehicle-seconds in the stop-line queues; TTS = vehicle-seconds in the network; vehicles generated
and completed. A day is cleared when, after the departures end, less than 0.5 vehicle remains in the network.
"""


import numpy as np

import ns_design as d


class Plant:
    def __init__(self, net, seed):
        network = net["network"]
        roads = network["roads"]
        self.edges = sorted(roads)
        self.idx = {e: i for i, e in enumerate(self.edges)}
        nE = len(self.edges)
        self.tt = np.array([max(1, round(float(roads[e]["lanes"][0]["length"]) / float(roads[e]["lanes"][0]["speed"])))
                            for e in self.edges])
        self.pipe = np.zeros((nE, int(self.tt.max())))
        self.rng = np.random.default_rng(seed)
        # movements: one per (approach, outgoing link); their saturation flow (0.5 veh/s x lanes) and stage
        mov, lanes_of_mov = {}, {}
        for c in network["connections"]:
            key = (c["from"], c["to"])
            if key not in mov:
                mov[key] = (c["tl"], c["served_native_phases"][0])
            lanes_of_mov.setdefault(key, set()).add(int(c["fromLane"]))
        keys = sorted(mov)
        self.mov_keys = keys
        self.mov_from = np.array([self.idx[z] for z, _ in keys])
        self.mov_to = np.array([self.idx[y] for _, y in keys])
        self.mov_phase = np.array([mov[k][1] for k in keys])
        lanes = sorted({(k[0], lane) for k in keys for lane in lanes_of_mov[k]})
        lane_idx = {k: i for i, k in enumerate(lanes)}
        pairs = [(i, lane_idx[(k[0], lane)]) for i, k in enumerate(keys) for lane in sorted(lanes_of_mov[k])]
        self.pair_mov = np.array([p[0] for p in pairs])
        self.pair_lane = np.array([p[1] for p in pairs])
        self.mov_nlanes = np.bincount(self.pair_mov, minlength=len(keys)).astype(float)
        self.nlanes = len(lanes)
        design = np.array([network["turn_priors"][z]["shares"][y] for z, y in keys])
        self.mov_design = design
        self.storage = np.array([sum(int(float(l["length"]) // 7.5) + 1 for l in roads[e]["lanes"]) for e in self.edges],
                                float)
        self.internal = np.array([">" in e for e in self.edges])
        self.max_occ = np.zeros(nE)
        self.max_occ_peak = np.zeros(nE)
        self.mov_share = self._day_shares(design)
        self.q = np.zeros(len(keys))
        self.is_exit = np.zeros(nE, bool)
        for e in network["exit_edges"]:
            self.is_exit[self.idx[e]] = True
        self.entries = list(network["entry_edges"])
        self.entry_idx = np.array([self.idx[e] for e in self.entries])
        factors = self.rng.normal(1.0, d.DAY_SIGMA, len(net["groups"])).clip(min=0.0)
        rates = net["entry_group_rates_veh_per_second"]
        self.entry_rate = np.array([float(np.dot(factors, rates[e])) for e in self.entries])
        self.day_group_factors = factors.tolist()
        self.arrival_log = []                     # per second: flow into each entry (causal window)
        self.tls = sorted(net["signals"])
        self.tls_idx = {a: i for i, a in enumerate(self.tls)}
        self.mov_tls_idx = np.array([self.tls_idx[mov[k][0]] for k in keys])
        self.t = 0
        self.generated = self.completed = 0.0
        self.delay = self.tts = 0.0

    def _day_shares(self, design):
        """Dirichlet turning shares of the day per approach (mean = design, concentration DIRICHLET_CONCENTRATION)."""
        out = design.copy()
        for z in np.unique(self.mov_from):
            sel = np.flatnonzero((self.mov_from == z) & (design > 0))
            if len(sel) > 1:
                out[sel] = self.rng.dirichlet(d.DIRICHLET_CONCENTRATION * design[sel])
        return out

    def vehicles_on(self):
        on = self.pipe.sum(axis=1)
        np.add.at(on, self.mov_from, self.q)
        return on

    def observe(self, previous_greens):
        on = self.vehicles_on()
        moving = self.pipe.sum(axis=1)
        movement = self.q + self.mov_design * moving[self.mov_from]
        window = self.arrival_log[-d.OBS_WINDOW:]
        rate = np.mean(window, axis=0) if window else np.zeros(len(self.entries))
        return {"time_seconds": self.t, "period": d.period(self.t),
                "edge_vehicle_count": {e: float(on[i]) for i, e in enumerate(self.edges)},
                "movement_vehicle_count": {k: float(v) for k, v in zip(self.mov_keys, movement)},
                "boundary_backlog_vehicles": {e: 0.0 for e in self.entries},   # no storage limit: nobody waits outside
                "boundary_release_rate_veh_per_second": {e: float(rate[i]) for i, e in enumerate(self.entries)},
                "signals": {a: {"previous_executed_green_seconds": list(g)} for a, g in previous_greens.items()}}

    def run(self, active):
        """Advance len(active) seconds; active[s, k] = native phase index shown by intersection k in second s."""
        for s in range(active.shape[0]):
            self._second(active[s])

    def _second(self, phase_of_tls):
        nE = len(self.edges)
        # 1. flow reaching the stop line; exits release it
        arrive = self.pipe[:, 0].copy()
        self.pipe[:, :-1] = self.pipe[:, 1:]
        self.pipe[:, -1] = 0.0
        self.completed += float(arrive[self.is_exit].sum())
        self.q += arrive[self.mov_from] * self.mov_share
        # 2. discharge while green: 0.5 veh/s per lane, a shared lane split between its green movements by queue
        green = phase_of_tls[self.mov_tls_idx] == self.mov_phase
        per_lane = np.where(green, self.q, 0.0) / self.mov_nlanes
        lane_queue = np.bincount(self.pair_lane, weights=per_lane[self.pair_mov], minlength=self.nlanes)
        part = np.divide(per_lane[self.pair_mov], lane_queue[self.pair_lane],
                         out=np.zeros(len(self.pair_mov)), where=lane_queue[self.pair_lane] > 0)
        capacity = np.bincount(self.pair_mov, weights=d.LANE_RATE * part, minlength=len(self.q))
        moved = np.where(green, np.minimum(self.q, capacity), 0.0)
        self.q -= moved
        inflow = np.bincount(self.mov_to, weights=moved, minlength=nE)
        # 3. entries: the demand flow of this second enters at once
        new = self.entry_rate * d.r(self.t)
        self.generated += float(new.sum())
        self.arrival_log.append(new)
        if len(self.arrival_log) > d.OBS_WINDOW:
            self.arrival_log.pop(0)
        inflow[self.entry_idx] += new
        self.pipe[np.arange(nE), self.tt - 1] += inflow
        # 4. measures
        self.delay += float(self.q.sum())
        self.tts += float(self.pipe.sum() + self.q.sum())
        occ = self.vehicles_on() / self.storage
        self.max_occ = np.maximum(self.max_occ, occ)
        if d.PEAK[0] <= self.t < d.PEAK[1]:
            self.max_occ_peak = np.maximum(self.max_occ_peak, occ)
        self.t += 1

    def in_system(self):
        return float(self.pipe.sum() + self.q.sum())

    def summary(self):
        return {"time_seconds": self.t, "generated": self.generated, "completed": self.completed,
                "in_system": self.in_system(), "delay_vehicle_seconds": self.delay, "tts_vehicle_seconds": self.tts,
                "mean_delay_seconds_per_vehicle": self.delay / max(self.completed, 1.0),
                "max_link_occupancy": float(self.max_occ[self.internal].max()),
                "max_link_occupancy_peak": float(self.max_occ_peak[self.internal].max()),
                "link_occupancy_peak_p90": float(np.percentile(self.max_occ_peak[self.internal], 90)),
                "links_over_storage_peak": int((self.max_occ_peak[self.internal] > 1).sum()),
                "day_group_factors": self.day_group_factors}
