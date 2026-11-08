"""Past and present entry observations without route or future-demand access."""

from collections import Counter
import math


class EntryObserver:
    """Count pending entries and record every simulation-step departure once."""

    def __init__(self, network):
        self.edges = tuple(network["entry_edges"])
        if len(self.edges) != len(set(self.edges)):
            raise ValueError("The entry edge list contains duplicates.")
        self.admitted = Counter()
        self.history = {edge: [(0.0, 0)] for edge in self.edges}
        self.departed = {}
        self.pending_by_edge = {edge: () for edge in self.edges}
        self._pending_origin = {}
        self._previous_pending = set()
        self._last_departure_time = None
        self._last_departure_event = ()
        self._last_snapshot_time = None
        self.max_depart_delay = None

    def _check_options(self, conn):
        if self.max_depart_delay is None:
            value = float(conn.simulation.getOption("max-depart-delay"))
            if not math.isfinite(value) or value >= 0:
                raise ValueError("Use max-depart-delay=-1. Discarded insertion attempts would invalidate entry counts.")
            self.max_depart_delay = value

    def record_departures(self, conn):
        """Call after each simulationStep, starting with the first step."""
        self._check_options(conn)
        now = float(conn.simulation.getTime())
        step = float(conn.simulation.getDeltaT())
        vehicles = tuple(conn.simulation.getDepartedIDList())
        if len(vehicles) != len(set(vehicles)):
            raise ValueError("The departure event contains duplicate vehicle IDs.")
        if now == self._last_departure_time:
            if vehicles != self._last_departure_event:
                raise ValueError("The departure event changed without a simulation step.")
            return
        if self._last_departure_time is None:
            if now > step + 1e-8:
                raise ValueError("Departure recording must start after the first simulation step.")
        elif now < self._last_departure_time or now - self._last_departure_time > step + 1e-8:
            raise ValueError("A simulation step is missing from the departure ledger.")
        additions = {}
        for vehicle in vehicles:
            if vehicle in self.departed:
                raise ValueError(f"A vehicle departed more than once: {vehicle}")
            edge = self._pending_origin.get(vehicle)
            if edge is None:
                edge = conn.vehicle.getRoadID(vehicle)
            if edge not in self.history:
                raise ValueError(f"The departure origin was not observed on an entry edge: {vehicle}, {edge}")
            additions[vehicle] = edge
        self.departed.update(additions)
        self.admitted.update(additions.values())
        self._last_departure_time = now
        self._last_departure_event = vehicles

    def snapshot(self, conn, now):
        """Return pending counts and causal release-rate estimates by entry edge."""
        self._check_options(conn)
        now = float(now)
        if not math.isfinite(now) or abs(now - float(conn.simulation.getTime())) > 1e-8:
            raise ValueError("The observation time must equal the current simulation time.")
        if self._last_snapshot_time is not None and now < self._last_snapshot_time:
            raise ValueError("The observation time decreased.")
        if now > 0 and self._last_departure_time != now:
            raise ValueError("Record the current simulation-step departures before the entry observation.")
        by_edge = {}
        observed = set()
        for edge in self.edges:
            ids = tuple(conn.edge.getPendingVehicles(edge))
            if len(ids) != len(set(ids)) or observed.intersection(ids):
                raise ValueError("A pending vehicle was counted on multiple entry edges.")
            for vehicle in ids:
                if vehicle in self.departed:
                    raise ValueError(f"A departed vehicle is still pending: {vehicle}")
                origin = self._pending_origin.get(vehicle, edge)
                if origin != edge:
                    raise ValueError(f"A pending vehicle changed its entry edge: {vehicle}")
            observed.update(ids)
            by_edge[edge] = ids
        global_ids = tuple(conn.simulation.getPendingVehicles())
        if len(global_ids) != len(set(global_ids)):
            raise ValueError("The global pending inventory contains duplicate vehicle IDs.")
        global_set = set(global_ids)
        if observed != global_set:
            missing, extra = sorted(global_set - observed), sorted(observed - global_set)
            raise ValueError(f"Entry edge pending inventory differs from SUMO: missing={missing}, extra={extra}")
        disappeared = self._previous_pending - observed - self.departed.keys()
        if disappeared:
            raise ValueError(f"Pending vehicles disappeared without recorded departures: {sorted(disappeared)}")
        pending = {edge: len(ids) for edge, ids in by_edge.items()}
        for edge in self.edges:
            if self.admitted[edge] + pending[edge] < self.history[edge][-1][1]:
                raise ValueError(f"The cumulative released count decreased on entry edge {edge}.")
        rates = {}
        for edge in self.edges:
            released = self.admitted[edge] + pending[edge]
            history = self.history[edge]
            if history[-1][0] == now:
                history[-1] = (now, released)
            else:
                history.append((now, released))
            while len(history) > 2 and history[1][0] <= now - 300:
                history.pop(0)
            earlier_time, earlier_count = history[0]
            rates[edge] = (released - earlier_count) / max(1.0, now - earlier_time)
        self.pending_by_edge = by_edge
        self._pending_origin.update((vehicle, edge) for edge, ids in by_edge.items() for vehicle in ids)
        self._previous_pending = observed
        self._last_snapshot_time = now
        return pending, rates
