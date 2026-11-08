#!/usr/bin/env bash
# Reproduce the IR-MPC scaling trial: 49, 100, 200 intersections from whole centralized MPC days, 500 from
# centralized MPC days up to the end of the peak (5400 s); then the tables. Python 3.13 with numpy 2.4.4, scipy 1.17.1, clarabel 0.11.1, osqp 1.1.1, networkx 3.6.1.
# Set PY to that interpreter. Run times on a 6-core, 16-GB PC: about 1 h (49), 1.5 h (100), 7 h (200), 11 h (500).
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${PY:-python}"
export NS_WORKERS=4 NS_DEMAND_FACTOR=0.6 NS_ROOT="$HERE/runs"
stamp() { date +"%F %T $*"; }
cd "$HERE/code"
days() {   # days N seeds...: closed-loop centralized MPC days, at most 4 at a time
  local n=$1; shift
  mkdir -p "$NS_ROOT/work/N$n/runs" "$NS_ROOT/work/N$n/logs"
  local k=0
  for s in "$@"; do
    [ -f "$NS_ROOT/work/N$n/runs/central_$s/SUMMARY.json" ] && continue
    rm -rf "$NS_ROOT/work/N$n/runs/central_$s"
    $PY -B ns_scenario.py $n $s central "$NS_ROOT/work/N$n/runs/central_$s" > "$NS_ROOT/work/N$n/logs/central_$s.log" 2>&1 &
    k=$((k + 1)); [ $((k % 4)) -eq 0 ] && wait
  done
  wait
}
for n in 49 100 200; do
  [ -f "$NS_ROOT/work/N$n/net.json" ] || $PY -B grid_net.py $n
  stamp "start N=$n days"; days $n 1 2 3 4 101 102; stamp "end N=$n days"
  $PY -B "$HERE/ir_trial.py" $n 2 6 || stamp "FAILED trial N=$n"
done
[ -f "$NS_ROOT/work/N500/net.json" ] || $PY -B grid_net.py 500
NS_DAY_STOP=5400 days 500 1 2 3 4 101 102
$PY -B "$HERE/ir_trial.py" 500 2 6 || stamp "FAILED trial N=500"
# a failed sampled construction can be replaced with: $PY -B "$HERE/resample_trial.py" 500 8
$PY -B "$HERE/make_tables.py" 49 100 200 500
stamp "done"
