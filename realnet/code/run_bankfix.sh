#!/usr/bin/env bash
# Bank extended to the clearance (clocks with >= 3 history days still running), then IR-MPC on the 10 test days,
# Table I/II (the OSD closed-loop runs wait for a decision on the deadlock in junction 1674). Table III, the partitions and the timing pass are kept
# (they use only clocks before 7200 s, whose bank entries are unchanged: all 20 history days run there).
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p ../results_ir_v1/runs ../results_ir_v1/logs
irrun() {
  [ -f ../results_ir_v1/runs/ir_seed$1/COMPLETE.json ] && { echo "skip $1"; return; }
  rm -rf ../results_ir_v1/runs/ir_seed$1
  $PY -B cc_scenario.py --seed $1 --out ../results_ir_v1/runs/ir_seed$1 --max-time 14400 --controller ir > ../results_ir_v1/logs/ir_seed$1.log 2>&1
  echo "ir $1 $(date +%T): $(tail -n 1 ../results_ir_v1/logs/ir_seed$1.log | cut -c1-220)"
}
export -f irrun
export PY
echo "bank $(date +%T)"; $PY -B ir_offline.py bank > ../design/ir/bank.log 2>&1; tail -n 2 ../design/ir/bank.log
echo "ir runs $(date +%T)"
$PY -B -c "import cc_design as d; d.enforce_threads(); import ir_common as ir; print(' '.join(map(str, ir.TEST)))" | tr ' ' '\n' | xargs -P 10 -I{} bash -c 'irrun {}'
echo "tables closed $(date +%T)"; $PY -B ir_tables.py closed > ../results_ir_v1/tables_closed.log 2>&1; tail -n 1 ../results_ir_v1/tables_closed.log
echo "all done $(date +%T)"
