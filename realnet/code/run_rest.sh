#!/usr/bin/env bash
# Seed-77 experiment, stages 2-8 after run_central.sh: IR design (coupling, bank), IR-MPC on the test
# days, closed-loop and open-loop tables, partitioned OSD, the timing pass alone, the closed-loop OSD runs, tables.
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p ../design/ir ../results_ir_v1/runs ../results_ir_v1/logs
irrun() {
  [ -f ../results_ir_v1/runs/ir_seed$1/COMPLETE.json ] && { echo "skip $1"; return; }
  rm -rf ../results_ir_v1/runs/ir_seed$1
  $PY -B cc_scenario.py --seed $1 --out ../results_ir_v1/runs/ir_seed$1 --max-time 14400 --controller ir > ../results_ir_v1/logs/ir_seed$1.log 2>&1
  echo "ir $1 $(date +%T): $(tail -n 1 ../results_ir_v1/logs/ir_seed$1.log | cut -c1-220)"
}
export -f irrun
export PY
$PY -B -c "import cc_design as d; d.enforce_threads(); import ir_common as ir; print('history', ir.HISTORY); print('test', ir.TEST)"
echo "coupling $(date +%T)"; $PY -B ir_offline.py coupling > ../design/ir/coupling.log 2>&1; tail -n 7 ../design/ir/coupling.log
echo "bank $(date +%T)"; $PY -B ir_offline.py bank > ../design/ir/bank.log 2>&1; tail -n 2 ../design/ir/bank.log
echo "ir runs $(date +%T)"
$PY -B -c "import cc_design as d; d.enforce_threads(); import ir_common as ir; print(' '.join(map(str, ir.TEST)))" | tr ' ' '\n' | xargs -P 10 -I{} bash -c 'irrun {}'
for s in closed samples; do echo "tables $s $(date +%T)"; $PY -B ir_tables.py $s > ../results_ir_v1/tables_$s.log 2>&1; tail -n 1 ../results_ir_v1/tables_$s.log; done
echo "partition accuracy $(date +%T)"; $PY -B ir_partition.py accuracy > ../results_ir_v1/partition_accuracy.log 2>&1; tail -n 1 ../results_ir_v1/partition_accuracy.log
echo "timing alone $(date +%T)"; $PY -B ir_partition_timing_onepass.py > ../results_ir_v1/partition_timing.log 2>&1; tail -n 1 ../results_ir_v1/partition_timing.log
echo "osd closed loop $(date +%T)"; bash run_osd.sh
echo "fill tables $(date +%T)"; $PY -B fill_tables.py > ../results_ir_v1/fill_tables.log 2>&1; tail -n 3 ../results_ir_v1/fill_tables.log
echo "all done $(date +%T)"
