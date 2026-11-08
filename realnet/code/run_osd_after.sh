#!/usr/bin/env bash
# After run_bankfix.sh: the 60 OSD closed-loop runs (teleports in junction 1674 accepted, as the user decided) and tables.
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
until grep -q "^all done" "$1"; do sleep 60; done
echo "osd closed loop $(date +%T)"; bash run_osd.sh
echo "fill tables $(date +%T)"; $PY -B fill_tables.py > ../results_ir_v1/fill_tables.log 2>&1; tail -n 3 ../results_ir_v1/fill_tables.log
echo "all done $(date +%T)"
