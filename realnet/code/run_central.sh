#!/usr/bin/env bash
# Centralized MPC on the 20 history days (1-20) and the 10 test days (101-104, 106-111; 77 and 105 were dropped,
# see EXPERIMENT_NOTES.md).
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p ../results_v1/runs ../results_v1/logs
central() {
  [ -f ../results_v1/runs/mix_seed$1/COMPLETE.json ] && { echo "skip $1"; return; }
  rm -rf ../results_v1/runs/mix_seed$1
  $PY -B cc_scenario.py --seed $1 --out ../results_v1/runs/mix_seed$1 --max-time 14400 > ../results_v1/logs/mix_seed$1.log 2>&1
  echo "central $1 $(date +%T): $(tail -n 1 ../results_v1/logs/mix_seed$1.log | cut -c1-220)"
}
export -f central
export PY
echo "central $(date +%T)"
printf '%s\n' ${DAYS:-$(seq 1 20) 101 102 103 104 106 107 108 109 110 111} | xargs -P ${PAR:-6} -I{} bash -c 'central {}'
echo "central done $(date +%T)"
