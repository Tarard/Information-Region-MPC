#!/usr/bin/env bash
# Closed-loop OSD runs for Table I: OSD-1, OSD-2, P-OSD with K = 1 and 4 on the 10 test days (8 at a time).
# K = 1 first.
set -u
cd "$(dirname "$0")"
PY="${PY:-python}"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
OUT=../results_osd_v1
mkdir -p $OUT/runs $OUT/logs
osdrun() {
  set -- $1
  name=$(echo "$1" | tr -d -)_K$2_seed$3
  [ -f $OUT/runs/$name/COMPLETE.json ] && { echo "skip $name"; return; }
  rm -rf $OUT/runs/$name
  $PY -B cc_scenario.py --seed $3 --out $OUT/runs/$name --max-time 14400 --controller osd --osd $1 --k $2 > $OUT/logs/$name.log 2>&1
  echo "$name $(date +%T): $(tail -n 1 $OUT/logs/$name.log | cut -c1-200)"
}
export -f osdrun
export PY OUT
echo "start $(date +%T)"
for k in 1 4; do for v in OSD-1 OSD-2 P-OSD; do for s in 101 102 103 104 106 107 108 109 110 111; do echo "$v $k $s"; done; done; done \
  | xargs -P 12 -I{} bash -c 'osdrun "{}"'
echo "done $(date +%T)"
