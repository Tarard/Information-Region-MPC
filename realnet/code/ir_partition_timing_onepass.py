"""Timing pass of ir_partition.py on all 50 test samples in one run.

One process, nothing else running, one warm-up solve that is not counted, then all 50 samples of `samples_list()`
in a single run. Writes
results_ir_v1/TABLES_partition_timing_onepass50.json (and a .partial.json after every sample).
"""
import json

import ir_partition as P   # sets the BLAS thread variables before NumPy is loaded


def main():
    P._init()
    allp = [tuple(x) for x in P.T.samples_list()]
    assert len(allp) == 50, len(allp)
    P.one(*allp[0], timing=True)                       # warm-up, not counted
    res = []
    for i, (s, t) in enumerate(allp, 1):
        r = P.one(s, t, timing=True)
        r.setdefault("seed", s)
        r.setdefault("t", t)
        res.append(r)
        print(f"timing {i}/{len(allp)} seed {s} t {t}", flush=True)
        (P.T.OUT / "TABLES_partition_timing_onepass50.partial.json").write_text(json.dumps(res, default=float),
                                                                              encoding="utf-8")
    (P.T.OUT / "TABLES_partition_timing_onepass50.json").write_text(json.dumps(res, default=float), encoding="utf-8")
    print("partition timing (50 samples, one pass) done", len(res))


if __name__ == "__main__":
    main()
