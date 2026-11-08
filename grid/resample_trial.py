"""Replace a failed sampled construction of an IR-MPC trial (user, 2026-11-01) and evaluate again.
For every failed target, the next intersection with the same stage count that is not yet sampled (random order,
seed 1) is constructed with the payload of the trial; the evaluation at TSTAR is then repeated with all controllers.
The first result is kept as IR_TRIAL_N{N}_first.json; IR_TRIAL_N{N}.json records the replacements.
Usage: resample_trial.py N [GB]
"""
import gzip
import json
import pickle
import shutil
import sys

import ir_trial as it
from ir_trial import d, grid_net, nc, no, np, ns_ir  # noqa: F401  (same settings as the trial)


def main(n, gb):
    root = d.ROOT
    res = json.loads((root / f"IR_TRIAL_N{n}.json").read_text(encoding="utf-8"))
    net = grid_net.load(n)
    labels = no.labels_of(net)
    tls = sorted(net["signals"])
    payload = root / f"trial_payload_N{n}.pkl"
    with gzip.open(root / f"trial_bank_N{n}.pkl.gz", "rb") as h:
        bank = pickle.load(h)
    ctrl, pbar = bank["controllers"], bank["pbar"]
    failed = dict(res["construction"]["failed"])
    sampled = set(ctrl) | set(failed)
    stages = {m: net["signals"][m]["stage_count"] for m in tls}
    rng = np.random.default_rng(1)
    replacements, extra_seconds = [], []
    for bad in sorted(failed):
        pool = [m for m in tls if stages[m] == stages[bad] and m not in sampled]
        for cand in rng.permutation(pool)[:5]:
            sampled.add(cand)
            built = it.construct_all(n, [str(cand)], 1, gb, payload)
            c, status, seconds = built[str(cand)]
            extra_seconds.append(seconds)
            replacements.append({"failed": bad, "candidate": str(cand), "status": status, "seconds": seconds})
            if c is not None:
                ctrl[str(cand)] = c
                break
            failed[str(cand)] = status
    shutil.copy(root / f"IR_TRIAL_N{n}.json", root / f"IR_TRIAL_N{n}_first.json")
    shutil.copy(root / f"trial_bank_N{n}.pkl.gz", root / f"trial_bank_N{n}_first.pkl.gz")
    with gzip.open(root / f"trial_bank_N{n}.pkl.gz", "wb") as h:
        pickle.dump({"pbar": pbar, "controllers": ctrl}, h)
    con = res["construction"]
    con.update(targets=con["targets"] + len(replacements), constructed=len(ctrl), failed=failed,
               replacements=replacements)
    res["local_variables"] = {"median": float(np.median([len(c["reduction"]["core"]) + c["boundary_dimension"]
                                                         for c in ctrl.values()]))}
    w = d.size_dir(n)
    hist, test = it.states(w, it.HIST), it.states(w, it.TEST)
    res.pop("days_without_state_at_tstar", None)
    res["evaluation"] = it.evaluate(n, net, labels, hist, test, ctrl, pbar, tls, res)
    (root / f"IR_TRIAL_N{n}.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(f"N={n}: replacements {json.dumps(replacements)}; constructed {len(ctrl)}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]), float(sys.argv[2]) if len(sys.argv) > 2 else 12.0)
