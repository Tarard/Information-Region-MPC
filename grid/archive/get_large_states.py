"""Download the grid state files that are too large for the repository (N = 200 and 500).

GitHub rejects files above 100 MB, so the twelve `states.npz` files of N = 200 and 500 are assets of the GitHub
release `grid-states-v1`. This script downloads them into the `NS_ROOT` layout of this folder and checks their
SHA-256 against `SHA256SUMS.txt` of the same release.

Usage (from the repository root or from this folder):
    python grid/archive/get_large_states.py            # N = 200 and 500
    python grid/archive/get_large_states.py 200        # one size
After the download, `NS_ROOT=grid/archive python grid/ir_trial.py N` runs on the archived states.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

REPO = "Tarard/Information-Region-MPC"
TAG = "grid-states-v1"
BASE = f"https://github.com/{REPO}/releases/download/{TAG}"
HERE = Path(__file__).resolve().parent
SEEDS = (1, 2, 3, 4, 101, 102)


def asset_name(n, seed):
    return f"N{n}_central_{seed}_states.npz"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(sizes):
    with urllib.request.urlopen(f"{BASE}/SHA256SUMS.txt") as r:
        sums = dict(line.split()[::-1] for line in r.read().decode().splitlines() if line.strip())
    for n in sizes:
        for seed in SEEDS:
            name = asset_name(n, seed)
            out = HERE / f"work/N{n}/runs/central_{seed}/states.npz"
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists() and sha256(out) == sums[name]:
                print("ok (present)", out.relative_to(HERE))
                continue
            print("download", name, flush=True)
            urllib.request.urlretrieve(f"{BASE}/{name}", out)
            if sha256(out) != sums[name]:
                out.unlink()
                raise SystemExit(f"SHA-256 mismatch for {name}")
            print("ok", out.relative_to(HERE))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [200, 500])
