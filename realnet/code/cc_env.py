"""Identity of the execution environment (DESIGN.md Sec. 7). Starts no SUMO process.
Import after cc_design.enforce_threads()."""

from hashlib import sha256
import os
from pathlib import Path
import sys

import cc_design as d

d.enforce_threads()

import clarabel  # noqa: E402
import numpy  # noqa: E402
import osqp  # noqa: E402
import scipy  # noqa: E402
import traci  # noqa: E402
import traci.constants as tc  # noqa: E402


def tree_sha256(folder):
    digest = sha256()
    for path in sorted(Path(folder).rglob("*.py")):
        digest.update(path.relative_to(folder).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def identity():
    folder = Path(traci.__file__).resolve().parent
    return {"python_version": sys.version, "python_executable": str(Path(sys.executable).resolve()),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "clarabel": clarabel.__version__,
            "osqp": osqp.__version__, "traci_client_api_version": int(tc.TRACI_VERSION),
            "traci_folder": str(folder), "traci_tree_sha256": tree_sha256(folder),
            "sumo_binary": str(d.SUMO), "sumo_sha256": sha256(d.SUMO.read_bytes()).hexdigest(),
            "threads": {k: os.environ.get(k) for k in d.THREADS}, "clarabel_max_threads": 1}


def differences(expected, actual):
    return sorted(k for k in set(expected) | set(actual) if expected.get(k) != actual.get(k))
