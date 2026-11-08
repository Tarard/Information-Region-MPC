"""Read-only path imports of the frozen v3 modules. Nothing is written to the v3 folder."""

from pathlib import Path
import sys

sys.dont_write_bytecode = True

from cc_design import V3  # noqa: E402

V3_CODE = V3 / "code"
if str(V3_CODE) not in sys.path:
    sys.path.append(str(V3_CODE))

import aligned_model as v3_model        # noqa: E402  Problem: sparse QP records
import causal_observation               # noqa: E402  EntryObserver: causal entry observation
import native_executor                  # noqa: E402  loaded by v3 scenario only; not used here
import scenario as v3_scenario          # noqa: E402  read/save/sha/plain and the v3 trip audits
import strict_qp                        # noqa: E402  Clarabel with the v3 closed-loop acceptance

from cc_design import CLOSED_LOOP_TOLERANCE  # noqa: E402

# main.tex Sec. VI: QPs are accepted at a KKT residual of 1e-5 in closed loop and 1e-7 in open loop. Closed-loop
# runs use this default; offline and open-loop scripts set strict_qp.TOLERANCE = OPEN_LOOP_TOLERANCE themselves.
strict_qp.TOLERANCE = CLOSED_LOOP_TOLERANCE


def _no_osqp(matrices, warm):
    raise RuntimeError("OSQP fall-back disabled: a QP not accepted from Clarabel is a failure")


# Decided 2026-10-30: no OSQP fall-back for any method (centralized, IR-MPC, OSD), open or closed loop. A QP that
# Clarabel does not solve to the KKT acceptance is a failure and follows the failure rule of its method.
strict_qp._osqp = _no_osqp

import local_controller                 # noqa: E402  IR-MPC local QP

_compile_local = local_controller.compile_local


def _compile_local_sparse(controller):
    """Fm selects one retained row per boundary component (entries 0 or 1); the least-squares solve in v12_boundary_v2
    leaves rounding noise below 1e-12 in every other entry, which made Fm about 50% dense. Dropping that noise gives
    the same QP (largest green-time change 1.2e-9 s) and halves the local solve time (decided 2026-10-31)."""
    if "P" not in controller:
        F = controller["reduction"]["Fm"].tocsc(copy=True)
        F.data[abs(F.data) < 1e-12] = 0.0
        F.eliminate_zeros()
        controller["reduction"]["Fm"] = F
    _compile_local(controller)


local_controller.compile_local = _compile_local_sparse

MODULES =(v3_model, causal_observation, native_executor, v3_scenario, strict_qp)
FILES = tuple(sorted({V3_CODE / f"{m.__name__}.py" for m in MODULES}, key=str))
for module in MODULES:
    if V3_CODE.resolve() not in Path(module.__file__).resolve().parents:
        raise ImportError(f"{module.__name__} was not loaded from the frozen v3 code folder.")
