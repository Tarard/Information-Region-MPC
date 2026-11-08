"""Solver settings of the v2 experiment (PLAN_numeric_scale_v2.md section 7), as in LYG: Clarabel with the strict
settings of strict_qp._clarabel (tol_gap_abs 1e-10, tol_feas 1e-11, tol_ktratio 1e-10, iterative refinement 1e-13,
one thread, qdldl, at most 300 iterations), no OSQP fall-back: a solve that Clarabel does not get accepted fails.
Acceptance by the five absolute KKT residuals: 1e-5 in closed loop (history and test days), 1e-7 offline and in the
open-loop samples (accept_offline())."""
import ns_design as d
import strict_qp


def install():
    strict_qp.FALLBACK = False
    strict_qp.TOLERANCE = d.TOLERANCE


def accept_offline():
    strict_qp.TOLERANCE = d.OFFLINE_TOLERANCE
