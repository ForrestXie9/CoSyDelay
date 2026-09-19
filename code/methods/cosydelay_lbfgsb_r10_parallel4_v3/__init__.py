"""Prospective CoSyDelay v3 training-only search components.

This package is intentionally isolated from the retained v2 method.  Importing
it does not patch the production runtime; callers must enter the explicit
context managers exported here.
"""

from .fitter import (
    V3_MAXFUN,
    V3_MAXITER,
    V3_PARALLEL_WORKERS,
    V3_RESTARTS,
    MixedRestartJacobianFitter,
    fit_lane_parameters_mixed_jacobian_parallel,
)
from .physics_audit import audit_fitted_physics
from .policy import V3_POLICY, V3SearchPolicy
from .training_selection import TrainingOnlySelectionController

__all__ = [
    "V3_MAXFUN",
    "V3_MAXITER",
    "V3_PARALLEL_WORKERS",
    "V3_RESTARTS",
    "MixedRestartJacobianFitter",
    "TrainingOnlySelectionController",
    "V3_POLICY",
    "V3SearchPolicy",
    "audit_fitted_physics",
    "fit_lane_parameters_mixed_jacobian_parallel",
]
