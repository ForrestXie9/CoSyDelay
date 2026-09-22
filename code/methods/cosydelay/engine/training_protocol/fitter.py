"""V16 adapter that omits the superseded numerical zero-green probe.

The retained all-log fitter historically evaluated a diagnostic condition at
``GR_phase=1e-6`` after every restart.  V15 already prevents that diagnostic
from choosing a restart, but computing and serializing it would still expose a
10,000-second threshold that is not part of the seven-rule manuscript method.

This module changes neither the objective nor the optimizer.  It temporarily
substitutes only the approach-worker task, calls the exact retained solver with
the probe replaced by a no-op, and restores the shared function immediately.
The active V15 confirmation runs in a separate process and is untouched.
"""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import Any, Dict, Iterator, Mapping

from methods.cosydelay.engine.optimizer_conditioning import (
    all_log_fitter as retained,
)


_ORIGINAL_SOLVER = retained._solve_approach_all_log
_ORIGINAL_MINIMIZE = retained.minimize
_PATCH_LOCK = threading.RLock()
_LEGACY_DIAGNOSTIC_KEYS = frozenset(
    {
        "r9_probe_pass",
        "r9_probe_delays_seconds",
        "r9_constrained_restart_selection",
        "r9_feasible_restart_count",
        "r9_feasible_solution_available",
    }
)

# These are the L-BFGS-B defaults in the frozen SciPy 1.16.3 environment.
# They were previously inherited implicitly.  Supplying the same values
# explicitly does not alter the optimization, but prevents a future SciPy
# upgrade from silently changing the formal method.
FROZEN_LBFGSB_OPTIONS = {
    "maxcor": 10,
    "ftol": 2.220446049250313e-9,
    "gtol": 1e-5,
    "maxls": 20,
}


def _probe_not_executed(*args, **kwargs) -> tuple[bool, Dict[str, float]]:
    """Satisfy the retained solver interface without evaluating the probe."""
    del args, kwargs
    return True, {}


def _minimize_with_frozen_lbfgsb_defaults(*args, **kwargs):
    """Freeze the formerly implicit SciPy L-BFGS-B stopping options."""
    method = str(kwargs.get("method", "")).upper()
    if method != "L-BFGS-B":
        return _ORIGINAL_MINIMIZE(*args, **kwargs)
    options = dict(kwargs.get("options") or {})
    for name, value in FROZEN_LBFGSB_OPTIONS.items():
        observed = options.setdefault(name, value)
        if observed != value:
            raise ValueError(
                f"V16 L-BFGS-B option drift for {name}: {observed} != {value}"
            )
    kwargs["options"] = options
    return _ORIGINAL_MINIMIZE(*args, **kwargs)


def scrub_legacy_probe_fields(value: Any) -> Any:
    """Remove only obsolete numerical-R9 diagnostics, recursively in place."""
    if isinstance(value, dict):
        for key in tuple(value):
            if str(key) in _LEGACY_DIAGNOSTIC_KEYS:
                value.pop(key, None)
            else:
                scrub_legacy_probe_fields(value[key])
    elif isinstance(value, list):
        for item in value:
            scrub_legacy_probe_fields(item)
    return value


def _solve_approach_without_legacy_probe(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run the retained all-log solver while omitting its unused probe."""
    local_payload = dict(payload)
    local_payload["r9_constrained_restart_selection"] = False
    previous_probe = retained._r9_probe_restart
    previous_minimize = retained.minimize
    retained._r9_probe_restart = _probe_not_executed
    retained.minimize = _minimize_with_frozen_lbfgsb_defaults
    try:
        result = _ORIGINAL_SOLVER(local_payload)
    finally:
        retained.minimize = previous_minimize
        retained._r9_probe_restart = previous_probe
    scrub_legacy_probe_fields(result)
    result["legacy_numeric_r9_probe_executed"] = False
    result["restart_selection"] = "minimum_training_mse"
    result["frozen_lbfgsb_options"] = dict(FROZEN_LBFGSB_OPTIONS)
    return result


@contextmanager
def legacy_numeric_r9_probe_disabled() -> Iterator[None]:
    """Scope the worker substitution to one sequential V16 evaluation."""
    with _PATCH_LOCK:
        previous_solver = retained._solve_approach_all_log
        retained._solve_approach_all_log = _solve_approach_without_legacy_probe
        try:
            yield
        finally:
            retained._solve_approach_all_log = previous_solver
