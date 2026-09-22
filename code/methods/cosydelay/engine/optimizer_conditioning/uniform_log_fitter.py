"""Uniform all-log restart schedule for a controlled optimizer ablation.

This module keeps the validated all-positive-log solver and objective intact,
but replaces the mixed official-local/one-wide start stream with ten starts
drawn by one deterministic scrambled-Sobol rule in log coefficient space.
There is no raw-coordinate restart, parent warm start, or second restart
family hidden in the schedule.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

from methods.cosydelay.engine.optimizer_parallel3.parallel_fitter import (
    _generate_payloads as _generate_official_payloads,
)
from methods.cosydelay.engine.optimizer_conditioning import all_log_fitter as _base


UNIFORM_LOG_RESTART_STRATEGY_ID = "uniform10_scrambled_sobol_all_log_v1"
UNIFORM_LOG_RESTARTS = 10


def _stable_seed(base_seed: int, expression: str, approach: str) -> int:
    digest = hashlib.sha256(
        f"{int(base_seed)}|{expression}|{approach}|uniform10-log".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _uniform_log_points(
    count: int,
    bounds: Sequence[Tuple[float, float]],
    seed: int,
    *,
    unit_low: float = 0.0,
    unit_high: float = 1.0,
) -> np.ndarray:
    if count < 1:
        raise ValueError("uniform log restart count must be positive")
    lower = np.asarray([float(item[0]) for item in bounds], dtype=float)
    upper = np.asarray([float(item[1]) for item in bounds], dtype=float)
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("uniform log bounds must be finite")
    if np.any(lower <= 0.0) or np.any(upper <= lower):
        raise ValueError("uniform log bounds must be strictly positive and ordered")
    if not (0.0 <= float(unit_low) < float(unit_high) <= 1.0):
        raise ValueError("Sobol unit interval must satisfy 0 <= low < high <= 1")
    sampler = qmc.Sobol(d=len(bounds), scramble=True, seed=int(seed))
    # Generate the next power-of-two Sobol block, then retain exactly the
    # declared ten points.  The optimizer still executes exactly ``count``
    # restarts, while avoiding the balance warning for non-power-of-two n.
    block_size = 1 << int(np.ceil(np.log2(max(1, int(count)))))
    unit = np.asarray(sampler.random_base2(m=int(np.log2(block_size)))[:count], dtype=float)
    unit = np.clip(unit, 1e-7, 1.0 - 1e-7)
    unit = float(unit_low) + unit * (float(unit_high) - float(unit_low))
    return np.exp(
        np.log(lower)[None, :]
        + unit * (np.log(upper)[None, :] - np.log(lower)[None, :])
    )


def generate_uniform_log_restart_payloads(
    *,
    universal_expr: str,
    lanes: Sequence[str],
    approach_targets: Mapping[str, Any],
    prepared: Mapping[str, Any],
    rng: np.random.Generator,
    n_restarts: int,
    maxiter: int,
    maxfun: int,
    coefficient_bounds_override: Optional[Mapping[str, Tuple[float, float]]],
    warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> tuple[list[str], list[Dict[str, Any]], int]:
    """Build payloads whose ten starts all share one log-Sobol mechanism."""
    if int(n_restarts) != UNIFORM_LOG_RESTARTS:
        raise ValueError(
            f"{UNIFORM_LOG_RESTART_STRATEGY_ID} requires exactly "
            f"{UNIFORM_LOG_RESTARTS} restarts"
        )
    coefficient_names, payloads = _generate_official_payloads(
        universal_expr=universal_expr,
        lanes=list(lanes),
        approach_targets=approach_targets,
        prepared=prepared,
        rng=rng,
        n_restarts=int(n_restarts),
        maxiter=int(maxiter),
        maxfun=int(maxfun),
        coefficient_bounds_override=coefficient_bounds_override,
    )
    base_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    for payload in payloads:
        bounds = [tuple(item) for item in payload["bounds"]]
        payload["initials"] = _uniform_log_points(
            int(n_restarts), bounds,
            _stable_seed(base_seed, str(universal_expr), str(payload["approach"])),
        )
        payload["restart_kinds"] = [
            "uniform_log_sobol"
        ] * int(n_restarts)
        payload["warm_values_inherited"] = 0
    return coefficient_names, payloads, base_seed


def generate_interior_log_restart_payloads(
    *,
    universal_expr: str,
    lanes: Sequence[str],
    approach_targets: Mapping[str, Any],
    prepared: Mapping[str, Any],
    rng: np.random.Generator,
    n_restarts: int,
    maxiter: int,
    maxfun: int,
    coefficient_bounds_override: Optional[Mapping[str, Tuple[float, float]]],
    warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> tuple[list[str], list[Dict[str, Any]], int]:
    """Use the same single log-Sobol mechanism, restricted to its interior."""
    if int(n_restarts) != UNIFORM_LOG_RESTARTS:
        raise ValueError(
            f"interior log restart requires exactly {UNIFORM_LOG_RESTARTS} restarts"
        )
    coefficient_names, payloads = _generate_official_payloads(
        universal_expr=universal_expr,
        lanes=list(lanes),
        approach_targets=approach_targets,
        prepared=prepared,
        rng=rng,
        n_restarts=int(n_restarts),
        maxiter=int(maxiter),
        maxfun=int(maxfun),
        coefficient_bounds_override=coefficient_bounds_override,
    )
    base_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    for payload in payloads:
        payload["initials"] = _uniform_log_points(
            int(n_restarts),
            [tuple(item) for item in payload["bounds"]],
            _stable_seed(base_seed, str(universal_expr), str(payload["approach"])),
            unit_low=0.05,
            unit_high=0.95,
        )
        payload["restart_kinds"] = ["interior_log_sobol"] * int(n_restarts)
        payload["warm_values_inherited"] = 0
    return coefficient_names, payloads, base_seed


def _fit_with_payload_generator(generator, strategy_id: str, *args, **kwargs):
    n_restarts = int(kwargs.get("n_restarts", UNIFORM_LOG_RESTARTS))
    if n_restarts != UNIFORM_LOG_RESTARTS:
        raise ValueError(
            f"{UNIFORM_LOG_RESTART_STRATEGY_ID} requires exactly "
            f"{UNIFORM_LOG_RESTARTS} restarts"
        )
    original = _base.generate_mixed_restart_payloads
    _base.generate_mixed_restart_payloads = generator
    try:
        result = _base.fit_lane_parameters_all_log_parallel(*args, **kwargs)
    finally:
        _base.generate_mixed_restart_payloads = original
    diagnostics = kwargs.get("diagnostics")
    if isinstance(diagnostics, dict):
        diagnostics.update(
            {
                "start_strategy_id": strategy_id,
                "restart_schedule": "ten_uniform_scrambled_sobol_points_in_log_space",
                "restart_parameterizations": ["all_positive_log"] * n_restarts,
                "uniform_restart_schedule": True,
                "parent_warm_start_available": False,
                "parent_warm_parameters_ignored": bool(
                    kwargs.get("warm_parameters") is not None
                ),
            }
        )
    return result


def fit_lane_parameters_uniform_log_parallel(*args, **kwargs):
    """Call the retained all-log solver with the full-range log-Sobol starts."""
    return _fit_with_payload_generator(
        generate_uniform_log_restart_payloads,
        UNIFORM_LOG_RESTART_STRATEGY_ID,
        *args,
        **kwargs,
    )


def fit_lane_parameters_interior_log_parallel(*args, **kwargs):
    """Call the retained all-log solver with interior [0.05, 0.95] starts."""
    return _fit_with_payload_generator(
        generate_interior_log_restart_payloads,
        UNIFORM_LOG_RESTART_STRATEGY_ID + "_interior05",
        *args,
        **kwargs,
    )
