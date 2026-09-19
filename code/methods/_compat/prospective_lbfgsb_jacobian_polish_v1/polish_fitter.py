"""Deterministic best-restart polish for the Jacobian L-BFGS-B fitter."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numexpr as ne
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from methods.cosydelay_lbfgsb_r10_parallel3.parallel_fitter import (
    OFFICIAL_MAXFUN,
    OFFICIAL_MAXITER,
    OFFICIAL_PARALLEL_WORKERS,
    OFFICIAL_RESTARTS,
    _generate_payloads,
)
from methods.prospective_lbfgsb_jacobian_v1.jacobian_fitter import (
    _mse_and_gradient,
    _symbolic_derivative_strings,
    fit_lane_parameters_with_jacobian_parallel,
)
from optimization_lane import (
    extract_coefficients,
    prepare_optimization_context,
    unpack_lane_parameters,
)


DEFAULT_POLISH_MAXITER = 200
DEFAULT_POLISH_MAXFUN = 20_000


def _solve_polish_approach(payload: Mapping[str, Any]) -> Dict[str, Any]:
    ne.set_num_threads(1)
    initial = np.asarray(payload["initials"][0], dtype=float)
    bounds = [tuple(item) for item in payload["bounds"]]
    initial_objective, initial_gradient = _mse_and_gradient(
        initial, payload
    )
    started = time.perf_counter()
    result = minimize(
        _mse_and_gradient,
        initial,
        args=(payload,),
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={
            "maxiter": int(payload["maxiter"]),
            "maxfun": int(payload["maxfun"]),
            "disp": False,
        },
    )
    wall_seconds = float(time.perf_counter() - started)
    candidate_objective = float(result.fun)
    accepted = bool(
        np.isfinite(candidate_objective)
        and candidate_objective < initial_objective
    )
    selected_values = (
        np.asarray(result.x, dtype=float) if accepted else initial
    )
    selected_objective = (
        candidate_objective if accepted else float(initial_objective)
    )
    return {
        "approach": str(payload["approach"]),
        "block_lanes": list(payload["block_lanes"]),
        "coefficient_names": list(payload["coefficient_names"]),
        "values": selected_values,
        "initial_objective": float(initial_objective),
        "initial_gradient_inf_norm": float(
            np.linalg.norm(initial_gradient, ord=np.inf)
        ),
        "candidate_objective": candidate_objective,
        "selected_objective": float(selected_objective),
        "objective_improvement": float(
            initial_objective - selected_objective
        ),
        "accepted": accepted,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(getattr(result, "nit", 0)),
        "function_evaluations": int(getattr(result, "nfev", 0)),
        "gradient_evaluations": int(getattr(result, "njev", 0)),
        "final_gradient_inf_norm": float(
            np.linalg.norm(
                np.asarray(getattr(result, "jac", []), dtype=float),
                ord=np.inf,
            )
        ),
        "wall_seconds": wall_seconds,
    }


def polish_lane_parameters_parallel(
    universal_expr: str,
    df: pd.DataFrame,
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    approach_targets: Mapping[str, pd.Series],
    intersection_id: int,
    lane_parameters: Mapping[str, Mapping[str, float]],
    prepared_context: Optional[Mapping[str, Any]] = None,
    diagnostics: Optional[Dict[str, Any]] = None,
    coefficient_bounds_override: Optional[
        Mapping[str, Tuple[float, float]]
    ] = None,
    *,
    polish_maxiter: int = DEFAULT_POLISH_MAXITER,
    polish_maxfun: int = DEFAULT_POLISH_MAXFUN,
    parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Dict[str, float]]:
    """Polish one existing best vector per approach without consuming RNG."""
    if min(polish_maxiter, polish_maxfun, parallel_workers) < 1:
        raise ValueError("polish budgets and workers must be positive")
    coefficient_names = extract_coefficients(universal_expr)
    if not coefficient_names:
        return {str(lane): {} for lane in lanes}
    prepared = prepared_context or prepare_optimization_context(
        df,
        list(lanes),
        dict(lane_to_approach),
        int(intersection_id),
    )
    derivative_expressions = _symbolic_derivative_strings(
        universal_expr, tuple(coefficient_names)
    )
    # A private deterministic generator is used only to reuse the frozen
    # payload/context construction. Its generated start is replaced below and
    # the caller's RNG is never touched.
    coefficient_names, payloads = _generate_payloads(
        universal_expr=universal_expr,
        lanes=list(lanes),
        approach_targets=approach_targets,
        prepared=prepared,
        rng=np.random.default_rng(0xC05D),
        n_restarts=1,
        maxiter=int(polish_maxiter),
        maxfun=int(polish_maxfun),
        coefficient_bounds_override=coefficient_bounds_override,
    )
    for payload in payloads:
        block_lanes = list(payload["block_lanes"])
        missing = [
            (lane, name)
            for lane in block_lanes
            for name in coefficient_names
            if lane not in lane_parameters
            or name not in lane_parameters[lane]
        ]
        if missing:
            raise KeyError(f"Missing starting coefficients: {missing}")
        payload["initials"] = np.asarray(
            [
                [
                    float(lane_parameters[lane][name])
                    for lane in block_lanes
                    for name in coefficient_names
                ]
            ],
            dtype=float,
        )
        payload["derivative_expressions"] = derivative_expressions

    started = time.perf_counter()
    workers = min(int(parallel_workers), len(payloads))
    owns_executor = executor is None
    active_executor = executor
    if workers > 1 and active_executor is None:
        active_executor = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
    try:
        if workers == 1:
            records = [
                _solve_polish_approach(payload) for payload in payloads
            ]
        else:
            assert active_executor is not None
            records = list(
                active_executor.map(_solve_polish_approach, payloads)
            )
    finally:
        if owns_executor and active_executor is not None:
            active_executor.shutdown(wait=True)
    wall_seconds = float(time.perf_counter() - started)

    polished = {
        str(lane): {
            str(name): float(value)
            for name, value in parameters.items()
        }
        for lane, parameters in lane_parameters.items()
    }
    for record in records:
        polished.update(
            unpack_lane_parameters(
                np.asarray(record["values"], dtype=float),
                list(record["block_lanes"]),
                coefficient_names,
            )
        )
    if diagnostics is not None:
        before = float(
            sum(record["initial_objective"] for record in records)
        )
        after = float(
            sum(record["selected_objective"] for record in records)
        )
        diagnostics.update(
            {
                "solver": "L-BFGS-B",
                "jacobian": "analytic_sympy_numexpr",
                "polish_restarts": 0,
                "polish_maxiter": int(polish_maxiter),
                "polish_maxfun": int(polish_maxfun),
                "parallel_workers": workers,
                "rng_values_consumed": 0,
                "objective_sum_before": before,
                "objective_sum_after": after,
                "objective_improvement": float(before - after),
                "outer_wall_seconds": wall_seconds,
                "accepted_approaches": int(
                    sum(record["accepted"] for record in records)
                ),
                "total_iterations": int(
                    sum(record["iterations"] for record in records)
                ),
                "total_function_evaluations": int(
                    sum(
                        record["function_evaluations"]
                        for record in records
                    )
                ),
                "approaches": [
                    {
                        key: value
                        for key, value in record.items()
                        if key
                        not in {
                            "values",
                            "block_lanes",
                            "coefficient_names",
                        }
                    }
                    for record in records
                ],
            }
        )
    return polished


def fit_lane_parameters_with_jacobian_polish_parallel(
    universal_expr: str,
    df: pd.DataFrame,
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    approach_targets: Mapping[str, pd.Series],
    intersection_id: int,
    verbose: bool = False,
    prepared_context: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = OFFICIAL_RESTARTS,
    diagnostics: Optional[Dict[str, Any]] = None,
    coefficient_bounds_override: Optional[
        Mapping[str, Tuple[float, float]]
    ] = None,
    *,
    parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
    base_maxiter: int = OFFICIAL_MAXITER,
    base_maxfun: int = OFFICIAL_MAXFUN,
    polish_maxiter: int = DEFAULT_POLISH_MAXITER,
    polish_maxfun: int = DEFAULT_POLISH_MAXFUN,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Dict[str, float]]:
    """Run the validated Jacobian r10 fit followed by deterministic polish."""
    base_diagnostics: dict[str, Any] = {}
    started = time.perf_counter()
    base_parameters = fit_lane_parameters_with_jacobian_parallel(
        universal_expr=universal_expr,
        df=df,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        approach_targets=approach_targets,
        intersection_id=intersection_id,
        verbose=verbose,
        prepared_context=prepared_context,
        rng=rng,
        n_restarts=n_restarts,
        diagnostics=base_diagnostics,
        coefficient_bounds_override=coefficient_bounds_override,
        parallel_workers=parallel_workers,
        maxiter=base_maxiter,
        maxfun=base_maxfun,
        executor=executor,
    )
    polish_diagnostics: dict[str, Any] = {}
    polished = polish_lane_parameters_parallel(
        universal_expr=universal_expr,
        df=df,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        approach_targets=approach_targets,
        intersection_id=intersection_id,
        lane_parameters=base_parameters,
        prepared_context=prepared_context,
        diagnostics=polish_diagnostics,
        coefficient_bounds_override=coefficient_bounds_override,
        polish_maxiter=polish_maxiter,
        polish_maxfun=polish_maxfun,
        parallel_workers=parallel_workers,
        executor=executor,
    )
    if diagnostics is not None:
        diagnostics.update(
            {
                "solver": "L-BFGS-B",
                "n_restarts": int(n_restarts),
                "base": base_diagnostics,
                "polish": polish_diagnostics,
                "objective_sum": polish_diagnostics[
                    "objective_sum_after"
                ],
                "total_wall_seconds": float(
                    time.perf_counter() - started
                ),
            }
        )
    return polished


class ParallelApproachPolisher:
    """Persistent process pool for polishing archived parameter vectors."""

    def __init__(
        self,
        *,
        parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
        polish_maxfun: int = DEFAULT_POLISH_MAXFUN,
    ) -> None:
        self.parallel_workers = int(parallel_workers)
        self.polish_maxfun = int(polish_maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def __call__(self, *, polish_maxiter: int, **kwargs):
        if self._executor is None:
            raise RuntimeError("ParallelApproachPolisher is closed")
        return polish_lane_parameters_parallel(
            polish_maxiter=polish_maxiter,
            polish_maxfun=self.polish_maxfun,
            parallel_workers=self.parallel_workers,
            executor=self._executor,
            **kwargs,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "ParallelApproachPolisher":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class JacobianPolishedApproachFitter:
    """Drop-in persistent r10 Jacobian fitter plus deterministic polish."""

    def __init__(
        self,
        *,
        parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
        base_maxiter: int = OFFICIAL_MAXITER,
        base_maxfun: int = OFFICIAL_MAXFUN,
        polish_maxiter: int = DEFAULT_POLISH_MAXITER,
        polish_maxfun: int = DEFAULT_POLISH_MAXFUN,
    ) -> None:
        self.parallel_workers = int(parallel_workers)
        self.base_maxiter = int(base_maxiter)
        self.base_maxfun = int(base_maxfun)
        self.polish_maxiter = int(polish_maxiter)
        self.polish_maxfun = int(polish_maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def __call__(self, **kwargs):
        if self._executor is None:
            raise RuntimeError("JacobianPolishedApproachFitter is closed")
        return fit_lane_parameters_with_jacobian_polish_parallel(
            parallel_workers=self.parallel_workers,
            base_maxiter=self.base_maxiter,
            base_maxfun=self.base_maxfun,
            polish_maxiter=self.polish_maxiter,
            polish_maxfun=self.polish_maxfun,
            executor=self._executor,
            **kwargs,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "JacobianPolishedApproachFitter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
