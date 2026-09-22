"""Prospective analytic-Jacobian implementation of the retained fitter.

The solver, bounds, restart points, objective, and approach-level parallelism
match ``cosydelay_lbfgsb_r10_parallel3``. The only numerical change is that
SciPy receives the exact symbolic derivative of the same summed-MSE objective
instead of estimating it with forward finite differences.
"""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numexpr as ne
import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import minimize

from expression_rules import parse_symbolic_expression
from methods.cosydelay.engine.optimizer_parallel3.parallel_fitter import (
    OFFICIAL_MAXFUN,
    OFFICIAL_MAXITER,
    OFFICIAL_PARALLEL_WORKERS,
    OFFICIAL_RESTARTS,
    _generate_payloads,
)
from optimization_lane import (
    extract_coefficients,
    prepare_optimization_context,
    unpack_lane_parameters,
)


PENALTY_OBJECTIVE = 1e12


@lru_cache(maxsize=256)
def _symbolic_derivative_strings(
    expression: str,
    coefficient_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Return derivatives in the fitter's coefficient order."""
    parsed, symbols = parse_symbolic_expression(expression)
    missing = [name for name in coefficient_names if name not in symbols]
    if missing:
        raise ValueError(f"Missing symbolic coefficients: {missing}")
    return tuple(
        str(sp.diff(parsed, symbols[name])) for name in coefficient_names
    )


def _row_array(value: Any, n_rows: int) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim == 0:
        return np.full(n_rows, float(result), dtype=float)
    if result.shape != (n_rows,):
        return np.asarray(np.broadcast_to(result, (n_rows,)), dtype=float)
    return result


def _penalty(values: np.ndarray) -> tuple[float, np.ndarray]:
    return PENALTY_OBJECTIVE, np.zeros_like(values, dtype=float)


def _mse_and_gradient(
    values: np.ndarray,
    payload: Mapping[str, Any],
) -> tuple[float, np.ndarray]:
    """Evaluate the unchanged approach MSE and its analytic gradient."""
    values = np.asarray(values, dtype=float)
    coefficient_names = list(payload["coefficient_names"])
    block_lanes = list(payload["block_lanes"])
    lane_parameters = unpack_lane_parameters(
        values, block_lanes, coefficient_names
    )
    expression = str(payload["expression"])
    derivative_expressions = list(payload["derivative_expressions"])
    target_values = np.asarray(payload["target_values"], dtype=float)
    valid = np.asarray(payload["valid"], dtype=bool)
    n_rows = len(target_values)
    prediction = np.zeros(n_rows, dtype=float)
    sensitivities = np.zeros((len(values), n_rows), dtype=float)

    try:
        for lane_index, lane in enumerate(block_lanes):
            weight = np.asarray(payload["weights"][lane], dtype=float)
            context = {
                **payload["lane_contexts"][lane],
                **lane_parameters[lane],
            }
            raw = _row_array(
                ne.evaluate(expression, local_dict=context), n_rows
            )
            if np.any(~np.isfinite(raw)):
                return _penalty(values)
            active = raw > 0.0
            prediction += weight * np.maximum(raw, 0.0)
            start = lane_index * len(coefficient_names)
            for coefficient_index, derivative_expression in enumerate(
                derivative_expressions
            ):
                derivative = _row_array(
                    ne.evaluate(
                        derivative_expression,
                        local_dict=context,
                    ),
                    n_rows,
                )
                # Zero lane flow can produce removable terms such as
                # 0**a * log(0). They have zero aggregate contribution when
                # the lane weight is zero. Non-finite derivatives with a
                # positive weight are treated as an invalid optimizer point.
                relevant = active & (np.abs(weight) > 0.0)
                if np.any((~np.isfinite(derivative)) & relevant):
                    return _penalty(values)
                derivative = np.where(
                    active,
                    np.nan_to_num(
                        derivative,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ),
                    0.0,
                )
                sensitivities[start + coefficient_index] = (
                    weight * derivative
                )
    except Exception:
        return _penalty(values)

    if not np.any(valid):
        return _penalty(values)
    residual = prediction[valid] - target_values[valid]
    if residual.size == 0 or np.any(~np.isfinite(residual)):
        return _penalty(values)
    objective = float(np.mean(residual**2))
    gradient = 2.0 * np.mean(
        sensitivities[:, valid] * residual[np.newaxis, :],
        axis=1,
    )
    if not np.isfinite(objective) or np.any(~np.isfinite(gradient)):
        return _penalty(values)
    return objective, np.asarray(gradient, dtype=float)


def _solve_approach_with_jacobian(
    payload: Mapping[str, Any],
) -> Dict[str, Any]:
    """Solve one approach with the same starts and an analytic Jacobian."""
    ne.set_num_threads(1)
    initials = np.asarray(payload["initials"], dtype=float)
    bounds = [tuple(item) for item in payload["bounds"]]
    maxiter = int(payload["maxiter"])
    maxfun = int(payload["maxfun"])
    records = []
    approach_started = time.perf_counter()

    for restart_index, initial in enumerate(initials):
        started = time.perf_counter()
        result = minimize(
            _mse_and_gradient,
            initial,
            args=(payload,),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={
                "maxiter": maxiter,
                "maxfun": maxfun,
                "disp": False,
            },
        )
        records.append(
            {
                "index": int(restart_index),
                "values": np.asarray(result.x, dtype=float),
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "objective": float(result.fun),
                "iterations": int(getattr(result, "nit", 0)),
                "function_evaluations": int(getattr(result, "nfev", 0)),
                "gradient_evaluations": int(getattr(result, "njev", 0)),
                "gradient_inf_norm": float(
                    np.linalg.norm(
                        np.asarray(getattr(result, "jac", []), dtype=float),
                        ord=np.inf,
                    )
                ),
                "wall_seconds": float(time.perf_counter() - started),
            }
        )

    chosen_index = int(
        np.argmin(
            [
                item["objective"]
                if np.isfinite(item["objective"])
                else np.inf
                for item in records
            ]
        )
    )
    chosen = records[chosen_index]
    return {
        "approach": str(payload["approach"]),
        "block_lanes": list(payload["block_lanes"]),
        "coefficient_names": list(payload["coefficient_names"]),
        "values": np.asarray(chosen["values"], dtype=float),
        "chosen_restart": chosen_index,
        "success": bool(chosen["success"]),
        "status": int(chosen["status"]),
        "message": str(chosen["message"]),
        "objective": float(chosen["objective"]),
        "iterations": int(chosen["iterations"]),
        "function_evaluations": int(chosen["function_evaluations"]),
        "gradient_evaluations": int(chosen["gradient_evaluations"]),
        "gradient_inf_norm": float(chosen["gradient_inf_norm"]),
        "chosen_wall_seconds": float(chosen["wall_seconds"]),
        "total_wall_seconds": float(
            time.perf_counter() - approach_started
        ),
        "fitted_rows": int(payload["fitted_rows"]),
        "zero_flow_rows_excluded": int(
            payload["zero_flow_rows_excluded"]
        ),
        "zero_flow_policy": str(payload["zero_flow_policy"]),
        "total_function_evaluations": int(
            sum(item["function_evaluations"] for item in records)
        ),
        "total_gradient_evaluations": int(
            sum(item["gradient_evaluations"] for item in records)
        ),
        "restarts": [
            {key: value for key, value in item.items() if key != "values"}
            for item in records
        ],
    }


def fit_lane_parameters_with_jacobian_parallel(
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
    maxiter: int = OFFICIAL_MAXITER,
    maxfun: int = OFFICIAL_MAXFUN,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit the retained objective using symbolic analytic derivatives."""
    if min(n_restarts, parallel_workers, maxiter, maxfun) < 1:
        raise ValueError("restarts, workers, and budgets must be positive")
    coefficient_names = extract_coefficients(universal_expr)
    if not coefficient_names:
        return {str(lane): {} for lane in lanes}
    rng = rng or np.random.default_rng()
    prepared = prepared_context or prepare_optimization_context(
        df,
        list(lanes),
        dict(lane_to_approach),
        int(intersection_id),
    )

    build_started = time.perf_counter()
    derivative_expressions = _symbolic_derivative_strings(
        universal_expr, tuple(coefficient_names)
    )
    coefficient_names, payloads = _generate_payloads(
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
    for payload in payloads:
        payload["derivative_expressions"] = derivative_expressions
    symbolic_build_seconds = float(time.perf_counter() - build_started)

    if not payloads:
        if diagnostics is not None:
            diagnostics.update(
                {
                    "solver": "L-BFGS-B",
                    "jacobian": "analytic_sympy_numexpr",
                    "n_restarts": int(n_restarts),
                    "maxiter": int(maxiter),
                    "maxfun": int(maxfun),
                    "parallel_workers": 0,
                    "objective_sum": 0.0,
                    "outer_wall_seconds": 0.0,
                    "symbolic_build_seconds": symbolic_build_seconds,
                    "approaches": [],
                }
            )
        return {}

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
                _solve_approach_with_jacobian(payload)
                for payload in payloads
            ]
        else:
            assert active_executor is not None
            records = list(
                active_executor.map(
                    _solve_approach_with_jacobian,
                    payloads,
                )
            )
    finally:
        if owns_executor and active_executor is not None:
            active_executor.shutdown(wait=True)
    outer_wall_seconds = float(time.perf_counter() - started)

    fitted: Dict[str, Dict[str, float]] = {}
    fit_records = []
    for record in records:
        fitted.update(
            unpack_lane_parameters(
                np.asarray(record["values"], dtype=float),
                list(record["block_lanes"]),
                coefficient_names,
            )
        )
        fit_records.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"values", "block_lanes", "coefficient_names"}
            }
        )
    objective_sum = float(
        sum(float(record["objective"]) for record in records)
    )
    if diagnostics is not None:
        serial_work_seconds = float(
            sum(float(record["total_wall_seconds"]) for record in records)
        )
        diagnostics.update(
            {
                "solver": "L-BFGS-B",
                "jacobian": "analytic_sympy_numexpr",
                "n_restarts": int(n_restarts),
                "maxiter": int(maxiter),
                "maxfun": int(maxfun),
                "parallel_workers": workers,
                "rng_initialization_semantics": (
                    "identical_to_official_random_uniform_r10"
                ),
                "objective_sum": objective_sum,
                "outer_wall_seconds": outer_wall_seconds,
                "symbolic_build_seconds": symbolic_build_seconds,
                "serial_work_seconds": serial_work_seconds,
                "parallel_efficiency_ratio": (
                    serial_work_seconds / outer_wall_seconds
                    if outer_wall_seconds > 0.0
                    else None
                ),
                "total_function_evaluations": int(
                    sum(
                        record["total_function_evaluations"]
                        for record in records
                    )
                ),
                "total_gradient_evaluations": int(
                    sum(
                        record["total_gradient_evaluations"]
                        for record in records
                    )
                ),
                "approaches": fit_records,
            }
        )
    if verbose:
        print(
            f"  Jacobian L-BFGS-B: approaches={len(records)}, "
            f"restarts={n_restarts}, objective={objective_sum:.4f}, "
            f"wall={outer_wall_seconds:.2f}s"
        )
    return fitted


class JacobianParallelApproachFitter:
    """Persistent three-process callable for repeated candidate fitting."""

    def __init__(
        self,
        *,
        parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
        maxiter: int = OFFICIAL_MAXITER,
        maxfun: int = OFFICIAL_MAXFUN,
    ) -> None:
        if min(parallel_workers, maxiter, maxfun) < 1:
            raise ValueError("workers and budgets must be positive")
        self.parallel_workers = int(parallel_workers)
        self.maxiter = int(maxiter)
        self.maxfun = int(maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def __call__(
        self,
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
    ) -> Dict[str, Dict[str, float]]:
        if self._executor is None:
            raise RuntimeError(
                "JacobianParallelApproachFitter is closed"
            )
        return fit_lane_parameters_with_jacobian_parallel(
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
            diagnostics=diagnostics,
            coefficient_bounds_override=coefficient_bounds_override,
            parallel_workers=self.parallel_workers,
            maxiter=self.maxiter,
            maxfun=self.maxfun,
            executor=self._executor,
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "JacobianParallelApproachFitter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
