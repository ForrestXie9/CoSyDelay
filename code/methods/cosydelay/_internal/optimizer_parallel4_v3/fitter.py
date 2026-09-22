"""Analytic-Jacobian L-BFGS-B with deterministic mixed restarts.

The objective, coefficient bounds, approach decomposition, restart count, and
solver remain those of the retained method.  The changes are limited to:

* the retained local restart sequence, with one slot reserved for wide Sobol;
* parent inheritance replacing only one local slot when available; and
* the already validated analytic objective Jacobian.

Restarts remain serial within each approach and approaches remain the only
parallel unit, capped at four workers.
"""

from __future__ import annotations

import hashlib
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc
import numexpr as ne

from expression_validation_lane import (
    ZERO_GREEN_LOW_PROBE_GREEN,
    ZERO_GREEN_MIN_DELAY_SECONDS,
)

from methods.cosydelay._internal.optimizer_parallel4.parallel_fitter import (
    OFFICIAL_MAXFUN as V3_MAXFUN,
    OFFICIAL_MAXITER as V3_MAXITER,
    OFFICIAL_PARALLEL_WORKERS as V3_PARALLEL_WORKERS,
    OFFICIAL_RESTARTS as V3_RESTARTS,
)
from methods.cosydelay._internal.optimizer_parallel3.parallel_fitter import (
    _generate_payloads as _generate_official_payloads,
)
from methods.cosydelay._internal.optimizer_jacobian.jacobian_fitter import (
    _mse_and_gradient,
    _symbolic_derivative_strings,
)
from optimization_lane import (
    extract_coefficients,
    prepare_optimization_context,
    unpack_lane_parameters,
)


START_STRATEGY_ID = "official_local9_raw_or_parent_official8_raw_plus_rolewide1_log_2026_08_05_v6"
LOG_BOUND_RATIO_THRESHOLD = 1_000.0
R9_PROBE_FLOW = 1.0
R9_PROBE_CYCLE = 120.0


def _stable_seed(base_seed: int, expression: str, approach: str, mode: str) -> int:
    digest = hashlib.sha256(
        f"{base_seed}|{expression}|{approach}|{mode}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little", signed=False)


def _reference_value(bound: Tuple[float, float]) -> float:
    lower, upper = map(float, bound)
    if lower > 0.0 and upper > 0.0:
        return float(math.sqrt(lower * upper))
    return float((lower + upper) / 2.0)


def _map_role_aware_points(
    points: np.ndarray,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    if np.any(lower <= 0.0):
        raise ValueError("role-aware restarts require strictly positive bounds")
    values = lower + points * (upper - lower)
    log_mask = (upper / lower) >= LOG_BOUND_RATIO_THRESHOLD
    if np.any(log_mask):
        values[:, log_mask] = np.exp(
            np.log(lower[log_mask])
            + points[:, log_mask]
            * (np.log(upper[log_mask]) - np.log(lower[log_mask]))
        )
    return values


def _wide_bound_mask(bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    return (upper / lower) >= LOG_BOUND_RATIO_THRESHOLD


def _to_search_space(values: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[log_mask] = np.log(result[log_mask])
    return result


def _from_search_space(values: np.ndarray, log_mask: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float).copy()
    result[log_mask] = np.exp(result[log_mask])
    return result


def _mse_and_selective_log_gradient(
    search_values: np.ndarray,
    payload: Mapping[str, Any],
) -> tuple[float, np.ndarray]:
    log_mask = np.asarray(payload["log_parameter_mask"], dtype=bool)
    raw_values = _from_search_space(search_values, log_mask)
    objective, raw_gradient = _mse_and_gradient_quiet(raw_values, payload)
    search_gradient = np.asarray(raw_gradient, dtype=float).copy()
    search_gradient[log_mask] *= raw_values[log_mask]
    return float(objective), search_gradient


def _mse_and_gradient_quiet(
    raw_values: np.ndarray,
    payload: Mapping[str, Any],
) -> tuple[float, np.ndarray]:
    """Suppress expected overflow probes without changing their inf result."""
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        return _mse_and_gradient(raw_values, payload)


def _r9_probe_restart(
    raw_values: np.ndarray,
    payload: Mapping[str, Any],
) -> tuple[bool, Dict[str, float]]:
    """Check the retained numerical R9 rule for one approach restart."""
    block_lanes = list(payload["block_lanes"])
    coefficient_names = list(payload["coefficient_names"])
    lane_parameters = unpack_lane_parameters(
        np.asarray(raw_values, dtype=float), block_lanes, coefficient_names
    )
    delays: Dict[str, float] = {}
    passed = True
    for lane in block_lanes:
        local = {
            "flow_lane": np.asarray([R9_PROBE_FLOW], dtype=float),
            "GR_phase": np.asarray([ZERO_GREEN_LOW_PROBE_GREEN], dtype=float),
            "Cycle_Time": np.asarray([R9_PROBE_CYCLE], dtype=float),
            **{
                name: float(lane_parameters[lane][name])
                for name in coefficient_names
            },
        }
        try:
            with np.errstate(all="ignore"):
                value = float(
                    np.asarray(
                        ne.evaluate(str(payload["expression"]), local_dict=local),
                        dtype=float,
                    ).reshape(-1)[0]
                )
        except Exception:
            value = float("nan")
        delays[str(lane)] = value
        lane_pass = bool(
            np.isposinf(value)
            or (np.isfinite(value) and value > ZERO_GREEN_MIN_DELAY_SECONDS)
        )
        passed = passed and lane_pass
    return passed, delays


def _solve_approach_selective_log(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Use raw coordinates locally and log coordinates for the wide start."""
    ne.set_num_threads(1)
    initials = np.asarray(payload["initials"], dtype=float)
    bounds = [tuple(item) for item in payload["bounds"]]
    log_mask = _wide_bound_mask(bounds)
    lower = np.asarray([item[0] for item in bounds], dtype=float)
    upper = np.asarray([item[1] for item in bounds], dtype=float)
    search_lower = _to_search_space(lower, log_mask)
    search_upper = _to_search_space(upper, log_mask)
    search_bounds = list(zip(search_lower.tolist(), search_upper.tolist()))
    working_payload = dict(payload)
    working_payload["log_parameter_mask"] = log_mask
    records = []
    approach_started = time.perf_counter()
    restart_kinds = list(payload["restart_kinds"])
    for restart_index, initial in enumerate(initials):
        start_kind = restart_kinds[restart_index]
        use_selective_log = start_kind == "sobol_role_wide"
        if use_selective_log:
            objective_function = _mse_and_selective_log_gradient
            optimizer_initial = _to_search_space(initial, log_mask)
            optimizer_bounds = search_bounds
        else:
            objective_function = _mse_and_gradient_quiet
            optimizer_initial = np.asarray(initial, dtype=float)
            optimizer_bounds = bounds
        started = time.perf_counter()
        result = minimize(
            objective_function,
            optimizer_initial,
            args=(working_payload,),
            method="L-BFGS-B",
            jac=True,
            bounds=optimizer_bounds,
            options={
                "maxiter": int(payload["maxiter"]),
                "maxfun": int(payload["maxfun"]),
                "disp": False,
            },
        )
        raw_values = (
            _from_search_space(result.x, log_mask)
            if use_selective_log
            else np.asarray(result.x, dtype=float)
        )
        r9_probe_pass, r9_probe_delays = _r9_probe_restart(
            raw_values, working_payload
        )
        records.append(
            {
                "index": int(restart_index),
                "parameterization": (
                    "selective_log" if use_selective_log else "raw"
                ),
                "values": raw_values,
                "success": bool(result.success),
                "status": int(result.status),
                "message": str(result.message),
                "objective": float(result.fun),
                "iterations": int(getattr(result, "nit", 0)),
                "function_evaluations": int(getattr(result, "nfev", 0)),
                "gradient_evaluations": int(getattr(result, "njev", 0)),
                "gradient_inf_norm_optimization_space": float(
                    np.linalg.norm(
                        np.asarray(getattr(result, "jac", []), dtype=float), ord=np.inf
                    )
                ),
                "r9_probe_pass": bool(r9_probe_pass),
                "r9_probe_delays_seconds": r9_probe_delays,
                "wall_seconds": float(time.perf_counter() - started),
            }
        )
    feasible_indices = [
        index
        for index, item in enumerate(records)
        if item["r9_probe_pass"] and np.isfinite(item["objective"])
    ]
    selection_indices = feasible_indices or list(range(len(records)))
    chosen_index = min(
        selection_indices,
        key=lambda index: (
            records[index]["objective"]
            if np.isfinite(records[index]["objective"])
            else np.inf
        ),
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
        "gradient_inf_norm_optimization_space": float(
            chosen["gradient_inf_norm_optimization_space"]
        ),
        "chosen_wall_seconds": float(chosen["wall_seconds"]),
        "total_wall_seconds": float(time.perf_counter() - approach_started),
        "fitted_rows": int(payload["fitted_rows"]),
        "zero_flow_rows_excluded": int(payload["zero_flow_rows_excluded"]),
        "zero_flow_policy": str(payload["zero_flow_policy"]),
        "chosen_parameterization": chosen["parameterization"],
        "wide_restart_parameterization": "selective_log",
        "local_restart_parameterization": "raw",
        "wide_log_parameter_count": int(np.count_nonzero(log_mask)),
        "r9_constrained_restart_selection": True,
        "r9_feasible_restart_count": int(len(feasible_indices)),
        "r9_feasible_solution_available": bool(feasible_indices),
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


def _sobol_points(count: int, dimension: int, seed: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, dimension), dtype=float)
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=int(seed))
    exponent = int(math.ceil(math.log2(max(1, count))))
    return np.asarray(sampler.random_base2(exponent)[:count], dtype=float)


def _block_reference_start(
    block_lanes: Sequence[str],
    coefficient_names: Sequence[str],
    block_bounds: Sequence[Tuple[float, float]],
    warm_parameters: Optional[Mapping[str, Mapping[str, float]]],
) -> tuple[np.ndarray, str, int]:
    values = []
    inherited = 0
    for index, (lane, name) in enumerate(
        (lane, name) for lane in block_lanes for name in coefficient_names
    ):
        lower, upper = map(float, block_bounds[index])
        candidate = None
        if warm_parameters is not None:
            candidate = warm_parameters.get(str(lane), {}).get(str(name))
        if candidate is not None and np.isfinite(float(candidate)):
            value = float(np.clip(float(candidate), lower, upper))
            inherited += 1
        else:
            value = _reference_value((lower, upper))
        values.append(value)
    kind = "parent_warm" if inherited else "geometric_reference"
    return np.asarray(values, dtype=float), kind, inherited


def generate_mixed_restart_payloads(
    *,
    universal_expr: str,
    lanes: Sequence[str],
    approach_targets: Mapping[str, pd.Series],
    prepared: Mapping[str, Any],
    rng: np.random.Generator,
    n_restarts: int,
    maxiter: int,
    maxfun: int,
    coefficient_bounds_override: Optional[Mapping[str, Tuple[float, float]]],
    warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> tuple[list[str], list[Dict[str, Any]], int]:
    """Replay retained local starts and exchange one slot for wide coverage."""
    if n_restarts < 2:
        raise ValueError("mixed restart fitting requires at least two restarts")
    coefficient_names, official_payloads = _generate_official_payloads(
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
    # Draw only after the full retained stream so every replayed local point is
    # exactly attributable to the old implementation.
    base_seed = int(rng.integers(0, 2**32, dtype=np.uint64))
    payloads: list[Dict[str, Any]] = []
    for official_payload in official_payloads:
        payload = dict(official_payload)
        approach = str(payload["approach"])
        block_lanes = list(payload["block_lanes"])
        block_bounds = [tuple(item) for item in payload["bounds"]]
        reference, reference_kind, inherited = _block_reference_start(
            block_lanes,
            coefficient_names,
            block_bounds,
            warm_parameters,
        )
        dimension = len(block_bounds)
        wide_unit = _sobol_points(
            1,
            dimension,
            _stable_seed(base_seed, universal_expr, str(approach), "role_wide"),
        )
        wide = _map_role_aware_points(wide_unit, block_bounds)[0]
        official_initials = list(np.asarray(payload["initials"], dtype=float))
        if inherited:
            initials = [reference, *official_initials[: n_restarts - 2], wide]
            kinds = (
                [reference_kind]
                + ["official_local_replay"] * (n_restarts - 2)
                + ["sobol_role_wide"]
            )
        else:
            initials = [*official_initials[: n_restarts - 1], wide]
            kinds = (
                ["official_local_replay"] * (n_restarts - 1)
                + ["sobol_role_wide"]
            )
        payload["initials"] = np.asarray(initials, dtype=float)
        payload["restart_kinds"] = kinds
        payload["warm_values_inherited"] = int(inherited)
        payloads.append(payload)
    return coefficient_names, payloads, base_seed


def fit_lane_parameters_mixed_jacobian_parallel(
    universal_expr: str,
    df: pd.DataFrame,
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    approach_targets: Mapping[str, pd.Series],
    intersection_id: int,
    verbose: bool = False,
    prepared_context: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = V3_RESTARTS,
    diagnostics: Optional[Dict[str, Any]] = None,
    coefficient_bounds_override: Optional[Mapping[str, Tuple[float, float]]] = None,
    *,
    warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
    parallel_workers: int = V3_PARALLEL_WORKERS,
    maxiter: int = V3_MAXITER,
    maxfun: int = V3_MAXFUN,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit one expression with ten mixed starts and analytic derivatives."""
    if min(n_restarts, parallel_workers, maxiter, maxfun) < 1:
        raise ValueError("restarts, workers, and budgets must be positive")
    coefficient_names = extract_coefficients(universal_expr)
    if not coefficient_names:
        return {str(lane): {} for lane in lanes}
    rng = rng or np.random.default_rng()
    prepared = prepared_context or prepare_optimization_context(
        df, list(lanes), dict(lane_to_approach), int(intersection_id)
    )

    build_started = time.perf_counter()
    derivative_expressions = _symbolic_derivative_strings(
        universal_expr, tuple(coefficient_names)
    )
    coefficient_names, payloads, base_seed = generate_mixed_restart_payloads(
        universal_expr=universal_expr,
        lanes=list(lanes),
        approach_targets=approach_targets,
        prepared=prepared,
        rng=rng,
        n_restarts=int(n_restarts),
        maxiter=int(maxiter),
        maxfun=int(maxfun),
        coefficient_bounds_override=coefficient_bounds_override,
        warm_parameters=warm_parameters,
    )
    for payload in payloads:
        payload["derivative_expressions"] = derivative_expressions
    symbolic_build_seconds = float(time.perf_counter() - build_started)
    if not payloads:
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
            records = [_solve_approach_selective_log(item) for item in payloads]
        else:
            assert active_executor is not None
            records = list(
                active_executor.map(_solve_approach_selective_log, payloads)
            )
    finally:
        if owns_executor and active_executor is not None:
            active_executor.shutdown(wait=True)
    outer_wall_seconds = float(time.perf_counter() - started)

    fitted: Dict[str, Dict[str, float]] = {}
    fit_records = []
    for record, payload in zip(records, payloads):
        fitted.update(
            unpack_lane_parameters(
                np.asarray(record["values"], dtype=float),
                list(record["block_lanes"]),
                coefficient_names,
            )
        )
        compact = {
            key: value
            for key, value in record.items()
            if key not in {"values", "block_lanes", "coefficient_names"}
        }
        compact["restart_kinds"] = list(payload["restart_kinds"])
        compact["warm_values_inherited"] = int(payload["warm_values_inherited"])
        for restart, kind in zip(compact["restarts"], payload["restart_kinds"]):
            restart["start_kind"] = kind
        fit_records.append(compact)

    objective_sum = float(sum(float(item["objective"]) for item in records))
    if diagnostics is not None:
        serial_work_seconds = float(
            sum(float(item["total_wall_seconds"]) for item in records)
        )
        diagnostics.update(
            {
                "solver": "L-BFGS-B",
                "jacobian": "analytic_sympy_numexpr",
                "parameterization": "raw_local_selective_log_wide",
                "start_strategy_id": START_STRATEGY_ID,
                "n_restarts": int(n_restarts),
                "maxiter": int(maxiter),
                "maxfun": int(maxfun),
                "parallel_workers": workers,
                "rng_initialization_semantics": (
                    "retained_r10_stream_generated_then_one_wide_sobol_seed"
                ),
                "sobol_base_seed": base_seed,
                "parent_warm_start_available": warm_parameters is not None,
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
                    sum(item["total_function_evaluations"] for item in records)
                ),
                "total_gradient_evaluations": int(
                    sum(item["total_gradient_evaluations"] for item in records)
                ),
                "approaches": fit_records,
            }
        )
    if verbose:
        print(
            f"  v3 Jacobian L-BFGS-B: approaches={len(records)}, "
            f"restarts={n_restarts}, objective={objective_sum:.4f}, "
            f"wall={outer_wall_seconds:.2f}s"
        )
    return fitted


class MixedRestartJacobianFitter:
    """Persistent four-worker fitter with optional parent coefficient reuse."""

    def __init__(
        self,
        *,
        parallel_workers: int = V3_PARALLEL_WORKERS,
        maxiter: int = V3_MAXITER,
        maxfun: int = V3_MAXFUN,
    ) -> None:
        self.parallel_workers = int(parallel_workers)
        self.maxiter = int(maxiter)
        self.maxfun = int(maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._parent_by_expression: Dict[str, str] = {}
        self._parameters_by_expression: Dict[str, Dict[str, Dict[str, float]]] = {}

    def register_parent(self, expression: str, parent_expression: Optional[str]) -> None:
        if parent_expression:
            self._parent_by_expression[str(expression)] = str(parent_expression)

    def remember_parameters(
        self, expression: str, parameters: Mapping[str, Mapping[str, float]]
    ) -> None:
        self._parameters_by_expression[str(expression)] = {
            str(lane): {str(name): float(value) for name, value in values.items()}
            for lane, values in parameters.items()
        }

    def parent_warm_parameters(
        self, expression: str
    ) -> Optional[Dict[str, Dict[str, float]]]:
        parent = self._parent_by_expression.get(str(expression))
        return self._parameters_by_expression.get(parent) if parent else None

    def fit(
        self,
        *,
        warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if self._executor is None:
            raise RuntimeError("MixedRestartJacobianFitter is closed")
        return fit_lane_parameters_mixed_jacobian_parallel(
            warm_parameters=warm_parameters,
            parallel_workers=self.parallel_workers,
            maxiter=self.maxiter,
            maxfun=self.maxfun,
            executor=self._executor,
            **kwargs,
        )

    def __call__(self, universal_expr: str, **kwargs) -> Dict[str, Dict[str, float]]:
        warm = self.parent_warm_parameters(universal_expr)
        result = self.fit(
            universal_expr=universal_expr,
            warm_parameters=warm,
            **kwargs,
        )
        self.remember_parameters(universal_expr, result)
        return result

    def fit_final_with_polish(
        self,
        *,
        universal_expr: str,
        polish_maxiter: int = 200,
        polish_maxfun: int = 20_000,
        warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
        diagnostics: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        from methods.cosydelay._internal.optimizer_jacobian_polish.polish_fitter import (
            polish_lane_parameters_parallel,
        )

        base_diagnostics: Dict[str, Any] = {}
        base = self.fit(
            universal_expr=universal_expr,
            warm_parameters=warm_parameters,
            diagnostics=base_diagnostics,
            **kwargs,
        )
        polish_diagnostics: Dict[str, Any] = {}
        polished = polish_lane_parameters_parallel(
            universal_expr=universal_expr,
            lane_parameters=base,
            diagnostics=polish_diagnostics,
            polish_maxiter=int(polish_maxiter),
            polish_maxfun=int(polish_maxfun),
            parallel_workers=self.parallel_workers,
            executor=self._executor,
            **{
                key: value
                for key, value in kwargs.items()
                if key not in {"rng", "n_restarts"}
            },
        )
        # Polish is accepted for accuracy only when it does not destroy an
        # already feasible base solution.  No expression term or optimizer is
        # added; this is a deterministic feasibility-preserving rollback.
        from .physics_audit import audit_fitted_physics

        lane_list = list(kwargs["lanes"])
        polished_physics = audit_fitted_physics(
            universal_expr, polished, lane_list
        )
        base_physics = None
        if not polished_physics["joint_pass"]:
            base_physics = audit_fitted_physics(
                universal_expr, base, lane_list
            )
        rollback = bool(
            base_physics is not None
            and base_physics["joint_pass"]
            and not polished_physics["joint_pass"]
        )
        selected = base if rollback else polished
        selected_physics = base_physics if rollback else polished_physics
        self.remember_parameters(universal_expr, selected)
        if diagnostics is not None:
            diagnostics.update(
                {
                    "solver": "L-BFGS-B",
                    "base": base_diagnostics,
                    "polish": polish_diagnostics,
                    "polish_physics_rollback": rollback,
                    "base_enhanced_physics": base_physics,
                    "polished_enhanced_physics": polished_physics,
                    "selected_enhanced_physics": selected_physics,
                    "objective_sum": (
                        polish_diagnostics["objective_sum_before"]
                        if rollback
                        else polish_diagnostics["objective_sum_after"]
                    ),
                }
            )
        return selected

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "MixedRestartJacobianFitter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
