"""
Joint optimization of lane-specific parameters for universal lane delay models.

The universal expression is shared by all lanes. Each lane receives its own
coefficient values, and all lane parameters are fitted jointly against
approach-level delay observations.
"""
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numexpr as ne
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from constants import INTERSECTION_CONFIGS
from expression_rules import coefficient_names, parameter_roles

MAX_DELAY = 1e6
DEFAULT_PARAM_BOUNDS = (0.001, 1000.0)
POWER_EXPONENT_BOUNDS = (0.05, 5.0)
EXP_COEFFICIENT_BOUNDS = (0.0001, 1.0)
OPTIMIZER_MAXITER = 100
OPTIMIZER_MAXFUN = 1000
ZERO_FLOW_EPSILON = 1e-6
ZERO_FLOW_POLICIES = {"na", "zero", "uniform"}


def extract_coefficients(expr: str) -> List[str]:
    """Extract coefficient names from an expression, e.g. a1, a2, a10."""
    return coefficient_names(expr)


def build_parameter_bounds(expr: str, coef_names: List[str]) -> List[Tuple[float, float]]:
    """Assign bounds from parameter roles in the parsed expression tree."""
    roles = parameter_roles(expr)

    bounds = []
    for name in coef_names:
        if "power_exponent" in roles.get(name, set()):
            bounds.append(POWER_EXPONENT_BOUNDS)
        elif "exp_coefficient" in roles.get(name, set()):
            bounds.append(EXP_COEFFICIENT_BOUNDS)
        else:
            bounds.append(DEFAULT_PARAM_BOUNDS)
    return bounds


def substitute_coefficients(expr: str, coefficients: Dict[str, float]) -> str:
    """Substitute coefficient values without replacing a1 inside a10."""
    substituted = expr
    for coef_name, coef_value in coefficients.items():
        substituted = re.sub(rf"\b{coef_name}\b", f"({coef_value})", substituted)
    return substituted


def sanitize_delay(delay, max_delay: float = MAX_DELAY) -> np.ndarray:
    """Return finite, non-negative delay values capped at max_delay."""
    delay = np.asarray(delay, dtype=float)
    delay = np.nan_to_num(delay, nan=max_delay, posinf=max_delay, neginf=0.0)
    delay = np.maximum(delay, 0.0)
    return np.minimum(delay, max_delay)


def get_controlling_green_ratios(config: Dict, lane: str) -> List[str]:
    """Return green-ratio column names controlling a lane."""
    direction, movement = lane.split("_", 1)
    green_ratios = []
    for phase, phase_movements in config["phase_movements"].items():
        if direction in phase_movements and movement in phase_movements[direction]:
            green_ratios.append(f"GR_{phase}")
    return green_ratios


def build_lane_contexts(
    df: pd.DataFrame,
    lanes: List[str],
    intersection_id: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Precompute numexpr contexts for each lane."""
    config = INTERSECTION_CONFIGS[intersection_id]
    contexts = {}

    for lane in lanes:
        flow_col = f"flow_{lane}"
        green_ratios = get_controlling_green_ratios(config, lane)

        if len(green_ratios) == 1:
            gr_phase = df[green_ratios[0]].values
        elif len(green_ratios) > 1:
            gr_phase = sum(df[gr].values for gr in green_ratios)
        else:
            gr_phase = np.ones(len(df)) * 0.5

        contexts[lane] = {
            "flow_lane": df[flow_col].values,
            "GR_phase": gr_phase,
            "Cycle_Time": df["Cycle_Time"].values,
        }

    return contexts


def group_lanes_by_approach(lane_to_approach: Dict[str, str]) -> Dict[str, List[str]]:
    """Group lane IDs by approach ID."""
    approach_lanes = {}
    for lane, approach in lane_to_approach.items():
        approach_lanes.setdefault(approach, []).append(lane)
    return approach_lanes


def build_approach_weights(
    lane_contexts: Dict[str, Dict[str, np.ndarray]],
    approach_lanes: Dict[str, List[str]],
    zero_flow_policy: str = "na",
) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    """Precompute flow weights used for lane-to-approach aggregation.

    ``na`` is the mathematical convention: a per-vehicle weighted mean is
    undefined when total approach flow is zero. Such rows receive NaN at
    prediction time and are excluded from coefficient fitting and metrics.
    ``zero`` is an explicit application-layer convention. ``uniform`` retains
    the historical fallback solely for legacy reproduction.
    """
    if zero_flow_policy not in ZERO_FLOW_POLICIES:
        raise ValueError(
            f"zero_flow_policy must be one of {sorted(ZERO_FLOW_POLICIES)}"
        )
    approach_weights = {}

    for approach, lanes_list in approach_lanes.items():
        lane_flows = {
            lane: lane_contexts[lane]["flow_lane"]
            for lane in lanes_list
            if lane in lane_contexts
        }
        if not lane_flows:
            continue

        total_flow = np.asarray(sum(lane_flows.values()), dtype=float)
        valid_mask = np.isfinite(total_flow) & (total_flow > ZERO_FLOW_EPSILON)
        weights = {}
        for lane, flow in lane_flows.items():
            weight = np.zeros_like(total_flow, dtype=float)
            np.divide(flow, total_flow, out=weight, where=valid_mask)
            if zero_flow_policy == "uniform":
                weight[~valid_mask] = 1.0 / len(lane_flows)
            weights[lane] = weight
        approach_weights[approach] = {
            "lanes": list(lane_flows.keys()),
            "weights": weights,
            "total_flow": total_flow,
            "valid_mask": valid_mask,
            "zero_flow_policy": zero_flow_policy,
            "zero_flow_rows": int(np.count_nonzero(~valid_mask)),
        }

    return approach_weights


def prepare_optimization_context(
    df: pd.DataFrame,
    lanes: List[str],
    lane_to_approach: Dict[str, str],
    intersection_id: int,
    zero_flow_policy: str = "na",
) -> Dict:
    """Build data-dependent arrays once for reuse by every candidate formula."""
    lane_contexts = build_lane_contexts(df, lanes, intersection_id)
    approach_lanes = group_lanes_by_approach(lane_to_approach)
    return {
        "lane_contexts": lane_contexts,
        "approach_lanes": approach_lanes,
        "approach_weights": build_approach_weights(
            lane_contexts, approach_lanes, zero_flow_policy=zero_flow_policy
        ),
        "n_rows": len(df),
        "zero_flow_policy": zero_flow_policy,
    }


def unpack_lane_parameters(
    params: np.ndarray,
    lanes: List[str],
    coef_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """Convert the flat optimizer vector into lane -> coefficient mappings."""
    n_coefs = len(coef_names)
    lane_parameters = {}

    for i, lane in enumerate(lanes):
        start = i * n_coefs
        lane_parameters[lane] = {
            coef_name: float(params[start + j])
            for j, coef_name in enumerate(coef_names)
        }

    return lane_parameters


def evaluate_lane_delay(
    universal_expr: str,
    lane_context: Dict[str, np.ndarray],
    lane_parameters: Dict[str, float],
) -> np.ndarray:
    """Evaluate one lane's delay using its fitted coefficients."""
    # Pass coefficients as variables instead of rebuilding the expression string
    # on every optimizer evaluation. This keeps the expression cache effective.
    evaluation_context = {**lane_context, **lane_parameters}
    values = np.asarray(
        ne.evaluate(universal_expr, local_dict=evaluation_context),
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite lane delay")

    # Boundedness of delay magnitudes is enforced by physical rule R5 on the
    # operational domain. Observed-data prediction no longer rejects candidates
    # for large but finite delays; clip negatives only for approach aggregation.
    return np.maximum(values, 0.0)


def calculate_approach_delays_from_universal(
    df: pd.DataFrame,
    universal_expr: str,
    lane_parameters: Dict[str, Dict[str, float]],
    lanes: List[str],
    lane_to_approach: Dict[str, str],
    intersection_id: int,
    prepared_context: Optional[Dict] = None,
    strict: bool = False,
) -> Dict[str, np.ndarray]:
    """Calculate flow-weighted approach delays from lane-level predictions."""
    prepared = prepared_context or prepare_optimization_context(
        df, lanes, lane_to_approach, intersection_id
    )
    lane_contexts = prepared["lane_contexts"]
    approach_weights = prepared["approach_weights"]
    approach_delays = {}

    for approach, weight_context in approach_weights.items():
        weighted_delay = np.zeros(len(df))

        for lane in weight_context["lanes"]:
            if lane not in lane_parameters:
                continue
            try:
                lane_delay = evaluate_lane_delay(
                    universal_expr,
                    lane_contexts[lane],
                    lane_parameters[lane],
                )
            except Exception as exc:
                if strict:
                    raise FloatingPointError(f"{lane}: {exc}") from exc
                print(f"    Error evaluating lane {lane}: {exc}")
                lane_delay = np.zeros(len(df))

            weighted_delay += weight_context["weights"][lane] * lane_delay

        if weight_context["zero_flow_policy"] == "na":
            weighted_delay = weighted_delay.astype(float, copy=False)
            weighted_delay[~weight_context["valid_mask"]] = np.nan

        approach_delays[approach] = weighted_delay

    return approach_delays


def fit_lane_parameters_to_approaches(
    universal_expr: str,
    df: pd.DataFrame,
    lanes: List[str],
    lane_to_approach: Dict[str, str],
    approach_targets: Dict[str, pd.Series],
    intersection_id: int,
    verbose: bool = False,
    prepared_context: Optional[Dict] = None,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = 1,
    diagnostics: Optional[Dict] = None,
    coefficient_bounds_override: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit separable approach blocks with the same summed-MSE objective."""
    coef_names = extract_coefficients(universal_expr)
    n_coefs = len(coef_names)
    if n_coefs == 0:
        return {lane: {} for lane in lanes}
    if n_restarts < 1:
        raise ValueError("n_restarts must be at least 1")
    rng = rng or np.random.default_rng()

    if verbose:
        print(f"  Fitting parameters for expression: {universal_expr[:80]}...")
        print(f"  Coefficients: {coef_names}, total parameters: {len(lanes) * n_coefs}")

    prepared = prepared_context or prepare_optimization_context(
        df, lanes, lane_to_approach, intersection_id
    )
    lane_contexts = prepared["lane_contexts"]
    approach_weights = prepared["approach_weights"]

    n_params = len(lanes) * n_coefs
    initial_params = rng.uniform(0.1, 1.0, n_params)
    coefficient_bounds = build_parameter_bounds(universal_expr, coef_names)
    if coefficient_bounds_override:
        coefficient_bounds = [
            coefficient_bounds_override.get(name, bound)
            for name, bound in zip(coef_names, coefficient_bounds)
        ]
    initial_by_lane = unpack_lane_parameters(initial_params, lanes, coef_names)
    fitted_parameters: Dict[str, Dict[str, float]] = {}
    objective_sum = 0.0
    fit_records = []

    # No prediction contains parameters from two different approaches. The old
    # objective was a sum of these independent blocks, so separate minimization
    # preserves the mathematical objective while reducing optimizer dimension.
    for approach, target_delay in approach_targets.items():
        if approach not in approach_weights:
            continue

        weight_context = approach_weights[approach]
        block_lanes = weight_context["lanes"]
        block_initial = np.asarray([
            initial_by_lane[lane][name]
            for lane in block_lanes
            for name in coef_names
        ])

        def objective(params):
            lane_parameters = unpack_lane_parameters(params, block_lanes, coef_names)
            approach_delay_pred = np.zeros(len(df))
            for lane in weight_context["lanes"]:
                try:
                    lane_delay = evaluate_lane_delay(
                        universal_expr,
                        lane_contexts[lane],
                        lane_parameters[lane],
                    )
                except Exception:
                    return 1e12
                approach_delay_pred += weight_context["weights"][lane] * lane_delay
            target_values = np.asarray(target_delay.values, dtype=float)
            valid_rows = weight_context["valid_mask"] & np.isfinite(target_values)
            if not np.any(valid_rows):
                return 1e12
            mse = np.mean(
                (approach_delay_pred[valid_rows] - target_values[valid_rows]) ** 2
            )
            return float(mse) if np.isfinite(mse) else 1e12

        block_bounds = coefficient_bounds * len(block_lanes)
        restart_results = []
        for restart_index in range(n_restarts):
            if restart_index == 0:
                restart_initial = block_initial
            else:
                lower = np.asarray([max(bound[0], 0.1) for bound in block_bounds])
                upper = np.asarray([min(bound[1], 1.0) for bound in block_bounds])
                upper = np.maximum(upper, lower)
                restart_initial = rng.uniform(lower, upper)
            started = time.perf_counter()
            candidate_result = minimize(
                objective,
                restart_initial,
                method="L-BFGS-B",
                bounds=block_bounds,
                options={
                    "maxiter": int(os.environ.get(
                        "COSY_OPTIMIZER_MAXITER", OPTIMIZER_MAXITER
                    )),
                    "maxfun": int(os.environ.get(
                        "COSY_OPTIMIZER_MAXFUN", OPTIMIZER_MAXFUN
                    )),
                    "disp": False,
                },
            )
            restart_results.append((candidate_result, time.perf_counter() - started))

        chosen_index = int(
            np.argmin([
                float(item[0].fun) if np.isfinite(item[0].fun) else np.inf
                for item in restart_results
            ])
        )
        result, chosen_seconds = restart_results[chosen_index]
        fitted_parameters.update(unpack_lane_parameters(result.x, block_lanes, coef_names))
        objective_sum += float(result.fun)
        fit_records.append({
            "approach": approach,
            "chosen_restart": chosen_index,
            "success": bool(result.success),
            "status": int(result.status),
            "message": str(result.message),
            "objective": float(result.fun),
            "iterations": int(getattr(result, "nit", 0)),
            "function_evaluations": int(getattr(result, "nfev", 0)),
            "wall_seconds": float(chosen_seconds),
            "fitted_rows": int(np.count_nonzero(weight_context["valid_mask"])),
            "zero_flow_rows_excluded": int(weight_context["zero_flow_rows"]),
            "zero_flow_policy": weight_context["zero_flow_policy"],
            "restarts": [
                {
                    "index": index,
                    "success": bool(item.success),
                    "objective": float(item.fun),
                    "wall_seconds": float(seconds),
                }
                for index, (item, seconds) in enumerate(restart_results)
            ],
        })
        if verbose:
            status = "Success" if result.success else "Stopped"
            print(f"  {approach}: {status}, MSE={result.fun:.2f}")

    if diagnostics is not None:
        diagnostics.update({
            "n_restarts": int(n_restarts),
            "objective_sum": float(objective_sum),
            "approaches": fit_records,
        })
    if verbose:
        print(f"  Equivalent total objective: {objective_sum:.2f}")
    return fitted_parameters
