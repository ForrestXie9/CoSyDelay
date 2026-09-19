"""Stronger, guard-free physical audit for fitted v3 expressions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
import sympy as sp

from expression_rules import coefficient_names, parse_symbolic_expression
from expression_validation_lane import (
    PhysicalVerifierConfig,
    _evaluate_zero_green_limit_symbolic_lane,
    _positive_parameter_expression,
    score_fitted_lanes_principlewise,
)


AUDIT_ID = "symbolic_r8_r9_dense_boundary_2026_08_05_v1"
SEARCH_AUDIT_ID = "standard_fitted_plus_symbolic_r8_r9_2026_08_05_v1"


def _continuous_at_zero_green(
    expression: sp.Expr,
    green: sp.Symbol,
) -> bool:
    """Conservatively certify continuity at ``GR_phase=0+``.

    This is deliberately a sufficient, not necessary, test.  It recognizes the
    positive exp/log/power grammar used by the finalized prompt.  Any ambiguous
    construction falls back to the existing isolated SymPy limit evaluator.
    """
    if green not in expression.free_symbols:
        return True
    if expression == green:
        return True
    if expression.is_Atom:
        return False
    if expression.is_Add or expression.is_Mul:
        return all(_continuous_at_zero_green(arg, green) for arg in expression.args)
    if expression.func == sp.exp:
        argument = expression.args[0]
        if not _continuous_at_zero_green(argument, green):
            return False
        endpoint = argument.subs(green, sp.Integer(0))
        return bool(getattr(endpoint, "is_finite", None) is True)
    if expression.func == sp.log:
        argument = expression.args[0]
        if not _continuous_at_zero_green(argument, green):
            return False
        endpoint = argument.subs(green, sp.Integer(0))
        return bool(
            getattr(endpoint, "is_finite", None) is True
            and getattr(endpoint, "is_positive", None) is True
        )
    if expression.is_Pow:
        base, exponent = expression.args
        if green in exponent.free_symbols:
            return False
        if not _continuous_at_zero_green(base, green):
            return False
        endpoint = base.subs(green, sp.Integer(0))
        if getattr(endpoint, "is_finite", None) is not True:
            return False
        if getattr(exponent, "is_integer", None) is True:
            return bool(
                getattr(exponent, "is_nonnegative", None) is True
                or getattr(endpoint, "is_nonzero", None) is True
            )
        if getattr(exponent, "is_positive", None) is True:
            return bool(getattr(endpoint, "is_nonnegative", None) is True)
        return bool(getattr(endpoint, "is_positive", None) is True)
    return False


def _fixed_inverse_green_certificate(
    parsed: sp.Expr,
    symbols: Mapping[str, sp.Symbol],
) -> tuple[bool, Optional[str]]:
    """Fast sufficient proof for the finalized ``Cycle*flow/GR*H`` form.

    For fixed positive flow and cycle, one explicit inverse-green factor tends
    to positive infinity.  If the remaining factor H is conservatively proved
    continuous, finite, and strictly positive at zero green, their product must
    also tend to positive infinity.  Inconclusive cases are never accepted by
    this function and continue through the former isolated symbolic proof.
    """
    flow = symbols["flow_lane"]
    green = symbols["GR_phase"]
    cycle = symbols["Cycle_Time"]
    positive_flow = sp.Symbol("_certificate_positive_flow", positive=True, finite=True)
    try:
        remainder = sp.cancel(parsed * green / (cycle * flow))
    except Exception:
        return False, None
    if cycle in remainder.free_symbols:
        return False, None
    remainder = remainder.subs(flow, positive_flow)
    if not _continuous_at_zero_green(remainder, green):
        return False, None
    try:
        endpoint = remainder.subs(green, sp.Integer(0))
    except Exception:
        return False, None
    invalid = (sp.oo, -sp.oo, sp.zoo, sp.nan)
    if any(endpoint.has(item) for item in invalid):
        return False, None
    if not (
        getattr(endpoint, "is_finite", None) is True
        and getattr(endpoint, "is_positive", None) is True
    ):
        return False, None
    return True, str(endpoint)


@lru_cache(maxsize=4096)
def _symbolic_structure_audit(expression: str) -> Dict[str, Any]:
    """Cache coefficient-robust endpoint proofs for search-time screening."""
    result: Dict[str, Any] = {
        "symbolic_r8_exact": False,
        "symbolic_r9_positive_infinity": False,
        "symbolic_r9_method": None,
        "symbolic_r9_detail": None,
        "errors": [],
    }
    try:
        parsed, symbols = parse_symbolic_expression(expression)
        flow = symbols["flow_lane"]
        green = symbols["GR_phase"]
        cycle = symbols["Cycle_Time"]
        names = coefficient_names(expression)
        zero_residual = sp.simplify(parsed.subs(flow, sp.Integer(0)))
        result["symbolic_r8_residual"] = str(zero_residual)
        result["symbolic_r8_exact"] = bool(
            zero_residual == 0 or getattr(zero_residual, "is_zero", None) is True
        )
        if not result["symbolic_r8_exact"]:
            result["errors"].append("R8 was not an exact symbolic zero-flow identity")
        fast_r9, fast_endpoint = _fixed_inverse_green_certificate(parsed, symbols)
        if fast_r9:
            result["symbolic_r9_positive_infinity"] = True
            result["symbolic_r9_limit"] = "+infinity"
            result["symbolic_r9_method"] = (
                "fixed_inverse_green_finite_positive_h_certificate"
            )
            result["symbolic_r9_detail"] = None
            result["symbolic_r9_h_at_zero"] = fast_endpoint
            return result
        positive_expression = _positive_parameter_expression(parsed, symbols, names)
        symbolic_pass, symbolic_limit, symbolic_method, symbolic_error = (
            _evaluate_zero_green_limit_symbolic_lane(
                positive_expression,
                flow,
                green,
                cycle,
                PhysicalVerifierConfig(),
            )
        )
        result["symbolic_r9_positive_infinity"] = bool(symbolic_pass)
        result["symbolic_r9_limit"] = symbolic_limit
        result["symbolic_r9_method"] = symbolic_method
        result["symbolic_r9_detail"] = symbolic_error
        if symbolic_pass is not True:
            result["errors"].append(
                "R9 positive-infinity limit was not symbolically established"
            )
    except Exception as exc:
        result["errors"].append(f"symbolic structure audit failed: {type(exc).__name__}: {exc}")
    return result


def audit_search_physics(
    expression: str,
    *,
    standard_joint_pass: bool,
) -> Dict[str, Any]:
    """Cheap search gate; dense fitted audit remains mandatory at CV/final."""
    symbolic = _symbolic_structure_audit(str(expression))
    result = {
        "audit_id": SEARCH_AUDIT_ID,
        "standard_joint_pass": bool(standard_joint_pass),
        "symbolic_r8_exact": bool(symbolic["symbolic_r8_exact"]),
        "symbolic_r9_positive_infinity": bool(
            symbolic["symbolic_r9_positive_infinity"]
        ),
        "symbolic_r8_residual": symbolic.get("symbolic_r8_residual"),
        "symbolic_r9_limit": symbolic.get("symbolic_r9_limit"),
        "symbolic_r9_method": symbolic.get("symbolic_r9_method"),
        "symbolic_r9_detail": symbolic.get("symbolic_r9_detail"),
        "dense_audit_deferred_to_cv": True,
        "fixed_r9_guard_applied": False,
        "errors": list(symbolic["errors"]),
    }
    if not standard_joint_pass:
        result["errors"].append("standard fitted verifier did not jointly pass")
    result["joint_pass"] = bool(
        result["standard_joint_pass"]
        and result["symbolic_r8_exact"]
        and result["symbolic_r9_positive_infinity"]
    )
    return result


def _dense_operational_grid(config: PhysicalVerifierConfig) -> tuple[np.ndarray, ...]:
    flow = np.unique(
        np.concatenate(
            (
                np.asarray(config.flow_domain, dtype=float),
                np.geomspace(config.flow_domain[0], config.flow_domain[1], 15),
                np.linspace(config.flow_domain[0], config.flow_domain[1], 15),
            )
        )
    )
    green = np.unique(
        np.concatenate(
            (
                np.asarray(config.green_domain, dtype=float),
                np.geomspace(config.green_domain[0], config.green_domain[1], 15),
                np.linspace(config.green_domain[0], config.green_domain[1], 15),
            )
        )
    )
    cycle = np.asarray(
        [
            config.cycle_domain_seconds[0],
            np.mean(config.cycle_domain_seconds),
            config.cycle_domain_seconds[1],
        ],
        dtype=float,
    )
    return np.meshgrid(flow, green, cycle, indexing="ij")


def _is_positive_infinite(value: Any) -> bool:
    return bool(
        value == sp.oo
        or (
            getattr(value, "is_infinite", False) is True
            and getattr(value, "is_positive", False) is True
        )
    )


def audit_fitted_physics(
    expression: str,
    lane_parameters: Mapping[str, Mapping[str, float]],
    lanes: Sequence[str],
    *,
    config: Optional[PhysicalVerifierConfig] = None,
    precomputed_standard: Optional[Mapping[str, Any]] = None,
    precomputed_symbolic: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Require the standard verifier plus symbolic and dense robustness checks.

    R9 is proved on the positive-coefficient expression.  A numerical near-zero
    probe is retained in the standard verifier diagnostics but cannot replace an
    undecidable symbolic limit in this stronger audit.  No term is appended to
    the generated expression.
    """
    config = config or PhysicalVerifierConfig()
    lane_list = [str(lane) for lane in lanes]
    standard_reused = bool(
        precomputed_standard is not None
        and "joint_pass" in precomputed_standard
        and "rule_scores" in precomputed_standard
    )
    if standard_reused:
        standard_joint_pass = bool(precomputed_standard["joint_pass"])
        standard_rule_scores = {
            str(name): float(score)
            for name, score in dict(precomputed_standard["rule_scores"]).items()
        }
    else:
        standard = score_fitted_lanes_principlewise(
            expression,
            {str(k): dict(v) for k, v in lane_parameters.items()},
            lane_list,
            ["flow_lane", "GR_phase", "Cycle_Time"],
            config=config,
        )
        standard_joint_pass = bool(standard.joint_pass)
        standard_rule_scores = dict(standard.rule_scores)
    result: Dict[str, Any] = {
        "audit_id": AUDIT_ID,
        "fixed_r9_guard_applied": False,
        "standard_joint_pass": standard_joint_pass,
        "standard_rule_scores": standard_rule_scores,
        "standard_audit_reused": standard_reused,
        "symbolic_r8_exact": False,
        "symbolic_r9_positive_infinity": False,
        "symbolic_r9_method": None,
        "symbolic_r9_detail": None,
        "dense_finite": False,
        "dense_nonnegative": False,
        "dense_flow_nondecreasing": False,
        "dense_green_nonincreasing": False,
        "dense_points_per_lane": 0,
        "lane_diagnostics": {},
        "errors": [],
    }
    failed_standard_rules = [
        str(name)
        for name, score in standard_rule_scores.items()
        if float(score) < 1.0
    ]
    if failed_standard_rules:
        result["errors"].append(
            "failed standard fitted rules " + ", ".join(failed_standard_rules)
        )
    try:
        parsed, symbols = parse_symbolic_expression(expression)
        flow = symbols["flow_lane"]
        green = symbols["GR_phase"]
        cycle = symbols["Cycle_Time"]
        names = coefficient_names(expression)

        symbolic_reused = bool(
            precomputed_symbolic is not None
            and "symbolic_r8_exact" in precomputed_symbolic
            and "symbolic_r9_positive_infinity" in precomputed_symbolic
        )
        symbolic = (
            dict(precomputed_symbolic)
            if symbolic_reused
            else _symbolic_structure_audit(str(expression))
        )
        result["symbolic_endpoint_audit_reused"] = symbolic_reused
        result["symbolic_r8_exact"] = bool(symbolic["symbolic_r8_exact"])
        result["symbolic_r8_residual"] = symbolic.get("symbolic_r8_residual")
        result["symbolic_r9_positive_infinity"] = bool(
            symbolic["symbolic_r9_positive_infinity"]
        )
        result["symbolic_r9_limit"] = symbolic.get("symbolic_r9_limit")
        result["symbolic_r9_method"] = symbolic.get("symbolic_r9_method")
        result["symbolic_r9_detail"] = symbolic.get("symbolic_r9_detail")
        if not result["symbolic_r8_exact"]:
            result["errors"].append("R8 was not an exact symbolic zero-flow identity")
        if not result["symbolic_r9_positive_infinity"]:
            result["errors"].append(
                "R9 positive-infinity limit was not symbolically established"
            )

        flow_grid, green_grid, cycle_grid = _dense_operational_grid(config)
        flat_flow = flow_grid.reshape(-1)
        flat_green = green_grid.reshape(-1)
        flat_cycle = cycle_grid.reshape(-1)
        result["dense_points_per_lane"] = int(flat_flow.size)
        value_expression = sp.lambdify(
            [flow, green, cycle, *[symbols[name] for name in names]],
            parsed,
            modules="numpy",
        )
        flow_derivative_expression = sp.lambdify(
            [flow, green, cycle, *[symbols[name] for name in names]],
            sp.diff(parsed, flow),
            modules="numpy",
        )
        green_derivative_expression = sp.lambdify(
            [flow, green, cycle, *[symbols[name] for name in names]],
            sp.diff(parsed, green),
            modules="numpy",
        )

        all_finite = True
        all_nonnegative = True
        all_flow = True
        all_green = True
        for lane in lane_list:
            missing = [
                name
                for name in names
                if lane not in lane_parameters or name not in lane_parameters[lane]
            ]
            if missing:
                result["errors"].append(
                    f"{lane}: missing fitted coefficients {', '.join(missing)}"
                )
                all_finite = all_nonnegative = all_flow = all_green = False
                continue
            coefficients = [float(lane_parameters[lane][name]) for name in names]
            with np.errstate(all="ignore"):
                values = np.asarray(
                    value_expression(flat_flow, flat_green, flat_cycle, *coefficients),
                    dtype=float,
                )
                flow_derivatives = np.asarray(
                    flow_derivative_expression(
                        flat_flow, flat_green, flat_cycle, *coefficients
                    ),
                    dtype=float,
                )
                green_derivatives = np.asarray(
                    green_derivative_expression(
                        flat_flow, flat_green, flat_cycle, *coefficients
                    ),
                    dtype=float,
                )
            values = np.broadcast_to(values, flat_flow.shape)
            flow_derivatives = np.broadcast_to(flow_derivatives, flat_flow.shape)
            green_derivatives = np.broadcast_to(green_derivatives, flat_flow.shape)
            finite = bool(np.all(np.isfinite(values)))
            nonnegative = bool(
                finite and np.all(values >= -config.zero_tolerance)
            )
            flow_ok = bool(
                np.all(np.isfinite(flow_derivatives))
                and np.all(flow_derivatives >= -config.derivative_tolerance)
            )
            green_ok = bool(
                np.all(np.isfinite(green_derivatives))
                and np.all(green_derivatives <= config.derivative_tolerance)
            )
            result["lane_diagnostics"][lane] = {
                "finite": finite,
                "nonnegative": nonnegative,
                "flow_nondecreasing": flow_ok,
                "green_nonincreasing": green_ok,
                "minimum_delay": (
                    float(np.min(values[np.isfinite(values)]))
                    if np.any(np.isfinite(values))
                    else None
                ),
                "minimum_flow_derivative": (
                    float(np.min(flow_derivatives[np.isfinite(flow_derivatives)]))
                    if np.any(np.isfinite(flow_derivatives))
                    else None
                ),
                "maximum_green_derivative": (
                    float(np.max(green_derivatives[np.isfinite(green_derivatives)]))
                    if np.any(np.isfinite(green_derivatives))
                    else None
                ),
            }
            all_finite &= finite
            all_nonnegative &= nonnegative
            all_flow &= flow_ok
            all_green &= green_ok
        result["dense_finite"] = all_finite
        result["dense_nonnegative"] = all_nonnegative
        result["dense_flow_nondecreasing"] = all_flow
        result["dense_green_nonincreasing"] = all_green
        for key in (
            "dense_finite",
            "dense_nonnegative",
            "dense_flow_nondecreasing",
            "dense_green_nonincreasing",
        ):
            if not result[key]:
                result["errors"].append(f"failed {key}")
    except Exception as exc:
        result["errors"].append(f"audit failed: {type(exc).__name__}: {exc}")

    result["joint_pass"] = bool(
        result["standard_joint_pass"]
        and result["symbolic_r8_exact"]
        and result["symbolic_r9_positive_infinity"]
        and result["dense_finite"]
        and result["dense_nonnegative"]
        and result["dense_flow_nondecreasing"]
        and result["dense_green_nonincreasing"]
    )
    return result
