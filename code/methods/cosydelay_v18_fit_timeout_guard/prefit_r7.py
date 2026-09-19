"""Conservative, coefficient-robust R7 screening before fitting.

The gate rejects only expressions whose complete response is proved finite as
``GR_phase -> 0+`` for arbitrary positive finite coefficients, flow, and cycle
time.  A proof failure is deliberately classified as ``unknown`` and allowed
to continue to the bounded fitter and the existing fitted seven-rule scorer.

No Training values, targets, Validation data, or Test data enter this check.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import time
from typing import Any

import sympy as sp

from expression_rules import coefficient_names, parse_symbolic_expression
from expression_validation_lane import _positive_parameter_expression


PREFIT_R7_AUDIT_ID = "positive_coefficient_finite_endpoint_r7_2026_08_14_v1"


@dataclass(frozen=True)
class PrefitR7Decision:
    """Tri-state result for the conservative pre-fit endpoint check."""

    audit_id: str
    verdict: str
    passed: bool
    method: str
    endpoint: str | None
    reason: str
    wall_seconds: float
    uses_training_values: bool = False
    uses_targets: bool = False
    uses_validation_or_test: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _continuous_finite_at_zero_green(
    expression: sp.Expr,
    green: sp.Symbol,
) -> bool:
    """Prove a sufficient class of right-continuous finite expressions.

    This intentionally avoids ``simplify``, ``factor``, ``cancel``, and
    ``limit``.  Those general symbolic routines can consume unbounded time on
    generated formulas.  Returning ``False`` means only "not proved here".
    """
    if green not in expression.free_symbols:
        return bool(getattr(expression, "is_finite", None) is True)
    if expression == green:
        return True
    if expression.is_Atom:
        return False
    if expression.is_Add or expression.is_Mul:
        return all(
            _continuous_finite_at_zero_green(argument, green)
            for argument in expression.args
        )
    if expression.func == sp.exp:
        argument = expression.args[0]
        if not _continuous_finite_at_zero_green(argument, green):
            return False
        endpoint = argument.subs(green, sp.Integer(0))
        return bool(getattr(endpoint, "is_finite", None) is True)
    if expression.func == sp.log:
        argument = expression.args[0]
        if not _continuous_finite_at_zero_green(argument, green):
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
        if not _continuous_finite_at_zero_green(base, green):
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


@lru_cache(maxsize=4096)
def classify_prefit_r7(expression: str) -> PrefitR7Decision:
    """Reject only a coefficient-robust proof of a finite R7 endpoint."""
    started = time.perf_counter()
    expression = str(expression)
    try:
        parsed, symbols = parse_symbolic_expression(expression)
        names = coefficient_names(expression)
        positive_expression = _positive_parameter_expression(
            parsed,
            symbols,
            names,
        )
        flow = symbols["flow_lane"]
        green = symbols["GR_phase"]
        cycle = symbols["Cycle_Time"]
        positive_flow = sp.Symbol(
            "_prefit_positive_flow", positive=True, finite=True
        )
        positive_cycle = sp.Symbol(
            "_prefit_positive_cycle", positive=True, finite=True
        )
        endpoint_expression = positive_expression.subs(
            {flow: positive_flow, cycle: positive_cycle}
        )
        if _continuous_finite_at_zero_green(endpoint_expression, green):
            endpoint = endpoint_expression.subs(green, sp.Integer(0))
            invalid = (sp.oo, -sp.oo, sp.zoo, sp.nan)
            if not any(endpoint.has(item) for item in invalid) and bool(
                getattr(endpoint, "is_finite", None) is True
            ):
                return PrefitR7Decision(
                    audit_id=PREFIT_R7_AUDIT_ID,
                    verdict="proven_finite",
                    passed=False,
                    method="recursive_continuity_and_direct_substitution",
                    endpoint=str(endpoint),
                    reason=(
                        "R7 pre-fit rejection: with arbitrary positive finite "
                        "coefficients, the complete delay has a finite limit as "
                        "GR_phase approaches zero; generate a distinct expression "
                        "with a genuine zero-green divergence"
                    ),
                    wall_seconds=float(time.perf_counter() - started),
                )
        return PrefitR7Decision(
            audit_id=PREFIT_R7_AUDIT_ID,
            verdict="unknown_allow",
            passed=True,
            method="finite_endpoint_not_proved",
            endpoint=None,
            reason=(
                "R7 pre-fit was inconclusive; candidate is allowed to the "
                "bounded coefficient fitter and fitted seven-rule evaluation"
            ),
            wall_seconds=float(time.perf_counter() - started),
        )
    except Exception as exc:
        return PrefitR7Decision(
            audit_id=PREFIT_R7_AUDIT_ID,
            verdict="unknown_allow",
            passed=True,
            method="classification_error_fail_open_to_bounded_fit",
            endpoint=None,
            reason=(
                "R7 pre-fit was inconclusive "
                f"({type(exc).__name__}); candidate is allowed to the bounded "
                "coefficient fitter"
            ),
            wall_seconds=float(time.perf_counter() - started),
        )
