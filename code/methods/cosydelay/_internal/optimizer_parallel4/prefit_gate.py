"""Conservative coefficient-independent checks before expensive fitting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, Tuple

import sympy as sp

from expression_rules import (
    coefficient_names,
    parse_symbolic_expression,
    validate_candidate_expression,
)


GATE_ID = "structural_exact_r8_no_neutral_r9_2026_08_03_v1"


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    reason: str
    checks: Dict[str, bool] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    gate_id: str = GATE_ID

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _zero_flow_identity(expression: str) -> Tuple[bool, str]:
    """Require zero flow to cancel structurally, before coefficients are known."""
    parsed, symbols = parse_symbolic_expression(expression)
    residual = parsed.subs(symbols["flow_lane"], sp.Integer(0))
    if residual == 0 or getattr(residual, "is_zero", None) is True:
        return True, "0"

    witness_substitutions = {
        symbols["GR_phase"]: 0.5,
        symbols["Cycle_Time"]: 120.0,
        **{
            symbols[name]: 1.0
            for name in coefficient_names(expression)
        },
    }
    try:
        witness = float(sp.N(residual.subs(witness_substitutions)))
    except (OverflowError, TypeError, ValueError):
        witness = float("nan")
    if math.isfinite(witness) and abs(witness) > 1e-10:
        return False, f"neutral witness={witness:.12g}; residual={str(residual)[:400]}"

    try:
        residual = sp.simplify(residual)
    except Exception:
        pass
    if residual == 0 or getattr(residual, "is_zero", None) is True:
        return True, "0"
    return False, str(residual)[:500]


def evaluate_structural_prefit_gate(expression: str) -> GateDecision:
    """Reject only structural failures and exact zero-flow violations.

    R9 is deliberately absent here. Its numerical 10,000-second threshold
    depends on fitted coefficients and remains authoritative post-fit.
    """
    checks: Dict[str, bool] = {}
    diagnostics: Dict[str, Any] = {
        "r9_prefit_policy": "deferred_to_postfit_fitted_verifier",
        "neutral_coefficient_r9_probe_run": False,
        "fixed_postfit_guard_applied": False,
    }

    structural_ok, structural_reason = validate_candidate_expression(expression)
    checks["structural_legality_units_domain_and_obvious_direction"] = bool(
        structural_ok
    )
    if not structural_ok:
        return GateDecision(
            False,
            f"structural: {structural_reason}",
            checks,
            diagnostics,
        )

    try:
        zero_ok, zero_residual = _zero_flow_identity(expression)
    except Exception as exc:
        zero_ok = False
        zero_residual = f"{type(exc).__name__}: {exc}"
    checks["r8_zero_flow_identity"] = bool(zero_ok)
    diagnostics["r8_zero_flow_residual"] = zero_residual
    if not zero_ok:
        return GateDecision(
            False,
            "R8: expression does not reduce identically to zero at flow_lane=0",
            checks,
            diagnostics,
        )

    return GateDecision(
        True,
        "Expression passed structural and exact-R8 pre-fit checks; R9 deferred post-fit",
        checks,
        diagnostics,
    )


def validate_structural_prefit_expression(expression: str) -> Tuple[bool, str]:
    decision = evaluate_structural_prefit_gate(expression)
    return decision.passed, decision.reason

