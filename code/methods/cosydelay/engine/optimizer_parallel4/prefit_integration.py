"""Scoped prompt alignment and conservative pre-fit gate installation."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional

from .prefit_gate import evaluate_structural_prefit_gate


ZERO_FLOW_STANDARD_OLD = """8. Low-demand boundary: for every fixed positive GR_phase and Cycle_Time, the
   one-sided limit as flow_lane -> 0+ exists, is finite, and is non-negative.
   No condition requires this limit to equal zero."""
ZERO_FLOW_STANDARD_NEW = """8. Zero-flow identity: for every fixed positive GR_phase and Cycle_Time,
   y(0, GR_phase, Cycle_Time) = 0 exactly."""
ZERO_FLOW_COMPACT_OLD = """As flow_lane -> 0+, its limit must exist, be
finite and non-negative; No condition requires that limit to equal zero."""
ZERO_FLOW_COMPACT_NEW = """At flow_lane=0, the expression must satisfy
y(0, GR_phase, Cycle_Time)=0 exactly for every positive green ratio and cycle."""


@contextmanager
def install_structural_prefit_gate(
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[List[Dict[str, Any]]]:
    """Align R8 wording, reject structural failures, and audit every attempt."""
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module

    previous_population_generator = (
        population_module.safe_generate_universal_lane_expression
    )
    previous_validator = adaptation_module.validate_candidate_expression
    previous_standard = adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD
    previous_compact = adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT
    audit: List[Dict[str, Any]] = []

    def recording_validator(expression: str):
        decision = evaluate_structural_prefit_gate(expression)
        audit.append({"expression": expression, **decision.to_dict()})
        return decision.passed, decision.reason

    def gated_generator(*args, **kwargs):
        kwargs["enforce_physical_prefilter"] = True
        return adaptation_module.safe_generate_universal_lane_expression(
            *args, **kwargs
        )

    if ZERO_FLOW_STANDARD_OLD not in previous_standard:
        raise RuntimeError(
            "Standard prompt no longer contains the expected stale R8 wording"
        )
    if ZERO_FLOW_COMPACT_OLD not in previous_compact:
        raise RuntimeError(
            "Compact prompt no longer contains the expected stale R8 wording"
        )

    population_module.safe_generate_universal_lane_expression = gated_generator
    adaptation_module.validate_candidate_expression = recording_validator
    adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = previous_standard.replace(
        ZERO_FLOW_STANDARD_OLD, ZERO_FLOW_STANDARD_NEW
    )
    adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = previous_compact.replace(
        ZERO_FLOW_COMPACT_OLD, ZERO_FLOW_COMPACT_NEW
    )
    try:
        yield audit
    finally:
        population_module.safe_generate_universal_lane_expression = (
            previous_population_generator
        )
        adaptation_module.validate_candidate_expression = previous_validator
        adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = previous_standard
        adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = previous_compact
