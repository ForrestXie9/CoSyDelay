"""Scoped accuracy-first parsimony prompt experiment; frozen V9 is untouched."""

from __future__ import annotations

import ast
from collections import Counter
from contextlib import contextmanager
from types import ModuleType
from typing import Any, Dict, Iterator, List, Mapping, Optional

import sympy as sp

from expression_rules import (
    coefficient_names,
    parse_symbolic_expression,
    structural_family_key,
)
from methods.cosydelay._internal.optimizer_conditioning.role_policy import (
    nonlinear_role_conflicts,
)


COMPLEXITY_CONTRACT_ID = "prospective_accuracy_first_minimum_complexity_v2"
EPSILON_FITNESS_TOLERANCE = 0.002
COMPLEXITY_PROMPT_NOTE = """
PROSPECTIVE ACCURACY-FIRST PARSIMONY AND DIVERSITY CONTRACT:
- Do not target a predetermined coefficient count. Use as many of the allowed
  consecutive coefficients a1 through a8 as are genuinely needed to capture
  changes in delay, but use no redundant coefficient, branch, or transform.
- First preserve enough demand and green-ratio response capacity to model the
  delay variation. Among structures with comparable capacity, prefer the one
  with fewer fitted coefficients, fewer operations, and shallower nonlinear
  nesting. Never use all eight coefficients merely because eight are allowed.
- A large mutation must create a meaningfully different traffic mechanism or
  functional family. Replace, reorganize, merge, or simplify the parent's
  demand amplification and green attenuation; do not interpret "large" as
  automatically appending terms or coefficients.
- Every retained term must have a distinct traffic interpretation and affect a
  response that a simpler term does not already capture. Avoid nested nonlinear
  transformations and avoid reusing one coefficient as both a fitted power
  exponent and an exp coefficient when a clearer uncoupled structure exists.
- Before responding, silently remove algebraically redundant factors and check
  that the expression remains structurally different from the prohibited
  families while satisfying all traffic-physics requirements.
"""


def expression_complexity(expression: str) -> Dict[str, Any]:
    parsed, _ = parse_symbolic_expression(str(expression))
    names = coefficient_names(str(expression))
    return {
        "coefficient_names": names,
        "coefficient_count": len(names),
        "sympy_operations": int(sp.count_ops(parsed, visual=False)),
        "python_ast_nodes": int(
            sum(1 for _ in ast.walk(ast.parse(str(expression), mode="eval")))
        ),
        "expression_characters": len(str(expression)),
        "structural_family_key": structural_family_key(str(expression)),
        "nonlinear_role_conflicts_diagnostic_only": nonlinear_role_conflicts(
            str(expression)
        ),
    }


def complexity_key(expression: str) -> tuple[int, int, int, int]:
    item = expression_complexity(expression)
    return (
        int(item["coefficient_count"]),
        int(item["sympy_operations"]),
        int(item["python_ast_nodes"]),
        int(item["expression_characters"]),
    )


def select_epsilon_parsimonious(
    evaluations: Mapping[str, Mapping[str, Any]],
    *,
    fitness_tolerance: float = EPSILON_FITNESS_TOLERANCE,
) -> Dict[str, Any]:
    """Diagnostic selection; never changes evolution or its paper Fitness."""
    if fitness_tolerance < 0.0:
        raise ValueError("fitness_tolerance must be nonnegative")
    eligible = [
        item
        for item in evaluations.values()
        if bool(item.get("enhanced_physics", {}).get("joint_pass"))
    ]
    if not eligible:
        raise ValueError("no physically passing evaluations")
    best_fitness = max(float(item["fitness"]) for item in eligible)
    near_best = [
        item
        for item in eligible
        if best_fitness - float(item["fitness"]) <= fitness_tolerance + 1e-12
    ]
    chosen = min(
        near_best,
        key=lambda item: (
            complexity_key(str(item["expression"])),
            -float(item["fitness"]),
        ),
    )
    expression = str(chosen["expression"])
    return {
        "diagnostic_only_not_used_for_evolution": True,
        "fitness_tolerance": float(fitness_tolerance),
        "best_paper_fitness": float(best_fitness),
        "near_best_candidates": len(near_best),
        "selected_expression": expression,
        "selected_fitness": float(chosen["fitness"]),
        "fitness_gap_from_best": float(best_fitness - float(chosen["fitness"])),
        "selected_complexity": expression_complexity(expression),
    }


@contextmanager
def install_accuracy_first_parsimony_contract(
    *,
    adaptation_module: Optional[ModuleType] = None,
    append_prompt_note: bool = True,
) -> Iterator[List[Dict[str, Any]]]:
    """Temporarily strengthen prompt guidance without adding a hard size gate."""
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module

    previous_standard = adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD
    previous_compact = adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT
    previous_parsimony = adaptation_module.INTERPRETABILITY_AND_PARSIMONY_RULES
    previous_validator = adaptation_module.validate_candidate_expression
    audit: List[Dict[str, Any]] = []

    def diagnostic_validator(expression: str):
        passed, reason = previous_validator(expression)
        try:
            complexity = expression_complexity(str(expression))
        except Exception as exc:
            complexity = {
                "complexity_audit_error": f"{type(exc).__name__}: {exc}"
            }
        audit.append(
            {
                "expression": str(expression),
                "contract_id": COMPLEXITY_CONTRACT_ID,
                "upstream_passed": bool(passed),
                "upstream_reason": str(reason),
                "passed": bool(passed),
                **complexity,
            }
        )
        return passed, reason

    if append_prompt_note:
        adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = (
            previous_standard + COMPLEXITY_PROMPT_NOTE
        )
        adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = (
            previous_compact + COMPLEXITY_PROMPT_NOTE
        )
        adaptation_module.INTERPRETABILITY_AND_PARSIMONY_RULES = (
            previous_parsimony + COMPLEXITY_PROMPT_NOTE
        )
    adaptation_module.validate_candidate_expression = diagnostic_validator
    try:
        yield audit
    finally:
        adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = previous_standard
        adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = previous_compact
        adaptation_module.INTERPRETABILITY_AND_PARSIMONY_RULES = previous_parsimony
        adaptation_module.validate_candidate_expression = previous_validator


def summarize_complexity_audit(audit: List[Dict[str, Any]]) -> Dict[str, Any]:
    passed = [item for item in audit if item.get("upstream_passed")]
    counts = Counter(
        int(item["coefficient_count"])
        for item in passed
        if item.get("coefficient_count") is not None
    )
    family_count = len(
        {
            str(item["structural_family_key"])
            for item in passed
            if item.get("structural_family_key")
        }
    )
    return {
        "records": len(audit),
        "upstream_physics_passes": len(passed),
        "coefficient_count_distribution_after_upstream": {
            str(key): int(value) for key, value in sorted(counts.items())
        },
        "unique_structural_families_after_upstream": int(family_count),
        "nonlinear_role_conflicts": int(
            sum(
                bool(item.get("nonlinear_role_conflicts_diagnostic_only"))
                for item in passed
            )
        ),
        "no_complexity_hard_rejection": True,
    }
