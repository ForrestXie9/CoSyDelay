"""Concise paper-style prompt used only by the V20 accelerated variant."""

from __future__ import annotations

from typing import Mapping, Optional

from methods.cosydelay._internal.training_protocol.prompt import OUTPUT_SCHEMA


_CONTEXT = """You are a traffic delay modeling expert tasked with generating a
symbolic expression for movement delay at intersection {intersection_id}.

Available variables:
- flow_lane: movement flow ratio (dimensionless)
- GR_phase: effective serving-green ratio (dimensionless)
- Cycle_Time: signal cycle time (seconds)

Use one universal expression structure for all movements. Its positive
movement-specific coefficients are jointly calibrated within each approach
using flow-weighted approach-average delay predictions. Generate the symbolic
structure only; do not provide numerical coefficient values."""


_PRINCIPLES = """The expression must satisfy:
1. Use flow_lane, GR_phase, and Cycle_Time.
2. Delay is nondecreasing with flow_lane when the other variables are fixed.
3. Delay is nonincreasing with GR_phase when the other variables are fixed.
4. The complete expression is homogeneous of degree one in Cycle_Time;
   Cycle_Time supplies the only time dimension, and exp/log arguments are
   dimensionless.
5. As flow_lane approaches zero from above, delay has a finite nonnegative limit.
6. Delay is real, finite, and nonnegative for valid positive inputs.
7. For positive flow_lane and Cycle_Time, delay approaches infinity as
   GR_phase approaches zero from above."""


def build_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    del feature_explanations, universal_features
    if not include_physical_knowledge or str(prompt_style).lower() != "standard":
        raise ValueError("V20 requires the paper-style physical prompt")
    return "\n\n".join(
        (
            _CONTEXT.format(intersection_id=int(intersection_id)).strip(),
            "INITIALIZATION TASK:\n"
            "Generate one compact and sufficiently expressive candidate from scratch, "
            "with a clear traffic-physical interpretation. Avoid redundant terms and "
            "unnecessary nesting. Use only +, -, *, /, **, exp, and log; never use ^.",
            _PRINCIPLES,
            OUTPUT_SCHEMA,
        )
    ) + "\n"


def build_regeneration_prompt(
    base_expr: str,
    base_thought: str,
    base_explanation: str,
    validation_result: tuple[bool, str],
    regeneration_type: str,
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
    search_feedback: Optional[str] = None,
) -> str:
    del base_thought, base_explanation, universal_features, search_feedback
    if not include_physical_knowledge or str(prompt_style).lower() != "standard":
        raise ValueError("V20 requires the paper-style physical prompt")
    passed, reason = validation_result
    del regeneration_type
    parent = (
        f"Parent expression:\ny = {str(base_expr).strip()}\n\n"
        f"Physical evaluation: {'Valid' if passed else 'Invalid or partial'}\n"
        f"{str(reason or 'No failed physical principle was reported.').strip()}"
    )
    task = (
        "REGENERATION TASK:\n"
        "Generate one compact and sufficiently expressive candidate with a clear "
        "traffic-physical interpretation. Replace or fundamentally reorganize the parent's main "
        "demand-green interaction. Renaming coefficients, reordering terms, changing "
        "parentheses, or making an algebraically equivalent rewrite is insufficient. "
        "No functional family is prescribed. Use only +, -, *, /, **, exp, and log; "
        "never use ^."
    )
    return "\n\n".join(
        (
            _CONTEXT.format(intersection_id=int(intersection_id)).strip(),
            parent,
            task,
            _PRINCIPLES,
            OUTPUT_SCHEMA,
        )
    ) + "\n"


# Compatibility hook required by the retained V16/V18 generator API.  V20
# names this operation "regeneration" in its prompt and reported methodology.
build_mutation_prompt = build_regeneration_prompt
