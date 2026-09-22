"""Structure-neutral prompts with no numerical Training feedback to the LLM."""

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Mapping, Optional

from methods.cosydelay._internal.evolution_support import final_prompt as base


PROMPT_CONTRACT_ID = "structure_neutral_physics_no_training_feedback_v2"

PHYSICAL_REQUIREMENTS = """PHYSICAL AND STRUCTURAL REQUIREMENTS:
1. Construct the complete delay response from physical requirements below.
   Do not assume or insert any fixed unit baseline inside a multiplier. Every
   numerical scale in the response must be learnable from positive fitted
   coefficients; constants such as 1 may be used only for safe denominators or
   dimensionless log/exp arguments, not as an unavoidable delay amplitude.
2. At fixed green ratio and cycle time, complete delay must be nondecreasing
   with flow_lane. At fixed flow and cycle time, complete delay must be
   nonincreasing with GR_phase. Apply these tests to the complete expression,
   not to an isolated factor.
3. The complete expression must be finite, real, and nonnegative throughout
   0.001<=flow_lane<=1.2, 0.04<=GR_phase<=0.85, and
   60<=Cycle_Time<=180.
4. It must satisfy y(0, GR_phase, Cycle_Time)=0 exactly, and for every fixed
   positive flow and cycle time it must approach +infinity as GR_phase->0+.
   Satisfy these endpoints through the generated expression itself; no fixed
   post-fit guard is added.
5. The output must have time dimension one. flow_lane, GR_phase, coefficients,
   powers, and exp/log arguments are dimensionless; Cycle_Time supplies the
   time dimension.
6. Use only +, -, *, /, **, exp, and log. Keep denominators and log arguments
   strictly positive across the domain. Avoid cancellation, coefficient
   differences, and interactions that can reverse either required monotonicity.
7. Use consecutive positive fitted coefficients beginning at a1, never beyond
   a8. Do not target a predetermined coefficient count. Prefer a compact
   expression that captures delay variation; each retained mechanism must have
   a distinct traffic interpretation.
8. Explore genuinely different admissible response families across candidates.
   The examples implicit in common traffic formulas are possibilities, not
   mandatory templates or fixed factorizations.
"""


def build_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    del feature_explanations, universal_features
    if not include_physical_knowledge or str(prompt_style).strip().lower() != "standard":
        raise ValueError("V13 requires the fixed physical standard prompt")
    return "\n\n".join(
        (
            base._SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            (
                "INITIALIZATION TASK:\nGenerate one new physically admissible "
                "structure from scratch. Use a compact demand and green-response "
                "mechanism that is structurally different from excluded candidates."
            ),
            PHYSICAL_REQUIREMENTS.strip(),
            base._OUTPUT.strip(),
        )
    ) + "\n"


def build_mutation_prompt(
    base_expr: str,
    base_thought: str,
    base_explanation: str,
    validation_result: tuple[bool, str],
    mutation_type: str,
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
    search_feedback: Optional[str] = None,
) -> str:
    """Expose only the parent formula and independent physical feedback."""
    del base_thought, base_explanation, universal_features, search_feedback
    if not include_physical_knowledge or str(prompt_style).strip().lower() != "standard":
        raise ValueError("V13 requires the fixed physical standard prompt")
    physical_pass, physical_reason = validation_result
    mutation = str(mutation_type).strip().lower()
    instruction = (
        "Make one compact structural refinement while retaining useful parent mechanisms."
        if mutation == "small"
        else "Create a meaningfully different structure by replacing, reorganizing, "
        "merging, or simplifying a response mechanism; do not merely append terms, "
        "rename coefficients, or reorder the parent."
    )
    parent = "\n".join(
        (
            "PARENT EXPRESSION:",
            f"y = {str(base_expr).strip()}",
            "",
            f"PARENT PHYSICAL AUDIT: {'PASS' if physical_pass else 'FAIL'}",
            str(physical_reason or "No detailed physical feedback.").strip(),
        )
    )
    task = "\n".join(
        (
            "MUTATION TASK:",
            instruction,
            "Use only the parent structure and its independent physical audit.",
        )
    )
    return "\n\n".join(
        (
            base._SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            parent,
            task,
            PHYSICAL_REQUIREMENTS.strip(),
            base._OUTPUT.strip(),
        )
    ) + "\n"


@contextmanager
def install_prompt_contract(*, adaptation_module: ModuleType) -> Iterator[None]:
    previous_init = adaptation_module.build_universal_lane_init_prompt
    previous_mutation = adaptation_module.build_universal_lane_mutation_prompt
    adaptation_module.build_universal_lane_init_prompt = build_init_prompt
    adaptation_module.build_universal_lane_mutation_prompt = build_mutation_prompt
    try:
        yield
    finally:
        adaptation_module.build_universal_lane_init_prompt = previous_init
        adaptation_module.build_universal_lane_mutation_prompt = previous_mutation
