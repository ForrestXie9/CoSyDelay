"""Paper-structured prompt for the prospective seven-rule method."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Mapping, Optional

from methods.cosydelay.engine.evolution_support import final_prompt as base


PROMPT_CONTRACT_ID = (
    "manuscript_seven_rule_physics_only_feedback_v4_output_schema_last"
)
OUTPUT_SCHEMA = base._OUTPUT.strip()

_SHARED_TASK = """You are a traffic delay modeling expert. Create one UNIVERSAL
symbolic skeleton for a movement-delay function at intersection {intersection_id}.
Every movement uses the same skeleton with positive movement-specific
coefficients. The available observed targets are approach-average delays, so
the movement coefficients within each approach are jointly calibrated through
the flow-weighted aggregate prediction. The LLM proposes only the symbolic
structure and never supplies fitted numerical coefficient values. Predict y in
seconds from:
- flow_lane: movement flow ratio (dimensionless)
- GR_phase: effective serving-green ratio (dimensionless)
- Cycle_Time: signal cycle time (seconds)
"""

PHYSICAL_REQUIREMENTS = """PHYSICAL AND STRUCTURAL REQUIREMENTS:
1. The complete expression must use flow_lane, GR_phase, and Cycle_Time.
   Every movement shares this structure and receives positive coefficients
   jointly calibrated within its approach.
2. At fixed green ratio and cycle time, complete delay must be nondecreasing
   with flow_lane. At fixed flow and cycle time, complete delay must be
   nonincreasing with GR_phase. Apply both conditions to the full response.
3. The output must have time dimension one. flow_lane, GR_phase, fitted
   coefficients, powers, and exp/log arguments are dimensionless;
   Cycle_Time supplies the time dimension.
4. As flow_lane approaches zero from above, delay must have a finite,
   nonnegative limit. This limit may be positive; do not impose an exact
   zero-delay identity.
5. Delay must remain finite, real, and nonnegative throughout
   0.001<=flow_lane<=2.0, 0.02<=GR_phase<=0.95, and
   30<=Cycle_Time<=240.
6. For every fixed positive flow and cycle time, the complete expression must
   approach +infinity as GR_phase approaches zero from above. Establish this
   through the generated expression itself; no fixed post-fit guard is added.
7. Use only +, -, *, /, **, exp, and log. Keep denominators and log arguments
   strictly positive over the declared domain. Avoid cancellation and
   coefficient differences that can reverse the required monotonicity.
8. Use consecutive positive fitted coefficients beginning at a1, never beyond
   a8. Do not target a predetermined coefficient count. Prefer a compact
   expression that captures delay variation, and retain only mechanisms with
   a distinct traffic interpretation. Do not reuse the same coefficient both
   as a fitted power exponent and inside an exp() exponent; use separate
   coefficients for those two nonlinear roles.
9. Explore genuinely different admissible response families across
   candidates. Common traffic formulas are possibilities, not mandatory
   templates or fixed factorizations.
"""


def build_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    del feature_explanations, universal_features
    if not include_physical_knowledge or str(prompt_style).lower() != "standard":
        raise ValueError("V16 requires the frozen manuscript prompt")
    return "\n\n".join(
        (
            _SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            (
                "INITIALIZATION TASK:\nGenerate one compact candidate from "
                "scratch. Vary the demand/green mechanism across candidates "
                "without copying an excluded expression."
            ),
            PHYSICAL_REQUIREMENTS.strip(),
            OUTPUT_SCHEMA,
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
    del base_thought, base_explanation, universal_features, search_feedback
    if not include_physical_knowledge or str(prompt_style).lower() != "standard":
        raise ValueError("V16 requires the frozen manuscript prompt")
    physical_pass, physical_reason = validation_result
    mutation = str(mutation_type).strip().lower()
    instruction = (
        "Make one compact structural refinement while retaining useful parent mechanisms."
        if mutation == "small"
        else "Create a meaningfully different compact structure by replacing, "
        "reorganizing, merging, or simplifying a response mechanism."
    )
    parent = "\n".join(
        (
            "PARENT EXPRESSION:",
            f"y = {str(base_expr).strip()}",
            "",
            f"PARENT PHYSICAL AUDIT: {'PASS' if physical_pass else 'PARTIAL/FAIL'}",
            str(physical_reason or "No failed physical principle was reported.").strip(),
        )
    )
    task = "\n".join(
        (
            "MUTATION TASK:",
            instruction,
            "Use the current parent and its concise physical-principle audit. "
            "Do not infer or request Training, Validation, or Test metrics.",
        )
    )
    return "\n\n".join(
        (
            _SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            parent,
            task,
            PHYSICAL_REQUIREMENTS.strip(),
            OUTPUT_SCHEMA,
        )
    ) + "\n"


def move_output_schema_last(prompt: str) -> str:
    """Put programmatic novelty/retry instructions before the output schema.

    The shared generator appends novelty and retry corrections after the base
    prompt has been built.  Reordering the one immutable output block at the
    final API boundary preserves every instruction while matching the paper's
    role/context/task/constraints/output organization.
    """
    text = str(prompt)
    count = text.count(OUTPUT_SCHEMA)
    if count != 1:
        raise RuntimeError(
            f"V16 prompt must contain exactly one output schema, observed {count}"
        )
    before, after = text.split(OUTPUT_SCHEMA, 1)
    if not after.strip():
        return text
    return "\n\n".join((before.strip(), after.strip(), OUTPUT_SCHEMA)) + "\n"


def output_schema_is_last(prompt: str) -> bool:
    """Return whether the single required output schema closes the prompt."""
    text = str(prompt)
    return text.count(OUTPUT_SCHEMA) == 1 and text.rstrip().endswith(OUTPUT_SCHEMA)


@contextmanager
def install_prompt_contract(*, adaptation_module: ModuleType) -> Iterator[None]:
    previous_init = adaptation_module.build_universal_lane_init_prompt
    previous_mutation = adaptation_module.build_universal_lane_mutation_prompt
    previous_run_llm = adaptation_module.run_llm

    def run_llm_with_output_schema_last(prompt: str, *args, **kwargs):
        return previous_run_llm(move_output_schema_last(prompt), *args, **kwargs)

    adaptation_module.build_universal_lane_init_prompt = build_init_prompt
    adaptation_module.build_universal_lane_mutation_prompt = build_mutation_prompt
    adaptation_module.run_llm = run_llm_with_output_schema_last
    try:
        yield
    finally:
        adaptation_module.run_llm = previous_run_llm
        adaptation_module.build_universal_lane_init_prompt = previous_init
        adaptation_module.build_universal_lane_mutation_prompt = previous_mutation
