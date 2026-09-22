"""Paper-structured final prompts for the prospective V10 method.

The templates retain the paper's original order: task and inputs, parent plus
evaluation feedback for mutation, physical requirements, mutation instruction,
and one strict output block.  They deliberately do not alter the paper Fitness
or any program-side verifier.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Any, Iterator, Mapping, Optional


FINAL_PROMPT_CONTRACT_ID = "paper_structured_clean_training_feedback_v6"
MAX_NOVELTY_EXAMPLES_IN_PROMPT = 10


_SHARED_TASK = """You are a traffic delay modeling expert. Create one UNIVERSAL
symbolic expression for lane-level delay at intersection {intersection_id}.
Every lane uses the same expression structure, while its positive coefficients
are fitted separately. Predict y in seconds from:
- flow_lane: lane flow ratio (dimensionless)
- GR_phase: effective green ratio (dimensionless)
- Cycle_Time: signal cycle time (seconds)
"""


_SHARED_REQUIREMENTS = """PHYSICAL AND STRUCTURAL REQUIREMENTS:
1. Use the coefficient-robust global form
   y = Cycle_Time * flow_lane / GR_phase * H(flow_lane, GR_phase; a1, ...).
   Algebraically equivalent factorizations are allowed, but the complete
   expression must retain the explicit flow_lane and fixed /GR_phase factors.
2. H must be finite, strictly positive as flow_lane -> 0+ and GR_phase -> 0+,
   nondecreasing in flow_lane, and nonincreasing in GR_phase. Use positive,
   well-defined sums, products, 1+x, log(1+x), exp(x), positive powers, and
   strictly positive denominators.
3. Consequently, y must be nonnegative and finite on 0.001<=flow_lane<=1.2,
   0.04<=GR_phase<=0.85, and 60<=Cycle_Time<=180; increase or remain constant
   with flow; decrease or remain constant with green; satisfy
   y(0, GR_phase, Cycle_Time)=0 exactly; and approach +infinity as
   GR_phase->0+ for fixed positive flow and cycle.
4. The output must have time dimension one. Every additive term must reduce to
   Cycle_Time times a dimensionless function. exp/log arguments and variable
   exponents must be dimensionless.
5. Use only +, -, *, /, **, exp, and log. Do not use subtractive or canceling
   branches, coefficient differences, or unsafe denominators. A unary negative
   is allowed inside a positive decreasing transform such as exp(-a1*GR_phase)
   when all monotonicity, endpoint, and domain requirements remain satisfied.
6. The fixed global /GR_phase factor alone must establish the zero-green limit.
   H must not introduce another zero-green singularity through 1/GR_phase,
   log(1/GR_phase), exp(1/GR_phase), or a negative/fitted green power. Do not
   append a fixed post-fit guard.
7. Use consecutive positive coefficients beginning at a1, never beyond a8.
   Do not target a predetermined coefficient count: retain only coefficients,
   terms, and transforms needed to capture delay variation. Accuracy capacity
   comes first; among similarly capable structures, prefer the simpler one.
8. Every retained term must have a distinct traffic interpretation. Avoid
   redundant branches, equivalent rewrites, unnecessary nonlinear nesting,
   and changes that only rename coefficients or reorder terms.
9. Give H enough nonlinear demand capacity to represent gradual delay growth at
   low demand and sharper growth toward high demand while retaining stable green
   attenuation. Do not include any mechanism mechanically or redundantly.
"""


_OUTPUT = """Return exactly one candidate and no chain of thought or extra text:
```
### Expression
y = <one expression using consecutive coefficients a1, a2, ...>
### Explanation
<brief physical interpretation of each term>
```
"""


def build_final_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    """Build the final initialization prompt in the paper's concise structure."""
    del feature_explanations, universal_features
    if not include_physical_knowledge:
        raise ValueError("the V10 final prompt requires physical knowledge")
    if str(prompt_style).strip().lower() != "standard":
        raise ValueError("the V10 final prompt is frozen to the standard style")
    return "\n\n".join(
        (
            _SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            (
                "INITIALIZATION TASK:\nGenerate a new physically admissible "
                "structure from scratch. Vary H meaningfully across candidates "
                "rather than repeating one template."
            ),
            _SHARED_REQUIREMENTS.strip(),
            _OUTPUT.strip(),
        )
    ) + "\n"


def build_final_mutation_prompt(
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
    """Build a mutation prompt containing the current parent and its feedback."""
    del base_thought, base_explanation, universal_features
    if not include_physical_knowledge:
        raise ValueError("the V10 final prompt requires physical knowledge")
    if str(prompt_style).strip().lower() != "standard":
        raise ValueError("the V10 final prompt is frozen to the standard style")

    physical_pass, physical_reason = validation_result
    physical_status = "PASS" if physical_pass else "FAIL"
    training_feedback = str(search_feedback or "").strip()
    if not training_feedback:
        training_feedback = (
            "PARENT TRAINING EVALUATION:\n"
            "- No numerical Training summary was available; use only the supplied "
            "physical feedback."
        )
    mutation = str(mutation_type).strip().lower()
    if mutation == "small":
        mutation_instruction = (
            "Make a small but real structural change that directly addresses the "
            "feedback while retaining useful parent mechanisms."
        )
    else:
        mutation_instruction = (
            "Make a meaningfully different structural mutation. Replace, reorganize, "
            "merge, or simplify demand amplification and green attenuation; do not "
            "merely append terms, rename coefficients, or reorder the parent."
        )

    parent_block = "\n".join(
        (
            "PARENT EXPRESSION:",
            f"y = {str(base_expr).strip()}",
            "",
            f"PARENT PHYSICAL AUDIT: {physical_status}",
            str(physical_reason or "No detailed physical feedback.").strip(),
            "",
            training_feedback,
        )
    )
    task_block = "\n".join(
        (
            "MUTATION TASK:",
            mutation_instruction,
            "Use the Training evaluation only as directional feedback: seek higher "
            "paper Fitness/R2 and lower RMSE/MAE without sacrificing a physical pass. "
            "Add complexity only when it represents a distinct mechanism needed to "
            "correct an identified weakness.",
        )
    )
    return "\n\n".join(
        (
            _SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            parent_block,
            task_block,
            _SHARED_REQUIREMENTS.strip(),
            _OUTPUT.strip(),
        )
    ) + "\n"


def format_parent_training_feedback(
    record: Optional[Mapping[str, Any]],
    *,
    current_best_fitness: Optional[float] = None,
) -> str:
    """Format only Training-side metrics for the parent's mutation prompt."""
    if not record:
        return (
            "PARENT TRAINING EVALUATION:\n"
            "- Numerical Training evaluation is unavailable for this parent."
        )
    metrics = record.get("metrics", {}) or {}
    enhanced = record.get("enhanced_physics", {}) or {}

    def number(value: Any) -> str:
        try:
            return f"{float(value):.6f}"
        except (TypeError, ValueError):
            return "unavailable"

    parent_fitness = record.get("fitness")
    comparison_lines = []
    try:
        parent_value = float(parent_fitness)
        best_value = float(current_best_fitness)
        gap = max(0.0, best_value - parent_value)
        comparison_lines.extend(
            (
                f"- Current best Training Fitness in the evaluated search: {best_value:.6f}",
                f"- Parent Fitness gap from current best: {gap:.6f}",
                (
                    "- Search direction: preserve the parent's effective mechanisms "
                    "and refine their response shape."
                    if gap <= 0.005
                    else "- Search direction: the parent is below the current best; "
                    "make a genuinely different response structure rather than a "
                    "cosmetic rewrite."
                ),
            )
        )
    except (TypeError, ValueError):
        comparison_lines.append(
            "- No current-best comparison was available for this parent."
        )
    lines = [
        "PARENT TRAINING EVALUATION (Training only):",
        f"- Paper Fitness: {number(parent_fitness)}",
        (
            "- Mean nonnegative approach R2: "
            f"{number(metrics.get('macro_nonnegative_r2'))}"
        ),
        f"- Pooled RMSE: {number(metrics.get('pooled_rmse'))}",
        f"- Pooled MAE: {number(metrics.get('pooled_mae'))}",
        (
            "- Enhanced physical gate: "
            f"{'PASS' if enhanced.get('joint_pass') else 'FAIL'}"
        ),
        *comparison_lines,
        "- Validation and Test metrics are not available to evolution.",
    ]
    return "\n".join(lines)


@contextmanager
def install_final_prompt_contract(
    *, adaptation_module: ModuleType
) -> Iterator[None]:
    """Install the two frozen templates without changing program-side checks."""
    previous_init = adaptation_module.build_universal_lane_init_prompt
    previous_mutation = adaptation_module.build_universal_lane_mutation_prompt
    adaptation_module.build_universal_lane_init_prompt = build_final_init_prompt
    adaptation_module.build_universal_lane_mutation_prompt = (
        build_final_mutation_prompt
    )
    try:
        yield
    finally:
        adaptation_module.build_universal_lane_init_prompt = previous_init
        adaptation_module.build_universal_lane_mutation_prompt = previous_mutation
