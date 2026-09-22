"""Paper-structured prompts for total-response physical evolution.

V11 changes only the expression-search instructions and Training feedback.  It
does not change the paper Fitness, coefficient fitting, or the post-fit hard
physics verifier.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Any, Iterator, Mapping, Optional


FINAL_PROMPT_CONTRACT_ID = "total_response_training_feedback_v1"
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
   expression must retain explicit flow_lane and fixed /GR_phase factors.
2. Apply monotonicity to the COMPLETE delay response, not to H in isolation.
   The complete flow factor flow_lane*H must be nondecreasing in flow_lane,
   and the complete green factor H/GR_phase must be nonincreasing in GR_phase.
   H itself may increase or decrease when the explicit outer factor safely
   dominates and the complete response still satisfies both conditions.
3. Safe compensated saturation is allowed. For example, a positive factor
   1/(1+a1*flow_lane) may attenuate H when the outer flow_lane makes the full
   branch flow_lane/(1+a1*flow_lane) nondecreasing. Likewise,
   1/(1+a1/GR_phase) may be used when the complete green response remains
   nonincreasing and another positive branch preserves the zero-green limit.
   These are admissible mechanisms, not templates that must be copied.
4. The complete y must be nonnegative and finite on 0.001<=flow_lane<=1.2,
   0.04<=GR_phase<=0.85, and 60<=Cycle_Time<=180; satisfy
   y(0, GR_phase, Cycle_Time)=0 exactly; and approach +infinity as
   GR_phase->0+ for fixed positive flow and cycle.
5. Use positive, well-defined sums, products, 1+x, log(1+x), exp(x), positive
   powers, and strictly positive denominators. Avoid subtractive/canceling
   branches, coefficient differences, unsafe log arguments, and response
   interactions such as exp(-a1*GR_phase*flow_lane) that can reverse the total
   flow derivative after fitting.
6. The fixed global /GR_phase factor must provide an uncancellable zero-green
   divergence through at least one strictly positive finite branch of H. A
   bounded compensating branch may coexist with it. Do not append a fixed
   post-fit guard.
7. The output must have time dimension one. Every additive term must reduce to
   Cycle_Time times a dimensionless function. exp/log arguments and variable
   exponents must be dimensionless. Use only +, -, *, /, **, exp, and log.
8. Use consecutive positive coefficients beginning at a1, never beyond a8.
   Do not target a predetermined coefficient count. Retain only mechanisms
   needed to capture delay variation; among similarly accurate structures,
   prefer the simpler one.
9. Give the complete response enough nonlinear capacity for gradual low-demand
   growth, high-demand amplification or saturation, and stable green
   attenuation. Every retained term must have a distinct traffic meaning.
"""


_OUTPUT = """Return exactly one candidate and no chain of thought or extra text:
```
### Expression
y = <one expression using consecutive coefficients a1, a2, ...>
### Explanation
<brief physical interpretation of each term>
```
"""


def build_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    """Build a concise initialization prompt without an incumbent formula."""
    del feature_explanations, universal_features
    if not include_physical_knowledge:
        raise ValueError("V11 requires physical knowledge")
    if str(prompt_style).strip().lower() != "standard":
        raise ValueError("V11 is fixed to the standard prompt style")
    return "\n\n".join(
        (
            _SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            (
                "INITIALIZATION TASK:\nGenerate a new physically admissible "
                "structure from scratch. Cover a genuinely different demand, "
                "saturation, or green-response mechanism from excluded candidates."
            ),
            _SHARED_REQUIREMENTS.strip(),
            _OUTPUT.strip(),
        )
    ) + "\n"


def _number(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "unavailable"


def format_parent_training_feedback(
    record: Optional[Mapping[str, Any]],
    *,
    current_best_fitness: Optional[float] = None,
) -> str:
    """Report macro and per-approach Training evidence to mutation only."""
    if not record:
        return (
            "PARENT TRAINING EVALUATION (Training only):\n"
            "- Numerical Training evaluation is unavailable for this parent."
        )

    metrics = record.get("metrics", {}) or {}
    enhanced = record.get("enhanced_physics", {}) or {}
    parent_fitness = record.get("fitness")
    lines = [
        "PARENT TRAINING EVALUATION (Training only):",
        f"- Paper Fitness: {_number(parent_fitness)}",
        (
            "- Mean nonnegative approach R2: "
            f"{_number(metrics.get('macro_nonnegative_r2'))}"
        ),
        f"- Pooled RMSE: {_number(metrics.get('pooled_rmse'))}",
        f"- Pooled MAE: {_number(metrics.get('pooled_mae'))}",
    ]

    by_approach = metrics.get("by_approach", {}) or {}
    available: list[tuple[str, float]] = []
    for approach in sorted(by_approach):
        item = by_approach.get(approach, {}) or {}
        try:
            available.append((str(approach), float(item.get("r2"))))
        except (TypeError, ValueError):
            pass
        lines.append(
            f"- Approach {approach} Training: R2={_number(item.get('r2'))}, "
            f"RMSE={_number(item.get('rmse'))}, MAE={_number(item.get('mae'))}"
        )
    if available:
        weakest, weakest_r2 = min(available, key=lambda item: item[1])
        lines.append(
            f"- Weakest Training approach: {weakest} (R2={weakest_r2:.6f}); "
            "improve its response shape without sacrificing stronger approaches."
        )

    lines.append(
        "- Enhanced physical gate: "
        f"{'PASS' if enhanced.get('joint_pass') else 'FAIL'}"
    )
    errors = [str(item).strip() for item in enhanced.get("errors", []) if str(item).strip()]
    if errors:
        lines.append("- Physical repair target: " + "; ".join(errors[:2]))

    try:
        parent_value = float(parent_fitness)
        best_value = float(current_best_fitness)
        gap = max(0.0, best_value - parent_value)
        lines.extend(
            (
                f"- Current best Training Fitness: {best_value:.6f}",
                f"- Parent Fitness gap: {gap:.6f}",
                (
                    "- Search direction: retain useful mechanisms and make a "
                    "targeted response-shape refinement."
                    if gap <= 0.005
                    else "- Search direction: make a genuinely different but "
                    "physically safe response structure."
                ),
            )
        )
    except (TypeError, ValueError):
        lines.append("- No current-best Training comparison was available.")
    lines.append("- Validation and Test metrics are not available to evolution.")
    return "\n".join(lines)


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
    """Build mutation input from the current parent and Training feedback."""
    del base_thought, base_explanation, universal_features
    if not include_physical_knowledge:
        raise ValueError("V11 requires physical knowledge")
    if str(prompt_style).strip().lower() != "standard":
        raise ValueError("V11 is fixed to the standard prompt style")

    physical_pass, physical_reason = validation_result
    training_feedback = str(search_feedback or "").strip() or (
        "PARENT TRAINING EVALUATION (Training only):\n"
        "- Numerical Training evaluation is unavailable."
    )
    mutation = str(mutation_type).strip().lower()
    if mutation == "small":
        instruction = (
            "Make a small but genuine structural refinement of the complete "
            "flow or green response, retaining useful parent mechanisms."
        )
    else:
        instruction = (
            "Make a meaningfully different structural mutation by replacing, "
            "reorganizing, merging, or simplifying a response mechanism. Do not "
            "merely append terms, rename coefficients, or reorder the parent."
        )
    parent_block = "\n".join(
        (
            "PARENT EXPRESSION:",
            f"y = {str(base_expr).strip()}",
            "",
            f"PARENT PHYSICAL AUDIT: {'PASS' if physical_pass else 'FAIL'}",
            str(physical_reason or "No detailed physical feedback.").strip(),
            "",
            training_feedback,
        )
    )
    task_block = "\n".join(
        (
            "MUTATION TASK:",
            instruction,
            "Use only the supplied Training evidence as directional feedback. "
            "Seek higher paper Fitness/R2 and lower RMSE/MAE while the complete "
            "expression continues to pass every physical rule.",
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


@contextmanager
def install_prompt_contract(*, adaptation_module: ModuleType) -> Iterator[None]:
    """Install V11 prompts for one scoped evolution run."""
    previous_init = adaptation_module.build_universal_lane_init_prompt
    previous_mutation = adaptation_module.build_universal_lane_mutation_prompt
    adaptation_module.build_universal_lane_init_prompt = build_init_prompt
    adaptation_module.build_universal_lane_mutation_prompt = build_mutation_prompt
    try:
        yield
    finally:
        adaptation_module.build_universal_lane_init_prompt = previous_init
        adaptation_module.build_universal_lane_mutation_prompt = previous_mutation
