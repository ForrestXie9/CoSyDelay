"""V14 prompts: no Training metrics, with compact post-fit physical repair."""

from contextlib import contextmanager
import re
from types import ModuleType
from typing import Iterator, Mapping, Optional

from methods.cosydelay.engine.prompt_support.prompt import (
    PHYSICAL_REQUIREMENTS,
)
from methods.cosydelay.engine.evolution_support import final_prompt as base


PROMPT_CONTRACT_ID = "structure_neutral_physics_only_retry_feedback_v1"
_ALLOWED_REPAIR_LABELS = frozenset(("R1", "R2", "R3", "R4", "R6", "R7", "R8", "R9", "Domain"))
_TRAINING_TERMS = re.compile(
    r"fitness|rmse|mae|training|validation|test",
    flags=re.IGNORECASE,
)


def extract_physical_repair_feedback(search_feedback: Optional[str]) -> str:
    """Whitelist rule-level repair lines and discard all numerical fit evidence."""
    lines = []
    for raw in str(search_feedback or "").splitlines():
        line = raw.strip()
        if not line.startswith("- ") or ":" not in line:
            continue
        label = line[2:].split(":", 1)[0].strip()
        if label not in _ALLOWED_REPAIR_LABELS or _TRAINING_TERMS.search(line):
            continue
        # Numeric lane counts, scores, and observed values must never enter the LLM.
        if re.search(r"\d", line.split(":", 1)[1]):
            continue
        lines.append(line)
    return "\n".join(dict.fromkeys(lines))


def build_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    del feature_explanations, universal_features
    if not include_physical_knowledge or str(prompt_style).strip().lower() != "standard":
        raise ValueError("V14 requires the fixed physical standard prompt")
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
    del base_thought, base_explanation, universal_features
    if not include_physical_knowledge or str(prompt_style).strip().lower() != "standard":
        raise ValueError("V14 requires the fixed physical standard prompt")
    physical_pass, physical_reason = validation_result
    mutation = str(mutation_type).strip().lower()
    instruction = (
        "Make one compact structural refinement while retaining useful parent mechanisms."
        if mutation == "small"
        else "Create a meaningfully different structure by replacing, reorganizing, "
        "merging, or simplifying a response mechanism; do not merely append terms, "
        "rename coefficients, or reorder the parent."
    )
    physical_repair = extract_physical_repair_feedback(search_feedback)
    repair_block = (
        "POST-FIT PHYSICAL REPAIR FROM THE REJECTED CANDIDATE:\n"
        + physical_repair
        + "\nApply these rule-level corrections without copying its expression."
        if physical_repair
        else "NO ADDITIONAL POST-FIT PHYSICAL REPAIR IS PENDING."
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
            "Use the parent structure, its independent physical audit, and only "
            "the rule-level physical repair supplied below.",
        )
    )
    return "\n\n".join(
        (
            base._SHARED_TASK.format(intersection_id=int(intersection_id)).strip(),
            parent,
            repair_block,
            task,
            PHYSICAL_REQUIREMENTS.strip(),
            base._OUTPUT.strip(),
        )
    ) + "\n"


@contextmanager
def install_prompt_contract(*, adaptation_module: ModuleType) -> Iterator[None]:
    previous_mutation = adaptation_module.build_universal_lane_mutation_prompt
    sentinel = object()
    previous_init_override = getattr(
        adaptation_module, "_cosydelay_init_prompt_override", sentinel
    )
    adaptation_module._cosydelay_init_prompt_override = build_init_prompt
    adaptation_module.build_universal_lane_mutation_prompt = build_mutation_prompt
    try:
        yield
    finally:
        adaptation_module.build_universal_lane_mutation_prompt = previous_mutation
        if previous_init_override is sentinel:
            delattr(adaptation_module, "_cosydelay_init_prompt_override")
        else:
            adaptation_module._cosydelay_init_prompt_override = previous_init_override
