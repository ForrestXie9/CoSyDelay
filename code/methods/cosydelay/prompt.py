"""CoSyDelay keeps the validated V21 prompt and narrows only regeneration wording."""

from __future__ import annotations

from typing import Mapping, Optional

from methods.cosydelay_v21_global_selection import prompt as v21


def build_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    """Initialization is intentionally identical to the valid V21 version."""
    return v21.build_init_prompt(
        feature_explanations,
        universal_features,
        intersection_id,
        include_physical_knowledge,
        prompt_style,
    )


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
    """Request structural exploration without prescribing a formula family."""
    value = v21.build_regeneration_prompt(
        base_expr,
        base_thought,
        base_explanation,
        validation_result,
        regeneration_type,
        universal_features,
        intersection_id,
        include_physical_knowledge,
        prompt_style,
        search_feedback,
    )
    old = (
        "REGENERATION TASK:\n"
        "Generate one compact and sufficiently expressive candidate with a clear "
    )
    new = (
        "REGENERATION TASK:\n"
        "Explore a structurally distinct candidate, not a minimal modification of "
        "the parent. Generate one compact and sufficiently expressive candidate with a clear "
    )
    if old not in value:
        raise RuntimeError("CoSyDelay regeneration insertion point missing")
    return value.replace(old, new, 1)


build_mutation_prompt = build_regeneration_prompt
