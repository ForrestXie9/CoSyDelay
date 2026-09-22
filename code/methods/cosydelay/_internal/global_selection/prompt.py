"""Formal V21 prompt: V20 prompt plus only the declared numeric domain."""
from __future__ import annotations

from typing import Mapping, Optional

from methods.cosydelay._internal.accelerated_search import paper_prompt as base


_OLD = "Delay is real, finite, and nonnegative for valid positive inputs."
_NEW = (
    "Delay is real, finite, and nonnegative throughout "
    "0.001<=flow_lane<=2.0, 0.02<=GR_phase<=0.95, and "
    "30<=Cycle_Time<=240."
)


def _add_numeric_domain(value: str) -> str:
    if value.count(_OLD) != 1:
        raise RuntimeError("V21 numeric-domain insertion point missing")
    return value.replace(_OLD, _NEW, 1)


def build_init_prompt(
    feature_explanations: Mapping[str, str],
    universal_features: list[str],
    intersection_id: int = 1,
    include_physical_knowledge: bool = True,
    prompt_style: str = "standard",
) -> str:
    return _add_numeric_domain(
        base.build_init_prompt(
            feature_explanations, universal_features, intersection_id,
            include_physical_knowledge, prompt_style,
        )
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
    return _add_numeric_domain(
        base.build_regeneration_prompt(
            base_expr, base_thought, base_explanation, validation_result,
            regeneration_type, universal_features, intersection_id,
            include_physical_knowledge, prompt_style, search_feedback,
        )
    )


build_mutation_prompt = build_regeneration_prompt

