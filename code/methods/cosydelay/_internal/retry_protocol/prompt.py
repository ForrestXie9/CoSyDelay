"""Concise cross-batch legality correction layered on the frozen V16 prompt."""

from __future__ import annotations

import re

from methods.cosydelay._internal.training_protocol.prompt import (
    OUTPUT_SCHEMA,
    move_output_schema_last,
)

from .policy import V17_PROMPT_CONTRACT_ID


PROMPT_CONTRACT_ID = V17_PROMPT_CONTRACT_ID
_SUPERVISION_TERMS = re.compile(
    r"\b(?:training|validation|test|fitness|r\s*\^?\s*2|rmse|mae)\b",
    flags=re.IGNORECASE,
)


def _one_line(value: object, *, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def format_cross_batch_legality_correction(
    *, reason: str, expression: str | None, sequence: int
) -> str:
    """Format one legality-only correction without accuracy supervision."""
    clean_reason = _one_line(reason, limit=320)
    clean_expression = _one_line(expression or "[not parseable]", limit=320)
    if not clean_reason:
        clean_reason = "the previous generation batch produced no valid expression"
    if _SUPERVISION_TERMS.search(clean_reason):
        raise ValueError("cross-batch legality feedback contains supervision metrics")
    return "\n".join(
        (
            "CROSS-BATCH LEGALITY CORRECTION:",
            f"Correction sequence: {int(sequence)}.",
            "The immediately preceding generation batch exhausted its output attempts.",
            f"Last rejected expression: {clean_expression}",
            f"Last legality failure: {clean_reason}",
            "Correct this exact legality issue. Return a structurally different formula; "
            "do not repeat the rejected construction or discuss the correction.",
        )
    )


def append_cross_batch_correction(
    prompt: str, *, reason: str, expression: str | None, sequence: int
) -> str:
    """Insert the correction before the single immutable output schema."""
    text = str(prompt)
    if text.count(OUTPUT_SCHEMA) != 1:
        raise RuntimeError("V17 requires exactly one output schema before repair")
    correction = format_cross_batch_legality_correction(
        reason=reason,
        expression=expression,
        sequence=sequence,
    )
    repaired = text.rstrip() + "\n\n" + correction + "\n"
    repaired = move_output_schema_last(repaired)
    if repaired.count(OUTPUT_SCHEMA) != 1 or not repaired.rstrip().endswith(
        OUTPUT_SCHEMA
    ):
        raise RuntimeError("V17 repair failed to preserve output schema last")
    return repaired


def correction_contains_supervision_metrics(text: str) -> bool:
    """Audit helper used by tests and formal result validation."""
    return bool(_SUPERVISION_TERMS.search(str(text)))
