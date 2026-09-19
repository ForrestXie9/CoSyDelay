"""Narrow, auditable expression normalization used only by V20."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator


def normalize_python_power_operator(expression: str | None) -> str | None:
    """Convert mathematical caret powers to the allowed Python ``**`` form."""

    if expression is None or "^" not in expression:
        return expression
    return str(expression).replace("^", "**")


@contextmanager
def install_expression_normalization(
    adaptation_module: ModuleType,
) -> Iterator[dict]:
    """Normalize only the parsed expression, preserving the raw LLM archive."""

    previous_parser = adaptation_module.parse_llm_response
    state = {"caret_power_conversions": 0}

    def parse_with_normalization(response: str):
        expression, thought, explanation = previous_parser(response)
        normalized = normalize_python_power_operator(expression)
        if normalized != expression:
            state["caret_power_conversions"] += 1
            print("  Normalized power operator: ^ -> **", flush=True)
        return normalized, thought, explanation

    adaptation_module.parse_llm_response = parse_with_normalization
    try:
        yield state
    finally:
        adaptation_module.parse_llm_response = previous_parser
