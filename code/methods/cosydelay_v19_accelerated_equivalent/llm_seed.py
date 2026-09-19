"""Deterministic per-call LLM seed scheduling for V20."""

from __future__ import annotations

from contextlib import contextmanager
import os
from types import ModuleType
from typing import Iterator


@contextmanager
def install_reproducible_seed_schedule(
    adaptation_module: ModuleType,
    *,
    base_seed: int,
    intersection_id: int,
) -> Iterator[dict]:
    """Use a distinct reproducible provider seed for every LLM call."""

    previous_run_llm = adaptation_module.run_llm
    previous_environment_seed = os.environ.get("LLM_SEED")
    state = {
        "strategy": "base_plus_intersection_100000_plus_call_index",
        "base_seed": int(base_seed),
        "intersection_id": int(intersection_id),
        "calls": 0,
    }

    def run_llm_with_scheduled_seed(prompt: str, *args, **kwargs):
        state["calls"] += 1
        call_seed = (
            int(base_seed)
            + int(intersection_id) * 100_000
            + int(state["calls"])
        )
        old_call_seed = os.environ.get("LLM_SEED")
        os.environ["LLM_SEED"] = str(call_seed)
        try:
            return previous_run_llm(prompt, *args, **kwargs)
        finally:
            if old_call_seed is None:
                os.environ.pop("LLM_SEED", None)
            else:
                os.environ["LLM_SEED"] = old_call_seed

    adaptation_module.run_llm = run_llm_with_scheduled_seed
    try:
        yield state
    finally:
        adaptation_module.run_llm = previous_run_llm
        if previous_environment_seed is None:
            os.environ.pop("LLM_SEED", None)
        else:
            os.environ["LLM_SEED"] = previous_environment_seed
