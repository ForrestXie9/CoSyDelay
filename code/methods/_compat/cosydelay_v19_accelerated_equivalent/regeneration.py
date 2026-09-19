"""V20 checked-regeneration with a from-scratch initialization fallback."""

from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from typing import Callable, Iterator


REGENERATION_CHECK_ATTEMPTS = 3
INITIALIZATION_FALLBACK_ATTEMPTS = 5
_EXHAUSTED_MESSAGE = "Failed to generate valid expression after all retries"


def with_initialization_fallback(generator: Callable) -> Callable:
    """Retry a rejected regeneration, then fall back to an initial prompt.

    The fallback is used only when all program-side candidate checks are
    exhausted. Provider/API exceptions retain their original failure mode.
    """

    @wraps(generator)
    def checked_regeneration(*args, **kwargs):
        operation = str(
            kwargs.get(
                "mutation_type",
                args[2] if len(args) > 2 else "initial",
            )
        ).lower()
        if operation == "initial":
            return generator(*args, **kwargs)

        regeneration_args = list(args)
        regeneration_kwargs = dict(kwargs)
        if len(regeneration_args) > 8:
            regeneration_args[8] = REGENERATION_CHECK_ATTEMPTS
        else:
            regeneration_kwargs["max_retries"] = REGENERATION_CHECK_ATTEMPTS
        try:
            return generator(*regeneration_args, **regeneration_kwargs)
        except RuntimeError as exc:
            if str(exc) != _EXHAUSTED_MESSAGE:
                raise

        print(
            "  Regeneration checks exhausted; switching to the initialization "
            "prompt for a new structure",
            flush=True,
        )
        initialization_args = list(args)
        initialization_kwargs = dict(kwargs)
        positional_replacements = {
            2: "initial",
            3: None,
            4: None,
            5: None,
            6: None,
            8: INITIALIZATION_FALLBACK_ATTEMPTS,
            13: None,
        }
        keyword_names = {
            2: "mutation_type",
            3: "base_expr",
            4: "base_thought",
            5: "base_explanation",
            6: "validation_result",
            8: "max_retries",
            13: "search_feedback",
        }
        for index, value in positional_replacements.items():
            if len(initialization_args) > index:
                initialization_args[index] = value
                initialization_kwargs.pop(keyword_names[index], None)
            else:
                initialization_kwargs[keyword_names[index]] = value
        return generator(*initialization_args, **initialization_kwargs)

    checked_regeneration.v20_initialization_fallback = True
    return checked_regeneration


@contextmanager
def install_after_compatibility_layers(
    install_candidate: Callable,
    population_module,
    **kwargs,
) -> Iterator[object]:
    """Install the fallback after inherited integrations wrap generation."""

    kwargs["population_module"] = population_module
    with install_candidate(**kwargs) as runtime:
        installed_generator = population_module.safe_generate_universal_lane_expression
        population_module.safe_generate_universal_lane_expression = (
            with_initialization_fallback(installed_generator)
        )
        try:
            yield runtime
        finally:
            population_module.safe_generate_universal_lane_expression = (
                installed_generator
            )
