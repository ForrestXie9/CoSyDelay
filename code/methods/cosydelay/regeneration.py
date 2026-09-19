"""One broad-regeneration retry policy for CoSyDelay.

The retained V20 overlay changed from regeneration to initialization after a
short failed sequence.  CoSyDelay deliberately does not mix those two operations:
all rejected offspring attempts use the same parent-conditioned broad
regeneration prompt and the same five-attempt budget.  A later outer offspring
draw may choose another parent, but no retry changes prompt type.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Iterator

from methods.uniform_regeneration_contract_v1 import (
    REGENERATION_ATTEMPTS,
    wrap_uniform_regeneration,
)


REGENERATION_ATTEMPTS = 5


def with_uniform_regeneration(generator: Callable) -> Callable:
    return wrap_uniform_regeneration(
        generator, marker="cosydelay_uniform_regeneration"
    )


@contextmanager
def install_after_compatibility_layers(
    install_candidate: Callable,
    population_module,
    **kwargs,
) -> Iterator[object]:
    """Install the one-policy retry wrapper after inherited wrappers."""
    kwargs["population_module"] = population_module
    with install_candidate(**kwargs) as runtime:
        original = population_module.safe_generate_universal_lane_expression
        population_module.safe_generate_universal_lane_expression = (
            with_uniform_regeneration(original)
        )
        try:
            yield runtime
        finally:
            population_module.safe_generate_universal_lane_expression = original
