"""Compose V13 over V11 without changing fitting, physics, or paper Fitness."""

from contextlib import contextmanager
import builtins
from types import ModuleType
from typing import Iterator, Mapping, Optional, Sequence

import pandas as pd

from methods.cosydelay._internal.data_protocol import integration as clean_integration
from methods.cosydelay._internal.evolution_support.integration import (
    V11Runtime,
    install_v11_candidate,
)

from .policy import V13_POLICY
from .prompt import install_prompt_contract
from .ranking import paper_fitness_training_tie_key


_BASE_REPAIR_GUIDANCE = clean_integration._compact_physical_repair_guidance


def structure_neutral_repair_guidance(reason: str) -> str:
    """Keep retry feedback physical without prescribing one formula family."""
    guidance = _BASE_REPAIR_GUIDANCE(reason)
    return guidance.replace(
        "R9: keep one explicit uncancellable fixed /GR_phase factor outside "
        "the complete positive demand term; do not use fitted green exponents.",
        "R9: make the complete expression itself approach positive infinity as "
        "GR_phase approaches zero for every allowed positive coefficient; use "
        "any safe structure and do not impose a mandatory global factorization.",
    ).replace(
        "R7: retain explicit flow_lane and /GR_phase responses that cannot "
        "collapse to a fitted constant.",
        "R7: retain non-degenerate flow and green responses that cannot collapse "
        "to a fitted constant; do not impose a mandatory global factorization.",
    ).replace(
        "obeys the v9 construction contract.",
        "obeys the stated complete-response physical requirements.",
    )


@contextmanager
def install_v13_candidate(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy=V13_POLICY,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[V11Runtime]:
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module

    with install_v11_candidate(
        df_train=df_train,
        targets=targets,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        intersection_id=intersection_id,
        policy=policy,
        population_module=population_module,
        adaptation_module=adaptation_module,
    ) as runtime:
        previous_sorted = getattr(population_module, "sorted", None)
        previous_repair = clean_integration._compact_physical_repair_guidance

        def training_tie_sorted(iterable, *, key=None, reverse=False):
            items = list(iterable)
            if (
                key is not None
                and items
                and all(isinstance(item, Mapping) and "fitness" in item for item in items)
            ):
                try:
                    fitness_only = all(
                        float(key(item)) == float(item["fitness"]) for item in items
                    )
                except (TypeError, ValueError, KeyError):
                    fitness_only = False
                if fitness_only:
                    key = paper_fitness_training_tie_key
            return builtins.sorted(items, key=key, reverse=reverse)

        with install_prompt_contract(adaptation_module=adaptation_module):
            population_module.sorted = training_tie_sorted
            clean_integration._compact_physical_repair_guidance = (
                structure_neutral_repair_guidance
            )
            try:
                yield runtime
            finally:
                clean_integration._compact_physical_repair_guidance = previous_repair
                if previous_sorted is None:
                    delattr(population_module, "sorted")
                else:
                    population_module.sorted = previous_sorted
