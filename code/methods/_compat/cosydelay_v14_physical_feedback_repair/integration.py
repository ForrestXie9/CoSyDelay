"""Compose V14 over V11 with physical-only feedback and explicit tie ranking."""

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Mapping, Optional, Sequence

import pandas as pd

from methods.cosydelay_v9_clean_from_scratch import integration as clean_integration
from methods.cosydelay_v11_total_response_evolution.integration import (
    V11Runtime,
    install_v11_candidate,
)
from methods.cosydelay_v13_unbiased_prompt_tie_break.integration import (
    structure_neutral_repair_guidance,
)
from methods.cosydelay_v13_unbiased_prompt_tie_break.ranking import (
    paper_fitness_training_tie_key,
)

from .policy import V14_POLICY
from .prompt import install_prompt_contract


@contextmanager
def install_v14_candidate(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy=V14_POLICY,
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
        previous_selection_key = population_module.individual_selection_key
        previous_repair = clean_integration._compact_physical_repair_guidance
        with install_prompt_contract(adaptation_module=adaptation_module):
            population_module.individual_selection_key = (
                paper_fitness_training_tie_key
            )
            clean_integration._compact_physical_repair_guidance = (
                structure_neutral_repair_guidance
            )
            try:
                yield runtime
            finally:
                clean_integration._compact_physical_repair_guidance = previous_repair
                population_module.individual_selection_key = previous_selection_key
