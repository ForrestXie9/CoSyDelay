"""Scoped v3 integration; the retained v2 runtime remains untouched."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import pandas as pd

from methods.cosydelay._internal.optimizer_parallel4.prefit_integration import (
    install_structural_prefit_gate,
)

from .fitter import MixedRestartJacobianFitter
from .policy import V3_POLICY, V3SearchPolicy
from .training_selection import TrainingOnlySelectionController


R9_GENERATION_NOTE = """
V3 COEFFICIENT-ROBUST PHYSICS GUIDANCE:
- Satisfy R8 structurally; do not add an offset independent of flow_lane.
- Make the R9 +infinity limit follow from the generated structure for every
  positive fitted coefficient. Do not append a fixed guard term.
- Prefer compact positive families such as flow_lane**p / GR_phase**r,
  flow_lane**p * log(1 + a/GR_phase), or
  flow_lane**p * (exp(a/GR_phase) - 1), multiplied by Cycle_Time.
- Avoid cancellation between green-dependent terms and avoid fitted shifts in
  a denominator that can remove the zero-green singularity.
"""


@dataclass
class V3Runtime:
    fitter: MixedRestartJacobianFitter
    selector: TrainingOnlySelectionController
    prefit_audit: List[Dict[str, Any]]


@contextmanager
def install_v3_evolution(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    seed: int,
    policy: V3SearchPolicy = V3_POLICY,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[V3Runtime]:
    """Install v3 fitter, prompt guidance, and Training-only selector."""
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module

    with install_structural_prefit_gate(
        population_module=population_module,
        adaptation_module=adaptation_module,
    ) as prefit_audit:
        previous_standard = adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD
        previous_compact = adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT
        previous_fitter = population_module.fit_lane_parameters_to_approaches
        adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = (
            previous_standard + R9_GENERATION_NOTE
        )
        adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = (
            previous_compact + R9_GENERATION_NOTE
        )
        with MixedRestartJacobianFitter(
            parallel_workers=policy.approach_workers_cap
        ) as fitter:
            selector = TrainingOnlySelectionController(
                df_train=df_train,
                targets=targets,
                lanes=lanes,
                lane_to_approach=lane_to_approach,
                intersection_id=intersection_id,
                fitter=fitter,
                seed=seed,
                policy=policy,
            )
            population_module.fit_lane_parameters_to_approaches = fitter
            try:
                with selector.install(population_module):
                    yield V3Runtime(
                        fitter=fitter,
                        selector=selector,
                        prefit_audit=prefit_audit,
                    )
            finally:
                population_module.fit_lane_parameters_to_approaches = previous_fitter
                adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = previous_standard
                adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = previous_compact
