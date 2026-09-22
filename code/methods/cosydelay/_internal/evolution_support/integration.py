"""Scoped V11 composition over clean V9 evaluation and V10 fitting."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator, Mapping, Optional, Sequence

import pandas as pd

from methods.cosydelay._internal.data_protocol.integration import (
    CleanRuntime,
    install_clean_single_evolution,
)
from methods.cosydelay._internal.data_protocol.policy import CleanSearchPolicy
from methods.cosydelay._internal.candidate_engine.fitter import (
    PersistentAllLogFitter,
)
from methods.cosydelay._internal.optimizer_conditioning.complexity_prompt import (
    install_accuracy_first_parsimony_contract,
)

from .final_prompt import (
    MAX_NOVELTY_EXAMPLES_IN_PROMPT,
    format_parent_training_feedback,
    install_prompt_contract,
)
from .policy import V11_POLICY


@dataclass
class V11Runtime:
    clean: CleanRuntime
    complexity_audit: list[dict]
    all_log_fitter: PersistentAllLogFitter

    @property
    def evaluations(self):
        return self.clean.evaluations

    @property
    def generation_audit(self):
        return self.clean.generation_audit

    @property
    def prefit_audit(self):
        return self.clean.prefit_audit

    @property
    def fitted_rejections(self):
        return self.clean.fitted_rejections


@contextmanager
def install_v11_candidate(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy: CleanSearchPolicy = V11_POLICY,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[V11Runtime]:
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module
    if policy.structural_diversity_mode != "canonical":
        raise ValueError("V11 requires canonical, not family-unique, diversity")
    if policy.reject_fitted_structural_families:
        raise ValueError("V11 must not blacklist an entire fitted family")
    if policy.optimizer_restarts != 10:
        raise ValueError("V11 requires exactly ten optimizer restarts")

    with install_prompt_contract(adaptation_module=adaptation_module):
        with install_clean_single_evolution(
            df_train=df_train,
            targets=targets,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            intersection_id=intersection_id,
            policy=policy,
            population_module=population_module,
            adaptation_module=adaptation_module,
        ) as clean_runtime:
            previous_fitter = population_module.fit_lane_parameters_to_approaches
            previous_generator = population_module.safe_generate_universal_lane_expression
            clean_runtime.fitter.close()
            with install_accuracy_first_parsimony_contract(
                adaptation_module=adaptation_module,
                append_prompt_note=False,
            ) as complexity_audit:
                with PersistentAllLogFitter(
                    parallel_workers=policy.approach_workers_cap,
                    maxiter=policy.optimizer_maxiter,
                    maxfun=policy.optimizer_maxfun,
                ) as all_log_fitter:

                    def fit_with_clean_parent_state(universal_expr: str, **kwargs):
                        warm = clean_runtime.fitter.parent_warm_parameters(
                            universal_expr
                        )
                        return all_log_fitter.fit(
                            universal_expr=universal_expr,
                            warm_parameters=warm,
                            **kwargs,
                        )

                    def generate_with_training_feedback(*args, **kwargs):
                        mutation_type = kwargs.get("mutation_type")
                        if mutation_type is None and len(args) > 2:
                            mutation_type = args[2]
                        mutation_type = str(mutation_type or "initial")
                        parent = kwargs.get("base_expr")
                        if parent is None and len(args) > 3:
                            parent = args[3]
                        if mutation_type != "initial" and parent:
                            best_fitness = max(
                                (
                                    float(item["fitness"])
                                    for item in clean_runtime.evaluations.values()
                                    if item.get("fitness") is not None
                                ),
                                default=None,
                            )
                            parent_feedback = format_parent_training_feedback(
                                clean_runtime.evaluations.get(str(parent)),
                                current_best_fitness=best_fitness,
                            )
                            existing = str(
                                kwargs.get("search_feedback", "") or ""
                            ).strip()
                            kwargs["search_feedback"] = "\n\n".join(
                                item for item in (parent_feedback, existing) if item
                            )
                        kwargs["max_prompt_exclusions"] = (
                            MAX_NOVELTY_EXAMPLES_IN_PROMPT
                        )
                        return previous_generator(*args, **kwargs)

                    population_module.fit_lane_parameters_to_approaches = (
                        fit_with_clean_parent_state
                    )
                    population_module.safe_generate_universal_lane_expression = (
                        generate_with_training_feedback
                    )
                    try:
                        yield V11Runtime(
                            clean=clean_runtime,
                            complexity_audit=complexity_audit,
                            all_log_fitter=all_log_fitter,
                        )
                    finally:
                        population_module.fit_lane_parameters_to_approaches = (
                            previous_fitter
                        )
                        population_module.safe_generate_universal_lane_expression = (
                            previous_generator
                        )
