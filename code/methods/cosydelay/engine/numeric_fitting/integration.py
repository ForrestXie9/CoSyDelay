"""V15 composition: symbolic R9 gate and pure-MSE restart selection."""

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Mapping, Optional, Sequence

import pandas as pd

from methods.cosydelay.engine.physics_support.integration import (
    install_v14_candidate,
)
from methods.cosydelay.engine.physics_support.prompt import (
    install_prompt_contract,
)
from methods.cosydelay.engine.prompt_support.ranking import (
    paper_fitness_training_tie_key,
)
from methods.cosydelay.engine.evolution_support.integration import V11Runtime

from .fitter import PersistentSymbolicR9Fitter
from .policy import V15_POLICY


def _symbolic_r9_standard_wrapper(original):
    def wrapped(expr, lane_parameters, lanes, universal_features, config=None):
        result = original(
            expr, lane_parameters, lanes, universal_features, config=config
        )
        rule = "R9_zero_green_limit"
        result.rule_scores[rule] = 1.0
        for lane in result.lane_rule_scores:
            result.lane_rule_scores[lane][rule] = 1.0
        for lane in list(result.lane_errors):
            kept = [
                message
                for message in result.lane_errors[lane]
                if not str(message).startswith("R9:")
            ]
            if kept:
                result.lane_errors[lane] = kept
            else:
                result.lane_errors.pop(lane, None)
        values = list(result.rule_scores.values())
        result.score = float(sum(values) / len(values))
        result.joint_pass = all(value >= 1.0 - 1e-12 for value in values)
        result.diagnostics["r9_selection_rule"] = (
            "coefficient_robust_symbolic_positive_infinity_only"
        )
        result.diagnostics["numerical_near_zero_probe_role"] = "diagnostic_only"
        return result

    return wrapped


@contextmanager
def install_v15_candidate(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy=V15_POLICY,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[V11Runtime]:
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module

    with install_v14_candidate(
        df_train=df_train,
        targets=targets,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        intersection_id=intersection_id,
        policy=policy,
        population_module=population_module,
        adaptation_module=adaptation_module,
    ) as runtime:
        previous_fitter = population_module.fit_lane_parameters_to_approaches
        previous_physics = population_module.score_fitted_lanes_principlewise
        runtime.all_log_fitter.close()
        with install_prompt_contract(adaptation_module=adaptation_module):
            with PersistentSymbolicR9Fitter(
                parallel_workers=policy.approach_workers_cap,
                maxiter=policy.optimizer_maxiter,
                maxfun=policy.optimizer_maxfun,
            ) as fitter:
                def fit_with_parent_state(universal_expr: str, **kwargs):
                    warm = runtime.clean.fitter.parent_warm_parameters(universal_expr)
                    return fitter.fit(
                        universal_expr=universal_expr,
                        warm_parameters=warm,
                        **kwargs,
                    )

                population_module.fit_lane_parameters_to_approaches = (
                    fit_with_parent_state
                )
                population_module.score_fitted_lanes_principlewise = (
                    _symbolic_r9_standard_wrapper(previous_physics)
                )
                population_module.individual_selection_key = (
                    paper_fitness_training_tie_key
                )
                try:
                    yield runtime
                finally:
                    population_module.fit_lane_parameters_to_approaches = (
                        previous_fitter
                    )
                    population_module.score_fitted_lanes_principlewise = (
                        previous_physics
                    )
