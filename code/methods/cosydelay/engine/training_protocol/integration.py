"""Compose the seven-rule score over V15 fitting without a physical hard gate."""

from __future__ import annotations

from contextlib import contextmanager
import time
from types import ModuleType
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import pandas as pd

import expression_validation_lane as shared_physics
from expression_rules import (
    coefficient_names,
    parameter_roles,
    validate_candidate_legality,
)
from methods.cosydelay.engine.data_protocol import integration as clean_integration
from methods.cosydelay.engine.evolution_support import integration as v11_integration
from methods.cosydelay.engine.evolution_support.integration import V11Runtime
from methods.cosydelay.engine.numeric_fitting.integration import install_v15_candidate

from .physics import (
    MANUSCRIPT_RULE_NAMES,
    ManuscriptPhysicalScoreResult,
    ManuscriptVerifierConfig,
    PersistentSymbolicLimitEvaluator,
    score_fitted_lanes_manuscript_principlewise,
)
from .fitter import (
    legacy_numeric_r9_probe_disabled,
    scrub_legacy_probe_fields,
)
from .policy import V16_POLICY
from .prompt import install_prompt_contract
from .contract import V16_MAX_PROMPT_EXCLUSIONS, validate_v16_contract


_RULE_FEEDBACK = {
    "R1_required_variables": "R1: include all three declared variables.",
    "R2_nondecreasing_flow": "R2: make the complete response nondecreasing in flow_lane.",
    "R3_nonincreasing_green": "R3: make the complete response nonincreasing in GR_phase.",
    "R4_time_dimension": "R4: make every additive branch Cycle_Time times a dimensionless function.",
    "R5_finite_nonnegative_low_demand_limit": "R5: ensure a finite nonnegative low-demand limit; it need not be zero.",
    "R6_nonnegative_delay": "R6: remove negative, complex, or non-finite values over the domain.",
    "R7_zero_green_limit": "R7: make the complete response approach positive infinity as GR_phase approaches zero.",
}


def validate_unambiguous_coefficient_roles(expression: str):
    """Reject coefficients assigned two incompatible nonlinear bounds.

    A coefficient used both as a fitted power and inside an ``exp`` argument
    would otherwise inherit one bound by precedence even though it controls
    two mechanisms. Requiring separate names makes the predeclared bounds
    unambiguous without inspecting any data or adding a physical rule.
    """
    passed, reason = validate_candidate_legality(str(expression))
    if not passed:
        return passed, reason
    names = coefficient_names(str(expression))
    expected_names = [f"a{index}" for index in range(1, len(names) + 1)]
    if names != expected_names:
        return (
            False,
            "coefficient identifiers must be consecutive beginning at a1 "
            f"(observed: {', '.join(names)})",
        )
    roles = parameter_roles(str(expression))
    nonlinear_roles = {"power_exponent", "exp_coefficient"}
    conflicts = {
        name: sorted(roles.get(name, set()) & nonlinear_roles)
        for name in names
        if len(roles.get(name, set()) & nonlinear_roles) > 1
    }
    if conflicts:
        detail = ", ".join(
            f"{name}={'+'.join(role_names)}"
            for name, role_names in conflicts.items()
        )
        return (
            False,
            "coefficient-role ambiguity: use different coefficients for "
            f"power and exp mechanisms ({detail})",
        )
    return True, "Expression passed grammar and coefficient-role checks"


def principlewise_fitness(accuracy: float, physical_score: float) -> float:
    """The revised paper Fitness, bounded to its declared [0, 2] range."""
    return float(min(max(float(accuracy) + float(physical_score), 0.0), 2.0))


def format_principle_feedback(result: ManuscriptPhysicalScoreResult) -> str:
    compact_names = {
        name: name.split("_", 1)[0] for name in MANUSCRIPT_RULE_NAMES
    }
    vector = ", ".join(
        f"{compact_names[name]}={float(result.rule_scores[name]):.3f}"
        for name in MANUSCRIPT_RULE_NAMES
    )
    header = (
        "SEVEN-COMPONENT PHYSICAL AUDIT "
        "(equal movement aggregation):\n"
        f"- {vector}\n"
        f"- Strict joint pass: {'YES' if result.joint_pass else 'NO'}"
    )
    failed = [
        name
        for name in MANUSCRIPT_RULE_NAMES
        if float(result.rule_scores[name]) < 1.0 - 1e-12
    ]
    if not failed:
        return header + "\n- All seven declared physical principles passed."
    return header + "\nFAILED OR PARTIAL PHYSICAL PRINCIPLES:\n" + "\n".join(
        f"- {_RULE_FEEDBACK[name]} (score={float(result.rule_scores[name]):.3f})"
        for name in failed
    )


def _permissive_runtime_score(
    expr, lane_parameters, lanes, universal_features, config=None
):
    """Bypass the superseded eight-rule scorer before V16 rescoring."""
    del expr, lane_parameters, universal_features, config
    names = shared_physics.PHYSICAL_RULE_NAMES
    scores = {name: 1.0 for name in names}
    return shared_physics.PhysicalScoreResult(
        score=1.0,
        joint_pass=True,
        rule_scores=dict(scores),
        lane_rule_scores={str(lane): dict(scores) for lane in lanes},
        diagnostics={"role": "superseded_runtime_bypass_before_v16_rescore"},
        config=shared_physics.PhysicalVerifierConfig(),
    )


def _permissive_enhanced_audit(*args, **kwargs) -> Dict[str, Any]:
    del args, kwargs
    return {
        "audit_id": "superseded_eight_rule_hard_gate_bypass_for_v16",
        "joint_pass": True,
        "standard_joint_pass": True,
        "standard_rule_scores": {
            name: 1.0 for name in shared_physics.PHYSICAL_RULE_NAMES
        },
        "standard_audit_reused": False,
        "symbolic_endpoint_audit_reused": False,
        "errors": [],
    }


@contextmanager
def install_v16_candidate(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy=V16_POLICY,
    verifier_config: ManuscriptVerifierConfig | None = None,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[V11Runtime]:
    """Install Training-only principle-wise evaluation for one isolated run."""
    validate_v16_contract()
    if policy.optimizer_restarts != 10:
        raise ValueError("V16 retains exactly ten L-BFGS-B restarts")
    if policy.fitted_enhanced_physics_is_hard_gate:
        raise ValueError("V16 strict joint pass is diagnostic, not a hard gate")
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module
    manuscript_config = verifier_config or ManuscriptVerifierConfig()
    limit_evaluator = PersistentSymbolicLimitEvaluator()

    with install_v15_candidate(
        df_train=df_train,
        targets=targets,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        intersection_id=intersection_id,
        policy=policy,
        population_module=population_module,
        adaptation_module=adaptation_module,
    ) as runtime:
        previous_validator = adaptation_module.validate_candidate_expression
        previous_generator = (
            population_module.safe_generate_universal_lane_expression
        )
        previous_score = population_module.score_fitted_lanes_principlewise
        previous_evaluator = population_module.evaluate_expression_with_fitting
        previous_enhanced_audit = clean_integration.audit_fitted_physics
        previous_prompt_exclusions = (
            v11_integration.MAX_NOVELTY_EXAMPLES_IN_PROMPT
        )

        def legality_only(expression: str):
            return validate_unambiguous_coefficient_roles(str(expression))

        def generate_with_explicit_legality_gate(*args, **kwargs):
            # ``True`` selects the dynamically installed validator.  For V16
            # that validator is legality/role-only, not the retained physical
            # prefilter.  Make this explicit here instead of relying on the
            # nesting order of inherited generator wrappers.
            kwargs["enforce_physical_prefilter"] = True
            return previous_generator(*args, **kwargs)

        def evaluate_principlewise(*args, **kwargs):
            with legacy_numeric_r9_probe_disabled():
                result = previous_evaluator(*args, **kwargs)
            _, _, _, parameters, _, details = result
            scrub_legacy_probe_fields(details)
            if details.get("status") != "evaluated" or not parameters:
                return result
            expression = str(args[0] if args else kwargs["expr"])
            compatibility_bypass_physical_wall_seconds = float(
                details.get("physical_wall_seconds", 0.0)
            )
            compatibility_bypass_enhanced_wall_seconds = float(
                details.get("enhanced_physics_wall_seconds", 0.0)
            )
            physical_started = time.perf_counter()
            physical = score_fitted_lanes_manuscript_principlewise(
                expression,
                parameters,
                lanes,
                ["flow_lane", "GR_phase", "Cycle_Time"],
                config=manuscript_config,
                limit_evaluator=limit_evaluator,
            )
            v16_physics_wall_seconds = time.perf_counter() - physical_started
            metrics = details.get("selection_metrics", {}) or {}
            accuracy = float(metrics.get("macro_nonnegative_r2", 0.0))
            fitness = principlewise_fitness(accuracy, physical.score)
            physical_dict = physical.to_dict()
            details.update(
                {
                    "score_mode": "principlewise",
                    "selection_source": "full_training",
                    "selection_accuracy": accuracy,
                    "selected_physical_component": float(physical.score),
                    "paper_fitness_components": {
                        "mean_nonnegative_approach_r2": accuracy,
                        "equal_mean_movement_by_seven_rule_score": float(
                            physical.score
                        ),
                        "lower_bound": 0.0,
                        "upper_bound": 2.0,
                    },
                    "accuracy_supervision_level": "approach",
                    "coefficient_parameterization_level": "movement",
                    "coefficient_calibration_objective": (
                        "joint_approach_error_after_flow_weighted_movement_aggregation"
                    ),
                    "verifier": physical_dict,
                    "clean_enhanced_physics": physical_dict,
                    # The legacy fields initially time only the permissive V15
                    # compatibility hooks.  Relabel those values and make the
                    # generic timing field report the actual seven-rule V16
                    # evaluation.  The enhanced audit reuses this same result,
                    # so it has no second physical-evaluation cost.
                    "compatibility_bypass_physical_wall_seconds": (
                        compatibility_bypass_physical_wall_seconds
                    ),
                    "compatibility_bypass_enhanced_physics_wall_seconds": (
                        compatibility_bypass_enhanced_wall_seconds
                    ),
                    "v16_physics_wall_seconds": float(
                        v16_physics_wall_seconds
                    ),
                    "physical_wall_seconds": float(
                        v16_physics_wall_seconds
                    ),
                    "enhanced_physics_wall_seconds": 0.0,
                    "physical_timing_source": "v16_seven_rule_rescore",
                    "v16_score_reused_as_enhanced_audit": True,
                    "strict_joint_pass_is_diagnostic_only": True,
                    "physical_hard_gate": False,
                    "legacy_numeric_r9_probe_executed": False,
                    "optimizer_restart_selection": "minimum_training_mse",
                    "outer_validation_accessed": False,
                    "test_file_opened": False,
                }
            )
            record = runtime.evaluations.get(expression)
            if record is not None:
                record.update(
                    {
                        "fitness": fitness,
                        "enhanced_physics": physical_dict,
                        "details": details,
                    }
                )
            return (
                float(physical.score),
                accuracy,
                fitness,
                parameters,
                format_principle_feedback(physical),
                details,
            )

        with install_prompt_contract(adaptation_module=adaptation_module):
            adaptation_module.validate_candidate_expression = legality_only
            population_module.safe_generate_universal_lane_expression = (
                generate_with_explicit_legality_gate
            )
            population_module.score_fitted_lanes_principlewise = (
                _permissive_runtime_score
            )
            population_module.evaluate_expression_with_fitting = (
                evaluate_principlewise
            )
            clean_integration.audit_fitted_physics = _permissive_enhanced_audit
            v11_integration.MAX_NOVELTY_EXAMPLES_IN_PROMPT = (
                V16_MAX_PROMPT_EXCLUSIONS
            )
            try:
                yield runtime
            finally:
                limit_evaluator.close()
                v11_integration.MAX_NOVELTY_EXAMPLES_IN_PROMPT = (
                    previous_prompt_exclusions
                )
                clean_integration.audit_fitted_physics = previous_enhanced_audit
                population_module.safe_generate_universal_lane_expression = (
                    previous_generator
                )
                population_module.evaluate_expression_with_fitting = (
                    previous_evaluator
                )
                population_module.score_fitted_lanes_principlewise = previous_score
                adaptation_module.validate_candidate_expression = previous_validator
