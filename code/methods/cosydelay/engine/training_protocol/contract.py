"""Machine-checkable frozen contract for prospective V16 experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

from expression_rules import (
    ALLOWED_FUNCTION_NAMES,
    FEATURE_NAMES,
    MAX_ABS_NUMERIC_LITERAL,
    MAX_COEFFICIENTS,
    MAX_EXPRESSION_CHARACTERS,
    MAX_EXPRESSION_NODES,
    MAX_PYTHON_AST_NODES,
    PREFIT_FLOW_POINTS,
    PREFIT_GREEN_POINTS,
)
from llm_config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_SEED,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT,
    DEFAULT_TOP_P,
    JIEKOU_API_ENDPOINT,
    JIEKOU_MODEL,
    JIEKOU_REQUEST_PATH,
)

from .physics import (
    MANUSCRIPT_RULE_NAMES,
    ManuscriptVerifierConfig,
    _generate_v16_test_points,
)
from .policy import V16_POLICY
from .prompt import PROMPT_CONTRACT_ID


V16_MAX_PROMPT_EXCLUSIONS = 3


@dataclass(frozen=True)
class V16MethodContract:
    schema_version: int = 10
    method_id: str = V16_POLICY.method_id
    population: int = 10
    generations: int = 10
    search_split: str = "Training_only"
    accuracy_supervision_level: str = "approach"
    coefficient_parameterization_level: str = "movement"
    coefficient_calibration_objective: str = (
        "sum_of_approach_MSE_after_flow_weighted_movement_aggregation"
    )
    accuracy_aggregation: str = "mean_nonnegative_R2_over_observed_approaches"
    physical_aggregation: str = (
        "equal_mean_over_declared_movements_and_seven_rules"
    )
    optimizer: str = "L-BFGS-B_all_positive_log_coordinates"
    optimizer_restarts: int = 10
    optimizer_maxiter: int = 200
    optimizer_maxfun: int = 20_000
    optimizer_maxcor: int = 10
    optimizer_ftol: float = 2.220446049250313e-9
    optimizer_gtol: float = 1e-5
    optimizer_maxls: int = 20
    restart_selection: str = "minimum_Training_MSE"
    initial_candidate_restart_composition: str = (
        "nine_retained_local_starts_plus_one_role_aware_Sobol_start"
    )
    mutation_candidate_restart_composition: str = (
        "when_parent_coefficients_overlap_one_current_run_parent_warm_start_"
        "plus_eight_retained_local_starts_plus_one_role_aware_Sobol_start;_"
        "otherwise_same_as_initial_candidate"
    )
    parent_warm_start_source: str = (
        "immediate_parent_coefficients_from_same_intersection_same_Training_run"
    )
    parent_warm_start_replaces_a_restart_not_added: bool = True
    external_warm_start_allowed: bool = False
    legacy_numeric_r9_probe_executed: bool = False
    approach_workers_cap: int = 4
    coefficient_bounds_scale: Tuple[float, float] = (0.001, 1000.0)
    coefficient_bounds_power: Tuple[float, float] = (0.05, 5.0)
    coefficient_bounds_exponential: Tuple[float, float] = (0.0001, 1.0)
    parameter_boundary_diagnostic_coordinates: str = (
        "normalized_natural_log_bound_interval"
    )
    parameter_boundary_diagnostic_used_for_selection: bool = False
    mixed_nonlinear_coefficient_roles_allowed: bool = False
    coefficient_identifiers_consecutive_from_a1: bool = True
    prefit_filter_scope: str = (
        "parser_grammar_sampled_numerical_legality_and_unambiguous_"
        "coefficient_roles_only"
    )
    prefit_physical_principle_rejection_enabled: bool = False
    prefit_sampled_screen_is_global_domain_proof: bool = False
    prefit_representative_positive_coefficient: float = 1.0
    prefit_representative_cycle_seconds: float = 120.0
    prefit_log_and_fitted_power_flow_points: Tuple[float, ...] = (
        PREFIT_FLOW_POINTS
    )
    prefit_log_and_fitted_power_green_points: Tuple[float, ...] = (
        PREFIT_GREEN_POINTS
    )
    prefit_flow_zero_role: str = "analytic_low_demand_boundary_only"
    prefit_denominator_flow_domain: Tuple[float, float] = (0.001, 2.0)
    prefit_denominator_green_domain: Tuple[float, float] = (0.02, 0.95)
    prefit_denominator_points_per_axis: int = 65
    prefit_denominator_near_zero_absolute_tolerance: float = 1e-10
    variables: Tuple[str, ...] = FEATURE_NAMES
    binary_operators: Tuple[str, ...] = ("+", "-", "*", "/", "**")
    unary_functions: Tuple[str, ...] = ALLOWED_FUNCTION_NAMES
    maximum_coefficients: int = MAX_COEFFICIENTS
    maximum_symbolic_nodes: int = MAX_EXPRESSION_NODES
    maximum_python_ast_nodes: int = MAX_PYTHON_AST_NODES
    maximum_expression_characters: int = MAX_EXPRESSION_CHARACTERS
    maximum_absolute_numeric_literal: float = MAX_ABS_NUMERIC_LITERAL
    coefficients_positive: bool = True
    canonical_duplicate_detection: bool = True
    full_duplicate_archive_checked: bool = True
    maximum_duplicate_examples_in_prompt: int = V16_MAX_PROMPT_EXCLUSIONS
    instantiated_prompt_output_schema_position: str = "last"
    invalid_outputs_per_generation_batch: int = 5
    generation_batches_per_population_slot: int = 30
    transport_attempts_per_output_attempt: int = 5
    provider_host: str = JIEKOU_API_ENDPOINT
    provider_api_style: str = "OpenAI_compatible_chat_completions"
    request_path: str = JIEKOU_REQUEST_PATH
    requested_model_alias: str = JIEKOU_MODEL
    exact_model_snapshot: str = "unknown_not_exposed_by_provider"
    temperature: None = DEFAULT_TEMPERATURE
    top_p: None = DEFAULT_TOP_P
    maximum_output_tokens: None = DEFAULT_MAX_TOKENS
    omitted_sampling_fields_use_provider_defaults: bool = True
    requested_seed: int = DEFAULT_SEED
    provider_seed_support_verified: bool = False
    request_timeout_seconds: float = DEFAULT_TIMEOUT
    llm_attempt_log_format: str = "JSONL_one_record_per_transport_attempt"
    instantiated_prompt_content_archived: bool = True
    normalized_assistant_response_content_archived: bool = True
    raw_provider_response_envelope_archived: bool = False
    prompt_and_response_sha256_archived: bool = True
    api_key_or_authorization_archived: bool = False
    physical_rule_names: Tuple[str, ...] = MANUSCRIPT_RULE_NAMES
    lhs_grid_points_per_movement: int = 128
    joint_domain_corner_anchors: int = 8
    unique_numerical_grid_points_per_movement: int = 137
    time_dimension_verification: str = (
        "exact_symbolic_degree_one_homogeneity_without_coefficient_substitution"
    )
    fitted_component_analytic_limit_evaluation: str = (
        "always_exact_decimal_rational_fitted_coefficients_with_8s_timeout_"
        "fail_closed;_redundant_generic_positive_parameter_checks_not_executed"
    )
    strict_joint_pass_role: str = "diagnostic_only"
    mutation_receives_compact_parent_seven_rule_feedback: bool = True
    legacy_eight_rule_targeted_feedback_augmenter_enabled: bool = False
    symbolic_limit_operation: str = "sympy.limit_in_isolated_process"
    symbolic_limit_worker_lifetime: str = (
        "persistent_per_search_restarted_after_timeout_or_protocol_failure"
    )
    physical_score_reused_as_enhanced_audit: bool = True
    validation_or_test_used_for_selection: bool = False
    population_update: str = "mu_plus_lambda_global_elitist_top_m"

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["verifier"] = asdict(ManuscriptVerifierConfig())
        return payload


V16_CONTRACT = V16MethodContract()


def validate_v16_contract() -> None:
    """Fail before a run if executable settings drift from the declaration."""
    expected_policy = {
        "population": V16_CONTRACT.population,
        "generations": V16_CONTRACT.generations,
        "optimizer": V16_CONTRACT.optimizer,
        "optimizer_restarts": V16_CONTRACT.optimizer_restarts,
        "optimizer_maxiter": V16_CONTRACT.optimizer_maxiter,
        "optimizer_maxfun": V16_CONTRACT.optimizer_maxfun,
        "approach_workers_cap": V16_CONTRACT.approach_workers_cap,
        "invalid_output_retries_per_generation_batch": (
            V16_CONTRACT.invalid_outputs_per_generation_batch
        ),
        "generation_batches_per_population_slot": (
            V16_CONTRACT.generation_batches_per_population_slot
        ),
        "llm_transport_attempts_per_output_attempt": (
            V16_CONTRACT.transport_attempts_per_output_attempt
        ),
        "llm_model_requested": V16_CONTRACT.requested_model_alias,
        "llm_temperature": V16_CONTRACT.temperature,
        "llm_top_p": V16_CONTRACT.top_p,
        "llm_max_tokens": V16_CONTRACT.maximum_output_tokens,
        "llm_seed_requested": V16_CONTRACT.requested_seed,
        "validation_or_test_used_for_selection": False,
        "fitted_enhanced_physics_is_hard_gate": False,
        "targeted_physical_feedback": False,
        "parent_coefficient_inheritance": True,
        "external_incumbent_allowed": False,
    }
    drift = {
        name: {"declared": expected, "observed": getattr(V16_POLICY, name)}
        for name, expected in expected_policy.items()
        if getattr(V16_POLICY, name) != expected
    }
    expected_bounds = {
        "scale": V16_CONTRACT.coefficient_bounds_scale,
        "power_exponent": V16_CONTRACT.coefficient_bounds_power,
        "exp_coefficient": V16_CONTRACT.coefficient_bounds_exponential,
    }
    if V16_POLICY.coefficient_bounds != expected_bounds:
        drift["coefficient_bounds"] = {
            "declared": expected_bounds,
            "observed": V16_POLICY.coefficient_bounds,
        }
    if V16_POLICY.execution_contract_version != (
        "training_only_manuscript_principlewise_v10"
    ):
        drift["execution_contract_version"] = {
            "declared": "training_only_manuscript_principlewise_v10",
            "observed": V16_POLICY.execution_contract_version,
        }
    if V16_POLICY.prompt_contract_version != PROMPT_CONTRACT_ID:
        drift["prompt_contract_version"] = {
            "declared": PROMPT_CONTRACT_ID,
            "observed": V16_POLICY.prompt_contract_version,
        }
    from .fitter import FROZEN_LBFGSB_OPTIONS

    expected_lbfgsb_options = {
        "maxcor": V16_CONTRACT.optimizer_maxcor,
        "ftol": V16_CONTRACT.optimizer_ftol,
        "gtol": V16_CONTRACT.optimizer_gtol,
        "maxls": V16_CONTRACT.optimizer_maxls,
    }
    if FROZEN_LBFGSB_OPTIONS != expected_lbfgsb_options:
        drift["lbfgsb_options"] = {
            "declared": expected_lbfgsb_options,
            "observed": FROZEN_LBFGSB_OPTIONS,
        }
    verifier = ManuscriptVerifierConfig()
    generated_points = _generate_v16_test_points(
        verifier.n_grid_samples,
        verifier.lhs_seed,
        verifier.flow_domain,
        verifier.green_domain,
        verifier.cycle_domain_seconds,
    )
    expected_verifier_grid = {
        "lhs_grid_points_per_movement": verifier.n_grid_samples,
        "joint_domain_corner_anchors": 8,
        "unique_numerical_grid_points_per_movement": len(generated_points) - 1,
    }
    for name, observed in expected_verifier_grid.items():
        expected = getattr(V16_CONTRACT, name)
        if observed != expected:
            drift[name] = {"declared": expected, "observed": observed}
    if drift:
        raise RuntimeError(f"V16 executable contract drift: {drift}")
