"""Scoped installation of the clean, one-stage full-Training evolution."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
import time
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence

import pandas as pd

from expression_rules import canonical_expression, structural_family_key
from optimization_lane import calculate_approach_delays_from_universal
from methods.cosydelay._internal.optimizer_parallel4.prefit_integration import (
    install_structural_prefit_gate,
)
from methods.cosydelay._internal.optimizer_parallel4_v3.physics_audit import (
    audit_fitted_physics,
    audit_search_physics,
)

from .clean_fitter import (
    CleanMixedRestartJacobianFitter as MixedRestartJacobianFitter,
)
from .metrics import score_predictions
from .policy import CLEAN_POLICY, CleanSearchPolicy


CLEAN_R9_GENERATION_NOTE = """
V9 MANDATORY COEFFICIENT-ROBUST CONSTRUCTION CONTRACT:
- Use the global skeleton
  Cycle_Time * flow_lane / GR_phase * H(flow_lane, GR_phase; a1, ...).
  Algebraically equivalent factorizations are allowed, but the complete
  expression闁炽儲鏀簅t merely one optional branch闁炽儲鏀穟st retain the explicit
  flow_lane and fixed first-power inverse-GR_phase factors.
- Construct H only from positive, finite sums/products of positive
  coefficients, positive constants, flow_lane, GR_phase, 1+x, log(1+x),
  exp(x), and positive powers whose domains are guaranteed. H must remain
  finite and strictly positive as flow_lane -> 0+ and GR_phase -> 0+.
- Keep H nondecreasing in flow_lane and nonincreasing in GR_phase. Prefer
  demand factors in the numerator and green factors in positive denominators.
- Do not use subtraction, coefficient differences such as a2-1 or 2-a2,
  canceling branches, or denominator shifts such as a2+GR_phase as the sole
  green mechanism. Their signs or endpoint behavior are not coefficient-robust.
- Do not make R9 depend on GR_phase**a2, log(1/GR_phase), or another fitted or
  weak singular exponent. The explicit uncancellable /GR_phase factor must
  establish positive-infinite delay for every allowed positive coefficient.
- Every additive branch must contain flow_lane. Never add a constant or
  green-only delay branch, because y(0, GR_phase, Cycle_Time) must equal zero.
- Never append a fixed post-fit guard. Use 3--6 consecutive coefficient names
  starting at a1 and make each coefficient control a distinct mechanism.
  The absolute identifier limit is a8: a9, a10, and every higher identifier
  are forbidden. Before responding, silently count the distinct coefficients
  and simplify the formula if the count or largest identifier exceeds eight.
- These are structural rules, not a request to repeat one formula. Change H
  meaningfully and obey the separately supplied novelty exclusions.
"""


def _compact_physical_repair_guidance(reason: str) -> str:
    """Translate verifier details into short structural instructions for the LLM."""
    text = str(reason or "").strip()
    lowered = text.lower()
    repairs: List[str] = []
    if "r8" in lowered or "zero flow" in lowered or "zero-flow" in lowered:
        repairs.append(
            "R8: factor flow_lane from the complete expression so every branch "
            "is exactly zero at flow_lane=0."
        )
    if "r9" in lowered or "zero-green" in lowered or "positive-infinity" in lowered:
        repairs.append(
            "R9: keep one explicit uncancellable fixed /GR_phase factor outside "
            "the complete positive demand term; do not use fitted green exponents."
        )
    if "r2" in lowered or "flow_nondecreasing" in lowered or "flow decreases" in lowered:
        repairs.append(
            "R2: use only positive nondecreasing demand factors and remove "
            "subtractive or canceling flow terms."
        )
    if "r3" in lowered or "green_nonincreasing" in lowered or "green ratio increases" in lowered:
        repairs.append(
            "R3: place green-dependent factors only in positive denominators "
            "or other structures that cannot increase with GR_phase."
        )
    if "r7" in lowered or "responsiveness" in lowered or "flat" in lowered:
        repairs.append(
            "R7: retain explicit flow_lane and /GR_phase responses that cannot "
            "collapse to a fitted constant."
        )
    if "negative" in lowered or "nonnegative" in lowered or "non-finite" in lowered:
        repairs.append(
            "Domain: use positive sums/products and strictly positive denominators; "
            "remove subtraction and unsafe log/exp constructions."
        )
    if "fit" in lowered and not repairs:
        repairs.append(
            "Fit stability: use a compact 3--6 coefficient positive structure and "
            "avoid nested exp/log terms or coupled fitted exponents."
        )
    if not repairs:
        repairs.append(
            "Generate a compact structurally different positive expression that "
            "obeys the v9 construction contract."
        )
    return "MANDATORY REPAIR:\n" + "\n".join(f"- {item}" for item in repairs)


@dataclass
class CleanRuntime:
    fitter: MixedRestartJacobianFitter
    prefit_audit: List[Dict[str, Any]]
    fitted_rejections: List[Dict[str, Any]]
    evaluations: Dict[str, Dict[str, Any]]
    generation_audit: List[Dict[str, Any]]
    rejected_family_feedback: Dict[str, str]
    incumbent_injected: bool = False


def _fitness(
    metrics: Mapping[str, float],
    consistency_bonus: float,
    policy: CleanSearchPolicy,
) -> float:
    """Reproduce the paper fitness: clip(R2 + consistency, 0, 2).

    As in the paper implementation, R2 is first clipped to zero separately for
    each approach and then macro-averaged.  The clean v9 hard gate means only
    candidates with a binary consistency bonus of one can enter the population.
    """
    value = metrics["macro_nonnegative_r2"] + float(consistency_bonus)
    return float(
        min(max(value, policy.fitness_lower_bound), policy.fitness_upper_bound)
    )


@contextmanager
def install_clean_single_evolution(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy: CleanSearchPolicy = CLEAN_POLICY,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[CleanRuntime]:
    """Fit, score, and hard-filter from-scratch candidates on Training only.

    There is deliberately no incumbent argument.  Every initialization call is
    delegated to the configured LLM generator.  An offspring may reuse its own
    immediate parent's coefficients in one of the same ten optimizer restart
    slots; no coefficients or expressions enter from an earlier experiment.
    """
    if policy.external_incumbent_allowed:
        raise ValueError("the clean policy must prohibit external incumbents")
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
        previous_prefit_validator = adaptation_module.validate_candidate_expression
        previous_fitter = population_module.fit_lane_parameters_to_approaches
        previous_evaluator = population_module.evaluate_expression_with_fitting
        previous_generator = population_module.safe_generate_universal_lane_expression
        previous_init_prompt = adaptation_module.build_universal_lane_init_prompt
        previous_initialization_batches = (
            population_module.MAX_INITIALIZATION_BATCHES_PER_SLOT
        )
        previous_offspring_batches = population_module.MAX_OFFSPRING_BATCHES_PER_SLOT
        adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = (
            previous_standard + CLEAN_R9_GENERATION_NOTE
        )
        adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = (
            previous_compact + CLEAN_R9_GENERATION_NOTE
        )
        fitted_rejections: List[Dict[str, Any]] = []
        evaluations: Dict[str, Dict[str, Any]] = {}
        generation_audit: List[Dict[str, Any]] = []
        feedback_state = {"text": ""}
        rejected_expressions: List[str] = []
        rejected_canonical = set()
        rejected_family_feedback: Dict[str, str] = {}
        endpoint_audits: Dict[str, Dict[str, Any]] = {}
        population_module.MAX_INITIALIZATION_BATCHES_PER_SLOT = (
            policy.generation_batches_per_population_slot
        )
        population_module.MAX_OFFSPRING_BATCHES_PER_SLOT = (
            policy.generation_batches_per_population_slot
        )

        def clean_endpoint_prefit_validator(expression: str):
            expression = str(expression)
            if policy.reject_fitted_structural_families:
                family = structural_family_key(expression)
                family_feedback = rejected_family_feedback.get(family)
                if family_feedback:
                    reason = (
                        "FITTED STRUCTURAL FAMILY REJECTION: this operator/feature "
                        "skeleton already failed after coefficient fitting. "
                        "Renaming coefficients or reordering terms is not a repair.\n"
                        + family_feedback
                    )
                    prefit_audit.append(
                        {
                            "expression": expression,
                            "passed": False,
                            "reason": reason,
                            "checks": {"fitted_rejected_family_unique": False},
                            "diagnostics": {
                                "rejected_before_symbolic_endpoint_audit": True,
                                "structural_family_key": family,
                            },
                            "gate_id": "fitted_rejected_family_cache_2026_08_11_v1",
                        }
                    )
                    return False, reason
            passed, reason = previous_prefit_validator(expression)
            if not passed:
                return passed, _compact_physical_repair_guidance(reason)
            endpoint = audit_search_physics(
                str(expression), standard_joint_pass=True
            )
            endpoint["dense_audit_deferred_to_cv"] = False
            endpoint["dense_audit_deferred_to_postfit"] = True
            endpoint_audits[expression] = dict(endpoint)
            if prefit_audit and prefit_audit[-1].get("expression") == expression:
                prefit_audit[-1]["coefficient_robust_endpoint_audit"] = endpoint
            if not endpoint["symbolic_r8_exact"]:
                return False, _compact_physical_repair_guidance(
                    "R8 exact symbolic identity was not established"
                )
            if not endpoint["symbolic_r9_positive_infinity"]:
                detail = endpoint.get("symbolic_r9_detail") or (
                    "positive-infinity limit was not established"
                )
                if prefit_audit and prefit_audit[-1].get("expression") == expression:
                    prefit_audit[-1]["passed"] = False
                    prefit_audit[-1]["reason"] = f"R9 pre-fit: {detail}"
                return False, _compact_physical_repair_guidance(
                    f"R9 pre-fit: {detail}"
                )
            return True, (
                "Expression passed structural, exact-R8, and coefficient-robust "
                "symbolic-R9 pre-fit checks"
            )

        adaptation_module.validate_candidate_expression = (
            clean_endpoint_prefit_validator
        )

        def remember_rejection(expression: str, reason: str) -> None:
            expression = str(expression)
            key = canonical_expression(expression)
            if key not in rejected_canonical:
                rejected_canonical.add(key)
                rejected_expressions.append(expression)
            if policy.reject_fitted_structural_families:
                family = structural_family_key(expression)
                rejected_family_feedback.setdefault(
                    family,
                    _compact_physical_repair_guidance(reason),
                )

        def feedback_init_prompt(*args, **kwargs):
            prompt_builder = getattr(
                adaptation_module,
                "_cosydelay_init_prompt_override",
                previous_init_prompt,
            )
            prompt = prompt_builder(*args, **kwargs)
            if not feedback_state["text"]:
                return prompt
            return (
                prompt
                + "\n\nMANDATORY CORRECTION FROM THE PREVIOUS FITTED CANDIDATE:\n"
                + feedback_state["text"]
                + "\nDo not repeat the rejected structural family."
            )

        adaptation_module.build_universal_lane_init_prompt = feedback_init_prompt

        with MixedRestartJacobianFitter(
            parallel_workers=policy.approach_workers_cap,
            maxiter=policy.optimizer_maxiter,
            maxfun=policy.optimizer_maxfun,
        ) as fitter:
            population_module.fit_lane_parameters_to_approaches = fitter

            def full_training_evaluator(
                expr,
                df_train_arg,
                lanes_arg,
                lane_to_approach_arg,
                approach_targets_arg,
                intersection_id_arg,
                prepared_context,
                score_mode="binary",
                physics_weight=1.0,
                verifier_config=None,
                rng=None,
                optimizer_restarts=1,
                df_validation=None,
                validation_targets=None,
                prepared_validation_context=None,
            ):
                if int(optimizer_restarts) != policy.optimizer_restarts:
                    raise ValueError(
                        "clean search requires exactly "
                        f"{policy.optimizer_restarts} optimizer restarts"
                    )
                if df_train_arg is not df_train:
                    raise RuntimeError(
                        "clean evaluator accepts only the installed Training frame"
                    )
                if approach_targets_arg is not targets:
                    raise RuntimeError(
                        "clean evaluator accepts only the installed Training targets"
                    )
                if int(intersection_id_arg) != int(intersection_id):
                    raise RuntimeError("intersection changed inside clean evolution")
                if (
                    df_validation is not None
                    or validation_targets is not None
                    or prepared_validation_context is not None
                ):
                    raise RuntimeError("clean evolution does not accept Validation data")

                base = previous_evaluator(
                    expr,
                    df_train_arg,
                    lanes_arg,
                    lane_to_approach_arg,
                    approach_targets_arg,
                    intersection_id_arg,
                    prepared_context,
                    score_mode=score_mode,
                    physics_weight=physics_weight,
                    verifier_config=verifier_config,
                    rng=rng,
                    optimizer_restarts=optimizer_restarts,
                )
                _, _, _, parameters, feedback, details = base
                if details.get("status") != "evaluated" or not parameters:
                    reason = details.get("reason", "candidate fitting failed")
                    remember_rejection(str(expr), str(reason))
                    fitted_rejections.append(
                        {
                            "expression": str(expr),
                            "stage": "fit",
                            "reason": str(reason),
                            "details": details,
                        }
                    )
                    feedback_state["text"] = _compact_physical_repair_guidance(
                        f"fit failed: {reason}"
                    )
                    raise RuntimeError(feedback_state["text"])

                predictions = calculate_approach_delays_from_universal(
                    df=df_train_arg,
                    universal_expr=expr,
                    lane_parameters=parameters,
                    lanes=list(lanes_arg),
                    lane_to_approach=dict(lane_to_approach_arg),
                    intersection_id=int(intersection_id_arg),
                    prepared_context=prepared_context,
                    strict=True,
                )
                metrics = score_predictions(approach_targets_arg, predictions)
                enhanced_started = time.perf_counter()
                enhanced = audit_fitted_physics(
                    str(expr),
                    parameters,
                    list(lanes_arg),
                    config=verifier_config,
                    precomputed_standard=(
                        details.get("verifier")
                        if policy.reuse_standard_fitted_physics_audit
                        else None
                    ),
                    precomputed_symbolic=(
                        endpoint_audits.get(str(expr))
                        if policy.reuse_prefit_symbolic_endpoint_audit
                        else None
                    ),
                )
                enhanced_wall_seconds = time.perf_counter() - enhanced_started
                consistency_bonus = 1.0 if enhanced["joint_pass"] else 0.0
                fitness = _fitness(metrics, consistency_bonus, policy)
                details.update(
                    {
                        "selection_source": "full_training",
                        "selection_metrics": metrics,
                        "selection_accuracy": float(
                            metrics["macro_nonnegative_r2"]
                        ),
                        "train_r2": float(metrics["macro_nonnegative_r2"]),
                        "paper_fitness_components": {
                            "mean_nonnegative_approach_r2": float(
                                metrics["macro_nonnegative_r2"]
                            ),
                            "binary_physical_consistency": consistency_bonus,
                            "lower_bound": policy.fitness_lower_bound,
                            "upper_bound": policy.fitness_upper_bound,
                        },
                        "train_rmse": float(metrics["macro_rmse"]),
                        "train_mae": float(metrics["macro_mae"]),
                        "outer_validation_accessed": False,
                        "test_file_opened": False,
                        "clean_enhanced_physics": enhanced,
                        "post_evolution_cv_reranking": False,
                        "post_evolution_refit": False,
                        "external_incumbent_used": False,
                        "enhanced_physics_wall_seconds": float(
                            enhanced_wall_seconds
                        ),
                        "efficiency_reuse": {
                            "standard_fitted_physics": bool(
                                enhanced.get("standard_audit_reused", False)
                            ),
                            "prefit_symbolic_endpoint": bool(
                                enhanced.get(
                                    "symbolic_endpoint_audit_reused", False
                                )
                            ),
                        },
                    }
                )
                details.setdefault("verifier", {})["standard_joint_pass"] = bool(
                    details.get("verifier", {}).get("joint_pass", False)
                )
                details["verifier"]["joint_pass"] = bool(enhanced["joint_pass"])
                details["verifier"]["enhanced_audit_id"] = enhanced["audit_id"]

                record = {
                    "expression": str(expr),
                    "parameters": parameters,
                    "metrics": metrics,
                    "fitness": fitness,
                    "enhanced_physics": enhanced,
                    "details": details,
                }
                if not enhanced["joint_pass"]:
                    failed_rules = [
                        str(name)
                        for name, score in enhanced.get(
                            "standard_rule_scores", {}
                        ).items()
                        if float(score) < 1.0
                    ]
                    error_items = list(enhanced.get("errors", []))
                    if failed_rules:
                        error_items.insert(
                            0,
                            "failed standard fitted rules " + ", ".join(failed_rules),
                        )
                    errors = "; ".join(error_items) or (
                        "enhanced fitted-physics audit failed"
                    )
                    remember_rejection(str(expr), errors)
                    record["rejection_reason"] = errors
                    fitted_rejections.append(record)
                    feedback_state["text"] = _compact_physical_repair_guidance(
                        errors
                    )
                    raise RuntimeError(feedback_state["text"])

                evaluations[str(expr)] = record
                feedback_state["text"] = ""
                fitter.remember_parameters(str(expr), parameters)
                return (
                    consistency_bonus,
                    float(metrics["macro_nonnegative_r2"]),
                    float(fitness),
                    parameters,
                    feedback,
                    details,
                )

            def from_scratch_recording_generator(*args, **kwargs):
                kwargs["max_retries"] = (
                    policy.invalid_output_retries_per_generation_batch
                )
                mutation_type = kwargs.get("mutation_type")
                if mutation_type is None and len(args) > 2:
                    mutation_type = args[2]
                mutation_type = str(mutation_type or "initial")
                parent = kwargs.get("base_expr")
                if parent is None and len(args) > 3:
                    parent = args[3]
                if mutation_type == "initial" and parent is not None:
                    raise RuntimeError("initial generation unexpectedly received a parent")
                if mutation_type != "initial" and not parent:
                    raise RuntimeError("offspring generation requires its current parent")
                if feedback_state["text"]:
                    existing = str(kwargs.get("search_feedback", "") or "").strip()
                    kwargs["search_feedback"] = "\n".join(
                        item for item in (existing, feedback_state["text"]) if item
                    )
                exclusions = list(kwargs.get("excluded_expressions", []) or [])
                exclusions.extend(rejected_expressions)
                kwargs["excluded_expressions"] = exclusions
                expression, thought, explanation = previous_generator(*args, **kwargs)
                if canonical_expression(expression) in rejected_canonical:
                    raise RuntimeError(
                        "generator returned a canonical expression already excluded "
                        "for fitted-physics failure"
                    )
                if (
                    policy.reject_fitted_structural_families
                    and structural_family_key(expression)
                    in rejected_family_feedback
                ):
                    raise RuntimeError(
                        "generator returned a structural family already excluded "
                        "for fitted-physics failure"
                    )
                fitter.register_parent(expression, parent)
                generation_audit.append(
                    {
                        "sequence": len(generation_audit) + 1,
                        "expression": str(expression),
                        "canonical_expression": canonical_expression(expression),
                        "mutation_type": mutation_type,
                        "parent_expression": None if parent is None else str(parent),
                        "source": "llm_generated_in_current_run",
                        "external_incumbent": False,
                    }
                )
                return expression, thought, explanation

            population_module.evaluate_expression_with_fitting = full_training_evaluator
            population_module.safe_generate_universal_lane_expression = (
                from_scratch_recording_generator
            )
            runtime = CleanRuntime(
                fitter=fitter,
                prefit_audit=prefit_audit,
                fitted_rejections=fitted_rejections,
                evaluations=evaluations,
                generation_audit=generation_audit,
                rejected_family_feedback=rejected_family_feedback,
            )
            try:
                yield runtime
            finally:
                population_module.fit_lane_parameters_to_approaches = previous_fitter
                population_module.evaluate_expression_with_fitting = previous_evaluator
                population_module.safe_generate_universal_lane_expression = (
                    previous_generator
                )
                adaptation_module.PHYSICAL_REQUIREMENTS_STANDARD = previous_standard
                adaptation_module.PHYSICAL_REQUIREMENTS_COMPACT = previous_compact
                adaptation_module.validate_candidate_expression = (
                    previous_prefit_validator
                )
                adaptation_module.build_universal_lane_init_prompt = previous_init_prompt
                population_module.MAX_INITIALIZATION_BATCHES_PER_SLOT = (
                    previous_initialization_batches
                )
                population_module.MAX_OFFSPRING_BATCHES_PER_SLOT = (
                    previous_offspring_batches
                )
