"""
Evolve a universal lane-level delay expression with lane-specific fitting.
"""
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from constants import feature_explanations
from expression_adaptation_lane import (
    PROMPT_STYLES,
    safe_generate_universal_lane_expression,
)
from expression_rules import (
    canonical_expression,
    structural_family_key,
    validate_candidate_legality,
)
from expression_validation_lane import (
    PhysicalVerifierConfig,
    format_physical_validation_feedback,
    score_fitted_lanes_principlewise,
    validate_fitted_lanes_batch_legacy,
)
from optimization_lane import (
    calculate_approach_delays_from_universal,
    fit_lane_parameters_to_approaches,
    prepare_optimization_context,
)
from residual_guidance_lane import (
    build_direction_invariant_residual_guidance,
)


MAX_INITIALIZATION_BATCHES_PER_SLOT = 10
MAX_OFFSPRING_BATCHES_PER_SLOT = 10
RESIDUAL_GUIDANCE_MODES = {
    "none",
    "shared_regime",
    "anonymous_bottleneck_regime",
}
STRUCTURAL_DIVERSITY_MODES = {
    "canonical",
    "family_unique",
}


def individual_selection_key(individual):
    """Default evolutionary order; scoped methods may add lower-priority ties."""
    return (float(individual.get("fitness", 0.0)),)
TARGETED_PHYSICAL_REPAIR_GUIDANCE = {
    "R2_nondecreasing_flow": (
        "R2 repair: replace subtractive/cancelling flow terms with a structure "
        "whose flow derivative remains nonnegative after coefficient fitting."
    ),
    "R7_operational_responsiveness": (
        "R7 repair: keep flow_lane and GR_phase structurally active so fitted "
        "coefficients cannot collapse either response to an effectively flat curve."
    ),
    "R9_zero_green_limit": (
        "R9 repair: replace the parent green-ratio structure with a "
        "coefficient-robust positive singular dependence as GR_phase approaches "
        "zero; do not append a fixed guard term."
    ),
}


def augment_targeted_physical_feedback(
    feedback: str,
    verifier: Dict,
) -> Tuple[str, List[str]]:
    """Add compact fitted R2/R7/R9 diagnostics without lane identifiers."""
    rule_scores = verifier.get("rule_scores", {})
    failed_rules = [
        name
        for name in TARGETED_PHYSICAL_REPAIR_GUIDANCE
        if float(rule_scores.get(name, 1.0)) < 1.0 - 1e-12
    ]
    if not failed_rules:
        return feedback, []

    lines = [str(feedback).strip(), "Targeted fitted-physics repair guidance:"]
    lane_errors = verifier.get("lane_errors", {})
    for name in failed_rules:
        lines.append(f"- {TARGETED_PHYSICAL_REPAIR_GUIDANCE[name]}")
        prefix = name.split("_", 1)[0] + ":"
        examples = []
        for errors in lane_errors.values():
            for error in errors:
                text = str(error).strip()
                if text.startswith(prefix):
                    detail = text.split(":", 1)[1].strip()
                    if detail and detail not in examples:
                        examples.append(detail)
                if len(examples) >= 2:
                    break
            if len(examples) >= 2:
                break
        for example in examples:
            lines.append(f"  Observed fitted failure: {example}")
    return "\n".join(line for line in lines if line), failed_rules


def validate_expression_basic(expr: str) -> Tuple[bool, str]:
    """Parser/numerical legality only; physics is scored after coefficient fitting."""
    return validate_candidate_legality(expr)


def _score_fitted_expression_on_dataset(
    expr: str,
    df: pd.DataFrame,
    targets: Dict[str, pd.Series],
    lane_parameters: Dict[str, Dict[str, float]],
    lanes: List[str],
    lane_to_approach: Dict[str, str],
    intersection_id: int,
    prepared_context: Dict,
) -> Tuple[float, Dict[str, Optional[float]], Optional[float], Dict[str, Optional[float]]]:
    """Score already-fitted coefficients without changing or refitting them."""
    predictions = calculate_approach_delays_from_universal(
        df=df,
        universal_expr=expr,
        lane_parameters=lane_parameters,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        intersection_id=intersection_id,
        prepared_context=prepared_context,
        strict=True,
    )
    raw_scores: Dict[str, Optional[float]] = {}
    raw_rmse: Dict[str, Optional[float]] = {}
    squared_error_sum = 0.0
    valid_count = 0
    for approach, y_true in targets.items():
        if approach not in predictions:
            continue
        try:
            truth = np.asarray(y_true.values, dtype=float)
            prediction = np.asarray(predictions[approach], dtype=float)
            valid_rows = np.isfinite(truth) & np.isfinite(prediction)
            if np.count_nonzero(valid_rows) < 2:
                raise ValueError("fewer than two defined rows")
            raw_scores[approach] = float(
                r2_score(truth[valid_rows], prediction[valid_rows])
            )
            squared_errors = (
                prediction[valid_rows] - truth[valid_rows]
            ) ** 2
            raw_rmse[approach] = float(np.sqrt(np.mean(squared_errors)))
            squared_error_sum += float(np.sum(squared_errors))
            valid_count += int(np.count_nonzero(valid_rows))
        except Exception:
            raw_scores[approach] = None
            raw_rmse[approach] = None
    finite = [
        value
        for value in raw_scores.values()
        if value is not None and np.isfinite(value)
    ]
    clipped_mean = (
        float(np.mean([max(0.0, value) for value in finite]))
        if finite
        else 0.0
    )
    pooled_rmse = (
        float(np.sqrt(squared_error_sum / valid_count))
        if valid_count
        else None
    )
    return clipped_mean, raw_scores, pooled_rmse, raw_rmse


def evaluate_expression_with_fitting(
    expr: str,
    df_train: pd.DataFrame,
    lanes: List[str],
    lane_to_approach: Dict[str, str],
    approach_targets: Dict[str, pd.Series],
    intersection_id: int,
    prepared_context: Dict,
    score_mode: str = "binary",
    physics_weight: float = 1.0,
    verifier_config: Optional[PhysicalVerifierConfig] = None,
    rng: Optional[np.random.Generator] = None,
    optimizer_restarts: int = 1,
    df_validation: Optional[pd.DataFrame] = None,
    validation_targets: Optional[Dict[str, pd.Series]] = None,
    prepared_validation_context: Optional[Dict] = None,
) -> Tuple[float, float, float, Dict[str, Dict[str, float]], str, Dict]:
    """Fit one expression and compute training fitness plus physical diagnostics.

    Optional validation data is diagnostic only. It never contributes to the
    returned fitness or to evolutionary selection.
    """
    allowed_modes = {"none", "legacy_binary", "binary", "principlewise"}
    if score_mode not in allowed_modes:
        raise ValueError(f"Unknown score_mode {score_mode!r}; expected {sorted(allowed_modes)}")
    if physics_weight < 0:
        raise ValueError("physics_weight must be non-negative")
    started = time.perf_counter()
    is_basic_valid, reason = validate_expression_basic(expr)
    if not is_basic_valid:
        print(f"  Basic validation failed: {reason}")
        details = {
            "score_mode": score_mode,
            "status": "illegal_expression",
            "reason": reason,
            "wall_seconds": time.perf_counter() - started,
        }
        return 0.0, 0.0, 0.0, {}, reason, details

    fit_diagnostics: Dict = {}
    fit_started = time.perf_counter()
    try:
        lane_parameters = fit_lane_parameters_to_approaches(
            universal_expr=expr,
            df=df_train,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            approach_targets=approach_targets,
            intersection_id=intersection_id,
            prepared_context=prepared_context,
            rng=rng,
            n_restarts=optimizer_restarts,
            diagnostics=fit_diagnostics,
        )
    except Exception as exc:
        reason = f"Parameter fitting failed: {exc}"
        print(f"  {reason}")
        details = {
            "score_mode": score_mode,
            "status": "fit_failed",
            "reason": reason,
            "fit": fit_diagnostics,
            "wall_seconds": time.perf_counter() - started,
        }
        return 0.0, 0.0, 0.0, {}, reason, details

    fit_wall_seconds = time.perf_counter() - fit_started
    print(f"  Parameter fitting completed in {fit_wall_seconds:.2f}s")

    physical_started = time.perf_counter()
    physical_result = score_fitted_lanes_principlewise(
        expr,
        lane_parameters,
        lanes,
        ["flow_lane", "GR_phase", "Cycle_Time"],
        config=verifier_config,
    )
    physical_wall_seconds = time.perf_counter() - physical_started
    print(f"  Physical verification completed in {physical_wall_seconds:.2f}s")
    lane_validation_errors = (
        validate_fitted_lanes_batch_legacy(
            expr,
            lane_parameters,
            lanes,
            ["flow_lane", "GR_phase", "Cycle_Time"],
        )
        if score_mode == "legacy_binary"
        else {}
    )
    legacy_errors = [
        f"{lane}: {lane_validation_errors[lane]}"
        for lane in lanes
        if lane in lane_validation_errors
    ]
    strict_errors = [
        f"{lane}: {'; '.join(physical_result.lane_errors[lane])}"
        for lane in lanes
        if lane in physical_result.lane_errors
    ]
    if score_mode == "none":
        physical_component = 0.0
    elif score_mode == "legacy_binary":
        physical_component = 0.0 if legacy_errors else 1.0
    elif score_mode == "binary":
        physical_component = float(physical_result.joint_pass)
    else:
        physical_component = physical_result.score

    displayed_errors = legacy_errors if score_mode == "legacy_binary" else strict_errors
    if displayed_errors and score_mode != "none":
        print(f"  Physical verification found issues for {len(displayed_errors)} lane(s)")
        print(f"  First issue: {displayed_errors[0].splitlines()[0]}")

    prediction_reason = None
    validation_r2 = None
    train_rmse = None
    validation_rmse = None
    train_r2_by_approach = {}
    validation_r2_by_approach = {}
    train_rmse_by_approach = {}
    validation_rmse_by_approach = {}
    try:
        train_r2, train_r2_by_approach, train_rmse, train_rmse_by_approach = (
            _score_fitted_expression_on_dataset(
                expr,
                df_train,
                approach_targets,
                lane_parameters,
                lanes,
                lane_to_approach,
                intersection_id,
                prepared_context,
            )
        )
        if df_validation is not None:
            if not validation_targets:
                raise ValueError("validation_targets are required with df_validation")
            validation_context = prepared_validation_context or prepare_optimization_context(
                df_validation, lanes, lane_to_approach, intersection_id
            )
            (
                validation_r2,
                validation_r2_by_approach,
                validation_rmse,
                validation_rmse_by_approach,
            ) = _score_fitted_expression_on_dataset(
                expr,
                df_validation,
                validation_targets,
                lane_parameters,
                lanes,
                lane_to_approach,
                intersection_id,
                validation_context,
            )

    except Exception as exc:
        prediction_reason = f"Prediction failed on observed data: {exc}"
        print(f"  {prediction_reason}")
        train_r2 = 0.0
        validation_r2 = 0.0 if df_validation is not None else None
        train_rmse = None
        validation_rmse = None
        physical_component = 0.0

    selection_accuracy = train_r2
    fitness = selection_accuracy + physics_weight * physical_component
    if score_mode == "none":
        validation_feedback = "Verifier disabled for selection; diagnostics retained"
    elif prediction_reason:
        validation_feedback = format_physical_validation_feedback(
            physical_result,
            prediction_reason=prediction_reason,
        )
    else:
        validation_feedback = format_physical_validation_feedback(physical_result)

    details = {
        "score_mode": score_mode,
        "status": "evaluated",
        "physics_weight": float(physics_weight),
        "selection_source": "training",
        "validation_role": "diagnostic_only_not_used_for_fitness",
        "selected_physical_component": float(physical_component),
        "legacy_joint_pass": (
            not legacy_errors if score_mode == "legacy_binary" else None
        ),
        "train_r2": float(train_r2),
        "validation_r2": None if validation_r2 is None else float(validation_r2),
        "train_rmse": None if train_rmse is None else float(train_rmse),
        "validation_rmse": (
            None if validation_rmse is None else float(validation_rmse)
        ),
        "selection_accuracy": float(selection_accuracy),
        "train_r2_by_approach": train_r2_by_approach,
        "validation_r2_by_approach": validation_r2_by_approach,
        "train_rmse_by_approach": train_rmse_by_approach,
        "validation_rmse_by_approach": validation_rmse_by_approach,
        "verifier": physical_result.to_dict(),
        "fit": fit_diagnostics,
        "fit_wall_seconds": float(fit_wall_seconds),
        "physical_wall_seconds": float(physical_wall_seconds),
        "wall_seconds": time.perf_counter() - started,
    }
    return (
        physical_component,
        train_r2,
        fitness,
        lane_parameters,
        validation_feedback,
        details,
    )


def evolve_universal_lane_expression(
    df_train: pd.DataFrame,
    lanes: List[str],
    lane_to_approach: Dict[str, str],
    approach_targets: Dict[str, pd.Series],
    universal_features: List[str],
    generations: int,
    pop_size: int,
    intersection_id: int = 1,
    score_mode: str = "binary",
    physics_weight: float = 1.0,
    prompt_knowledge: bool = True,
    seed: int = 42,
    optimizer_restarts: int = 1,
    verifier_config: Optional[PhysicalVerifierConfig] = None,
    df_validation: Optional[pd.DataFrame] = None,
    validation_targets: Optional[Dict[str, pd.Series]] = None,
    prompt_style: str = "standard",
    max_wall_seconds: Optional[float] = None,
    residual_guidance_mode: str = "none",
    structural_diversity_mode: str = "canonical",
    use_feasible_archive: bool = False,
    early_stop_feasible_count: Optional[int] = None,
    early_stop_train_gap: Optional[float] = None,
    targeted_physical_feedback: bool = False,
) -> Tuple[str, str, str, Dict, Dict]:
    """
    Evolve a universal lane expression with integrated fitting and evaluation.

    Returns:
        best_expr: Best expression.
        best_thought: Thought process.
        best_explanation: Explanation.
        best_lane_parameters: Fitted parameters for the best expression.
        history: Evolution history.
    """
    prompt_style = str(prompt_style).strip().lower()
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(
            f"Unknown prompt_style {prompt_style!r}; expected one of {PROMPT_STYLES}"
        )
    if max_wall_seconds is not None and max_wall_seconds <= 0:
        raise ValueError("max_wall_seconds must be positive when provided")
    if early_stop_feasible_count is not None and early_stop_feasible_count < 1:
        raise ValueError("early_stop_feasible_count must be positive when supplied")
    if early_stop_train_gap is not None and early_stop_train_gap < 0:
        raise ValueError("early_stop_train_gap must be non-negative when supplied")
    if (
        early_stop_feasible_count is not None
        or early_stop_train_gap is not None
    ) and not use_feasible_archive:
        raise ValueError("training-side early stopping requires use_feasible_archive=True")
    if (early_stop_feasible_count is None) != (early_stop_train_gap is None):
        raise ValueError(
            "early_stop_feasible_count and early_stop_train_gap must be supplied together"
        )
    residual_guidance_mode = str(residual_guidance_mode).strip().lower()
    if residual_guidance_mode not in RESIDUAL_GUIDANCE_MODES:
        raise ValueError(
            f"Unknown residual_guidance_mode {residual_guidance_mode!r}; "
            f"expected one of {sorted(RESIDUAL_GUIDANCE_MODES)}"
        )
    structural_diversity_mode = str(structural_diversity_mode).strip().lower()
    if structural_diversity_mode not in STRUCTURAL_DIVERSITY_MODES:
        raise ValueError(
            f"Unknown structural_diversity_mode {structural_diversity_mode!r}; "
            f"expected one of {sorted(STRUCTURAL_DIVERSITY_MODES)}"
        )

    print(f"\n{'=' * 70}")
    print("EVOLVING UNIVERSAL LANE EXPRESSION")
    print(f"{'=' * 70}")
    print(f"Population size: {pop_size}")
    print(f"Generations: {generations}")
    print(f"Lanes: {len(lanes)}")
    print(f"Approaches: {list(approach_targets.keys())}")
    print(f"Physical score mode: {score_mode}")
    print(f"Prompt knowledge: {prompt_knowledge}")
    print(f"Prompt style: {prompt_style}")
    print(f"Random seed: {seed}")
    print(f"Final-only validation evaluation: {df_validation is not None}")
    print(f"Residual guidance: {residual_guidance_mode}")
    print(f"Structural diversity: {structural_diversity_mode}")
    print(f"Feasible archive: {use_feasible_archive}")
    print(f"Targeted physical feedback: {targeted_physical_feedback}")
    if early_stop_feasible_count is not None:
        print(
            "Training-side early stop: "
            f"feasible_count>={early_stop_feasible_count}, "
            f"train_R2_gap<={early_stop_train_gap:.4f}"
        )
    if max_wall_seconds is not None:
        print(f"Wall-time budget: {max_wall_seconds:.1f}s")

    population = []
    history = []
    seen_canonical_keys = set()
    seen_family_keys = set()
    rng = np.random.default_rng(seed)
    next_candidate_id = 1
    evolution_started = time.perf_counter()
    wall_budget_exhausted = False
    completed_generations = 0
    evaluated_candidates = []
    feasible_archive = {}
    early_stop_triggered = False
    early_stop_reason = None
    early_stop_diagnostics = {}

    def wall_elapsed() -> float:
        return time.perf_counter() - evolution_started

    def wall_budget_remaining() -> bool:
        if max_wall_seconds is None:
            return True
        return wall_elapsed() < float(max_wall_seconds)
    prepared_context = prepare_optimization_context(
        df_train, lanes, lane_to_approach, intersection_id
    )
    if df_validation is not None:
        if not validation_targets:
            raise ValueError("validation_targets are required when df_validation is provided")

    def evaluate_candidate(candidate_expr: str):
        # Frozen validation is deliberately absent here: every generation and
        # every parent/offspring decision must depend on training fitness only.
        return evaluate_expression_with_fitting(
            candidate_expr,
            df_train,
            lanes,
            lane_to_approach,
            approach_targets,
            intersection_id,
            prepared_context,
            score_mode=score_mode,
            physics_weight=physics_weight,
            verifier_config=verifier_config,
            rng=rng,
            optimizer_restarts=optimizer_restarts,
        )

    def training_side_stop_status():
        if early_stop_feasible_count is None or early_stop_train_gap is None:
            return False, {}
        feasible = list(feasible_archive.values())
        if not evaluated_candidates or not feasible:
            return False, {
                "feasible_count": len(feasible),
                "required_feasible_count": early_stop_feasible_count,
                "train_r2_gap": None,
                "maximum_train_r2_gap": early_stop_train_gap,
            }
        best_any = max(
            float(item.get("train_r2", 0.0)) for item in evaluated_candidates
        )
        best_feasible = max(
            float(item.get("train_r2", 0.0)) for item in feasible
        )
        train_gap = max(0.0, best_any - best_feasible)
        diagnostics = {
            "feasible_count": len(feasible),
            "required_feasible_count": early_stop_feasible_count,
            "best_train_r2": best_any,
            "best_feasible_train_r2": best_feasible,
            "train_r2_gap": train_gap,
            "maximum_train_r2_gap": early_stop_train_gap,
        }
        passed = (
            len(feasible) >= early_stop_feasible_count
            and train_gap <= early_stop_train_gap
        )
        return passed, diagnostics

    def build_individual(expr, thought, explanation, evaluation):
        nonlocal next_candidate_id
        consistency, train_r2, fitness, lane_params, feedback, details = evaluation
        verifier = details.get("verifier", {})
        targeted_failed_rules = []
        if targeted_physical_feedback:
            feedback, targeted_failed_rules = augment_targeted_physical_feedback(
                feedback, verifier
            )
        residual_guidance = {
            "schema_version": 1,
            "status": "disabled",
            "prompt_text": "",
            "source_split": "coefficient_fit",
            "approach_labels_exposed_to_prompt": False,
        }
        if (
            residual_guidance_mode != "none"
            and lane_params
            and details.get("status") == "evaluated"
        ):
            try:
                fit_predictions = calculate_approach_delays_from_universal(
                    df=df_train,
                    universal_expr=expr,
                    lane_parameters=lane_params,
                    lanes=lanes,
                    lane_to_approach=lane_to_approach,
                    intersection_id=intersection_id,
                    prepared_context=prepared_context,
                    strict=True,
                )
                residual_guidance = (
                    build_direction_invariant_residual_guidance(
                        approach_targets,
                        fit_predictions,
                        prepared_context,
                        include_stable_bottleneck=(
                            residual_guidance_mode
                            == "anonymous_bottleneck_regime"
                        ),
                    )
                )
            except Exception as exc:
                residual_guidance = {
                    "schema_version": 1,
                    "status": "diagnostic_failed",
                    "prompt_text": "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "source_split": "coefficient_fit",
                    "approach_labels_exposed_to_prompt": False,
                }
        details["residual_guidance"] = residual_guidance
        details["targeted_physical_feedback"] = {
            "enabled": bool(targeted_physical_feedback),
            "failed_rules": targeted_failed_rules,
        }
        individual = {
            "candidate_id": next_candidate_id,
            "expr": expr,
            "thought": thought,
            "explanation": explanation,
            # Compatibility name retained; this is the component selected by score_mode.
            "consistency_bonus": consistency,
            # This is the component actually used by fitness. In binary mode
            # it is strictly 0 or 1. Keep the fractional verifier summary only
            # as an audit diagnostic.
            "physical_score": float(consistency),
            "principlewise_diagnostic_score": verifier.get("score", 0.0),
            "physical_joint_pass": verifier.get("joint_pass", False),
            "rule_scores": verifier.get("rule_scores", {}),
            "train_r2": train_r2,
            "validation_r2": details.get("validation_r2"),
            "train_rmse": details.get("train_rmse"),
            "validation_rmse": details.get("validation_rmse"),
            "selection_accuracy": details.get("selection_accuracy", train_r2),
            "fitness": fitness,
            "lane_parameters": lane_params,
            "validation_feedback": feedback,
            "evaluation_details": details,
            "residual_guidance": residual_guidance,
        }
        next_candidate_id += 1
        evaluated_candidates.append(individual)
        if use_feasible_archive and individual["physical_joint_pass"]:
            archive_key = canonical_expression(individual["expr"])
            previous = feasible_archive.get(archive_key)
            if (
                previous is None
                or float(individual["train_r2"]) > float(previous["train_r2"])
            ):
                feasible_archive[archive_key] = individual
        return individual

    def history_record(individual, generation, slot, event="evaluated", **extra):
        record = {
            "generation": generation,
            "individual": slot,
            "event": event,
            "candidate_id": individual.get("candidate_id"),
            "expression": individual.get("expr"),
            "consistency_bonus": individual.get("consistency_bonus", 0.0),
            "physical_score": individual.get("physical_score", 0.0),
            "principlewise_diagnostic_score": individual.get(
                "principlewise_diagnostic_score", 0.0
            ),
            "physical_joint_pass": individual.get("physical_joint_pass", False),
            "rule_scores": individual.get("rule_scores", {}),
            "train_r2": individual.get("train_r2", 0.0),
            "validation_r2": individual.get("validation_r2"),
            "train_rmse": individual.get("train_rmse"),
            "validation_rmse": individual.get("validation_rmse"),
            "selection_accuracy": individual.get("selection_accuracy", 0.0),
            "fitness": individual.get("fitness", 0.0),
            "validation_feedback": individual.get("validation_feedback", ""),
            "evaluation_details": individual.get("evaluation_details", {}),
            "residual_guidance": individual.get("residual_guidance", {}),
            "residual_guidance_mode": residual_guidance_mode,
            "structural_diversity_mode": structural_diversity_mode,
            "score_mode": score_mode,
            "prompt_knowledge": prompt_knowledge,
            "prompt_style": prompt_style,
            "seed": seed,
            "max_wall_seconds": max_wall_seconds,
            "wall_elapsed_seconds": wall_elapsed(),
        }
        record.update(extra)
        return record

    print(f"\n{'=' * 70}")
    print("INITIALIZATION")
    print(f"{'=' * 70}")

    while len(population) < pop_size:
        if not wall_budget_remaining():
            wall_budget_exhausted = True
            print(
                f"\nWall-time budget exhausted during initialization "
                f"({wall_elapsed():.1f}s >= {max_wall_seconds:.1f}s); "
                f"population so far: {len(population)}/{pop_size}"
            )
            break
        slot_index = len(population) + 1
        slot_batches = 0
        initialized = False
        slot_excluded = [item["expr"] for item in population]
        while not initialized:
            if not wall_budget_remaining():
                wall_budget_exhausted = True
                break
            slot_batches += 1
            if slot_batches > MAX_INITIALIZATION_BATCHES_PER_SLOT:
                raise RuntimeError(
                    f"Unable to fill initialization slot {slot_index}/{pop_size} after "
                    f"{MAX_INITIALIZATION_BATCHES_PER_SLOT} generation batches; "
                    "aborting instead of continuing with an incomplete population"
                )
            try:
                expr, thought, explanation = safe_generate_universal_lane_expression(
                    feature_explanations,
                    universal_features,
                    mutation_type="initial",
                    intersection_id=intersection_id,
                    include_physical_knowledge=prompt_knowledge,
                    enforce_physical_prefilter=False,
                    prompt_style=prompt_style,
                    excluded_expressions=slot_excluded,
                )

                expression_key = canonical_expression(expr)
                if expression_key in seen_canonical_keys:
                    slot_excluded.append(expr)
                    raise ValueError(
                        "algebraically equivalent to an already evaluated expression"
                    )
                family_key = structural_family_key(expr)
                if (
                    structural_diversity_mode == "family_unique"
                    and family_key in seen_family_keys
                ):
                    slot_excluded.append(expr)
                    raise ValueError(
                        "same coefficient-agnostic structural family as an "
                        "already evaluated expression"
                    )

                print(f"\n[Individual {slot_index}/{pop_size}]")
                print(f"Expression: {expr}")

                individual = build_individual(
                    expr, thought, explanation, evaluate_candidate(expr)
                )
                seen_canonical_keys.add(expression_key)
                seen_family_keys.add(family_key)
                population.append(individual)

                history.append(history_record(individual, 0, slot_index))

                print(
                    f"Joint pass: {individual['physical_joint_pass']}, "
                    f"Physical score: {individual['physical_score']:.4f}, "
                    f"Train R2: {individual['train_r2']:.4f}, "
                    f"Selection score: {individual['selection_accuracy']:.4f}, "
                    f"Fitness: {individual['fitness']:.4f}"
                )
                initialized = True

            except Exception as exc:
                print(
                    f"Initialization slot {slot_index}/{pop_size} generation batch "
                    f"{slot_batches} failed: {exc}; regenerating the same slot"
                )
        if wall_budget_exhausted:
            break

    if not population:
        raise RuntimeError(
            "Wall-time budget exhausted before any initialization candidate "
            "could be evaluated"
        )
    if max_wall_seconds is None and len(population) != pop_size:
        raise RuntimeError(
            f"Population initialization invariant failed: {len(population)} != {pop_size}"
        )

    print(f"\nInitialized {len(population)} individuals")

    early_stop_triggered, early_stop_diagnostics = training_side_stop_status()
    if early_stop_triggered:
        early_stop_reason = "training_feasible_archive_criterion_after_initialization"
        print(
            "Training-side early stop after initialization: "
            f"feasible={early_stop_diagnostics['feasible_count']}, "
            f"train_R2_gap={early_stop_diagnostics['train_r2_gap']:.4f}"
        )

    for gen in range(generations):
        if early_stop_triggered:
            break
        if not wall_budget_remaining():
            wall_budget_exhausted = True
            print(
                f"\nWall-time budget exhausted before generation {gen + 1}/{generations} "
                f"({wall_elapsed():.1f}s >= {max_wall_seconds:.1f}s)"
            )
            break
        print(f"\n{'=' * 70}")
        print(f"GENERATION {gen + 1}/{generations}")
        print(f"{'=' * 70}")

        population = sorted(population, key=individual_selection_key, reverse=True)
        best = population[0]

        print("\nBest individual:")
        print(f"  Expression: {best['expr']}")
        print(f"  Fitness: {best['fitness']:.4f}")
        print(f"  Physical score: {best['physical_score']:.4f}")
        print(f"  Joint pass: {best['physical_joint_pass']}")
        print(f"  Train R2: {best['train_r2']:.4f}")
        if best["validation_r2"] is not None:
            print(f"  Validation R2: {best['validation_r2']:.4f}")

        valid_count = sum(1 for ind in population if ind["physical_joint_pass"])
        avg_r2 = np.mean([ind["train_r2"] for ind in population])
        print("\nPopulation statistics:")
        print(f"  Valid expressions: {valid_count}/{len(population)}")
        print(f"  Average train R2: {avg_r2:.4f}")

        new_population = []
        occupied_keys = {canonical_expression(item["expr"]) for item in population}
        offspring_keys = set()
        for i, parent in enumerate(population):
            if not wall_budget_remaining():
                wall_budget_exhausted = True
                print(
                    f"\nWall-time budget exhausted during generation {gen + 1} "
                    f"at offspring slot {i + 1}/{len(population)} "
                    f"({wall_elapsed():.1f}s >= {max_wall_seconds:.1f}s)"
                )
                break
            mutation_type = "large"
            offspring_created = False
            slot_batches = 0
            slot_excluded = (
                [item["expr"] for item in population]
                + [item["expr"] for item in new_population]
            )

            while not offspring_created:
                if not wall_budget_remaining():
                    wall_budget_exhausted = True
                    break
                slot_batches += 1
                if slot_batches > MAX_OFFSPRING_BATCHES_PER_SLOT:
                    print(
                        f"  Unable to create a novel offspring for parent "
                        f"{i + 1}/{pop_size} after "
                        f"{MAX_OFFSPRING_BATCHES_PER_SLOT} batches; keeping parent"
                    )
                    history.append(history_record(
                        parent,
                        gen + 1,
                        i + 1,
                        event="duplicate_exhausted",
                        mutation_type=mutation_type,
                        parent_candidate_id=parent["candidate_id"],
                    ))
                    break

                try:
                    if slot_batches == 1:
                        print(f"\n[Offspring {i + 1}/{pop_size}] (regeneration)")

                    validation_result = (
                        (
                            parent["physical_joint_pass"],
                            parent.get("validation_feedback", "Valid"),
                        )
                        if prompt_knowledge
                        else (True, "Physical feedback withheld by experiment design")
                    )

                    child_expr, child_thought, child_explanation = (
                        safe_generate_universal_lane_expression(
                            feature_explanations,
                            universal_features,
                            mutation_type,
                            parent["expr"],
                            parent["thought"],
                            parent["explanation"],
                            validation_result,
                            intersection_id,
                            include_physical_knowledge=prompt_knowledge,
                            enforce_physical_prefilter=False,
                            prompt_style=prompt_style,
                            excluded_expressions=slot_excluded,
                            search_feedback=parent.get(
                                "residual_guidance", {}
                            ).get("prompt_text", ""),
                        )
                    )

                    if slot_batches == 1:
                        print(f"Expression: {child_expr}")
                    child_key = canonical_expression(child_expr)
                    child_family_key = structural_family_key(child_expr)
                    if (
                        child_key in occupied_keys
                        or child_key in offspring_keys
                        or child_key in seen_canonical_keys
                        or (
                            structural_diversity_mode == "family_unique"
                            and child_family_key in seen_family_keys
                        )
                    ):
                        slot_excluded.append(child_expr)
                        print(
                            "  Canonical duplicate after generation; "
                            f"regenerating (batch {slot_batches})"
                        )
                        continue

                    child = build_individual(
                        child_expr,
                        child_thought,
                        child_explanation,
                        evaluate_candidate(child_expr),
                    )
                    seen_canonical_keys.add(child_key)
                    seen_family_keys.add(child_family_key)
                    new_population.append(child)
                    offspring_keys.add(child_key)

                    history.append(history_record(
                        child,
                        gen + 1,
                        i + 1,
                        mutation_type=mutation_type,
                        parent_candidate_id=parent["candidate_id"],
                    ))

                    print(
                        f"Joint pass: {child['physical_joint_pass']}, "
                        f"Physical score: {child['physical_score']:.4f}, "
                        f"Train R2: {child['train_r2']:.4f}, "
                        f"Selection score: {child['selection_accuracy']:.4f}, "
                        f"Fitness: {child['fitness']:.4f}"
                    )
                    offspring_created = True

                except Exception as exc:
                    print(
                        f"  Offspring batch {slot_batches} for parent "
                        f"{i + 1}/{pop_size} failed: {exc}; regenerating"
                    )

        if new_population:
            combined = population + new_population
            unique = {}
            for item in combined:
                key = (
                    structural_family_key(item["expr"])
                    if structural_diversity_mode == "family_unique"
                    else canonical_expression(item["expr"])
                )
                if (
                    key not in unique
                    or individual_selection_key(item)
                    > individual_selection_key(unique[key])
                ):
                    unique[key] = item
            population = sorted(
                unique.values(), key=individual_selection_key, reverse=True
            )[:pop_size]
        completed_generations = gen + 1
        early_stop_triggered, early_stop_diagnostics = training_side_stop_status()
        if early_stop_triggered:
            early_stop_reason = (
                "training_feasible_archive_criterion_after_generation_"
                f"{completed_generations}"
            )
            print(
                "Training-side early stop: "
                f"feasible={early_stop_diagnostics['feasible_count']}, "
                f"train_R2_gap={early_stop_diagnostics['train_r2_gap']:.4f}"
            )
            break
        if wall_budget_exhausted:
            break

    if use_feasible_archive and feasible_archive:
        best = max(
            feasible_archive.values(),
            key=lambda item: float(item.get("train_r2", 0.0)),
        )
        final_selection_policy = "best_training_r2_from_physical_feasible_archive"
    else:
        best = sorted(population, key=individual_selection_key, reverse=True)[0]
        final_selection_policy = "best_training_fitness_from_surviving_population"
    total_wall_seconds = wall_elapsed()
    history.append(
        {
            "event": "search_budget_summary",
            "max_wall_seconds": max_wall_seconds,
            "wall_budget_exhausted": bool(wall_budget_exhausted),
            "completed_generations": int(completed_generations),
            "population_size_final": len(population),
            "wall_elapsed_seconds": float(total_wall_seconds),
            "score_mode": score_mode,
            "residual_guidance_mode": residual_guidance_mode,
            "structural_diversity_mode": structural_diversity_mode,
            "seed": seed,
            "selection_source": "training",
            "validation_accessed_during_evolution": False,
            "candidate_evaluations": len(evaluated_candidates),
            "use_feasible_archive": bool(use_feasible_archive),
            "feasible_archive_size": len(feasible_archive),
            "feasible_archive_candidate_ids": sorted(
                int(item["candidate_id"])
                for item in feasible_archive.values()
            ),
            "final_selection_policy": final_selection_policy,
            "final_candidate_id": int(best["candidate_id"]),
            "final_expression": best["expr"],
            "early_stop_triggered": bool(early_stop_triggered),
            "early_stop_reason": early_stop_reason,
            "early_stop_diagnostics": early_stop_diagnostics,
            "targeted_physical_feedback": bool(targeted_physical_feedback),
        }
    )

    if df_validation is not None:
        validation_started = time.perf_counter()
        validation_error = None
        try:
            final_validation_context = prepare_optimization_context(
                df_validation, lanes, lane_to_approach, intersection_id
            )
            (
                validation_r2,
                validation_r2_by_approach,
                validation_rmse,
                validation_rmse_by_approach,
            ) = _score_fitted_expression_on_dataset(
                best["expr"],
                df_validation,
                validation_targets,
                best["lane_parameters"],
                lanes,
                lane_to_approach,
                intersection_id,
                final_validation_context,
            )
        except Exception as exc:
            validation_r2 = None
            validation_r2_by_approach = {}
            validation_rmse = None
            validation_rmse_by_approach = {}
            validation_error = f"{type(exc).__name__}: {exc}"

        validation_wall_seconds = time.perf_counter() - validation_started
        best["validation_r2"] = validation_r2
        best["validation_rmse"] = validation_rmse
        # History records intentionally retain their generation-time snapshot.
        # Copy before attaching final-only evaluation so validation cannot
        # appear retroactively inside an evolutionary candidate record.
        best["evaluation_details"] = dict(best["evaluation_details"])
        best["evaluation_details"]["validation_r2"] = validation_r2
        best["evaluation_details"]["validation_rmse"] = validation_rmse
        best["evaluation_details"]["validation_r2_by_approach"] = (
            validation_r2_by_approach
        )
        best["evaluation_details"]["validation_rmse_by_approach"] = (
            validation_rmse_by_approach
        )
        best["evaluation_details"]["validation_role"] = (
            "final_best_only_not_used_for_fitness"
        )
        best["evaluation_details"]["validation_wall_seconds"] = float(
            validation_wall_seconds
        )
        best["evaluation_details"]["validation_error"] = validation_error
        history.append(
            {
                "event": "final_validation_evaluation",
                "candidate_id": best["candidate_id"],
                "expression": best["expr"],
                "selection_source": "training",
                "fitness_unchanged_by_validation": True,
                "validation_r2": validation_r2,
                "validation_rmse": validation_rmse,
                "validation_r2_by_approach": validation_r2_by_approach,
                "validation_rmse_by_approach": validation_rmse_by_approach,
                "validation_wall_seconds": float(validation_wall_seconds),
                "error": validation_error,
            }
        )

    print(f"\n{'=' * 70}")
    print("FINAL BEST EXPRESSION")
    print(f"{'=' * 70}")
    print(f"Expression: {best['expr']}")
    print(f"Fitness: {best['fitness']:.4f}")
    print(f"Physical score: {best['physical_score']:.4f}")
    print(f"Physical joint pass: {best['physical_joint_pass']}")
    print(f"Train R2: {best['train_r2']:.4f}")
    if best["validation_r2"] is not None:
        print(f"Validation R2: {best['validation_r2']:.4f}")
    if max_wall_seconds is not None:
        print(
            f"Wall budget: exhausted={wall_budget_exhausted}, "
            f"elapsed={total_wall_seconds:.1f}s / {max_wall_seconds:.1f}s, "
            f"completed_generations={completed_generations}/{generations}"
        )
    print(f"Explanation: {best['explanation']}")

    print("\nLane parameters:")
    for lane, params in best["lane_parameters"].items():
        approach = lane_to_approach[lane]
        param_str = ", ".join([f"{key}={value:.4f}" for key, value in params.items()])
        print(f"  {lane} ({approach}): {param_str}")

    return best["expr"], best["thought"], best["explanation"], best["lane_parameters"], history
