"""Training-only holdout evolution and four-fold candidate reranking."""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import time
from types import ModuleType
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from optimization_lane import (
    calculate_approach_delays_from_universal,
    prepare_optimization_context,
)

from .fitter import MixedRestartJacobianFitter, V3_RESTARTS
from .physics_audit import audit_fitted_physics, audit_search_physics
from .policy import V3_POLICY, V3SearchPolicy


def balanced_regression_folds(
    targets: Mapping[str, pd.Series],
    *,
    n_folds: int,
    seed: int,
) -> np.ndarray:
    """Deterministically distribute the multi-output target range across folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    arrays = [np.asarray(series.values, dtype=float) for series in targets.values()]
    if not arrays:
        raise ValueError("at least one target is required")
    row_count = len(arrays[0])
    if row_count < n_folds:
        raise ValueError("row count must be at least the number of folds")
    if any(len(values) != row_count for values in arrays):
        raise ValueError("target lengths differ")
    standardized = []
    for values in arrays:
        finite = values[np.isfinite(values)]
        center = float(np.median(finite)) if finite.size else 0.0
        scale = float(np.std(finite)) if finite.size else 1.0
        scale = max(scale, 1e-12)
        standardized.append(np.nan_to_num((values - center) / scale, nan=0.0))
    severity = np.mean(np.vstack(standardized), axis=0)
    tie_keys = [
        hashlib.sha256(f"{seed}|{index}".encode("utf-8")).hexdigest()
        for index in range(row_count)
    ]
    order = sorted(range(row_count), key=lambda i: (severity[i], tie_keys[i]))
    assignments = np.empty(row_count, dtype=int)
    # A snake assignment avoids systematically placing every bin edge in the
    # same fold while keeping fold sizes balanced to one row.
    for rank, index in enumerate(order):
        block, offset = divmod(rank, n_folds)
        fold = offset if block % 2 == 0 else n_folds - 1 - offset
        assignments[index] = fold
    return assignments


def score_predictions(
    targets: Mapping[str, pd.Series | np.ndarray],
    predictions: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    """Return macro R2/RMSE/MAE and dimensionless error penalties."""
    by_approach: Dict[str, Dict[str, float]] = {}
    pooled_truth = []
    pooled_prediction = []
    for approach, target in targets.items():
        if approach not in predictions:
            continue
        truth = np.asarray(
            target.values if hasattr(target, "values") else target,
            dtype=float,
        )
        prediction = np.asarray(predictions[approach], dtype=float)
        valid = np.isfinite(truth) & np.isfinite(prediction)
        if np.count_nonzero(valid) < 2:
            continue
        truth = truth[valid]
        prediction = prediction[valid]
        pooled_truth.append(truth)
        pooled_prediction.append(prediction)
        raw_r2 = float(r2_score(truth, prediction))
        rmse = float(np.sqrt(mean_squared_error(truth, prediction)))
        mae = float(mean_absolute_error(truth, prediction))
        scale = max(float(np.std(truth)), 1e-12)
        by_approach[str(approach)] = {
            "r2": raw_r2,
            "clipped_r2": max(0.0, raw_r2),
            "rmse": rmse,
            "mae": mae,
            "normalized_rmse": rmse / scale,
            "normalized_mae": mae / scale,
            "rows": int(len(truth)),
        }
    if not by_approach:
        return {
            "macro_r2": 0.0,
            "macro_raw_r2": 0.0,
            "macro_rmse": float("inf"),
            "macro_mae": float("inf"),
            "pooled_rmse": float("inf"),
            "pooled_mae": float("inf"),
            "normalized_rmse": float("inf"),
            "normalized_mae": float("inf"),
            "by_approach": {},
        }
    records = list(by_approach.values())
    pooled_truth_array = np.concatenate(pooled_truth)
    pooled_prediction_array = np.concatenate(pooled_prediction)
    return {
        "macro_r2": float(np.mean([item["clipped_r2"] for item in records])),
        "macro_raw_r2": float(np.mean([item["r2"] for item in records])),
        "macro_rmse": float(np.mean([item["rmse"] for item in records])),
        "macro_mae": float(np.mean([item["mae"] for item in records])),
        "pooled_rmse": float(
            np.sqrt(mean_squared_error(pooled_truth_array, pooled_prediction_array))
        ),
        "pooled_mae": float(
            mean_absolute_error(pooled_truth_array, pooled_prediction_array)
        ),
        "normalized_rmse": float(
            np.mean([item["normalized_rmse"] for item in records])
        ),
        "normalized_mae": float(
            np.mean([item["normalized_mae"] for item in records])
        ),
        "by_approach": by_approach,
    }


def composite_accuracy(metrics: Mapping[str, float], policy: V3SearchPolicy) -> float:
    return float(
        policy.r2_weight * metrics["macro_r2"]
        - policy.normalized_rmse_weight * metrics["normalized_rmse"]
        - policy.normalized_mae_weight * metrics["normalized_mae"]
    )


def _stable_fit_seed(seed: int, expression: str, fold: int, stage: str) -> int:
    digest = hashlib.sha256(
        f"{seed}|{fold}|{stage}|{expression}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def select_search_candidates(
    records: Sequence[Mapping[str, Any]],
    *,
    top_k: int,
) -> list[Mapping[str, Any]]:
    """Reserve one CV slot for a retained incumbent, when available."""
    if top_k < 1:
        raise ValueError("top_k must be positive")
    eligible = [
        item
        for item in records
        if item["standard_physical_pass"] and item["enhanced_physical_pass"]
    ]
    incumbents = sorted(
        (
            item
            for item in eligible
            if item.get("candidate_source") == "retained_v2_incumbent"
        ),
        key=lambda item: item["search_composite"],
        reverse=True,
    )
    generated = sorted(
        (
            item
            for item in eligible
            if item.get("candidate_source") != "retained_v2_incumbent"
        ),
        key=lambda item: item["search_composite"],
        reverse=True,
    )
    selected = incumbents[:1]
    selected.extend(generated[: max(0, top_k - len(selected))])
    return selected


class TrainingOnlySelectionController:
    """Patch one evolution run to use only an internal Training holdout."""

    def __init__(
        self,
        *,
        df_train: pd.DataFrame,
        targets: Mapping[str, pd.Series],
        lanes: Sequence[str],
        lane_to_approach: Mapping[str, str],
        intersection_id: int,
        fitter: MixedRestartJacobianFitter,
        seed: int,
        policy: V3SearchPolicy = V3_POLICY,
    ) -> None:
        self.df_train = df_train.reset_index(drop=True)
        self.targets = {
            str(name): pd.Series(np.asarray(value.values, dtype=float))
            for name, value in targets.items()
        }
        self.lanes = [str(lane) for lane in lanes]
        self.lane_to_approach = {
            str(lane): str(approach)
            for lane, approach in lane_to_approach.items()
        }
        self.intersection_id = int(intersection_id)
        self.fitter = fitter
        self.seed = int(seed)
        self.policy = policy
        self.fold_assignments = balanced_regression_folds(
            self.targets,
            n_folds=policy.inner_folds,
            seed=seed,
        )
        self.search_fold = int(policy.search_holdout_fold % policy.inner_folds)
        self.records: Dict[str, Dict[str, Any]] = {}
        self.generated: Dict[str, Dict[str, Any]] = {}
        self._cv_warm: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.last_cv_results: list[Dict[str, Any]] = []
        self.last_final_attempts: list[Dict[str, Any]] = []
        self._installed_evaluator = None

    def _split(self, fold: int) -> tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.Series], Dict[str, pd.Series]]:
        validation_positions = np.flatnonzero(self.fold_assignments == int(fold))
        fit_positions = np.flatnonzero(self.fold_assignments != int(fold))
        fit = self.df_train.iloc[fit_positions].reset_index(drop=True)
        validation = self.df_train.iloc[validation_positions].reset_index(drop=True)
        fit_targets = {
            name: series.iloc[fit_positions].reset_index(drop=True)
            for name, series in self.targets.items()
        }
        validation_targets = {
            name: series.iloc[validation_positions].reset_index(drop=True)
            for name, series in self.targets.items()
        }
        return fit, validation, fit_targets, validation_targets

    @contextmanager
    def install(self, module: ModuleType) -> Iterator["TrainingOnlySelectionController"]:
        previous_evaluator = module.evaluate_expression_with_fitting
        previous_generator = module.safe_generate_universal_lane_expression
        (
            search_fit,
            search_validation,
            fit_targets,
            search_validation_targets,
        ) = self._split(self.search_fold)
        fit_context = prepare_optimization_context(
            search_fit,
            self.lanes,
            self.lane_to_approach,
            self.intersection_id,
        )

        def training_inner_evaluator(
            expr,
            df_train,
            lanes,
            lane_to_approach,
            approach_targets,
            intersection_id,
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
            if int(optimizer_restarts) != self.policy.optimizer_restarts:
                raise ValueError(
                    f"v3 requires exactly {self.policy.optimizer_restarts} restarts"
                )
            evaluation = previous_evaluator(
                expr,
                search_fit,
                self.lanes,
                self.lane_to_approach,
                fit_targets,
                self.intersection_id,
                fit_context,
                score_mode=score_mode,
                physics_weight=physics_weight,
                verifier_config=verifier_config,
                rng=rng,
                optimizer_restarts=optimizer_restarts,
                df_validation=search_validation,
                validation_targets=search_validation_targets,
            )
            physical_component, inner_fit_r2, _, parameters, feedback, details = evaluation
            try:
                predictions = calculate_approach_delays_from_universal(
                    df=search_validation,
                    universal_expr=expr,
                    lane_parameters=parameters,
                    lanes=self.lanes,
                    lane_to_approach=self.lane_to_approach,
                    intersection_id=self.intersection_id,
                    strict=True,
                )
                metrics = score_predictions(search_validation_targets, predictions)
                selection_accuracy = composite_accuracy(metrics, self.policy)
            except Exception as exc:
                metrics = score_predictions({}, {})
                selection_accuracy = -1e9
                details["training_inner_prediction_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )
                physical_component = 0.0
            fitness = float(selection_accuracy + physics_weight * physical_component)
            inner_fit_metrics = {
                "macro_r2": float(inner_fit_r2),
                "pooled_rmse": details.get("train_rmse"),
                "r2_by_approach": details.get("train_r2_by_approach", {}),
                "rmse_by_approach": details.get("train_rmse_by_approach", {}),
            }
            details.update(
                {
                    "selection_source": "training_inner_holdout",
                    "validation_role": "not_supplied_to_v3_runtime",
                    "outer_validation_accessed": False,
                    "test_file_opened": False,
                    "inner_fit_metrics_reporting_only": inner_fit_metrics,
                    "training_inner_metrics": metrics,
                    "selection_accuracy": float(selection_accuracy),
                    "train_r2": float(metrics["macro_r2"]),
                    "train_rmse": float(metrics["macro_rmse"]),
                    "train_mae": float(metrics["macro_mae"]),
                    "validation_r2": None,
                    "validation_rmse": None,
                    "validation_r2_by_approach": {},
                    "validation_rmse_by_approach": {},
                    "search_holdout_fold": self.search_fold,
                    "inner_fold_count": self.policy.inner_folds,
                }
            )
            standard_pass = bool(details.get("verifier", {}).get("joint_pass", False))
            enhanced = audit_search_physics(
                expr, standard_joint_pass=standard_pass
            )
            enhanced_pass = bool(enhanced["joint_pass"])
            if not enhanced_pass:
                physical_component = 0.0
                fitness = float(selection_accuracy)
                failed = "; ".join(enhanced.get("errors", [])) or (
                    "enhanced fitted-physics audit failed"
                )
                feedback = f"{feedback} V3 enhanced audit: {failed}".strip()
            details["v3_search_physics"] = enhanced
            self.records[str(expr)] = {
                "expression": str(expr),
                "standard_physical_pass": standard_pass,
                "enhanced_physical_pass": enhanced_pass,
                "candidate_source": "generated_v3",
                "search_metrics": metrics,
                "search_composite": float(selection_accuracy),
                "parameters": parameters,
                "evaluation_details": details,
            }
            self.fitter.remember_parameters(str(expr), parameters)
            # The core's compatibility field is an R2, not the composite.  The
            # composite is the fitness accuracy component used for survivors.
            return (
                physical_component,
                float(metrics["macro_r2"]),
                fitness,
                parameters,
                feedback,
                details,
            )

        def recording_generator(*args, **kwargs):
            expression, thought, explanation = previous_generator(*args, **kwargs)
            parent = kwargs.get("base_expr")
            if parent is None and len(args) > 3:
                parent = args[3]
            self.fitter.register_parent(expression, parent)
            self.generated[str(expression)] = {
                "thought": thought,
                "explanation": explanation,
                "parent_expression": parent,
            }
            return expression, thought, explanation

        module.evaluate_expression_with_fitting = training_inner_evaluator
        module.safe_generate_universal_lane_expression = recording_generator
        self._installed_evaluator = training_inner_evaluator
        try:
            yield self
        finally:
            self._installed_evaluator = None
            module.evaluate_expression_with_fitting = previous_evaluator
            module.safe_generate_universal_lane_expression = previous_generator

    def evaluate_incumbent(
        self,
        expression: str,
        *,
        thought: str = "",
        explanation: str = "",
    ) -> Dict[str, Any]:
        """Fit one retained v2 expression on the same Training search split."""
        if self._installed_evaluator is None:
            raise RuntimeError("incumbent evaluation requires an installed controller")
        expression = str(expression)
        self._installed_evaluator(
            expression,
            self.df_train,
            self.lanes,
            self.lane_to_approach,
            self.targets,
            self.intersection_id,
            None,
            score_mode="binary",
            physics_weight=1.0,
            rng=np.random.default_rng(
                _stable_fit_seed(self.seed, expression, self.search_fold, "incumbent")
            ),
            optimizer_restarts=self.policy.optimizer_restarts,
        )
        record = self.records[expression]
        record["candidate_source"] = "retained_v2_incumbent"
        self.generated[expression] = {
            "thought": str(thought or ""),
            "explanation": str(explanation or ""),
            "parent_expression": None,
            "candidate_source": "retained_v2_incumbent",
        }
        return record

    def cross_validate_candidates(self) -> list[Dict[str, Any]]:
        """Four-fold rerank of the best search candidates, using Training only."""
        selected = select_search_candidates(
            list(self.records.values()),
            top_k=self.policy.cross_validation_top_k,
        )
        results = []
        for candidate in selected:
            expression = candidate["expression"]
            started = time.perf_counter()
            oof_targets = {name: np.full(len(self.df_train), np.nan) for name in self.targets}
            oof_predictions = {name: np.full(len(self.df_train), np.nan) for name in self.targets}
            fold_records = []
            warm_parameters = None
            for fold in range(self.policy.inner_folds):
                fit, validation, fit_targets, validation_targets = self._split(fold)
                fit_positions = np.flatnonzero(self.fold_assignments != fold)
                validation_positions = np.flatnonzero(self.fold_assignments == fold)
                diagnostics: Dict[str, Any] = {}
                parameters = self.fitter.fit(
                    universal_expr=expression,
                    df=fit,
                    lanes=self.lanes,
                    lane_to_approach=self.lane_to_approach,
                    approach_targets=fit_targets,
                    intersection_id=self.intersection_id,
                    prepared_context=prepare_optimization_context(
                        fit,
                        self.lanes,
                        self.lane_to_approach,
                        self.intersection_id,
                    ),
                    rng=np.random.default_rng(
                        _stable_fit_seed(self.seed, expression, fold, "four_fold_cv")
                    ),
                    n_restarts=V3_RESTARTS,
                    diagnostics=diagnostics,
                    warm_parameters=None,
                )
                predictions = calculate_approach_delays_from_universal(
                    df=validation,
                    universal_expr=expression,
                    lane_parameters=parameters,
                    lanes=self.lanes,
                    lane_to_approach=self.lane_to_approach,
                    intersection_id=self.intersection_id,
                    strict=True,
                )
                metrics = score_predictions(validation_targets, predictions)
                enhanced = audit_fitted_physics(
                    expression, parameters, self.lanes
                )
                for approach in self.targets:
                    oof_targets[approach][validation_positions] = np.asarray(
                        validation_targets[approach].values, dtype=float
                    )
                    oof_predictions[approach][validation_positions] = np.asarray(
                        predictions[approach], dtype=float
                    )
                fold_records.append(
                    {
                        "fold": fold,
                        "fit_rows": int(len(fit_positions)),
                        "selection_rows": int(len(validation_positions)),
                        "metrics": metrics,
                        "standard_physical_pass": bool(
                            enhanced["standard_joint_pass"]
                        ),
                        "enhanced_physical_pass": bool(enhanced["joint_pass"]),
                        "enhanced_physics": enhanced,
                        "optimizer": diagnostics,
                    }
                )
                warm_parameters = parameters
            oof_metrics = score_predictions(oof_targets, oof_predictions)
            cv_composite = composite_accuracy(oof_metrics, self.policy)
            self._cv_warm[expression] = warm_parameters or {}
            results.append(
                {
                    "expression": expression,
                    "candidate_source": candidate.get(
                        "candidate_source", "generated_v3"
                    ),
                    "search_metrics": candidate["search_metrics"],
                    "search_composite": candidate["search_composite"],
                    "oof_metrics": oof_metrics,
                    "cv_composite": cv_composite,
                    "all_folds_standard_physical": all(
                        item["standard_physical_pass"] for item in fold_records
                    ),
                    "all_folds_enhanced_physical": all(
                        item["enhanced_physical_pass"] for item in fold_records
                    ),
                    "folds": fold_records,
                    "wall_seconds": float(time.perf_counter() - started),
                }
            )
        results.sort(
            key=lambda item: (
                item["all_folds_enhanced_physical"],
                item["all_folds_standard_physical"],
                item["cv_composite"],
            ),
            reverse=True,
        )
        self.last_cv_results = results
        return results

    def finalize(self, core_result) -> tuple[tuple, Dict[str, Any]]:
        """Rerank candidates, refit on all Training rows, and apply polish."""
        _, core_thought, core_explanation, _, history = core_result
        cv_results = self.cross_validate_candidates()
        if not cv_results:
            raise RuntimeError("v3 search produced no physically feasible candidate")
        final_attempts = []
        selected = None
        for rank, candidate in enumerate(cv_results, start=1):
            expression = candidate["expression"]
            diagnostics: Dict[str, Any] = {}
            parameters = self.fitter.fit_final_with_polish(
                universal_expr=expression,
                df=self.df_train,
                lanes=self.lanes,
                lane_to_approach=self.lane_to_approach,
                approach_targets=self.targets,
                intersection_id=self.intersection_id,
                prepared_context=prepare_optimization_context(
                    self.df_train,
                    self.lanes,
                    self.lane_to_approach,
                    self.intersection_id,
                ),
                rng=np.random.default_rng(
                    _stable_fit_seed(self.seed, expression, -1, "full_training_polish")
                ),
                n_restarts=V3_RESTARTS,
                warm_parameters=self._cv_warm.get(expression),
                polish_maxiter=self.policy.polish_maxiter,
                polish_maxfun=self.policy.polish_maxfun,
                diagnostics=diagnostics,
            )
            predictions = calculate_approach_delays_from_universal(
                df=self.df_train,
                universal_expr=expression,
                lane_parameters=parameters,
                lanes=self.lanes,
                lane_to_approach=self.lane_to_approach,
                intersection_id=self.intersection_id,
                strict=True,
            )
            train_metrics = score_predictions(self.targets, predictions)
            enhanced = diagnostics.get("selected_enhanced_physics")
            if not isinstance(enhanced, dict):
                enhanced = audit_fitted_physics(
                    expression, parameters, self.lanes
                )
            attempt = {
                "cv_rank": rank,
                "expression": expression,
                "cv_composite": candidate["cv_composite"],
                "oof_metrics": candidate["oof_metrics"],
                "full_training_metrics_reporting_only": train_metrics,
                "enhanced_physics": enhanced,
                "optimizer": diagnostics,
            }
            final_attempts.append(attempt)
            self.last_final_attempts = list(final_attempts)
            if enhanced["joint_pass"]:
                selected = (candidate, parameters, train_metrics, enhanced)
                break
        if selected is None:
            raise RuntimeError(
                "no cross-validated candidate passed the enhanced full-Training physics audit"
            )

        candidate, parameters, train_metrics, enhanced = selected
        expression = candidate["expression"]
        metadata = self.generated.get(expression, {})
        thought = metadata.get("thought")
        explanation = metadata.get("explanation")
        if thought is None and expression == core_result[0]:
            thought = core_thought
        if explanation is None and expression == core_result[0]:
            explanation = core_explanation
        history = list(history)
        history.append(
            {
                "event": "v3_training_inner_cv_selection",
                "selection_source": "training_inner_four_fold_oof",
                "outer_validation_accessed": False,
                "test_file_opened": False,
                "candidate_count_reranked": len(cv_results),
                "selected_expression": expression,
                "selected_cv_composite": candidate["cv_composite"],
                "selected_oof_metrics": candidate["oof_metrics"],
                "fixed_r9_guard_applied": False,
            }
        )
        history.append(
            {
                "event": "v3_full_training_refit",
                "selection_already_frozen": True,
                "metrics_role": "reporting_only",
                "full_training_metrics": train_metrics,
                "enhanced_physics": enhanced,
            }
        )
        report = {
            "method_id": self.policy.method_id,
            "policy": self.policy.to_dict(),
            "selection_source": "training_inner_four_fold_oof",
            "outer_validation_accessed": False,
            "test_file_opened": False,
            "fold_assignments": self.fold_assignments.tolist(),
            "search_holdout_fold": self.search_fold,
            "search_candidate_count": len(self.records),
            "cv_candidates": cv_results,
            "final_refit_attempts": final_attempts,
            "selected_expression": expression,
            "selected_candidate_source": candidate.get(
                "candidate_source", "generated_v3"
            ),
            "selected_oof_metrics": candidate["oof_metrics"],
            "selected_full_training_metrics_reporting_only": train_metrics,
            "selected_enhanced_physics": enhanced,
        }
        return (
            expression,
            thought or "",
            explanation or "",
            parameters,
            history,
        ), report
