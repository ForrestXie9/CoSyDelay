"""Compose V17 with conservative R7 screening and a killable fit boundary."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import time
from types import ModuleType
from typing import Any, Iterator, Mapping, Optional, Sequence

import pandas as pd

from expression_rules import canonical_expression
from optimization_lane import prepare_optimization_context
from methods.cosydelay._internal.numeric_fitting import integration as v15_integration
from methods.cosydelay._internal.training_protocol.physics import (
    ManuscriptVerifierConfig,
)
from methods.cosydelay._internal.retry_protocol.integration import (
    install_v17_candidate,
)

from .contract import V18_CONTRACT, validate_v18_contract
from .fit_timeout import GuardedCandidateFitter
from .policy import V18_POLICY
from .prefit_r7 import classify_prefit_r7


_LAST_FIT_GUARD_AUDIT: dict[str, Any] = {
    "fit_events": [],
    "candidate_events": [],
    "prefit_r7_events": [],
    "rejected_canonical_expressions": [],
}


class CandidateEvaluationRejected(RuntimeError):
    """A generated expression failed before becoming a scored candidate."""


def _hash_text(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


class FitRejectionRegistry:
    """Run-local exact-canonical ban list and supervision-free audit."""

    def __init__(self) -> None:
        self._rejected: dict[str, dict[str, Any]] = {}
        self.candidate_events: list[dict[str, Any]] = []

    @property
    def rejected_keys(self) -> tuple[str, ...]:
        return tuple(self._rejected)

    def reject(self, expression: str, *, stage: str, exception_type: str) -> None:
        try:
            key = canonical_expression(str(expression))
        except Exception:
            key = "raw_sha256:" + _hash_text(str(expression))
        item = {
            "event": "candidate_rejected",
            "expression_sha256": _hash_text(str(expression)),
            "canonical_sha256": _hash_text(key),
            "stage": str(stage),
            "exception_type": str(exception_type),
            "counts_toward_successful_candidate_budget": False,
            "accuracy_metrics_present": False,
            "validation_or_test_data_present": False,
        }
        self._rejected[key] = item
        self.candidate_events.append(item)

    def accept(self, expression: str) -> None:
        try:
            key = canonical_expression(str(expression))
        except Exception:
            key = "raw_sha256:" + _hash_text(str(expression))
        self.candidate_events.append(
            {
                "event": "candidate_accepted",
                "expression_sha256": _hash_text(str(expression)),
                "canonical_sha256": _hash_text(key),
                "counts_toward_successful_candidate_budget": True,
                "accuracy_metrics_present": False,
                "validation_or_test_data_present": False,
            }
        )

    def validate(self, expression: str) -> tuple[bool, str]:
        try:
            key = canonical_expression(str(expression))
        except Exception:
            key = "raw_sha256:" + _hash_text(str(expression))
        if key in self._rejected:
            return (
                False,
                "canonical expression was already rejected after a bounded "
                "coefficient-fit/evaluation failure; generate a distinct legal "
                "expression",
            )
        return True, "expression has no prior V18 runtime rejection"


@dataclass
class V18Runtime:
    base: object
    guarded_fitter: GuardedCandidateFitter
    rejection_registry: FitRejectionRegistry
    prefit_r7_audit: list[dict[str, Any]]

    @property
    def fit_guard_audit(self) -> dict[str, Any]:
        return {
            "fit_events": copy.deepcopy(self.guarded_fitter.audit),
            "candidate_events": copy.deepcopy(
                self.rejection_registry.candidate_events
            ),
            "prefit_r7_events": copy.deepcopy(self.prefit_r7_audit),
            "rejected_canonical_expressions": [
                _hash_text(key) for key in self.rejection_registry.rejected_keys
            ],
        }

    def __getattr__(self, name):
        return getattr(self.base, name)


def get_last_fit_guard_audit() -> dict[str, Any]:
    return copy.deepcopy(_LAST_FIT_GUARD_AUDIT)


@contextmanager
def install_v18_candidate(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy=V18_POLICY,
    verifier_config: ManuscriptVerifierConfig | None = None,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[V18Runtime]:
    """Install V18 without changing V17 Fitness or optimizer settings."""
    global _LAST_FIT_GUARD_AUDIT
    validate_v18_contract()
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module

    prepared_context = prepare_optimization_context(
        df_train,
        list(lanes),
        dict(lane_to_approach),
        int(intersection_id),
    )
    static_fit_kwargs = {
        "df": df_train,
        "lanes": list(lanes),
        "lane_to_approach": dict(lane_to_approach),
        "approach_targets": targets,
        "intersection_id": int(intersection_id),
        "prepared_context": prepared_context,
    }
    previous_fitter_type = v15_integration.PersistentSymbolicR9Fitter
    created_fitters: list[GuardedCandidateFitter] = []
    registry = FitRejectionRegistry()
    prefit_r7_audit: list[dict[str, Any]] = []

    def guarded_fitter_factory(*, parallel_workers: int, maxiter: int, maxfun: int):
        if created_fitters:
            raise RuntimeError("V18 permits one fitting supervisor per run")
        if (
            int(parallel_workers) != int(policy.approach_workers_cap)
            or int(maxiter) != int(policy.optimizer_maxiter)
            or int(maxfun) != int(policy.optimizer_maxfun)
        ):
            raise RuntimeError("V18 received optimizer settings outside the frozen policy")
        fitter = GuardedCandidateFitter(
            static_fit_kwargs=static_fit_kwargs,
            parallel_workers=parallel_workers,
            maxiter=maxiter,
            maxfun=maxfun,
            fit_timeout_seconds=V18_CONTRACT.candidate_fit_wall_timeout_seconds,
            startup_timeout_seconds=(
                V18_CONTRACT.candidate_fit_startup_timeout_seconds
            ),
            close_timeout_seconds=V18_CONTRACT.candidate_fit_close_timeout_seconds,
        )
        created_fitters.append(fitter)
        return fitter

    v15_integration.PersistentSymbolicR9Fitter = guarded_fitter_factory
    runtime = None
    try:
        with install_v17_candidate(
            df_train=df_train,
            targets=targets,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            intersection_id=intersection_id,
            policy=policy,
            verifier_config=verifier_config,
            population_module=population_module,
            adaptation_module=adaptation_module,
        ) as base_runtime:
            if len(created_fitters) != 1:
                raise RuntimeError("V18 fitting supervisor was not installed exactly once")
            guarded_fitter = created_fitters[0]
            previous_evaluator = population_module.evaluate_expression_with_fitting
            previous_validator = adaptation_module.validate_candidate_expression

            def reject_impossible_or_previously_failed(expression: str):
                passed, reason = previous_validator(expression)
                if not passed:
                    return passed, reason
                passed, reason = registry.validate(str(expression))
                if not passed:
                    return passed, reason
                started = time.perf_counter()
                decision = classify_prefit_r7(str(expression))
                event = decision.to_dict()
                try:
                    canonical_key = canonical_expression(str(expression))
                except Exception:
                    canonical_key = "raw_sha256:" + _hash_text(str(expression))
                event.update(
                    {
                        "event": "prefit_r7_decision",
                        "expression_sha256": _hash_text(str(expression)),
                        "canonical_sha256": _hash_text(canonical_key),
                        "validator_wall_seconds": float(
                            time.perf_counter() - started
                        ),
                        "accuracy_metrics_present": False,
                        "validation_or_test_data_present": False,
                        "counts_toward_successful_candidate_budget": False,
                    }
                )
                prefit_r7_audit.append(event)
                if not decision.passed:
                    return False, decision.reason
                return True, reason

            def evaluate_with_runtime_rejection(*args, **kwargs):
                expression = str(args[0] if args else kwargs["expr"])
                try:
                    result = previous_evaluator(*args, **kwargs)
                except Exception as exc:
                    registry.reject(
                        expression,
                        stage="candidate_evaluation_exception",
                        exception_type=type(exc).__name__,
                    )
                    raise
                parameters = result[3]
                details = result[5]
                if details.get("status") != "evaluated" or not parameters:
                    registry.reject(
                        expression,
                        stage="candidate_evaluation_incomplete",
                        exception_type=str(details.get("status", "unknown")),
                    )
                    raise CandidateEvaluationRejected(
                        "candidate did not complete bounded coefficient fitting; "
                        "regenerate the same population slot"
                    )
                registry.accept(expression)
                return result

            adaptation_module.validate_candidate_expression = (
                reject_impossible_or_previously_failed
            )
            population_module.evaluate_expression_with_fitting = (
                evaluate_with_runtime_rejection
            )
            runtime = V18Runtime(
                base=base_runtime,
                guarded_fitter=guarded_fitter,
                rejection_registry=registry,
                prefit_r7_audit=prefit_r7_audit,
            )
            try:
                yield runtime
            finally:
                population_module.evaluate_expression_with_fitting = (
                    previous_evaluator
                )
                adaptation_module.validate_candidate_expression = previous_validator
    finally:
        v15_integration.PersistentSymbolicR9Fitter = previous_fitter_type
        if created_fitters:
            _LAST_FIT_GUARD_AUDIT = {
                "fit_events": copy.deepcopy(created_fitters[0].audit),
                "candidate_events": copy.deepcopy(registry.candidate_events),
                "prefit_r7_events": copy.deepcopy(prefit_r7_audit),
                "rejected_canonical_expressions": [
                    _hash_text(key) for key in registry.rejected_keys
                ],
            }
        else:
            _LAST_FIT_GUARD_AUDIT = {
                "fit_events": [],
                "candidate_events": copy.deepcopy(registry.candidate_events),
                "prefit_r7_events": copy.deepcopy(prefit_r7_audit),
                "rejected_canonical_expressions": [],
            }
