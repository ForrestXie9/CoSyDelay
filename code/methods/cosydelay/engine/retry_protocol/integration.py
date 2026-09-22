"""V17 composition: V16 plus legality feedback retained across failed batches."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Iterator, Mapping, Optional, Sequence

import pandas as pd

from methods.cosydelay.engine.training_protocol.integration import (
    install_v16_candidate,
)
from methods.cosydelay.engine.training_protocol.physics import (
    ManuscriptVerifierConfig,
)

from .contract import validate_v17_contract
from .policy import V17_POLICY
from .prompt import append_cross_batch_correction


_LAST_RETRY_AUDIT: list[dict] = []


@dataclass
class V17Runtime:
    base: object
    cross_batch_retry_audit: list[dict]

    @property
    def evaluations(self):
        return self.base.evaluations

    @property
    def prefit_audit(self):
        return self.base.prefit_audit

    @property
    def fitted_rejections(self):
        return self.base.fitted_rejections

    @property
    def generation_audit(self):
        return self.base.generation_audit

    @property
    def complexity_audit(self):
        return self.base.complexity_audit

    def __getattr__(self, name):
        """Preserve the complete inherited runtime interface."""
        return getattr(self.base, name)


def get_last_retry_audit() -> list[dict]:
    return [dict(item) for item in _LAST_RETRY_AUDIT]


@contextmanager
def install_cross_batch_legality_repair(
    *, population_module: ModuleType, adaptation_module: ModuleType
) -> Iterator[list[dict]]:
    """Carry only the last legality failure into the next generation batch."""
    previous_generator = population_module.safe_generate_universal_lane_expression
    previous_validator = adaptation_module.validate_candidate_expression
    previous_run_llm = adaptation_module.run_llm
    previous_expression_audit_writer = (
        adaptation_module._write_expression_attempt_audit
    )
    audit: list[dict] = []
    state = {
        "sequence": 0,
        "current_failure": None,
        "carried_failure": None,
        "active_correction": None,
        "correction_pending": False,
        "correction_injected": False,
        "last_response_expression": None,
    }

    def recording_validator(expression: str):
        passed, reason = previous_validator(expression)
        if not passed:
            state["current_failure"] = {
                "reason": str(reason),
                "expression": str(expression),
            }
        return passed, reason

    def recording_expression_attempt(record: dict):
        previous_expression_audit_writer(record)
        if record.get("status") in {
            "format_invalid",
            "symbolic_parse_invalid",
            "canonical_duplicate",
            "validator_rejected",
        }:
            state["current_failure"] = {
                "reason": str(
                    record.get("reason")
                    or "the candidate failed the declared legality checks"
                ),
                "expression": (
                    str(record["expression"])
                    if record.get("expression") is not None
                    else None
                ),
            }

    def run_llm_with_cross_batch_repair(prompt: str, *args, **kwargs):
        correction = state.get("active_correction")
        if correction and state.get("correction_pending"):
            prompt = append_cross_batch_correction(
                prompt,
                reason=correction["reason"],
                expression=correction.get("expression"),
                sequence=int(state["sequence"]),
            )
            state["correction_pending"] = False
            state["correction_injected"] = True
        response = previous_run_llm(prompt, *args, **kwargs)
        try:
            parsed, _, _ = adaptation_module.parse_llm_response(response)
        except Exception:
            parsed = None
        state["last_response_expression"] = (
            str(parsed) if parsed is not None else None
        )
        # The dynamically installed validator below replaces this generic
        # reason whenever it sees a more precise grammar/role failure.  The
        # fallback still retains the last returned formula for format,
        # symbolic-parse, and novelty failures that occur before validation.
        state["current_failure"] = {
            "reason": (
                "the latest candidate did not complete format, symbolic, "
                "novelty, and coefficient-role acceptance"
            ),
            "expression": state["last_response_expression"],
        }
        return response

    def generator_with_cross_batch_repair(*args, **kwargs):
        state["sequence"] += 1
        state["current_failure"] = None
        state["active_correction"] = state.get("carried_failure")
        state["correction_pending"] = state["active_correction"] is not None
        state["correction_injected"] = False
        state["last_response_expression"] = None
        try:
            result = previous_generator(*args, **kwargs)
        except Exception as exc:
            failure = state.get("current_failure") or {
                "reason": (
                    "the previous generation batch produced no valid novel "
                    f"expression ({type(exc).__name__})"
                ),
                "expression": None,
            }
            state["carried_failure"] = dict(failure)
            audit.append(
                {
                    "event": "generation_batch_failed",
                    "sequence": int(state["sequence"]),
                    "correction_injected": bool(state["correction_injected"]),
                    "last_reason": str(failure["reason"]),
                    "last_expression": failure.get("expression"),
                    "exception_type": type(exc).__name__,
                }
            )
            raise
        else:
            audit.append(
                {
                    "event": "generation_batch_succeeded",
                    "sequence": int(state["sequence"]),
                    "correction_injected": bool(state["correction_injected"]),
                }
            )
            state["carried_failure"] = None
            state["active_correction"] = None
            state["correction_pending"] = False
            return result

    adaptation_module.validate_candidate_expression = recording_validator
    adaptation_module._write_expression_attempt_audit = recording_expression_attempt
    adaptation_module.run_llm = run_llm_with_cross_batch_repair
    population_module.safe_generate_universal_lane_expression = (
        generator_with_cross_batch_repair
    )
    try:
        yield audit
    finally:
        population_module.safe_generate_universal_lane_expression = previous_generator
        adaptation_module.run_llm = previous_run_llm
        adaptation_module._write_expression_attempt_audit = (
            previous_expression_audit_writer
        )
        adaptation_module.validate_candidate_expression = previous_validator


@contextmanager
def install_v17_candidate(
    *,
    df_train: pd.DataFrame,
    targets: Mapping[str, pd.Series],
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    intersection_id: int,
    policy=V17_POLICY,
    verifier_config: ManuscriptVerifierConfig | None = None,
    population_module: Optional[ModuleType] = None,
    adaptation_module: Optional[ModuleType] = None,
) -> Iterator[V17Runtime]:
    """Install the V17 repair without changing V16 scoring or fitting."""
    global _LAST_RETRY_AUDIT
    validate_v17_contract()
    if population_module is None:
        import population_evolution_lane as population_module
    if adaptation_module is None:
        import expression_adaptation_lane as adaptation_module
    audit: list[dict] = []
    try:
        with install_v16_candidate(
            df_train=df_train,
            targets=targets,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            intersection_id=intersection_id,
            policy=policy,
            verifier_config=verifier_config,
            population_module=population_module,
            adaptation_module=adaptation_module,
        ) as runtime:
            with install_cross_batch_legality_repair(
                population_module=population_module,
                adaptation_module=adaptation_module,
            ) as audit:
                yield V17Runtime(
                    base=runtime,
                    cross_batch_retry_audit=audit,
                )
    finally:
        _LAST_RETRY_AUDIT = [dict(item) for item in audit]
