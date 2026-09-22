"""V20 exact pre-fit R4 time-dimension gate."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator

from methods.cosydelay._internal.training_protocol.physics import (
    _check_exact_time_dimension,
)


def validate_exact_time_dimension(expression: str) -> tuple[bool, str]:
    passed, reason = _check_exact_time_dimension(str(expression))
    if passed:
        return True, "R4 exact pre-fit time dimension passed"
    return False, f"R4 pre-fit rejection: {reason or 'time dimension is not seconds'}"


@contextmanager
def install_prefit_r4_gate(adaptation_module: ModuleType) -> Iterator[None]:
    """Compose exact R4 after all inherited pre-fit legality checks."""

    previous_validator = adaptation_module.validate_candidate_expression

    def validate_with_r4(expression: str):
        passed, reason = previous_validator(expression)
        if not passed:
            return passed, reason
        return validate_exact_time_dimension(str(expression))

    adaptation_module.validate_candidate_expression = validate_with_r4
    try:
        yield
    finally:
        adaptation_module.validate_candidate_expression = previous_validator
