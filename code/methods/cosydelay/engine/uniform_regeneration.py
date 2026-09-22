"""Shared retry contract for experimental CoSyDelay regeneration arms."""

from __future__ import annotations

from functools import wraps
from typing import Callable


REGENERATION_ATTEMPTS = 5


def wrap_uniform_regeneration(generator: Callable, *, marker: str) -> Callable:
    """Force every non-initial generation call through the same retry budget."""

    @wraps(generator)
    def checked(*args, **kwargs):
        operation = str(
            kwargs.get("mutation_type", args[2] if len(args) > 2 else "initial")
        ).lower()
        if operation == "initial":
            return generator(*args, **kwargs)
        changed_args = list(args)
        changed_kwargs = dict(kwargs)
        if len(changed_args) > 8:
            changed_args[8] = REGENERATION_ATTEMPTS
            changed_kwargs.pop("max_retries", None)
        else:
            changed_kwargs["max_retries"] = REGENERATION_ATTEMPTS
        return generator(*changed_args, **changed_kwargs)

    setattr(checked, marker, True)
    return checked
