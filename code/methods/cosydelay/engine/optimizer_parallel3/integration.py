"""Scoped integration with the existing evolutionary search."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Optional

from .parallel_fitter import ParallelApproachFitter


@contextmanager
def install_population_evolution_fitter(
    module: Optional[ModuleType] = None,
) -> Iterator[ParallelApproachFitter]:
    """Temporarily install the official fitter and restore the old one.

    The evolutionary module imports the coefficient fitter into a module-level
    name. This context manager replaces only that name for the duration of one
    search, reuses a persistent three-process pool, and always restores the
    previous callable.
    """
    if module is None:
        import population_evolution_lane as module

    attribute = "fit_lane_parameters_to_approaches"
    if not hasattr(module, attribute):
        raise AttributeError(
            f"{module.__name__!r} does not expose {attribute!r}"
        )
    previous = getattr(module, attribute)
    fitter = ParallelApproachFitter()
    setattr(module, attribute, fitter)
    try:
        yield fitter
    finally:
        setattr(module, attribute, previous)
        fitter.close()
