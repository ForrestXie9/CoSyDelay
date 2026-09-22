"""Scoped integration of the validated Jacobian fitter."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Optional

from .jacobian_fitter import JacobianParallelApproachFitter


@contextmanager
def install_population_evolution_jacobian_fitter(
    module: Optional[ModuleType] = None,
) -> Iterator[JacobianParallelApproachFitter]:
    """Use the Jacobian fitter for one search and restore the baseline."""
    if module is None:
        import population_evolution_lane as module

    attribute = "fit_lane_parameters_to_approaches"
    if not hasattr(module, attribute):
        raise AttributeError(
            f"{module.__name__!r} does not expose {attribute!r}"
        )
    previous = getattr(module, attribute)
    fitter = JacobianParallelApproachFitter()
    setattr(module, attribute, fitter)
    try:
        yield fitter
    finally:
        setattr(module, attribute, previous)
        fitter.close()
