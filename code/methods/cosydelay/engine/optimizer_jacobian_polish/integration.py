"""Scoped integration of the selected Jacobian plus polish fitter."""

from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType
from typing import Iterator, Optional

from .polish_fitter import JacobianPolishedApproachFitter


@contextmanager
def install_population_evolution_polished_fitter(
    module: Optional[ModuleType] = None,
) -> Iterator[JacobianPolishedApproachFitter]:
    """Use the selected polish=200 fitter for one evolutionary search."""
    if module is None:
        import population_evolution_lane as module

    attribute = "fit_lane_parameters_to_approaches"
    if not hasattr(module, attribute):
        raise AttributeError(
            f"{module.__name__!r} does not expose {attribute!r}"
        )
    previous = getattr(module, attribute)
    fitter = JacobianPolishedApproachFitter()
    setattr(module, attribute, fitter)
    try:
        yield fitter
    finally:
        setattr(module, attribute, previous)
        fitter.close()
