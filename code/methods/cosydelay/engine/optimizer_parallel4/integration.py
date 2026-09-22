"""Scoped installation of the retained four-approach coefficient fitter."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Dict, Iterator, List, Optional

from .parallel_fitter import ParallelApproachFitter
from .prefit_integration import install_structural_prefit_gate


@dataclass
class RetainedEvolutionRuntime:
    fitter: ParallelApproachFitter
    prefit_audit: List[Dict[str, Any]]


@contextmanager
def install_population_evolution_fitter(
    module: Optional[ModuleType] = None,
) -> Iterator[ParallelApproachFitter]:
    """Install one persistent four-worker pool for an evolutionary run."""
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


@contextmanager
def install_retained_evolution(
    module: Optional[ModuleType] = None,
) -> Iterator[RetainedEvolutionRuntime]:
    """Install the structural pre-fit gate and four-worker fitter together."""
    with install_structural_prefit_gate(population_module=module) as gate_audit:
        with install_population_evolution_fitter(module=module) as fitter:
            yield RetainedEvolutionRuntime(
                fitter=fitter,
                prefit_audit=gate_audit,
            )
