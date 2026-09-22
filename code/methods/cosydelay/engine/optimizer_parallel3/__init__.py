"""Official CoSyDelay coefficient fitter: L-BFGS-B r10, parallel3."""

from .parallel_fitter import (
    OFFICIAL_MAXFUN,
    OFFICIAL_MAXITER,
    OFFICIAL_PARALLEL_WORKERS,
    OFFICIAL_RESTARTS,
    ParallelApproachFitter,
    fit_lane_parameters_to_approaches_parallel,
)
from .integration import install_population_evolution_fitter

__all__ = [
    "OFFICIAL_MAXFUN",
    "OFFICIAL_MAXITER",
    "OFFICIAL_PARALLEL_WORKERS",
    "OFFICIAL_RESTARTS",
    "ParallelApproachFitter",
    "fit_lane_parameters_to_approaches_parallel",
    "install_population_evolution_fitter",
]
