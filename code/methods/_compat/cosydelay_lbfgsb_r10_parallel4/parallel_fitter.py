"""Official four-approach wrapper around the verified L-BFGS-B r10 fitter."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..cosydelay_lbfgsb_r10_parallel3.parallel_fitter import (
    OFFICIAL_MAXFUN,
    OFFICIAL_MAXITER,
    OFFICIAL_RESTARTS,
    ParallelApproachFitter as _ParallelApproachFitter,
    fit_lane_parameters_to_approaches_parallel as _fit_parallel,
)


OFFICIAL_PARALLEL_WORKERS = 4


def fit_lane_parameters_to_approaches_parallel(
    universal_expr: str,
    df: pd.DataFrame,
    lanes: Sequence[str],
    lane_to_approach: Mapping[str, str],
    approach_targets: Mapping[str, pd.Series],
    intersection_id: int,
    verbose: bool = False,
    prepared_context: Optional[Mapping[str, Any]] = None,
    rng: Optional[np.random.Generator] = None,
    n_restarts: int = OFFICIAL_RESTARTS,
    diagnostics: Optional[Dict[str, Any]] = None,
    coefficient_bounds_override: Optional[
        Mapping[str, Tuple[float, float]]
    ] = None,
    *,
    parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
    maxiter: Optional[int] = None,
    maxfun: Optional[int] = None,
    executor: Optional[ProcessPoolExecutor] = None,
) -> Dict[str, Dict[str, float]]:
    """Fit every available approach with at most four concurrent workers."""
    return _fit_parallel(
        universal_expr=universal_expr,
        df=df,
        lanes=lanes,
        lane_to_approach=lane_to_approach,
        approach_targets=approach_targets,
        intersection_id=intersection_id,
        verbose=verbose,
        prepared_context=prepared_context,
        rng=rng,
        n_restarts=n_restarts,
        diagnostics=diagnostics,
        coefficient_bounds_override=coefficient_bounds_override,
        parallel_workers=parallel_workers,
        maxiter=maxiter,
        maxfun=maxfun,
        executor=executor,
    )


class ParallelApproachFitter(_ParallelApproachFitter):
    """Persistent process pool whose official concurrency cap is four."""

    def __init__(
        self,
        *,
        parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
        maxiter: int = OFFICIAL_MAXITER,
        maxfun: int = OFFICIAL_MAXFUN,
    ) -> None:
        super().__init__(
            parallel_workers=parallel_workers,
            maxiter=maxiter,
            maxfun=maxfun,
        )

