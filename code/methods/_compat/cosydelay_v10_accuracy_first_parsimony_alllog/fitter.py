"""Persistent all-log L-BFGS-B adapter for the prospective V10 runtime."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Any, Dict, Mapping, Optional

from methods.prospective_optimizer_conditioning_v1.all_log_fitter import (
    fit_lane_parameters_all_log_parallel,
)


V10_START_STRATEGY_ID = "v10_all10_positive_log_retained_raw_starts_r10"


class PersistentAllLogFitter:
    def __init__(self, *, parallel_workers: int, maxiter: int, maxfun: int) -> None:
        self.parallel_workers = int(parallel_workers)
        self.maxiter = int(maxiter)
        self.maxfun = int(maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def fit(
        self,
        *,
        warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if self._executor is None:
            raise RuntimeError("PersistentAllLogFitter is closed")
        result = fit_lane_parameters_all_log_parallel(
            warm_parameters=warm_parameters,
            parallel_workers=self.parallel_workers,
            maxiter=self.maxiter,
            maxfun=self.maxfun,
            executor=self._executor,
            **kwargs,
        )
        diagnostics = kwargs.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["start_strategy_id"] = V10_START_STRATEGY_ID
            diagnostics["formal_method_adapter"] = (
                "cosydelay_v10_accuracy_first_parsimony_alllog.fitter"
            )
        return result

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "PersistentAllLogFitter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
