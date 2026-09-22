"""V15 all-log L-BFGS-B fitter selecting restarts solely by Training MSE."""

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Mapping, Optional

from methods.cosydelay.engine.optimizer_conditioning.all_log_fitter import (
    fit_lane_parameters_all_log_parallel,
)


class PersistentSymbolicR9Fitter:
    def __init__(self, *, parallel_workers: int, maxiter: int, maxfun: int) -> None:
        self.parallel_workers = int(parallel_workers)
        self.maxiter = int(maxiter)
        self.maxfun = int(maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )

    def fit(self, *, warm_parameters: Optional[Mapping] = None, **kwargs):
        if self._executor is None:
            raise RuntimeError("PersistentSymbolicR9Fitter is closed")
        result = fit_lane_parameters_all_log_parallel(
            warm_parameters=warm_parameters,
            parallel_workers=self.parallel_workers,
            maxiter=self.maxiter,
            maxfun=self.maxfun,
            executor=self._executor,
            r9_constrained_restart_selection=False,
            **kwargs,
        )
        diagnostics = kwargs.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["formal_method_adapter"] = (
                "cosydelay_v15_symbolic_r9.fitter"
            )
            diagnostics["restart_selection"] = "minimum_training_mse"
        return result

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
