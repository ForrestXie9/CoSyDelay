"""Narrow v9 optimizer adapter with no CV, incumbent, or polish interface."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from typing import Any, Dict, Mapping, Optional

from methods.cosydelay._internal.optimizer_parallel4.parallel_fitter import (
    OFFICIAL_MAXFUN,
    OFFICIAL_MAXITER,
    OFFICIAL_PARALLEL_WORKERS,
)
from methods.cosydelay._internal.optimizer_parallel4_v3.fitter import (
    fit_lane_parameters_mixed_jacobian_parallel,
)


CLEAN_START_STRATEGY_ID = (
    "v9_clean_local9_or_parent8_plus_rolewide1_selective_log_r10"
)


class CleanMixedRestartJacobianFitter:
    """Persistent four-worker L-BFGS-B fitter for the clean runtime only."""

    def __init__(
        self,
        *,
        parallel_workers: int = OFFICIAL_PARALLEL_WORKERS,
        maxiter: int = OFFICIAL_MAXITER,
        maxfun: int = OFFICIAL_MAXFUN,
    ) -> None:
        self.parallel_workers = int(parallel_workers)
        self.maxiter = int(maxiter)
        self.maxfun = int(maxfun)
        self._executor: Optional[ProcessPoolExecutor] = ProcessPoolExecutor(
            max_workers=self.parallel_workers,
            mp_context=multiprocessing.get_context("spawn"),
        )
        self._parent_by_expression: Dict[str, str] = {}
        self._parameters_by_expression: Dict[str, Dict[str, Dict[str, float]]] = {}

    def register_parent(self, expression: str, parent_expression: Optional[str]) -> None:
        if parent_expression:
            self._parent_by_expression[str(expression)] = str(parent_expression)

    def remember_parameters(
        self, expression: str, parameters: Mapping[str, Mapping[str, float]]
    ) -> None:
        self._parameters_by_expression[str(expression)] = {
            str(lane): {str(name): float(value) for name, value in values.items()}
            for lane, values in parameters.items()
        }

    def parent_warm_parameters(
        self, expression: str
    ) -> Optional[Dict[str, Dict[str, float]]]:
        parent = self._parent_by_expression.get(str(expression))
        return self._parameters_by_expression.get(parent) if parent else None

    def fit(
        self,
        *,
        warm_parameters: Optional[Mapping[str, Mapping[str, float]]] = None,
        **kwargs,
    ) -> Dict[str, Dict[str, float]]:
        if self._executor is None:
            raise RuntimeError("CleanMixedRestartJacobianFitter is closed")
        result = fit_lane_parameters_mixed_jacobian_parallel(
            warm_parameters=warm_parameters,
            parallel_workers=self.parallel_workers,
            maxiter=self.maxiter,
            maxfun=self.maxfun,
            executor=self._executor,
            **kwargs,
        )
        diagnostics = kwargs.get("diagnostics")
        if isinstance(diagnostics, dict):
            diagnostics["start_strategy_id"] = CLEAN_START_STRATEGY_ID
            diagnostics["formal_method_adapter"] = (
                "cosydelay_v9_clean_from_scratch.clean_fitter"
            )
        return result

    def __call__(self, universal_expr: str, **kwargs) -> Dict[str, Dict[str, float]]:
        warm = self.parent_warm_parameters(universal_expr)
        result = self.fit(
            universal_expr=universal_expr,
            warm_parameters=warm,
            **kwargs,
        )
        self.remember_parameters(universal_expr, result)
        return result

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None

    def __enter__(self) -> "CleanMixedRestartJacobianFitter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
