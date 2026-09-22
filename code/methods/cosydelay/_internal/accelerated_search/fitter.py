"""Persistent task pool retaining every V18 coefficient-fit decision."""
from __future__ import annotations

import multiprocessing
import copy
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from typing import Mapping, Optional

from .parallel_restart_impl import fit_lane_parameters_restart_parallel
from methods.cosydelay._internal.fit_guard.fit_timeout import _restore_rng
from methods.cosydelay._internal.training_protocol.fitter import (
    legacy_numeric_r9_probe_disabled,
    scrub_legacy_probe_fields,
)


class PersistentAcceleratedEquivalentFitter:
    """V19 simple restart policy: 9 independent local starts plus 1 Sobol."""
    def __init__(self, *, parallel_workers: int, maxiter: int, maxfun: int) -> None:
        if int(parallel_workers) != 4:
            raise ValueError("V18-equivalent overlay requires the declared four approaches")
        self.parallel_workers=8
        self.maxiter=int(maxiter); self.maxfun=int(maxfun)
        self._executor=ProcessPoolExecutor(max_workers=8,mp_context=multiprocessing.get_context("spawn"))

    def fit(self, *, warm_parameters: Optional[Mapping]=None, **kwargs):
        # Structure-level regeneration need not preserve a coefficient's role.
        # Deliberately ignore inherited parent values: every candidate has the
        # same independent ten-start schedule.
        result=fit_lane_parameters_restart_parallel(warm_parameters=None,parallel_workers=self.parallel_workers,maxiter=self.maxiter,maxfun=self.maxfun,executor=self._executor,r9_constrained_restart_selection=False,precompile_numexpr=True,**kwargs)
        diagnostics=kwargs.get("diagnostics")
        if isinstance(diagnostics,dict):
            diagnostics["formal_method_adapter"]="cosydelay_v19_accelerated_equivalent.fitter"
            diagnostics["restart_selection"]="minimum_training_mse"
            diagnostics["equivalence_scope"]="same_expression_starts_bounds_objective_gradient_and_selection_as_v18"
            diagnostics["parent_warm_start_available"]=False
            diagnostics["restart_policy"]="nine_official_local_replay_plus_one_sobol_role_wide_for_all_candidates"
            diagnostics["parent_warm_parameters_ignored"]=bool(warm_parameters is not None)
        return result

    def close(self):
        if self._executor is not None:
            self._executor.shutdown(wait=True); self._executor=None


def accelerated_fit_supervisor_main(connection, static_kwargs: dict, settings: dict) -> None:
    """Spawn-safe V18 guard supervisor that explicitly owns the V19 fitter."""
    fitter = None
    try:
        fitter = PersistentAcceleratedEquivalentFitter(
            parallel_workers=int(settings["parallel_workers"]),
            maxiter=int(settings["maxiter"]),
            maxfun=int(settings["maxfun"]),
        )
        connection.send({"kind": "ready", "pid": os.getpid()})
        while True:
            request = connection.recv()
            kind = request.get("kind")
            if kind == "shutdown":
                fitter.close(); fitter = None
                connection.send({"kind": "stopped", "pid": os.getpid()})
                return
            if kind != "fit":
                raise ValueError(f"unknown supervisor request: {kind!r}")
            request_id = int(request["request_id"])
            diagnostics = {}
            rng = _restore_rng(request.get("rng"))
            fit_kwargs = dict(static_kwargs)
            fit_kwargs.update(dict(request.get("dynamic_kwargs") or {}))
            try:
                with legacy_numeric_r9_probe_disabled():
                    fitted = fitter.fit(
                        universal_expr=str(request["expression"]),
                        warm_parameters=request.get("warm_parameters"),
                        diagnostics=diagnostics,
                        rng=rng,
                        **fit_kwargs,
                    )
                scrub_legacy_probe_fields(diagnostics)
                connection.send({
                    "kind": "fit_succeeded",
                    "request_id": request_id,
                    "fitted": fitted,
                    "diagnostics": diagnostics,
                    "rng_state": copy.deepcopy(rng.bit_generator.state) if rng is not None else None,
                })
            except BaseException as exc:
                try:
                    connection.send({
                        "kind": "fit_failed",
                        "request_id": request_id,
                        "exception_type": type(exc).__name__,
                        "message": str(exc)[:500],
                        "traceback": traceback.format_exc()[-4000:],
                    })
                finally:
                    return
    except (EOFError, BrokenPipeError):
        return
    except BaseException as exc:
        try:
            connection.send({
                "kind": "supervisor_failed",
                "exception_type": type(exc).__name__,
                "message": str(exc)[:500],
                "traceback": traceback.format_exc()[-4000:],
            })
        except BaseException:
            pass
    finally:
        if fitter is not None:
            try: fitter.close()
            except BaseException: pass
        try: connection.close()
        except BaseException: pass
