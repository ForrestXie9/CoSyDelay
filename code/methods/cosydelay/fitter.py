"""CoSyDelay accelerated fitter with a pre-screened nonlinear range profile.

The objective, all-positive log coordinates, ten starts, restart scheduler,
and Training-only restart selection are fixed by the released protocol. The
nonlinear range profile uses power exponents ``[0.01, 8]`` and exponential
coefficients ``[1e-5, 2]``.
"""

from __future__ import annotations

import copy
import os
import traceback
from typing import Mapping, Optional

from methods.cosydelay_v19_accelerated_equivalent.fitter import PersistentAcceleratedEquivalentFitter
from methods.cosydelay_v18_fit_timeout_guard.fit_timeout import _restore_rng
from methods.cosydelay_v16_manuscript_principlewise.fitter import legacy_numeric_r9_probe_disabled, scrub_legacy_probe_fields
from methods.prospective_optimizer_conditioning_v1.role_policy import coefficient_bounds_for_expression
from .range_policy import (
    COSYDELAY_RANGE_PROFILE,
    COSYDELAY_RANGE_PROFILE_ID,
    COSYDELAY_RANGE_SELECTION_MANIFEST,
)


class StableRangeAcceleratedFitter(PersistentAcceleratedEquivalentFitter):
    """Run the released restart schedule with audited role-wise bounds."""

    def fit(self, *, universal_expr: str, warm_parameters: Optional[Mapping] = None, **kwargs):
        bounds = coefficient_bounds_for_expression(
            str(universal_expr), COSYDELAY_RANGE_PROFILE, reject_nonlinear_role_conflicts=False
        )
        diagnostics = kwargs.get("diagnostics")
        result = super().fit(
            universal_expr=str(universal_expr),
            warm_parameters=warm_parameters,
            coefficient_bounds_override=bounds,
            **kwargs,
        )
        if isinstance(diagnostics, dict):
            diagnostics.update(
                {
                    "formal_method_adapter": "cosydelay.fitter",
                    "coefficient_range_profile": COSYDELAY_RANGE_PROFILE_ID,
                    "coefficient_bounds_by_name": {
                        name: [float(lower), float(upper)]
                        for name, (lower, upper) in bounds.items()
                    },
                    "coefficient_bound_selection_data": "predeclared_after_training_only_paired_screen",
                    "coefficient_range_selection_manifest": dict(COSYDELAY_RANGE_SELECTION_MANIFEST),
                }
            )
        return result


def accelerated_fit_supervisor_main(connection, static_kwargs: dict, settings: dict) -> None:
    """Spawn-safe fitting supervisor for the released CoSyDelay profile."""
    fitter = None
    try:
        fitter = StableRangeAcceleratedFitter(
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
            diagnostics: dict = {}
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
                # Preserve an explicit provenance marker at the supervisor
                # boundary.  The V18 guarded wrapper serializes this
                # dictionary unchanged; recording it here makes the CoSyDelay
                # post-run audit robust even when an inner compatibility
                # layer replaces the diagnostics object.
                diagnostics.setdefault(
                    "formal_method_adapter",
                    "cosydelay.fitter",
                )
                connection.send({
                    "kind": "fit_succeeded", "request_id": request_id,
                    "fitted": fitted, "diagnostics": diagnostics,
                    "rng_state": copy.deepcopy(rng.bit_generator.state) if rng is not None else None,
                })
            except BaseException as exc:
                try:
                    connection.send({
                        "kind": "fit_failed", "request_id": request_id,
                        "exception_type": type(exc).__name__, "message": str(exc)[:500],
                        "traceback": traceback.format_exc()[-4000:],
                    })
                finally:
                    return
    except (EOFError, BrokenPipeError):
        return
    except BaseException as exc:
        try:
            connection.send({
                "kind": "supervisor_failed", "exception_type": type(exc).__name__,
                "message": str(exc)[:500], "traceback": traceback.format_exc()[-4000:],
            })
        except BaseException:
            pass
    finally:
        if fitter is not None:
            try:
                fitter.close()
            except BaseException:
                pass
        try:
            connection.close()
        except BaseException:
            pass
