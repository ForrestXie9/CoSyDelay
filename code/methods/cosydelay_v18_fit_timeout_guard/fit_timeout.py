"""Killable wall-clock guard around one complete coefficient-fitting call.

The retained fitter and its persistent four-approach process pool live inside
one supervisor process.  The evolutionary process commits the returned
optimizer RNG state only after a successful response.  A timeout or worker
failure terminates the supervisor and every descendant, starts a clean
supervisor, and rejects the current expression without consuming the official
110-success candidate budget.
"""

from __future__ import annotations

import copy
import hashlib
import multiprocessing
import os
import subprocess
import time
import traceback
from typing import Any, Callable, Mapping, Optional

import numpy as np
import psutil

from methods.cosydelay_v15_symbolic_r9.fitter import (
    PersistentSymbolicR9Fitter,
)
from methods.cosydelay_v16_manuscript_principlewise.fitter import (
    legacy_numeric_r9_probe_disabled,
    scrub_legacy_probe_fields,
)


class CandidateFitGuardError(RuntimeError):
    """Base class for bounded-fit failures visible to the parent evaluator."""


class CandidateFitTimeoutError(CandidateFitGuardError):
    """The complete candidate fit exceeded its fixed wall-clock allowance."""


class CandidateFitWorkerError(CandidateFitGuardError):
    """The isolated fitting supervisor failed or violated its protocol."""


def _expression_sha256(expression: str) -> str:
    return hashlib.sha256(str(expression).encode("utf-8")).hexdigest()


def terminate_process_tree(
    pid: int,
    *,
    expected_create_time: Optional[float] = None,
    grace_seconds: float = 2.0,
) -> dict[str, Any]:
    """Terminate one exact process and all descendants, then hard-kill survivors."""
    pid = int(pid)
    report: dict[str, Any] = {
        "root_pid": pid,
        "targeted_pids": [],
        "terminated_pids": [],
        "killed_pids": [],
        "still_alive_pids": [],
        "root_identity_matched": False,
    }
    if pid <= 0 or pid == os.getpid():
        report["refused_reason"] = "invalid_or_current_process_pid"
        return report
    try:
        root = psutil.Process(pid)
        observed_create_time = float(root.create_time())
    except psutil.Error:
        report["root_missing"] = True
        return report
    if expected_create_time is not None and abs(
        observed_create_time - float(expected_create_time)
    ) > 0.01:
        report.update(
            {
                "refused_reason": "pid_creation_time_mismatch",
                "observed_create_time": observed_create_time,
                "expected_create_time": float(expected_create_time),
            }
        )
        return report
    report["root_identity_matched"] = True
    try:
        descendants = root.children(recursive=True)
    except psutil.Error:
        descendants = []
    processes = list(dict.fromkeys([*descendants, root]))
    report["targeted_pids"] = [int(process.pid) for process in processes]
    for process in reversed(processes):
        try:
            process.terminate()
            report["terminated_pids"].append(int(process.pid))
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            report.setdefault("terminate_errors", []).append(
                {"pid": int(process.pid), "type": type(exc).__name__}
            )
    _, alive = psutil.wait_procs(processes, timeout=max(0.0, float(grace_seconds)))
    for process in alive:
        try:
            process.kill()
            report["killed_pids"].append(int(process.pid))
        except psutil.NoSuchProcess:
            pass
        except psutil.Error as exc:
            report.setdefault("kill_errors", []).append(
                {"pid": int(process.pid), "type": type(exc).__name__}
            )
    _, alive = psutil.wait_procs(alive, timeout=max(0.0, float(grace_seconds)))
    report["still_alive_pids"] = [int(process.pid) for process in alive]
    return report


def wait_subprocess_bounded(
    process: subprocess.Popen,
    *,
    timeout_seconds: float,
    terminate_grace_seconds: float = 2.0,
) -> tuple[int, bool, dict[str, Any]]:
    """Wait for a subprocess with a hard process-tree cleanup fallback."""
    try:
        create_time = float(psutil.Process(process.pid).create_time())
    except psutil.Error:
        create_time = None
    try:
        return int(process.wait(timeout=float(timeout_seconds))), False, {}
    except subprocess.TimeoutExpired:
        report = terminate_process_tree(
            process.pid,
            expected_create_time=create_time,
            grace_seconds=terminate_grace_seconds,
        )
        try:
            process.wait(timeout=max(1.0, float(terminate_grace_seconds)))
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=max(1.0, float(terminate_grace_seconds)))
        return 124, True, report


def _restore_rng(payload: Optional[Mapping[str, Any]]) -> Optional[np.random.Generator]:
    if payload is None:
        return None
    bit_generator_name = str(payload["bit_generator"])
    bit_generator_type = getattr(np.random, bit_generator_name, None)
    if bit_generator_type is None:
        raise ValueError(f"unsupported NumPy bit generator: {bit_generator_name}")
    rng = np.random.Generator(bit_generator_type())
    rng.bit_generator.state = copy.deepcopy(payload["state"])
    return rng


def _fit_supervisor_main(connection, static_kwargs: dict, settings: dict) -> None:
    """Child protocol loop; intentionally owns every approach worker process."""
    fitter = None
    try:
        fitter = PersistentSymbolicR9Fitter(
            parallel_workers=int(settings["parallel_workers"]),
            maxiter=int(settings["maxiter"]),
            maxfun=int(settings["maxfun"]),
        )
        connection.send({"kind": "ready", "pid": os.getpid()})
        while True:
            request = connection.recv()
            kind = request.get("kind")
            if kind == "shutdown":
                fitter.close()
                fitter = None
                connection.send({"kind": "stopped", "pid": os.getpid()})
                return
            if kind != "fit":
                raise ValueError(f"unknown supervisor request: {kind!r}")
            request_id = int(request["request_id"])
            diagnostics: dict[str, Any] = {}
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
                connection.send(
                    {
                        "kind": "fit_succeeded",
                        "request_id": request_id,
                        "fitted": fitted,
                        "diagnostics": diagnostics,
                        "rng_state": (
                            copy.deepcopy(rng.bit_generator.state)
                            if rng is not None
                            else None
                        ),
                    }
                )
            except BaseException as exc:
                try:
                    connection.send(
                        {
                            "kind": "fit_failed",
                            "request_id": request_id,
                            "exception_type": type(exc).__name__,
                            "message": str(exc)[:500],
                            "traceback": traceback.format_exc()[-4000:],
                        }
                    )
                finally:
                    # The pool may be broken or may retain unfinished work.
                    # The parent replaces this entire process tree.
                    return
    except (EOFError, BrokenPipeError):
        return
    except BaseException as exc:
        try:
            connection.send(
                {
                    "kind": "supervisor_failed",
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:500],
                    "traceback": traceback.format_exc()[-4000:],
                }
            )
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


class GuardedCandidateFitter:
    """V15-compatible fitter with a transaction-like wall-clock boundary."""

    _STATIC_KEYS = (
        "df",
        "lanes",
        "lane_to_approach",
        "approach_targets",
        "intersection_id",
        "prepared_context",
    )

    def __init__(
        self,
        *,
        static_fit_kwargs: Mapping[str, Any],
        parallel_workers: int,
        maxiter: int,
        maxfun: int,
        fit_timeout_seconds: float,
        startup_timeout_seconds: float,
        close_timeout_seconds: float,
        supervisor_target: Callable = _fit_supervisor_main,
    ) -> None:
        missing = [key for key in self._STATIC_KEYS if key not in static_fit_kwargs]
        if missing:
            raise ValueError(f"missing static fit arguments: {missing}")
        if min(
            parallel_workers,
            maxiter,
            maxfun,
            fit_timeout_seconds,
            startup_timeout_seconds,
            close_timeout_seconds,
        ) <= 0:
            raise ValueError("worker counts, optimizer budgets, and timeouts must be positive")
        self.static_fit_kwargs = {
            key: static_fit_kwargs[key] for key in self._STATIC_KEYS
        }
        self.settings = {
            "parallel_workers": int(parallel_workers),
            "maxiter": int(maxiter),
            "maxfun": int(maxfun),
        }
        self.fit_timeout_seconds = float(fit_timeout_seconds)
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.close_timeout_seconds = float(close_timeout_seconds)
        self._supervisor_target = supervisor_target
        self._context = multiprocessing.get_context("spawn")
        self._connection = None
        self._process = None
        self._process_create_time: Optional[float] = None
        self._supervisor_generation = 0
        self._request_id = 0
        self._closed = False
        self.audit: list[dict[str, Any]] = []

    @property
    def supervisor_pid(self) -> Optional[int]:
        return int(self._process.pid) if self._process is not None else None

    def _start_supervisor(self) -> None:
        if self._closed:
            raise RuntimeError("GuardedCandidateFitter is closed")
        if self._process is not None:
            raise RuntimeError("a fitting supervisor is already attached")
        parent_connection, child_connection = self._context.Pipe(duplex=True)
        process = self._context.Process(
            target=self._supervisor_target,
            args=(child_connection, self.static_fit_kwargs, self.settings),
            name="cosydelay-v18-fit-supervisor",
            daemon=False,
        )
        started = time.perf_counter()
        try:
            process.start()
            child_connection.close()
            self._connection = parent_connection
            self._process = process
            try:
                self._process_create_time = float(
                    psutil.Process(process.pid).create_time()
                )
            except psutil.Error:
                self._process_create_time = None
            if not parent_connection.poll(self.startup_timeout_seconds):
                report = self._detach_supervisor(
                    force=True, reason="startup_timeout"
                )
                raise CandidateFitWorkerError(
                    "coefficient-fit supervisor did not become ready within "
                    f"{self.startup_timeout_seconds:.1f}s; cleanup={report}"
                )
            response = parent_connection.recv()
            if response.get("kind") != "ready" or int(
                response.get("pid", -1)
            ) != int(process.pid):
                report = self._detach_supervisor(
                    force=True, reason="invalid_startup_response"
                )
                raise CandidateFitWorkerError(
                    f"invalid coefficient-fit supervisor handshake; cleanup={report}"
                )
            self._supervisor_generation += 1
            self.audit.append(
                {
                    "event": "supervisor_started",
                    "supervisor_generation": self._supervisor_generation,
                    "pid": int(process.pid),
                    "wall_seconds": float(time.perf_counter() - started),
                }
            )
        except BaseException:
            try:
                child_connection.close()
            except BaseException:
                pass
            if self._process is not None:
                self._detach_supervisor(force=True, reason="startup_exception")
            else:
                parent_connection.close()
            raise

    def _detach_supervisor(self, *, force: bool, reason: str) -> dict[str, Any]:
        process = self._process
        connection = self._connection
        create_time = self._process_create_time
        report: dict[str, Any] = {}
        if process is not None and process.is_alive() and not force:
            try:
                connection.send({"kind": "shutdown"})
                if connection.poll(self.close_timeout_seconds):
                    response = connection.recv()
                    if response.get("kind") != "stopped":
                        report["shutdown_protocol_error"] = response.get("kind")
                process.join(timeout=self.close_timeout_seconds)
            except (BrokenPipeError, EOFError, OSError) as exc:
                report["shutdown_error"] = type(exc).__name__
        if process is not None and process.is_alive():
            report["tree_cleanup"] = terminate_process_tree(
                process.pid,
                expected_create_time=create_time,
                grace_seconds=min(2.0, self.close_timeout_seconds),
            )
            process.join(timeout=max(1.0, self.close_timeout_seconds))
        if process is not None and process.is_alive():
            process.kill()
            process.join(timeout=max(1.0, self.close_timeout_seconds))
            report["multiprocessing_kill_fallback"] = True
        if process is not None:
            report["return_code"] = process.exitcode
            report["pid"] = process.pid
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None and not process.is_alive():
            try:
                process.close()
            except (OSError, ValueError):
                pass
        self._connection = None
        self._process = None
        self._process_create_time = None
        self.audit.append(
            {
                "event": "supervisor_detached",
                "supervisor_generation": self._supervisor_generation,
                "reason": str(reason),
                "forced": bool(force),
                "cleanup": report,
            }
        )
        return report

    def _restart_after_failure(self, reason: str) -> dict[str, Any]:
        cleanup = self._detach_supervisor(force=True, reason=reason)
        self._start_supervisor()
        return cleanup

    def _rng_payload(
        self, rng: Optional[np.random.Generator]
    ) -> Optional[dict[str, Any]]:
        if rng is None:
            return None
        return {
            "bit_generator": type(rng.bit_generator).__name__,
            "state": copy.deepcopy(rng.bit_generator.state),
        }

    def _validate_and_remove_static_kwargs(self, kwargs: dict[str, Any]) -> None:
        observed_intersection = int(kwargs.pop("intersection_id"))
        if observed_intersection != int(self.static_fit_kwargs["intersection_id"]):
            raise ValueError("intersection changed inside one guarded fitting runtime")
        observed_lanes = list(kwargs.pop("lanes"))
        if observed_lanes != list(self.static_fit_kwargs["lanes"]):
            raise ValueError("movement layout changed inside one guarded fitting runtime")
        observed_mapping = dict(kwargs.pop("lane_to_approach"))
        if observed_mapping != dict(self.static_fit_kwargs["lane_to_approach"]):
            raise ValueError("approach mapping changed inside one guarded fitting runtime")
        observed_targets = kwargs.pop("approach_targets")
        if set(observed_targets) != set(self.static_fit_kwargs["approach_targets"]):
            raise ValueError("approach targets changed inside one guarded fitting runtime")
        kwargs.pop("df")
        kwargs.pop("prepared_context")

    def fit(
        self,
        *,
        universal_expr: str,
        warm_parameters: Optional[Mapping] = None,
        **kwargs,
    ):
        if self._closed:
            raise RuntimeError("GuardedCandidateFitter is closed")
        if self._process is None:
            self._start_supervisor()
        diagnostics = kwargs.pop("diagnostics", None)
        rng = kwargs.pop("rng", None)
        self._validate_and_remove_static_kwargs(kwargs)
        self._request_id += 1
        request_id = self._request_id
        expression_hash = _expression_sha256(universal_expr)
        request = {
            "kind": "fit",
            "request_id": request_id,
            "expression": str(universal_expr),
            "warm_parameters": warm_parameters,
            "dynamic_kwargs": kwargs,
            "rng": self._rng_payload(rng),
        }
        started = time.perf_counter()
        try:
            self._connection.send(request)
        except (BrokenPipeError, EOFError, OSError) as exc:
            cleanup = self._restart_after_failure("request_send_failed")
            self.audit.append(
                {
                    "event": "fit_worker_error",
                    "request_id": request_id,
                    "expression_sha256": expression_hash,
                    "exception_type": type(exc).__name__,
                    "cleanup": cleanup,
                    "optimizer_rng_state_committed": False,
                }
            )
            raise CandidateFitWorkerError(
                "coefficient-fit supervisor failed while accepting a request"
            ) from exc
        elapsed = time.perf_counter() - started
        remaining = max(0.0, self.fit_timeout_seconds - elapsed)
        if not self._connection.poll(remaining):
            cleanup = self._restart_after_failure("candidate_fit_timeout")
            wall_seconds = float(time.perf_counter() - started)
            self.audit.append(
                {
                    "event": "fit_timeout",
                    "request_id": request_id,
                    "expression_sha256": expression_hash,
                    "wall_seconds": wall_seconds,
                    "timeout_seconds": self.fit_timeout_seconds,
                    "cleanup": cleanup,
                    "optimizer_rng_state_committed": False,
                }
            )
            raise CandidateFitTimeoutError(
                "candidate coefficient fitting exceeded the fixed "
                f"{self.fit_timeout_seconds:.1f}s wall-clock limit"
            )
        try:
            response = self._connection.recv()
        except (BrokenPipeError, EOFError, OSError) as exc:
            cleanup = self._restart_after_failure("response_receive_failed")
            self.audit.append(
                {
                    "event": "fit_worker_error",
                    "request_id": request_id,
                    "expression_sha256": expression_hash,
                    "exception_type": type(exc).__name__,
                    "cleanup": cleanup,
                    "optimizer_rng_state_committed": False,
                }
            )
            raise CandidateFitWorkerError(
                "coefficient-fit supervisor exited without a complete response"
            ) from exc
        response_wall_seconds = float(time.perf_counter() - started)
        if response_wall_seconds > self.fit_timeout_seconds:
            cleanup = self._restart_after_failure(
                "candidate_fit_timeout_during_result_collection"
            )
            self.audit.append(
                {
                    "event": "fit_timeout",
                    "request_id": request_id,
                    "expression_sha256": expression_hash,
                    "wall_seconds": response_wall_seconds,
                    "timeout_seconds": self.fit_timeout_seconds,
                    "cleanup": cleanup,
                    "optimizer_rng_state_committed": False,
                    "timeout_stage": "result_collection",
                }
            )
            raise CandidateFitTimeoutError(
                "candidate coefficient fitting and result collection exceeded "
                f"the fixed {self.fit_timeout_seconds:.1f}s wall-clock limit"
            )
        if int(response.get("request_id", -1)) != request_id:
            cleanup = self._restart_after_failure("response_id_mismatch")
            self.audit.append(
                {
                    "event": "fit_worker_error",
                    "request_id": request_id,
                    "expression_sha256": expression_hash,
                    "exception_type": "SupervisorProtocolError",
                    "cleanup": cleanup,
                    "optimizer_rng_state_committed": False,
                }
            )
            raise CandidateFitWorkerError(
                "coefficient-fit supervisor returned a mismatched request id; "
                f"cleanup={cleanup}"
            )
        if response.get("kind") != "fit_succeeded":
            exception_type = str(response.get("exception_type", "unknown"))
            cleanup = self._restart_after_failure("isolated_fit_failed")
            self.audit.append(
                {
                    "event": "fit_worker_error",
                    "request_id": request_id,
                    "expression_sha256": expression_hash,
                    "exception_type": exception_type,
                    "wall_seconds": float(time.perf_counter() - started),
                    "cleanup": cleanup,
                    "optimizer_rng_state_committed": False,
                }
            )
            raise CandidateFitWorkerError(
                "isolated coefficient fitting failed "
                f"({exception_type}); the candidate was not scored"
            )
        returned_rng_state = response.get("rng_state")
        if rng is not None:
            if returned_rng_state is None:
                cleanup = self._restart_after_failure("missing_rng_state")
                self.audit.append(
                    {
                        "event": "fit_worker_error",
                        "request_id": request_id,
                        "expression_sha256": expression_hash,
                        "exception_type": "MissingRngState",
                        "cleanup": cleanup,
                        "optimizer_rng_state_committed": False,
                    }
                )
                raise CandidateFitWorkerError(
                    "successful coefficient fit omitted optimizer RNG state; "
                    f"cleanup={cleanup}"
                )
            rng.bit_generator.state = copy.deepcopy(returned_rng_state)
        returned_diagnostics = dict(response.get("diagnostics") or {})
        returned_diagnostics.update(
            {
                "candidate_fit_guard": "v18_isolated_supervisor_wall_timeout",
                "candidate_fit_timeout_seconds": self.fit_timeout_seconds,
                "fit_supervisor_generation": self._supervisor_generation,
                "optimizer_rng_state_committed_after_success": rng is not None,
            }
        )
        if isinstance(diagnostics, dict):
            diagnostics.update(returned_diagnostics)
        wall_seconds = float(time.perf_counter() - started)
        self.audit.append(
            {
                "event": "fit_succeeded",
                "request_id": request_id,
                "expression_sha256": expression_hash,
                "wall_seconds": wall_seconds,
                "timeout_seconds": self.fit_timeout_seconds,
                "supervisor_generation": self._supervisor_generation,
                "optimizer_rng_state_committed": rng is not None,
            }
        )
        return response["fitted"]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process is not None:
            self._detach_supervisor(force=False, reason="normal_close")

    def __enter__(self):
        self._start_supervisor()
        return self

    def __exit__(self, exc_type, exc, traceback_value) -> None:
        del exc_type, exc, traceback_value
        self.close()
