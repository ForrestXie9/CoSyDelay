"""Bound native symbolic checks without changing successful verifier results.

Some exact fitted-coefficient substitutions allocate enormous rational powers
before the verifier's per-limit timeout starts. The whole symbolic operation,
not just ``limit()``, must therefore run outside the search process.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
import multiprocessing
import os
from pathlib import Path
import time

import psutil


PREFIT_TIMEOUT_SECONDS = 8.0
PHYSICS_TIMEOUT_SECONDS = 30.0
SYMBOLIC_MEMORY_MB = 1536
SYMBOLIC_STARTUP_SECONDS = 20.0


class SymbolicCheckError(RuntimeError):
    """The current symbolic operation could not complete safely."""


def _memory_job(memory_mb):
    """On Windows, cap this worker AND its descendants with a kernel Job."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    class BasicLimit(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_longlong), ("job_time", ctypes.c_longlong),
                    ("flags", wintypes.DWORD), ("min_working_set", ctypes.c_size_t),
                    ("max_working_set", ctypes.c_size_t), ("active_processes", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class IOCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_ulonglong) for name in
                    ("reads", "writes", "other", "read_bytes", "write_bytes", "other_bytes")]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [("basic", BasicLimit), ("io", IOCounters),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int,
                                               ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    job = kernel.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits = ExtendedLimit()
    limits.basic.flags = 0x200 | 0x2000  # JOB_MEMORY | KILL_ON_JOB_CLOSE
    limits.job_memory = int(memory_mb * 1024**2)
    if not kernel.SetInformationJobObject(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    if not kernel.AssignProcessToJobObject(job, kernel.GetCurrentProcess()):
        error = ctypes.WinError(ctypes.get_last_error())
        kernel.CloseHandle(job)
        raise error
    # Keep this handle alive until process exit; never inherit it to children.
    return job


def _symbolic_worker(connection, memory_mb):
    job = None
    limits = None
    try:
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[name] = "1"
        from expression_rules import canonical_expression, structural_family_key
        from methods.cosydelay.engine.training_protocol.integration import validate_unambiguous_coefficient_roles
        from methods.cosydelay.engine.training_protocol.physics import (
            ManuscriptVerifierConfig, PersistentSymbolicLimitEvaluator,
            score_fitted_lanes_manuscript_principlewise,
        )
        from methods.cosydelay.engine.fit_guard.prefit_r7 import classify_prefit_r7
        from methods.cosydelay.engine.accelerated_search.prefit_r4 import validate_exact_time_dimension

        job = _memory_job(memory_mb)
        limits = PersistentSymbolicLimitEvaluator()
        connection.send({"kind": "ready", "hard_memory_job": job is not None})
        while True:
            request = connection.recv()
            if request["operation"] == "shutdown":
                return
            try:
                expression = request["expression"]
                if request["operation"] == "prefit":
                    for gate in (validate_unambiguous_coefficient_roles, validate_exact_time_dimension):
                        passed, reason = gate(expression)
                        if not passed:
                            raise ValueError(reason)
                    decision = classify_prefit_r7(expression)
                    if not decision.passed:
                        raise ValueError(decision.reason)
                    value = {"canonical": canonical_expression(expression),
                             "family": structural_family_key(expression)}
                elif request["operation"] == "physical":
                    value = score_fitted_lanes_manuscript_principlewise(
                        expression, request["parameters"], request["lanes"], request["features"],
                        config=ManuscriptVerifierConfig(), limit_evaluator=limits,
                    ).to_dict()
                else:
                    raise ValueError("unknown symbolic operation")
                connection.send({"kind": "result", "request_id": request["request_id"], "value": value})
            except Exception as exc:
                connection.send({"kind": "error", "request_id": request["request_id"],
                                 "error": f"{type(exc).__name__}: {str(exc)[:600]}",
                                 "fatal": isinstance(exc, (MemoryError, RecursionError))})
                if isinstance(exc, (MemoryError, RecursionError)):
                    return
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as exc:
        try:
            connection.send({"kind": "error", "error": f"{type(exc).__name__}: {str(exc)[:600]}", "fatal": True})
        except BaseException:
            pass
    finally:
        if limits is not None:
            limits.close()
        connection.close()
        # OS process cleanup closes job and terminates any surviving descendants.


class GuardedSymbolicChecks:
    def __init__(self, output, *, memory_mb=SYMBOLIC_MEMORY_MB, worker_target=_symbolic_worker):
        self.output = Path(output)
        self.memory_mb = memory_mb
        self.worker_target = worker_target
        self.process = self.connection = None
        self.worker_starts = self.request_id = 0
        self.hard_memory_job = False
        self.identity = None

    def _log(self, record):
        self.output.mkdir(parents=True, exist_ok=True)
        record.update(updated_utc=datetime.now(timezone.utc).isoformat(), worker_starts=self.worker_starts)
        with (self.output / "symbolic_guard.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _stop(self):
        process, connection = self.process, self.connection
        self.process = self.connection = None
        if process is not None:
            try:
                owned = psutil.Process(process.pid)
                if self.identity is not None and abs(owned.create_time() - self.identity) < .01:
                    for child in reversed(owned.children(recursive=True)):
                        try:
                            child.kill()
                        except psutil.NoSuchProcess:
                            pass
                    owned.kill()
            except psutil.NoSuchProcess:
                pass
            process.join(timeout=1)
            if not process.is_alive():
                process.close()
        if connection is not None:
            connection.close()

    def _receive(self, deadline):
        peak_mb = 0.
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise SymbolicCheckError("symbolic operation timed out")
            if self.connection.poll(min(.05, remaining)):
                try:
                    return self.connection.recv(), peak_mb
                except (EOFError, OSError) as exc:
                    raise SymbolicCheckError("symbolic worker exited without a result") from exc
            if not self.process.is_alive():
                raise SymbolicCheckError(f"symbolic worker exited with code {self.process.exitcode}")
            try:
                root = psutil.Process(self.process.pid)
                processes = [root, *root.children(recursive=True)]
                rss = sum(p.memory_info().rss for p in processes if p.is_running()) / 1024**2
                peak_mb = max(peak_mb, rss)
                if rss > self.memory_mb:
                    raise SymbolicCheckError(f"symbolic worker tree exceeded {self.memory_mb} MiB")
            except psutil.NoSuchProcess:
                continue

    def _start(self, deadline):
        if self.process is not None and self.process.is_alive():
            return
        self._stop()
        context = multiprocessing.get_context("spawn")
        self.connection, child = context.Pipe()
        self.process = context.Process(target=self.worker_target, args=(child, self.memory_mb))
        self.process.start()
        child.close()
        self.identity = psutil.Process(self.process.pid).create_time()
        self.worker_starts += 1
        response, _ = self._receive(min(deadline, time.monotonic() + SYMBOLIC_STARTUP_SECONDS))
        if response.get("kind") != "ready":
            raise SymbolicCheckError(response.get("error", "symbolic worker startup failed"))
        self.hard_memory_job = response.get("hard_memory_job", False)

    def call(self, operation, expression, deadline, *, timeout=None, **payload):
        started = time.monotonic()
        record = {"operation": operation, "expression": expression}
        try:
            if deadline <= started:
                raise SymbolicCheckError("shared search deadline reached")
            self._start(deadline)
            self.request_id += 1
            self.connection.send({"operation": operation, "expression": expression,
                                  "request_id": self.request_id, **payload})
            limit = timeout if timeout is not None else (
                PREFIT_TIMEOUT_SECONDS if operation == "prefit" else PHYSICS_TIMEOUT_SECONDS)
            response, peak_mb = self._receive(min(deadline, time.monotonic() + limit))
            record.update(peak_worker_tree_rss_mb=peak_mb, hard_memory_job=self.hard_memory_job)
            if response.get("request_id") != self.request_id:
                raise SymbolicCheckError("symbolic worker response identity mismatch")
            if response.get("kind") == "error":
                if response.get("fatal"):
                    raise SymbolicCheckError(response.get("error", "fatal symbolic error"))
                raise ValueError(response.get("error", "symbolic operation rejected"))
            if response.get("kind") != "result":
                raise SymbolicCheckError("invalid symbolic worker response")
            record["status"] = "complete"
            return response["value"]
        except (SymbolicCheckError, OSError, EOFError) as exc:
            record.update(status="resource_failure", error=str(exc))
            self._stop()
            raise SymbolicCheckError(str(exc)) from exc
        except ValueError as exc:
            record.update(status="rejected", error=str(exc))
            raise
        finally:
            record["wall_seconds"] = time.monotonic() - started
            self._log(record)

    def close(self):
        self._stop()
