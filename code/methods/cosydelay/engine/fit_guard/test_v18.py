"""Regression and fault-injection tests for the V18 bounded-fit repair."""

from __future__ import annotations

import copy
import json
import multiprocessing
import os
from pathlib import Path
import time
import unittest

import numpy as np
import pandas as pd
import psutil

from methods.cosydelay.engine.retry_protocol import (
    launch_p10g10_i1_i6 as v17_launcher,
)
from methods.cosydelay.engine.retry_protocol.contract import V17_CONTRACT
from methods.cosydelay.engine.retry_protocol.policy import V17_POLICY

from .contract import V18_CONTRACT, validate_v18_contract
from .fit_timeout import (
    CandidateFitGuardError,
    GuardedCandidateFitter,
    _restore_rng,
)
from .integration import FitRejectionRegistry
from .policy import V18_POLICY
from .prefit_r7 import classify_prefit_r7
from .run_p10g10_training import validate_fit_guard_audit
from .source_manifest import HERE, validate_v18_source_manifest


def _sleep_forever() -> None:
    while True:
        time.sleep(60.0)


def _fake_supervisor(connection, static_kwargs: dict, settings: dict) -> None:
    del static_kwargs, settings
    descendants: list[multiprocessing.Process] = []
    try:
        connection.send({"kind": "ready", "pid": os.getpid()})
        while True:
            request = connection.recv()
            if request.get("kind") == "shutdown":
                connection.send({"kind": "stopped", "pid": os.getpid()})
                return
            expression = str(request["expression"])
            if expression == "HANG":
                child = multiprocessing.get_context("spawn").Process(
                    target=_sleep_forever,
                    name="cosydelay-v18-fault-descendant",
                    daemon=False,
                )
                child.start()
                descendants.append(child)
                _sleep_forever()
            if expression == "CRASH":
                os._exit(23)
            rng = _restore_rng(request.get("rng"))
            if rng is not None:
                rng.random()
            connection.send(
                {
                    "kind": "fit_succeeded",
                    "request_id": int(request["request_id"]),
                    "fitted": {"lane": {"a1": 1.0}},
                    "diagnostics": {"fake": True},
                    "rng_state": (
                        copy.deepcopy(rng.bit_generator.state)
                        if rng is not None
                        else None
                    ),
                }
            )
    except (EOFError, BrokenPipeError):
        return
    finally:
        for child in descendants:
            if child.is_alive():
                child.terminate()
            child.join(timeout=1.0)


def _static_kwargs() -> dict:
    frame = pd.DataFrame(
        {
            "flow_lane": [0.1, 0.2],
            "GR_phase": [0.3, 0.4],
            "Cycle_Time": [90.0, 100.0],
        }
    )
    return {
        "df": frame,
        "lanes": ["lane"],
        "lane_to_approach": {"lane": "S"},
        "approach_targets": {"S": pd.Series([1.0, 2.0])},
        "intersection_id": 1,
        "prepared_context": {},
    }


def _fit_call_kwargs(static: dict, *, rng: np.random.Generator) -> dict:
    return {
        **static,
        "rng": rng,
        "diagnostics": {},
        "n_restarts": 10,
    }


class V18ContractTests(unittest.TestCase):
    def test_contract_and_manifest(self) -> None:
        validate_v18_contract()
        validate_v18_source_manifest()
        self.assertEqual(V18_CONTRACT.population, 10)
        self.assertEqual(V18_CONTRACT.generations, 10)
        self.assertEqual(V18_CONTRACT.optimizer_restarts, 10)
        self.assertEqual(V18_CONTRACT.approach_workers_cap, 4)
        self.assertEqual(V18_CONTRACT.candidate_fit_wall_timeout_seconds, 100.0)
        self.assertTrue(V18_CONTRACT.prefit_r7_gate_enabled)
        self.assertTrue(V18_POLICY.coefficient_robust_r9_prefit_gate)
        self.assertFalse(V18_CONTRACT.timed_out_candidate_counts_toward_budget)

    def test_only_declared_policy_and_contract_changes(self) -> None:
        allowed_policy = {
            "method_id",
            "method_status",
            "execution_contract_version",
            "coefficient_robust_r9_prefit_gate",
        }
        for key, value in V17_POLICY.to_dict().items():
            if key not in allowed_policy:
                self.assertEqual(V18_POLICY.to_dict()[key], value, key)
        allowed_contract = {
            "schema_version",
            "method_id",
            "prefit_filter_scope",
            "prefit_physical_principle_rejection_enabled",
            "physical_contract_changed_from_v16",
        }
        for key, value in V17_CONTRACT.to_dict().items():
            if key not in allowed_contract:
                self.assertEqual(V18_CONTRACT.to_dict()[key], value, key)

    def test_completed_v17_shared_source_drift_is_known_and_limited(self) -> None:
        protocol_path = (
            HERE.parent
            / "cosydelay_v17_cross_batch_retry"
            / "experiments"
            / "v17_p10g10_i1_i6_20260813_run01"
            / "FROZEN_PROTOCOL.json"
        )
        if not protocol_path.is_file():
            self.skipTest("completed V17 run01 protocol is not present")
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        expected = protocol["source_sha256"]
        observed = v17_launcher._source_hashes()
        drift = {
            key
            for key in set(expected) | set(observed)
            if expected.get(key) != observed.get(key)
        }
        # V17's completed artifacts remain frozen.  A later shared prompt/parser
        # edit changed this one active workspace file before V18
        # was frozen; V18 records the current file in its own source manifest.
        self.assertEqual(drift, {"expression_adaptation_lane.py"})
        self.assertEqual(
            expected["expression_adaptation_lane.py"],
            "8d86959457a8228d7e6682088c8d6a7a1c9defa03e540bcd270bfee433b6eb6c",
        )
        self.assertEqual(
            observed["expression_adaptation_lane.py"],
            "e71652368739b7651c0274ad23c5df90ad2373d0c97b98f1917635e0960ac629",
        )

    def test_active_v18_launchers_use_bounded_popen(self) -> None:
        for name in ("launch_p10g10_i1_i6.py", "launch_formal_and_test.py"):
            text = (HERE / name).read_text(encoding="utf-8")
            self.assertIn("subprocess.Popen(", text)
            self.assertIn("wait_subprocess_bounded(", text)
            self.assertNotIn("subprocess.run(", text)


class V18RegistryTests(unittest.TestCase):
    def test_rejected_canonical_expression_cannot_reenter(self) -> None:
        registry = FitRejectionRegistry()
        expression = "Cycle_Time * a1 / GR_phase"
        registry.reject(
            expression,
            stage="candidate_evaluation_exception",
            exception_type="CandidateFitTimeoutError",
        )
        passed, reason = registry.validate("a1 * Cycle_Time / GR_phase")
        self.assertFalse(passed)
        self.assertIn("already rejected", reason)
        self.assertFalse(
            registry.candidate_events[0][
                "counts_toward_successful_candidate_budget"
            ]
        )


class V18PrefitR7Tests(unittest.TestCase):
    def test_observed_stalling_formula_is_rejected_as_proven_finite(self) -> None:
        expression = (
            "Cycle_Time * (a1 + a2 * flow_lane**a3 * "
            "(1 + a4 * exp(-a5 * GR_phase)) + "
            "a6 / (GR_phase**a7 + a8))"
        )
        decision = classify_prefit_r7(expression)
        self.assertEqual(decision.verdict, "proven_finite")
        self.assertFalse(decision.passed)
        self.assertIn("_positive_a6/_positive_a8", decision.endpoint or "")
        self.assertFalse(decision.uses_training_values)
        self.assertFalse(decision.uses_targets)
        self.assertFalse(decision.uses_validation_or_test)

    def test_unproved_inverse_green_candidate_is_allowed_to_bounded_fit(self) -> None:
        expression = (
            "Cycle_Time * (a1 + a2 * (flow_lane / GR_phase)**a3 "
            "+ a4 / GR_phase**a5)"
        )
        decision = classify_prefit_r7(expression)
        self.assertEqual(decision.verdict, "unknown_allow")
        self.assertTrue(decision.passed)

    def test_prefit_implementation_avoids_general_symbolic_limit(self) -> None:
        source = (HERE / "prefit_r7.py").read_text(encoding="utf-8")
        self.assertNotIn("sp.limit(", source)
        self.assertNotIn("sp.simplify(", source)
        self.assertNotIn("sp.factor(", source)
        self.assertNotIn("sp.cancel(", source)


class V18ProcessFaultTests(unittest.TestCase):
    def _fitter(self, *, timeout: float = 0.35) -> GuardedCandidateFitter:
        return GuardedCandidateFitter(
            static_fit_kwargs=_static_kwargs(),
            parallel_workers=4,
            maxiter=200,
            maxfun=20_000,
            fit_timeout_seconds=timeout,
            startup_timeout_seconds=5.0,
            close_timeout_seconds=1.0,
            supervisor_target=_fake_supervisor,
        )

    def test_success_commits_exact_rng_state(self) -> None:
        fitter = self._fitter()
        static = fitter.static_fit_kwargs
        observed_rng = np.random.default_rng(12345)
        expected_rng = np.random.default_rng(12345)
        expected_rng.random()
        diagnostics = {}
        kwargs = _fit_call_kwargs(static, rng=observed_rng)
        kwargs["diagnostics"] = diagnostics
        with fitter:
            fitted = fitter.fit(
                universal_expr="OK",
                warm_parameters=None,
                **kwargs,
            )
        self.assertEqual(fitted, {"lane": {"a1": 1.0}})
        self.assertEqual(observed_rng.bit_generator.state, expected_rng.bit_generator.state)
        self.assertTrue(diagnostics["fake"])
        self.assertTrue(
            diagnostics["optimizer_rng_state_committed_after_success"]
        )

    def test_hang_kills_descendant_restarts_and_preserves_rng(self) -> None:
        fitter = self._fitter(timeout=0.4)
        static = fitter.static_fit_kwargs
        rng = np.random.default_rng(99)
        initial_state = copy.deepcopy(rng.bit_generator.state)
        with fitter:
            first_pid = fitter.supervisor_pid
            with self.assertRaises(CandidateFitGuardError):
                fitter.fit(
                    universal_expr="HANG",
                    warm_parameters=None,
                    **_fit_call_kwargs(static, rng=rng),
                )
            restarted_pid = fitter.supervisor_pid
            self.assertNotEqual(first_pid, restarted_pid)
            self.assertEqual(rng.bit_generator.state, initial_state)
            fitter.fit(
                universal_expr="RECOVERED",
                warm_parameters=None,
                **_fit_call_kwargs(static, rng=rng),
            )
        timeout_event = next(
            event for event in fitter.audit if event.get("event") == "fit_timeout"
        )
        tree = timeout_event["cleanup"]["tree_cleanup"]
        self.assertGreaterEqual(len(tree["targeted_pids"]), 2)
        self.assertEqual(tree["still_alive_pids"], [])
        for pid in tree["targeted_pids"]:
            self.assertFalse(psutil.pid_exists(int(pid)))

    def test_crash_restarts_and_next_fit_succeeds(self) -> None:
        fitter = self._fitter(timeout=2.0)
        static = fitter.static_fit_kwargs
        rng = np.random.default_rng(7)
        initial_state = copy.deepcopy(rng.bit_generator.state)
        with fitter:
            with self.assertRaises(CandidateFitGuardError):
                fitter.fit(
                    universal_expr="CRASH",
                    warm_parameters=None,
                    **_fit_call_kwargs(static, rng=rng),
                )
            self.assertEqual(rng.bit_generator.state, initial_state)
            result = fitter.fit(
                universal_expr="AFTER_CRASH",
                warm_parameters=None,
                **_fit_call_kwargs(static, rng=rng),
            )
        self.assertEqual(result, {"lane": {"a1": 1.0}})
        self.assertTrue(
            any(event.get("event") == "fit_worker_error" for event in fitter.audit)
            or any(event.get("event") == "fit_timeout" for event in fitter.audit)
        )

    def test_audit_recomputes_success_budget_and_timeout_rejection(self) -> None:
        fitter = self._fitter(timeout=0.4)
        static = fitter.static_fit_kwargs
        rng = np.random.default_rng(101)
        registry = FitRejectionRegistry()
        with fitter:
            fitter.fit(
                universal_expr="ACCEPTED",
                warm_parameters=None,
                **_fit_call_kwargs(static, rng=rng),
            )
            registry.accept("ACCEPTED")
            with self.assertRaises(CandidateFitGuardError):
                fitter.fit(
                    universal_expr="HANG",
                    warm_parameters=None,
                    **_fit_call_kwargs(static, rng=rng),
                )
            registry.reject(
                "HANG",
                stage="candidate_evaluation_exception",
                exception_type="CandidateFitTimeoutError",
            )
        summary = validate_fit_guard_audit(
            {
                "fit_events": fitter.audit,
                "candidate_events": registry.candidate_events,
                "prefit_r7_events": [
                    {
                        "event": "prefit_r7_decision",
                        "verdict": "unknown_allow",
                        "passed": True,
                        "expression_sha256": next(
                            event["expression_sha256"]
                            for event in registry.candidate_events
                            if event["event"] == "candidate_accepted"
                        ),
                        "accuracy_metrics_present": False,
                        "validation_or_test_data_present": False,
                        "counts_toward_successful_candidate_budget": False,
                    }
                ],
                "rejected_canonical_expressions": [],
            },
            expected_candidates=1,
        )
        self.assertEqual(summary["successful_candidates_counted"], 1)
        self.assertEqual(summary["fit_timeouts"], 1)
        self.assertEqual(summary["remaining_descendant_pids"], [])


if __name__ == "__main__":
    unittest.main()
