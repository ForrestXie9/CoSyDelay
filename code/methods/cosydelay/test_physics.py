"""Regression tests for complete, killable symbolic verification."""
import os
import time

import pytest

from .physics import GuardedSymbolicChecks, SymbolicCheckError, _memory_job


EXPRESSION = "Cycle_Time * (a1 + a2 * flow_lane / GR_phase)"
FEATURES = ["flow_lane", "GR_phase", "Cycle_Time"]


def fault_worker(connection, memory_mb):
    job = _memory_job(memory_mb)
    connection.send({"kind": "ready", "hard_memory_job": job is not None})
    while True:
        request = connection.recv()
        if request["operation"] == "hang":
            time.sleep(60)
        if request["operation"] == "exit":
            os._exit(7)
        if request["operation"] == "memory":
            connection.send({"kind": "error", "request_id": request["request_id"],
                             "error": "MemoryError: simulated regression", "fatal": True})
            return
        connection.send({"kind": "result", "request_id": request["request_id"], "value": {"ok": True}})


@pytest.mark.parametrize("failure", ["hang", "exit", "memory"])
def test_faulty_symbolic_worker_does_not_crash_next_check(tmp_path, failure):
    guard = GuardedSymbolicChecks(tmp_path, worker_target=fault_worker)
    try:
        assert guard.call("echo", "", time.monotonic() + 20)["ok"]
        started = time.monotonic()
        with pytest.raises(SymbolicCheckError):
            guard.call(failure, "", time.monotonic() + 3, timeout=.2)
        assert time.monotonic() - started < 3
        assert guard.process is None
        assert guard.call("echo", "", time.monotonic() + 20)["ok"]
        assert guard.worker_starts == 2
        if os.name == "nt":
            assert guard.hard_memory_job
    finally:
        guard.close()


def test_successful_isolated_physics_equals_native_result(tmp_path):
    from .engine.training_protocol.physics import (
        ManuscriptVerifierConfig, PersistentSymbolicLimitEvaluator,
        score_fitted_lanes_manuscript_principlewise,
    )
    guard = GuardedSymbolicChecks(tmp_path)
    native = PersistentSymbolicLimitEvaluator()
    parameters = {"S_L": {"a1": .1, "a2": .5}}
    try:
        assert guard.call("prefit", EXPRESSION, time.monotonic() + 25)["canonical"]
        actual = guard.call("physical", EXPRESSION, time.monotonic() + 30,
                            parameters=parameters, lanes=["S_L"], features=FEATURES)
        expected = score_fitted_lanes_manuscript_principlewise(
            EXPRESSION, parameters, ["S_L"], FEATURES,
            config=ManuscriptVerifierConfig(), limit_evaluator=native,
        ).to_dict()
        for key in ("joint_pass", "score", "rule_scores", "lane_rule_scores"):
            assert actual[key] == expected[key]
        assert actual["joint_pass"]
    finally:
        native.close()
        guard.close()


def test_exact_rational_power_substitution_is_bounded(tmp_path):
    guard = GuardedSymbolicChecks(tmp_path)
    expression = "Cycle_Time * (a1 + a2 * (1 / GR_phase + flow_lane / (a3 ** a4)) / GR_phase)"
    parameters = {"S_L": {"a1": .10394857392015937, "a2": .5239485739201593,
                           "a3": .9735939283745983, "a4": .4375832983749837}}
    try:
        guard.call("prefit", expression, time.monotonic() + 25)
        started = time.monotonic()
        try:
            guard.call("physical", expression, time.monotonic() + 3, timeout=.3,
                       parameters=parameters, lanes=["S_L"], features=FEATURES)
        except SymbolicCheckError:
            pass
        assert time.monotonic() - started < 3
        assert guard.call("prefit", EXPRESSION, time.monotonic() + 25)["canonical"]
    finally:
        guard.close()
