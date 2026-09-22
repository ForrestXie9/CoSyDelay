"""Operationally finish run03 after its coordinator lost its console host.

I1 and I2 continue in their original from-scratch runner processes. This
supervisor waits for those exact processes; it never restarts or resumes an
incomplete intersection. Only intersections that have never started are
launched, in fresh pairs, after the prior pair has completed successfully.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np


GMINI = Path(
    r"Z:\CoSydelay\Paper_SM_traffic_delay\Parameters_sensitive\gmini"
)
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from methods.cosydelay._internal.data_protocol import (  # noqa: E402
    launch_formal_i1_i6 as launcher,
)
from methods.cosydelay._internal.data_protocol.policy import (  # noqa: E402
    CLEAN_POLICY,
)
from methods.cosydelay._internal.data_protocol.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
)


ROOT = GMINI / (
    "methods/cosydelay_v9_clean_from_scratch/experiments/"
    "formal_v9_paper_fitness_promptv3_providerdefaults_"
    "p10g10_i1_i6_run03_20260810"
)
PROTOCOL_PATH = ROOT / "FROZEN_PROTOCOL.json"
STATUS_PATH = ROOT / "RECOVERY_SUPERVISOR_STATUS.json"
ERROR_PATH = ROOT.parent / f"{ROOT.name}.recovery.error.log"
HISTORY_FILE = Path(
    r"Z:\CoSydelay\Paper_SM_traffic_delay\Baseline_methods\paper_ml_reproduction"
    r"\local_history_recovery\pycharm2025_1_vfs_resolved"
    r"\20260710_153720_150_record_0000000352_change_0000_jiekou_llm_embedded.py"
)
EXISTING_PROCESS_IDS = {1: 59512, 2: 17508}


def write_status(value: dict[str, Any]) -> None:
    launcher.write_json(STATUS_PATH, value)


def wait_for_process(process_id: int) -> None:
    if os.name != "nt":
        raise RuntimeError("run03 recovery is Windows-specific")
    synchronize = 0x00100000
    infinite = 0xFFFFFFFF
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = (
        ctypes.c_uint32,
        ctypes.c_int,
        ctypes.c_uint32,
    )
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.WaitForSingleObject.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
    kernel32.WaitForSingleObject.restype = ctypes.c_uint32
    kernel32.CloseHandle.argtypes = (ctypes.c_void_p,)
    kernel32.CloseHandle.restype = ctypes.c_int
    handle = kernel32.OpenProcess(synchronize, False, process_id)
    if not handle:
        return
    try:
        outcome = kernel32.WaitForSingleObject(handle, infinite)
        if outcome != 0:
            raise OSError(
                ctypes.get_last_error(),
                f"WaitForSingleObject failed for PID {process_id}",
            )
    finally:
        kernel32.CloseHandle(handle)


def result_path(intersection: int) -> Path:
    return ROOT / f"intersection_{intersection:02d}" / "run_01" / "result.json"


def read_result(intersection: int) -> dict[str, Any]:
    path = result_path(intersection)
    if not path.is_file():
        raise RuntimeError(
            f"I{intersection} process exited without a complete result.json"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def validate_frozen_inputs(protocol: dict[str, Any]) -> None:
    mismatches = []
    for relative, expected in protocol["source_sha256"].items():
        path = GMINI / relative
        observed = launcher.sha256_file(path) if path.is_file() else None
        if observed != expected:
            mismatches.append(str(relative))
    for intersection in range(1, 7):
        expected = protocol["training_sha256"][str(intersection)]["sha256"]
        path = DEFAULT_DATA_DIR / f"Intersection_{intersection}_Train.jsonl"
        observed = launcher.sha256_file(path) if path.is_file() else None
        if observed != expected:
            mismatches.append(str(path))
    if mismatches:
        raise RuntimeError(
            "frozen source/data hashes changed: " + ", ".join(mismatches)
        )


def recover_api_key() -> str:
    text = HISTORY_FILE.read_text(encoding="utf-8")
    match = re.search(
        r"(?m)^JIEKOU_API_KEY\s*=\s*['\"](sk-[A-Za-z0-9_-]{20,})['\"]\s*$",
        text,
    )
    if not match:
        raise RuntimeError("API credential could not be recovered")
    return match.group(1)


def completed_status(intersection: int, result: dict[str, Any]) -> dict[str, Any]:
    parent = ROOT / f"intersection_{intersection:02d}"
    return {
        "intersection_id": intersection,
        "started_utc": result.get("started_utc"),
        "completed_utc": result.get("completed_utc"),
        "return_code": 0,
        "wall_seconds": result.get("wall_seconds"),
        "run_output": str((parent / "run_01").resolve()),
        "stdout": str((parent / "run_01.stdout.log").resolve()),
        "stderr": str((parent / "run_01.stderr.log").resolve()),
        "result_exists": True,
        "coordination_recovery": "original runner completed without resume",
    }


def run_pair(pair: tuple[int, int]) -> list[dict[str, Any]]:
    statuses = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                launcher.run_one,
                intersection=intersection,
                data_dir=DEFAULT_DATA_DIR.resolve(),
                output_root=ROOT.resolve(),
            ): intersection
            for intersection in pair
        }
        for future in as_completed(futures):
            statuses.append(future.result())
    statuses.sort(key=lambda item: item["intersection_id"])
    failed = [
        item
        for item in statuses
        if item["return_code"] != 0 or not item["result_exists"]
    ]
    if failed:
        raise RuntimeError(
            "fresh pair failed: "
            + json.dumps(failed, ensure_ascii=False, sort_keys=True)
        )
    return statuses


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "scope": "full Training search metrics; not external generalization",
        "intersections": list(range(1, 7)),
        "mean_raw_macro_r2": float(
            np.mean(
                [
                    item["formal_comparison_training_metrics"][
                        "raw_approach_macro_r2"
                    ]
                    for item in results
                ]
            )
        ),
        "mean_pooled_rmse": float(
            np.mean(
                [
                    item["formal_comparison_training_metrics"]["pooled_rmse"]
                    for item in results
                ]
            )
        ),
        "mean_pooled_mae": float(
            np.mean(
                [
                    item["formal_comparison_training_metrics"]["pooled_mae"]
                    for item in results
                ]
            )
        ),
        "all_selected_physics_pass": all(
            item["selected_enhanced_physics_from_evolution"]["joint_pass"]
            for item in results
        ),
        "mean_invalid_expression_ratio": float(
            np.mean(
                [
                    item["expression_attempt_summary"]["invalid_output_ratio"]
                    for item in results
                ]
            )
        ),
        "mean_prefit_pass_rate": float(
            np.mean([item["prefit_gate_pass_rate"] for item in results])
        ),
        "mean_postfit_pass_rate": float(
            np.mean([item["postfit_pass_rate"] for item in results])
        ),
        "total_api_attempts": int(
            sum(item["llm_attempt_summary"]["api_attempts"] for item in results)
        ),
        "total_wall_seconds_sum": float(
            sum(item["wall_seconds"] for item in results)
        ),
        "response_models": sorted(
            {
                model
                for item in results
                for model in item["llm_attempt_summary"]["response_models"]
            }
        ),
        "coordination_recovery": {
            "reason": "original launcher lost Windows console host",
            "incomplete_intersection_resumed": False,
            "i1_i2_original_processes_waited": True,
            "i3_i6_fresh_from_scratch": True,
        },
    }


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    validate_frozen_inputs(protocol)
    write_status(
        {
            "status": "waiting_for_original_i1_i2",
            "updated_utc": launcher.utc_now(),
            "existing_process_ids": EXISTING_PROCESS_IDS,
            "incomplete_intersection_resumed": False,
        }
    )

    for process_id in EXISTING_PROCESS_IDS.values():
        wait_for_process(process_id)
    results_by_id = {
        intersection: read_result(intersection) for intersection in (1, 2)
    }
    for intersection, result in results_by_id.items():
        launcher.validate_formal_result(
            result, protocol, intersection=intersection
        )
    statuses = [
        completed_status(intersection, results_by_id[intersection])
        for intersection in (1, 2)
    ]
    launcher.write_json(ROOT / "RUN_STATUS.json", statuses)

    os.environ["LLM_API_KEY"] = recover_api_key()
    for name in ("LLM_TEMPERATURE", "LLM_TOP_P", "LLM_MAX_TOKENS"):
        os.environ.pop(name, None)
    try:
        for pair in ((3, 4), (5, 6)):
            validate_frozen_inputs(protocol)
            write_status(
                {
                    "status": "running_fresh_pair",
                    "updated_utc": launcher.utc_now(),
                    "pair": list(pair),
                    "incomplete_intersection_resumed": False,
                }
            )
            pair_statuses = run_pair(pair)
            statuses.extend(pair_statuses)
            statuses.sort(key=lambda item: item["intersection_id"])
            launcher.write_json(ROOT / "RUN_STATUS.json", statuses)
            for intersection in pair:
                result = read_result(intersection)
                launcher.validate_formal_result(
                    result, protocol, intersection=intersection
                )
                results_by_id[intersection] = result
    finally:
        os.environ.pop("LLM_API_KEY", None)

    results = [results_by_id[index] for index in range(1, 7)]
    launcher.write_json(ROOT / "TRAINING_AGGREGATE.json", aggregate(results))
    launcher.write_json(
        ROOT / "BATCH_COMPLETE.json",
        {
            "status": "complete",
            "completed_utc": launcher.utc_now(),
            "formal_training_runs": 6,
            "fresh_unseen_test_still_required": True,
            "coordination_recovery_used": True,
            "incomplete_intersection_resumed": False,
        },
    )
    write_status(
        {
            "status": "complete",
            "updated_utc": launcher.utc_now(),
            "incomplete_intersection_resumed": False,
        }
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        ERROR_PATH.write_text(
            f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
        )
        try:
            launcher.write_json(
                ROOT / "BATCH_COMPLETE.json",
                {
                    "status": "failed_recovery_supervisor",
                    "completed_utc": launcher.utc_now(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "incomplete_intersection_resumed": False,
                },
            )
        finally:
            raise
    raise SystemExit(exit_code)
