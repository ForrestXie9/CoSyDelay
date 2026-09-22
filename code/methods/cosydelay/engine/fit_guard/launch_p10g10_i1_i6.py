"""Freeze, run, and integrity-check six bounded V18 Training searches."""

from __future__ import annotations

import argparse
import copy
from importlib import metadata
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from methods.cosydelay.engine.retry_protocol import (
    launch_p10g10_i1_i6 as base,
)

from .contract import V18_CONTRACT, validate_v18_contract
from .fit_timeout import wait_subprocess_bounded
from .policy import V18_POLICY
from .run_p10g10_training import STATUS, validate_fit_guard_audit
from .source_manifest import (
    GMINI,
    V18_SOURCE_FILES,
    validate_v18_source_manifest,
)


PROTOCOL_STATUS = "frozen_before_any_v18_formal_api_call"
_BASE_VALIDATE_FORMAL_RESULT = base.validate_formal_result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base.base.DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _source_hashes() -> dict[str, str]:
    return {
        (
            str(path.relative_to(GMINI)).replace("\\", "/")
            if path.is_relative_to(GMINI)
            else str(path)
        ): base.base.sha256_file(path)
        for path in V18_SOURCE_FILES
    }


def validate_formal_result(
    result: dict[str, Any], protocol: dict[str, Any], *, intersection: int
) -> None:
    """Reuse V17 audits, then prove bounded-fit transactional invariants."""
    normalized = copy.deepcopy(result)
    normalized["v17_source_sha256"] = result.get("v18_source_sha256")
    _BASE_VALIDATE_FORMAL_RESULT(
        normalized, protocol, intersection=intersection
    )
    required = {
        "status": STATUS,
        "method_id": V18_POLICY.method_id,
        "policy": protocol["policy"],
        "method_contract": protocol["method_contract"],
        "v18_source_sha256": protocol["source_sha256"],
        "candidate_fit_timeout_used_for_fitness": False,
        "timed_out_or_failed_candidates_counted": False,
        "candidate_fit_timeout_contains_accuracy_metrics": False,
        "candidate_fit_timeout_contains_validation_or_test_data": False,
    }
    mismatches = {
        key: {"observed": result.get(key), "required": expected}
        for key, expected in required.items()
        if result.get(key) != expected
    }
    audit = result.get("fit_guard_audit")
    if not isinstance(audit, dict):
        mismatches["fit_guard_audit"] = {
            "observed": type(audit).__name__,
            "required": "dict",
        }
    else:
        recomputed = validate_fit_guard_audit(audit)
        if result.get("fit_guard_summary") != recomputed:
            mismatches["fit_guard_summary"] = {
                "observed": result.get("fit_guard_summary"),
                "required": recomputed,
            }
    if mismatches:
        raise RuntimeError(
            f"I{intersection} failed V18 bounded-result integrity checks: "
            f"{mismatches}"
        )


def _run_one(
    *, intersection: int, data_dir: Path, output_root: Path, env: dict[str, str]
) -> dict[str, Any]:
    parent = output_root / f"intersection_{intersection:02d}"
    parent.mkdir(parents=True, exist_ok=True)
    output = parent / "run_01"
    stdout_path = parent / "run_01.stdout.log"
    stderr_path = parent / "run_01.stderr.log"
    started_utc = base.base._now()
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "methods.cosydelay.engine.fit_guard.run_p10g10_training",
                "--intersection",
                str(intersection),
                "--data-dir",
                str(data_dir),
                "--output",
                str(output),
            ],
            cwd=GMINI,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        return_code, timed_out, cleanup = wait_subprocess_bounded(
            process,
            timeout_seconds=V18_CONTRACT.intersection_process_wall_timeout_seconds,
        )
    return {
        "intersection_id": intersection,
        "started_utc": started_utc,
        "completed_utc": base.base._now(),
        "return_code": int(return_code),
        "wall_seconds": time.perf_counter() - started,
        "process_wall_timeout_seconds": (
            V18_CONTRACT.intersection_process_wall_timeout_seconds
        ),
        "process_timed_out": bool(timed_out),
        "process_tree_cleanup": cleanup,
        "result_exists": (output / "result.json").is_file(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> int:
    validate_v18_contract()
    validate_v18_source_manifest()
    v16_launcher = base.base
    previous = (
        base.V17_CONTRACT,
        base.V17_POLICY,
        base.validate_v17_contract,
        base.validate_v17_source_manifest,
        base.V17_SOURCE_FILES,
        base._run_one,
        base.validate_formal_result,
        base.PROTOCOL_STATUS,
        base.STATUS,
        base.arguments,
        v16_launcher._package_versions,
        v16_launcher._write,
    )
    original_package_versions = v16_launcher._package_versions
    original_write = v16_launcher._write

    def package_versions_with_process_guard() -> dict[str, str]:
        versions = dict(original_package_versions())
        versions["psutil"] = metadata.version("psutil")
        return versions

    def write_with_bounded_execution(path: Path, value: Any) -> None:
        if path.name == "FROZEN_PROTOCOL.json" and isinstance(value, dict):
            value = dict(value)
            value["bounded_execution"] = {
                "candidate_fit_wall_timeout_seconds": (
                    V18_CONTRACT.candidate_fit_wall_timeout_seconds
                ),
                "candidate_fit_startup_timeout_seconds": (
                    V18_CONTRACT.candidate_fit_startup_timeout_seconds
                ),
                "candidate_fit_close_timeout_seconds": (
                    V18_CONTRACT.candidate_fit_close_timeout_seconds
                ),
                "intersection_process_wall_timeout_seconds": (
                    V18_CONTRACT.intersection_process_wall_timeout_seconds
                ),
                "process_tree_cleanup": "psutil_terminate_then_kill",
                "timeout_candidate_counted": False,
                "optimizer_rng_commit": "successful_fit_only",
            }
            value["prefit_r7"] = {
                "enabled": V18_CONTRACT.prefit_r7_gate_enabled,
                "coefficient_assumption": (
                    V18_CONTRACT.prefit_r7_coefficient_assumption
                ),
                "reject_verdict": V18_CONTRACT.prefit_r7_reject_verdict,
                "unknown_action": V18_CONTRACT.prefit_r7_unknown_action,
                "uses_training_values": (
                    V18_CONTRACT.prefit_r7_uses_training_values
                ),
                "uses_targets": V18_CONTRACT.prefit_r7_uses_targets,
                "uses_validation_or_test": (
                    V18_CONTRACT.prefit_r7_uses_validation_or_test
                ),
                "changes_fitness_definition": (
                    V18_CONTRACT.prefit_r7_changes_fitness_definition
                ),
            }
        original_write(path, value)

    try:
        base.V17_CONTRACT = V18_CONTRACT
        base.V17_POLICY = V18_POLICY
        base.validate_v17_contract = validate_v18_contract
        base.validate_v17_source_manifest = validate_v18_source_manifest
        base.V17_SOURCE_FILES = V18_SOURCE_FILES
        base._run_one = _run_one
        base.validate_formal_result = validate_formal_result
        base.PROTOCOL_STATUS = PROTOCOL_STATUS
        base.STATUS = STATUS
        base.arguments = arguments
        v16_launcher._package_versions = package_versions_with_process_guard
        v16_launcher._write = write_with_bounded_execution
        return int(base.main())
    finally:
        (
            base.V17_CONTRACT,
            base.V17_POLICY,
            base.validate_v17_contract,
            base.validate_v17_source_manifest,
            base.V17_SOURCE_FILES,
            base._run_one,
            base.validate_formal_result,
            base.PROTOCOL_STATUS,
            base.STATUS,
            base.arguments,
            v16_launcher._package_versions,
            v16_launcher._write,
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
