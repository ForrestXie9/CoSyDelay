"""Freeze, run, and integrity-check six V17 Training-only searches."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

from methods.cosydelay_v16_manuscript_principlewise import (
    launch_p10g10_i1_i6 as base,
)

from .contract import V17_CONTRACT, validate_v17_contract
from .policy import V17_POLICY
from .run_p10g10_training import STATUS, validate_retry_audit
from .source_manifest import (
    GMINI,
    V17_SOURCE_FILES,
    validate_v17_source_manifest,
)


PROTOCOL_STATUS = "frozen_before_any_v17_formal_api_call"
_BASE_VALIDATE_FORMAL_RESULT = base.validate_formal_result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=base.DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _source_hashes() -> dict[str, str]:
    return {
        (
            str(path.relative_to(GMINI)).replace("\\", "/")
            if path.is_relative_to(GMINI)
            else str(path)
        ): base.sha256_file(path)
        for path in V17_SOURCE_FILES
    }


def validate_formal_result(
    result: dict[str, Any], protocol: dict[str, Any], *, intersection: int
) -> None:
    """Reuse every V16 audit, then enforce the isolated V17 additions."""
    normalized = copy.deepcopy(result)
    normalized["status"] = "completed_v16_p10g10_training_only_search"
    normalized["method_id"] = base.V16_POLICY.method_id
    normalized["v16_source_sha256"] = result.get("v17_source_sha256")
    _BASE_VALIDATE_FORMAL_RESULT(
        normalized, protocol, intersection=intersection
    )
    required = {
        "status": STATUS,
        "method_id": V17_POLICY.method_id,
        "policy": protocol["policy"],
        "method_contract": protocol["method_contract"],
        "v17_source_sha256": protocol["source_sha256"],
        "cross_batch_retry_used_for_selection": False,
        "cross_batch_retry_contains_accuracy_metrics": False,
        "cross_batch_retry_contains_validation_or_test_data": False,
    }
    mismatches = {
        key: {"observed": result.get(key), "required": expected}
        for key, expected in required.items()
        if result.get(key) != expected
    }
    retry_records = result.get("cross_batch_retry_audit")
    if not isinstance(retry_records, list):
        mismatches["cross_batch_retry_audit"] = {
            "observed": type(retry_records).__name__,
            "required": "list",
        }
    else:
        recomputed = validate_retry_audit(retry_records)
        if recomputed["trailing_failure_pending"]:
            mismatches["cross_batch_retry_trailing_failure"] = {
                "observed": True,
                "required": False,
            }
        if result.get("cross_batch_retry_summary") != recomputed:
            mismatches["cross_batch_retry_summary"] = {
                "observed": result.get("cross_batch_retry_summary"),
                "required": recomputed,
            }
    prompt_audit = result.get("prompt_audit", {})
    required_prompt_flags = {
        "at_most_one_cross_batch_correction_per_prompt": True,
        "cross_batch_correction_has_sequence": True,
        "cross_batch_correction_has_last_reason": True,
        "cross_batch_correction_has_no_accuracy_or_split_supervision": True,
    }
    for key, expected in required_prompt_flags.items():
        if prompt_audit.get(key) != expected:
            mismatches[f"prompt_audit.{key}"] = {
                "observed": prompt_audit.get(key),
                "required": expected,
            }
    if mismatches:
        raise RuntimeError(
            f"I{intersection} failed V17 frozen-result integrity checks: "
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
    started_utc = base._now()
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "methods.cosydelay_v17_cross_batch_retry.run_p10g10_training",
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
            check=False,
            env=env,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    return {
        "intersection_id": intersection,
        "started_utc": started_utc,
        "completed_utc": base._now(),
        "return_code": int(completed.returncode),
        "wall_seconds": time.perf_counter() - started,
        "result_exists": (output / "result.json").is_file(),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def main() -> int:
    validate_v17_contract()
    validate_v17_source_manifest()
    previous = (
        base.V16_CONTRACT,
        base.V16_POLICY,
        base.validate_v16_contract,
        base.validate_v16_source_manifest,
        base.V16_SOURCE_FILES,
        base._run_one,
        base.validate_formal_result,
        base._write,
        base.arguments,
    )
    original_write = base._write

    def write_with_v17_protocol(path: Path, value: Any) -> None:
        if path.name == "FROZEN_PROTOCOL.json" and isinstance(value, dict):
            value = dict(value)
            if value.get("status") == "frozen_before_any_v16_formal_api_call":
                value["status"] = PROTOCOL_STATUS
        original_write(path, value)

    try:
        base.V16_CONTRACT = V17_CONTRACT
        base.V16_POLICY = V17_POLICY
        base.validate_v16_contract = validate_v17_contract
        base.validate_v16_source_manifest = validate_v17_source_manifest
        base.V16_SOURCE_FILES = V17_SOURCE_FILES
        base._run_one = _run_one
        base.validate_formal_result = validate_formal_result
        base._write = write_with_v17_protocol
        base.arguments = arguments
        return int(base.main())
    finally:
        (
            base.V16_CONTRACT,
            base.V16_POLICY,
            base.validate_v16_contract,
            base.validate_v16_source_manifest,
            base.V16_SOURCE_FILES,
            base._run_one,
            base.validate_formal_result,
            base._write,
            base.arguments,
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
