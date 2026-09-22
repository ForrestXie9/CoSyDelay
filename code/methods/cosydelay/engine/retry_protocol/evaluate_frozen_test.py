"""Prediction-only Test evaluation after six V17 searches are frozen."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from methods.cosydelay.engine.training_protocol import (
    evaluate_frozen_test as base,
)

from . import launch_p10g10_i1_i6 as launcher
from .policy import V17_POLICY


METHOD_LABEL = "cosydelay_v17_p10g10"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=base.DEFAULT_DATA_DIR)
    parser.add_argument(
        "--baseline-dir", type=Path, default=base.DEFAULT_BASELINE_DIR
    )
    parser.add_argument("--skip-baseline-ranking", action="store_true")
    return parser.parse_args()


def verify_frozen_search(
    search_root: Path, data_dir: Path
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Verify all Training artifacts before any Test file is opened."""
    protocol_path = search_root / "FROZEN_PROTOCOL.json"
    batch_path = search_root / "BATCH_COMPLETE.json"
    status_path = search_root / "RUN_STATUS.json"
    protocol = base._read_json(protocol_path)
    batch = base._read_json(batch_path)
    status = base._read_json(status_path)
    if protocol.get("status") != launcher.PROTOCOL_STATUS:
        raise RuntimeError("search protocol was not frozen before V17 API calls")
    if protocol.get("test_sha256") is not None:
        raise RuntimeError("search protocol unexpectedly contains a Test hash")
    if protocol.get("existing_test_allowed_during_search") is not False:
        raise RuntimeError("search protocol did not prohibit Test access")
    if batch.get("status") != "complete" or batch.get("formal_training_runs") != 6:
        raise RuntimeError("V17 Training batch is not complete")
    if status.get("status") != "completed" or len(status.get("runs", [])) != 6:
        raise RuntimeError("V17 run status is not complete for all intersections")
    if batch.get("frozen_protocol_sha256") != base.sha256_file(protocol_path):
        raise RuntimeError("frozen protocol hash differs from completion record")

    current_source = launcher._source_hashes()
    if current_source != protocol.get("source_sha256"):
        changed = sorted(
            key
            for key in set(current_source) | set(protocol.get("source_sha256", {}))
            if current_source.get(key) != protocol.get("source_sha256", {}).get(key)
        )
        raise RuntimeError(
            "frozen V17 source changed before Test evaluation: " + ", ".join(changed)
        )
    results = []
    for intersection in base.INTERSECTIONS:
        train_path = data_dir / f"Intersection_{intersection}_Train.jsonl"
        if not train_path.is_file():
            raise FileNotFoundError(train_path)
        expected_train_hash = protocol["training_sha256"][str(intersection)][
            "sha256"
        ]
        if base.sha256_file(train_path) != expected_train_hash:
            raise RuntimeError(f"I{intersection} Training data changed after freeze")
        result_path = (
            search_root
            / f"intersection_{intersection:02d}"
            / "run_01"
            / "result.json"
        )
        if base.sha256_file(result_path) != batch["result_sha256"][str(intersection)]:
            raise RuntimeError(f"I{intersection} result changed after completion")
        result = base._read_json(result_path)
        launcher.validate_formal_result(result, protocol, intersection=intersection)
        results.append(result)
    return protocol, results


def _rename_v16_comparison_fields(output: Path) -> None:
    path = output / "fixed17_comparison.json"
    if not path.is_file():
        return
    payload = base._read_json(path)
    renamed = {}
    for key, value in payload.items():
        renamed[key.replace("v16_", "v17_")] = value
    if isinstance(renamed.get("comparison_scope"), str):
        renamed["comparison_scope"] = renamed["comparison_scope"].replace(
            "V16", "V17"
        )
    base._write_json(path, renamed)


def main() -> int:
    previous = (
        base.launcher,
        base.V16_POLICY,
        base.METHOD_LABEL,
        base.verify_frozen_search,
        base.__file__,
        base.arguments,
    )
    try:
        base.launcher = launcher
        base.V16_POLICY = V17_POLICY
        base.METHOD_LABEL = METHOD_LABEL
        base.verify_frozen_search = verify_frozen_search
        base.__file__ = __file__
        base.arguments = arguments
        code = int(base.main())
        if code == 0:
            args = base.arguments()
            _rename_v16_comparison_fields(args.output.resolve())
        return code
    finally:
        (
            base.launcher,
            base.V16_POLICY,
            base.METHOD_LABEL,
            base.verify_frozen_search,
            base.__file__,
            base.arguments,
        ) = previous


if __name__ == "__main__":
    raise SystemExit(main())
