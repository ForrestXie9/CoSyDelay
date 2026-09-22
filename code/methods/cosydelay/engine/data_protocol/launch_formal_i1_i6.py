"""Freeze and launch the six fixed clean formal Training searches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from llm_config import load_llm_config  # noqa: E402
from methods.cosydelay.engine.data_protocol.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay.engine.data_protocol.source_manifest import (  # noqa: E402
    SOURCE_FILES,
    validate_source_manifest,
)
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (  # noqa: E402
    DEFAULT_DATA_DIR,
    jsonable,
)


INTERSECTIONS = tuple(range(1, 7))
MAX_PARALLEL_INTERSECTIONS = 2
RUNNER = HERE / "run_formal_training_search.py"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def package_versions() -> dict[str, str]:
    names = ("numpy", "pandas", "scipy", "sympy", "scikit-learn", "numexpr")
    return {name: metadata.version(name) for name in names}


def preflight(args: argparse.Namespace) -> tuple[object, dict[str, Path]]:
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite or resume formal batch: {args.output_root}"
        )
    if not RUNNER.is_file():
        raise FileNotFoundError(RUNNER)
    validate_source_manifest()
    training_paths = {
        str(intersection): (
            args.data_dir / f"Intersection_{intersection}_Train.jsonl"
        ).resolve()
        for intersection in INTERSECTIONS
    }
    for path in training_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    llm = load_llm_config()
    observed = {
        "model": llm.model,
        "temperature": llm.temperature,
        "top_p": llm.top_p,
        "max_tokens": llm.max_tokens,
        "seed": llm.seed,
    }
    if observed != CLEAN_POLICY.llm_sampling:
        raise RuntimeError(
            f"LLM configuration is not frozen: {observed} != "
            f"{CLEAN_POLICY.llm_sampling}"
        )
    return llm, training_paths


def run_one(
    *, intersection: int, data_dir: Path, output_root: Path
) -> dict:
    parent = output_root / f"intersection_{intersection:02d}"
    parent.mkdir(parents=True, exist_ok=True)
    run_output = parent / "run_01"
    stdout_path = parent / "run_01.stdout.log"
    stderr_path = parent / "run_01.stderr.log"
    command = [
        sys.executable,
        str(RUNNER),
        "--intersection",
        str(intersection),
        "--data-dir",
        str(data_dir),
        "--output",
        str(run_output),
    ]
    started_utc = utc_now()
    started = time.perf_counter()
    with stdout_path.open("w", encoding="utf-8") as stdout_handle, stderr_path.open(
        "w", encoding="utf-8"
    ) as stderr_handle:
        completed = subprocess.run(
            command,
            cwd=str(GMINI),
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            check=False,
            env=os.environ.copy(),
            creationflags=(
                getattr(subprocess, "CREATE_NO_WINDOW", 0)
                if os.name == "nt"
                else 0
            ),
        )
    return {
        "intersection_id": intersection,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "return_code": int(completed.returncode),
        "wall_seconds": time.perf_counter() - started,
        "run_output": str(run_output.resolve()),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "result_exists": (run_output / "result.json").is_file(),
    }


def validate_formal_result(
    result: dict, protocol: dict, *, intersection: int
) -> None:
    """Fail closed if a child result differs from the pre-call freeze."""
    required = {
        "formal_protocol_complete": True,
        "method_id": protocol["method_id"],
        "intersection_id": int(intersection),
        "policy": protocol["policy"],
        "source_sha256": protocol["source_sha256"],
        "train_file_sha256": protocol["training_sha256"][str(intersection)][
            "sha256"
        ],
        "selection_source": "full_training_evolution_only",
        "accessed_splits": ["train"],
        "project_artifact_read_policy": "allowlist_only",
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "post_evolution_cv_reranking": False,
        "post_evolution_refit": False,
        "incumbent_argument_supported": False,
        "incumbent_injected_into_initial_population": False,
        "initial_population_all_generated_in_current_run": True,
        "initial_population_size_verified": CLEAN_POLICY.population,
        "completed_generations": CLEAN_POLICY.generations,
        "legacy_validation_labels_removed_from_formal_history": True,
    }
    mismatches = {
        key: {"observed": result.get(key), "required": expected}
        for key, expected in required.items()
        if result.get(key) != expected
    }
    physics = result.get("selected_enhanced_physics_from_evolution", {})
    if physics.get("joint_pass") is not True:
        mismatches["selected_enhanced_physics_from_evolution.joint_pass"] = {
            "observed": physics.get("joint_pass"),
            "required": True,
        }
    if mismatches:
        raise RuntimeError(
            f"I{intersection} failed frozen-result integrity checks: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def main() -> int:
    args = arguments()
    llm, training_paths = preflight(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    protocol = {
        "schema_version": 1,
        "status": "frozen_before_any_v9_formal_api_call",
        "frozen_utc": utc_now(),
        "method_id": CLEAN_POLICY.method_id,
        "policy": CLEAN_POLICY.to_dict(),
        "intersections": list(INTERSECTIONS),
        "runs_per_intersection": 1,
        "max_parallel_intersections": MAX_PARALLEL_INTERSECTIONS,
        "selection_split": "Training only",
        "validation_allowed": False,
        "existing_test_allowed": False,
        "project_wide_existing_test_previously_seen": True,
        "fresh_unseen_test_required": True,
        "llm": {
            "provider_endpoint": llm.endpoint,
            "request_path": llm.request_path,
            "sampling": CLEAN_POLICY.llm_sampling,
            "sampling_parameter_sources": llm.sampling_parameter_sources,
            "original_provider_defaults_preserved": {
                "temperature_sent": llm.temperature is not None,
                "top_p_sent": llm.top_p is not None,
                "max_tokens_sent": llm.max_tokens is not None,
            },
            "api_key_recorded": False,
            "response_snapshot_to_be_read_from_call_audits": True,
        },
        "python": sys.version,
        "packages": package_versions(),
        "source_sha256": {
            str(path.relative_to(GMINI)): sha256_file(path)
            for path in SOURCE_FILES
        },
        "training_sha256": {
            key: {
                "file_name": path.name,
                "sha256": sha256_file(path),
            }
            for key, path in training_paths.items()
        },
        "test_sha256": None,
    }
    write_json(args.output_root / "FROZEN_PROTOCOL.json", protocol)

    statuses = []
    with ThreadPoolExecutor(
        max_workers=MAX_PARALLEL_INTERSECTIONS
    ) as executor:
        futures = {
            executor.submit(
                run_one,
                intersection=intersection,
                data_dir=args.data_dir.resolve(),
                output_root=args.output_root.resolve(),
            ): intersection
            for intersection in INTERSECTIONS
        }
        for future in as_completed(futures):
            status = future.result()
            statuses.append(status)
            statuses.sort(key=lambda item: item["intersection_id"])
            write_json(args.output_root / "RUN_STATUS.json", statuses)
            print(
                f"I{status['intersection_id']} return={status['return_code']} "
                f"wall={status['wall_seconds']:.1f}s"
            )

    failed = [
        item
        for item in statuses
        if item["return_code"] != 0 or not item["result_exists"]
    ]
    if failed:
        write_json(
            args.output_root / "BATCH_COMPLETE.json",
            {
                "status": "failed",
                "completed_utc": utc_now(),
                "failed_intersections": [
                    item["intersection_id"] for item in failed
                ],
            },
        )
        return 1

    results = [
        json.loads(
            (
                args.output_root
                / f"intersection_{intersection:02d}"
                / "run_01"
                / "result.json"
            ).read_text(encoding="utf-8")
        )
        for intersection in INTERSECTIONS
    ]
    integrity_errors = []
    for intersection, result in zip(INTERSECTIONS, results):
        try:
            validate_formal_result(
                result, protocol, intersection=intersection
            )
        except Exception as exc:
            integrity_errors.append(
                {
                    "intersection_id": intersection,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    if integrity_errors:
        write_json(
            args.output_root / "BATCH_COMPLETE.json",
            {
                "status": "failed_integrity",
                "completed_utc": utc_now(),
                "integrity_errors": integrity_errors,
            },
        )
        return 1
    aggregate = {
        "schema_version": 1,
        "scope": "full Training search metrics; not external generalization",
        "intersections": list(INTERSECTIONS),
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
    }
    write_json(args.output_root / "TRAINING_AGGREGATE.json", aggregate)
    write_json(
        args.output_root / "BATCH_COMPLETE.json",
        {
            "status": "complete",
            "completed_utc": utc_now(),
            "formal_training_runs": 6,
            "fresh_unseen_test_still_required": True,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
