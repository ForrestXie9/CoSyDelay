"""Freeze and sequentially run I1--I6 prompt-v6 efficiency-v1 P10/G10."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from llm_config import load_llm_config
from methods.cosydelay.engine.data_protocol.operations.run04_pythonw_supervisor import (
    recover_api_key,
)
from methods.cosydelay.engine.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    jsonable,
    sha256_file,
)

from .policy import V10_POLICY
from .source_manifest import SOURCE_FILES, validate_source_manifest


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[1]
INTERSECTIONS = tuple(range(1, 7))
RUNNER_MODULE = (
    "methods.cosydelay.engine.candidate_engine."
    "run_formal_training_search"
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def preflight(args: argparse.Namespace) -> tuple[object, dict[str, Path], dict]:
    if args.output_root.exists():
        raise FileExistsError(
            f"refusing to overwrite or resume formal batch: {args.output_root}"
        )
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
    if observed != V10_POLICY.llm_sampling:
        raise RuntimeError(
            f"LLM configuration is not frozen: {observed} != "
            f"{V10_POLICY.llm_sampling}"
        )
    source_sha = {
        str(path.relative_to(GMINI)): sha256_file(path) for path in SOURCE_FILES
    }
    return llm, training_paths, source_sha


def validate_result(
    result: dict,
    *,
    intersection: int,
    source_sha: dict,
    train_sha: str,
) -> None:
    required = {
        "formal_protocol_complete": True,
        "method_id": V10_POLICY.method_id,
        "policy": V10_POLICY.to_dict(),
        "intersection_id": intersection,
        "source_sha256": source_sha,
        "train_file_sha256": train_sha,
        "population": V10_POLICY.population,
        "generations": V10_POLICY.generations,
        "completed_generations": V10_POLICY.generations,
        "optimizer_restarts": V10_POLICY.optimizer_restarts,
        "prompt_contract_version": V10_POLICY.prompt_contract_version,
        "execution_contract_version": V10_POLICY.execution_contract_version,
        "selection_and_evaluation_scope": "Training only",
        "accessed_splits": ["train"],
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "post_evolution_cv_reranking": False,
        "post_evolution_refit": False,
        "initial_population_all_generated_in_current_run": True,
        "initial_population_size_verified": V10_POLICY.population,
    }
    mismatches = {
        key: {"observed": result.get(key), "required": value}
        for key, value in required.items()
        if result.get(key) != value
    }
    if result.get("selected_enhanced_physics", {}).get("joint_pass") is not True:
        mismatches["selected_enhanced_physics.joint_pass"] = {
            "observed": result.get("selected_enhanced_physics", {}).get(
                "joint_pass"
            ),
            "required": True,
        }
    if mismatches:
        raise RuntimeError(
            f"I{intersection} failed frozen integrity checks: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def run_one(intersection: int, args: argparse.Namespace) -> dict:
    parent = args.output_root / f"intersection_{intersection:02d}"
    parent.mkdir(parents=True, exist_ok=False)
    output = parent / "run_01"
    stdout_path = parent / "run_01.stdout.log"
    stderr_path = parent / "run_01.stderr.log"
    command = [
        sys.executable,
        "-m",
        RUNNER_MODULE,
        "--intersection",
        str(intersection),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--output",
        str(output.resolve()),
    ]
    started = time.perf_counter()
    started_utc = utc_now()
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
        "run_output": str(output.resolve()),
        "stdout": str(stdout_path.resolve()),
        "stderr": str(stderr_path.resolve()),
        "result_exists": (output / "result.json").is_file(),
    }


def main() -> int:
    args = arguments()
    environment_names = (
        "LLM_API_KEY",
        "LLM_TEMPERATURE",
        "LLM_TOP_P",
        "LLM_MAX_TOKENS",
    )
    previous = {name: os.environ.get(name) for name in environment_names}
    try:
        os.environ["LLM_API_KEY"] = recover_api_key()
        for name in environment_names[1:]:
            os.environ.pop(name, None)
        llm, training_paths, source_sha = preflight(args)
        if args.preflight_only:
            print(
                json.dumps(
                    {
                        "status": "preflight_passed",
                        "policy": V10_POLICY.to_dict(),
                        "source_files": len(source_sha),
                        "training_files": len(training_paths),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return 0

        args.output_root.mkdir(parents=True, exist_ok=False)
        protocol = {
            "schema_version": 1,
            "status": "frozen_before_any_v10_formal_api_call",
            "frozen_utc": utc_now(),
            "method_id": V10_POLICY.method_id,
            "policy": V10_POLICY.to_dict(),
            "intersections": list(INTERSECTIONS),
            "runs_per_intersection": 1,
            "max_parallel_intersections": 1,
            "selection_split": "Training only",
            "validation_allowed": False,
            "existing_test_allowed": False,
            "llm": {
                "provider_endpoint": llm.endpoint,
                "request_path": llm.request_path,
                "sampling": V10_POLICY.llm_sampling,
                "sampling_parameter_sources": llm.sampling_parameter_sources,
                "api_key_recorded": False,
            },
            "python": sys.version,
            "packages": package_versions(),
            "source_sha256": source_sha,
            "training_sha256": {
                key: {"file_name": path.name, "sha256": sha256_file(path)}
                for key, path in training_paths.items()
            },
            "test_sha256": None,
        }
        write_json(args.output_root / "FROZEN_PROTOCOL.json", protocol)

        statuses = []
        results = []
        for intersection in INTERSECTIONS:
            status = run_one(intersection, args)
            statuses.append(status)
            write_json(args.output_root / "RUN_STATUS.json", statuses)
            print(
                f"I{intersection} return={status['return_code']} "
                f"wall={status['wall_seconds']:.1f}s",
                flush=True,
            )
            if status["return_code"] != 0 or not status["result_exists"]:
                write_json(
                    args.output_root / "BATCH_COMPLETE.json",
                    {
                        "status": "failed",
                        "completed_utc": utc_now(),
                        "failed_intersection": intersection,
                    },
                )
                return 1
            result = json.loads(
                (
                    args.output_root
                    / f"intersection_{intersection:02d}"
                    / "run_01"
                    / "result.json"
                ).read_text(encoding="utf-8")
            )
            validate_result(
                result,
                intersection=intersection,
                source_sha=source_sha,
                train_sha=protocol["training_sha256"][str(intersection)][
                    "sha256"
                ],
            )
            results.append(result)

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
            "mean_wall_seconds": float(
                np.mean([float(item["wall_seconds"]) for item in results])
            ),
            "all_selected_physics_pass": all(
                item["selected_enhanced_physics"]["joint_pass"]
                for item in results
            ),
        }
        write_json(args.output_root / "AGGREGATE.json", aggregate)
        write_json(
            args.output_root / "BATCH_COMPLETE.json",
            {"status": "completed", "completed_utc": utc_now()},
        )
        return 0
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


if __name__ == "__main__":
    raise SystemExit(main())
