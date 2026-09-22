"""Run one fixed P10/G10 from-scratch search using only one Training file."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np


HERE = Path(__file__).resolve().parent
GMINI = HERE.parents[2]
if str(GMINI) not in sys.path:
    sys.path.insert(0, str(GMINI))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from llm_config import load_llm_config  # noqa: E402
import optimization_lane  # noqa: E402
from population_evolution_lane import evolve_universal_lane_expression  # noqa: E402
from methods.cosydelay._internal.data_protocol.integration import (  # noqa: E402
    install_clean_single_evolution,
)
from methods.cosydelay._internal.data_protocol.diagnostics import (  # noqa: E402
    audit_parameter_quality,
    summarize_compute_efficiency,
)
from methods.cosydelay._internal.data_protocol.policy import CLEAN_POLICY  # noqa: E402
from methods.cosydelay._internal.data_protocol.source_manifest import (  # noqa: E402
    SOURCE_FILES,
    validate_source_manifest,
)


# The release carries the frozen, training-only splits under ``data``.  Keep
# the default relocatable so the public runner works after cloning elsewhere;
# callers can still override it with ``--data-dir``.
DEFAULT_DATA_DIR = GMINI.parent.parent / "data" / "locked_splits"
DATASET_NAME = re.compile(
    r"^Intersection_(?P<intersection>\d+)_(?P<split>Train|Validation|Test)\.jsonl$",
    re.IGNORECASE,
)
ARTIFACT_SUFFIXES = frozenset(
    {
        ".csv",
        ".feather",
        ".joblib",
        ".json",
        ".jsonl",
        ".npy",
        ".npz",
        ".parquet",
        ".pickle",
        ".pkl",
        ".xls",
        ".xlsx",
    }
)
_ACTIVE_ACCESS_GUARD = None
_ACCESS_AUDIT_HOOK_INSTALLED = False


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_read_open(arguments) -> bool:
    if len(arguments) < 2:
        return True
    mode = arguments[1]
    if isinstance(mode, str):
        return "r" in mode or "+" in mode
    return True


def _dispatch_access_audit(event, arguments) -> None:
    guard = _ACTIVE_ACCESS_GUARD
    if guard is not None:
        guard.audit_event(event, arguments)


class SplitAccessGuard:
    """Allow-list project data/artifact reads during a formal search."""

    def __init__(
        self,
        allowed_training_path: Path,
        *,
        guarded_roots=(),
        allowed_artifact_paths=(),
    ) -> None:
        self.allowed_training_path = allowed_training_path.resolve()
        self.guarded_roots = tuple(Path(path).resolve() for path in guarded_roots)
        self.allowed_artifact_paths = {
            self.allowed_training_path,
            *(Path(path).resolve() for path in allowed_artifact_paths),
        }
        self.records: list[dict] = []
        self.active = True

    def audit_event(self, event, arguments) -> None:
        if not self.active or event != "open" or not arguments:
            return
        raw_path = arguments[0]
        if not isinstance(raw_path, (str, bytes, os.PathLike)):
            return
        if not _is_read_open(arguments):
            return
        path = Path(os.fsdecode(raw_path))
        match = DATASET_NAME.match(path.name)
        resolved = path.resolve()
        project_artifact = (
            path.suffix.lower() in ARTIFACT_SUFFIXES
            and any(_path_is_within(resolved, root) for root in self.guarded_roots)
        )
        if not match and not project_artifact:
            return
        split = match.group("split").lower() if match else "artifact"
        allowed = resolved in self.allowed_artifact_paths
        self.records.append(
            {
                "path": str(resolved),
                "file_name": path.name,
                "split": split,
                "allowed": bool(allowed),
                "read_allowlisted": bool(allowed),
            }
        )
        if not allowed:
            raise PermissionError(
                "clean formal search blocked a non-allowlisted project data or "
                f"artifact read: {resolved}"
            )

    def install(self) -> None:
        global _ACCESS_AUDIT_HOOK_INSTALLED, _ACTIVE_ACCESS_GUARD
        if _ACTIVE_ACCESS_GUARD is not None and _ACTIVE_ACCESS_GUARD.active:
            raise RuntimeError("another formal data-access guard is already active")
        if not _ACCESS_AUDIT_HOOK_INSTALLED:
            sys.addaudithook(_dispatch_access_audit)
            _ACCESS_AUDIT_HOOK_INSTALLED = True
        _ACTIVE_ACCESS_GUARD = self

    def disable(self) -> None:
        global _ACTIVE_ACCESS_GUARD
        self.active = False
        if _ACTIVE_ACCESS_GUARD is self:
            _ACTIVE_ACCESS_GUARD = None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def lanes_for(config):
    lanes = []
    mapping = {}
    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            mapping[lane] = approach
    return lanes, mapping


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    return value


def fs_path(path: Path | str) -> Path:
    """Path usable under Windows MAX_PATH (~260) when LongPathsEnabled=0.

    Uses the ``\\\\?\\`` extended-length prefix. Prefer short output roots
    (subst/junction + ``Path.absolute()``, not ``resolve()``) as well.
    """
    text = os.path.abspath(str(path))
    if os.name != "nt" or text.startswith("\\\\?\\"):
        return Path(text)
    if text.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + text[2:])
    return Path("\\\\?\\" + text)


def write_json(path: Path, value) -> None:
    path = Path(path)
    fs_path(path.parent).mkdir(parents=True, exist_ok=True)
    # Short temp name: ``.json.tmp`` pushed several cells to exactly 260 chars.
    temporary = path.with_name(path.stem + ".tmp")
    payload = (
        json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )
    fs_path(temporary).write_text(payload, encoding="utf-8")
    os.replace(str(fs_path(temporary)), str(fs_path(path)))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"non-object JSONL at {path}:{line_number}")
        records.append(value)
    return records


def summarize_expression_attempts(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown"))
        counts[status] = counts.get(status, 0) + 1
    invalid_statuses = {
        "format_invalid",
        "symbolic_parse_invalid",
        "canonical_duplicate",
        "validator_rejected",
    }
    invalid = sum(counts.get(status, 0) for status in invalid_statuses)
    accepted = counts.get("accepted", 0)
    output_attempts = invalid + accepted
    return {
        "records": len(records),
        "status_counts": counts,
        "accepted_outputs": accepted,
        "invalid_outputs": invalid,
        "invalid_output_ratio": (
            invalid / output_attempts if output_attempts else None
        ),
        "generation_exceptions": counts.get("generation_exception", 0),
        "duplicate_outputs": counts.get("canonical_duplicate", 0),
    }


def summarize_llm_attempts(records: list[dict]) -> dict:
    success = [item for item in records if item.get("status") == "success"]
    errors = [item for item in records if item.get("status") == "error"]
    usage_totals: dict[str, float] = {}
    for item in records:
        usage = item.get("usage")
        if not isinstance(usage, dict):
            continue
        for name, value in usage.items():
            if isinstance(value, (int, float)) and np.isfinite(value):
                usage_totals[str(name)] = usage_totals.get(str(name), 0.0) + float(value)
    return {
        "api_attempts": len(records),
        "successful_responses": len(success),
        "failed_attempts": len(errors),
        "retry_attempts": sum(bool(item.get("retry")) for item in records),
        "response_models": sorted(
            {
                str(item["response_model"])
                for item in success
                if item.get("response_model")
            }
        ),
        "http_statuses": sorted(
            {
                int(item["http_status"])
                for item in records
                if item.get("http_status") is not None
            }
        ),
        "usage_totals": usage_totals,
    }


def accessed_dataset_splits(records: list[dict]) -> list[str]:
    """Return data splits only; allow-listed runtime artifacts are not data."""
    dataset_splits = {"train", "validation", "test"}
    return sorted(
        {
            str(item["split"])
            for item in records
            if item.get("split") in dataset_splits
        }
    )


def sanitize_formal_history(value):
    """Remove misleading legacy Validation labels from Training-only artifacts."""
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            key = str(key)
            if key == "validation_feedback":
                cleaned["physics_feedback"] = sanitize_formal_history(item)
            elif key.startswith("validation_") or key == "outer_validation_accessed":
                continue
            else:
                cleaned[key] = sanitize_formal_history(item)
        return cleaned
    if isinstance(value, list):
        return [sanitize_formal_history(item) for item in value]
    if isinstance(value, tuple):
        return [sanitize_formal_history(item) for item in value]
    return value


def _main() -> int:
    args = arguments()
    if args.intersection not in INTERSECTION_CONFIGS:
        raise ValueError(f"unknown intersection {args.intersection}")
    if args.output.exists():
        raise FileExistsError(
            f"refusing to overwrite or resume a formal run: {args.output}"
        )
    llm = load_llm_config()
    observed_llm = {
        "model": llm.model,
        "temperature": llm.temperature,
        "top_p": llm.top_p,
        "max_tokens": llm.max_tokens,
        "seed": llm.seed,
    }
    if observed_llm != CLEAN_POLICY.llm_sampling:
        raise RuntimeError(
            "formal LLM configuration differs from the frozen clean policy: "
            f"observed={observed_llm}, required={CLEAN_POLICY.llm_sampling}"
        )
    validate_source_manifest()

    train_path = (
        args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    ).resolve()
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    audit_path = args.output / "llm_audit.jsonl"
    expression_audit_path = args.output / "expression_attempt_audit.jsonl"
    access_guard = SplitAccessGuard(
        train_path,
        guarded_roots=(GMINI.parent.parent,),
        allowed_artifact_paths=(
            HERE / "method_config.json",
            audit_path,
            expression_audit_path,
        ),
    )
    access_guard.install()
    args.output.mkdir(parents=True, exist_ok=False)

    config = INTERSECTION_CONFIGS[args.intersection]
    train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), args.intersection),
        args.intersection,
    ).reset_index(drop=True)
    approaches = list(config["approaches"])
    targets = {
        approach: train[f"Delay_{approach}"].reset_index(drop=True)
        for approach in approaches
    }
    lanes, lane_to_approach = lanes_for(config)
    evolution_seed = (
        CLEAN_POLICY.evolution_base_seed + args.intersection * 10_000
    )

    llm_public_config = {
        "endpoint": llm.endpoint,
        "request_path": llm.request_path,
        "model_requested": llm.model,
        "temperature": llm.temperature,
        "temperature_sent_to_provider": llm.temperature is not None,
        "top_p": llm.top_p,
        "top_p_sent_to_provider": llm.top_p is not None,
        "max_tokens": llm.max_tokens,
        "max_tokens_sent_to_provider": llm.max_tokens is not None,
        "sampling_parameter_sources": llm.sampling_parameter_sources,
        "seed_requested": llm.seed,
        "seed_sent_to_provider": llm.seed is not None,
        "provider_seed_support_verified": False,
        "timeout_seconds": llm.timeout,
        "api_key_recorded": False,
        "matches_frozen_policy": True,
    }
    previous_audit = os.environ.get("LLM_AUDIT_LOG")
    previous_content = os.environ.get("LLM_AUDIT_INCLUDE_CONTENT")
    previous_expression_audit = os.environ.get("EXPRESSION_ATTEMPT_AUDIT_LOG")
    previous_bounds = {
        "scale": optimization_lane.DEFAULT_PARAM_BOUNDS,
        "power_exponent": optimization_lane.POWER_EXPONENT_BOUNDS,
        "exp_coefficient": optimization_lane.EXP_COEFFICIENT_BOUNDS,
    }
    bounds = CLEAN_POLICY.coefficient_bounds
    optimization_lane.DEFAULT_PARAM_BOUNDS = bounds["scale"]
    optimization_lane.POWER_EXPONENT_BOUNDS = bounds["power_exponent"]
    optimization_lane.EXP_COEFFICIENT_BOUNDS = bounds["exp_coefficient"]
    # absolute() keeps subst/junction short roots; resolve() expands to MAX_PATH.
    os.environ["LLM_AUDIT_LOG"] = str(audit_path.absolute())
    os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = "true"
    os.environ["EXPRESSION_ATTEMPT_AUDIT_LOG"] = str(
        expression_audit_path.absolute()
    )

    started_utc = utc_now()
    started = time.perf_counter()
    try:
        with install_clean_single_evolution(
            df_train=train,
            targets=targets,
            lanes=lanes,
            lane_to_approach=lane_to_approach,
            intersection_id=args.intersection,
            policy=CLEAN_POLICY,
        ) as runtime:
            expression, thought, explanation, parameters, history = (
                evolve_universal_lane_expression(
                    df_train=train,
                    lanes=lanes,
                    lane_to_approach=lane_to_approach,
                    approach_targets=targets,
                    universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                    generations=CLEAN_POLICY.generations,
                    pop_size=CLEAN_POLICY.population,
                    intersection_id=args.intersection,
                    score_mode="binary",
                    physics_weight=1.0,
                    prompt_knowledge=CLEAN_POLICY.prompt_knowledge,
                    prompt_style=CLEAN_POLICY.prompt_style,
                    seed=evolution_seed,
                    optimizer_restarts=CLEAN_POLICY.optimizer_restarts,
                    max_wall_seconds=None,
                    residual_guidance_mode=CLEAN_POLICY.residual_guidance_mode,
                    structural_diversity_mode=(
                        CLEAN_POLICY.structural_diversity_mode
                    ),
                    use_feasible_archive=False,
                    targeted_physical_feedback=(
                        CLEAN_POLICY.targeted_physical_feedback
                    ),
                )
            )
            selected = runtime.evaluations.get(str(expression))
            if selected is None:
                raise RuntimeError("winner is not a physically passing evaluation")
            if parameters != selected["parameters"]:
                raise RuntimeError("winner parameters changed after evolution")
            if runtime.incumbent_injected:
                raise RuntimeError("clean runtime unexpectedly injected an incumbent")
            prefit_audit = list(runtime.prefit_audit)
            fitted_rejections = list(runtime.fitted_rejections)
            generation_audit = list(runtime.generation_audit)
            passing_evaluations = len(runtime.evaluations)
    finally:
        optimization_lane.DEFAULT_PARAM_BOUNDS = previous_bounds["scale"]
        optimization_lane.POWER_EXPONENT_BOUNDS = previous_bounds[
            "power_exponent"
        ]
        optimization_lane.EXP_COEFFICIENT_BOUNDS = previous_bounds[
            "exp_coefficient"
        ]
        if previous_audit is None:
            os.environ.pop("LLM_AUDIT_LOG", None)
        else:
            os.environ["LLM_AUDIT_LOG"] = previous_audit
        if previous_content is None:
            os.environ.pop("LLM_AUDIT_INCLUDE_CONTENT", None)
        else:
            os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = previous_content
        if previous_expression_audit is None:
            os.environ.pop("EXPRESSION_ATTEMPT_AUDIT_LOG", None)
        else:
            os.environ["EXPRESSION_ATTEMPT_AUDIT_LOG"] = (
                previous_expression_audit
            )

    budget = next(
        item
        for item in reversed(history)
        if item.get("event") == "search_budget_summary"
    )
    initialization = [
        item
        for item in history
        if item.get("event") == "evaluated" and item.get("generation") == 0
    ]
    if len(initialization) != CLEAN_POLICY.population:
        raise RuntimeError(
            "formal initialization did not contain exactly "
            f"{CLEAN_POLICY.population} evaluated expressions"
        )
    if int(budget["completed_generations"]) != CLEAN_POLICY.generations:
        raise RuntimeError("formal search did not complete all generations")
    initialization_expressions = {str(item["expression"]) for item in initialization}
    generated_initial = {
        str(item["expression"])
        for item in generation_audit
        if item["mutation_type"] == "initial"
        and item["parent_expression"] is None
        and not item["external_incumbent"]
    }
    if not initialization_expressions.issubset(generated_initial):
        raise RuntimeError("an initial survivor lacks from-scratch generation evidence")
    if any(item["external_incumbent"] for item in generation_audit):
        raise RuntimeError("generation audit contains an external incumbent")
    expression_attempts = read_jsonl(expression_audit_path)
    llm_attempts = read_jsonl(audit_path)
    expression_attempt_summary = summarize_expression_attempts(
        expression_attempts
    )
    llm_attempt_summary = summarize_llm_attempts(llm_attempts)
    if expression_attempt_summary["accepted_outputs"] < len(generation_audit):
        raise RuntimeError(
            "expression-attempt audit has fewer accepted outputs than the "
            "current-run generation audit"
        )
    accessed_splits = accessed_dataset_splits(access_guard.records)
    if accessed_splits != ["train"]:
        raise RuntimeError(f"unexpected dataset split access: {accessed_splits}")

    output = {
        "schema_version": 1,
        "method_id": CLEAN_POLICY.method_id,
        "method_status": CLEAN_POLICY.method_status,
        "formal_protocol_complete": True,
        "started_utc": started_utc,
        "completed_utc": utc_now(),
        "intersection_id": args.intersection,
        "base_seed": CLEAN_POLICY.evolution_base_seed,
        "evolution_seed": evolution_seed,
        "policy": CLEAN_POLICY.to_dict(),
        "active_coefficient_bounds": bounds,
        "train_file_name": train_path.name,
        "train_file_sha256": sha256_file(train_path),
        "train_rows": len(train),
        "data_access_audit": access_guard.records,
        "project_artifact_read_policy": "allowlist_only",
        "accessed_splits": accessed_splits,
        "selection_source": "full_training_evolution_only",
        "outer_validation_accessed": False,
        "test_file_opened": False,
        "project_wide_prior_test_exposure": True,
        "fresh_unseen_test_still_required": True,
        "post_evolution_cv_reranking": False,
        "post_evolution_refit": False,
        "winner_reuses_in_evolution_parameters": True,
        "incumbent_argument_supported": False,
        "incumbent_result": None,
        "incumbent_injected_into_initial_population": False,
        "initial_population_all_generated_in_current_run": True,
        "initial_population_size_verified": len(initialization),
        "llm_runtime": llm_public_config,
        "llm_attempt_summary": llm_attempt_summary,
        "expression_attempt_summary": expression_attempt_summary,
        "expression": expression,
        "thought": thought,
        "explanation": explanation,
        "lane_parameters": parameters,
        "training_metrics": selected["metrics"],
        "formal_comparison_training_metrics": {
            "raw_approach_macro_r2": selected["metrics"]["macro_raw_r2"],
            "pooled_rmse": selected["metrics"]["pooled_rmse"],
            "pooled_mae": selected["metrics"]["pooled_mae"],
        },
        "training_fitness": selected["fitness"],
        "parameter_quality_audit": audit_parameter_quality(
            expression, parameters, bounds
        ),
        "compute_efficiency": summarize_compute_efficiency(history),
        "selected_enhanced_physics_from_evolution": selected["enhanced_physics"],
        "completed_generations": budget["completed_generations"],
        "candidate_evaluations": budget["candidate_evaluations"],
        "final_selection_policy": budget["final_selection_policy"],
        "prefit_gate_attempts": len(prefit_audit),
        "prefit_gate_passes": sum(
            bool(item.get("passed")) for item in prefit_audit
        ),
        "prefit_gate_rejections": sum(
            not bool(item.get("passed")) for item in prefit_audit
        ),
        "prefit_gate_pass_rate": (
            sum(bool(item.get("passed")) for item in prefit_audit)
            / len(prefit_audit)
            if prefit_audit
            else None
        ),
        "fitted_physics_rejections": len(fitted_rejections),
        "physically_passing_evaluations": passing_evaluations,
        "postfit_pass_rate": (
            passing_evaluations / (passing_evaluations + len(fitted_rejections))
            if passing_evaluations + len(fitted_rejections)
            else None
        ),
        "population_physical_pass_rate_by_construction": 1.0,
        "population_pass_rate_interpretation": (
            "hard gate admits only enhanced-physics passing candidates; use "
            "prefit_gate_pass_rate and postfit_pass_rate for generator yield"
        ),
        "wall_seconds": time.perf_counter() - started,
        "source_sha256": {
            str(path.relative_to(GMINI)): sha256_file(path) for path in SOURCE_FILES
        },
        "legacy_validation_labels_removed_from_formal_history": True,
    }
    write_json(args.output / "result.json", output)
    write_json(args.output / "history.json", sanitize_formal_history(history))
    write_json(args.output / "generation_audit.json", generation_audit)
    write_json(args.output / "prefit_gate_audit.json", prefit_audit)
    write_json(args.output / "fitted_rejections.json", fitted_rejections)
    metrics = selected["metrics"]
    print(
        f"I{args.intersection} clean P10/G10: "
        f"fitness={selected['fitness']:.6f}, "
        f"raw_macro_R2={metrics['macro_raw_r2']:.6f}, "
        f"pooled_RMSE={metrics['pooled_rmse']:.6f}, "
        f"pooled_MAE={metrics['pooled_mae']:.6f}, "
        f"physics={selected['enhanced_physics']['joint_pass']}, "
        f"wall={output['wall_seconds']:.1f}s"
    )
    return 0


def main() -> int:
    try:
        return _main()
    finally:
        guard = _ACTIVE_ACCESS_GUARD
        if guard is not None:
            guard.disable()


if __name__ == "__main__":
    raise SystemExit(main())
