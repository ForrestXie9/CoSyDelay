"""Freeze an Intersection-1 model from a completed reviewer evolution matrix.

This is deliberately stricter than the historical exporter.  It accepts only
the complete nine-intersection ``physics_score`` matrix produced by the reviewer runner, audits every
planned run and selects an Intersection-1 ``principlewise`` candidate by its
locked validation predictions.  Test predictions are opened only after the
winner has been fixed and are report-only.

The historical ``traffic_models_lane_level_1`` directory is not accepted.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Dict, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parent
GMINI = ROOT.parent
DEFAULT_MATRIX = GMINI / "reviewer_revision_experiments" / "matrix_runs" / "physics_score"
DEFAULT_OUTPUT = ROOT / "models" / "symbolic_lane_model.json"

LANES = (
    "S_R", "S_T", "S_L",
    "N_R", "N_T", "N_L",
    "E_R", "E_T", "E_L",
    "W_R", "W_T", "W_L",
)
APPROACHES = ("S", "E", "N", "W")
FEATURES = ("flow_lane", "GR_phase", "Cycle_Time")
SHA256_RE = re.compile(r"[0-9a-f]{64}")
COEFFICIENT_RE = re.compile(r"\ba\d+\b")
EXPECTED_RULE_SCHEMA_ID = "operational_domain_binary_hybrid_2026_07_14_v12"
EXPECTED_RULE_ORDER = (
    "R1_required_variables",
    "R2_nondecreasing_flow",
    "R3_nonincreasing_green",
    "R4_time_dimension",
    "R6_nonnegative_delay",
    "R7_operational_responsiveness",
    "R8_zero_flow_boundary",
    "R9_zero_green_limit",
)

REQUIRED_RUN_ARTIFACTS = (
    "config.json",
    "config.sha256",
    "history.json",
    "model.json",
    "validation_predictions_long.csv",
    "test_predictions_long.csv",
    "predictions_long.csv",
    "metrics.json",
    "llm_audit.jsonl",
    "console.log",
)
HASHED_RUN_ARTIFACTS = tuple(
    name for name in REQUIRED_RUN_ARTIFACTS if name != "config.sha256"
)


class MatrixAuditError(RuntimeError):
    """The source matrix is not safe to promote to a formal controller."""


@dataclass(frozen=True)
class FormalMinimums:
    """Minimum design declared by the reviewer revision protocol."""

    generations: int = 10
    population: int = 20
    runs_per_cell: int = 10
    optimizer_restarts: int = 3
    required_intersections: tuple[int, ...] = tuple(range(1, 10))


@dataclass(frozen=True)
class AuditedRun:
    run_dir: Path
    relative_run_dir: str
    config_hash: str
    config: Dict[str, Any]
    status: Dict[str, Any]
    model: Dict[str, Any]
    artifact_sha256: Dict[str, str]


@dataclass(frozen=True)
class MatrixAudit:
    root: Path
    plan: Dict[str, Any]
    plan_sha256: str
    runs: tuple[AuditedRun, ...]
    source_data_sha256: Dict[str, Dict[str, str]]


def _json_pairs(pairs: Sequence[tuple[str, Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_json_pairs,
                parse_constant=_reject_nonfinite_json,
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise MatrixAuditError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MatrixAuditError(f"Expected a JSON object: {path}")
    return value


def _read_json_list(path: Path) -> list[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_json_pairs,
                parse_constant=_reject_nonfinite_json,
            )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise MatrixAuditError(f"Invalid JSON artifact {path}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise MatrixAuditError(f"Expected a JSON array of objects: {path}")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise MatrixAuditError(f"Cannot hash required artifact {path}: {exc}") from exc
    return digest.hexdigest()


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _configuration_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and SHA256_RE.fullmatch(value) is not None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise MatrixAuditError(message)


def _safe_relative(root: Path, value: Any, description: str) -> Path:
    _require(isinstance(value, str) and bool(value.strip()), f"Missing {description}")
    relative = Path(value)
    _require(not relative.is_absolute(), f"{description} must be relative: {value}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise MatrixAuditError(f"{description} escapes matrix root: {value}") from exc
    return resolved


def _as_positive_int(value: Any, description: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool) and value > 0,
             f"{description} must be a positive integer")
    return int(value)


def _valid_physical_verifier_protocol(value: Any) -> bool:
    return bool(
        isinstance(value, dict)
        and value.get("rule_schema_id") == EXPECTED_RULE_SCHEMA_ID
        and tuple(value.get("rule_order", ())) == EXPECTED_RULE_ORDER
        and isinstance(value.get("config"), dict)
    )


def _validate_plan_header(
    plan: Mapping[str, Any], minimums: FormalMinimums
) -> tuple[list[int], int]:
    _require(plan.get("schema_version") == 3, "Only reviewer matrix schema_version=3 is accepted")
    _require(
        _valid_physical_verifier_protocol(plan.get("physical_verifier")),
        "Matrix does not declare the finalized Table-I verifier protocol",
    )
    _require(plan.get("execution_status") == "complete", "Matrix is not complete")
    _require(plan.get("plan_only_is_not_an_experiment_run") is False,
             "Plan-only matrix cannot be frozen")
    _require(plan.get("experiment") == "physics_score",
             "Legacy or non-physics_score matrices are rejected")

    variants = plan.get("variants")
    _require(isinstance(variants, list), "Matrix variants are missing")
    variant_map: Dict[str, Dict[str, Any]] = {}
    for item in variants:
        _require(isinstance(item, dict), "Invalid variant declaration")
        name = item.get("variant")
        _require(isinstance(name, str) and name not in variant_map,
                 "Duplicate or missing variant declaration")
        variant_map[name] = item
    _require(set(variant_map) == {"binary", "principlewise"},
             "physics_score matrix must contain both binary and principlewise cells")
    expected_variants = {
        "binary": ("cosydelay_binary", "binary"),
        "principlewise": ("cosydelay_principlewise", "principlewise"),
    }
    for name, (method, score_mode) in expected_variants.items():
        item = variant_map[name]
        _require(item.get("method") == method and item.get("score_mode") == score_mode,
                 f"Variant {name} does not match the reviewer runner contract")
        _require(item.get("prompt_knowledge") is True,
                 f"Variant {name} has an unexpected prompt condition")

    intersections_raw = plan.get("intersections")
    _require(isinstance(intersections_raw, list) and intersections_raw,
             "Matrix intersections are missing")
    intersections = [_as_positive_int(value, "intersection id") for value in intersections_raw]
    _require(len(intersections) == len(set(intersections)) and intersections == sorted(intersections),
             "Matrix intersections must be unique and sorted")
    required_intersections = tuple(minimums.required_intersections)
    _require(
        tuple(intersections) == required_intersections,
        "Reviewer matrix intersections must exactly match "
        f"{list(required_intersections)}; received {intersections}",
    )
    _require(1 in intersections, "Intersection 1 is absent from the matrix")

    generations = _as_positive_int(plan.get("generations"), "generations")
    population = _as_positive_int(plan.get("population"), "population")
    runs_per_cell = _as_positive_int(plan.get("runs_per_cell"), "runs_per_cell")
    optimizer_restarts = _as_positive_int(
        plan.get("optimizer_restarts"), "optimizer_restarts"
    )
    _require(generations >= minimums.generations,
             f"Pilot matrix rejected: generations must be >= {minimums.generations}")
    _require(population >= minimums.population,
             f"Pilot matrix rejected: population must be >= {minimums.population}")
    _require(runs_per_cell >= minimums.runs_per_cell,
             f"Pilot matrix rejected: runs_per_cell must be >= {minimums.runs_per_cell}")
    _require(optimizer_restarts >= minimums.optimizer_restarts,
             f"Pilot matrix rejected: optimizer_restarts must be >= {minimums.optimizer_restarts}")
    physics_weight = plan.get("physics_weight")
    _require(
        isinstance(physics_weight, (int, float))
        and not isinstance(physics_weight, bool)
        and math.isfinite(float(physics_weight))
        and float(physics_weight) > 0.0,
        "physics_weight must be finite and positive",
    )
    return intersections, runs_per_cell


def _manifest_semantic_hash(path: Path, expected_split: str | None) -> tuple[str, Dict[str, int], set[str]]:
    compact: list[Dict[str, Any]] = []
    counts: Dict[str, int] = {}
    dialogue_hashes: set[str] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"source_index", "sample_id", "dialogue_sha256", "split"}
            _require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                     f"Split manifest schema mismatch: {path}")
            for line_number, row in enumerate(reader, start=2):
                try:
                    source_index = int(row["source_index"])
                except (TypeError, ValueError) as exc:
                    raise MatrixAuditError(
                        f"Invalid source_index in {path}:{line_number}"
                    ) from exc
                split = str(row["split"])
                if expected_split is not None:
                    _require(split == expected_split,
                             f"Unexpected split {split!r} in {path}:{line_number}")
                _require(split in {"fit", "validation", "test"},
                         f"Unknown split in {path}:{line_number}")
                dialogue_sha256 = str(row["dialogue_sha256"])
                _require(_is_sha256(dialogue_sha256),
                         f"Invalid dialogue SHA-256 in {path}:{line_number}")
                compact.append({
                    "source_index": source_index,
                    "sample_id": str(row["sample_id"]),
                    "dialogue_sha256": dialogue_sha256,
                    "split": split,
                })
                counts[split] = counts.get(split, 0) + 1
                dialogue_hashes.add(dialogue_sha256)
    except OSError as exc:
        raise MatrixAuditError(f"Cannot read split manifest {path}: {exc}") from exc
    _require(bool(compact), f"Empty split manifest: {path}")
    return _configuration_hash(compact), counts, dialogue_hashes


def _validate_source_data(
    matrix_root: Path,
    plan: Mapping[str, Any],
    intersections: Iterable[int],
    data_dir_override: Path | None,
) -> Dict[str, Dict[str, str]]:
    plan_data = plan.get("data")
    _require(isinstance(plan_data, dict), "Plan data registry is missing")
    result: Dict[str, Dict[str, str]] = {}
    train_dialogues: Dict[int, set[str]] = {}
    test_dialogues: Dict[int, set[str]] = {}
    for intersection in intersections:
        info = plan_data.get(str(intersection))
        _require(isinstance(info, dict), f"Missing data declaration for Intersection {intersection}")
        configured_data_dir = info.get("data_dir")
        _require(isinstance(configured_data_dir, str), "Configured data_dir is missing")
        data_dir = data_dir_override.resolve() if data_dir_override else Path(configured_data_dir).resolve()
        train_name = info.get("train_file")
        test_name = info.get("test_file")
        _require(isinstance(train_name, str) and Path(train_name).name == train_name,
                 f"Unsafe train filename for Intersection {intersection}")
        _require(isinstance(test_name, str) and Path(test_name).name == test_name,
                 f"Unsafe test filename for Intersection {intersection}")
        train_path = data_dir / train_name
        test_path = data_dir / test_name
        _require(train_path.is_file() and test_path.is_file(),
                 f"Source data files are unavailable for Intersection {intersection}; use --data-dir if moved")
        train_sha = _sha256_file(train_path)
        test_sha = _sha256_file(test_path)
        _require(train_sha == info.get("train_file_sha256"),
                 f"Train data hash mismatch for Intersection {intersection}")
        _require(test_sha == info.get("test_file_sha256"),
                 f"Test data hash mismatch for Intersection {intersection}")

        train_manifest = _safe_relative(
            matrix_root, info.get("train_manifest"), "train split manifest"
        )
        test_manifest = _safe_relative(
            matrix_root, info.get("test_manifest"), "test sample manifest"
        )
        train_manifest_hash, train_counts, train_hashes = _manifest_semantic_hash(
            train_manifest, None
        )
        test_manifest_hash, test_counts, test_hashes = _manifest_semantic_hash(
            test_manifest, "test"
        )
        _require(set(train_counts) == {"fit", "validation"},
                 f"Train split must contain fit and validation for Intersection {intersection}")
        _require(train_manifest_hash == info.get("train_manifest_sha256"),
                 f"Train split manifest hash mismatch for Intersection {intersection}")
        _require(test_manifest_hash == info.get("test_manifest_sha256"),
                 f"Test split manifest hash mismatch for Intersection {intersection}")
        _require(not (train_hashes & test_hashes),
                 f"Train/test dialogue leakage for Intersection {intersection}")
        sample_counts = info.get("sample_counts")
        _require(isinstance(sample_counts, dict), "Data sample counts are missing")
        expected_counts = {
            "fit": train_counts.get("fit", 0),
            "validation": train_counts.get("validation", 0),
            "test": test_counts.get("test", 0),
        }
        _require(sample_counts == expected_counts,
                 f"Sample count mismatch for Intersection {intersection}")
        train_dialogues[intersection] = train_hashes
        test_dialogues[intersection] = test_hashes
        result[str(intersection)] = {
            "train_file": str(train_path.resolve()),
            "train_file_sha256": train_sha,
            "test_file": str(test_path.resolve()),
            "test_file_sha256": test_sha,
            "train_manifest": str(train_manifest),
            "train_manifest_sha256": train_manifest_hash,
            "test_manifest": str(test_manifest),
            "test_manifest_sha256": test_manifest_hash,
            "data_config_sha256": _configuration_hash(info),
        }
    return result


def _validate_run_config(
    config: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_data: Mapping[str, Any],
) -> None:
    _require(config.get("schema_version") == 3, "Run config schema mismatch")
    _require(
        config.get("physical_verifier") == plan.get("physical_verifier"),
        "Run verifier protocol differs from the matrix plan",
    )
    _require(config.get("experiment") == "physics_score", "Non-physics_score run rejected")
    variant = config.get("variant")
    expected = {
        "binary": ("binary", "cosydelay_binary"),
        "principlewise": ("principlewise", "cosydelay_principlewise"),
    }
    _require(variant in expected, "Legacy or unknown run variant rejected")
    expected_score, expected_method = expected[str(variant)]
    _require(config.get("score_mode") == expected_score,
             f"Run {variant} has the wrong score_mode")
    _require(config.get("method") == expected_method,
             f"Run {variant} has the wrong method")
    _require(config.get("selection_split") == "validation",
             "Run did not use validation for selection")
    _require(config.get("test_used_for_selection") is False,
             "Run declares test-assisted selection")
    _require(config.get("prompt_knowledge") is True,
             "physics_score run has an unexpected prompt condition")
    _require(tuple(config.get("universal_features", ())) == FEATURES,
             "Run feature contract mismatch")
    for name in ("generations", "population", "runs_in_matrix", "split_seed",
                 "optimizer_restarts", "physics_weight", "validation_fraction"):
        plan_name = "runs_per_cell" if name == "runs_in_matrix" else name
        _require(config.get(name) == plan.get(plan_name),
                 f"Run/plan configuration mismatch for {name}")
    intersection = _as_positive_int(config.get("intersection_id"), "run intersection_id")
    _as_positive_int(config.get("run_id"), "run_id")
    _require(config.get("data") == plan_data.get(str(intersection)),
             f"Run data configuration mismatch for Intersection {intersection}")
    code_hashes = config.get("code_sha256")
    _require(isinstance(code_hashes, dict) and bool(code_hashes), "Run code hashes are missing")
    for filename, digest in code_hashes.items():
        _require(isinstance(filename, str) and _is_sha256(digest),
                 f"Invalid source code hash for {filename!r}")
    _require("expression_validation_lane.py" in code_hashes,
             "Run does not identify the principle-wise verifier source")


def _validate_history(path: Path, config: Mapping[str, Any]) -> list[Dict[str, Any]]:
    history = _read_json_list(path)
    population = int(config["population"])
    generations = int(config["generations"])
    expected = population * (generations + 1)
    _require(len(history) == expected,
             f"Incomplete history {path}: {len(history)} records, expected {expected}")
    generation_counts: Dict[int, int] = {}
    for index, record in enumerate(history):
        generation = record.get("generation")
        _require(isinstance(generation, int) and 0 <= generation <= generations,
                 f"Invalid generation in {path} record {index}")
        generation_counts[generation] = generation_counts.get(generation, 0) + 1
        _require(record.get("score_mode") == config.get("score_mode"),
                 f"History score_mode mismatch in {path} record {index}")
    _require(generation_counts == {generation: population for generation in range(generations + 1)},
             f"History generation coverage mismatch: {path}")
    return history


def _validate_model_metadata(
    model: Mapping[str, Any], config: Mapping[str, Any], path: Path
) -> None:
    for field in ("experiment", "variant", "method", "intersection_id", "run_id"):
        _require(model.get(field) == config.get(field),
                 f"Model/config mismatch for {field}: {path}")
    status = model.get("status")
    _require(status is None or status not in {"fallback", "fallback_not_for_formal_control", "legacy"},
             f"Fallback/legacy model rejected: {path}")
    selection = model.get("selection")
    _require(isinstance(selection, dict), f"Model selection metadata missing: {path}")
    _require(selection.get("score_mode") == config.get("score_mode"),
             f"Model score mode mismatch: {path}")
    _require(selection.get("physical_verifier") == config.get("physical_verifier"),
             f"Model verifier protocol mismatch: {path}")
    _require(selection.get("selection_split") == "validation",
             f"Model did not use validation selection: {path}")
    _require(selection.get("test_used_for_selection") is False,
             f"Model declares test-assisted selection: {path}")


def _validate_one_run(
    matrix_root: Path,
    entry: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> AuditedRun:
    relative_run_dir = entry.get("run_dir")
    run_dir = _safe_relative(matrix_root, relative_run_dir, "run directory")
    _require(run_dir.is_dir(), f"Missing planned run directory: {run_dir}")
    planned_config = entry.get("config")
    _require(isinstance(planned_config, dict), f"Missing planned config: {run_dir}")
    _require("config_hash" not in planned_config,
             f"Planned config contains forbidden self hash: {run_dir}")
    expected_hash = entry.get("config_hash")
    _require(_is_sha256(expected_hash), f"Invalid planned config hash: {run_dir}")
    _require(_configuration_hash(planned_config) == expected_hash,
             f"Plan config hash mismatch: {run_dir}")
    _validate_run_config(planned_config, plan, plan["data"])

    for name in REQUIRED_RUN_ARTIFACTS:
        _require((run_dir / name).is_file(), f"Partial run; missing {name}: {run_dir}")
    stored_hash = (run_dir / "config.sha256").read_text(encoding="utf-8").strip()
    _require(stored_hash == expected_hash, f"config.sha256 mismatch: {run_dir}")
    run_config = _read_json(run_dir / "config.json")
    _require(run_config.get("config_hash") == expected_hash,
             f"Run config_hash mismatch: {run_dir}")
    comparable_config = dict(run_config)
    comparable_config.pop("config_hash", None)
    _require(comparable_config == planned_config,
             f"Run config differs from preregistered plan: {run_dir}")

    status = _read_json(run_dir / "status.json")
    _require(status.get("execution_status") == "complete",
             f"Partial/failed run rejected: {run_dir}")
    _require(status.get("plan_only") is False, f"Plan-only run rejected: {run_dir}")
    _require(status.get("config_hash") == expected_hash,
             f"Status config hash mismatch: {run_dir}")
    artifact_hashes = status.get("artifact_sha256")
    _require(isinstance(artifact_hashes, dict), f"Run artifact hashes missing: {run_dir}")
    normalized_hashes: Dict[str, str] = {}
    for name in HASHED_RUN_ARTIFACTS:
        recorded = artifact_hashes.get(name)
        _require(_is_sha256(recorded), f"Missing/invalid hash for {name}: {run_dir}")
        actual = _sha256_file(run_dir / name)
        _require(actual == recorded, f"Artifact hash mismatch for {name}: {run_dir}")
        normalized_hashes[name] = actual

    history = _validate_history(run_dir / "history.json", planned_config)
    model = _read_json(run_dir / "model.json")
    _validate_model_metadata(model, planned_config, run_dir / "model.json")
    if planned_config.get("variant") == "principlewise" and planned_config.get("intersection_id") == 1:
        expression_block = model.get("universal_expression")
        _require(isinstance(expression_block, dict), f"Missing expression: {run_dir}")
        expression = expression_block.get("template")
        _require(isinstance(expression, str) and expression.strip(),
                 f"Empty expression: {run_dir}")
        matching = [record for record in history if record.get("expression") == expression]
        _require(bool(matching), f"Final model expression is absent from history: {run_dir}")

    return AuditedRun(
        run_dir=run_dir,
        relative_run_dir=str(relative_run_dir).replace("\\", "/"),
        config_hash=str(expected_hash),
        config=dict(planned_config),
        status=status,
        model=model,
        artifact_sha256=normalized_hashes,
    )


def audit_matrix(
    matrix_root: Path,
    *,
    data_dir_override: Path | None = None,
    minimums: FormalMinimums = FormalMinimums(),
) -> MatrixAudit:
    """Audit the complete matrix without reading any test prediction values."""
    root = matrix_root.resolve()
    plan_path = root / "plan_manifest.json"
    _require(plan_path.is_file(), f"Missing plan_manifest.json: {root}")
    plan = _read_json(plan_path)
    intersections, runs_per_cell = _validate_plan_header(plan, minimums)
    source_data = _validate_source_data(root, plan, intersections, data_dir_override)

    entries = plan.get("runs")
    _require(isinstance(entries, list), "Plan runs are missing")
    expected_count = 2 * len(intersections) * runs_per_cell
    _require(plan.get("run_count") == expected_count and len(entries) == expected_count,
             f"Partial matrix: expected {expected_count} planned runs")

    seen_dirs: set[str] = set()
    seen_cells: set[tuple[str, int, int]] = set()
    runs: list[AuditedRun] = []
    reference_code_hashes: Dict[str, str] | None = None
    for entry in entries:
        _require(isinstance(entry, dict), "Invalid run entry in plan")
        run = _validate_one_run(root, entry, plan)
        _require(run.relative_run_dir not in seen_dirs,
                 f"Duplicate run directory in plan: {run.relative_run_dir}")
        cell = (
            str(run.config["variant"]),
            int(run.config["intersection_id"]),
            int(run.config["run_id"]),
        )
        _require(cell not in seen_cells, f"Duplicate run cell in plan: {cell}")
        seen_dirs.add(run.relative_run_dir)
        seen_cells.add(cell)
        code_hashes = dict(run.config["code_sha256"])
        if reference_code_hashes is None:
            reference_code_hashes = code_hashes
        else:
            _require(code_hashes == reference_code_hashes,
                     "Matrix mixes different source-code hash bundles")
        runs.append(run)

    expected_cells = {
        (variant, intersection, run_id)
        for variant in ("binary", "principlewise")
        for intersection in intersections
        for run_id in range(1, runs_per_cell + 1)
    }
    _require(seen_cells == expected_cells, "Matrix cells do not match the preregistered factorial design")
    return MatrixAudit(
        root=root,
        plan=plan,
        plan_sha256=_sha256_file(plan_path),
        runs=tuple(runs),
        source_data_sha256=source_data,
    )


def _vector_metrics(rows: Sequence[Mapping[str, float]]) -> Dict[str, Any]:
    _require(bool(rows), "Prediction table is empty")
    observed = [float(row["y_true"]) for row in rows]
    predicted = [float(row["y_pred"]) for row in rows]
    errors = [prediction - truth for truth, prediction in zip(observed, predicted)]
    n = len(rows)
    mse = sum(error * error for error in errors) / n
    mae = sum(abs(error) for error in errors) / n
    mean_observed = sum(observed) / n
    denominator = sum((value - mean_observed) ** 2 for value in observed)
    numerator = sum(error * error for error in errors)
    r2 = None if denominator <= 0.0 else 1.0 - numerator / denominator
    percentage_errors = [
        abs(error / truth) * 100.0
        for truth, error in zip(observed, errors)
        if truth != 0.0
    ]
    return {
        "n": n,
        "r2": r2,
        "rmse": math.sqrt(mse),
        "mae": mae,
        "mape": sum(percentage_errors) / len(percentage_errors) if percentage_errors else None,
        "zero_target_count": sum(value == 0.0 for value in observed),
    }


def _split_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    by_approach = {
        approach: _vector_metrics([row for row in rows if row["approach"] == approach])
        for approach in APPROACHES
    }
    macro: Dict[str, Any] = {}
    for metric in ("r2", "rmse", "mae", "mape"):
        values = [item[metric] for item in by_approach.values() if item[metric] is not None]
        macro[metric] = sum(values) / len(values) if values else None
    return {
        "all_approaches_micro": _vector_metrics(rows),
        "macro_average": macro,
        "by_approach": by_approach,
    }


def _read_prediction_rows(run: AuditedRun, split: str) -> list[Dict[str, Any]]:
    filename = f"{split}_predictions_long.csv"
    path = run.run_dir / filename
    rows: list[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {
                "method", "run_id", "intersection_id", "approach",
                "sample_id", "y_true", "y_pred", "split",
            }
            _require(reader.fieldnames is not None and required.issubset(reader.fieldnames),
                     f"Prediction schema mismatch: {path}")
            for line_number, row in enumerate(reader, start=2):
                _require(row["split"] == split, f"Wrong split in {path}:{line_number}")
                _require(row["method"] == run.config["method"],
                         f"Wrong method in {path}:{line_number}")
                _require(int(row["run_id"]) == int(run.config["run_id"]),
                         f"Wrong run_id in {path}:{line_number}")
                _require(int(row["intersection_id"]) == 1,
                         f"Wrong intersection in {path}:{line_number}")
                approach = row["approach"]
                _require(approach in APPROACHES, f"Unknown approach in {path}:{line_number}")
                key = (approach, row["sample_id"])
                _require(key not in seen, f"Duplicate prediction key in {path}:{line_number}")
                try:
                    y_true = float(row["y_true"])
                    y_pred = float(row["y_pred"])
                except ValueError as exc:
                    raise MatrixAuditError(f"Non-numeric prediction in {path}:{line_number}") from exc
                _require(math.isfinite(y_true) and math.isfinite(y_pred),
                         f"Non-finite prediction in {path}:{line_number}")
                rows.append({
                    "approach": approach,
                    "sample_id": row["sample_id"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                })
                seen.add(key)
    except OSError as exc:
        raise MatrixAuditError(f"Cannot read predictions {path}: {exc}") from exc
    expected_samples = int(run.config["data"]["sample_counts"][split])
    _require(len(rows) == expected_samples * len(APPROACHES),
             f"Prediction row count mismatch for {path}")
    for approach in APPROACHES:
        _require(sum(row["approach"] == approach for row in rows) == expected_samples,
                 f"Approach coverage mismatch for {path}: {approach}")
    return rows


def _metric_close(actual: Any, expected: Any) -> bool:
    if actual is None or expected is None:
        return actual is None and expected is None
    if isinstance(expected, int):
        return actual == expected
    try:
        actual_number = float(actual)
        expected_number = float(expected)
    except (TypeError, ValueError):
        return False
    return math.isfinite(actual_number) and math.isclose(
        actual_number, expected_number, rel_tol=1e-10, abs_tol=1e-10
    )


def _verify_recorded_split(
    recorded: Mapping[str, Any], recomputed: Mapping[str, Any], description: str
) -> None:
    for section in ("all_approaches_micro", "macro_average"):
        _require(isinstance(recorded.get(section), dict), f"Missing {description} {section}")
        for metric, expected in recomputed[section].items():
            _require(_metric_close(recorded[section].get(metric), expected),
                     f"Recorded/recomputed {description} mismatch: {section}.{metric}")
    recorded_by_approach = recorded.get("by_approach")
    _require(isinstance(recorded_by_approach, dict), f"Missing {description} by_approach")
    for approach, values in recomputed["by_approach"].items():
        _require(isinstance(recorded_by_approach.get(approach), dict),
                 f"Missing {description} approach {approach}")
        for metric, expected in values.items():
            _require(_metric_close(recorded_by_approach[approach].get(metric), expected),
                     f"Recorded/recomputed {description} mismatch: {approach}.{metric}")


def _validation_result(run: AuditedRun) -> Dict[str, Any]:
    """Recompute validation metrics without opening metrics.json or any test values."""
    rows = _read_prediction_rows(run, "validation")
    recomputed = _split_metrics(rows)
    macro = recomputed["macro_average"]
    for metric in ("r2", "rmse", "mae"):
        _require(isinstance(macro.get(metric), (int, float)) and math.isfinite(float(macro[metric])),
                 f"Non-finite validation {metric}: {run.run_dir}")
    return recomputed


def select_intersection1_principlewise(
    audit: MatrixAudit,
) -> tuple[AuditedRun, Dict[str, Any], list[Dict[str, Any]]]:
    """Select only by validation macro R2, with deterministic validation tie-breaks."""
    candidates = [
        run for run in audit.runs
        if run.config.get("intersection_id") == 1
        and run.config.get("variant") == "principlewise"
        and run.config.get("score_mode") == "principlewise"
    ]
    expected = int(audit.plan["runs_per_cell"])
    _require(len(candidates) == expected,
             f"Expected {expected} Intersection-1 principlewise candidates, found {len(candidates)}")
    ranked: list[tuple[AuditedRun, Dict[str, Any]]] = []
    table: list[Dict[str, Any]] = []
    for run in candidates:
        validation = _validation_result(run)
        macro = validation["macro_average"]
        ranked.append((run, validation))
        table.append({
            "run_id": int(run.config["run_id"]),
            "run_dir": run.relative_run_dir,
            "config_sha256": run.config_hash,
            "validation_macro_r2": float(macro["r2"]),
            "validation_macro_rmse": float(macro["rmse"]),
            "validation_macro_mae": float(macro["mae"]),
        })
    winner, winner_validation = min(
        ranked,
        key=lambda item: (
            -float(item[1]["macro_average"]["r2"]),
            float(item[1]["macro_average"]["rmse"]),
            float(item[1]["macro_average"]["mae"]),
            int(item[0].config["run_id"]),
            item[0].config_hash,
        ),
    )
    table.sort(key=lambda row: int(row["run_id"]))
    return winner, winner_validation, table


def _winner_test_report(
    winner: AuditedRun, validation_metrics: Mapping[str, Any]
) -> Dict[str, Any]:
    """Open the selected run's test values only after selection is complete."""
    rows = _read_prediction_rows(winner, "test")
    recomputed = _split_metrics(rows)
    metrics = _read_json(winner.run_dir / "metrics.json")
    statement = metrics.get("selection_statement")
    _require(isinstance(statement, dict), f"Selection statement missing: {winner.run_dir}")
    _require(statement.get("validation_used_for_expression_selection") is True,
             f"Validation selection not declared: {winner.run_dir}")
    _require(statement.get("test_used_for_selection") is False,
             f"Test-assisted selection declared: {winner.run_dir}")
    recorded_validation = metrics.get("validation")
    _require(isinstance(recorded_validation, dict),
             f"Validation metrics missing: {winner.run_dir}")
    _verify_recorded_split(
        recorded_validation, validation_metrics, "selected-run validation"
    )
    recorded = metrics.get("test")
    _require(isinstance(recorded, dict), f"Test metrics missing: {winner.run_dir}")
    _verify_recorded_split(recorded, recomputed, "selected-run test")
    return recomputed


def _normalize_symbolic_model(winner: AuditedRun) -> tuple[str, Dict[str, Dict[str, float]], str, str]:
    expression_block = winner.model.get("universal_expression")
    _require(isinstance(expression_block, dict), "Selected model expression block is missing")
    expression = expression_block.get("template")
    _require(isinstance(expression, str) and expression.strip(), "Selected model expression is empty")
    _require("fallback" not in expression.lower(), "Fallback expression marker rejected")
    coefficient_names = set(COEFFICIENT_RE.findall(expression))
    _require(bool(coefficient_names), "Selected expression has no fitted coefficients")
    raw_parameters = winner.model.get("lane_parameters")
    _require(isinstance(raw_parameters, dict), "Selected lane parameters are missing")
    _require(set(raw_parameters) == set(LANES),
             "Selected model must contain exactly the 12 Intersection-1 lanes")
    normalized: Dict[str, Dict[str, float]] = {}
    for lane in LANES:
        values = raw_parameters[lane]
        _require(isinstance(values, dict) and set(values) == coefficient_names,
                 f"Coefficient set mismatch for lane {lane}")
        normalized[lane] = {}
        for name, value in values.items():
            _require(not isinstance(value, bool), f"Boolean coefficient rejected: {lane}.{name}")
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise MatrixAuditError(f"Non-numeric coefficient {lane}.{name}") from exc
            _require(math.isfinite(numeric), f"Non-finite coefficient {lane}.{name}")
            normalized[lane][name] = numeric
    thought = expression_block.get("thought", "")
    explanation = expression_block.get("explanation", "")
    return expression, normalized, str(thought), str(explanation)


def _require_selected_source_joint_pass(winner: AuditedRun, expression: str) -> None:
    """Do not silently fall back to a lower-ranked physically passing run."""
    history = _read_json_list(winner.run_dir / "history.json")
    matching = [record for record in history if record.get("expression") == expression]
    _require(bool(matching), "Selected expression is absent from its source history")
    _require(any(record.get("physical_joint_pass") is True for record in matching),
             "Validation-selected model did not pass the source eight-rule joint verifier; "
             "refusing to substitute a lower-ranked candidate")


def _physical_recheck(expression: str, lane_parameters: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    if str(GMINI) not in sys.path:
        sys.path.insert(0, str(GMINI))
    try:
        from expression_validation_lane import score_fitted_lanes_principlewise
    except ImportError as exc:
        raise MatrixAuditError(f"Cannot load the current principle-wise verifier: {exc}") from exc
    result = score_fitted_lanes_principlewise(
        expression,
        lane_parameters,
        list(LANES),
        list(FEATURES),
    )
    _require(result.joint_pass is True,
             "Selected model fails the current eight-rule joint physical verifier")
    return {
        "rule_schema_id": EXPECTED_RULE_SCHEMA_ID,
        "rule_order": list(EXPECTED_RULE_ORDER),
        "score": float(result.score),
        "joint_pass": bool(result.joint_pass),
        "rule_scores": {name: float(value) for name, value in result.rule_scores.items()},
        "lane_errors": result.lane_errors,
        "config": {
            "n_grid_samples": result.config.n_grid_samples,
            "lhs_seed": result.config.lhs_seed,
            "flow_domain": list(result.config.flow_domain),
            "green_domain": list(result.config.green_domain),
            "cycle_domain_seconds": list(result.config.cycle_domain_seconds),
            "zero_tolerance": result.config.zero_tolerance,
            "derivative_tolerance": result.config.derivative_tolerance,
            "rule_weights": list(result.config.rule_weights),
        },
        "verifier_source_sha256": _sha256_file(GMINI / "expression_validation_lane.py"),
    }


def _model_hash(expression: str, lane_parameters: Mapping[str, Any]) -> str:
    return _configuration_hash({"expression": expression, "lane_parameters": lane_parameters})


def _build_frozen_payload(
    audit: MatrixAudit,
    winner: AuditedRun,
    validation_metrics: Dict[str, Any],
    candidate_table: list[Dict[str, Any]],
    test_metrics: Dict[str, Any],
) -> Dict[str, Any]:
    expression, lane_parameters, thought, explanation = _normalize_symbolic_model(winner)
    _require_selected_source_joint_pass(winner, expression)
    physical = _physical_recheck(expression, lane_parameters)
    model_sha256 = _model_hash(expression, lane_parameters)
    code_hashes = dict(winner.config["code_sha256"])
    data_config = dict(winner.config["data"])
    return {
        "schema_version": 1,
        "status": "frozen",
        "model_sha256": model_sha256,
        "expression": expression,
        "lane_parameters": lane_parameters,
        "metrics": {
            "validation": validation_metrics,
            "test_report_only": test_metrics,
        },
        "selection_protocol": {
            "source_experiment": "physics_score",
            "eligible_variant": "principlewise",
            "eligible_intersection": 1,
            "primary_rule": "maximum validation macro-average R2",
            "tie_breakers": [
                "minimum validation macro-average RMSE",
                "minimum validation macro-average MAE",
                "minimum run_id",
                "lexicographic config_sha256",
            ],
            "test_used_for_selection": False,
            "test_opened_after_winner_fixed": True,
            "candidate_count": len(candidate_table),
            "candidate_validation_metrics": candidate_table,
            "selected_run_id": int(winner.config["run_id"]),
            "selected_run_dir": winner.relative_run_dir,
        },
        "evolution": {
            "intersection_id": 1,
            "variant": "principlewise",
            "score_mode": "principlewise",
            "generations": int(winner.config["generations"]),
            "population": int(winner.config["population"]),
            "independent_runs": int(winner.config["runs_in_matrix"]),
            "optimizer_restarts": int(winner.config["optimizer_restarts"]),
            "run_seed": int(winner.config["run_seed"]),
        },
        "validation": {
            "matrix_complete": True,
            "source_artifact_hashes_verified": True,
            "history_complete": True,
            "current_principlewise_physical_recheck": physical,
            "symbolic_model_schema_probe": "passed before atomic promotion",
        },
        "provenance": {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_matrix": str(audit.root),
            "source_plan": str(audit.root / "plan_manifest.json"),
            "source_plan_sha256": audit.plan_sha256,
            "source_run": str(winner.run_dir),
            "source_config_sha256": winner.config_hash,
            "source_config": winner.config,
            "source_code_sha256": code_hashes,
            "source_code_bundle_sha256": _configuration_hash(code_hashes),
            "source_data_config_sha256": _configuration_hash(data_config),
            "source_data": audit.source_data_sha256["1"],
            "source_artifact_sha256": {
                **winner.artifact_sha256,
                "status.json": _sha256_file(winner.run_dir / "status.json"),
                "config.sha256": _sha256_file(winner.run_dir / "config.sha256"),
            },
        },
        "thought": thought,
        "explanation": explanation,
    }


def _validate_with_controller_loader(path: Path) -> None:
    try:
        from experiment.controllers import SymbolicModel
        model = SymbolicModel.load(path)
    except Exception as exc:
        raise MatrixAuditError(f"Exported payload fails SymbolicModel.load: {exc}") from exc
    _require(model.schema_version == 1 and model.status == "frozen",
             "Exported payload failed the formal SymbolicModel contract")
    _require(set(model.lane_parameters) == set(LANES),
             "Exported payload failed the 12-lane SymbolicModel contract")


def _write_frozen(payload: Mapping[str, Any], output: Path, force: bool) -> Path | None:
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not force:
        raise FileExistsError(f"Frozen model already exists: {output}; use --force to replace it")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)
    )
    temporary = Path(temporary_name)
    archive_path: Path | None = None
    try:
        with open(descriptor, "w", encoding="utf-8", newline="\n", closefd=True) as handle:
            json.dump(payload, handle, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
        _validate_with_controller_loader(temporary)
        if output.exists():
            archive_dir = output.parent / "archive"
            archive_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
            archive_path = archive_dir / f"{output.stem}_{stamp}{output.suffix}"
            shutil.copy2(output, archive_path)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return archive_path


def freeze_from_matrix(
    matrix_root: Path,
    output: Path = DEFAULT_OUTPUT,
    *,
    data_dir_override: Path | None = None,
    force: bool = False,
    minimums: FormalMinimums = FormalMinimums(),
) -> Dict[str, Any]:
    """Audit, validation-select, test-report, and atomically freeze one model."""
    if output.exists() and not force:
        raise FileExistsError(f"Frozen model already exists: {output.resolve()}; use --force to replace it")
    audit = audit_matrix(
        matrix_root,
        data_dir_override=data_dir_override,
        minimums=minimums,
    )
    winner, validation_metrics, candidate_table = select_intersection1_principlewise(audit)
    # Selection is now immutable.  Only the chosen run's test file is opened.
    test_metrics = _winner_test_report(winner, validation_metrics)
    payload = _build_frozen_payload(
        audit, winner, validation_metrics, candidate_table, test_metrics
    )
    archive_path = _write_frozen(payload, output, force)
    if archive_path is not None:
        payload["_archived_previous_model"] = str(archive_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze an audited Intersection-1 principlewise model from a completed formal matrix."
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--data-dir",
        type=Path,
        help="Optional relocated directory containing the source Intersection_* JSONL files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Archive and atomically replace an existing formal frozen model.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = freeze_from_matrix(
            args.matrix,
            args.output,
            data_dir_override=args.data_dir,
            force=args.force,
        )
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    selection = payload["selection_protocol"]
    validation = payload["metrics"]["validation"]["macro_average"]
    test = payload["metrics"]["test_report_only"]["macro_average"]
    print(f"Frozen model: {args.output.resolve()}")
    print(f"Selected run: {selection['selected_run_dir']}")
    print(f"Model SHA-256: {payload['model_sha256']}")
    print(f"Validation macro R2: {validation['r2']:.6f} (selection)")
    print(f"Test macro R2: {test['r2']:.6f} (report only, opened after selection)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
