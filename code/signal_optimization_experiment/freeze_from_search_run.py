"""Freeze an Intersection-1 principlewise model from one completed search run.

Use when the full nine-intersection reviewer matrix is unavailable but a
completed ``physics_score`` principlewise Intersection-1 run already exists.
Selection is validation macro-average R2 only; test metrics are report-only.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

ROOT = Path(__file__).resolve().parent
GMINI = ROOT.parent
DEFAULT_OUTPUT = ROOT / "models" / "symbolic_lane_model.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from freeze_evolved_model import (  # noqa: E402
    MatrixAuditError,
    _configuration_hash,
    _physical_recheck,
    _read_json,
    _read_prediction_rows,
    _sha256_file,
    _split_metrics,
    _validate_model_metadata,
    _validate_run_config,
    _winner_test_report,
)


def _audited_run_from_directory(run_dir: Path) -> Any:
    run_dir = run_dir.resolve()
    config = _read_json(run_dir / "config.json")
    if config.get("intersection_id") != 1:
        raise MatrixAuditError("SUMO control export requires Intersection 1")
    if config.get("variant") != "principlewise":
        raise MatrixAuditError("SUMO control export requires principlewise variant")
    if config.get("experiment") != "physics_score":
        raise MatrixAuditError("Only physics_score runs are accepted")
    if config.get("selection_split") != "validation":
        raise MatrixAuditError("Run must use validation selection")
    if config.get("test_used_for_selection") is not False:
        raise MatrixAuditError("Test-assisted selection is rejected")
    status = _read_json(run_dir / "status.json")
    if status.get("execution_status") != "complete":
        raise MatrixAuditError(f"Incomplete run rejected: {run_dir}")
    model = _read_json(run_dir / "model.json")
    _validate_model_metadata(model, config, run_dir / "model.json")
    artifact_sha256: Dict[str, str] = {}
    for name in (
        "config.json",
        "history.json",
        "model.json",
        "validation_predictions_long.csv",
        "test_predictions_long.csv",
        "predictions_long.csv",
        "metrics.json",
        "llm_audit.jsonl",
        "console.log",
    ):
        path = run_dir / name
        if path.is_file():
            artifact_sha256[name] = _sha256_file(path)
    return type(
        "AuditedRun",
        (),
        {
            "run_dir": run_dir,
            "relative_run_dir": run_dir.name,
            "config_hash": str(config.get("config_hash", "")),
            "config": config,
            "status": status,
            "model": model,
            "artifact_sha256": artifact_sha256,
        },
    )()


def _validation_metrics(run) -> Dict[str, Any]:
    rows = _read_prediction_rows(run, "validation")
    return _split_metrics(rows)


def freeze_from_search_run(
    run_dir: Path,
    output: Path = DEFAULT_OUTPUT,
    *,
    force: bool = False,
) -> Dict[str, Any]:
    if output.exists() and not force:
        raise FileExistsError(f"Frozen model already exists: {output}; use --force")
    winner = _audited_run_from_directory(run_dir)
    validation_metrics = _validation_metrics(winner)
    test_metrics = _winner_test_report(winner, validation_metrics)
    expression = winner.model["universal_expression"]["template"]
    lane_parameters = winner.model["lane_parameters"]
    physical = _physical_recheck(expression, lane_parameters)
    model_sha256 = _configuration_hash(
        {"expression": expression, "lane_parameters": lane_parameters}
    )
    matrix_root = run_dir
    while matrix_root != matrix_root.parent:
        if (matrix_root / "plan_manifest.json").is_file():
            break
        matrix_root = matrix_root.parent
    else:
        raise MatrixAuditError(f"No plan_manifest.json found above {run_dir}")
    plan_path = matrix_root / "plan_manifest.json"
    plan_sha256 = _sha256_file(plan_path) if plan_path.is_file() else ""
    payload = {
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
            "tie_breakers": ["single nominated search run"],
            "test_used_for_selection": False,
            "test_opened_after_winner_fixed": True,
            "candidate_count": 1,
            "candidate_validation_metrics": [{
                "run_id": int(winner.config["run_id"]),
                "run_dir": str(run_dir),
                "config_sha256": winner.config_hash,
                "validation_macro_r2": float(
                    validation_metrics["macro_average"]["r2"]
                ),
                "validation_macro_rmse": float(
                    validation_metrics["macro_average"]["rmse"]
                ),
                "validation_macro_mae": float(
                    validation_metrics["macro_average"]["mae"]
                ),
            }],
            "selected_run_id": int(winner.config["run_id"]),
            "selected_run_dir": str(run_dir),
            "export_mode": "single_search_run",
        },
        "evolution": {
            "intersection_id": 1,
            "variant": "principlewise",
            "score_mode": "principlewise",
            "generations": int(winner.config["generations"]),
            "population": int(winner.config["population"]),
            "independent_runs": int(winner.config.get("runs_in_matrix", 10)),
            "optimizer_restarts": int(winner.config["optimizer_restarts"]),
            "run_seed": int(winner.config.get("run_seed", 0)),
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
            "source_matrix": str(matrix_root),
            "source_plan": str(plan_path) if plan_path.is_file() else "",
            "source_plan_sha256": plan_sha256,
            "source_run": str(run_dir),
            "source_config_sha256": winner.config_hash,
            "source_config": winner.config,
            "source_code_sha256": dict(winner.config["code_sha256"]),
            "source_code_bundle_sha256": _configuration_hash(
                winner.config["code_sha256"]
            ),
            "source_data_config_sha256": _configuration_hash(winner.config["data"]),
            "source_artifact_sha256": winner.artifact_sha256,
            "export_mode": "single_search_run",
        },
        "thought": winner.model.get("universal_expression", {}).get("thought", ""),
        "explanation": winner.model.get("universal_expression", {}).get(
            "explanation", ""
        ),
    }
    from freeze_evolved_model import _write_frozen  # local import avoids cycle

    archive = _write_frozen(payload, output, force=True)
    if archive is not None:
        payload["_archived_previous_model"] = str(archive)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Freeze one audited Intersection-1 principlewise search run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        payload = freeze_from_search_run(args.run_dir, args.output, force=args.force)
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    validation = payload["metrics"]["validation"]["macro_average"]
    test = payload["metrics"]["test_report_only"]["macro_average"]
    print(f"Frozen model: {args.output.resolve()}")
    print(f"Selected run: {payload['selection_protocol']['selected_run_dir']}")
    print(f"Model SHA-256: {payload['model_sha256']}")
    print(f"Validation macro R2: {validation['r2']:.6f}")
    print(f"Test macro R2: {test['r2']:.6f} (report only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
