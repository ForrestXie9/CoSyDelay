"""Small API smoke run that never opens the locked Test JSONL files."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any, Dict, Iterable


GMINI_DIR = Path(__file__).resolve().parents[2]
if str(GMINI_DIR) not in sys.path:
    sys.path.insert(0, str(GMINI_DIR))

from constants import INTERSECTION_CONFIGS  # noqa: E402
from data_processing import load_dataset_flexible, preprocess_data_flexible  # noqa: E402
from population_evolution_lane import evolve_universal_lane_expression  # noqa: E402
from reviewer_revision_experiments.evolution_matrix.shared import (  # noqa: E402
    read_jsonl_dialogues,
    stable_fit_validation_split,
    to_jsonable,
)

from . import (  # noqa: E402
    OFFICIAL_PARALLEL_WORKERS,
    OFFICIAL_RESTARTS,
    P4G2_ADAPTIVE_SEARCH,
    install_population_evolution_fitter,
    install_retained_evolution,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersections", nargs="+", type=int, default=[1, 6])
    parser.add_argument("--population", type=_positive_int, default=2)
    parser.add_argument("--generations", type=_positive_int, default=1)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--split-seed", type=int, default=20260712)
    parser.add_argument(
        "--prefit-mode",
        choices=("none", "conservative"),
        default="conservative",
        help="A/B switch: coefficient fitter only, or retained conservative gate",
    )
    parser.add_argument(
        "--search-policy",
        choices=("standard", "feasible_adaptive"),
        default="standard",
        help="standard evolution, or the retained training-only feasible archive policy",
    )
    parser.add_argument(
        "--validation-mode",
        choices=("final", "none"),
        default="final",
        help="score the reserved Train holdout once after selection, or omit it",
    )
    parser.add_argument("--max-wall-seconds", type=float, default=900.0)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=GMINI_DIR.parent.parent / "Final_cosy_delay" / "jsonl_files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "experiments"
        / "smoke_i1_i6_trainonly_p2g1_20260803",
    )
    return parser.parse_args()


@contextmanager
def _install_experiment_runtime(prefit_mode: str):
    """Install an explicitly labelled A/B runtime for one smoke run."""
    if prefit_mode == "conservative":
        with install_retained_evolution() as runtime:
            yield runtime
        return
    if prefit_mode == "none":
        with install_population_evolution_fitter() as fitter:
            yield SimpleNamespace(fitter=fitter, prefit_audit=[])
        return
    raise ValueError(f"unknown prefit mode: {prefit_mode}")


def _lanes(config: Dict[str, Any]):
    lanes = []
    lane_to_approach = {}
    for approach in config["approaches"]:
        for movement in config["movements"][approach]:
            lane = f"{approach}_{movement}"
            lanes.append(lane)
            lane_to_approach[lane] = approach
    return lanes, lane_to_approach


def _targets(frame, approaches: Iterable[str]):
    return {
        approach: frame[f"Delay_{approach}"]
        for approach in approaches
    }


def _bundle(train_path: Path, intersection_id: int, split_seed: int):
    raw_train = read_jsonl_dialogues(train_path)
    manifest = stable_fit_validation_split(raw_train, split_seed, 0.20)
    full_train = preprocess_data_flexible(
        load_dataset_flexible(str(train_path), intersection_id),
        intersection_id,
    )
    if len(full_train) != len(manifest):
        raise ValueError("training loader and split manifest row counts differ")
    fit_indices = [
        int(row["source_index"])
        for row in manifest
        if row["split"] == "fit"
    ]
    validation_indices = [
        int(row["source_index"])
        for row in manifest
        if row["split"] == "validation"
    ]
    return (
        full_train.iloc[fit_indices].reset_index(drop=True),
        full_train.iloc[validation_indices].reset_index(drop=True),
        manifest,
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            to_jsonable(value),
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = _arguments()
    if args.max_wall_seconds <= 0:
        raise ValueError("--max-wall-seconds must be positive")
    if args.search_policy == "feasible_adaptive" and (
        args.population != P4G2_ADAPTIVE_SEARCH.population
        or args.generations != P4G2_ADAPTIVE_SEARCH.generations
    ):
        raise ValueError(
            "feasible_adaptive is the frozen P4/G2 pilot policy; use "
            f"--population {P4G2_ADAPTIVE_SEARCH.population} "
            f"--generations {P4G2_ADAPTIVE_SEARCH.generations}"
        )
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(args.output.glob("intersection_*/result.json"))
    ]

    for intersection_id in args.intersections:
        if intersection_id not in INTERSECTION_CONFIGS:
            raise ValueError(f"unknown intersection {intersection_id}")
        config = INTERSECTION_CONFIGS[intersection_id]
        approaches = list(config["approaches"])
        train_path = args.data_dir / f"Intersection_{intersection_id}_Train.jsonl"
        if not train_path.exists():
            raise FileNotFoundError(train_path)
        fit, validation, manifest = _bundle(
            train_path, intersection_id, args.split_seed
        )
        lanes, lane_to_approach = _lanes(config)
        fit_targets = _targets(fit, approaches)
        validation_targets = (
            _targets(validation, approaches)
            if args.validation_mode == "final"
            else None
        )
        run_dir = args.output / f"intersection_{intersection_id:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        audit_path = run_dir / "llm_audit.jsonl"
        previous_audit = os.environ.get("LLM_AUDIT_LOG")
        previous_content = os.environ.get("LLM_AUDIT_INCLUDE_CONTENT")
        os.environ["LLM_AUDIT_LOG"] = str(audit_path.resolve())
        os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = "false"
        started = time.perf_counter()
        try:
            with _install_experiment_runtime(args.prefit_mode) as runtime:
                search_kwargs = (
                    P4G2_ADAPTIVE_SEARCH.evolution_kwargs()
                    if args.search_policy == "feasible_adaptive"
                    else {}
                )
                result = evolve_universal_lane_expression(
                    df_train=fit,
                    lanes=lanes,
                    lane_to_approach=lane_to_approach,
                    approach_targets=fit_targets,
                    universal_features=["flow_lane", "GR_phase", "Cycle_Time"],
                    generations=args.generations,
                    pop_size=args.population,
                    intersection_id=intersection_id,
                    score_mode="binary",
                    physics_weight=1.0,
                    prompt_knowledge=True,
                    prompt_style="standard",
                    seed=args.seed + intersection_id * 10_000,
                    optimizer_restarts=OFFICIAL_RESTARTS,
                    df_validation=(
                        validation if args.validation_mode == "final" else None
                    ),
                    validation_targets=validation_targets,
                    max_wall_seconds=args.max_wall_seconds,
                    **search_kwargs,
                )
                configured_workers = runtime.fitter.parallel_workers
                prefit_audit = list(runtime.prefit_audit)
        finally:
            if previous_audit is None:
                os.environ.pop("LLM_AUDIT_LOG", None)
            else:
                os.environ["LLM_AUDIT_LOG"] = previous_audit
            if previous_content is None:
                os.environ.pop("LLM_AUDIT_INCLUDE_CONTENT", None)
            else:
                os.environ["LLM_AUDIT_INCLUDE_CONTENT"] = previous_content

        expression, thought, explanation, lane_parameters, history = result
        search_summary = next(
            item
            for item in reversed(history)
            if item.get("event") == "search_budget_summary"
        )
        final_event = next(
            (
                item
                for item in reversed(history)
                if item.get("event") == "final_validation_evaluation"
            ),
            None,
        )
        if args.validation_mode == "final" and final_event is None:
            raise RuntimeError("final validation event is missing")
        if args.validation_mode == "none" and final_event is not None:
            raise RuntimeError("validation was evaluated despite --validation-mode none")
        candidate_id = (
            final_event["candidate_id"]
            if final_event is not None
            else search_summary["final_candidate_id"]
        )
        candidate_event = next(
            item
            for item in reversed(history)
            if item.get("candidate_id") == candidate_id
            and item.get("event") == "evaluated"
        )
        fit_diagnostics = candidate_event.get("evaluation_details", {}).get(
            "fit", {}
        )
        summary = {
            "method_id": (
                "cosydelay_lbfgsb_r10_parallel4_p4g2_training_selection_v2"
                if args.search_policy == "feasible_adaptive"
                else (
                    "cosydelay_lbfgsb_r10_parallel4_training_selection_v1"
                    if args.prefit_mode == "none"
                    else "cosydelay_lbfgsb_r10_parallel4_conservative_prefit_v1"
                )
            ),
            "prefit_mode": args.prefit_mode,
            "search_policy": args.search_policy,
            "search_policy_config": (
                P4G2_ADAPTIVE_SEARCH.to_dict()
                if args.search_policy == "feasible_adaptive"
                else {}
            ),
            "intersection_id": intersection_id,
            "approaches": approaches,
            "approach_count": len(approaches),
            "configured_worker_cap": configured_workers,
            "actual_workers_for_final_candidate": fit_diagnostics.get(
                "parallel_workers"
            ),
            "optimizer_restarts": OFFICIAL_RESTARTS,
            "population": args.population,
            "generations": args.generations,
            "run_seed": args.seed,
            "evolution_seed": args.seed + intersection_id * 10_000,
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "validation_mode": args.validation_mode,
            "validation_evaluated": final_event is not None,
            "split_seed": args.split_seed,
            "test_file_opened": False,
            "validation_accessed_during_evolution": False,
            "expression": expression,
            "thought": thought,
            "explanation": explanation,
            "lane_parameters": lane_parameters,
            "train_r2": candidate_event.get("train_r2"),
            "fitness": candidate_event.get("fitness"),
            "physical_joint_pass": candidate_event.get("physical_joint_pass"),
            "physical_rule_scores": candidate_event.get("rule_scores", {}),
            "validation_r2": (
                final_event.get("validation_r2") if final_event else None
            ),
            "validation_rmse": (
                final_event.get("validation_rmse") if final_event else None
            ),
            "validation_r2_by_approach": (
                final_event.get("validation_r2_by_approach", {})
                if final_event
                else {}
            ),
            "validation_rmse_by_approach": (
                final_event.get("validation_rmse_by_approach", {})
                if final_event
                else {}
            ),
            "wall_seconds": time.perf_counter() - started,
            "candidate_evaluations": sum(
                item.get("event") == "evaluated" for item in history
            ),
            "completed_generations": search_summary.get(
                "completed_generations"
            ),
            "early_stop_triggered": search_summary.get(
                "early_stop_triggered", False
            ),
            "early_stop_reason": search_summary.get("early_stop_reason"),
            "early_stop_diagnostics": search_summary.get(
                "early_stop_diagnostics", {}
            ),
            "feasible_archive_size": search_summary.get(
                "feasible_archive_size", 0
            ),
            "final_selection_policy": search_summary.get(
                "final_selection_policy"
            ),
            "validation_evaluation_events": sum(
                item.get("event") == "final_validation_evaluation"
                for item in history
            ),
            "prefit_gate_attempts": len(prefit_audit),
            "prefit_gate_passes": sum(
                bool(item.get("passed")) for item in prefit_audit
            ),
            "prefit_gate_rejections": sum(
                not bool(item.get("passed")) for item in prefit_audit
            ),
            "prefit_rejection_reasons": [
                item.get("reason")
                for item in prefit_audit
                if not bool(item.get("passed"))
            ],
            "split_manifest": manifest,
        }
        _write_json(run_dir / "result.json", summary)
        _write_json(run_dir / "history.json", history)
        _write_json(run_dir / "prefit_gate_audit.json", prefit_audit)
        summaries = [
            item
            for item in summaries
            if int(item["intersection_id"]) != intersection_id
        ]
        summaries.append(summary)
        summaries.sort(key=lambda item: int(item["intersection_id"]))
        _write_json(args.output / "summary.json", summaries)
        validation_text = (
            f"{summary['validation_r2']:.5f}"
            if summary["validation_r2"] is not None
            else "not_evaluated"
        )
        print(
            f"I{intersection_id}: physical={summary['physical_joint_pass']}, "
            f"train_R2={summary['train_r2']:.5f}, "
            f"validation_R2={validation_text}, "
            f"wall={summary['wall_seconds']:.1f}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
