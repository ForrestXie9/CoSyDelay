"""Run one V11 P10/G10 Training-only comparison without an incumbent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from llm_config import load_llm_config
from methods.cosydelay._internal.data_protocol.diagnostics import (
    audit_parameter_quality,
    summarize_compute_efficiency,
)
from methods.cosydelay._internal.data_protocol.run_formal_training_search import (
    DEFAULT_DATA_DIR,
    sha256_file,
    utc_now,
)
from methods.cosydelay._internal.candidate_engine import (
    run_final_prompt_pilot as engine,
)

from .integration import install_v11_candidate
from .policy import V11_POLICY
from .run_training_pilot import _v11_prompt_audit


POPULATION = 10
GENERATIONS = 10


def _configure_utf8_console() -> None:
    """Keep diagnostic printing from aborting an otherwise completed run."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--intersection", type=int, choices=range(1, 7), required=True)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    _configure_utf8_console()
    args = arguments()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite: {args.output}")
    train_path = (
        args.data_dir / f"Intersection_{args.intersection}_Train.jsonl"
    ).resolve()
    if not train_path.is_file():
        raise FileNotFoundError(train_path)
    llm = load_llm_config()
    observed = {
        "model": llm.model,
        "temperature": llm.temperature,
        "top_p": llm.top_p,
        "max_tokens": llm.max_tokens,
        "seed": llm.seed,
    }
    if observed != V11_POLICY.llm_sampling:
        raise RuntimeError(
            f"LLM configuration changed: {observed} != {V11_POLICY.llm_sampling}"
        )

    started_utc = utc_now()
    previous = (
        engine.PILOT_POPULATION,
        engine.PILOT_GENERATIONS,
        engine.install_v10_candidate,
        engine.V10_POLICY,
        engine._prompt_audit,
        sys.argv[:],
    )
    try:
        engine.PILOT_POPULATION = POPULATION
        engine.PILOT_GENERATIONS = GENERATIONS
        engine.install_v10_candidate = install_v11_candidate
        engine.V10_POLICY = V11_POLICY
        engine._prompt_audit = _v11_prompt_audit
        sys.argv = [
            str(engine.__file__),
            "--intersection",
            str(args.intersection),
            "--data-dir",
            str(args.data_dir),
            "--output",
            str(args.output),
        ]
        code = engine.main()
    finally:
        (
            engine.PILOT_POPULATION,
            engine.PILOT_GENERATIONS,
            engine.install_v10_candidate,
            engine.V10_POLICY,
            engine._prompt_audit,
            old_argv,
        ) = previous
        sys.argv = old_argv
    if code:
        return int(code)

    result_path = args.output / "result.json"
    history_path = args.output / "history.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    history = json.loads(history_path.read_text(encoding="utf-8"))
    budget = next(
        item
        for item in reversed(history)
        if item.get("event") == "search_budget_summary"
    )
    required = {
        "population": POPULATION,
        "generations": GENERATIONS,
        "completed_generations": GENERATIONS,
        "candidate_evaluations": POPULATION * (GENERATIONS + 1),
    }
    observed_budget = {
        "population": result.get("population"),
        "generations": result.get("generations"),
        "completed_generations": int(budget["completed_generations"]),
        "candidate_evaluations": int(budget["candidate_evaluations"]),
    }
    if observed_budget != required:
        raise RuntimeError(
            f"P10/G10 budget mismatch: {observed_budget} != {required}"
        )
    if result.get("selected_enhanced_physics", {}).get("joint_pass") is not True:
        raise RuntimeError("V11 winner failed the hard physical gate")
    if result.get("accessed_splits") != ["train"]:
        raise RuntimeError(f"split leakage detected: {result.get('accessed_splits')}")

    metrics = dict(result["selected_training_metrics"])
    source_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
    result.update(
        {
            "status": "completed_v11_p10g10_training_only_comparison",
            "not_a_formal_result": False,
            "formal_protocol_complete": True,
            "method_id": V11_POLICY.method_id,
            "method_status": "v11_p10g10_i1_i2_comparison",
            "policy": V11_POLICY.to_dict(),
            "formal_started_utc": started_utc,
            "formal_completed_utc": utc_now(),
            "train_file_name": train_path.name,
            "train_file_sha256": sha256_file(train_path),
            "selection_source": "full_training_evolution_only",
            "post_evolution_cv_reranking": False,
            "post_evolution_refit": False,
            "external_incumbent_allowed": False,
            "initial_population_all_generated_in_current_run": True,
            "completed_generations": GENERATIONS,
            "expression": result["selected_expression"],
            "lane_parameters": result["selected_parameters"],
            "training_metrics": metrics,
            "formal_comparison_training_metrics": {
                "raw_approach_macro_r2": metrics["macro_raw_r2"],
                "pooled_rmse": metrics["pooled_rmse"],
                "pooled_mae": metrics["pooled_mae"],
            },
            "training_fitness": result["selected_training_fitness"],
            "parameter_quality_audit": audit_parameter_quality(
                result["selected_expression"],
                result["selected_parameters"],
                V11_POLICY.coefficient_bounds,
            ),
            "compute_efficiency": summarize_compute_efficiency(history),
            "llm_runtime": {
                "provider_endpoint": llm.endpoint,
                "request_path": llm.request_path,
                "sampling": V11_POLICY.llm_sampling,
                "sampling_parameter_sources": llm.sampling_parameter_sources,
                "api_key_recorded": False,
            },
            "v11_source_sha256": {
                path.name: sha256_file(path) for path in source_files
            },
        }
    )
    engine.write_json(result_path, result)
    print(
        f"I{args.intersection} V11 P10/G10: "
        f"R2={metrics['macro_raw_r2']:.6f}, "
        f"RMSE={metrics['pooled_rmse']:.6f}, "
        f"MAE={metrics['pooled_mae']:.6f}, physics=True",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
