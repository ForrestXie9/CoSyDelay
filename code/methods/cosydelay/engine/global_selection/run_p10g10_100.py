"""Run one formal V21 global-selection search with the unchanged V20 prompt."""
from __future__ import annotations

import json
from pathlib import Path


def _source() -> str:
    path = (
        Path(__file__).resolve().parents[1]
        / "accelerated_search"
        / "run_i1_p10g10_100.py"
    )
    source = path.read_text(encoding="utf-8")
    replacements = (
        (
            "from methods.cosydelay.engine.accelerated_search import pairwise_population",
            "from methods.cosydelay.engine.global_selection import global_population as pairwise_population",
        ),
        (
            "from methods.cosydelay.engine.accelerated_search.prompt_audit import audit_v20_prompts",
            "from methods.cosydelay.engine.global_selection.prompt_audit import audit_v21_prompts as audit_v20_prompts",
        ),
        (
            "from methods.cosydelay.engine.accelerated_search import paper_prompt as v20_prompt",
            "from methods.cosydelay.engine.global_selection import prompt as v20_prompt",
        ),
        (
            "cosydelay_v20_pairwise_simple_restarts_p10g10_100_t1",
            "cosydelay_v21_global_numeric_domain_p10g10_100_t1",
        ),
        (
            "v20_pairwise_training_only_p10g10_exactly_100_temperature_1",
            "v21_global_numeric_domain_training_only_p10g10_100_t1",
        ),
        ("'pairwise_parent_child_selection'", "'global_population_selection'"),
        ("'prompt_omits_numeric_operating_domain'", "'prompt_declares_numeric_operating_domain'"),
        ("'v20_changes':", "'v21_changes':"),
        ("V20 I{args.intersection}", "V21 I{args.intersection}"),
    )
    for old, new in replacements:
        if old not in source:
            raise RuntimeError(f"V21 runner could not locate source fragment: {old}")
        source = source.replace(old, new)
    return source


def main() -> int:
    namespace = {"__name__": "cosydelay_v21_embedded"}
    exec(compile(_source(), "<cosydelay-v21-global-selection>", "exec"), namespace)
    code = int(namespace["main"]())
    if code:
        return code
    # Add an explicit promotion record without altering the V20 prompt text.
    import sys
    output = Path(sys.argv[sys.argv.index("--output") + 1])
    result_path = output / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["formal_version"] = "V21"
    result["survivor_selection"] = "global_mu_plus_lambda"
    result["prompt_contract"] = "V20_plus_numeric_operating_domain_only"
    result["denominator_prefit_policy"] = "unchanged_from_V20"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
