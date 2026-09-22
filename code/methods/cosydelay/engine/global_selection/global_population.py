"""Retained global population selection with V20 full-history novelty checks."""
from __future__ import annotations

from pathlib import Path

_source_path = Path(__file__).resolve().parents[2] / "support" / "population_evolution_lane.py"
_source = _source_path.read_text(encoding="utf-8")
_old = '''            slot_excluded = (
                [item["expr"] for item in population]
                + [item["expr"] for item in new_population]
            )
'''
_new = '''            slot_excluded = list(dict.fromkeys(
                [item["expr"] for item in evaluated_candidates]
                + [item["expr"] for item in population]
                + [item["expr"] for item in new_population]
                + [parent["expr"]]
            ))
'''
if _old not in _source:
    raise RuntimeError("V21 overlay could not locate the exclusion block")
_source = _source.replace(_old, _new, 1)
exec(compile(_source, str(_source_path), "exec"), globals())

PROGRAM_SIDE_REGENERATION_NOVELTY_SCOPE = "full_evaluated_history"
PROMPT_HISTORY_EXPRESSION_COUNT = 0
