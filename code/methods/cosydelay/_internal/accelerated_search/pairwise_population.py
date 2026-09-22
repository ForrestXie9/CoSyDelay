"""V19 pairwise-(1+1) evolution overlay derived from the retained engine."""
from __future__ import annotations

from pathlib import Path


def _physical_principles_fraction(individual: dict) -> str:
    scores = individual.get("rule_scores", {}) or {}
    if not scores:
        return "7/7" if individual.get("physical_joint_pass") else "0/7"
    passed = sum(float(value) >= 1.0 - 1e-12 for value in scores.values())
    return f"{passed}/{len(scores)}"

_source_path = Path(__file__).resolve().parents[2] / "_runtime" / "population_evolution_lane.py"
_source = _source_path.read_text(encoding="utf-8")

_old_exclusions = '''            slot_excluded = (
                [item["expr"] for item in population]
                + [item["expr"] for item in new_population]
            )
'''
_new_exclusions = '''            # V20 regeneration checks its parent and every historical
            # expression program-side. The V20 prompt
            # exclusion limit is zero, so none of these formulas is sent to
            # the LLM even though the canonical retry gate sees all of them.
            slot_excluded = list(dict.fromkeys(
                [item["expr"] for item in evaluated_candidates]
                + [item["expr"] for item in population]
                + [item["expr"] for item in new_population]
                + [parent["expr"]]
            ))
'''
if _old_exclusions not in _source:
    raise RuntimeError("V19 pairwise overlay could not locate prompt exclusions")
_source = _source.replace(_old_exclusions, _new_exclusions, 1)

_old_selection = '''        if new_population:
            combined = population + new_population
            unique = {}
            for item in combined:
                key = (
                    structural_family_key(item["expr"])
                    if structural_diversity_mode == "family_unique"
                    else canonical_expression(item["expr"])
                )
                if (
                    key not in unique
                    or individual_selection_key(item)
                    > individual_selection_key(unique[key])
                ):
                    unique[key] = item
            population = sorted(
                unique.values(), key=individual_selection_key, reverse=True
            )[:pop_size]
'''
_new_selection = '''        if len(new_population) != len(population):
            raise RuntimeError(
                "V19 requires one successfully evaluated novel child for every "
                "parent; incomplete generation is not selectable"
            )
        # Strict (1+1) selection: a child competes only with the parent that
        # generated it.  Ties retain the parent, so no global cross-family
        # displacement occurs.
        population = [
            child
            if individual_selection_key(child) > individual_selection_key(parent)
            else parent
            for parent, child in zip(population, new_population)
        ]
'''
if _old_selection not in _source:
    raise RuntimeError("V19 pairwise overlay could not locate survivor selection")
_source = _source.replace(_old_selection, _new_selection, 1)

_display_replacements = {
    'print(f"Physical score mode: {score_mode}")':
        'print(f"Physical compliance mode: {score_mode}")',
    'f"Physical score: {individual[\'physical_score\']:.4f}, "':
        'f"Physical principles fully satisfied: {_physical_principles_fraction(individual)}, "',
    'print(f"  Physical score: {best[\'physical_score\']:.4f}")':
        'print(f"  Physical principles fully satisfied: {_physical_principles_fraction(best)}")',
    'f"Physical score: {child[\'physical_score\']:.4f}, "':
        'f"Physical principles fully satisfied: {_physical_principles_fraction(child)}, "',
    'print(f"Physical score: {best[\'physical_score\']:.4f}")':
        'print(f"Physical principles fully satisfied: {_physical_principles_fraction(best)}")',
    'print(f"Physical joint pass: {best[\'physical_joint_pass\']}")':
        'print(f"Joint pass: {best[\'physical_joint_pass\']}")',
}
for old, new in _display_replacements.items():
    if old not in _source:
        raise RuntimeError(f"V20 display overlay could not locate: {old}")
    _source = _source.replace(old, new)

exec(compile(_source, str(_source_path), "exec"), globals())

PROGRAM_SIDE_REGENERATION_NOVELTY_SCOPE = "full_evaluated_history"
PROMPT_HISTORY_EXPRESSION_COUNT = 0
