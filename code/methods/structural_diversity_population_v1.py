"""Global-selection population overlay with strict structural-family novelty."""

from __future__ import annotations

import inspect
from functools import wraps
from types import FunctionType

from methods.cosydelay_v21_global_selection import global_population as _base


_evolve = _base.evolve_universal_lane_expression
_signature = inspect.signature(_evolve)
_parameters = list(_signature.parameters.values())
_defaults = list(_evolve.__defaults__ or ())
_offset = len(_parameters) - len(_defaults)
_structural_index = next(
    index for index, parameter in enumerate(_parameters)
    if parameter.name == "structural_diversity_mode"
)
if _structural_index < _offset:
    raise RuntimeError("structural_diversity_mode is not a defaulted parameter")
_defaults[_structural_index - _offset] = "family_unique"
_evolve.__defaults__ = tuple(_defaults)

# ``global_population`` is executed into its own module namespace.  The
# historical integration stack mutates the population module at runtime
# (fitter, evaluator, retry limits, and generator), so a bare reference to
# the base function would keep looking at the base module's globals and miss
# those patched bindings.  Mirror the base namespace into this overlay and
# clone the evolution function with the overlay globals.
for _name, _value in vars(_base).items():
    if _name not in {"__name__", "__package__", "__loader__", "__spec__", "__file__"}:
        globals().setdefault(_name, _value)


def wrap_without_history_prompt(generator):
    """Keep full program-side exclusions while hiding them from the LLM.

    The generator still checks the complete ``excluded_expressions`` list in
    code.  Only its optional prompt prefix is suppressed, so the historical
    archive cannot bloat or bias regeneration prompts.
    """
    if getattr(generator, "_cosydelay_history_prompt_suppressed", False):
        return generator

    @wraps(generator)
    def wrapped(*args, **kwargs):
        changed = dict(kwargs)
        changed["max_prompt_exclusions"] = 0
        return generator(*args, **changed)

    wrapped._cosydelay_history_prompt_suppressed = True
    wrapped._cosydelay_full_history_program_check = True
    return wrapped


# ``global_population`` imported the generator directly into its execution
# globals.  Replace only that binding after the V21 source has been loaded;
# the frozen V21 runner never imports this overlay.
_safe_generator_name = "safe_generate_universal_lane_expression"
_safe_generator = _evolve.__globals__.get(_safe_generator_name)
if _safe_generator is None:
    raise RuntimeError("population evolution generator binding is missing")
_evolve.__globals__[_safe_generator_name] = wrap_without_history_prompt(
    _safe_generator
)
# Older integration layers access the generator as a module attribute rather
# than through ``evolve_universal_lane_expression``'s globals. Export the
# wrapped binding as well, so the V22 overlay remains API-compatible with the
# frozen V16--V21 execution stack.
safe_generate_universal_lane_expression = _evolve.__globals__[_safe_generator_name]

# The retained runner checks that the installed evolution function belongs to
# the imported population module.  Keep that provenance explicit after the
# default is changed.
_evolve_overlay = FunctionType(
    _evolve.__code__,
    globals(),
    _evolve.__name__,
    _evolve.__defaults__,
    _evolve.__closure__,
)
_evolve_overlay.__kwdefaults__ = _evolve.__kwdefaults__
_evolve_overlay.__annotations__ = dict(_evolve.__annotations__)
_evolve_overlay.__module__ = __name__
evolve_universal_lane_expression = _evolve_overlay

PROGRAM_SIDE_REGENERATION_NOVELTY_SCOPE = "full_evaluated_history_and_structural_family"
STRUCTURAL_DIVERSITY_MODE = "family_unique"
PROMPT_HISTORY_EXPRESSION_COUNT = 0
FULL_HISTORY_PROGRAM_CHECK_WITHOUT_PROMPT = True
