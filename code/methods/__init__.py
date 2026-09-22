"""Public package namespace for the final CoSyDelay implementation.

Only :mod:`methods.cosydelay` is part of the public API.  Its private
``_runtime`` subdirectory contains the numerical/LLM helpers used by the
method; implementation details live below ``methods.cosydelay._internal``
and are not separate algorithms.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_RUNTIME = _ROOT / "cosydelay" / "_runtime"

if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

__all__ = ["cosydelay"]
