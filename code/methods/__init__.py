"""Public package namespace for the final CoSyDelay implementation.

Only :mod:`methods.cosydelay` is part of the public API.  Its private
``support`` subdirectory contains the numerical/LLM helpers used by the
method; implementation details live below ``methods.cosydelay.engine``
and are not separate algorithms.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_SUPPORT = _ROOT / "cosydelay" / "support"

if str(_SUPPORT) not in sys.path:
    sys.path.insert(0, str(_SUPPORT))

__all__ = ["cosydelay"]
