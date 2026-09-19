"""Public CoSyDelay package with private runtime compatibility paths.

The release exposes only :mod:`methods.cosydelay`.  The final implementation
still has a few absolute imports inherited from the research code, so the
packaged runtime and compatibility modules are added to the package search
path here.  They are intentionally internal and are not separate published
methods.
"""

from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parent
_RUNTIME = _ROOT / "_runtime"
_COMPAT = _ROOT / "_compat"

if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))
if str(_COMPAT) not in __path__:
    __path__.append(str(_COMPAT))

__all__ = ["cosydelay"]
