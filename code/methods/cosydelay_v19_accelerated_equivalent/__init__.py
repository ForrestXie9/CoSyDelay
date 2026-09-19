"""Runtime-only V18-equivalent acceleration overlay (not frozen V18 source)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# ProcessPoolExecutor pickles worker callables by module name.  Register the
# reviewed experimental implementation below a real package name so spawned
# workers can import exactly the same callable rather than a transient spec.
_IMPL_NAME = __name__ + ".parallel_restart_impl"
if _IMPL_NAME not in sys.modules:
    _path = Path(__file__).resolve().parents[2] / "reviewer_revision_experiments" / "18_v18_efficiency_equivalence" / "parallel_restart_fitter.py"
    _spec = importlib.util.spec_from_file_location(_IMPL_NAME, _path)
    if _spec is None or _spec.loader is None:
        raise RuntimeError("cannot load restart-parallel equivalence implementation")
    parallel_restart_impl = importlib.util.module_from_spec(_spec)
    sys.modules[_IMPL_NAME] = parallel_restart_impl
    _spec.loader.exec_module(parallel_restart_impl)
