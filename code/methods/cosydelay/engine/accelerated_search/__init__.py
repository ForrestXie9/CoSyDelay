"""Internal acceleration support used by the public CoSyDelay method."""

# Keep the worker module inside this package so the release has no dependency
# on reviewer-only experiment paths.  A normal package import also gives
# ProcessPoolExecutor a stable module name to pickle on Windows spawn.
from . import parallel_restart_impl
