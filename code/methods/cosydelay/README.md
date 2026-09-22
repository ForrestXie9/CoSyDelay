# CoSyDelay method package

This directory is the public Python entry point for the CoSyDelay symbolic
traffic-delay method.

The implementation uses the fixed Training/Validation/Test protocol, a
P10/G10 candidate budget, training-only fitting and selection, traffic-physics
checks, and a frozen Test evaluation. Implementation helpers live in
`methods/cosydelay/_internal/` and generic runtime helpers live in
`methods/_runtime/`. They are private implementation details of this one
method, not additional methods or historical variants.

Run from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\code).Path
python -u -m methods.cosydelay.run_p10g10_100 `
  --intersection 1 `
  --data-dir .\data\locked_splits `
  --output .\runs\cosydelay_i1
```

Replace `1` with an intersection ID from `1` to `6`. The runner requires a
locally supplied `LLM_API_KEY`; no credentials or generated run files are
distributed. If an existing Windows/Anaconda installation reports an
optional `gmpy2` DLL error, set `SYMPY_GROUND_TYPES=python` and
`MPMATH_NOGMPY=1` before running.
