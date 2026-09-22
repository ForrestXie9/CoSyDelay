# CoSyDelay method package

This directory is the public Python entry point for the CoSyDelay symbolic
traffic-delay method.

The implementation uses the fixed Training/Validation/Test protocol, a
P10/G10 candidate budget, training-only fitting and selection, traffic-physics
checks, and a frozen Test evaluation. The search and fitting engine lives in
`methods/cosydelay/engine/`, while data, expression, numerical, and LLM
support lives in `methods/cosydelay/support/`. These are details of this one
method, not additional methods or historical variants.

Run from the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\code).Path
python -u -m methods.cosydelay.run `
  --intersection 1 `
  --data-dir .\data\locked_splits `
  --output .\runs\cosydelay_i1 `
  --population 10 `
  --generations 10
```

Replace `1` with an intersection ID from `1` to `6`. The population and
generation options are optional and default to `10` each; the candidate budget
is their product. The runner requires a locally supplied `LLM_API_KEY`; no
credentials or generated run files are distributed. If an existing
Windows/Anaconda installation reports an
optional `gmpy2` DLL error, set `SYMPY_GROUND_TYPES=python` and
`MPMATH_NOGMPY=1` before running.
