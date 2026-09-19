# CoSyDelay method

This directory is the public entry point for the released CoSyDelay symbolic
traffic-delay method. The search uses the fixed Training/Validation/Test
protocol, a P10/G10 candidate budget, training-only parameter fitting and
selection, physical admissibility checks, and a final frozen Test evaluation.

The implementation-specific runtime modules are kept under
`methods/_runtime/` and `methods/_compat/`; those directories are private
support code, not additional public methods or reviewer experiments.

## Run one intersection

From the repository root:

```powershell
$env:PYTHONPATH = (Resolve-Path .\code).Path
python -u -m methods.cosydelay.run_p10g10_100 `
  --intersection 1 `
  --data-dir .\data\locked_splits `
  --output .\runs\cosydelay_i1
```

Use a different intersection ID (`1`-`6`) and output directory for each run.
The LLM interface reads a locally supplied `LLM_API_KEY`; credentials are never
part of this release.
