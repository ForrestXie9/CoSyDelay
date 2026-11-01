# CoSyDelay method package

This directory is the public Python entry point for the CoSyDelay symbolic
traffic-delay method.

The implementation uses the fixed Training/Validation/Test splits. Training
alone fits candidate coefficients and drives evolution. Validation Fitness
selects the final expression from the final population of `P` survivors; its
structure is fixed before coefficients are
refitted on Training+Validation. All seven physical principles are rechecked
with the refitted coefficients before the final model is frozen and tested.
The search and fitting engine lives in
`methods/cosydelay/engine/`, while data, expression, numerical, and LLM
support lives in `methods/cosydelay/support/`. These are details of this one
method, not additional methods or historical variants.

The public stages are coordinated by `pipeline.py`; `physics.py` isolates the
complete native verifier in a time/memory-bounded worker. Defaults are stored
in `configs/cosydelay.json` at the repository root.

Validation selects the maximum of `mean_a(max(0, R2_val,a)) + P_train`, with
lower direction-average Validation RMSE and then lower candidate ID breaking
ties. Every completed candidate retains its Training coefficients, so selection
does not introduce an unreported recalibration. R2 reporting itself is not
clipped. I4-E and I6-W are excluded from direction-average reporting because
they are the predeclared invalid/constant target directions.

The engine records the actual final survivor IDs before any held-out access.
Validation evaluates only these `P` candidates, including surviving parents
from earlier generations. The entire archive and the last `P` offspring are
not alternative selection pools. `selection_pool.json` freezes the exact
eligible population before Validation is opened; archives lacking the survivor
snapshot are rejected.

After coefficient-only Training+Validation refitting, the native seven-rule
verifier runs again. Failure or an incomplete resource-limited check sets the
final physical component to zero; the refitted model is still the frozen Test
model. Numerical fitting or prediction failures fail the run without a
Test-driven fallback. Model and physical-report hashes are saved before Test.

LLM content uses `### Expression` and `### Explanation` text sections, followed
by safe symbolic parsing. The HTTP response envelope being JSON does not mean
that the generated expression content is a JSON object.

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
