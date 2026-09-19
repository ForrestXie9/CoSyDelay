# CoSyDelay V21

V21 promotes the successful global-selection and numeric-domain ablations:

- V20 initialization and regeneration prompts with only the empirically
  selected numeric operating-domain clause added;
- global `(mu + lambda)` population survivor selection;
- 10 initial candidates plus 9 rounds of 10 candidates (100 total);
- V20 accelerated fitting, full-history duplicate checks, R4/R7 pre-fit gates,
  expression normalization, reproducible per-call seeds, and binary strict-
  physics fitness.
- denominator pre-fit rejection is inherited unchanged from V20; the only
  search-algorithm change in V21 is global survivor selection.

Run all six intersections from `Parameters_sensitive/gmini`:

```powershell
python -u -m methods.cosydelay_v21_global_selection.launch_i1_i6 `
  --output-root "methods\cosydelay_v21_global_selection\experiments\v21_i1_i6_<stamp>"
```
