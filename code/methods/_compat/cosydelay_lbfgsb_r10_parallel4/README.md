# CoSyDelay retained fitter: L-BFGS-B r10 parallel4

This version preserves the verified coefficient-fitting semantics of
`cosydelay_lbfgsb_r10_parallel3` and raises only the approach-level concurrency
cap from three to four.

- solver: L-BFGS-B;
- ten deterministic restarts per independent approach;
- `maxiter=200`, `maxfun=20000`;
- all available approaches are fitted;
- `workers = min(4, number_of_approaches)`;
- restarts remain serial inside each approach;
- all starts are generated in the parent process before dispatch.

The evolutionary core uses training accuracy for every parent/offspring
decision. If frozen validation data is supplied, it is scored exactly once,
after the final best expression has already been selected. Validation metrics
never change fitness, physical feedback, mutation, or survivor selection.

Before coefficient fitting, the retained generation gate rejects parser,
grammar, unit, domain-safety, provably wrong-direction, and exact R8 zero-flow
failures. Rejected expressions are regenerated with the recorded reason and do
not consume an optimizer call. R9 is not hard-filtered at neutral coefficients:
its 10,000-second threshold is checked only after fitting with the actual lane
coefficients. No fixed post-fit R9 guard is added.

The former `parallel3` folder remains unchanged as the archived verified
three-worker implementation.

## Retained P4/G2 adaptive search

`search_policy.py` defines the retained training-only policy with population 4 and
at most two generations (12 fitted candidates before early stopping). It keeps
a cross-generation archive of post-fit physical passes and selects the archived
candidate with the highest training R2. Fitted R2/R7/R9 failures receive compact
rule-specific repair feedback without approach or lane labels.

The pilot stops early only after at least two physical candidates have been
found and the best archived training R2 is within 0.02 of the best training R2
seen anywhere in the search. Validation is not consulted. In five training-only
runs on each of I1 and I6, the final physical pass rate was 5/5 for both while
mean training R2 was not degraded relative to the earlier P2/G1 smoke. The
policy is therefore retained as the default search configuration. The
optimizer, gate, and no-fixed-R9-guard invariants are unchanged.
