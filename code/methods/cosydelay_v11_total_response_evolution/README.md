# CoSyDelay V11 total-response evolution

This is an isolated, prospective Training-only repair candidate. It does not
modify or seed from any V9/V10 result expression.

V11 keeps the paper Training Fitness, the fitted hard-physics gate, positive
all-log L-BFGS-B with ten restarts, up to four approach workers, coefficient
bounds, and the no-Validation/no-Test/no-incumbent protocol.

It changes only the search behavior:

- physics monotonicity is stated for the complete delay response rather than
  for `H` in isolation;
- physically safe compensated saturation structures are permitted;
- canonical expression uniqueness replaces global structural-family
  uniqueness, so a useful family can continue to evolve;
- one failed fitted expression no longer blacklists its whole family; and
- mutation receives per-approach Training metrics and the weakest approach.

The first authorized experiment is P3/G2 and is diagnostic, not formal.
