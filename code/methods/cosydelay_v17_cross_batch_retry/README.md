# CoSyDelay V17: cross-batch legality retry

V17 is the prospective formal candidate derived from the frozen V16 method.
It changes one reliability mechanism only: when all five output attempts in a
generation batch fail, the next batch receives the last legality/novelty reason
and one truncated rejected expression. A successful batch immediately clears
that memory.

The correction never contains Training accuracy, Fitness, RMSE, MAE,
Validation, or Test information and is not used for selection. The following
V16 contracts are unchanged:

- paper Training Fitness and its tie order;
- seven-rule physical score and diagnostic strict joint pass;
- L-BFGS-B in positive log coordinates with 10 restarts;
- up to four approach workers;
- P10/G10 and exactly 110 fitted candidate evaluations;
- full-Training evolution only, with no Validation, CV reranking, refit,
  incumbent injection, or early stop;
- Test remains inaccessible until all six Training searches are frozen.

`test_v17.py` reproduces the I2 failure mode with the real expression generator:
one complete batch returns coefficient-role-conflicted expressions, while the
next batch receives the legality-only correction and succeeds. The formal
launcher additionally recomputes the inherited V16 integrity checks and audits
the retry state machine from the archived result.

The failed V16 batch is diagnostic evidence only. V17 must run I1--I6 from
scratch; no V16 result is reused in the formal V17 comparison.
